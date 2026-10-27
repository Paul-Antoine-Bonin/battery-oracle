"""Turn the raw command text stored by the collector into numbers.

The collector stores `ioreg` and `pmset` output verbatim. This module reads that
text back and extracts the fields the model needs. It never runs a command, so it
works the same on stored samples from months ago as on one taken a second ago.

Two traps are handled here and nowhere else:

- `ioreg` prints `Amperage` and `InstantAmperage` as unsigned 64 bit integers.
  A discharge of 1417 mA comes back as 18446744073709550199. `signed_64` undoes
  that with two's complement, so negative means discharging.
- `MaxCapacity` drifts as the battery ages. The charge fraction is always
  computed against the maximum read in the same sample, never the design figure.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union

Value = Union[int, bool, str]

_IOREG_LINE = re.compile(r'^\s*"([^"]+)" = (.*?)\s*$')

_TWO_63 = 2**63
_TWO_64 = 2**64


def signed_64(value: int) -> int:
    """Read an unsigned 64 bit integer as the signed value it encodes."""
    return value - _TWO_64 if value >= _TWO_63 else value


def _ioreg_value(text: str) -> Value:
    if text == "Yes":
        return True
    if text == "No":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    # Nested dicts and arrays stay as text. Nothing the model needs lives
    # only in them, and the raw sample still holds them if that changes.
    return text


def ioreg_fields(text: str) -> Dict[str, Value]:
    """Every top level `"Key" = value` pair of an `ioreg -w0` battery dump.

    With `-w0` a nested dict sits on its key's line, so it cannot shadow a top
    level key of the same name. If a key appears twice, the first one wins.
    """
    fields: Dict[str, Value] = {}
    for line in text.splitlines():
        match = _IOREG_LINE.match(line)
        if match and match.group(1) not in fields:
            fields[match.group(1)] = _ioreg_value(match.group(2))
    return fields


@dataclass
class BatteryState:
    """The hardware view of one sample. Any field can be None if absent."""

    current_mah: Optional[int] = None
    max_mah: Optional[int] = None
    design_mah: Optional[int] = None
    charge_fraction: Optional[float] = None   # against max_mah, not design
    health_fraction: Optional[float] = None   # max_mah over design_mah
    cycle_count: Optional[int] = None
    voltage_mv: Optional[int] = None
    amperage_ma: Optional[int] = None          # negative while discharging
    instant_amperage_ma: Optional[int] = None
    power_w: Optional[float] = None            # signed like amperage
    temperature_c: Optional[float] = None
    is_charging: Optional[bool] = None
    external_connected: Optional[bool] = None
    fully_charged: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _int(fields: Dict[str, Value], key: str) -> Optional[int]:
    value = fields.get(key)
    # bool is a subclass of int; a Yes/No where a number belongs is not a number.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _bool(fields: Dict[str, Value], key: str) -> Optional[bool]:
    value = fields.get(key)
    return value if isinstance(value, bool) else None


def _capacities(fields: Dict[str, Value]):
    """Current and maximum charge in mAh.

    Apple silicon reports `CurrentCapacity` and `MaxCapacity` as a percentage
    (MaxCapacity is 100) and puts the mAh figures in `AppleRawCurrentCapacity`
    and `AppleRawMaxCapacity`. Intel Macs report mAh directly. The raw pair is
    preferred whenever it is there.
    """
    current = _int(fields, "AppleRawCurrentCapacity")
    maximum = _int(fields, "AppleRawMaxCapacity")
    if current is None or maximum is None:
        current = _int(fields, "CurrentCapacity")
        maximum = _int(fields, "MaxCapacity")
    return current, maximum


def parse_ioreg(text: str) -> BatteryState:
    """Extract battery state from `ioreg -r -c AppleSmartBattery -w0` output."""
    fields = ioreg_fields(text)
    state = BatteryState()

    state.current_mah, state.max_mah = _capacities(fields)
    state.design_mah = _int(fields, "DesignCapacity")
    if state.current_mah is not None and state.max_mah:
        state.charge_fraction = state.current_mah / state.max_mah
    if state.max_mah is not None and state.design_mah:
        state.health_fraction = state.max_mah / state.design_mah

    state.cycle_count = _int(fields, "CycleCount")
    state.voltage_mv = _int(fields, "Voltage")

    amperage = _int(fields, "Amperage")
    if amperage is not None:
        state.amperage_ma = signed_64(amperage)
    instant = _int(fields, "InstantAmperage")
    if instant is not None:
        state.instant_amperage_ma = signed_64(instant)
    if state.amperage_ma is not None and state.voltage_mv is not None:
        state.power_w = state.amperage_ma * state.voltage_mv / 1e6

    # Hundredths of a degree Celsius.
    temperature = _int(fields, "Temperature")
    if temperature is not None:
        state.temperature_c = temperature / 100

    state.is_charging = _bool(fields, "IsCharging")
    state.external_connected = _bool(fields, "ExternalConnected")
    state.fully_charged = _bool(fields, "FullyCharged")
    return state


_PMSET_SOURCE = re.compile(r"Now drawing from '([^']+)'")
_PMSET_BATTERY = re.compile(
    r"-InternalBattery-\d+(?: \([^)]*\))?\s+"   # battery name and id
    r"(\d+)%;\s*"                                # percentage
    r"([^;]+?)\s*"                                # state
    r"(?:;\s*(.*?))?\s*"                          # estimate, when there is one
    r"(?:present:.*)?$"
)
_PMSET_REMAINING = re.compile(r"(\d+):(\d{2}) remaining")


@dataclass
class PmsetState:
    """What `pmset -g batt` says, including the macOS estimate to beat."""

    power_source: Optional[str] = None       # "Battery Power" or "AC Power"
    percent: Optional[int] = None
    state: Optional[str] = None              # discharging, charging, charged, ...
    estimate_min: Optional[int] = None       # None when macOS gives no estimate
    has_estimate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def parse_pmset(text: str) -> PmsetState:
    """Extract the macOS view from `pmset -g batt` output.

    The estimate reads `1:15 remaining` while discharging and charging, and
    `(no estimate)` for a while after the power source changes. That gap is
    kept as None: filling it in would hand the baseline numbers macOS never
    gave. With `AC attached; not charging` the state is `AC attached` and the
    third field holds no estimate.
    """
    out = PmsetState()

    source = _PMSET_SOURCE.search(text)
    if source:
        out.power_source = source.group(1)

    for line in text.splitlines():
        battery = _PMSET_BATTERY.search(line)
        if not battery:
            continue
        out.percent = int(battery.group(1))
        out.state = battery.group(2)
        remaining = _PMSET_REMAINING.search(battery.group(3) or "")
        if remaining:
            out.estimate_min = int(remaining.group(1)) * 60 + int(remaining.group(2))
            out.has_estimate = True
        break
    return out


_LOADAVG = re.compile(r"\{\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\}")
_WIFI_POWER = re.compile(r"Wi-Fi Power \([^)]*\):\s*(On|Off)\b")
_BRIGHTNESS = re.compile(r'"brightness"=\{([^}]*)\}')
_PLIST_INT = re.compile(r'"(\w+)"=(-?\d+)')


@dataclass
class ContextState:
    """What the machine was doing at the moment of the sample."""

    load_1m: Optional[float] = None
    load_5m: Optional[float] = None
    load_15m: Optional[float] = None
    logical_cpus: Optional[int] = None
    load_per_cpu: Optional[float] = None     # load_1m over logical_cpus
    wifi_on: Optional[bool] = None
    brightness_fraction: Optional[float] = None   # 0.0 to 1.0, built-in panel

    def to_dict(self) -> dict:
        return asdict(self)


def parse_cpu_load(text: str, state: Optional[ContextState] = None) -> ContextState:
    """Load averages and core count from `sysctl -n vm.loadavg hw.logicalcpu`.

    The output is `{ 2.15 1.98 1.87 }` on the first line and the core count on
    the second. Load is also given per core, since a load of 4 means something
    different on 8 cores than on 4.
    """
    state = state or ContextState()
    loads = _LOADAVG.search(text)
    if loads:
        state.load_1m, state.load_5m, state.load_15m = (
            float(loads.group(i)) for i in (1, 2, 3)
        )
    for line in text.splitlines():
        if line.strip().isdigit():
            state.logical_cpus = int(line.strip())
            break
    if state.load_1m is not None and state.logical_cpus:
        state.load_per_cpu = state.load_1m / state.logical_cpus
    return state


def parse_wifi_power(text: str, state: Optional[ContextState] = None) -> ContextState:
    """Whether Wi-Fi is powered, from `networksetup -getairportpower en0`.

    Anything other than a clear On or Off, such as the error printed when en0
    is not a Wi-Fi interface, leaves `wifi_on` as None.
    """
    state = state or ContextState()
    match = _WIFI_POWER.search(text)
    if match:
        state.wifi_on = match.group(1) == "On"
    return state


def parse_brightness(text: str, state: Optional[ContextState] = None) -> ContextState:
    """Built-in display brightness from `ioreg -r -k IODisplayParameters`.

    The `brightness` entry holds `min`, `max` and `value` in no fixed order.
    The fraction is `(value - min) / (max - min)`. With the lid closed or only
    an external display attached there is no such entry, and the fraction
    stays None rather than reading as a dark screen.
    """
    state = state or ContextState()
    match = _BRIGHTNESS.search(text)
    if not match:
        return state
    numbers = {key: int(value) for key, value in _PLIST_INT.findall(match.group(1))}
    low, high, value = numbers.get("min"), numbers.get("max"), numbers.get("value")
    if low is None or high is None or value is None or high <= low:
        return state
    fraction = (value - low) / (high - low)
    state.brightness_fraction = min(1.0, max(0.0, fraction))
    return state


_CONTEXT_PARSERS = {
    "cpu_load": parse_cpu_load,
    "wifi_power": parse_wifi_power,
    "display_brightness": parse_brightness,
}


def parse_context(commands: dict) -> ContextState:
    """The context half of a sample. Missing or failed commands leave None."""
    state = ContextState()
    for name, parser in _CONTEXT_PARSERS.items():
        result = commands.get(name, {})
        if result.get("ok"):
            parser(result.get("stdout", ""), state)
    return state


def parse_sample(sample: dict) -> dict:
    """One stored sample as a flat dict of numbers, ready for a table row.

    Keys from ioreg keep their BatteryState names, keys from pmset are
    prefixed `macos_`, and the context fields keep their ContextState names.
    A command that failed in that sample leaves its fields as None, and
    `ioreg_ok` and `pmset_ok` say which battery half is missing. Schema 1
    samples have no context commands, so their context fields are all None.
    """
    commands = sample.get("commands", {})
    ioreg = commands.get("ioreg_battery", {})
    pmset = commands.get("pmset_batt", {})

    ioreg_ok = bool(ioreg.get("ok"))
    pmset_ok = bool(pmset.get("ok"))

    battery = parse_ioreg(ioreg.get("stdout", "") if ioreg_ok else "")
    macos = parse_pmset(pmset.get("stdout", "") if pmset_ok else "")

    row = {
        "sampled_at": sample.get("sampled_at"),
        "ioreg_ok": ioreg_ok,
        "pmset_ok": pmset_ok,
    }
    row.update(battery.to_dict())
    row.update({f"macos_{key}": value for key, value in macos.to_dict().items()})
    row.update(parse_context(commands).to_dict())
    return row
