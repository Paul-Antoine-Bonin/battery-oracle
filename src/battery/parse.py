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


def parse_sample(sample: dict) -> dict:
    """One stored sample as a flat dict of numbers, ready for a table row.

    Keys from ioreg keep their BatteryState names, keys from pmset are
    prefixed `macos_`. A command that failed in that sample leaves its fields
    as None, and `ioreg_ok` and `pmset_ok` say which half is missing.
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
    return row
