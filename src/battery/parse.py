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
