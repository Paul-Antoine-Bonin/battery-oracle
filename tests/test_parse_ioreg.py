"""Tests for the ioreg parser, against fixtures shaped like real output.

The numbers in the fixtures are synthetic. They are chosen so the expected
values are easy to check by hand, not taken from any measurement.
"""

from pathlib import Path

import pytest

from battery.parse import ioreg_fields, parse_ioreg, signed_64

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def discharging():
    return parse_ioreg(fixture("ioreg_discharging.txt"))


@pytest.fixture
def charging():
    return parse_ioreg(fixture("ioreg_charging_no_raw_fields.txt"))


def test_signed_64_leaves_positive_values_alone():
    assert signed_64(2100) == 2100
    assert signed_64(0) == 0


def test_signed_64_converts_the_unsigned_discharge_value():
    assert signed_64(18446744073709550199) == -1417
    assert signed_64(2**64 - 1) == -1


def test_signed_64_boundary():
    assert signed_64(2**63 - 1) == 2**63 - 1
    assert signed_64(2**63) == -(2**63)


def test_discharge_amperage_is_negative(discharging):
    # The trap: read naively, this sample looks like an enormous charge.
    assert discharging.amperage_ma == -1000
    assert discharging.instant_amperage_ma == -1050


def test_charge_amperage_stays_positive(charging):
    assert charging.amperage_ma == 2100
    assert charging.instant_amperage_ma == 2150


def test_raw_capacity_fields_win_over_the_percentage_ones(discharging):
    # Apple silicon: MaxCapacity is 100, the mAh figures are under AppleRaw*.
    assert discharging.current_mah == 1250
    assert discharging.max_mah == 4000


def test_capacity_falls_back_when_raw_fields_are_absent(charging):
    assert charging.current_mah == 2400
    assert charging.max_mah == 4000


def test_charge_fraction_uses_the_current_maximum_not_design(discharging):
    assert discharging.charge_fraction == pytest.approx(1250 / 4000)
    assert discharging.charge_fraction != pytest.approx(1250 / 5000)


def test_health_is_maximum_over_design(discharging):
    assert discharging.health_fraction == pytest.approx(0.8)


def test_cycles_voltage_and_temperature(discharging):
    assert discharging.cycle_count == 500
    assert discharging.voltage_mv == 11400
    assert discharging.temperature_c == pytest.approx(30.5)


def test_power_is_signed_like_amperage(discharging, charging):
    assert discharging.power_w == pytest.approx(-11.4)
    assert charging.power_w == pytest.approx(2100 * 12300 / 1e6)


def test_flags(discharging, charging):
    assert discharging.is_charging is False
    assert discharging.external_connected is False
    assert charging.is_charging is True
    assert charging.external_connected is True


def test_nested_dicts_do_not_shadow_top_level_keys():
    # LegacyBatteryInfo carries its own Amperage and Voltage inline.
    fields = ioreg_fields(fixture("ioreg_discharging.txt"))
    assert fields["Voltage"] == 11400
    assert fields["BatteryData"].startswith("{")


def test_quoted_strings_are_unquoted():
    fields = ioreg_fields(fixture("ioreg_discharging.txt"))
    assert fields["DeviceName"] == "bq40z651"


def test_empty_text_gives_an_empty_state():
    state = parse_ioreg("")
    assert all(value is None for value in state.to_dict().values())


def test_missing_temperature_is_none_not_zero():
    text = fixture("ioreg_charging_no_raw_fields.txt").replace(
        '      "Temperature" = 3320\n', ""
    )
    assert parse_ioreg(text).temperature_c is None


def test_zero_maximum_does_not_divide():
    state = parse_ioreg('"CurrentCapacity" = 10\n"MaxCapacity" = 0\n')
    assert state.charge_fraction is None
