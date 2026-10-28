"""Tests for the context parsers, on synthetic fixtures."""

from pathlib import Path

import pytest

from battery.parse import (
    parse_brightness,
    parse_context,
    parse_cpu_load,
    parse_foreground_app,
    parse_sample,
    parse_wifi_power,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def ok(name):
    return {"ok": True, "stdout": fixture(name), "error": "", "duration_s": 0.0}


def failed(error):
    return {"ok": False, "stdout": "", "error": error, "duration_s": 0.0}


def test_cpu_load_reads_all_three_averages_and_the_core_count():
    state = parse_cpu_load(fixture("sysctl_cpu_load.txt"))
    assert (state.load_1m, state.load_5m, state.load_15m) == (2.15, 1.98, 1.87)
    assert state.logical_cpus == 8
    assert state.load_per_cpu == pytest.approx(2.15 / 8)


def test_cpu_load_without_a_core_count_has_no_per_core_figure():
    state = parse_cpu_load("{ 1.00 0.50 0.25 }\n")
    assert state.load_1m == 1.0
    assert state.logical_cpus is None
    assert state.load_per_cpu is None


def test_wifi_on_and_off():
    assert parse_wifi_power(fixture("networksetup_wifi_on.txt")).wifi_on is True
    assert parse_wifi_power(fixture("networksetup_wifi_off.txt")).wifi_on is False


def test_wifi_error_text_is_unknown_not_off():
    state = parse_wifi_power("en0 is not a Wi-Fi interface.\n")
    assert state.wifi_on is None


def test_brightness_is_a_fraction_of_the_panel_range():
    state = parse_brightness(fixture("ioreg_display_brightness.txt"))
    assert state.brightness_fraction == pytest.approx(16384 / 65536)


def test_brightness_ignores_linear_brightness_and_key_order():
    text = '"IODisplayParameters" = {"linear-brightness"={"value"=9,"max"=10,"min"=0},"brightness"={"max"=200,"value"=150,"min"=100}}'
    assert parse_brightness(text).brightness_fraction == pytest.approx(0.5)


def test_no_built_in_display_is_unknown_not_dark():
    # Lid closed or external display only: nothing matches.
    assert parse_brightness("").brightness_fraction is None


def test_a_degenerate_brightness_range_is_unknown():
    text = '"brightness"={"min"=5,"value"=5,"max"=5}'
    assert parse_brightness(text).brightness_fraction is None


def test_foreground_app_is_the_bundle_id():
    state = parse_foreground_app(fixture("lsappinfo_foreground.txt"))
    assert state.foreground_app == "com.example.editor"


def test_nothing_in_front_is_unknown():
    # At the login window lsappinfo resolves no application.
    assert parse_foreground_app("").foreground_app is None
    assert parse_foreground_app("[ NULL ]\n").foreground_app is None


def test_parse_context_skips_failed_commands():
    state = parse_context(
        {
            "cpu_load": ok("sysctl_cpu_load.txt"),
            "wifi_power": failed("exit 4"),
            "display_brightness": ok("ioreg_display_brightness.txt"),
        }
    )
    assert state.load_1m == 2.15
    assert state.wifi_on is None
    assert state.brightness_fraction == pytest.approx(0.25)


def test_a_schema_2_sample_row_carries_the_context():
    sample = {
        "sampled_at": "2026-08-16T20:00:00+00:00",
        "schema": 2,
        "commands": {
            "pmset_batt": ok("pmset_discharging.txt"),
            "ioreg_battery": ok("ioreg_discharging.txt"),
            "cpu_load": ok("sysctl_cpu_load.txt"),
            "wifi_power": ok("networksetup_wifi_on.txt"),
            "display_brightness": ok("ioreg_display_brightness.txt"),
        },
    }
    row = parse_sample(sample)
    assert row["foreground_app"] is None
    assert row["amperage_ma"] < 0
    assert row["macos_estimate_min"] == 75
    assert row["load_per_cpu"] == pytest.approx(2.15 / 8)
    assert row["wifi_on"] is True
    assert row["brightness_fraction"] == pytest.approx(0.25)


def test_a_schema_1_sample_row_has_empty_context():
    sample = {
        "sampled_at": "2026-08-16T20:00:00+00:00",
        "schema": 1,
        "commands": {
            "pmset_batt": ok("pmset_discharging.txt"),
            "ioreg_battery": ok("ioreg_discharging.txt"),
        },
    }
    row = parse_sample(sample)
    assert row["macos_percent"] == 31
    for key in (
        "load_1m", "logical_cpus", "wifi_on", "brightness_fraction", "foreground_app"
    ):
        assert key in row and row[key] is None


def test_a_schema_3_sample_row_carries_the_foreground_app():
    sample = {
        "sampled_at": "2026-08-16T20:00:00+00:00",
        "schema": 3,
        "commands": {
            "pmset_batt": ok("pmset_discharging.txt"),
            "ioreg_battery": ok("ioreg_discharging.txt"),
            "foreground_app": ok("lsappinfo_foreground.txt"),
        },
    }
    assert parse_sample(sample)["foreground_app"] == "com.example.editor"


def test_a_failed_foreground_app_command_leaves_none():
    state = parse_context({"foreground_app": failed("exit 127")})
    assert state.foreground_app is None
