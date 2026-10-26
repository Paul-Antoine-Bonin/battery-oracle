"""Tests for the pmset parser and the per-sample row, on synthetic fixtures."""

from pathlib import Path

from battery.parse import parse_pmset, parse_sample

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_discharging_with_an_estimate():
    state = parse_pmset(fixture("pmset_discharging.txt"))
    assert state.power_source == "Battery Power"
    assert state.percent == 31
    assert state.state == "discharging"
    assert state.estimate_min == 75
    assert state.has_estimate


def test_charging_estimate_is_time_to_full():
    state = parse_pmset(fixture("pmset_charging.txt"))
    assert state.power_source == "AC Power"
    assert state.state == "charging"
    assert state.estimate_min == 48


def test_no_estimate_stays_none_not_zero():
    # Filling this in would give the baseline an answer macOS never gave.
    state = parse_pmset(fixture("pmset_no_estimate.txt"))
    assert state.percent == 88
    assert state.state == "discharging"
    assert state.estimate_min is None
    assert not state.has_estimate


def test_ac_attached_not_charging():
    state = parse_pmset(fixture("pmset_ac_not_charging.txt"))
    assert state.percent == 80
    assert state.state == "AC attached"
    assert state.estimate_min is None


def test_charged_reports_zero_minutes():
    state = parse_pmset(fixture("pmset_charged.txt"))
    assert state.state == "charged"
    assert state.estimate_min == 0
    assert state.has_estimate


def test_hours_are_converted_to_minutes():
    text = " -InternalBattery-0 (id=1)\t95%; discharging; 10:05 remaining present: true"
    assert parse_pmset(text).estimate_min == 605


def test_empty_text():
    state = parse_pmset("")
    assert state.percent is None
    assert state.power_source is None


def stored(ioreg_ok=True, pmset_ok=True):
    return {
        "sampled_at": "2026-08-16T20:00:00+00:00",
        "schema": 1,
        "commands": {
            "pmset_batt": {
                "ok": pmset_ok,
                "stdout": fixture("pmset_discharging.txt") if pmset_ok else "",
                "error": "" if pmset_ok else "exit 1",
            },
            "ioreg_battery": {
                "ok": ioreg_ok,
                "stdout": fixture("ioreg_discharging.txt") if ioreg_ok else "",
                "error": "" if ioreg_ok else "exit 1",
            },
        },
    }


def test_parse_sample_merges_both_sources():
    row = parse_sample(stored())
    assert row["sampled_at"] == "2026-08-16T20:00:00+00:00"
    assert row["amperage_ma"] == -1000
    assert row["cycle_count"] == 500
    assert row["macos_estimate_min"] == 75
    assert row["macos_percent"] == 31
    assert row["ioreg_ok"] and row["pmset_ok"]


def test_parse_sample_keeps_the_half_that_worked():
    row = parse_sample(stored(ioreg_ok=False))
    assert not row["ioreg_ok"]
    assert row["amperage_ma"] is None
    assert row["macos_estimate_min"] == 75


def test_failed_command_output_is_not_parsed():
    sample = stored()
    sample["commands"]["pmset_batt"]["ok"] = False
    assert parse_sample(sample)["macos_percent"] is None


def test_every_row_has_the_same_keys():
    # A table needs the same columns whichever command failed.
    flags = [(a, b) for a in (True, False) for b in (True, False)]
    keys = {frozenset(parse_sample(stored(a, b))) for a, b in flags}
    assert len(keys) == 1


def test_a_sample_without_commands():
    row = parse_sample({"sampled_at": "2026-08-16T20:00:00+00:00"})
    assert row["ioreg_ok"] is False and row["pmset_ok"] is False
