"""Tests for the macOS baseline columns. Synthetic samples built from fixtures."""

from pathlib import Path

from battery.baseline import (
    NO_ESTIMATE,
    NOT_DISCHARGING,
    ON_AC,
    PMSET_FAILED,
    POWER_MISMATCH,
    SCORABLE,
    STATUSES,
    baseline_fields,
    baseline_status,
    status_counts,
)
from battery.parse import parse_sample

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLED_AT = "2026-08-16T20:00:00+00:00"


def ok(name):
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return {"ok": True, "stdout": text, "error": "", "duration_s": 0.0}


def failed():
    return {"ok": False, "stdout": "", "error": "exit 1: boom", "duration_s": 0.0}


def row(pmset="pmset_discharging.txt", ioreg="ioreg_discharging.txt"):
    commands = {
        "pmset_batt": failed() if pmset is None else ok(pmset),
        "ioreg_battery": failed() if ioreg is None else ok(ioreg),
    }
    return parse_sample({"sampled_at": SAMPLED_AT, "schema": 3, "commands": commands})


def test_a_discharge_with_an_estimate_is_scorable():
    fields = baseline_fields(row())
    assert fields == {
        "baseline_status": SCORABLE,
        "macos_remaining_min": 75,
        "macos_empty_at": "2026-08-16T21:15:00+00:00",
    }


def test_the_empty_instant_keeps_the_sample_offset():
    parsed = row()
    parsed["sampled_at"] = "2026-08-16T22:00:00+02:00"
    assert baseline_fields(parsed)["macos_empty_at"] == "2026-08-16T23:15:00+02:00"


def test_no_estimate_stays_a_gap():
    fields = baseline_fields(row(pmset="pmset_no_estimate.txt"))
    assert fields["baseline_status"] == NO_ESTIMATE
    assert fields["macos_remaining_min"] is None
    assert fields["macos_empty_at"] is None


def test_a_charging_estimate_is_a_time_to_full_not_a_prediction():
    parsed = row(pmset="pmset_charging.txt", ioreg="ioreg_charging_no_raw_fields.txt")
    assert parsed["macos_estimate_min"] == 48
    assert baseline_status(parsed) == ON_AC
    assert baseline_fields(parsed)["macos_remaining_min"] is None


def test_charged_and_not_charging_are_on_ac():
    assert baseline_status(row(pmset="pmset_charged.txt")) == ON_AC
    assert baseline_status(row(pmset="pmset_ac_not_charging.txt")) == ON_AC


def test_a_failed_pmset_cannot_be_scored():
    assert baseline_status(row(pmset=None)) == PMSET_FAILED


def test_a_plug_event_between_the_two_commands_is_flagged():
    # pmset says battery, ioreg already sees the charger and a positive current.
    parsed = row(ioreg="ioreg_charging_no_raw_fields.txt")
    assert parsed["macos_has_estimate"]
    assert baseline_status(parsed) == POWER_MISMATCH
    assert baseline_fields(parsed)["macos_empty_at"] is None


def test_a_positive_current_alone_is_a_mismatch():
    parsed = row()
    parsed["external_connected"] = False
    parsed["amperage_ma"] = 350
    assert baseline_status(parsed) == POWER_MISMATCH


def test_the_unsigned_discharge_current_is_not_a_mismatch():
    # ioreg_discharging.txt stores Amperage unsigned. After the two's
    # complement conversion it is negative, so this must stay scorable.
    parsed = row()
    assert parsed["amperage_ma"] < 0
    assert baseline_status(parsed) == SCORABLE


def test_a_failed_ioreg_leaves_the_pmset_reading_as_is():
    assert baseline_status(row(ioreg=None)) == SCORABLE


def test_a_battery_state_other_than_discharging_is_not_scored():
    parsed = row()
    parsed["macos_state"] = "finishing charge"
    assert baseline_status(parsed) == NOT_DISCHARGING


def test_counts_list_every_status():
    rows = [row(), row(), row(pmset="pmset_no_estimate.txt"), row(pmset=None)]
    counts = status_counts(rows)

    assert list(counts) == list(STATUSES)
    assert counts[SCORABLE] == 2
    assert counts[NO_ESTIMATE] == 1
    assert counts[PMSET_FAILED] == 1
    assert counts[ON_AC] == 0
    assert sum(counts.values()) == len(rows)
