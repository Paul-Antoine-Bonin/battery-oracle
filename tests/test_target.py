"""Tests for the reconstructed target. Rows are synthetic, built by hand."""

from datetime import datetime, timedelta, timezone

from battery.sessions import EMPTY, GAP, OPEN, PLUGGED
from battery.target import (
    CENSORED,
    EXACT,
    NONE,
    TARGET_COLUMNS,
    TARGET_STATUSES,
    target_counts,
    target_fields,
)

START = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)


def row(minute, charge, on_battery=True, start=START):
    return {
        "sampled_at": (start + timedelta(minutes=minute)).isoformat(),
        "ioreg_ok": True,
        "pmset_ok": True,
        "charge_fraction": charge,
        "amperage_ma": -1400 if on_battery else 900,
        "external_connected": not on_battery,
        "macos_power_source": "Battery Power" if on_battery else "AC Power",
        "macos_state": "discharging" if on_battery else "charging",
        "macos_percent": round(charge * 100),
    }


def drain(start_minute, start_charge, count, per_step=0.01, start=START):
    return [
        row(start_minute + 5 * k, round(start_charge - k * per_step, 4), start=start)
        for k in range(count)
    ]


def test_a_session_that_ran_out_gets_an_exact_target():
    rows = drain(0, 0.06, 5) + [row(400, 0.20, on_battery=False)]
    fields = target_fields(rows)

    assert [f["target_status"] for f in fields] == [EXACT] * 5 + [NONE]
    assert [f["actual_remaining_min"] for f in fields[:5]] == [20, 15, 10, 5, 0]
    assert all(f["session_end_reason"] == EMPTY for f in fields[:5])
    assert fields[0]["empty_at"] == (START + timedelta(minutes=20)).isoformat()


def test_an_exact_row_also_carries_the_lower_bound():
    fields = target_fields(drain(0, 0.04, 3) + drain(300, 0.9, 1))
    assert fields[0]["remaining_at_least_min"] == fields[0]["actual_remaining_min"]


def test_a_plugged_session_only_gets_a_lower_bound():
    rows = drain(0, 0.60, 4) + [row(20, 0.58, on_battery=False)]
    fields = target_fields(rows)

    assert [f["target_status"] for f in fields[:4]] == [CENSORED] * 4
    assert [f["remaining_at_least_min"] for f in fields[:4]] == [15, 10, 5, 0]
    assert all(f["actual_remaining_min"] is None for f in fields)
    assert all(f["empty_at"] is None for f in fields)
    assert fields[0]["session_end_reason"] == PLUGGED


def test_gap_and_open_sessions_are_censored_too():
    rows = drain(0, 0.80, 3) + drain(200, 0.70, 3)
    fields = target_fields(rows)

    assert [f["session_end_reason"] for f in fields] == [GAP] * 3 + [OPEN] * 3
    assert {f["target_status"] for f in fields} == {CENSORED}


def test_rows_on_a_charger_have_no_target():
    fields = target_fields([row(0, 0.9, on_battery=False)])
    assert fields == [
        {
            "session_id": None,
            "session_end_reason": None,
            "target_status": NONE,
            "empty_at": None,
            "actual_remaining_min": None,
            "remaining_at_least_min": None,
        }
    ]
    assert list(fields[0]) == list(TARGET_COLUMNS)


def test_each_row_points_at_its_own_session():
    rows = drain(0, 0.8, 2) + [row(10, 0.8, on_battery=False)] + drain(15, 0.8, 2)
    assert [f["session_id"] for f in target_fields(rows)] == [0, 0, None, 1, 1]


def test_the_empty_instant_keeps_the_sample_offset():
    paris = timezone(timedelta(hours=2))
    local = START.astimezone(paris)
    rows = drain(0, 0.04, 2, start=local) + drain(300, 0.9, 1, start=local)
    assert target_fields(rows)[0]["empty_at"].endswith("+02:00")


def test_counts_list_every_status():
    rows = drain(0, 0.04, 2) + [row(300, 0.5, on_battery=False)] + drain(305, 0.5, 3)
    counts = target_counts(target_fields(rows))

    assert list(counts) == list(TARGET_STATUSES)
    assert counts == {EXACT: 2, CENSORED: 3, NONE: 1}


def test_no_rows_no_targets():
    assert target_fields([]) == []
