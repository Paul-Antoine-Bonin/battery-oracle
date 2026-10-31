"""Tests for the discharge sessions. Rows are synthetic, built by hand."""

from datetime import datetime, timedelta, timezone

from battery.sessions import (
    EMPTY,
    END_REASONS,
    GAP,
    OPEN,
    PLUGGED,
    charge_of,
    find_sessions,
    is_discharging,
    session_counts,
)

START = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)


def row(minute, charge, on_battery=True, ioreg_ok=True):
    """A parsed row reduced to the fields the sessions look at."""
    return {
        "sampled_at": (START + timedelta(minutes=minute)).isoformat(),
        "ioreg_ok": ioreg_ok,
        "pmset_ok": True,
        "charge_fraction": charge if ioreg_ok else None,
        "amperage_ma": (-1400 if on_battery else 900) if ioreg_ok else None,
        "external_connected": (not on_battery) if ioreg_ok else None,
        "macos_power_source": "Battery Power" if on_battery else "AC Power",
        "macos_state": "discharging" if on_battery else "charging",
        "macos_percent": round(charge * 100),
    }


def drain(start_minute, start_charge, count, step=5, per_step=0.01):
    return [
        row(start_minute + k * step, round(start_charge - k * per_step, 4))
        for k in range(count)
    ]


def test_a_charger_ends_the_session_as_plugged():
    rows = drain(0, 0.60, 4) + [row(20, 0.57, on_battery=False)]
    sessions = find_sessions(rows)

    assert len(sessions) == 1
    assert sessions[0].rows == [0, 1, 2, 3]
    assert sessions[0].end_reason == PLUGGED


def test_running_out_ends_the_session_as_empty():
    # The last sample reads 2%, then nothing for hours, then the Mac is back
    # on a charger.
    rows = drain(0, 0.05, 4) + [row(300, 0.10, on_battery=False)]
    sessions = find_sessions(rows)

    assert sessions[0].end_reason == EMPTY
    assert sessions[0].last == 3


def test_plugging_in_at_two_percent_is_not_empty():
    rows = drain(0, 0.05, 4) + [row(20, 0.03, on_battery=False)]
    assert find_sessions(rows)[0].end_reason == PLUGGED


def test_a_long_gap_with_charge_left_breaks_the_session():
    rows = drain(0, 0.80, 3) + drain(120, 0.76, 3)
    sessions = find_sessions(rows)

    assert [s.end_reason for s in sessions] == [GAP, OPEN]
    assert sessions[1].rows == [3, 4, 5]


def test_charge_going_up_across_a_gap_means_it_was_plugged():
    # Asleep on a charger: both sides of the gap look like a discharge.
    rows = drain(0, 0.40, 3) + drain(240, 0.90, 3)
    assert find_sessions(rows)[0].end_reason == PLUGGED


def test_charge_going_up_between_close_samples_splits_the_session():
    rows = drain(0, 0.40, 3) + drain(15, 0.45, 3)
    sessions = find_sessions(rows)

    assert [s.end_reason for s in sessions] == [PLUGGED, OPEN]
    assert sessions[1].first == 3


def test_a_session_still_running_is_open():
    assert find_sessions(drain(0, 0.50, 5))[0].end_reason == OPEN


def test_rows_on_a_charger_belong_to_no_session():
    rows = [row(0, 0.9, on_battery=False)] + drain(5, 0.9, 2)
    rows += [row(15, 0.9, on_battery=False), row(20, 0.9, on_battery=False)]
    sessions = find_sessions(rows)

    assert [s.rows for s in sessions] == [[1, 2]]


def test_sessions_are_numbered_in_order():
    rows = drain(0, 0.8, 2) + [row(10, 0.8, on_battery=False)] + drain(15, 0.8, 2)
    assert [s.index for s in find_sessions(rows)] == [0, 1]


def test_the_gap_threshold_can_be_changed():
    rows = drain(0, 0.80, 2) + drain(17, 0.78, 2)
    assert len(find_sessions(rows)) == 1
    assert len(find_sessions(rows, max_gap_s=600)) == 2


def test_discharging_follows_ioreg_when_it_succeeded():
    on_ac = row(0, 0.5, on_battery=False)
    on_ac["macos_power_source"] = "Battery Power"
    on_ac["macos_state"] = "discharging"
    assert not is_discharging(on_ac)


def test_discharging_falls_back_to_pmset_when_ioreg_failed():
    assert is_discharging(row(0, 0.5, ioreg_ok=False))
    assert not is_discharging(row(0, 0.5, on_battery=False, ioreg_ok=False))


def test_a_row_with_neither_reading_is_not_discharging():
    broken = row(0, 0.5)
    broken["ioreg_ok"] = False
    broken["pmset_ok"] = False
    assert not is_discharging(broken)


def test_charge_falls_back_to_the_macos_percentage():
    assert charge_of(row(0, 0.42)) == 0.42
    assert charge_of(row(0, 0.42, ioreg_ok=False)) == 0.42
    assert charge_of({"charge_fraction": None, "macos_percent": None}) is None


def test_counts_list_every_reason():
    rows = drain(0, 0.03, 2) + drain(300, 0.70, 2)
    counts = session_counts(find_sessions(rows))

    assert list(counts) == list(END_REASONS)
    assert counts == {EMPTY: 1, PLUGGED: 0, GAP: 0, OPEN: 1}


def test_no_rows_no_sessions():
    assert find_sessions([]) == []
