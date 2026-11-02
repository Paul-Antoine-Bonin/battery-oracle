"""Tests for the training examples. Rows are synthetic, built by hand."""

from datetime import datetime, timedelta, timezone

import pytest

from battery.dataset import example_counts, examples, session_folds
from battery.series import COLUMNS, read_csv, write_csv
from battery.target import target_fields

START = datetime(2026, 8, 16, 20, 0, tzinfo=timezone(timedelta(hours=2)))
MAX_MAH = 3000


def row(minute, mah, on_battery=True):
    return {
        "sampled_at": (START + timedelta(minutes=minute)).isoformat(),
        "ioreg_ok": True,
        "pmset_ok": True,
        "current_mah": mah,
        "max_mah": MAX_MAH,
        "charge_fraction": mah / MAX_MAH,
        "amperage_ma": -1500 if on_battery else 900,
        "external_connected": not on_battery,
        "macos_power_source": "Battery Power" if on_battery else "AC Power",
        "macos_state": "discharging" if on_battery else "charging",
        "macos_percent": round(mah / MAX_MAH * 100),
    }


def with_target(rows):
    for r, t in zip(rows, target_fields(rows)):
        r.update(t)
    return rows


def full_discharge(start_minute=0):
    """500 mAh down to 50 mAh, 2%, then nothing until a charger an hour later."""
    return [row(start_minute + 5 * k, 500 - 50 * k) for k in range(10)] + [
        row(start_minute + 105, 300, on_battery=False)
    ]


def unplugged_day():
    """A charger row, then a session that a charger ends with charge left."""
    return (
        [row(0, 2000, on_battery=False)]
        + [row(5 + 5 * k, 2000 - 50 * k) for k in range(4)]
        + [row(25, 1900, on_battery=False)]
    )


def test_a_session_that_ran_out_gives_exact_examples():
    items = examples(with_target(full_discharge()))
    assert len(items) == 10
    assert not any(item.censored for item in items)
    assert items[0].minutes == pytest.approx(45)
    assert items[-1].minutes == pytest.approx(0)


def test_a_session_cut_short_gives_lower_bounds():
    items = examples(with_target(unplugged_day()))
    assert len(items) == 4
    assert all(item.censored for item in items)
    assert items[0].minutes == pytest.approx(15)


def test_rows_on_a_charger_give_no_example():
    rows = with_target(unplugged_day())
    assert [item.index for item in examples(rows)] == [1, 2, 3, 4]


def test_an_example_keeps_the_features_of_its_row():
    rows = with_target(full_discharge())
    items = examples(rows)
    assert items[3].features["current_mah"] == rows[3]["current_mah"]


def test_examples_survive_the_csv_round_trip(tmp_path):
    rows = with_target(full_discharge())
    for r in rows:
        r.update({column: r.get(column) for column in COLUMNS})
    table = write_csv(rows, tmp_path / "samples.csv")
    items = examples(read_csv(table))
    assert [item.minutes for item in items] == [
        item.minutes for item in examples(rows)
    ]


def fold_rows():
    rows = unplugged_day() + full_discharge(start_minute=120)
    return with_target(rows)


def test_a_session_is_never_on_both_sides_of_a_fold():
    items = examples(fold_rows())
    folds = list(session_folds(items))
    assert len(folds) == 2
    for session, train, test in folds:
        assert {items[p].session for p in test} == {session}
        assert session not in {items[p].session for p in train}
        assert sorted(train + test) == list(range(len(items)))


def test_the_counts_separate_rows_from_sessions():
    assert example_counts(examples(fold_rows())) == {
        "examples": 14,
        "exact": 10,
        "censored": 4,
        "sessions": 2,
        "sessions_exact": 1,
    }


def test_no_rows_means_no_examples_and_no_folds():
    assert examples([]) == []
    assert list(session_folds([])) == []
    assert example_counts([])["sessions"] == 0
