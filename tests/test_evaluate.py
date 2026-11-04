"""Tests for the scoring. Every number is made up to check the arithmetic."""

from datetime import datetime, timedelta, timezone

import pytest

from battery.dataset import Example
from battery.evaluate import (
    Scored,
    bound_misses,
    by_charge,
    compare,
    main,
    per_session,
    report_lines,
    scored_rows,
)
from battery.series import write_csv
from battery.target import target_fields


def example(index, session=1, minutes=100.0, censored=False):
    return Example(index, session, {}, minutes, censored)


def scored(session, actual, model, macos, charge=0.5):
    return Scored(0, session, charge, actual, model, macos)


def test_a_row_is_scored_only_with_an_outcome_and_both_estimates():
    rows = [
        {"macos_remaining_min": 90, "charge_fraction": 0.4},
        {"macos_remaining_min": None, "charge_fraction": 0.4},
        {"macos_remaining_min": 90, "charge_fraction": 0.4},
        {"macos_remaining_min": 90, "charge_fraction": 0.4},
    ]
    items = [
        example(0),
        example(1),  # macOS gave no estimate
        example(2),  # the model could not be fitted
        example(3, censored=True),  # only a lower bound
    ]

    out = scored_rows(rows, items, [110.0, 110.0, None, 110.0])

    assert out == [Scored(0, 1, 0.4, 100.0, 110.0, 90.0)]


def test_examples_are_looked_up_by_their_row_index():
    rows = [
        {"macos_remaining_min": 1, "charge_fraction": 0.9},  # on a charger, no example
        {"macos_remaining_min": 90, "charge_fraction": 0.4},
    ]

    out = scored_rows(rows, [example(1)], [100.0])

    assert out[0].macos_min == 90.0
    assert out[0].charge_fraction == 0.4


def test_predictions_must_line_up_with_the_examples():
    with pytest.raises(ValueError):
        scored_rows([{}], [example(0)], [])
    with pytest.raises(ValueError):
        bound_misses([{}], [example(0)], [])


def test_errors_are_absolute_on_both_sides():
    item = scored(1, actual=100.0, model=80.0, macos=130.0)

    assert item.model_error == 20.0
    assert item.macos_error == 30.0


def test_compare_with_nothing_scored_gives_no_means():
    out = compare([])

    assert out["rows"] == 0
    assert out["sessions"] == 0
    assert out["model_mae"] is None
    assert out["macos_mae"] is None
    assert out["model_session_mae"] is None


def test_the_row_mean_weighs_a_long_session_and_the_session_mean_does_not():
    long_session = [scored(1, 100.0, 110.0, 100.0)] * 3  # model off by 10
    short_session = [scored(2, 100.0, 150.0, 100.0)]  # model off by 50

    out = compare(long_session + short_session)

    assert out["rows"] == 4
    assert out["sessions"] == 2
    assert out["model_mae"] == 20.0
    assert out["model_session_mae"] == 30.0


def test_compare_counts_the_sessions_each_side_won_and_leaves_ties_out():
    items = [
        scored(1, 100.0, 105.0, 120.0),  # model
        scored(2, 100.0, 140.0, 110.0),  # macOS
        scored(3, 100.0, 90.0, 110.0),  # tie
    ]

    out = compare(items)

    assert out["sessions_model_better"] == 1
    assert out["sessions_macos_better"] == 1


def test_per_session_is_in_session_order():
    items = [scored(2, 100.0, 100.0, 100.0), scored(1, 100.0, 90.0, 80.0)]

    out = per_session(items)

    assert [s["session"] for s in out] == [1, 2]
    assert out[0] == {"session": 1, "rows": 1, "model_mae": 10.0, "macos_mae": 20.0}


def test_bound_misses_counts_estimates_below_the_bound_on_censored_rows_only():
    rows = [
        {"macos_remaining_min": 50},
        {"macos_remaining_min": 150},
        {"macos_remaining_min": 100},  # on the bound, not below it
        {"macos_remaining_min": None},
        {"macos_remaining_min": 50},
        {"macos_remaining_min": 50},
    ]
    items = [
        example(0, censored=True),
        example(1, censored=True),
        example(2, censored=True),
        example(3, censored=True),  # no macOS estimate
        example(4, censored=True),  # no prediction
        example(5),  # exact, scored elsewhere
    ]

    out = bound_misses(rows, items, [120.0, 90.0, 100.0, 10.0, None, 10.0])

    assert out == {"rows": 3, "model_below": 1, "macos_below": 1}


def test_a_band_includes_its_lower_edge():
    items = [scored(1, 100.0, 100.0, 100.0, charge=c) for c in (0.1, 0.19, 0.2)]

    out = by_charge(items)

    assert [(b["low_pct"], b["high_pct"], b["rows"]) for b in out] == [
        (10, 20, 2),
        (20, 30, 1),
    ]


def test_binary_rounding_does_not_move_a_row_into_the_band_below():
    assert 0.29 * 100 < 29  # the trap this guards against

    out = by_charge([scored(1, 100.0, 100.0, 100.0, charge=0.29)], band_pct=29)

    assert (out[0]["low_pct"], out[0]["high_pct"]) == (29, 58)


def test_a_full_battery_goes_in_the_top_band():
    items = [scored(1, 100.0, 100.0, 100.0, charge=c) for c in (0.95, 1.0)]

    out = by_charge(items)

    assert [(b["low_pct"], b["high_pct"], b["rows"]) for b in out] == [(90, 100, 2)]


def test_bands_that_do_not_divide_100_end_at_100():
    out = by_charge([scored(1, 100.0, 100.0, 100.0, charge=1.0)], band_pct=30)

    assert (out[0]["low_pct"], out[0]["high_pct"]) == (90, 100)


def test_a_band_reports_both_errors_and_its_sessions():
    items = [
        scored(1, 100.0, 110.0, 130.0, charge=0.42),
        scored(2, 100.0, 130.0, 110.0, charge=0.47),
    ]

    assert by_charge(items) == [
        {
            "low_pct": 40,
            "high_pct": 50,
            "rows": 2,
            "sessions": 2,
            "model_mae": 20.0,
            "macos_mae": 20.0,
        }
    ]


def test_a_row_with_no_charge_reading_is_left_out_of_the_bands():
    assert by_charge([scored(1, 100.0, 100.0, 100.0, charge=None)]) == []


@pytest.mark.parametrize("band_pct", [0, -10, 101])
def test_an_unusable_band_is_refused(band_pct):
    with pytest.raises(ValueError):
        by_charge([], band_pct)


def test_report_shows_both_estimators_and_the_bands():
    rows = [{"macos_remaining_min": 90, "charge_fraction": 0.45}]

    lines = report_lines(rows, [example(0)], [104.0])

    assert "scored_sessions         1" in lines
    assert "model_mae_min           4.0" in lines
    assert "macos_mae_min           10.0" in lines
    assert lines[-1].split() == ["40-50%", "1", "1", "4.0", "10.0"]


def test_report_with_nothing_scored_prints_dashes_and_no_band_table():
    lines = report_lines([{"macos_remaining_min": 90}], [example(0)], [None])

    assert "model_mae_min           -" in lines
    assert not any(line.startswith("charge") for line in lines)


def test_the_report_command_runs_from_a_written_table(tmp_path, capsys):
    pytest.importorskip("sklearn")

    start = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)

    def row(minute, mah, on_battery=True):
        return {
            "sampled_at": (start + timedelta(minutes=minute)).isoformat(),
            "ioreg_ok": True,
            "pmset_ok": True,
            "current_mah": mah,
            "max_mah": 3000,
            "charge_fraction": mah / 3000,
            "amperage_ma": -1500 if on_battery else 900,
            "external_connected": not on_battery,
            "macos_power_source": "Battery Power" if on_battery else "AC Power",
            "macos_state": "discharging" if on_battery else "charging",
            "macos_percent": round(mah / 30),
            "macos_remaining_min": 60 if on_battery else None,
        }

    def discharge(first_minute):
        """500 mAh down to 50 mAh, then a charger an hour later."""
        return [row(first_minute + 5 * k, 500 - 50 * k) for k in range(10)] + [
            row(first_minute + 105, 300, on_battery=False)
        ]

    rows = discharge(0) + discharge(300)
    for r, t in zip(rows, target_fields(rows)):
        r.update(t)
    table = write_csv(rows, tmp_path / "samples.csv")

    assert main(["--table", str(table), "report", "--band", "20"]) == 0

    out = capsys.readouterr().out
    assert "scored_sessions         2" in out
    assert "0-20%" in out


def test_the_report_command_says_when_there_is_no_table(tmp_path, capsys):
    assert main(["--table", str(tmp_path / "missing.csv"), "report"]) == 1
    assert "no table" in capsys.readouterr().out


@pytest.mark.parametrize("band", ["0", "101", "ten"])
def test_the_report_command_refuses_a_bad_band_before_reading_the_table(
    tmp_path, capsys, band
):
    with pytest.raises(SystemExit) as stopped:
        main(["--table", str(tmp_path / "missing.csv"), "report", "--band", band])
    assert stopped.value.code == 2
    assert "--band" in capsys.readouterr().err
