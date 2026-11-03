"""Tests for the scoring. Every number is made up to check the arithmetic."""

import pytest

from battery.dataset import Example
from battery.evaluate import Scored, bound_misses, compare, per_session, scored_rows


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
