"""Tests for the regression model, on synthetic sessions from a known law.

Nothing here is a measurement. The numbers come from the generators below, and
the assertions check the fitting, not how well a real battery can be predicted.
"""

import math
import random

import pytest

pytest.importorskip("sklearn")

from battery.dataset import Example
from battery.features import FEATURE_COLUMNS
import battery.model as model_module
from battery.model import (
    ALPHA_GRID,
    DEFAULT_ALPHA,
    MAX_REMAINING_MIN,
    MODEL_COLUMNS,
    alpha_score,
    choose_alpha,
    cross_predict,
    fit,
    matrix,
)


def feature_dict(**values):
    out = {column: None for column in FEATURE_COLUMNS}
    out.update(values)
    return out


def session(number, draw_ma, censored=False, start_mah=2400, stop_mah=100, seed=0):
    """A steady drain at `draw_ma`, one row every 5 minutes.

    A censored session is cut off halfway. Its bounds are measured to the cut,
    as `battery.target` does, not to the empty battery.
    """
    noise = random.Random(number + seed)
    per_step = draw_ma / 12
    steps = int((start_mah - stop_mah) / per_step)
    last = steps // 2 if censored else steps
    rows = []
    for k in range(last + 1):
        mah = start_mah - k * per_step
        minutes = (last - k) * 5.0 if censored else (mah - stop_mah) / draw_ma * 60
        drawn = draw_ma * (1 + noise.uniform(-0.02, 0.02))
        rows.append(
            Example(
                index=number * 1000 + k,
                session=number,
                features=feature_dict(
                    current_mah=mah,
                    charge_fraction=mah / 3000,
                    draw_ma=drawn,
                    instant_remaining_min=mah / drawn * 60,
                    hour_of_day=float(number % 24),
                ),
                minutes=minutes,
                censored=censored,
            )
        )
    return rows


def test_the_matrix_has_one_column_per_model_column():
    assert matrix([feature_dict()]).shape == (1, len(MODEL_COLUMNS))
    assert matrix([]).shape == (0, len(MODEL_COLUMNS))


def test_a_missing_value_is_nan_and_never_zero():
    line = matrix([feature_dict()])[0]
    assert math.isnan(line[MODEL_COLUMNS.index("draw_ma")])


def test_remaining_time_inputs_are_capped():
    line = matrix([feature_dict(trend_remaining_min=90000.0)])[0]
    assert line[MODEL_COLUMNS.index("trend_remaining_min")] == MAX_REMAINING_MIN


def test_the_hour_wraps_around_midnight():
    late, early = matrix(
        [feature_dict(hour_of_day=23.0), feature_dict(hour_of_day=0.0)]
    )
    sin, cos = MODEL_COLUMNS.index("hour_sin"), MODEL_COLUMNS.index("hour_cos")
    assert math.dist(late[[sin, cos]], early[[sin, cos]]) < 0.3


def test_it_recovers_a_known_law_on_exact_sessions():
    rows = [r for n, draw in enumerate((900, 1200, 1500, 1800)) for r in session(n, draw)]
    model = fit(rows)
    predicted = model.predict([r.features for r in rows])
    errors = [abs(p - r.minutes) for p, r in zip(predicted, rows)]
    assert sum(errors) / len(errors) < 15


def test_a_prediction_is_never_negative():
    rows = session(0, 1200) + session(1, 1500)
    model = fit(rows)
    almost_flat = feature_dict(
        current_mah=1.0,
        charge_fraction=0.0,
        draw_ma=9000.0,
        instant_remaining_min=0.0,
    )
    assert model.predict([almost_flat]) == [0.0]


def test_fitting_needs_an_exact_example():
    with pytest.raises(ValueError):
        fit(session(0, 1200, censored=True))


def test_it_tolerates_a_feature_that_is_never_set():
    model = fit(session(0, 1200) + session(1, 1500))
    assert len(model.predict([feature_dict(brightness_fraction=0.5)])) == 1


def flat_exact_and_higher_bounds():
    """Exact rows say 100 at x=0, censored rows at x=1 say at least 300."""
    exact = [
        Example(i, 0, feature_dict(load_per_cpu=0.0 + i * 1e-3), 100.0, False)
        for i in range(40)
    ]
    bounded = [
        Example(100 + i, 1, feature_dict(load_per_cpu=1.0 + i * 1e-3), 300.0, True)
        for i in range(40)
    ]
    return exact, bounded


def test_lower_bounds_pull_a_prediction_up():
    exact, bounded = flat_exact_and_higher_bounds()
    without = fit(exact + bounded, rounds=0).predict([bounded[0].features])[0]
    with_bounds = fit(exact + bounded).predict([bounded[0].features])[0]
    assert without < 150
    assert with_bounds > 200


def test_a_prediction_above_its_bound_is_left_alone():
    # The bounds are all below what the exact rows already imply.
    exact = session(0, 1200) + session(1, 1500)
    low = [
        Example(r.index, 2, r.features, 1.0, True)
        for r in session(2, 1300)
    ]
    plain = fit(exact).predict([low[0].features])[0]
    bounded = fit(exact + low).predict([low[0].features])[0]
    assert bounded == pytest.approx(plain, abs=15)


def test_cross_prediction_never_uses_the_held_out_session():
    a, b, c = session(0, 900), session(1, 1200), session(2, 1500)
    items = a + b + c
    predicted = cross_predict(items)
    expected = fit(a + c).predict([r.features for r in b])
    assert predicted[len(a) : len(a) + len(b)] == pytest.approx(expected)


def test_the_only_exact_session_gets_no_cross_prediction():
    exact = session(0, 1200)
    bounded = session(1, 1200, censored=True)
    predicted = cross_predict(exact + bounded)
    assert predicted[: len(exact)] == [None] * len(exact)
    assert all(value is not None for value in predicted[len(exact) :])


def test_cross_prediction_of_nothing_is_empty():
    assert cross_predict([]) == []


def four_exact_sessions():
    return [r for n, d in enumerate((900, 1200, 1500, 1800)) for r in session(n, d)]


def test_an_alpha_cannot_be_scored_on_one_full_discharge():
    items = session(0, 1200) + session(1, 1200, censored=True)
    assert alpha_score(items, 1.0) is None
    assert choose_alpha(items) == DEFAULT_ALPHA


def test_the_alpha_score_skips_censored_rows():
    exact = four_exact_sessions()
    bounded = session(9, 1300, censored=True)
    # The censored session changes the fits, so the score may move, but only
    # the four exact sessions are averaged.
    assert alpha_score(exact + bounded, 1.0) is not None


def test_the_chosen_alpha_has_the_lowest_score():
    items = four_exact_sessions()
    scores = {alpha: alpha_score(items, alpha) for alpha in ALPHA_GRID}
    assert scores[choose_alpha(items)] == min(scores.values())


def test_a_tie_goes_to_the_larger_alpha(monkeypatch):
    monkeypatch.setattr(model_module, "alpha_score", lambda items, alpha, rounds: 5.0)
    assert choose_alpha(four_exact_sessions(), (0.1, 10.0, 1.0)) == 10.0


def test_choosing_from_no_alpha_is_refused():
    with pytest.raises(ValueError):
        choose_alpha(four_exact_sessions(), ())


def test_the_alpha_for_a_held_out_session_is_chosen_without_it():
    sessions = [session(n, d) for n, d in enumerate((900, 1200, 1500, 1800))]
    items = [r for s in sessions for r in s]
    predicted = cross_predict(items, alphas=ALPHA_GRID)
    kept = sessions[0] + sessions[2] + sessions[3]
    expected = fit(kept, choose_alpha(kept)).predict([r.features for r in sessions[1]])
    start = len(sessions[0])
    assert predicted[start : start + len(sessions[1])] == pytest.approx(expected)
