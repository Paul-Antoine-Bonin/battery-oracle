"""The regression model: time left until empty from the drain rate and context.

Needs the `analysis` extra. The collector and the parsers do not import this
module, so they keep running on the standard library alone.

The model is a ridge regression. It is deliberately plain: full discharges
are rare, so there will be few independent sessions to learn from, and a small
linear model is harder to fool itself with than a flexible one.

Inputs are prepared in `matrix`:

- Missing values get the median of the training rows, plus a column saying the
  value was missing, so a missing Wi-Fi reading is not read as a typical one.
- `hour_of_day` becomes a sine and a cosine, so 23:00 sits next to 00:00.
- `instant_remaining_min` and `trend_remaining_min` are capped at a day. They
  divide by the drain, so a nearly idle machine gives huge values that a linear
  fit would chase.

Censored rows only say the battery lasted at least so long, so `fit` cannot
treat them as answers. It starts from the exact rows alone. Then, for a few
rounds, each censored row takes as its target the larger of the current
prediction and its lower bound, and the model is refitted. A prediction above
the bound costs nothing and one below it is pulled up. Without this, the model
would learn only from days the battery ran flat, which are not typical days.

`cross_predict` is what any score must come from. Each row is predicted by a
model that never saw its session. Scoring `fit` on its own training rows would
report a number no live prediction could reach.

The ridge strength can be fixed or chosen from a list with `choose_alpha`. The
choice holds out one session at a time too, and scores only exact rows, since
a censored row has no error to measure. Inside `cross_predict` it runs again
for every held out session, on the kept sessions alone, so the session being
predicted never weighs on its own alpha. Choosing once on all the data and
then scoring would leak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import Example, session_folds
from .features import FEATURE_COLUMNS

DEFAULT_ALPHA = 1.0
DEFAULT_ROUNDS = 5
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
MAX_REMAINING_MIN = 1440.0

CAPPED_COLUMNS = ("instant_remaining_min", "trend_remaining_min")
PLAIN_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c != "hour_of_day")
MODEL_COLUMNS = PLAIN_COLUMNS + ("hour_sin", "hour_cos")


def _vector(item: Mapping[str, Optional[float]]) -> List[float]:
    values = []
    for column in PLAIN_COLUMNS:
        value = item.get(column)
        if value is None:
            values.append(math.nan)
        elif column in CAPPED_COLUMNS:
            values.append(min(value, MAX_REMAINING_MIN))
        else:
            values.append(value)
    hour = item.get("hour_of_day")
    if hour is None:
        values.extend((math.nan, math.nan))
    else:
        angle = 2 * math.pi * hour / 24
        values.extend((math.sin(angle), math.cos(angle)))
    return values


def matrix(items: Sequence[Mapping[str, Optional[float]]]) -> np.ndarray:
    """One line per feature dict, one column per name in `MODEL_COLUMNS`."""
    return np.array([_vector(item) for item in items], dtype=float).reshape(
        len(items), len(MODEL_COLUMNS)
    )


def _pipeline(alpha: float) -> Pipeline:
    return make_pipeline(
        # Keeps a column with no value at all, so a feature nobody collected
        # yet does not change the shape between folds.
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


@dataclass
class Model:
    pipeline: Pipeline

    def predict(
        self, items: Sequence[Mapping[str, Optional[float]]]
    ) -> List[float]:
        """Minutes until empty for each feature dict, never below zero."""
        if not items:
            return []
        return [max(0.0, float(y)) for y in self.pipeline.predict(matrix(items))]


def fit(
    items: Sequence[Example],
    alpha: float = DEFAULT_ALPHA,
    rounds: int = DEFAULT_ROUNDS,
) -> Model:
    """Fit on exact examples, then bring the censored ones in. See the module doc.

    Raises ValueError when no example is exact: lower bounds alone say nothing
    about when a battery runs out.
    """
    exact = [item for item in items if not item.censored]
    if not exact:
        raise ValueError("no exact examples to fit on")
    x_exact = matrix([item.features for item in exact])
    y_exact = np.array([item.minutes for item in exact])
    pipeline = _pipeline(alpha)
    pipeline.fit(x_exact, y_exact)

    censored = [item for item in items if item.censored]
    if not censored or rounds <= 0:
        return Model(pipeline)

    x_censored = matrix([item.features for item in censored])
    bounds = np.array([item.minutes for item in censored])
    x_all = np.vstack((x_exact, x_censored))
    previous = None
    for _ in range(rounds):
        targets = np.maximum(pipeline.predict(x_censored), bounds)
        if previous is not None and np.allclose(targets, previous, atol=0.01):
            break
        previous = targets
        pipeline = _pipeline(alpha)
        pipeline.fit(x_all, np.concatenate((y_exact, targets)))
    return Model(pipeline)


def alpha_score(
    items: Sequence[Example],
    alpha: float,
    rounds: int = DEFAULT_ROUNDS,
) -> Optional[float]:
    """Mean over sessions of the held out absolute error on exact rows, in minutes.

    Each session counts once, however many rows it has. None when no session
    with an exact row could be predicted, which needs at least two sessions
    with a full discharge.
    """
    predicted = _held_out(items, lambda kept: fit(kept, alpha, rounds))
    errors = {}
    for item, value in zip(items, predicted):
        if item.censored or value is None:
            continue
        errors.setdefault(item.session, []).append(abs(value - item.minutes))
    if not errors:
        return None
    return sum(sum(e) / len(e) for e in errors.values()) / len(errors)


def choose_alpha(
    items: Sequence[Example],
    alphas: Sequence[float] = ALPHA_GRID,
    rounds: int = DEFAULT_ROUNDS,
) -> float:
    """The alpha in `alphas` with the lowest `alpha_score` on `items`.

    A tie goes to the larger alpha, the plainer model. With too few full
    discharges to score any of them, this falls back to `DEFAULT_ALPHA`.
    """
    if not alphas:
        raise ValueError("no alpha to choose from")
    best, best_score = DEFAULT_ALPHA, None
    for alpha in sorted(alphas, reverse=True):
        score = alpha_score(items, alpha, rounds)
        if score is None:
            continue
        if best_score is None or score < best_score - 1e-9:
            best, best_score = alpha, score
    return best


def _held_out(items: Sequence[Example], train_model) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(items)
    for _, train, test in session_folds(items):
        kept = [items[p] for p in train]
        if all(item.censored for item in kept):
            continue
        model = train_model(kept)
        predicted = model.predict([items[p].features for p in test])
        for position, value in zip(test, predicted):
            out[position] = value
    return out


def cross_predict(
    items: Sequence[Example],
    alpha: float = DEFAULT_ALPHA,
    rounds: int = DEFAULT_ROUNDS,
    alphas: Optional[Sequence[float]] = None,
) -> List[Optional[float]]:
    """A prediction for each example from a model that never saw its session.

    An entry stays None when the other sessions hold no exact example, which
    happens to the only full discharge collected so far. Scoring must skip
    those rows, not fill them.

    With `alphas`, `alpha` is ignored and each held out session gets the alpha
    `choose_alpha` picks on the other sessions only.
    """
    if alphas is None:
        return _held_out(items, lambda kept: fit(kept, alpha, rounds))
    return _held_out(
        items, lambda kept: fit(kept, choose_alpha(kept, alphas, rounds), rounds)
    )
