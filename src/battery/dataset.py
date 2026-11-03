"""Training examples for the model, and how to hold sessions out.

`battery.features` gives the inputs of a row and `battery.target` gives what
is known about its outcome. This module pairs the two into one example per
discharging row, so the model step never has to know where either came from.

An example is one of two kinds:

- exact: the session ran out, `minutes` is the time the battery really lasted.
- censored: the session ended first, `minutes` is only a lower bound.

Rows on a charger have no outcome and produce no example.

Rows of one session sit minutes apart and are nearly copies of each other. A
random split would put neighbours on both sides and make any model look far
better than it is, so `session_folds` always holds out a whole session. Each
fold trains on sessions from before and after the held out one. That is
acceptable because full discharges are too rare to keep only the past, but it
means a fold can see a battery slightly older than the one it is tested on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .features import DEFAULT_MIN_SPAN_S, DEFAULT_WINDOW_S, features
from .target import CENSORED, EXACT


@dataclass(frozen=True)
class Example:
    """One discharging row, ready for fitting or scoring."""

    index: int  # position of the row in the time series
    session: int
    features: Dict[str, Optional[float]]
    minutes: float
    censored: bool


def examples(
    rows: Sequence[dict],
    window_s: float = DEFAULT_WINDOW_S,
    min_span_s: float = DEFAULT_MIN_SPAN_S,
) -> List[Example]:
    """One example per row that has features and an outcome, in row order.

    `rows` must be sorted by instant and carry the target columns, as
    `battery.series` builds them.
    """
    out = []
    for index, (row, item) in enumerate(
        zip(rows, features(rows, window_s, min_span_s))
    ):
        if item is None:
            continue
        status = row["target_status"]
        if status == EXACT:
            minutes = row["actual_remaining_min"]
        elif status == CENSORED:
            minutes = row["remaining_at_least_min"]
        else:
            continue
        out.append(
            Example(index, row["session_id"], item, minutes, status == CENSORED)
        )
    return out


def session_folds(
    items: Sequence[Example],
) -> Iterator[Tuple[int, List[int], List[int]]]:
    """For each session in turn: its id, the positions kept, the positions held out.

    Positions index `items`. A session is never on both sides of a fold.
    """
    for session in sorted({item.session for item in items}):
        train, test = [], []
        for position, item in enumerate(items):
            (test if item.session == session else train).append(position)
        yield session, train, test


def example_counts(items: Sequence[Example]) -> Dict[str, int]:
    """How much there is to learn from: rows and sessions of each kind."""
    exact = [item for item in items if not item.censored]
    return {
        "examples": len(items),
        "exact": len(exact),
        "censored": len(items) - len(exact),
        "sessions": len({item.session for item in items}),
        "sessions_exact": len({item.session for item in exact}),
    }
