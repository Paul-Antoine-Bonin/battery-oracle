"""Score the model against the macOS estimate on the same rows.

The comparison is only fair if both estimators answer the same questions, so a
row is scored only when all three of these exist: the time the battery really
lasted, a prediction from a model that never saw the row's session, and a
macOS estimate. A row missing any of them is dropped for both sides.

Only exact rows give an error. A censored row says the battery lasted at least
so long, so `bound_misses` counts the estimates below that bound, which are
wrong for certain, and says nothing about the ones above it.

Rows of one session are near copies of each other, so the row count overstates
how much evidence there is. The number of sessions is what matters. A long
session also weighs more than a short one in a mean over rows, so `compare`
gives a mean over sessions as well, and how many sessions each estimator won.

Scoring needs only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .dataset import Example


@dataclass(frozen=True)
class Scored:
    """One row both estimators can be scored on. Times are in minutes."""

    index: int  # position of the row in the time series
    session: int
    charge_fraction: Optional[float]
    actual_min: float
    model_min: float
    macos_min: float

    @property
    def model_error(self) -> float:
        return abs(self.model_min - self.actual_min)

    @property
    def macos_error(self) -> float:
        return abs(self.macos_min - self.actual_min)


def _check_lengths(items: Sequence[Example], predicted: Sequence) -> None:
    if len(items) != len(predicted):
        raise ValueError(
            f"{len(items)} examples but {len(predicted)} predictions"
        )


def scored_rows(
    rows: Sequence[dict],
    items: Sequence[Example],
    predicted: Sequence[Optional[float]],
) -> List[Scored]:
    """The exact rows that have a prediction and a macOS estimate.

    `predicted` must come from `battery.model.cross_predict` on `items`, and
    `items` from `battery.dataset.examples` on `rows`.
    """
    _check_lengths(items, predicted)
    out = []
    for item, model_min in zip(items, predicted):
        row = rows[item.index]
        macos_min = row.get("macos_remaining_min")
        if item.censored or model_min is None or macos_min is None:
            continue
        out.append(
            Scored(
                item.index,
                item.session,
                row.get("charge_fraction"),
                item.minutes,
                model_min,
                float(macos_min),
            )
        )
    return out


def _mae(errors: Sequence[float]) -> Optional[float]:
    return sum(errors) / len(errors) if errors else None


def per_session(scored: Sequence[Scored]) -> List[dict]:
    """Rows and mean absolute errors of each session, in session order."""
    groups: Dict[int, List[Scored]] = {}
    for item in scored:
        groups.setdefault(item.session, []).append(item)
    return [
        {
            "session": session,
            "rows": len(group),
            "model_mae": _mae([item.model_error for item in group]),
            "macos_mae": _mae([item.macos_error for item in group]),
        }
        for session, group in sorted(groups.items())
    ]


def compare(scored: Sequence[Scored]) -> dict:
    """Both estimators over all scored rows. The means are None with no rows."""
    sessions = per_session(scored)
    return {
        "rows": len(scored),
        "sessions": len(sessions),
        "model_mae": _mae([item.model_error for item in scored]),
        "macos_mae": _mae([item.macos_error for item in scored]),
        "model_session_mae": _mae([s["model_mae"] for s in sessions]),
        "macos_session_mae": _mae([s["macos_mae"] for s in sessions]),
        "sessions_model_better": sum(
            s["model_mae"] < s["macos_mae"] for s in sessions
        ),
        "sessions_macos_better": sum(
            s["macos_mae"] < s["model_mae"] for s in sessions
        ),
    }


def bound_misses(
    rows: Sequence[dict],
    items: Sequence[Example],
    predicted: Sequence[Optional[float]],
) -> dict:
    """Censored rows where an estimate fell below what the battery really lasted.

    Counted on the rows that have both estimates, for the same reason as
    `scored_rows`.
    """
    _check_lengths(items, predicted)
    out = {"rows": 0, "model_below": 0, "macos_below": 0}
    for item, model_min in zip(items, predicted):
        macos_min = rows[item.index].get("macos_remaining_min")
        if not item.censored or model_min is None or macos_min is None:
            continue
        out["rows"] += 1
        out["model_below"] += model_min < item.minutes
        out["macos_below"] += macos_min < item.minutes
    return out
