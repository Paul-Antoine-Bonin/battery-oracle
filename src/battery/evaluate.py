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

Scoring needs only the standard library. The `report` command needs the
`analysis` extra, because it has to fit the model first.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .dataset import Example, example_counts, examples

DEFAULT_BAND_PCT = 10


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


def by_charge(
    scored: Sequence[Scored], band_pct: int = DEFAULT_BAND_PCT
) -> List[dict]:
    """Both estimators per charge band, lowest first, skipping empty bands.

    Bands include their lower edge. A full battery goes in the top band, so
    there is no band holding only 100%. A row with no charge reading is left
    out. This is what the error chart plots.
    """
    if not 1 <= band_pct <= 100:
        raise ValueError(f"band_pct must be between 1 and 100, got {band_pct}")
    top = math.ceil(100 / band_pct) - 1
    groups: Dict[int, List[Scored]] = {}
    for item in scored:
        if item.charge_fraction is None:
            continue
        # Rounded first, since 0.29 * 100 lands just under 29 in binary.
        percent = round(item.charge_fraction * 100, 6)
        groups.setdefault(min(int(percent // band_pct), top), []).append(item)
    return [
        {
            "low_pct": band * band_pct,
            "high_pct": min((band + 1) * band_pct, 100),
            "rows": len(group),
            "sessions": len({item.session for item in group}),
            "model_mae": _mae([item.model_error for item in group]),
            "macos_mae": _mae([item.macos_error for item in group]),
        }
        for band, group in sorted(groups.items())
    ]


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


def _minutes(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def report_lines(
    rows: Sequence[dict],
    items: Sequence[Example],
    predicted: Sequence[Optional[float]],
    band_pct: int = DEFAULT_BAND_PCT,
) -> List[str]:
    """The text of the `report` command, from rows already predicted."""
    counts = example_counts(items)
    scored = scored_rows(rows, items, predicted)
    totals = compare(scored)
    misses = bound_misses(rows, items, predicted)
    lines = [
        f"{'examples':<24}{counts['examples']}",
        f"{'sessions_exact':<24}{counts['sessions_exact']}",
        f"{'scored_rows':<24}{totals['rows']}",
        f"{'scored_sessions':<24}{totals['sessions']}",
        f"{'model_mae_min':<24}{_minutes(totals['model_mae'])}",
        f"{'macos_mae_min':<24}{_minutes(totals['macos_mae'])}",
        f"{'model_session_mae_min':<24}{_minutes(totals['model_session_mae'])}",
        f"{'macos_session_mae_min':<24}{_minutes(totals['macos_session_mae'])}",
        f"{'sessions_model_better':<24}{totals['sessions_model_better']}",
        f"{'sessions_macos_better':<24}{totals['sessions_macos_better']}",
        f"{'censored_rows':<24}{misses['rows']}",
        f"{'model_below_bound':<24}{misses['model_below']}",
        f"{'macos_below_bound':<24}{misses['macos_below']}",
    ]
    bands = by_charge(scored, band_pct)
    if bands:
        lines.append("")
        lines.append(
            f"{'charge':<12}{'rows':<8}{'sessions':<10}{'model':<10}{'macos':<10}"
        )
    for band in bands:
        name = f"{band['low_pct']}-{band['high_pct']}%"
        lines.append(
            f"{name:<12}{band['rows']:<8}{band['sessions']:<10}"
            f"{_minutes(band['model_mae']):<10}{_minutes(band['macos_mae']):<10}".rstrip()
        )
    return lines


def _band_pct(text: str) -> int:
    """A `--band` value, checked here so a typo fails before the model is fitted."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError(f"must be between 1 and 100, got {text}")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here so scoring works without the analysis extra.
    from .series import DEFAULT_OUT, read_csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="mean absolute error of both estimators")
    report.add_argument("--band", type=_band_pct, default=DEFAULT_BAND_PCT)
    args = parser.parse_args(argv)

    if not args.table.exists():
        print(f"no table at {args.table}, run python -m battery.series build first")
        return 1

    from .model import cross_predict

    rows = read_csv(args.table)
    items = examples(rows)
    for line in report_lines(rows, items, cross_predict(items), args.band):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
