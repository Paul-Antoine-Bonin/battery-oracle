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

The report also checks the model against the plainest alternative, the
charge left divided by the rate it fell over the last half hour. If the fitted
model does not beat that, it is not worth having, whatever it does against
macOS. `against_trend` scores both on the same rows, the ones where that
extrapolation exists.

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

# Limits of the situations in `by_situation`. They are round numbers picked
# before any real score exists, so they cannot have been tuned to favour either
# estimator.
LIGHT_LOAD_PER_CPU = 0.25
HEAVY_LOAD_PER_CPU = 0.75
BRIGHT_FRACTION = 0.5


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


def against_trend(
    items: Sequence[Example], scored: Sequence[Scored]
) -> dict:
    """The model against the trend extrapolation, on the rows that have one.

    The extrapolation is the `trend_remaining_min` feature, used as it is with
    no cap, since that is what anyone would get by dividing the charge left by
    the recent rate. A row whose session has under the minimum history has no
    rate, so it has no extrapolation and is left out on both sides.
    """
    by_index = {item.index: item for item in items}
    model_errors, trend_errors, sessions = [], [], set()
    for row in scored:
        item = by_index.get(row.index)
        trend = None if item is None else item.features.get("trend_remaining_min")
        if trend is None:
            continue
        model_errors.append(row.model_error)
        trend_errors.append(abs(trend - row.actual_min))
        sessions.add(row.session)
    return {
        "rows": len(model_errors),
        "sessions": len(sessions),
        "model_mae": _mae(model_errors),
        "trend_mae": _mae(trend_errors),
    }


def _group_errors(group: Sequence[Scored]) -> dict:
    """How much a group of rows rests on, and the row mean of each estimator."""
    return {
        "rows": len(group),
        "sessions": len({item.session for item in group}),
        "model_mae": _mae([item.model_error for item in group]),
        "macos_mae": _mae([item.macos_error for item in group]),
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
            **_group_errors(group),
        }
        for band, group in sorted(groups.items())
    ]


def _load_group(row: dict) -> Optional[str]:
    load = row.get("load_per_cpu")
    if load is None:
        return None
    if load < LIGHT_LOAD_PER_CPU:
        return "light load"
    return "moderate load" if load < HEAVY_LOAD_PER_CPU else "heavy load"


def _brightness_group(row: dict) -> Optional[str]:
    brightness = row.get("brightness_fraction")
    if brightness is None:
        return None
    return "dim display" if brightness < BRIGHT_FRACTION else "bright display"


def _wifi_group(row: dict) -> Optional[str]:
    wifi = row.get("wifi_on")
    if wifi is None:
        return None
    return "wifi on" if wifi else "wifi off"


SITUATIONS = (
    (_load_group, ("light load", "moderate load", "heavy load")),
    (_brightness_group, ("dim display", "bright display")),
    (_wifi_group, ("wifi on", "wifi off")),
)


def by_situation(rows: Sequence[dict], scored: Sequence[Scored]) -> List[dict]:
    """Both estimators per usage situation, skipping situations with no row.

    A situation is read from the row being scored, so a session can land in
    several of them. A row missing a reading is left out of that split only.
    The splits overlap, so their row counts do not add up to the total.
    """
    out = []
    for group_of, names in SITUATIONS:
        groups: Dict[str, List[Scored]] = {}
        for item in scored:
            name = group_of(rows[item.index])
            if name is not None:
                groups.setdefault(name, []).append(item)
        out.extend(
            {"name": name, **_group_errors(groups[name])}
            for name in names
            if name in groups
        )
    return out


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
    trend = against_trend(items, scored)
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
        "",
        f"{'trend_rows':<24}{trend['rows']}",
        f"{'trend_sessions':<24}{trend['sessions']}",
        f"{'model_mae_on_trend_min':<24}{_minutes(trend['model_mae'])}",
        f"{'trend_mae_min':<24}{_minutes(trend['trend_mae'])}",
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
    situations = by_situation(rows, scored)
    if situations:
        lines.append("")
        lines.append(
            f"{'situation':<18}{'rows':<8}{'sessions':<10}{'model':<10}{'macos'}"
        )
    for group in situations:
        lines.append(
            f"{group['name']:<18}{group['rows']:<8}{group['sessions']:<10}"
            f"{_minutes(group['model_mae']):<10}{_minutes(group['macos_mae']):<10}".rstrip()
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here so scoring works without the analysis extra.
    from .series import DEFAULT_OUT, read_csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="mean absolute error of both estimators")
    report.add_argument("--band", type=int, default=DEFAULT_BAND_PCT)
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
