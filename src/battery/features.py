"""Model inputs for each discharging row: the current drain rate and the context.

    python -m battery.features coverage

The regression model predicts the time left until empty from what is known
at the moment of a sample. This module turns a row of the time series into
those inputs. It is the first half of the modelling step; fitting comes on
top of it.

The one rule that matters here is that a feature only ever looks backwards.
Every value is computed from the row itself and the rows before it in the
same discharge session, never from a later row and never from the target
columns. A feature that peeked ahead would make the model look better than
anything that could run live, and the comparison with macOS would be unfair
in the model's favour.

Only rows inside a discharge session get features. A row on a charger has no
time to empty to predict, so `row_features` returns None for it.

| Feature | Meaning |
|---|---|
| `charge_fraction` | charge against the maximum read in the same sample |
| `current_mah` | charge left in mAh |
| `health_fraction` | maximum against design capacity |
| `draw_ma` | current drawn right now, positive while discharging |
| `power_w` | power drawn right now, positive while discharging |
| `rate_mah_per_h` | charge lost per hour over the trailing window |
| `rate_fraction_per_h` | the same rate as a fraction of the maximum |
| `instant_remaining_min` | `current_mah` over `draw_ma`, in minutes |
| `trend_remaining_min` | `current_mah` over `rate_mah_per_h`, in minutes |
| `session_elapsed_min` | minutes since the session started |
| `load_per_cpu` | 1 minute load average per logical core |
| `wifi_on` | 1.0 when Wi-Fi was powered, 0.0 when not |
| `brightness_fraction` | built-in display brightness |
| `temperature_c` | battery temperature |
| `hour_of_day` | local hour of the sample, from its stored offset |

Any feature can be None: a command failed, the sample predates a context
field, or the session is too young for a trailing rate. The model step
decides how to fill those, not this module.

The rate is measured from the charge readings rather than taken from the
current, because the instant current swings with every burst of work while
the charge only moves as fast as the battery really drains. Both are kept,
since which one predicts better is a question for the data. The window
defaults to 30 minutes and a rate needs at least 10 minutes of history, so
the first two samples of a session have none at the default interval.

The foreground application is left out. It is a bundle id from a personal
machine, and turning it into a number needs a choice of categories that is
better made once there is data to make it with.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_WINDOW_S = 1800.0
DEFAULT_MIN_SPAN_S = 600.0

FEATURE_COLUMNS = (
    "charge_fraction",
    "current_mah",
    "health_fraction",
    "draw_ma",
    "power_w",
    "rate_mah_per_h",
    "rate_fraction_per_h",
    "instant_remaining_min",
    "trend_remaining_min",
    "session_elapsed_min",
    "load_per_cpu",
    "wifi_on",
    "brightness_fraction",
    "temperature_c",
    "hour_of_day",
)


def _instant(row: dict) -> datetime:
    return datetime.fromisoformat(row["sampled_at"])


def _positive_draw(value) -> Optional[float]:
    """A signed discharge reading turned positive, None when not discharging."""
    if value is None or value >= 0:
        return None
    return float(-value)


def _rate_per_h(earlier, later, hours: float) -> Optional[float]:
    if earlier is None or later is None:
        return None
    return (earlier - later) / hours


def _remaining_min(charge, per_hour) -> Optional[float]:
    """Minutes until `charge` reaches zero at `per_hour`, None if it never does."""
    if charge is None or per_hour is None or per_hour <= 0:
        return None
    return charge / per_hour * 60.0


def row_features(
    rows: Sequence[dict],
    i: int,
    window_s: float = DEFAULT_WINDOW_S,
    min_span_s: float = DEFAULT_MIN_SPAN_S,
) -> Optional[Dict[str, Optional[float]]]:
    """The features of `rows[i]`, from that row and the ones before it.

    `rows` must be sorted by instant and carry `session_id`, as
    `battery.series` builds them. Returns None for a row outside a session.
    """
    row = rows[i]
    session = row.get("session_id")
    if session is None:
        return None
    now = _instant(row)

    # Walk back through the same session. Sessions are contiguous in a
    # sorted series, so the first row with another id ends the walk.
    start = i
    while start > 0 and rows[start - 1].get("session_id") == session:
        start -= 1
    oldest = i
    while oldest > start:
        if (now - _instant(rows[oldest - 1])).total_seconds() > window_s:
            break
        oldest -= 1

    rate_mah = rate_fraction = None
    span_s = (now - _instant(rows[oldest])).total_seconds()
    if span_s >= min_span_s:
        hours = span_s / 3600.0
        past = rows[oldest]
        rate_mah = _rate_per_h(past.get("current_mah"), row.get("current_mah"), hours)
        rate_fraction = _rate_per_h(
            past.get("charge_fraction"), row.get("charge_fraction"), hours
        )

    current_mah = row.get("current_mah")
    draw_ma = _positive_draw(row.get("amperage_ma"))
    wifi = row.get("wifi_on")
    return {
        "charge_fraction": row.get("charge_fraction"),
        "current_mah": current_mah,
        "health_fraction": row.get("health_fraction"),
        "draw_ma": draw_ma,
        "power_w": _positive_draw(row.get("power_w")),
        "rate_mah_per_h": rate_mah,
        "rate_fraction_per_h": rate_fraction,
        "instant_remaining_min": _remaining_min(current_mah, draw_ma),
        "trend_remaining_min": _remaining_min(current_mah, rate_mah),
        "session_elapsed_min": (now - _instant(rows[start])).total_seconds() / 60.0,
        "load_per_cpu": row.get("load_per_cpu"),
        "wifi_on": None if wifi is None else float(wifi),
        "brightness_fraction": row.get("brightness_fraction"),
        "temperature_c": row.get("temperature_c"),
        "hour_of_day": float(now.hour),
    }


def features(
    rows: Sequence[dict],
    window_s: float = DEFAULT_WINDOW_S,
    min_span_s: float = DEFAULT_MIN_SPAN_S,
) -> List[Optional[Dict[str, Optional[float]]]]:
    """`row_features` for every row, in the same order as `rows`."""
    return [row_features(rows, i, window_s, min_span_s) for i in range(len(rows))]


def coverage(items: Sequence[Optional[dict]]) -> Dict[str, int]:
    """For each feature, how many rows with features have a value for it.

    A feature that is almost always None is not worth fitting on, and this
    is the quickest way to see that before any model is trained.
    """
    present = [item for item in items if item is not None]
    counts = {"rows": len(present)}
    for column in FEATURE_COLUMNS:
        counts[column] = sum(1 for item in present if item[column] is not None)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here so the feature functions stay usable on plain rows
    # without pulling in the series module.
    from .series import DEFAULT_OUT, read_csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("coverage", help="how often each feature has a value")
    args = parser.parse_args(argv)

    if not args.table.exists():
        print(f"no table at {args.table}, run python -m battery.series build first")
        return 1
    counts = coverage(features(read_csv(args.table)))
    total = counts.pop("rows")
    print(f"{'rows':<24}{total}")
    for column, count in counts.items():
        share = f"{count / total:.0%}" if total else "-"
        print(f"{column:<24}{count:<8}{share}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
