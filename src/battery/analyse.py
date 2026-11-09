"""Run the whole analysis from the raw samples in one go.

    python -m battery.analyse
    python -m battery.analyse --band 20
    python -m battery.analyse --tune

This does in one command what `battery.series build`, `battery.evaluate
report`, `battery.chart` and `battery.results` do one after the other:

1. Rebuild the time series table from `data/raw/`.
2. Fit the model once, holding out one session at a time.
3. Print the report, then write the chart and the results summary.

The separate commands each refit the model on their own, so running all three
fits it three times. Here the held out predictions are computed once and every
output is drawn from the same ones, so the report, the chart and the summary
cannot disagree.

The table is read back from the file it was just written to, exactly as the
separate commands read it, so both paths score the same rows.

By default the ridge strength is fixed. `--tune` picks it from
`battery.model.ALPHA_GRID` for every held out session, on the other sessions
only. That refits the model many times over, so it is slower, and with fewer
than three full discharges it changes nothing.

With no scored row yet, the table and the report are still produced, the chart
and the summary are not, and the command exits with 1. Everything it writes
goes under `data/`, git-ignored like the samples themselves.

Needs the `analysis` extra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from .chart import DEFAULT_CHART, write_chart
from .dataset import examples
from .evaluate import DEFAULT_BAND_PCT, against_trend, report_lines, scored_rows
from .results import DEFAULT_RESULTS, write_results
from .series import DEFAULT_OUT, load_series, read_csv, summarize, write_csv
from .store import DEFAULT_ROOT


def analyse(
    root: Path = DEFAULT_ROOT,
    table: Path = DEFAULT_OUT,
    chart: Path = DEFAULT_CHART,
    results: Path = DEFAULT_RESULTS,
    band_pct: int = DEFAULT_BAND_PCT,
    echo: Callable[[str], None] = print,
    tune: bool = False,
) -> bool:
    """Run every step and report through `echo`. True when something was scored."""
    # Imported here so the rest of the module loads without the analysis extra.
    from .model import ALPHA_GRID, cross_predict

    series = load_series(root=root)
    written = write_csv(series.rows, table)
    counts = summarize(series)
    echo(f"{len(series)} rows -> {written}")
    if series.duplicates:
        echo(f"dropped {series.duplicates} repeated instant(s)")
    if series.skipped_lines:
        echo(f"skipped {series.skipped_lines} unreadable line(s)")
    echo(f"full discharges so far: {counts['sessions_empty']}")
    echo("")

    rows = read_csv(written)
    items = examples(rows)
    if tune:
        grid = ", ".join(f"{alpha:g}" for alpha in ALPHA_GRID)
        echo(f"ridge strength chosen per held out session from {grid}")
        predicted = cross_predict(items, alphas=ALPHA_GRID)
    else:
        predicted = cross_predict(items)
    for line in report_lines(rows, items, predicted, band_pct):
        echo(line)
    echo("")

    scored = scored_rows(rows, items, predicted)
    if not scored:
        echo("no scored rows yet, so no chart and no results were written")
        return False
    echo(f"wrote {write_chart(scored, chart, band_pct)}")
    trend = against_trend(items, scored)
    echo(f"wrote {write_results(rows, scored, results, band_pct, trend)}")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--table", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--band", type=int, default=DEFAULT_BAND_PCT)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="choose the ridge strength per held out session",
    )
    args = parser.parse_args(argv)

    ok = analyse(
        args.root, args.table, args.chart, args.results, args.band, tune=args.tune
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
