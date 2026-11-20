"""Chart the error of both estimators as a function of charge level.

This draws the same numbers as the band table of `python -m battery.evaluate
report`, so the chart can never show something the table does not. Every band
of the charge range gets a slot on the axis. A band with no scored row stays
blank, because closing it up would hide where the data runs out.

Each band is labelled with its session count. Rows of one session are near
copies of each other, so a tall or short bar resting on one session says little,
and the reader has to be able to see that.

The chart goes under `data/` with everything else derived from the samples, so
it stays out of the repository. Working out the bands needs only the standard
library. Drawing them needs the `analysis` extra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .dataset import examples
from .evaluate import DEFAULT_BAND_PCT, Scored, by_charge, compare, scored_rows

DEFAULT_CHART = Path("data/charts/error_by_charge.png")

MODEL_COLOR = "#2a78d6"
MACOS_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e4e3df"


def chart_bands(
    scored: Sequence[Scored], band_pct: int = DEFAULT_BAND_PCT
) -> List[dict]:
    """Every band of the charge range, lowest first, empty ones included."""
    filled = {band["low_pct"]: band for band in by_charge(scored, band_pct)}
    return [
        filled.get(
            low,
            {
                "low_pct": low,
                "high_pct": min(low + band_pct, 100),
                "rows": 0,
                "sessions": 0,
                "model_mae": None,
                "macos_mae": None,
            },
        )
        for low in range(0, 100, band_pct)
    ]


def band_label(band: dict) -> str:
    """The charge range of a band, and how many sessions it rests on."""
    sessions = band["sessions"]
    unit = "session" if sessions == 1 else "sessions"
    return f"{band['low_pct']}-{band['high_pct']}%\n{sessions} {unit}"


def subtitle(totals: dict) -> str:
    return (
        f"{totals['rows']} scored rows from {totals['sessions']} discharge "
        "sessions, predictions held out by session"
    )


def build_figure(bands: Sequence[dict], totals: dict):
    """The chart as a matplotlib figure. A filled band gets one bar per estimator."""
    # A bare Figure rather than pyplot, which would need a display backend.
    from matplotlib.figure import Figure

    width = 0.38
    filled = [i for i, band in enumerate(bands) if band["rows"]]
    figure = Figure(figsize=(max(10.0, 1.0 * len(bands)), 5.2), layout="constrained")
    axes = figure.add_subplot()
    axes.bar(
        [i - width / 2 - 0.01 for i in filled],
        [bands[i]["model_mae"] for i in filled],
        width,
        color=MODEL_COLOR,
        label="Model",
        zorder=2,
    )
    axes.bar(
        [i + width / 2 + 0.01 for i in filled],
        [bands[i]["macos_mae"] for i in filled],
        width,
        color=MACOS_COLOR,
        label="macOS",
        zorder=2,
    )

    axes.set_xticks(range(len(bands)), [band_label(band) for band in bands])
    axes.set_xlim(-0.6, len(bands) - 0.4)
    axes.set_xlabel("Charge, as a share of the current maximum capacity", color=MUTED_INK)
    axes.set_ylabel("Mean absolute error (minutes)", color=MUTED_INK)
    axes.tick_params(colors=MUTED_INK, length=0)
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(MUTED_INK)
    axes.legend(frameon=False, loc="upper left", ncols=2, labelcolor=INK)

    # Headroom, so the legend never sits on a tall bar.
    axes.set_ylim(top=axes.get_ylim()[1] * 1.15)

    axes.set_title(subtitle(totals), loc="left", color=MUTED_INK, fontsize=10)
    axes.annotate(
        "Error of both estimators by charge level",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 24),
        textcoords="offset points",
        color=INK,
        fontsize=14,
    )
    return figure


def write_chart(
    scored: Sequence[Scored],
    path: Path = DEFAULT_CHART,
    band_pct: int = DEFAULT_BAND_PCT,
) -> Optional[Path]:
    """Draw the chart to `path`. With nothing scored, write nothing and return None."""
    if not scored:
        return None
    figure = build_figure(chart_bands(scored, band_pct), compare(scored))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here so the bands work without the analysis extra.
    from .series import DEFAULT_OUT, read_csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--band", type=int, default=DEFAULT_BAND_PCT)
    args = parser.parse_args(argv)

    if not args.table.exists():
        print(f"no table at {args.table}, run python -m battery.series build first")
        return 1

    from .model import cross_predict

    rows = read_csv(args.table)
    items = examples(rows)
    written = write_chart(
        scored_rows(rows, items, cross_predict(items)), args.out, args.band
    )
    if written is None:
        print("no scored rows yet, so no chart was written")
        return 1
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
