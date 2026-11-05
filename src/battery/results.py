"""Write the honest summary of the comparison, ready to go in the README.

The last step of the project is to say by how much the model beats macOS and
in which situations. This module turns the scores into that text, so the
sentences are worked out from the numbers and nobody picks the flattering ones.

The summary always says which estimator came out ahead, including when that is
macOS, and it lists every place macOS won. A split resting on fewer than
`MIN_SESSIONS` sessions gets no verdict, because rows of one session are near
copies of each other and one good day could carry it.

The file goes under `data/` with everything else derived from the samples. It
is a draft for a person to read and copy over, and nothing here edits the
README. Building the text needs only the standard library. The command needs
the `analysis` extra, because it has to fit the model first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .dataset import examples
from .evaluate import (
    DEFAULT_BAND_PCT,
    Scored,
    by_charge,
    by_situation,
    compare,
    scored_rows,
)

DEFAULT_RESULTS = Path("data/results.md")

MIN_SESSIONS = 3

MODEL = "model"
MACOS = "macOS"
TIE = "tie"


def winner(model_mae: float, macos_mae: float) -> str:
    """Which estimator has the lower error, to a tenth of a minute as printed."""
    model, macos = round(model_mae, 1), round(macos_mae, 1)
    if model == macos:
        return TIE
    return MODEL if model < macos else MACOS


def headline(totals: dict) -> str:
    """One sentence on the whole comparison, from the means over sessions.

    Those means give each session one vote, so one long discharge cannot
    decide the result.
    """
    model, macos = totals["model_session_mae"], totals["macos_session_mae"]
    errors = f"{model:.1f} minutes for the model and {macos:.1f} for macOS"
    who = winner(model, macos)
    if who == TIE:
        return f"The two estimators tied, at {errors} per session."
    if who == MODEL:
        return (
            f"The model beat macOS, with a mean error of {errors} per session, "
            f"{(macos - model) / macos:.0%} lower."
        )
    return (
        f"macOS beat the model, with a mean error of {errors} per session, "
        f"{(model - macos) / model:.0%} lower."
    )


def _count(number: int, unit: str) -> str:
    return f"{number} {unit}" if number == 1 else f"{number} {unit}s"


def _band_name(band: dict) -> str:
    return f"{band['low_pct']}-{band['high_pct']}% charge"


def _split_line(name: str, split: dict) -> str:
    return (
        f"- {name}: {split['model_mae']:.1f} against {split['macos_mae']:.1f} "
        f"minutes for macOS, {_count(split['sessions'], 'session')}"
    )


def _section(title: str, lines: Sequence[str]) -> List[str]:
    return [f"### {title}", "", *(lines or ["None."]), ""]


def results_lines(
    rows: Sequence[dict],
    scored: Sequence[Scored],
    band_pct: int = DEFAULT_BAND_PCT,
    min_sessions: int = MIN_SESSIONS,
) -> List[str]:
    """The summary as Markdown lines. `scored` must not be empty."""
    if not scored:
        raise ValueError("nothing scored, so there is nothing to summarise")
    totals = compare(scored)
    named = [(_band_name(band), band) for band in by_charge(scored, band_pct)]
    named += [(group["name"], group) for group in by_situation(rows, scored)]

    model_wins, macos_wins, ties, thin = [], [], [], []
    for name, split in named:
        if split["sessions"] < min_sessions:
            thin.append(f"- {name}: {split['sessions']} of {min_sessions} sessions needed")
            continue
        who = winner(split["model_mae"], split["macos_mae"])
        target = {MODEL: model_wins, MACOS: macos_wins, TIE: ties}[who]
        target.append(_split_line(name, split))

    lines = ["## Results", ""]
    if totals["sessions"] < min_sessions:
        lines.append(
            f"Too early to call: {_count(totals['sessions'], 'session')} scored, "
            f"and a verdict needs {min_sessions}. Provisional figures follow."
        )
        lines.append("")
    lines += [
        headline(totals),
        "",
        f"Scored on {_count(totals['rows'], 'row')} from "
        f"{_count(totals['sessions'], 'full discharge')}, with predictions held out "
        f"by session. The model did better in "
        f"{_count(totals['sessions_model_better'], 'session')} and macOS in "
        f"{totals['sessions_macos_better']}.",
        "",
        "The errors below are mean absolute errors over rows, model first.",
        "",
    ]
    lines += _section("Where the model wins", model_wins)
    lines += _section("Where macOS wins", macos_wins)
    if ties:
        lines += _section("Where they tie", ties)
    lines += _section("Too little data to say", thin)
    return lines[:-1]


def write_results(
    rows: Sequence[dict],
    scored: Sequence[Scored],
    path: Path = DEFAULT_RESULTS,
    band_pct: int = DEFAULT_BAND_PCT,
) -> Optional[Path]:
    """Write the summary to `path`. With nothing scored, write nothing and return None."""
    if not scored:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(results_lines(rows, scored, band_pct)) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here so the text can be built without the analysis extra.
    from .series import DEFAULT_OUT, read_csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--band", type=int, default=DEFAULT_BAND_PCT)
    args = parser.parse_args(argv)

    if not args.table.exists():
        print(f"no table at {args.table}, run python -m battery.series build first")
        return 1

    from .model import cross_predict

    rows = read_csv(args.table)
    items = examples(rows)
    written = write_results(
        rows, scored_rows(rows, items, cross_predict(items)), args.out, args.band
    )
    if written is None:
        print("no scored rows yet, so no results were written")
        return 1
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
