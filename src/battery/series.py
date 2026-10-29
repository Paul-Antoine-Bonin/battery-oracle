"""The collected samples as one time series, one row per sample.

    python -m battery.series build
    python -m battery.series summary

The raw JSON Lines under `data/raw/` stay the source of truth. This module turns
them into a flat table sorted by time, with one column per parsed field, and
writes it to `data/series/samples.csv`. The table is derived, so it is rebuilt
from scratch every time rather than appended to: a parser fix then reaches
every past sample, not only the new ones.

CSV rather than Parquet so the collector side keeps working with the standard
library alone. An empty cell means None, the same None the parser returned.

Two things are added to what `parse_sample` gives:

- `schema`, so a row from before the context commands existed can be told
  apart from one where they failed.
- `gap_s`, the seconds since the previous sample. A gap much longer than the
  sampling interval means the Mac slept or the collector stopped, and the
  target reconstruction has to know where those are.

Two samples with the same instant are the same sample written twice, for
example by a manual `once` racing the launchd agent. The first one read is
kept and the rest are counted, not silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import typing
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .parse import BatteryState, ContextState, PmsetState, parse_sample
from .store import DEFAULT_ROOT, read_samples

DEFAULT_OUT = Path("data/series/samples.csv")


def _field_types(cls, prefix: str = "") -> Dict[str, type]:
    """Column name to Python type, read from a parser dataclass."""
    hints = typing.get_type_hints(cls)
    out = {}
    for item in fields(cls):
        hint = hints[item.name]
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        out[prefix + item.name] = args[0] if args else hint
    return out


# The column order of the table. It follows parse_sample, with the two
# additions placed right after the timestamp.
COLUMN_TYPES: Dict[str, type] = {
    "sampled_at": str,
    "schema": int,
    "gap_s": float,
    "ioreg_ok": bool,
    "pmset_ok": bool,
    **_field_types(BatteryState),
    **_field_types(PmsetState, prefix="macos_"),
    **_field_types(ContextState),
}

COLUMNS: List[str] = list(COLUMN_TYPES)


@dataclass
class Series:
    """The table, plus what had to be left out of it."""

    rows: List[dict]
    duplicates: int = 0
    skipped_lines: int = 0

    def __len__(self) -> int:
        return len(self.rows)


def build_series(samples: Iterable[dict]) -> Series:
    """Parse, sort by instant, drop repeated instants and fill `gap_s`.

    Instants are compared as datetimes, not strings, so a sample stored with
    a non UTC offset still sorts where it belongs.
    """
    keyed = []
    for sample in samples:
        moment = datetime.fromisoformat(sample["sampled_at"])
        keyed.append((moment, sample))
    # Stable sort, so among equal instants the first one read stays first.
    keyed.sort(key=lambda pair: pair[0])

    rows: List[dict] = []
    duplicates = 0
    previous: Optional[datetime] = None
    for moment, sample in keyed:
        if previous is not None and moment == previous:
            duplicates += 1
            continue
        parsed = parse_sample(sample)
        row = {column: None for column in COLUMNS}
        row.update({key: value for key, value in parsed.items() if key in row})
        row["schema"] = sample.get("schema")
        row["gap_s"] = (
            None if previous is None else (moment - previous).total_seconds()
        )
        rows.append(row)
        previous = moment
    return Series(rows=rows, duplicates=duplicates)


def load_series(root: Path = DEFAULT_ROOT) -> Series:
    """Build the series from everything stored under `root`."""
    report = read_samples(root=root)
    series = build_series(report.samples)
    series.skipped_lines = report.skipped_lines
    return series


def _cell(value) -> str:
    return "" if value is None else str(value)


def write_csv(rows: Sequence[dict], path: Path = DEFAULT_OUT) -> Path:
    """Write the table, replacing any previous one in a single step.

    The file is written next to its destination and then renamed over it, so
    a reader never sees half a table, and an interrupted build leaves the
    previous table in place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in COLUMNS])
    os.replace(partial, path)
    return path


def _typed(text: str, kind: type):
    if text == "":
        return None
    if kind is bool:
        if text not in ("True", "False"):
            raise ValueError(f"not a bool: {text!r}")
        return text == "True"
    return kind(text)


def read_csv(path: Path = DEFAULT_OUT) -> List[dict]:
    """Read a table written by `write_csv`, with the original types restored.

    A column this version does not know is kept as text, so an older reader
    still opens a newer table.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                key: _typed(value, COLUMN_TYPES.get(key, str))
                for key, value in record.items()
            }
            for record in reader
        ]


def summarize(series: Series, gap_threshold_s: float = 900.0) -> Dict[str, object]:
    """Counts that show whether the series is usable yet.

    `on_battery_with_estimate` is how many rows the baseline can be scored
    on: an estimate on AC power is a time to full, not a time to empty.
    `gaps` counts the breaks longer than `gap_threshold_s`, three missed
    samples at the default five minute interval.
    """
    rows = series.rows
    on_battery = [row for row in rows if row["macos_power_source"] == "Battery Power"]
    gaps = [row["gap_s"] for row in rows if row["gap_s"] is not None]
    return {
        "rows": len(rows),
        "first": rows[0]["sampled_at"] if rows else None,
        "last": rows[-1]["sampled_at"] if rows else None,
        "duplicates": series.duplicates,
        "skipped_lines": series.skipped_lines,
        "on_battery": len(on_battery),
        "on_battery_with_estimate": sum(
            1 for row in on_battery if row["macos_has_estimate"]
        ),
        "gaps": sum(1 for gap in gaps if gap > gap_threshold_s),
        "longest_gap_s": max(gaps) if gaps else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="rebuild the table from the raw samples")
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)

    sub.add_parser("summary", help="how much usable series there is")

    args = parser.parse_args(argv)
    series = load_series(root=args.root)

    if args.command == "build":
        path = write_csv(series.rows, args.out)
        print(f"{len(series)} rows -> {path}")
        if series.duplicates:
            print(f"dropped {series.duplicates} repeated instant(s)")
        if series.skipped_lines:
            print(f"skipped {series.skipped_lines} unreadable line(s)")
    elif args.command == "summary":
        for key, value in summarize(series).items():
            print(f"{key:<20}{_cell(value) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
