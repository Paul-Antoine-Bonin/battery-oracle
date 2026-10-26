"""Append-only JSON Lines storage, one file per UTC day.

    data/raw/YYYY-MM-DD.jsonl

JSON Lines rather than Parquet, deliberately. This collector runs unattended
for months and will be killed mid-write eventually, by a reboot, a lid close,
or a battery that actually ran out. A truncated final line costs one sample and
nothing else, where a half-written columnar file can cost the whole day. Reads
skip unparseable lines and report how many they skipped.

Roughly 1 MiB per day at one sample every five minutes. The parsing step of the
roadmap turns this into something compact, after which the raw files can be
pruned or compressed.

The tree is git-ignored. `pmset -g log` and the process context this project
will eventually add both carry application names, and the brief says those
never enter a commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

DEFAULT_ROOT = Path("data/raw")


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("naive datetime rejected: pass an aware UTC datetime")
    return moment.astimezone(timezone.utc)


def day_path(moment: datetime, root: Path = DEFAULT_ROOT) -> Path:
    """The file a sample taken at `moment` belongs in."""
    return Path(root) / f"{_as_utc(moment).strftime('%Y-%m-%d')}.jsonl"


def append_sample(sample: dict, root: Path = DEFAULT_ROOT) -> Path:
    """Append one sample and return the file it went to.

    The line is built in memory first, so a serialisation error cannot leave a
    half-written line behind.
    """
    moment = datetime.fromisoformat(sample["sampled_at"])
    path = day_path(moment, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path


def list_days(root: Path = DEFAULT_ROOT) -> List[Path]:
    """Every day file, oldest first."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted(base.glob("*.jsonl"))


@dataclass
class ReadReport:
    """What a read found, including what it had to throw away."""

    samples: List[dict]
    skipped_lines: int
    files: int

    @property
    def count(self) -> int:
        return len(self.samples)


def read_samples(
    root: Path = DEFAULT_ROOT,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> ReadReport:
    """Read samples across days, tolerating a truncated last line.

    `start` and `end` are inclusive bounds on the sample instant.
    """
    samples: List[dict] = []
    skipped = 0
    files = 0

    for path in list_days(root):
        files += 1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    moment = datetime.fromisoformat(sample["sampled_at"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # A line cut in half by a shutdown, valid JSON that is not
                    # an object, or a sample without a usable timestamp. One
                    # lost sample, not a lost day.
                    skipped += 1
                    continue
                if moment.tzinfo is None:
                    # The collector always writes an offset. A naive timestamp
                    # cannot be placed in UTC, and comparing it with the bounds
                    # would raise, so it counts as damage too.
                    skipped += 1
                    continue
                if start is not None and moment < _as_utc(start):
                    continue
                if end is not None and moment > _as_utc(end):
                    continue
                samples.append(sample)

    return ReadReport(samples=samples, skipped_lines=skipped, files=files)


def coverage(root: Path = DEFAULT_ROOT) -> List[Tuple[str, int]]:
    """(day, sample count) pairs, so an interrupted collection is visible.

    The roadmap's main risk is a collector that quietly stopped. A day with far
    fewer samples than its neighbours is how that shows up.
    """
    out = []
    for path in list_days(root):
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        out.append((path.stem, count))
    return out
