"""Split the time series into discharge sessions.

A discharge session is a run of consecutive rows where the Mac draws on its
battery, with no break in the sampling. The target, the time actually left
until the battery runs out, can only be reconstructed inside one session, so
this is the first step of building it.

A row counts as discharging from the hardware view when `ioreg` succeeded:
no external supply connected and a negative current. When `ioreg` failed,
the `pmset` view is used instead. The two disagree only for a fraction of a
second around a plug event, and the baseline already flags those rows.

Every session records why it ended, in `end_reason`:

| Reason | Meaning |
|---|---|
| `empty` | the charge was at or below `empty_fraction`, then sampling stopped |
| `plugged` | the next row is on a charger, or the charge went up |
| `gap` | sampling stopped for longer than `max_gap_s` with charge left |
| `open` | the series ends while the session is still running |

`empty` is the only reason that gives an exact target. A Mac that runs out
stops sampling, so the session ends on a long gap with the charge near zero.
A Mac plugged in at 2% shows the next row on a charger and no gap, so it
counts as `plugged`, not `empty`: it never actually ran out.

A gap breaks a session even when the Mac only slept on battery. The target
then never spans time the collector did not observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

EMPTY = "empty"
PLUGGED = "plugged"
GAP = "gap"
OPEN = "open"

END_REASONS = (EMPTY, PLUGGED, GAP, OPEN)

# Three missed samples at the default five minute interval, the same
# threshold `battery.series.summarize` uses for its gap count.
DEFAULT_MAX_GAP_S = 900.0

# macOS forces a sleep around 1% and the last sample before that can read a
# little higher, so the default leaves some room above zero.
DEFAULT_EMPTY_FRACTION = 0.03


@dataclass
class Session:
    """One discharge session, as indices into the rows it was built from."""

    index: int
    rows: List[int] = field(default_factory=list)
    end_reason: str = OPEN

    @property
    def first(self) -> int:
        return self.rows[0]

    @property
    def last(self) -> int:
        return self.rows[-1]


def is_discharging(row: dict) -> bool:
    """Whether the Mac was running on its battery at this row."""
    if row.get("ioreg_ok"):
        if row.get("external_connected") is True:
            return False
        amperage = row.get("amperage_ma")
        if amperage is not None:
            return amperage < 0
        return row.get("external_connected") is False
    if row.get("pmset_ok"):
        return (
            row.get("macos_power_source") == "Battery Power"
            and row.get("macos_state") == "discharging"
        )
    return False


def charge_of(row: dict) -> Optional[float]:
    """Charge as a fraction of the current maximum.

    The hardware figure first, since it is computed against the maximum read
    in the same sample. The macOS percentage is the fallback when `ioreg`
    failed.
    """
    fraction = row.get("charge_fraction")
    if fraction is not None:
        return fraction
    percent = row.get("macos_percent")
    return None if percent is None else percent / 100.0


def _instant(row: dict) -> datetime:
    return datetime.fromisoformat(row["sampled_at"])


def _charge_rose(before: dict, after: dict) -> bool:
    a, b = charge_of(before), charge_of(after)
    return a is not None and b is not None and b > a


def find_sessions(
    rows: Sequence[dict],
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    empty_fraction: float = DEFAULT_EMPTY_FRACTION,
) -> List[Session]:
    """Discharge sessions in `rows`, which must be sorted by instant.

    The gap between two rows is computed from their instants rather than read
    from `gap_s`, so this works on any sorted subset of the series too.
    """
    sessions: List[Session] = []
    current: Optional[Session] = None

    def close(reason: str) -> None:
        nonlocal current
        if current is not None:
            current.end_reason = reason
            sessions.append(current)
            current = None

    for i, row in enumerate(rows):
        if current is not None:
            previous = rows[current.last]
            gap = (_instant(row) - _instant(previous)).total_seconds()
            if gap > max_gap_s:
                charge = charge_of(previous)
                if charge is not None and charge <= empty_fraction:
                    close(EMPTY)
                elif _charge_rose(previous, row):
                    close(PLUGGED)
                else:
                    close(GAP)
            elif not is_discharging(row) or _charge_rose(previous, row):
                close(PLUGGED)

        if not is_discharging(row):
            continue
        if current is None:
            current = Session(index=len(sessions))
        current.rows.append(i)

    close(OPEN)
    return sessions


def session_counts(sessions: Sequence[Session]) -> dict:
    """How many sessions ended for each reason, every reason listed."""
    counts = {reason: 0 for reason in END_REASONS}
    for session in sessions:
        counts[session.end_reason] += 1
    return counts
