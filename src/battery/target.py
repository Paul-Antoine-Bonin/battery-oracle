"""The target: how long the battery actually lasted from each row.

This is reconstructed after the fact from the discharge sessions in
`battery.sessions`. For every row inside a session it answers the question
the macOS estimate tries to answer at that moment: how many minutes until
the battery runs out.

Only a session that ends `empty` gives that answer. Every other session ends
before the battery did, by a charger, a gap or the end of the data, so its
rows only get a lower bound: the battery lasted at least until the session
ended. Those rows are censored, and `target_status` says which kind a row
is:

| Status | Meaning |
|---|---|
| `exact` | the session ran out, `actual_remaining_min` is set |
| `censored` | the session ended first, only `remaining_at_least_min` is set |
| `none` | the row is not part of a discharge session |

The empty instant is taken as the last sample of the session. The battery
really ran out somewhere between that sample and the next one that never
came, so `actual_remaining_min` can be short by up to one sampling interval.
That error is the same for every row of a session, and it is small next to
the hours a full discharge takes, so it is accepted rather than guessed at.

Censored rows are kept, not dropped. Full discharges are rare, and a model
or a comparison that ignored the lower bounds would only ever see the days
the battery ran flat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .sessions import (
    DEFAULT_EMPTY_FRACTION,
    DEFAULT_MAX_GAP_S,
    EMPTY,
    find_sessions,
)

EXACT = "exact"
CENSORED = "censored"
NONE = "none"

TARGET_STATUSES = (EXACT, CENSORED, NONE)

TARGET_COLUMNS = (
    "session_id",
    "session_end_reason",
    "target_status",
    "empty_at",
    "actual_remaining_min",
    "remaining_at_least_min",
)


def _instant(row: dict) -> datetime:
    return datetime.fromisoformat(row["sampled_at"])


def _empty_target() -> Dict[str, Optional[object]]:
    out: Dict[str, Optional[object]] = {column: None for column in TARGET_COLUMNS}
    out["target_status"] = NONE
    return out


def target_fields(
    rows: Sequence[dict],
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    empty_fraction: float = DEFAULT_EMPTY_FRACTION,
) -> List[Dict[str, Optional[object]]]:
    """The target columns for every row, in the same order as `rows`.

    `rows` must be sorted by instant, as `battery.series` builds them.
    `empty_at` keeps the offset of the sample it comes from, so it compares
    with `macos_empty_at` directly.
    """
    out = [_empty_target() for _ in rows]
    for session in find_sessions(rows, max_gap_s, empty_fraction):
        end = _instant(rows[session.last])
        exact = session.end_reason == EMPTY
        for i in session.rows:
            minutes = (end - _instant(rows[i])).total_seconds() / 60.0
            fields = out[i]
            fields["session_id"] = session.index
            fields["session_end_reason"] = session.end_reason
            fields["remaining_at_least_min"] = minutes
            if exact:
                fields["target_status"] = EXACT
                fields["empty_at"] = rows[session.last]["sampled_at"]
                fields["actual_remaining_min"] = minutes
            else:
                fields["target_status"] = CENSORED
    return out


def target_counts(fields: Sequence[dict]) -> Dict[str, int]:
    """How many rows have each target status, every status listed."""
    counts = {status: 0 for status in TARGET_STATUSES}
    for item in fields:
        counts[item["target_status"]] += 1
    return counts
