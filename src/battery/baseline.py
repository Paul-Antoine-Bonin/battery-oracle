"""The macOS estimate as a baseline prediction, one per sample.

`pmset -g batt` and `ioreg` are captured in the same sample, a fraction of a
second apart, so every sample already holds the macOS estimate next to the
hardware readings. Not every estimate is a prediction the model can be
compared against, though:

- On AC power the figure is a time to full, not a time to empty.
- After a power source change macOS shows `(no estimate)` for a while. That
  stays a gap. Filling it in would score the baseline on answers it never gave.
- The two commands run one after the other. If the charger is plugged in or
  pulled between them, `pmset` and `ioreg` describe two different moments.
  Such a sample is flagged rather than scored, since neither estimator should
  be judged on a moment that was not observed consistently.

`baseline_status` says which case a row is in. Only `scorable` rows carry
`macos_remaining_min` and `macos_empty_at`, the instant macOS expected the
battery to run out. The comparison step scores those against the time to
empty reconstructed afterwards.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional

SCORABLE = "scorable"
PMSET_FAILED = "pmset_failed"
ON_AC = "on_ac"
POWER_MISMATCH = "power_mismatch"
NOT_DISCHARGING = "not_discharging"
NO_ESTIMATE = "no_estimate"

# In the order they are checked, which is also the order they are reported.
STATUSES = (
    SCORABLE,
    PMSET_FAILED,
    ON_AC,
    POWER_MISMATCH,
    NOT_DISCHARGING,
    NO_ESTIMATE,
)


def _power_mismatch(row: dict) -> bool:
    """Whether ioreg contradicts pmset saying the Mac runs on battery.

    Either sign of the contradiction is enough: an external supply reported
    as connected, or a current flowing into the battery. When ioreg failed
    there is nothing to compare, and the pmset reading is taken as is.
    """
    if not row.get("ioreg_ok"):
        return False
    if row.get("external_connected") is True:
        return True
    amperage = row.get("amperage_ma")
    return amperage is not None and amperage > 0


def baseline_status(row: dict) -> str:
    """Why a parsed row can or cannot be used to score the macOS estimate."""
    if not row.get("pmset_ok"):
        return PMSET_FAILED
    if row.get("macos_power_source") != "Battery Power":
        return ON_AC
    if _power_mismatch(row):
        return POWER_MISMATCH
    if row.get("macos_state") != "discharging":
        return NOT_DISCHARGING
    if not row.get("macos_has_estimate") or row.get("macos_estimate_min") is None:
        return NO_ESTIMATE
    return SCORABLE


def baseline_fields(row: dict) -> Dict[str, Optional[object]]:
    """The baseline columns for one parsed row.

    `macos_empty_at` keeps the offset of `sampled_at`, so it compares with
    the sample instants as they are stored.
    """
    status = baseline_status(row)
    out: Dict[str, Optional[object]] = {
        "baseline_status": status,
        "macos_remaining_min": None,
        "macos_empty_at": None,
    }
    if status != SCORABLE:
        return out
    minutes = row["macos_estimate_min"]
    moment = datetime.fromisoformat(row["sampled_at"])
    out["macos_remaining_min"] = minutes
    out["macos_empty_at"] = (moment + timedelta(minutes=minutes)).isoformat()
    return out


def status_counts(rows: Iterable[dict]) -> Dict[str, int]:
    """How many rows fall in each status, every status listed even at 0."""
    counts = Counter(baseline_status(row) for row in rows)
    return {status: counts.get(status, 0) for status in STATUSES}
