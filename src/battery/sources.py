"""The macOS commands this project reads battery state from.

Two sources, captured verbatim:

- `pmset -g batt`                    the percentage, the charging state, and
                                     macOS's own time-remaining estimate, which
                                     is the baseline the project has to beat
- `ioreg -r -c AppleSmartBattery`    the hardware view: raw capacity, cycle
                                     count, voltage, amperage, temperature

Raw text is stored rather than parsed values. Parsing is the next step of the
roadmap, and the ordering matters: collection has to start now, because the
target variable needs months of real discharge cycles. Keeping the raw output
means a field nobody thought to extract today is still recoverable from
samples taken today.

Nothing here reaches the network, and no command output leaves the machine.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

# `-w0` stops ioreg wrapping lines at the terminal width, which would make the
# stored text depend on the window it was captured from.
COMMANDS: Dict[str, Sequence[str]] = {
    "pmset_batt": ("pmset", "-g", "batt"),
    "ioreg_battery": ("ioreg", "-r", "-c", "AppleSmartBattery", "-w0"),
}

DEFAULT_TIMEOUT = 15.0


@dataclass
class CommandResult:
    """One command's output, or the reason there is none."""

    ok: bool
    stdout: str = ""
    error: str = ""
    duration_s: float = 0.0


@dataclass
class Sample:
    """One moment of battery state, as raw command output."""

    sampled_at: str                          # ISO 8601, UTC, with offset
    schema: int = 1
    commands: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_command(
    argv: Sequence[str], timeout: float = DEFAULT_TIMEOUT
) -> CommandResult:
    """Run one command and capture stdout, never raising.

    A collector that dies because `ioreg` hiccuped once would leave a hole in a
    dataset that cannot be backfilled, so every failure mode returns a result
    instead.
    """
    if shutil.which(argv[0]) is None:
        return CommandResult(ok=False, error=f"{argv[0]} not found on PATH")

    start = utcnow()
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(ok=False, error=f"timed out after {timeout}s")
    except OSError as exc:
        return CommandResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    duration = (utcnow() - start).total_seconds()
    if completed.returncode != 0:
        return CommandResult(
            ok=False,
            error=f"exit {completed.returncode}: {completed.stderr.strip()[:200]}",
            duration_s=duration,
        )
    return CommandResult(ok=True, stdout=completed.stdout, duration_s=duration)


def collect_sample(
    commands: Optional[Dict[str, Sequence[str]]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    now: Optional[datetime] = None,
) -> Sample:
    """Capture one sample. Partial samples are kept, not discarded.

    If `ioreg` fails but `pmset` works, the pmset half is still worth having,
    and the failure is recorded so a later analysis can see the gap rather than
    guess at it.
    """
    commands = COMMANDS if commands is None else commands
    moment = now or utcnow()

    sample = Sample(sampled_at=moment.isoformat())
    for name, argv in commands.items():
        result = run_command(argv, timeout=timeout)
        sample.commands[name] = {
            "ok": result.ok,
            "stdout": result.stdout,
            "error": result.error,
            "duration_s": round(result.duration_s, 4),
        }
    return sample


def missing_tools(commands: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
    """Commands that are not on PATH, so setup fails loudly rather than silently."""
    commands = COMMANDS if commands is None else commands
    return [argv[0] for argv in commands.values() if shutil.which(argv[0]) is None]
