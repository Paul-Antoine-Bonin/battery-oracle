"""The macOS commands this project reads battery state and context from.

Two required battery sources, captured verbatim:

- `pmset -g batt`                    the percentage, the charging state, and
                                     macOS's own time-remaining estimate, which
                                     is the baseline the project has to beat
- `ioreg -r -c AppleSmartBattery`    the hardware view: raw capacity, cycle
                                     count, voltage, amperage, temperature

And four context sources, also verbatim, because how fast the battery drains
depends on what the machine is doing:

- `sysctl -n vm.loadavg hw.logicalcpu`   load average and core count
- `networksetup -getairportpower en0`    whether Wi-Fi is powered
- `ioreg -r -k IODisplayParameters`      built-in display brightness
- `lsappinfo info -only bundleid`        bundle id of the frontmost application

The frontmost application is personal. It is stored under `data/` like every
other sample and never belongs in the repository, in fixtures or anywhere else.

The context sources are optional. A sample where one of them failed is still a
full battery sample, so they never block the collector from starting.

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

# `lsappinfo front` prints the session id of the frontmost application and
# `lsappinfo info` resolves it. It needs two calls, hence the shell. The `&&`
# makes a failing first call fail the whole command, so it is recorded as a
# failure rather than as empty output. Only the bundle id is asked for: it is
# stable across languages and versions, unlike the display name.
FOREGROUND_APP_SCRIPT = (
    'asn="$(lsappinfo front)" && lsappinfo info -only bundleid "$asn"'
)

# `en0` is the Wi-Fi interface on every MacBook without a built-in Ethernet
# port. On other layouts the command reports an error, which is stored like
# any other failure rather than guessed around. `-d 1` keeps the brightness
# dump to the matching objects themselves, not their whole subtree.
CONTEXT_COMMANDS: Dict[str, Sequence[str]] = {
    "cpu_load": ("sysctl", "-n", "vm.loadavg", "hw.logicalcpu"),
    "wifi_power": ("networksetup", "-getairportpower", "en0"),
    "display_brightness": (
        "ioreg", "-r", "-k", "IODisplayParameters", "-d", "1", "-w0",
    ),
    "foreground_app": ("/bin/sh", "-c", FOREGROUND_APP_SCRIPT),
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
    # 1: battery commands only. 2: adds the context commands.
    # 3: adds the frontmost application.
    schema: int = 3
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
    commands = {**COMMANDS, **CONTEXT_COMMANDS} if commands is None else commands
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
    """Required commands that are not on PATH, so setup fails loudly.

    Only the battery commands are checked by default. A missing context command
    costs some context, not the sample.
    """
    commands = COMMANDS if commands is None else commands
    return [argv[0] for argv in commands.values() if shutil.which(argv[0]) is None]
