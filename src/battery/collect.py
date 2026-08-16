"""Take battery samples, once or on a loop.

    python -m battery.collect once
    python -m battery.collect loop --interval 300
    python -m battery.collect status

`once` is what the scheduler calls. `loop` is for running by hand in a terminal
while watching it work.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .sources import collect_sample, missing_tools
from .store import DEFAULT_ROOT, append_sample, coverage, read_samples

DEFAULT_INTERVAL = 300.0


def take_one(root: Path = DEFAULT_ROOT, quiet: bool = False) -> Path:
    """Capture one sample and append it."""
    sample = collect_sample()
    path = append_sample(sample.to_dict(), root=root)
    if not quiet:
        failed = [
            name for name, result in sample.commands.items() if not result["ok"]
        ]
        state = "ok" if not failed else f"partial, failed: {', '.join(failed)}"
        print(f"{sample.sampled_at}  {state}  -> {path}")
    return path


def run_loop(
    interval: float = DEFAULT_INTERVAL,
    root: Path = DEFAULT_ROOT,
    max_samples: Optional[int] = None,
    sleep=time.sleep,
) -> int:
    """Sample every `interval` seconds until interrupted.

    Returns the number of samples taken. A failure inside one sample is caught
    and reported: the loop must outlive a transient error, not die on it.
    """
    taken = 0
    while max_samples is None or taken < max_samples:
        try:
            take_one(root=root)
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            print(f"sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        taken += 1
        if max_samples is not None and taken >= max_samples:
            break
        sleep(interval)
    return taken


def print_status(root: Path = DEFAULT_ROOT) -> None:
    report = read_samples(root=root)
    days = coverage(root=root)

    print(f"samples : {report.count}")
    print(f"files   : {report.files}")
    if report.skipped_lines:
        print(f"skipped : {report.skipped_lines} unreadable line(s)")
    if days:
        print("per day :")
        for day, count in days[-14:]:
            print(f"    {day}  {count:>5}")
    else:
        print("no samples yet")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("once", help="take a single sample")

    loop = sub.add_parser("loop", help="sample repeatedly until interrupted")
    loop.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    loop.add_argument("--max-samples", type=int, default=None)

    sub.add_parser("status", help="what has been collected so far")

    args = parser.parse_args(argv)

    absent = missing_tools()
    if absent and args.command != "status":
        print(
            f"missing required command(s): {', '.join(absent)}. "
            f"This collector only runs on macOS.",
            file=sys.stderr,
        )
        return 2

    if args.command == "once":
        take_one(root=args.root)
    elif args.command == "loop":
        try:
            run_loop(
                interval=args.interval,
                root=args.root,
                max_samples=args.max_samples,
            )
        except KeyboardInterrupt:
            print("\nstopped", file=sys.stderr)
    elif args.command == "status":
        print_status(root=args.root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
