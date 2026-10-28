"""Install the collector as a launchd agent, so it survives logout and reboot.

    python -m battery.schedule install --interval 300
    python -m battery.schedule status
    python -m battery.schedule uninstall

launchd rather than cron: cron does not run when the machine is asleep and
never catches up, whereas launchd fires a missed `StartInterval` job as soon as
the Mac wakes. For a battery study, the samples around a sleep are exactly the
interesting ones.

The plist is generated at install time, never committed. It has to contain
absolute paths to this checkout and to the Python interpreter, and those are
personal paths that the project brief keeps out of the repository.
"""

from __future__ import annotations

import argparse
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

LABEL = "local.battery-oracle"

DEFAULT_INTERVAL = 300


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(label: str = LABEL) -> Path:
    return agents_dir() / f"{label}.plist"


def build_plist(
    python: Path,
    repo: Path,
    interval: int = DEFAULT_INTERVAL,
    label: str = LABEL,
) -> dict:
    """The launchd job definition.

    `RunAtLoad` fires one sample immediately, which turns a silent misinstall
    into an obvious one.
    """
    log_dir = repo / "data" / "logs"
    return {
        "Label": label,
        "ProgramArguments": [str(python), "-m", "battery.collect", "once"],
        "WorkingDirectory": str(repo),
        "EnvironmentVariables": {"PYTHONPATH": str(repo / "src")},
        "StartInterval": int(interval),
        "RunAtLoad": True,
        "StandardOutPath": str(log_dir / "collect.out"),
        "StandardErrorPath": str(log_dir / "collect.err"),
        "ProcessType": "Background",
    }


def write_plist(definition: dict, path: Optional[Path] = None) -> Path:
    path = path or plist_path(definition["Label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(definition["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(definition, handle)
    return path


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, check=False
    )


def install(python: Path, repo: Path, interval: int = DEFAULT_INTERVAL) -> Path:
    path = write_plist(build_plist(python, repo, interval))
    # bootout first so reinstalling picks up a changed interval.
    _launchctl("bootout", f"gui/{_uid()}/{LABEL}")
    result = _launchctl("bootstrap", f"gui/{_uid()}", str(path))
    if result.returncode != 0:
        # launchctl sometimes fails with nothing on either stream, so the exit
        # code is always included to keep the error from reading as blank.
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"launchctl bootstrap failed (exit {result.returncode}): {detail}"
        )
    return path


def uninstall(label: str = LABEL) -> bool:
    _launchctl("bootout", f"gui/{_uid()}/{label}")
    path = plist_path(label)
    if path.exists():
        path.unlink()
        return True
    return False


def is_loaded(label: str = LABEL) -> bool:
    return _launchctl("print", f"gui/{_uid()}/{label}").returncode == 0


def _uid() -> int:
    import os

    return os.getuid()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    inst = sub.add_parser("install", help="load the launchd agent")
    inst.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    inst.add_argument("--python", type=Path, default=Path(sys.executable))
    inst.add_argument("--repo", type=Path, default=Path.cwd())

    sub.add_parser("uninstall", help="unload and remove the agent")
    sub.add_parser("status", help="is the agent loaded")
    show = sub.add_parser("show", help="print the plist that would be written")
    show.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    show.add_argument("--python", type=Path, default=Path(sys.executable))
    show.add_argument("--repo", type=Path, default=Path.cwd())

    args = parser.parse_args(argv)

    if args.command == "install":
        path = install(args.python.resolve(), args.repo.resolve(), args.interval)
        print(f"installed {LABEL}, every {args.interval}s")
        print(f"plist: {path}")
    elif args.command == "uninstall":
        removed = uninstall()
        print("removed" if removed else "nothing to remove")
    elif args.command == "status":
        loaded = is_loaded()
        print(f"{LABEL}: {'loaded' if loaded else 'not loaded'}")
        return 0 if loaded else 1
    elif args.command == "show":
        definition = build_plist(
            args.python.resolve(), args.repo.resolve(), args.interval
        )
        sys.stdout.write(plistlib.dumps(definition).decode())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
