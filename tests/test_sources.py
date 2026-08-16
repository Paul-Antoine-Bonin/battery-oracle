"""Tests for the command capture layer. No real pmset or ioreg calls."""

import subprocess
from datetime import datetime, timezone

from battery import sources
from battery.sources import CommandResult, collect_sample, missing_tools, run_command

MOMENT = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)


def test_run_command_captures_stdout(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        sources.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "94%; charging", ""),
    )
    result = run_command(("pmset", "-g", "batt"))

    assert result.ok
    assert result.stdout == "94%; charging"


def test_a_missing_binary_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    result = run_command(("ioreg",))

    assert not result.ok
    assert "not found" in result.error


def test_a_nonzero_exit_is_reported(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda name: "/usr/bin/ioreg")
    monkeypatch.setattr(
        sources.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "boom"),
    )
    result = run_command(("ioreg",))

    assert not result.ok
    assert "exit 1" in result.error and "boom" in result.error


def test_a_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda name: "/usr/bin/ioreg")

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ioreg", timeout=15)

    monkeypatch.setattr(sources.subprocess, "run", hang)
    result = run_command(("ioreg",), timeout=15)

    assert not result.ok
    assert "timed out" in result.error


def test_an_os_error_is_reported(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda name: "/usr/bin/ioreg")

    def explode(*a, **k):
        raise OSError("too many open files")

    monkeypatch.setattr(sources.subprocess, "run", explode)
    result = run_command(("ioreg",))

    assert not result.ok
    assert "OSError" in result.error


def test_collect_sample_records_every_command(monkeypatch):
    monkeypatch.setattr(
        sources, "run_command", lambda argv, timeout: CommandResult(True, "output")
    )
    sample = collect_sample(now=MOMENT)

    assert sample.sampled_at == "2026-08-16T20:00:00+00:00"
    assert set(sample.commands) == {"pmset_batt", "ioreg_battery"}
    assert all(entry["ok"] for entry in sample.commands.values())


def test_a_partial_sample_is_kept(monkeypatch):
    # ioreg failing must not throw away the pmset half.
    def one_fails(argv, timeout):
        if argv[0] == "ioreg":
            return CommandResult(False, error="exit 1")
        return CommandResult(True, "94%; charging")

    monkeypatch.setattr(sources, "run_command", one_fails)
    sample = collect_sample(now=MOMENT)

    assert sample.commands["pmset_batt"]["ok"]
    assert not sample.commands["ioreg_battery"]["ok"]
    assert sample.commands["ioreg_battery"]["error"] == "exit 1"


def test_sample_serialises_to_a_plain_dict(monkeypatch):
    monkeypatch.setattr(
        sources, "run_command", lambda argv, timeout: CommandResult(True, "x")
    )
    payload = collect_sample(now=MOMENT).to_dict()

    assert payload["schema"] == 1
    assert isinstance(payload["commands"], dict)


def test_ioreg_is_called_without_line_wrapping():
    # -w0 keeps the captured text independent of the terminal width.
    assert "-w0" in sources.COMMANDS["ioreg_battery"]


def test_missing_tools_lists_absent_binaries(monkeypatch):
    monkeypatch.setattr(
        sources.shutil, "which", lambda name: None if name == "ioreg" else "/bin/x"
    )
    assert missing_tools() == ["ioreg"]
