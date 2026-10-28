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
    assert set(sample.commands) == {
        "pmset_batt",
        "ioreg_battery",
        "cpu_load",
        "wifi_power",
        "display_brightness",
        "foreground_app",
    }
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

    assert payload["schema"] == 3
    assert isinstance(payload["commands"], dict)


def test_ioreg_is_called_without_line_wrapping():
    # -w0 keeps the captured text independent of the terminal width.
    assert "-w0" in sources.COMMANDS["ioreg_battery"]


def test_missing_tools_lists_absent_binaries(monkeypatch):
    monkeypatch.setattr(
        sources.shutil, "which", lambda name: None if name == "ioreg" else "/bin/x"
    )
    assert missing_tools() == ["ioreg"]


def test_a_failed_context_command_keeps_the_battery_half(monkeypatch):
    def no_wifi(argv, timeout):
        if argv[0] == "networksetup":
            return CommandResult(False, error="exit 4: en0 is not a Wi-Fi interface")
        return CommandResult(True, "output")

    monkeypatch.setattr(sources, "run_command", no_wifi)
    sample = collect_sample(now=MOMENT)

    assert not sample.commands["wifi_power"]["ok"]
    assert sample.commands["pmset_batt"]["ok"]
    assert sample.commands["ioreg_battery"]["ok"]


def test_missing_context_tools_do_not_block_the_collector(monkeypatch):
    monkeypatch.setattr(
        sources.shutil,
        "which",
        lambda name: None if name in ("sysctl", "networksetup") else "/usr/bin/x",
    )
    assert missing_tools() == []


def test_brightness_dump_is_unwrapped_and_shallow():
    argv = sources.CONTEXT_COMMANDS["display_brightness"]
    assert "-w0" in argv
    assert argv[argv.index("-d") + 1] == "1"


def test_foreground_app_resolves_the_front_session_in_one_shell_call():
    argv = sources.CONTEXT_COMMANDS["foreground_app"]
    assert argv[:2] == ("/bin/sh", "-c")
    script = argv[2]
    # The second call must only run when the first one worked.
    assert "lsappinfo front" in script and "&&" in script
    assert "-only bundleid" in script


def _fake_lsappinfo(tmp_path, front_exit):
    tool = tmp_path / "lsappinfo"
    tool.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = front ]; then echo "ASN:0x0-0x1001:"; exit ' + str(front_exit) + "; fi\n"
        'echo "\\"CFBundleIdentifier\\"=\\"com.example.editor\\""\n'
    )
    tool.chmod(0o755)
    return tool


def test_foreground_app_script_runs_both_calls(tmp_path, monkeypatch):
    _fake_lsappinfo(tmp_path, front_exit=0)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    result = run_command(sources.CONTEXT_COMMANDS["foreground_app"])

    assert result.ok
    assert result.stdout.strip() == '"CFBundleIdentifier"="com.example.editor"'


def test_foreground_app_script_fails_when_front_fails(tmp_path, monkeypatch):
    _fake_lsappinfo(tmp_path, front_exit=1)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    result = run_command(sources.CONTEXT_COMMANDS["foreground_app"])

    assert not result.ok
    assert "exit 1" in result.error
