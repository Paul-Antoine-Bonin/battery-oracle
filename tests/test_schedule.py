"""Tests for the launchd agent and its command line. Nothing is loaded into launchd."""

import plistlib
import subprocess
from pathlib import Path

import pytest

from battery import schedule
from battery.schedule import LABEL, build_plist, write_plist


def definition(tmp_path, interval=300):
    return build_plist(
        python=Path("/usr/bin/python3"), repo=tmp_path, interval=interval
    )


def test_the_agent_runs_the_once_subcommand(tmp_path):
    # loop inside a scheduled job would leave two collectors running.
    assert definition(tmp_path)["ProgramArguments"][-3:] == [
        "-m",
        "battery.collect",
        "once",
    ]


def test_interval_is_carried_through(tmp_path):
    assert definition(tmp_path, interval=600)["StartInterval"] == 600


def test_interval_is_an_int_not_a_float(tmp_path):
    # launchd rejects a real where it expects an integer.
    assert isinstance(definition(tmp_path, interval=300)["StartInterval"], int)


def test_run_at_load_makes_a_bad_install_obvious(tmp_path):
    assert definition(tmp_path)["RunAtLoad"] is True


def test_pythonpath_points_at_the_src_layout(tmp_path):
    env = definition(tmp_path)["EnvironmentVariables"]
    assert env["PYTHONPATH"] == str(tmp_path / "src")


def test_logs_stay_inside_the_ignored_data_tree(tmp_path):
    plist = definition(tmp_path)
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert plist[key].startswith(str(tmp_path / "data"))


def test_label_is_not_personal():
    # The label shows up in launchctl output and in the plist filename.
    assert LABEL == "local.battery-oracle"


def test_written_plist_parses_back(tmp_path):
    path = write_plist(definition(tmp_path), path=tmp_path / "agent.plist")
    with path.open("rb") as handle:
        loaded = plistlib.load(handle)

    assert loaded["Label"] == LABEL
    assert loaded["StartInterval"] == 300


def test_writing_creates_the_log_directory(tmp_path):
    write_plist(definition(tmp_path), path=tmp_path / "agent.plist")
    assert (tmp_path / "data" / "logs").is_dir()


# The install, uninstall and status paths, with launchctl replaced by a stub
# and the LaunchAgents directory moved under tmp_path.


class FakeLaunchctl:
    """Records every launchctl call and answers with a fixed return code."""

    def __init__(self, codes=None):
        self.calls = []
        self.codes = codes or {}

    def __call__(self, *args):
        self.calls.append(args)
        code, stderr = self.codes.get(args[0], (0, ""))
        return subprocess.CompletedProcess(
            ["launchctl", *args], code, stdout="", stderr=stderr
        )

    def verbs(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def agents(tmp_path, monkeypatch):
    directory = tmp_path / "LaunchAgents"
    monkeypatch.setattr(schedule, "agents_dir", lambda: directory)
    return directory


def stub(monkeypatch, codes=None):
    fake = FakeLaunchctl(codes)
    monkeypatch.setattr(schedule, "_launchctl", fake)
    return fake


def test_install_boots_out_before_bootstrapping(tmp_path, agents, monkeypatch):
    fake = stub(monkeypatch)
    path = schedule.install(Path("/usr/bin/python3"), tmp_path, interval=600)

    assert path == agents / f"{LABEL}.plist"
    assert fake.verbs() == ["bootout", "bootstrap"]
    assert fake.calls[1][-1] == str(path)
    with path.open("rb") as handle:
        assert plistlib.load(handle)["StartInterval"] == 600


def test_install_ignores_a_failed_bootout(tmp_path, agents, monkeypatch):
    # On a first install there is nothing to boot out, and launchctl says so.
    stub(monkeypatch, {"bootout": (3, "No such process")})
    assert schedule.install(Path("/usr/bin/python3"), tmp_path).exists()


def test_a_failed_bootstrap_raises_with_the_launchctl_message(
    tmp_path, agents, monkeypatch
):
    stub(monkeypatch, {"bootstrap": (5, "Bootstrap failed: 5: Input/output error\n")})
    with pytest.raises(RuntimeError, match="Input/output error"):
        schedule.install(Path("/usr/bin/python3"), tmp_path)


def test_uninstall_removes_the_plist(tmp_path, agents, monkeypatch):
    fake = stub(monkeypatch)
    schedule.install(Path("/usr/bin/python3"), tmp_path)

    assert schedule.uninstall() is True
    assert not (agents / f"{LABEL}.plist").exists()
    assert fake.verbs()[-1] == "bootout"


def test_uninstall_without_a_plist_reports_nothing_removed(agents, monkeypatch):
    stub(monkeypatch)
    assert schedule.uninstall() is False


def test_status_exit_code_follows_launchctl_print(agents, monkeypatch, capsys):
    stub(monkeypatch)
    assert schedule.main(["status"]) == 0
    assert "not loaded" not in capsys.readouterr().out

    stub(monkeypatch, {"print": (113, "Could not find service")})
    assert schedule.main(["status"]) == 1
    assert "not loaded" in capsys.readouterr().out


def test_show_prints_the_plist_without_writing_it(tmp_path, agents, monkeypatch, capsys):
    fake = stub(monkeypatch)
    assert schedule.main(["show", "--repo", str(tmp_path), "--interval", "120"]) == 0

    printed = plistlib.loads(capsys.readouterr().out.encode())
    assert printed["StartInterval"] == 120
    assert printed["WorkingDirectory"] == str(tmp_path.resolve())
    assert fake.calls == []
    assert not agents.exists()
