"""Tests for the launchd agent. launchctl is faked, nothing is loaded into launchd."""

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


# The launchctl calls themselves, with launchctl replaced by a recorder.


class FakeLaunchctl:
    """Records every launchctl call and answers with a chosen exit code."""

    def __init__(self, codes=None, stderr="", stdout=""):
        self.calls = []
        self.codes = codes or {}
        self.stderr = stderr
        self.stdout = stdout

    def __call__(self, *args):
        self.calls.append(args)
        code = self.codes.get(args[0], 0)
        return subprocess.CompletedProcess(
            ["launchctl", *args], code, stdout=self.stdout, stderr=self.stderr
        )


@pytest.fixture
def agents(tmp_path, monkeypatch):
    home = tmp_path / "agents"
    monkeypatch.setattr(schedule, "agents_dir", lambda: home)
    monkeypatch.setattr(schedule, "_uid", lambda: 501)
    return home


def use(monkeypatch, fake):
    monkeypatch.setattr(schedule, "_launchctl", fake)
    return fake


def test_install_boots_out_before_bootstrapping(tmp_path, agents, monkeypatch):
    fake = use(monkeypatch, FakeLaunchctl())
    path = schedule.install(Path("/usr/bin/python3"), tmp_path / "repo", 600)

    assert path == agents / f"{LABEL}.plist"
    assert [call[0] for call in fake.calls] == ["bootout", "bootstrap"]
    assert fake.calls[0] == ("bootout", f"gui/501/{LABEL}")
    assert fake.calls[1] == ("bootstrap", "gui/501", str(path))


def test_install_survives_a_bootout_of_an_agent_that_was_not_loaded(
    tmp_path, agents, monkeypatch
):
    # A first install has nothing to boot out, and launchctl says so with a
    # nonzero exit. That must not stop the install.
    use(monkeypatch, FakeLaunchctl(codes={"bootout": 3}))
    path = schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")
    assert path.exists()


def test_a_failed_bootstrap_raises_with_launchctl_stderr(
    tmp_path, agents, monkeypatch
):
    fake = FakeLaunchctl(codes={"bootstrap": 5}, stderr="Input/output error\n")
    use(monkeypatch, fake)
    with pytest.raises(RuntimeError, match=r"exit 5\): Input/output error$"):
        schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")


def test_a_silent_bootstrap_failure_still_names_the_exit_code(
    tmp_path, agents, monkeypatch
):
    use(monkeypatch, FakeLaunchctl(codes={"bootstrap": 5}))
    with pytest.raises(RuntimeError, match=r"exit 5\): no output$"):
        schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")


def test_uninstall_removes_the_plist(tmp_path, agents, monkeypatch):
    use(monkeypatch, FakeLaunchctl())
    path = schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")

    fake = use(monkeypatch, FakeLaunchctl())
    assert schedule.uninstall() is True
    assert not path.exists()
    assert fake.calls == [("bootout", f"gui/501/{LABEL}")]


def test_uninstall_with_nothing_installed_reports_nothing_removed(
    agents, monkeypatch
):
    use(monkeypatch, FakeLaunchctl(codes={"bootout": 3}))
    assert schedule.uninstall() is False


def test_status_exit_code_follows_launchctl_print(agents, monkeypatch, capsys):
    use(monkeypatch, FakeLaunchctl())
    assert schedule.main(["status"]) == 0
    assert capsys.readouterr().out.strip() == f"{LABEL}: loaded"

    use(monkeypatch, FakeLaunchctl(codes={"print": 113}))
    assert schedule.main(["status"]) == 1
    assert capsys.readouterr().out.strip() == f"{LABEL}: not loaded"
