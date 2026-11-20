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
    """Records every launchctl call and answers per verb.

    `answers` maps a verb such as "bootstrap" to (exit code, stderr). Verbs
    not in the map succeed with no output.
    """

    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    def __call__(self, *args):
        self.calls.append(args)
        code, stderr = self.answers.get(args[0], (0, ""))
        return subprocess.CompletedProcess(
            ["launchctl", *args], code, stdout="", stderr=stderr
        )


@pytest.fixture
def agents(tmp_path, monkeypatch):
    home = tmp_path / "agents"
    monkeypatch.setattr(schedule, "agents_dir", lambda: home)
    monkeypatch.setattr(schedule, "_uid", lambda: 501)
    return home


def use(monkeypatch, answers=None):
    fake = FakeLaunchctl(answers)
    monkeypatch.setattr(schedule, "_launchctl", fake)
    return fake


def test_install_boots_out_before_bootstrapping(tmp_path, agents, monkeypatch):
    fake = use(monkeypatch)
    path = schedule.install(Path("/usr/bin/python3"), tmp_path / "repo", 600)

    assert path == agents / f"{LABEL}.plist"
    assert fake.calls == [
        ("bootout", f"gui/501/{LABEL}"),
        ("bootstrap", "gui/501", str(path)),
    ]


def test_install_survives_a_bootout_of_an_agent_that_was_not_loaded(
    tmp_path, agents, monkeypatch
):
    # A first install has nothing to boot out, and launchctl says so with a
    # nonzero exit. That must not stop the install.
    use(monkeypatch, {"bootout": (3, "No such process\n")})
    path = schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")
    assert path.exists()


def test_a_failed_bootstrap_raises_with_launchctl_stderr(
    tmp_path, agents, monkeypatch
):
    # Different messages per verb prove the bootstrap one reached the error.
    use(
        monkeypatch,
        {
            "bootout": (3, "No such process\n"),
            "bootstrap": (5, "Input/output error\n"),
        },
    )
    with pytest.raises(RuntimeError, match=r"exit 5\): Input/output error$"):
        schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")

    # A plist left in LaunchAgents would be loaded at the next login.
    assert not (agents / f"{LABEL}.plist").exists()


def test_a_silent_bootstrap_failure_still_names_the_exit_code(
    tmp_path, agents, monkeypatch
):
    use(monkeypatch, {"bootstrap": (5, "")})
    with pytest.raises(RuntimeError, match=r"exit 5\): no output$"):
        schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")


@pytest.mark.parametrize("interval", [0, -300])
def test_install_rejects_an_interval_that_is_not_positive(
    tmp_path, agents, monkeypatch, interval
):
    fake = use(monkeypatch)
    with pytest.raises(ValueError, match="positive"):
        schedule.install(Path("/usr/bin/python3"), tmp_path / "repo", interval)

    assert fake.calls == []
    assert not agents.exists()


def test_uninstall_removes_the_plist(tmp_path, agents, monkeypatch):
    use(monkeypatch)
    path = schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")

    fake = use(monkeypatch)
    assert schedule.uninstall() is True
    assert not path.exists()
    assert fake.calls == [("bootout", f"gui/501/{LABEL}")]


def test_uninstall_with_nothing_installed_reports_nothing_removed(
    agents, monkeypatch
):
    use(monkeypatch, {"bootout": (3, "No such process\n")})
    assert schedule.uninstall() is False


def test_install_command_prints_the_label_interval_and_plist(
    tmp_path, agents, monkeypatch, capsys
):
    use(monkeypatch)
    repo = tmp_path / "repo"
    code = schedule.main(
        ["install", "--repo", str(repo), "--python", "/usr/bin/python3",
         "--interval", "120"]
    )

    assert code == 0
    path = agents / f"{LABEL}.plist"
    assert capsys.readouterr().out.splitlines() == [
        f"installed {LABEL}, every 120s",
        f"plist: {path}",
    ]
    with path.open("rb") as handle:
        assert plistlib.load(handle)["StartInterval"] == 120


def test_uninstall_command_says_what_it_did(tmp_path, agents, monkeypatch, capsys):
    use(monkeypatch)
    schedule.install(Path("/usr/bin/python3"), tmp_path / "repo")

    assert schedule.main(["uninstall"]) == 0
    assert capsys.readouterr().out.strip() == "removed"

    assert schedule.main(["uninstall"]) == 0
    assert capsys.readouterr().out.strip() == "nothing to remove"


def test_status_exit_code_follows_launchctl_print(agents, monkeypatch, capsys):
    use(monkeypatch)
    assert schedule.main(["status"]) == 0
    assert capsys.readouterr().out.strip() == f"{LABEL}: loaded"

    use(monkeypatch, {"print": (113, "Could not find service\n")})
    assert schedule.main(["status"]) == 1
    assert capsys.readouterr().out.strip() == f"{LABEL}: not loaded"


def test_show_prints_the_plist_without_writing_it(
    tmp_path, agents, monkeypatch, capsys
):
    fake = use(monkeypatch)
    assert schedule.main(["show", "--repo", str(tmp_path), "--interval", "120"]) == 0

    printed = plistlib.loads(capsys.readouterr().out.encode())
    assert printed["StartInterval"] == 120
    assert printed["WorkingDirectory"] == str(tmp_path.resolve())
    assert fake.calls == []
    assert not agents.exists()
