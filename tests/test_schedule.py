"""Tests for the launchd agent definition. Nothing is loaded into launchd."""

import plistlib
from pathlib import Path

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
