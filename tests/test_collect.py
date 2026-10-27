"""Tests for the collect command line. No real pmset or ioreg calls."""

from datetime import datetime, timezone
from pathlib import Path

from battery import collect
from battery.sources import Sample
from battery.store import read_samples

FIXTURES = Path(__file__).parent / "fixtures"
MOMENT = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)


def fake_sample(pmset_ok=True):
    return Sample(
        sampled_at=MOMENT.isoformat(),
        commands={
            "pmset_batt": {
                "ok": pmset_ok,
                "stdout": (FIXTURES / "pmset_discharging.txt").read_text()
                if pmset_ok
                else "",
                "error": "" if pmset_ok else "timed out after 15.0s",
                "duration_s": 0.0,
            },
            "ioreg_battery": {
                "ok": True,
                "stdout": (FIXTURES / "ioreg_discharging.txt").read_text(),
                "error": "",
                "duration_s": 0.0,
            },
        },
    )


def test_take_one_appends_to_the_day_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(collect, "collect_sample", fake_sample)
    path = collect.take_one(root=tmp_path)

    assert path == tmp_path / "2026-08-16.jsonl"
    assert read_samples(root=tmp_path).count == 1
    assert " ok " in capsys.readouterr().out


def test_take_one_names_the_command_that_failed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(collect, "collect_sample", lambda: fake_sample(pmset_ok=False))
    collect.take_one(root=tmp_path)

    assert "partial, failed: pmset_batt" in capsys.readouterr().out
    # The ioreg half is still worth keeping.
    assert read_samples(root=tmp_path).count == 1


def test_take_one_quiet_prints_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(collect, "collect_sample", fake_sample)
    collect.take_one(root=tmp_path, quiet=True)

    assert capsys.readouterr().out == ""


def test_loop_sleeps_between_samples_not_after_the_last(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "take_one", lambda root: None)
    sleeps = []

    taken = collect.run_loop(
        interval=60, root=tmp_path, max_samples=3, sleep=sleeps.append
    )

    assert taken == 3
    assert sleeps == [60, 60]


def test_loop_survives_a_failing_sample(monkeypatch, tmp_path, capsys):
    calls = []

    def flaky(root):
        calls.append(root)
        if len(calls) == 1:
            raise OSError("disk full")

    monkeypatch.setattr(collect, "take_one", flaky)
    taken = collect.run_loop(
        interval=1, root=tmp_path, max_samples=2, sleep=lambda s: None
    )

    assert taken == 2
    assert len(calls) == 2
    assert "sample failed: OSError: disk full" in capsys.readouterr().err


def test_loop_with_zero_max_samples_takes_none(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "take_one", lambda root: None)
    assert collect.run_loop(root=tmp_path, max_samples=0, sleep=lambda s: None) == 0


def test_missing_tools_stop_once_with_exit_code_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(collect, "missing_tools", lambda: ["pmset", "ioreg"])

    assert collect.main(["--root", str(tmp_path), "once"]) == 2
    err = capsys.readouterr().err
    assert "pmset, ioreg" in err
    assert "only runs on macOS" in err
    assert not list(tmp_path.iterdir())


def test_status_still_works_without_the_tools(monkeypatch, tmp_path, capsys):
    # Reading what was collected must work anywhere, Linux included.
    monkeypatch.setattr(collect, "missing_tools", lambda: ["pmset", "ioreg"])

    assert collect.main(["--root", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "samples : 0" in out
    assert "no samples yet" in out


def test_status_lists_samples_per_day(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(collect, "collect_sample", fake_sample)
    collect.take_one(root=tmp_path, quiet=True)
    collect.take_one(root=tmp_path, quiet=True)

    collect.print_status(root=tmp_path)
    out = capsys.readouterr().out
    assert "samples : 2" in out
    assert "2026-08-16" in out
