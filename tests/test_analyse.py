"""End to end: raw command text in the store, a results summary out.

Every sample here is synthetic. The ioreg text is the discharging fixture with
the charge and the current rewritten, the current in the unsigned 64 bit form
ioreg really prints, so the whole chain is exercised from the stored text on.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from battery.analyse import analyse, main
from battery.series import read_csv
from battery.store import append_sample

FIXTURES = Path(__file__).parent / "fixtures"
START = datetime(2026, 8, 16, 8, 0, tzinfo=timezone.utc)
MAX_MAH = 4000


def text(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def ok(stdout):
    return {"ok": True, "stdout": stdout, "error": "", "duration_s": 0.0}


def ioreg_discharging(mah, draw_ma):
    """The discharging fixture at `mah` left, drawing `draw_ma`."""
    unsigned = 2**64 - draw_ma
    out = text("ioreg_discharging.txt")
    out = re.sub(r'"AppleRawCurrentCapacity" = \d+', f'"AppleRawCurrentCapacity" = {mah}', out)
    out = re.sub(r'"CurrentCapacity" = \d+', f'"CurrentCapacity" = {round(100 * mah / MAX_MAH)}', out)
    out = re.sub(r'"(Instant)?Amperage" ?= ?\d+', lambda m: f'"{m.group(1) or ""}Amperage" = {unsigned}', out)
    return out


def pmset_discharging(mah, estimate_min):
    hours, minutes = divmod(estimate_min, 60)
    return (
        "Now drawing from 'Battery Power'\n"
        f" -InternalBattery-0 (id=4653155)\t{round(100 * mah / MAX_MAH)}%; "
        f"discharging; {hours}:{minutes:02d} remaining present: true\n"
    )


def sample(moment, pmset, ioreg):
    return {
        "sampled_at": moment.isoformat(),
        "schema": 3,
        "commands": {
            "pmset_batt": ok(pmset),
            "ioreg_battery": ok(ioreg),
            "cpu_load": ok(text("sysctl_cpu_load.txt")),
            "wifi_power": ok(text("networksetup_wifi_on.txt")),
            "display_brightness": ok(text("ioreg_display_brightness.txt")),
            "foreground_app": ok(text("lsappinfo_foreground.txt")),
        },
    }


def full_discharge(root, start, draw_ma, macos_factor):
    """About 2000 mAh down to 2% every five minutes, then a charger after a gap.

    macOS is given an estimate `macos_factor` times the true time left, so it
    is wrong by a known amount.
    """
    step = draw_ma / 12  # mAh per five minutes
    last = 0.02 * MAX_MAH
    steps = int(1920 / step)
    for k in range(steps + 1):
        mah = last + (steps - k) * step
        left_min = mah / draw_ma * 60
        append_sample(
            sample(
                start + timedelta(minutes=5 * k),
                pmset_discharging(round(mah), round(left_min * macos_factor)),
                ioreg_discharging(round(mah), draw_ma),
            ),
            root=root,
        )
    append_sample(
        sample(
            start + timedelta(minutes=5 * k + 60),
            text("pmset_charging.txt"),
            text("ioreg_charging_no_raw_fields.txt"),
        ),
        root=root,
    )


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "raw"
    for day, (draw, factor) in enumerate([(1200, 1.3), (1500, 0.7), (1800, 1.2)]):
        full_discharge(root, START + timedelta(days=day), draw, factor)
    return root


def paths(tmp_path):
    return {
        "table": tmp_path / "series" / "samples.csv",
        "chart": tmp_path / "charts" / "error.png",
        "results": tmp_path / "results.md",
    }


def test_raw_samples_go_all_the_way_to_a_summary(tmp_path, store):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    out = paths(tmp_path)
    lines = []

    assert analyse(store, band_pct=20, echo=lines.append, **out)

    assert "full discharges so far: 3" in lines
    assert out["chart"].stat().st_size > 0
    summary = out["results"].read_text(encoding="utf-8")
    assert summary.startswith("## Results")
    assert "3 full discharges" in summary


def test_the_stored_current_is_read_as_a_discharge(tmp_path, store):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    out = paths(tmp_path)

    analyse(store, echo=lambda line: None, **out)

    rows = [r for r in read_csv(out["table"]) if r["macos_state"] == "discharging"]
    assert rows
    assert {r["amperage_ma"] for r in rows} == {-1200, -1500, -1800}
    assert all(r["target_status"] == "exact" for r in rows)


def test_report_chart_and_summary_come_from_one_fit(tmp_path, store, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    from battery import model

    calls = []
    real = model.cross_predict
    monkeypatch.setattr(
        model, "cross_predict", lambda items: calls.append(1) or real(items)
    )

    analyse(store, echo=lambda line: None, **paths(tmp_path))

    assert len(calls) == 1


def test_an_empty_store_writes_the_table_and_nothing_else(tmp_path, capsys):
    pytest.importorskip("sklearn")
    out = paths(tmp_path)

    code = main(
        [
            "--root", str(tmp_path / "raw"),
            "--table", str(out["table"]),
            "--chart", str(out["chart"]),
            "--results", str(out["results"]),
        ]
    )

    assert code == 1
    assert out["table"].exists()
    assert not out["chart"].exists()
    assert not out["results"].exists()
    assert "no scored rows yet" in capsys.readouterr().out
