"""Tests for the error chart. Every number is made up to check the drawing."""

from datetime import datetime, timedelta, timezone

import pytest

from battery.chart import band_label, chart_bands, main, subtitle, write_chart
from battery.evaluate import Scored, by_charge, compare
from battery.series import write_csv
from battery.target import target_fields


def scored(session, actual, model, macos, charge=0.5):
    return Scored(0, session, charge, actual, model, macos)


def test_every_band_of_the_range_is_present_and_empty_ones_stay_empty():
    out = chart_bands([scored(1, 100.0, 110.0, 130.0, charge=0.45)], band_pct=25)

    assert [(b["low_pct"], b["high_pct"], b["rows"]) for b in out] == [
        (0, 25, 0),
        (25, 50, 1),
        (50, 75, 0),
        (75, 100, 0),
    ]
    assert out[0]["model_mae"] is None
    assert out[0]["macos_mae"] is None


def test_a_filled_band_carries_the_same_numbers_as_the_report():
    items = [
        scored(1, 100.0, 110.0, 130.0, charge=0.42),
        scored(2, 100.0, 130.0, 110.0, charge=0.87),
    ]

    out = chart_bands(items)

    assert [b for b in out if b["rows"]] == by_charge(items)


def test_bands_that_do_not_divide_100_end_at_100():
    out = chart_bands([], band_pct=30)

    assert [(b["low_pct"], b["high_pct"]) for b in out] == [
        (0, 30),
        (30, 60),
        (60, 90),
        (90, 100),
    ]


def test_a_full_battery_lands_on_the_last_band_of_the_axis():
    out = chart_bands([scored(1, 100.0, 100.0, 100.0, charge=1.0)])

    assert len(out) == 10
    assert out[-1]["rows"] == 1


def test_a_band_label_shows_the_range_and_the_sessions():
    assert band_label({"low_pct": 40, "high_pct": 50, "sessions": 1}) == (
        "40-50%\n1 session"
    )
    assert band_label({"low_pct": 0, "high_pct": 10, "sessions": 0}) == (
        "0-10%\n0 sessions"
    )


def test_the_subtitle_gives_the_sample_size():
    text = subtitle({"rows": 12, "sessions": 3})

    assert "12 scored rows" in text
    assert "3 discharge sessions" in text


def test_nothing_scored_writes_nothing(tmp_path):
    path = tmp_path / "chart.png"

    assert write_chart([], path) is None
    assert not path.exists()


def test_the_bars_are_the_band_errors_and_empty_bands_have_none():
    pytest.importorskip("matplotlib")
    from battery.chart import build_figure

    items = [
        scored(1, 100.0, 110.0, 130.0, charge=0.15),
        scored(2, 100.0, 140.0, 105.0, charge=0.75),
    ]

    figure = build_figure(chart_bands(items, 25), compare(items))
    axes = figure.axes[0]

    model, macos = axes.containers
    assert [bar.get_height() for bar in model] == [10.0, 40.0]
    assert [bar.get_height() for bar in macos] == [30.0, 5.0]
    assert [bar.get_x() < 1 for bar in model] == [True, False]
    labels = [label.get_text() for label in axes.get_xticklabels()]
    assert labels[0] == "0-25%\n1 session"
    assert labels[1] == "25-50%\n0 sessions"
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Model",
        "macOS",
    ]


def test_write_chart_makes_the_folder_and_a_png(tmp_path):
    pytest.importorskip("matplotlib")

    path = tmp_path / "charts" / "chart.png"

    assert write_chart([scored(1, 100.0, 110.0, 130.0)], path) == path
    assert path.read_bytes().startswith(b"\x89PNG")


def test_the_chart_command_runs_from_a_written_table(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")

    start = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)

    def row(minute, mah, on_battery=True):
        return {
            "sampled_at": (start + timedelta(minutes=minute)).isoformat(),
            "ioreg_ok": True,
            "pmset_ok": True,
            "current_mah": mah,
            "max_mah": 3000,
            "charge_fraction": mah / 3000,
            "amperage_ma": -1500 if on_battery else 900,
            "external_connected": not on_battery,
            "macos_power_source": "Battery Power" if on_battery else "AC Power",
            "macos_state": "discharging" if on_battery else "charging",
            "macos_percent": round(mah / 30),
            "macos_remaining_min": 60 if on_battery else None,
        }

    def discharge(first_minute):
        """500 mAh down to 50 mAh, then a charger an hour later."""
        return [row(first_minute + 5 * k, 500 - 50 * k) for k in range(10)] + [
            row(first_minute + 105, 300, on_battery=False)
        ]

    rows = discharge(0) + discharge(300)
    for r, t in zip(rows, target_fields(rows)):
        r.update(t)
    table = write_csv(rows, tmp_path / "samples.csv")
    out = tmp_path / "chart.png"

    assert main(["--table", str(table), "--out", str(out), "--band", "20"]) == 0

    assert out.exists()
    assert "wrote" in capsys.readouterr().out


def test_the_chart_command_says_when_there_is_nothing_to_draw(tmp_path, capsys):
    pytest.importorskip("sklearn")

    table = write_csv([{"sampled_at": "2026-08-16T20:00:00+00:00"}], tmp_path / "s.csv")
    out = tmp_path / "chart.png"

    assert main(["--table", str(table), "--out", str(out)]) == 1

    assert not out.exists()
    assert "no scored rows" in capsys.readouterr().out


def test_the_chart_command_says_when_there_is_no_table(tmp_path, capsys):
    assert main(["--table", str(tmp_path / "missing.csv")]) == 1
    assert "no table" in capsys.readouterr().out
