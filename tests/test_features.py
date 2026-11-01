"""Tests for the model features. Rows are synthetic, built by hand."""

from datetime import datetime, timedelta, timezone

import pytest

from battery.features import (
    FEATURE_COLUMNS,
    coverage,
    features,
    main,
    row_features,
)
from battery.series import COLUMNS, write_csv
from battery.target import target_fields

START = datetime(2026, 8, 16, 20, 0, tzinfo=timezone(timedelta(hours=2)))
MAX_MAH = 3000


def row(minute, mah, on_battery=True, amperage=-1500, **extra):
    out = {
        "sampled_at": (START + timedelta(minutes=minute)).isoformat(),
        "ioreg_ok": True,
        "pmset_ok": True,
        "current_mah": mah,
        "max_mah": MAX_MAH,
        "charge_fraction": mah / MAX_MAH,
        "health_fraction": 0.8,
        "amperage_ma": amperage if on_battery else 900,
        "power_w": (amperage if on_battery else 900) * 11.0 / 1000,
        "external_connected": not on_battery,
        "macos_power_source": "Battery Power" if on_battery else "AC Power",
        "macos_state": "discharging" if on_battery else "charging",
        "macos_percent": round(mah / MAX_MAH * 100),
    }
    out.update(extra)
    return out


def with_target(rows):
    for r, t in zip(rows, target_fields(rows)):
        r.update(t)
    return rows


def drain(count, start_mah=2400, per_step=100, start_minute=0):
    """One sample every 5 minutes, losing `per_step` mAh each time: 1200 mAh/h."""
    return [
        row(start_minute + 5 * k, start_mah - k * per_step) for k in range(count)
    ]


def test_a_charging_row_has_no_features():
    rows = with_target([row(0, 2000, on_battery=False)] + drain(3, start_minute=5))
    assert row_features(rows, 0) is None
    assert row_features(rows, 1) is not None


def test_every_feature_column_is_present():
    rows = with_target(drain(4))
    assert set(row_features(rows, 3)) == set(FEATURE_COLUMNS)


def test_the_rate_comes_from_the_charge_over_the_window():
    rows = with_target(drain(10))
    item = row_features(rows, 9)
    # 30 minute window, so rows 3 to 9: 600 mAh over half an hour.
    assert item["rate_mah_per_h"] == pytest.approx(1200)
    assert item["rate_fraction_per_h"] == pytest.approx(1200 / MAX_MAH)
    assert item["trend_remaining_min"] == pytest.approx(1500 / 1200 * 60)


def test_a_young_session_has_no_rate_yet():
    rows = with_target(drain(4))
    assert row_features(rows, 0)["rate_mah_per_h"] is None
    assert row_features(rows, 1)["rate_mah_per_h"] is None
    assert row_features(rows, 2)["rate_mah_per_h"] == pytest.approx(1200)


def test_the_rate_never_reaches_into_the_previous_session():
    # A first session, a charger, then a second one that drains much slower.
    first = drain(6)
    plug = [row(30, 2000, on_battery=False)]
    second = drain(3, start_mah=2800, per_step=10, start_minute=35)
    rows = with_target(first + plug + second)

    last = row_features(rows, len(rows) - 1)
    assert last["rate_mah_per_h"] == pytest.approx(120)
    assert last["session_elapsed_min"] == pytest.approx(10)


def test_features_do_not_change_when_later_rows_change():
    rows = with_target(drain(8))
    before = row_features(rows, 4)

    altered = [dict(r) for r in rows]
    for r in altered[5:]:
        r["current_mah"] = 10
        r["charge_fraction"] = 10 / MAX_MAH
        r["amperage_ma"] = -9000
    after = row_features(with_target(altered), 4)
    assert before == after


def test_features_never_read_the_target():
    rows = with_target(drain(6))
    before = row_features(rows, 5)
    for r in rows:
        r["actual_remaining_min"] = 12345.0
        r["remaining_at_least_min"] = 12345.0
    assert row_features(rows, 5) == before


def test_draw_and_instant_remaining_are_positive():
    rows = with_target(drain(2))
    item = row_features(rows, 1)
    assert item["draw_ma"] == 1500
    assert item["power_w"] == pytest.approx(16.5)
    assert item["instant_remaining_min"] == pytest.approx(2300 / 1500 * 60)


def test_a_flat_charge_gives_no_trend_estimate():
    rows = with_target([row(5 * k, 2000) for k in range(4)])
    item = row_features(rows, 3)
    assert item["rate_mah_per_h"] == 0
    assert item["trend_remaining_min"] is None


def test_missing_readings_stay_none():
    rows = with_target(drain(4))
    rows[3]["current_mah"] = None
    rows[3]["load_per_cpu"] = None
    item = row_features(rows, 3)
    assert item["current_mah"] is None
    assert item["rate_mah_per_h"] is None
    assert item["instant_remaining_min"] is None
    assert item["load_per_cpu"] is None
    # The fraction based rate still works without the mAh figure.
    assert item["rate_fraction_per_h"] is not None


def test_context_is_carried_over_and_wifi_becomes_a_number():
    rows = with_target(drain(2))
    rows[1].update(
        load_per_cpu=0.4, wifi_on=False, brightness_fraction=0.6, temperature_c=31.5
    )
    item = row_features(rows, 1)
    assert item["load_per_cpu"] == 0.4
    assert item["wifi_on"] == 0.0
    assert item["brightness_fraction"] == 0.6
    assert item["temperature_c"] == 31.5


def test_the_hour_follows_the_stored_offset():
    rows = with_target(drain(1))
    assert row_features(rows, 0)["hour_of_day"] == 20.0


def test_the_foreground_app_is_not_a_feature():
    assert "foreground_app" not in FEATURE_COLUMNS


def test_coverage_counts_values_per_feature():
    rows = with_target([row(0, 2000, on_battery=False)] + drain(4, start_minute=5))
    counts = coverage(features(rows))
    assert counts["rows"] == 4
    assert counts["current_mah"] == 4
    assert counts["rate_mah_per_h"] == 2
    assert counts["load_per_cpu"] == 0



def test_the_coverage_command_reads_the_built_table(tmp_path, capsys):
    rows = with_target([row(0, 2000, on_battery=False)] + drain(4, start_minute=5))
    table = tmp_path / "samples.csv"
    write_csv([{c: r.get(c) for c in COLUMNS} for r in rows], table)

    assert main(["--table", str(table), "coverage"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["rows", "4"]
    assert "rate_mah_per_h" in out and "50%" in out


def test_the_coverage_command_asks_for_a_build_first(tmp_path, capsys):
    assert main(["--table", str(tmp_path / "missing.csv"), "coverage"]) == 1
    assert "battery.series build" in capsys.readouterr().out
