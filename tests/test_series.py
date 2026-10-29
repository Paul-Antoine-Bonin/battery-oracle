"""Tests for the time series table. Synthetic samples built from fixtures."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from battery import series as series_mod
from battery.series import (
    COLUMNS,
    build_series,
    load_series,
    read_csv,
    summarize,
    write_csv,
)
from battery.store import append_sample

FIXTURES = Path(__file__).parent / "fixtures"
START = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)


def ok(name):
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return {"ok": True, "stdout": text, "error": "", "duration_s": 0.0}


def sample(moment, pmset="pmset_discharging.txt", schema=3, context=True):
    commands = {
        "pmset_batt": ok(pmset),
        "ioreg_battery": ok("ioreg_discharging.txt"),
    }
    if context:
        commands.update(
            {
                "cpu_load": ok("sysctl_cpu_load.txt"),
                "wifi_power": ok("networksetup_wifi_on.txt"),
                "display_brightness": ok("ioreg_display_brightness.txt"),
                "foreground_app": ok("lsappinfo_foreground.txt"),
            }
        )
    return {"sampled_at": moment.isoformat(), "schema": schema, "commands": commands}


def test_rows_come_out_sorted_by_instant():
    moments = [START + timedelta(minutes=m) for m in (10, 0, 5)]
    result = build_series(sample(m) for m in moments)

    assert [row["sampled_at"] for row in result.rows] == [
        (START + timedelta(minutes=m)).isoformat() for m in (0, 5, 10)
    ]


def test_sorting_compares_instants_not_strings():
    # 21:30 at +02:00 is 19:30 UTC, so it comes before 20:00 UTC even though
    # the string sorts after it.
    paris = timezone(timedelta(hours=2))
    early = datetime(2026, 8, 16, 21, 30, tzinfo=paris)
    result = build_series([sample(START), sample(early)])

    assert result.rows[0]["sampled_at"] == early.isoformat()
    assert result.rows[1]["gap_s"] == 1800.0


def test_gap_is_seconds_since_the_previous_sample():
    moments = [START, START + timedelta(minutes=5), START + timedelta(hours=2)]
    result = build_series(sample(m) for m in moments)

    assert [row["gap_s"] for row in result.rows] == [None, 300.0, 6900.0]


def test_a_repeated_instant_is_kept_once_and_counted():
    first = sample(START, pmset="pmset_discharging.txt")
    again = sample(START, pmset="pmset_charging.txt")
    result = build_series([first, again, sample(START + timedelta(minutes=5))])

    assert len(result) == 2
    assert result.duplicates == 1
    assert result.rows[0]["macos_state"] == "discharging"
    assert result.rows[1]["gap_s"] == 300.0


def test_every_row_has_every_column_in_order():
    result = build_series([sample(START), sample(START, context=False)])
    for row in result.rows:
        assert list(row) == COLUMNS


def test_a_schema_1_row_keeps_its_schema_and_has_no_context():
    result = build_series([sample(START, schema=1, context=False)])
    row = result.rows[0]

    assert row["schema"] == 1
    assert row["load_1m"] is None
    assert row["wifi_on"] is None
    # The battery half is still there.
    assert row["amperage_ma"] < 0


def test_csv_round_trip_restores_types(tmp_path):
    result = build_series(
        [sample(START), sample(START + timedelta(minutes=5), pmset="pmset_no_estimate.txt")]
    )
    path = write_csv(result.rows, tmp_path / "samples.csv")
    back = read_csv(path)

    assert back == result.rows
    first = back[0]
    assert isinstance(first["amperage_ma"], int)
    assert isinstance(first["charge_fraction"], float)
    assert first["ioreg_ok"] is True
    assert back[1]["macos_estimate_min"] is None
    assert back[1]["macos_has_estimate"] is False


def test_negative_amperage_survives_the_round_trip(tmp_path):
    # The two's complement conversion must not be undone by the table.
    result = build_series([sample(START)])
    back = read_csv(write_csv(result.rows, tmp_path / "samples.csv"))
    assert back[0]["amperage_ma"] == result.rows[0]["amperage_ma"] < 0


def test_write_replaces_the_previous_table(tmp_path):
    path = tmp_path / "samples.csv"
    write_csv(build_series([sample(START)]).rows, path)
    write_csv(
        build_series([sample(START), sample(START + timedelta(minutes=5))]).rows, path
    )

    assert len(read_csv(path)) == 2
    assert not (tmp_path / "samples.csv.partial").exists()


def test_an_empty_series_still_writes_a_header(tmp_path):
    path = write_csv([], tmp_path / "samples.csv")
    assert path.read_text(encoding="utf-8").splitlines() == [",".join(COLUMNS)]
    assert read_csv(path) == []


def test_an_unknown_column_is_read_as_text(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("sampled_at,future_field\n2026-08-16T20:00:00+00:00,12\n")
    assert read_csv(path) == [
        {"sampled_at": "2026-08-16T20:00:00+00:00", "future_field": "12"}
    ]


def test_a_malformed_bool_is_an_error(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("ioreg_ok\nyes\n")
    with pytest.raises(ValueError):
        read_csv(path)


def test_load_series_reads_the_raw_store_and_reports_damage(tmp_path):
    raw = tmp_path / "raw"
    append_sample(sample(START), root=raw)
    append_sample(sample(START + timedelta(minutes=5)), root=raw)
    with (raw / "2026-08-16.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"sampled_at": "2026-08-16T20:10:00+00:00", "comm')

    result = load_series(root=raw)
    assert len(result) == 2
    assert result.skipped_lines == 1


def test_summary_counts_battery_rows_estimates_and_gaps():
    moments = [
        (START, "pmset_discharging.txt"),
        (START + timedelta(minutes=5), "pmset_no_estimate.txt"),
        (START + timedelta(minutes=10), "pmset_charging.txt"),
        (START + timedelta(hours=1), "pmset_charged.txt"),
    ]
    result = build_series(sample(m, pmset=p) for m, p in moments)
    summary = summarize(result)

    assert summary["rows"] == 4
    assert summary["first"] == START.isoformat()
    assert summary["on_battery"] == 2
    # The charging estimate is a time to full, so it does not count.
    assert summary["on_battery_with_estimate"] == 1
    assert summary["gaps"] == 1
    assert summary["longest_gap_s"] == 3000.0


def test_summary_of_nothing():
    summary = summarize(build_series([]))
    assert summary["rows"] == 0
    assert summary["first"] is None
    assert summary["longest_gap_s"] is None


def test_build_command_writes_the_table(tmp_path, capsys):
    raw = tmp_path / "raw"
    out = tmp_path / "series" / "samples.csv"
    append_sample(sample(START), root=raw)
    append_sample(sample(START), root=raw)

    assert series_mod.main(["--root", str(raw), "build", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1 rows" in printed
    assert "dropped 1 repeated instant" in printed
    assert len(read_csv(out)) == 1


def test_summary_command_runs_on_an_empty_store(tmp_path, capsys):
    assert series_mod.main(["--root", str(tmp_path), "summary"]) == 0
    assert "rows" in capsys.readouterr().out
