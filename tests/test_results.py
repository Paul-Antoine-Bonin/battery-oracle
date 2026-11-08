"""Tests for the written summary. Every number is made up to check the sentences."""

from datetime import datetime, timedelta, timezone

import pytest

from battery.evaluate import Scored, compare
from battery.results import (
    MACOS,
    MODEL,
    TIE,
    headline,
    main,
    results_lines,
    trend_lines,
    winner,
    write_results,
)
from battery.series import write_csv
from battery.target import target_fields


def scored(session, model, macos, charge=0.5, index=0):
    return Scored(index, session, charge, 100.0, 100.0 + model, 100.0 + macos)


def sessions(count, model, macos, charge=0.5, index=0):
    return [scored(s + 1, model, macos, charge, index) for s in range(count)]


def section(lines, title):
    start = lines.index(f"### {title}") + 2
    out = []
    for line in lines[start:]:
        if not line:
            break
        out.append(line)
    return out


def test_the_lower_error_wins_and_a_gap_too_small_to_print_is_a_tie():
    assert winner(10.0, 20.0) == MODEL
    assert winner(20.0, 10.0) == MACOS
    assert winner(10.01, 10.04) == TIE


def test_the_headline_says_how_far_the_model_is_ahead():
    assert headline(compare(sessions(3, 10.0, 20.0))) == (
        "The model beat macOS, with a mean error of 10.0 minutes for the model "
        "and 20.0 for macOS per session, 50% lower."
    )


def test_the_headline_says_so_when_macos_wins():
    assert headline(compare(sessions(3, 40.0, 30.0))).startswith(
        "macOS beat the model,"
    )
    assert headline(compare(sessions(3, 40.0, 30.0))).endswith("25% lower.")


def test_the_headline_reports_a_tie():
    assert headline(compare(sessions(3, 10.0, 10.0))).startswith(
        "The two estimators tied,"
    )


def test_one_long_session_cannot_decide_the_headline():
    long_good_session = [scored(1, 1.0, 50.0)] * 100
    two_bad_sessions = [scored(2, 50.0, 1.0), scored(3, 50.0, 1.0)]

    out = compare(long_good_session + two_bad_sessions)

    assert out["model_mae"] < out["macos_mae"]
    assert headline(out).startswith("macOS beat the model,")


def test_each_band_lands_under_the_side_that_won_it():
    items = sessions(3, 10.0, 20.0, charge=0.15) + sessions(3, 40.0, 30.0, charge=0.75)

    lines = results_lines([{}], items)

    assert section(lines, "Where the model wins") == [
        "- 10-20% charge: 10.0 against 20.0 minutes for macOS, 3 sessions"
    ]
    assert section(lines, "Where macOS wins") == [
        "- 70-80% charge: 40.0 against 30.0 minutes for macOS, 3 sessions"
    ]


def test_situations_are_listed_next_to_the_bands():
    lines = results_lines([{"wifi_on": True}], sessions(3, 40.0, 30.0))

    assert section(lines, "Where macOS wins") == [
        "- 50-60% charge: 40.0 against 30.0 minutes for macOS, 3 sessions",
        "- wifi on: 40.0 against 30.0 minutes for macOS, 3 sessions",
    ]


def test_macos_wins_get_a_section_even_when_empty():
    lines = results_lines([{}], sessions(3, 10.0, 20.0))

    assert section(lines, "Where macOS wins") == ["None."]


def test_a_split_on_too_few_sessions_gets_no_verdict():
    items = sessions(3, 10.0, 20.0, charge=0.15) + [scored(9, 1.0, 50.0, charge=0.75)]

    lines = results_lines([{}], items)

    assert section(lines, "Too little data to say") == [
        "- 70-80% charge: 1 of 3 sessions needed"
    ]
    assert all("70-80%" not in line for line in section(lines, "Where the model wins"))


def test_too_few_sessions_overall_open_with_a_warning():
    lines = results_lines([{}], sessions(2, 10.0, 20.0))

    assert lines[2] == (
        "Too early to call: 2 sessions scored, and a verdict needs 3. "
        "Provisional figures follow."
    )


def test_enough_sessions_carry_no_warning():
    lines = results_lines([{}], sessions(3, 10.0, 20.0))

    assert not any(line.startswith("Too early") for line in lines)


def test_the_sessions_each_side_won_are_counted():
    items = sessions(1, 10.0, 20.0) + [scored(2, 40.0, 30.0), scored(3, 40.0, 30.0)]

    text = "\n".join(results_lines([{}], items))

    assert (
        "Scored on 3 rows from 3 full discharges, with predictions held out by "
        "session. The model did better in 1 session and macOS in 2." in text
    )


def trend(rows, sessions, model, plain):
    return {"rows": rows, "sessions": sessions, "model_mae": model, "trend_mae": plain}


def test_the_summary_says_when_the_model_beats_the_trend():
    assert trend_lines(trend(30, 3, 10.0, 25.0)) == [
        "The model beat the trend extrapolation, at 10.0 minutes for the model "
        "and 25.0 for the trend, mean over 30 rows from 3 sessions."
    ]


def test_the_summary_says_plainly_when_the_trend_beats_the_model():
    (line,) = trend_lines(trend(30, 3, 25.0, 10.0))

    assert line.startswith("The trend extrapolation beat the model,")
    assert "the fit adds nothing" in line


def test_the_trend_check_reports_a_tie():
    (line,) = trend_lines(trend(30, 3, 10.01, 10.04))

    assert line.startswith("The model and the trend extrapolation tied,")


def test_the_trend_check_gives_no_verdict_on_too_few_sessions():
    (line,) = trend_lines(trend(1, 1, 10.0, 25.0))

    assert line.startswith("Too little data to say: 1 of 3 sessions needed.")
    assert "1 row from 1 session." in line


def test_the_trend_check_says_when_no_row_had_a_trend():
    assert trend_lines(trend(0, 0, None, None)) == [
        "No scored row had enough history for a trend extrapolation."
    ]


def test_the_trend_section_comes_last_and_only_when_asked_for():
    items = sessions(3, 10.0, 20.0)

    without = results_lines([{}], items)
    with_trend = results_lines([{}], items, trend=trend(3, 3, 10.0, 25.0))

    assert "### Against a plain trend" not in without
    assert with_trend[: len(without)] == without
    assert section(with_trend, "Against a plain trend") == trend_lines(
        trend(3, 3, 10.0, 25.0)
    )
    assert with_trend[-1].startswith("The model beat the trend extrapolation,")


def test_nothing_scored_is_refused():
    with pytest.raises(ValueError):
        results_lines([], [])


def test_nothing_scored_writes_no_file(tmp_path):
    out = tmp_path / "results.md"

    assert write_results([], [], out) is None
    assert not out.exists()


def test_the_file_ends_with_one_newline_and_its_folder_is_created(tmp_path):
    out = tmp_path / "new" / "results.md"

    assert write_results([{}], sessions(3, 10.0, 20.0), out) == out
    text = out.read_text(encoding="utf-8")
    assert text.startswith("## Results\n")
    assert text.endswith("None.\n")


def test_the_results_command_runs_from_a_written_table(tmp_path, capsys):
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
    out = tmp_path / "results.md"

    assert main(["--table", str(table), "--out", str(out), "--band", "20"]) == 0

    text = out.read_text(encoding="utf-8")
    assert "Too early to call: 2 sessions scored" in text
    assert "### Against a plain trend" in text
    assert "wrote" in capsys.readouterr().out


def test_the_results_command_says_when_there_is_nothing_to_write(tmp_path, capsys):
    pytest.importorskip("sklearn")

    table = write_csv([{"sampled_at": "2026-08-16T20:00:00+00:00"}], tmp_path / "s.csv")
    out = tmp_path / "results.md"

    assert main(["--table", str(table), "--out", str(out)]) == 1

    assert not out.exists()
    assert "no scored rows" in capsys.readouterr().out


def test_the_results_command_says_when_there_is_no_table(tmp_path, capsys):
    assert main(["--table", str(tmp_path / "missing.csv")]) == 1
    assert "no table" in capsys.readouterr().out
