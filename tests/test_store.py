"""Tests for the append-only JSONL store."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from battery.store import append_sample, coverage, day_path, list_days, read_samples

MOMENT = datetime(2026, 8, 16, 20, 0, tzinfo=timezone.utc)


def sample(moment=MOMENT, ok=True):
    return {
        "sampled_at": moment.isoformat(),
        "schema": 1,
        "commands": {"pmset_batt": {"ok": ok, "stdout": "94%; charging", "error": ""}},
    }


def test_day_path_is_one_file_per_utc_day(tmp_path):
    assert day_path(MOMENT, root=tmp_path).name == "2026-08-16.jsonl"


def test_naive_datetimes_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        day_path(datetime(2026, 8, 16, 20, 0), root=tmp_path)


def test_local_time_is_converted_to_utc(tmp_path):
    # 00:30 in Paris on the 17th is still the 16th in UTC.
    paris = timezone(timedelta(hours=2))
    local = datetime(2026, 8, 17, 0, 30, tzinfo=paris)
    assert day_path(local, root=tmp_path).name == "2026-08-16.jsonl"


def test_append_then_read(tmp_path):
    append_sample(sample(), root=tmp_path)
    report = read_samples(root=tmp_path)

    assert report.count == 1
    assert report.samples[0]["commands"]["pmset_batt"]["stdout"] == "94%; charging"


def test_appending_never_overwrites(tmp_path):
    for i in range(3):
        append_sample(sample(MOMENT + timedelta(minutes=5 * i)), root=tmp_path)

    assert read_samples(root=tmp_path).count == 3
    assert len(list_days(root=tmp_path)) == 1


def test_samples_split_across_days(tmp_path):
    append_sample(sample(MOMENT), root=tmp_path)
    append_sample(sample(MOMENT + timedelta(days=1)), root=tmp_path)

    assert [p.stem for p in list_days(root=tmp_path)] == ["2026-08-16", "2026-08-17"]


def test_a_truncated_last_line_costs_one_sample(tmp_path):
    # This is the realistic failure: the Mac dies mid-write.
    append_sample(sample(), root=tmp_path)
    path = day_path(MOMENT, root=tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sampled_at": "2026-08-16T20:05:00+00:00", "comm')

    report = read_samples(root=tmp_path)
    assert report.count == 1
    assert report.skipped_lines == 1


def test_a_sample_after_a_truncated_line_is_kept(tmp_path):
    # The next run appends to the same day file. Without a newline first, the
    # new sample would join the broken fragment and be lost with it.
    append_sample(sample(), root=tmp_path)
    path = day_path(MOMENT, root=tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sampled_at": "2026-08-16T20:05:00+00:00", "comm')

    after = MOMENT + timedelta(minutes=10)
    append_sample(sample(after), root=tmp_path)

    report = read_samples(root=tmp_path)
    assert [s["sampled_at"] for s in report.samples] == [
        MOMENT.isoformat(),
        after.isoformat(),
    ]
    assert report.skipped_lines == 1


def test_a_line_cut_inside_a_character_costs_one_sample(tmp_path):
    good = append_sample(sample(), root=tmp_path)
    cut = json.dumps({**sample(), "note": "é"}, ensure_ascii=False).encode("utf-8")
    with good.open("ab") as handle:
        handle.write(cut[: cut.index("é".encode("utf-8")) + 1])

    report = read_samples(root=tmp_path)
    assert report.count == 1
    assert report.skipped_lines == 1
    assert coverage(root=tmp_path) == [("2026-08-16", 1)]


def test_appending_adds_no_blank_lines(tmp_path):
    append_sample(sample(), root=tmp_path)
    append_sample(sample(MOMENT + timedelta(minutes=5)), root=tmp_path)

    text = day_path(MOMENT, root=tmp_path).read_text(encoding="utf-8")
    assert text.count("\n") == 2
    assert "\n\n" not in text


def test_a_line_without_a_timestamp_is_skipped(tmp_path):
    append_sample(sample(), root=tmp_path)
    path = day_path(MOMENT, root=tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": 1}\n')

    report = read_samples(root=tmp_path)
    assert report.count == 1
    assert report.skipped_lines == 1


@pytest.mark.parametrize(
    "line",
    [
        "[1, 2, 3]",
        "42",
        '"2026-08-16T20:05:00+00:00"',
        '{"sampled_at": 1786219500}',
        '{"sampled_at": null}',
        '{"sampled_at": "yesterday evening"}',
    ],
    ids=["array", "number", "bare string", "numeric stamp", "null stamp", "free text"],
)
def test_valid_json_that_is_not_a_sample_is_skipped(tmp_path, line):
    # These used to raise TypeError out of the read instead of being counted.
    append_sample(sample(), root=tmp_path)
    with day_path(MOMENT, root=tmp_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    report = read_samples(root=tmp_path)
    assert report.count == 1
    assert report.skipped_lines == 1


def test_a_naive_timestamp_is_skipped(tmp_path):
    # Without an offset it cannot be ordered against the other samples, and
    # comparing it with an aware bound or neighbour would raise TypeError.
    append_sample(sample(), root=tmp_path)
    with day_path(MOMENT, root=tmp_path).open("a", encoding="utf-8") as handle:
        handle.write('{"sampled_at": "2026-08-16T20:05:00", "schema": 1}\n')

    report = read_samples(root=tmp_path, start=MOMENT - timedelta(hours=1))
    assert report.count == 1
    assert report.skipped_lines == 1


def test_blank_lines_are_not_counted_as_damage(tmp_path):
    append_sample(sample(), root=tmp_path)
    with day_path(MOMENT, root=tmp_path).open("a", encoding="utf-8") as handle:
        handle.write("\n\n")

    report = read_samples(root=tmp_path)
    assert report.count == 1
    assert report.skipped_lines == 0


def test_date_range_filter(tmp_path):
    for day in range(4):
        append_sample(sample(MOMENT + timedelta(days=day)), root=tmp_path)

    report = read_samples(root=tmp_path, start=MOMENT + timedelta(days=2))
    assert report.count == 2


def test_reading_an_empty_tree(tmp_path):
    report = read_samples(root=tmp_path)
    assert report.count == 0
    assert report.files == 0


def test_coverage_counts_samples_per_day(tmp_path):
    append_sample(sample(MOMENT), root=tmp_path)
    append_sample(sample(MOMENT + timedelta(minutes=5)), root=tmp_path)
    append_sample(sample(MOMENT + timedelta(days=1)), root=tmp_path)

    assert coverage(root=tmp_path) == [("2026-08-16", 2), ("2026-08-17", 1)]


def test_coverage_leaves_out_damaged_lines(tmp_path):
    path = append_sample(sample(), root=tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sampled_at": "2026-08\n')
        handle.write('["not", "a", "sample"]\n')

    assert coverage(root=tmp_path) == [("2026-08-16", 1)]
    assert sum(n for _, n in coverage(root=tmp_path)) == read_samples(root=tmp_path).count


def test_coverage_on_an_empty_tree(tmp_path):
    assert coverage(root=tmp_path) == []


def test_a_naive_sample_is_refused_and_nothing_is_written(tmp_path):
    with pytest.raises(ValueError):
        append_sample(sample(datetime(2026, 8, 16, 20, 0)), root=tmp_path)
    assert list_days(tmp_path) == []
