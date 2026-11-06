# battery-oracle

Predicting Mac battery life better than macOS does.

The macOS time-remaining estimate is unstable and often wrong. This repo collects what
the hardware actually reports, alongside the macOS estimate itself, so a model can be
trained and then compared against that baseline honestly.

The collector works and is ready to run, stored samples parse into one time series
table, and each row carries the macOS estimate as the baseline to beat and the time the
battery actually lasted, reconstructed afterwards. The regression model is written and
tested on synthetic sessions only, as are the scoring against macOS, its chart and the written summary. None
of it has run on real data, so there are no results yet.

## Everything stays on this machine

Nothing here touches the network. There is no API, no upload and no telemetry. The
`data/` tree is git-ignored: it holds raw `pmset` and `ioreg` output from a personal
machine, and the bundle id of whichever application was in front at each sample.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The collector itself has no third-party dependencies, only the standard library. That
is deliberate: it has to keep running for months without a virtualenv drifting
underneath it. Analysis dependencies come later, under the `analysis` extra.

## Collect

```bash
python -m battery.collect once                    # one sample
python -m battery.collect loop --interval 300     # until you stop it
python -m battery.collect status                  # what has been collected
```

Samples append to `data/raw/YYYY-MM-DD.jsonl`, one JSON object per line.

JSON Lines rather than Parquet on purpose. This collector will be killed mid-write
eventually, by a reboot, a lid close, or a battery that genuinely ran out. A truncated
last line costs one sample; a half-written columnar file can cost the whole day. Reads
skip damaged lines and report how many they skipped.

## Run it unattended

```bash
python -m battery.schedule install --interval 300
python -m battery.schedule status
python -m battery.schedule uninstall
```

This installs a launchd agent, not a cron job. cron does not fire while the Mac is
asleep and never catches up, whereas launchd runs a missed job on wake, and the
samples around a sleep are exactly the ones worth having.

The plist is generated at install time and never committed, because it has to carry
absolute paths to this checkout.

The agent writes its output to `data/logs/collect.out` and `data/logs/collect.err`,
git-ignored like the rest of `data/`. If samples stop appearing, look there first.

## What one sample contains

Raw output from six commands, stored verbatim:

| Source | What it gives |
|---|---|
| `pmset -g batt` | percentage, charging state, and the macOS estimate to beat |
| `ioreg -r -c AppleSmartBattery` | capacity, cycles, voltage, amperage, temperature |
| `sysctl -n vm.loadavg hw.logicalcpu` | CPU load average and core count |
| `networksetup -getairportpower en0` | whether Wi-Fi is powered |
| `ioreg -r -k IODisplayParameters` | built-in display brightness |
| `lsappinfo info -only bundleid` | bundle id of the frontmost application |

The first two are required and the collector refuses to start without them. The
last four are context: if one fails, the sample is still kept and the failure is
recorded next to it. Samples carry a `schema` number: `1` has no context half, `2`
adds load, Wi-Fi and brightness, and `3` adds the frontmost application.

Keeping the raw text means a field nobody thought to extract today is still recoverable
from samples taken today.

## Parse

`battery.parse` reads that stored text back into numbers. It runs no command, so it
works the same on a sample from months ago as on a fresh one.

```python
from battery.parse import parse_sample
from battery.store import read_samples

rows = [parse_sample(s) for s in read_samples().samples]
```

Each row has the hardware view (`charge_fraction`, `cycle_count`, `voltage_mv`,
`amperage_ma`, `power_w`, `temperature_c`, ...) and the macOS view, prefixed `macos_`
(`macos_percent`, `macos_state`, `macos_estimate_min`). A command that failed leaves
its half as None, and `ioreg_ok` and `pmset_ok` say which.

The context fields sit alongside: `load_1m`, `load_5m`, `load_15m`, `logical_cpus`,
`load_per_cpu`, `wifi_on`, `brightness_fraction` and `foreground_app`. Each is None
when its command failed or the sample predates it.

- `Amperage` and `InstantAmperage` are converted with two's complement, so negative
  means discharging.
- The charge fraction is computed against the maximum capacity read in the same
  sample. On Apple silicon the mAh figures come from `AppleRawCurrentCapacity` and
  `AppleRawMaxCapacity`, since `MaxCapacity` there is a percentage.
- When macOS shows `(no estimate)`, `macos_estimate_min` stays None. Filling it in
  would give the baseline answers it never gave.
- With the lid closed or only an external display, `brightness_fraction` is None,
  not 0. A missing panel is not a dark one.
- `foreground_app` is a bundle id such as `com.example.editor`, kept as is. Grouping
  applications into categories would need application names in the code, and those
  stay out of the repository. Fixtures use made up ids for the same reason.

## Time series

```bash
python -m battery.series build      # data/raw/*.jsonl -> data/series/samples.csv
python -m battery.series summary    # rows, span, gaps, rows the baseline can be scored on
```

One row per sample, sorted by instant, one column per parsed field. The table is
derived from the raw samples and rebuilt from scratch each time, so a parser fix
reaches every past sample. It lives under `data/` like everything else collected.

- `gap_s` is the seconds since the previous sample. A gap far above the sampling
  interval marks a sleep or a stopped collector.
- `schema` tells a sample that predates a context command from one where it failed.
- Two samples with the same instant are kept once and counted in the build output.
- CSV, read back with `battery.series.read_csv`, which restores ints, floats and
  bools. An empty cell is None.

About 4.1 KiB per sample, so roughly 1.2 MiB a day at one sample every five minutes.

## The macOS baseline

`pmset -g batt` runs in the same sample as `ioreg`, so every row already holds the
macOS estimate next to the hardware readings. `battery.baseline` decides which of
those estimates count as a prediction of the time to empty, and records why the
others do not in `baseline_status`:

| Status | Meaning |
|---|---|
| `scorable` | on battery, discharging, with an estimate |
| `pmset_failed` | no macOS reading in this sample |
| `on_ac` | on AC power, where the figure is a time to full |
| `power_mismatch` | pmset says battery, ioreg sees a charger or a charging current |
| `not_discharging` | on battery but in another state |
| `no_estimate` | macOS showed `(no estimate)` |

Only `scorable` rows get `macos_remaining_min` and `macos_empty_at`, the instant
macOS expected the battery to run out. `power_mismatch` exists because the commands
run one after the other: a charger plugged in or pulled between them leaves two
readings of two different moments, and neither estimator should be scored on that.
`summary` prints `baseline_scorable` and `power_mismatch` alongside the other counts.

## The target

What both estimators are scored against is the time the battery actually lasted from
each row. It can only be known afterwards, so `battery.sessions` and `battery.target`
rebuild it from the series every time the table is built.

First the rows are split into discharge sessions: consecutive samples on battery with
no gap longer than 15 minutes. Each session records how it ended in
`session_end_reason`:

| Reason | Meaning |
|---|---|
| `empty` | charge at or below 3%, then sampling stopped |
| `plugged` | the next row is on a charger, or the charge went up |
| `gap` | sampling stopped with charge left, a sleep or a stopped collector |
| `open` | the data ends while the session is still running |

Then each row in a session gets its target, with `target_status` saying which kind:

| Status | Columns set |
|---|---|
| `exact` | `actual_remaining_min`, `empty_at` and `remaining_at_least_min`, the session ran out |
| `censored` | `remaining_at_least_min` only, the session ended first |
| `none` | nothing, the row is on a charger |

- `empty` needs the gap after the last sample. A Mac plugged in at 2% shows the next
  row on a charger with no gap, so it never ran out and stays censored.
- `empty_at` is the last sample of the session. The battery ran out somewhere before
  the next sample that never came, so `actual_remaining_min` can be short by up to one
  sampling interval.
- A gap ends a session even after a sleep on battery, so a target never covers time
  the collector did not see.
- Censored rows are kept. Full discharges are rare, and leaving out the lower bounds
  would mean learning only from the days the battery ran flat.

`summary` prints `target_exact`, `target_censored` and `sessions_empty`, the number of
full discharges collected so far.

## Model features

`battery.features` turns each discharging row into the inputs the regression model
will be fitted on: the drain rate and the context at that moment.

```python
from battery.features import coverage, features
from battery.series import read_csv

rows = read_csv()
items = features(rows)        # None for rows on a charger
print(coverage(items))        # how often each feature has a value
```

- A feature only looks backwards: the row itself and earlier rows of the same
  discharge session. Never a later row and never a target column, so the model is
  scored on what it could have known live, like macOS.
- `rate_mah_per_h` is the charge lost per hour over the last 30 minutes, measured from
  the charge readings. `draw_ma` is the instant current. Both are kept, along with the
  time to empty each one implies on its own (`trend_remaining_min`,
  `instant_remaining_min`).
- A rate needs at least 10 minutes of history in the session, so the first samples
  after unplugging have none.
- Context: `load_per_cpu`, `wifi_on`, `brightness_fraction`, `temperature_c` and
  `hour_of_day`. The hour is read at the offset stored with the sample, and the
  collector stores UTC, so it is not the local time of day. The foreground application is not a feature yet, since turning bundle
  ids into categories is a choice better made with data in hand.
- Missing values stay None. Filling them is the model's job.

```bash
python -m battery.features coverage   # per feature, how many rows have a value
```

It reads the table written by `battery.series build`. A feature that is almost never
set is not worth fitting on, and this shows it before any model is trained.

## The model

`battery.dataset` pairs each discharging row with its target, and `battery.model` fits a
ridge regression on those pairs. It needs the analysis dependencies:

```bash
pip install -e '.[analysis]'
```

```python
from battery.dataset import examples
from battery.model import cross_predict
from battery.series import read_csv

items = examples(read_csv())
predicted = cross_predict(items)   # one per example, None where it cannot be fitted
```

- A plain linear model on purpose. Full discharges are rare, so there are few
  independent sessions to learn from.
- Missing values get the median of the training rows and a column saying they were
  missing. `hour_of_day` becomes a sine and a cosine. The two remaining time features
  are capped at a day, since they blow up on a nearly idle machine.
- Censored rows count as lower bounds. The first fit uses exact rows alone. Each
  censored row then takes the larger of its prediction and its bound as a target, and
  the model is refitted for up to five rounds.
- `cross_predict` holds out one whole session at a time. Rows of a session are near
  copies of each other, so a random split would flatter the model. A score must come
  from these predictions, never from rows the model was fitted on.
- With only one full discharge collected, its rows get no prediction, because the
  other sessions hold no exact row to fit on. Those entries stay None.

## Scoring against macOS

`battery.evaluate` compares the held out predictions with the macOS estimate:

```bash
python -m battery.evaluate report             # 10 point charge bands
python -m battery.evaluate report --band 20
```

- A row is scored only when it has the real outcome, a held out prediction and a macOS
  estimate. A row missing one of them is dropped for both sides, so both estimators
  answer the same questions.
- Censored rows give no error. They only count how often an estimate fell below what
  the battery really lasted (`model_below_bound`, `macos_below_bound`).
- Rows of a session are near copies, so `scored_sessions` is the real sample size. The
  session means give each session one vote, where the row means favour long sessions.
- The charge table shows where macOS wins, and the chart below draws the same numbers.
- A second table splits the same rows by situation: light, moderate or heavy load per
  CPU, dim or bright display, Wi-Fi on or off. A row missing a reading is left out of
  that split only. The splits overlap, so their rows do not add up to the total.

```bash
python -m battery.chart                       # data/charts/error_by_charge.png
python -m battery.chart --band 20 --out error.png
```

- One pair of bars per charge band, the model and macOS, on one axis in minutes.
- Every band keeps its slot. A band with no scored row stays blank instead of being
  closed up, so the chart shows where the data runs out.
- Each band is labelled with its session count, and the subtitle gives the totals. A
  bar resting on one session says little, whichever estimator it favours.
- With nothing scored the command says so and writes no file.
- The chart lands under `data/`, git-ignored like everything else derived from the
  samples.

```bash
python -m battery.results                     # data/results.md
python -m battery.results --band 20 --out results.md
```

- This writes the summary for this README from the scores, so the sentences follow the
  numbers. It is a draft to read and copy over. Nothing edits the README.
- The headline uses the means over sessions and names whichever estimator came out
  ahead, macOS included. Every band and situation is then listed under where the model
  wins, where macOS wins, or too little data.
- A band or situation under three sessions gets no verdict. Under three sessions in
  total, the summary opens by saying it is too early to call.
- With nothing scored the command says so and writes no file.

### All of it in one command

```bash
python -m battery.analyse                     # table, report, chart and summary
python -m battery.analyse --band 20
```

- Rebuilds the table from `data/raw/`, prints the number of full discharges so far and
  the report, then writes the chart and the summary to their usual places.
- The model is fitted once and all three outputs come from the same held out
  predictions. Running the separate commands fits it once each.
- With nothing scored yet it still writes the table and prints the report, writes no
  chart and no summary, and exits with 1.
- Its test starts from raw command text in the store, with the current in the
  unsigned form `ioreg` prints, and checks that a summary comes out the other end.

The code is tested on made up numbers only. There are no real scores yet.

## Verified on this machine

A four sample run, all fields recovered from the stored text:

| | |
|---|---|
| Charge | 21.7% of current maximum |
| Battery health | 80.4%, 3522 against a design 4381 |
| Cycle count | 728 |
| Draw | -1417 mA at 10828 mV, so 15.3 W |
| Tests | 32, no real `pmset` or `ioreg` calls |

Two things that shape the modelling step. `ioreg` reports `Amperage` unsigned, so a
discharge reads as 18446744073709550199 rather than -1417 and needs a two's complement
conversion. And `MaxCapacity` is not constant: at 80.4% health and still ageing, the
charge fraction has to be computed against the current maximum, never the design
capacity.

## Run the tests

```bash
pip install -e '.[dev]'
pytest
```

The tests run on fixtures under `tests/` and never call `pmset` or `ioreg`, so they
pass on any system. `tests/test_model.py` is skipped unless the `analysis` extra is
installed too.

Roadmap and progress: [TODO.md](TODO.md)
