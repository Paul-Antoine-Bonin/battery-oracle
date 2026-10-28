# battery-oracle

Predicting Mac battery life better than macOS does.

The macOS time-remaining estimate is unstable and often wrong. This repo collects what
the hardware actually reports, alongside the macOS estimate itself, so a model can be
trained and then compared against that baseline honestly.

No model yet. The collector works and is ready to run, and stored samples parse into
numbers.

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

About 4.1 KiB per sample, so roughly 1.2 MiB a day at one sample every five minutes.

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
pytest
```

Roadmap and progress: [TODO.md](TODO.md)
