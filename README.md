# battery-oracle

Predicting Mac battery life better than macOS does.

The macOS time-remaining estimate is unstable and often wrong. This repo collects what
the hardware actually reports, alongside the macOS estimate itself, so a model can be
trained and then compared against that baseline honestly.

No model yet. The collector works and is ready to run.

## Everything stays on this machine

Nothing here touches the network. There is no API, no upload and no telemetry. The
`data/` tree is git-ignored: it holds raw `pmset` and `ioreg` output from a personal
machine, and once the roadmap adds process context it will hold application names too.

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

Raw output from two commands, stored verbatim:

| Source | What it gives |
|---|---|
| `pmset -g batt` | percentage, charging state, and the macOS estimate to beat |
| `ioreg -r -c AppleSmartBattery` | capacity, cycles, voltage, amperage, temperature |

Parsed values come in the next step. Keeping the raw text means a field nobody thought
to extract today is still recoverable from samples taken today.

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
