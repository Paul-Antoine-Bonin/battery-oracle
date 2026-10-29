# battery-oracle: predicting Mac battery life better than macOS does

## Goal
macOS gives an unstable and often wrong battery estimate. Build a model that learns from
your real usage and predicts better, then prove it.

## Data
- Local and personal: `pmset -g log`, `ioreg -l` for battery state, plus sampling of CPU
  load and the active applications.
- No data leaves the machine. Say so in the README.

## Steps
- [x] Set up repo, collection script run periodically
- [x] Extract: charge level, cycles, voltage, amperage, temperature if available
- [x] Add the context: brightness, foreground applications, CPU load, Wi-Fi
- [x] Store as a timestamped time series, one row per sample
- [ ] Record the macOS estimate at the same moment too, to have a baseline to beat
- [ ] After a few weeks: build the target, the time actually remaining until discharge,
      reconstructed after the fact
- [ ] Regression model on the current discharge rate and the context
- [ ] Compare the mean absolute error of your model against the macOS one
- [ ] Chart: error of both estimators as a function of charge level
- [ ] README: by how much you beat macOS, and in which situations

## Done when
Several weeks of collection, and an honest quantified comparison, including where macOS
wins.

## Traps
- Full discharge cycles are rare, so the target is the real bottleneck. Start collecting
  early and never interrupt it.
- Commit nothing containing personal file or application names. `data/` is git-ignored
  for that reason, and it holds the collector logs too.
- `ioreg` reports `Amperage` as an unsigned 64 bit integer. While discharging it comes
  back as 18446744073709550199, not -1417. Convert with two's complement, subtracting
  2^64 above 2^63, or every discharge sample will look like an enormous charge.
  `InstantAmperage` has the same problem.
- `MaxCapacity` is not constant. This battery is at 3522 against a design 4381, so 80.4%
  health after 728 cycles, and it keeps ageing during the study. Re-read it every
  sample and compute the charge fraction against the current maximum, never against
  the design figure.
- Use launchd, not cron. cron does not fire while the Mac is asleep and never catches
  up, and the samples around a sleep are the interesting ones.
- Storage is raw command text, not parsed fields, so a value nobody thought to extract
  today is still recoverable from today's samples. Cost is about 4.1 KiB per sample,
  roughly 1.2 MiB a day at one sample every five minutes.
