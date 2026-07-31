# battery-oracle: predicting Mac battery life better than macOS does

## Goal
macOS gives an unstable and often wrong battery estimate. Build a model that learns from
your real usage and predicts better, then prove it.

## Data
- Local and personal: `pmset -g log`, `ioreg -l` for battery state, plus sampling of CPU
  load and the active applications.
- No data leaves the machine. Say so in the README.

## Steps
- [ ] Set up repo, collection script run periodically
- [ ] Extract: charge level, cycles, voltage, amperage, temperature if available
- [ ] Add the context: brightness, foreground applications, CPU load, Wi-Fi
- [ ] Store as a timestamped time series, one row per sample
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
- Commit nothing containing personal file or application names.
