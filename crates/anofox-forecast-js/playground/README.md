# Laplace playground (WebAssembly demo)

**Live demo (once GitHub Pages is enabled): https://sipemu.github.io/anofox-forecast/**

Interactive browser demo of the streaming Laplace forecaster from
`anofox-forecast`. Mirrors the shape / interaction of
[skaters' playground](https://skaters.microprediction.org/demos/playground.html):
one observation is added per tick, the forecast fan is re-fit every
step, and the router picks the recipe based on the observed data shape.

## Deployment (GitHub Pages)

Automatic via `.github/workflows/deploy-playground.yml` — runs on
pushes to `main` that touch the playground sources or the Laplace
crate, plus a manual "Run workflow" button in the Actions tab.

**One-time repo setup required.** In GitHub:

  Settings → Pages → Build and deployment → Source = **"GitHub Actions"**

Without this the workflow will error at the deploy step. After
enabling, the next push (or a manual trigger) will publish the site
at `https://sipemu.github.io/anofox-forecast/`.

## Contents

- `index.html` — self-contained page: archetype dropdown, horizon /
  speed / warm-up sliders, play / pause / step / regenerate controls,
  canvas chart (observations + median forecast + 10-90 % + 25-75 %
  quantile bands), and a recipe-picked tag.
- `pkg/` — WASM build output from `wasm-pack build --target web`.

## Build

```sh
# From `crates/anofox-forecast-js/` (this crate's root):
wasm-pack build --release --target web --out-dir playground/pkg
```

Requires:
- Rust toolchain with `wasm32-unknown-unknown` target
  (`rustup target add wasm32-unknown-unknown`).
- `wasm-pack` on PATH.

## Run locally

WASM has to be served over HTTP — file:// won't work due to CORS on
the ES-module import.

```sh
cd crates/anofox-forecast-js/playground
python3 -m http.server 8000
# then open http://localhost:8000/
```

Any static-file server works (`npx serve`, `caddy file-server`, etc).

## Tabs

- **Forecast** — the classic streaming demo: pick an archetype, watch the
  1..H-step predictive fan re-fit as new observations arrive.
- **Anomaly detection** — same streaming loop, but the harness injects
  random-sign spikes at a user-controlled rate and the Laplace 1-step
  predictive mixture scores each incoming observation. Observations
  whose surprise `-log p(y | mixture)` exceeds the threshold are
  flagged (red X); the ground-truth injected obs are shown as red
  rings. Metrics: precision, recall, TP, FP.

The threshold slider is interactive — re-scores the entire history
against the new cut without re-fitting, so you can find the
precision/recall trade-off by dragging.

## Archetypes

Same generators as `examples/synthetic_bakeoff.rs`:

- `pure_gaussian_noise` — sanity check, flat noise.
- `seasonal_linear_trend` — sine + trend + noise (period 12).
- `random_walk` — non-parametric path-dependent.
- `mean_reverting_ou` — Ornstein-Uhlenbeck around 50.
- `intermittent_bursty` — 80 % zeros + Poisson(5) bursts.
- `heavy_tail_cauchy` — Cauchy innovations, wild spikes.
- `level_shift_midway` — flat 50 → jump → flat 55.
- `multi_seasonal_hourly` — daily + weekly cycle.
- `all_zeros_rare_spikes` — 99 % zeros + rare Poisson(10) bursts.

## What the recipe tag means

`recommended_for` picks one of five branches on cheap data-shape checks:

- `short_history` — N < 60, Laplace fallback (classical is usually better).
- `retail_count_aid` — integer-valued with heavy zeros → AID-selected count leaf.
- `heavy_tailed_crps` — kurtosis-on-differences or max-z detects fat tails.
- `continuous_multiscale_3sh` — seasonal + long enough → the fev-27 SOTA recipe.
- `continuous_plain_skaters` — everything else, plain skaters + tuned knobs.

Change archetype and watch the tag update. Some archetypes will
re-route mid-stream as more observations reveal the shape.
