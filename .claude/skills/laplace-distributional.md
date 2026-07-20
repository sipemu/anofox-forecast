---
name: laplace-distributional
description: How to use LaplaceForecaster and its opt-in extensions (MultiScaleLaplace, scoring knobs, GPD tails, parade PIT) for streaming distributional forecasting
user_invocable: true
---

# Distributional forecasting with `LaplaceForecaster`

Streaming per-observation ensemble that emits a `Vec<GaussianMixture>` per forecast. Ported from and interoperable with [microprediction/skaters](https://github.com/microprediction/skaters).

**Picking parameters?** For rule-based decision trees keyed on your data
type (continuous / count / intermittent / tick-grid / short-history),
see `docs/LAPLACE_PARAMETER_GUIDE.md`. That guide is the source of
truth for *which* builder chain to use; this skill covers *how* to use
the resulting API.

Requires the `distributional` cargo feature:

```toml
anofox-forecast = { version = "0.15.4", features = ["distributional"] }
```

## Quick start

```rust
use anofox_forecast::models::laplace::{LaplaceForecaster, DistributionalForecaster};
use anofox_forecast::models::Forecaster;

let mut f = LaplaceForecaster::new().auto();
f.fit(&ts)?;

// Point forecast
let fc = f.predict(12)?;
let means = fc.primary();

// Distributional forecast
let dists = f.forecast_dist(12)?;       // Vec<GaussianMixture>, len == 12
for (h, mix) in dists.iter().enumerate() {
    println!("h={}: μ={:.2}, σ={:.2}", h + 1, mix.mean(), mix.std());
    let p05 = mix.quantile(0.05);
    let p95 = mix.quantile(0.95);
}
```

## Three zero-config selectors

Which one to pick depends on the data type. See the module docs at `src/models/laplace/mod.rs` for full rules of thumb.

```rust
LaplaceForecaster::new().auto()        // Continuous / economic series (M-competition, FRED)
LaplaceForecaster::new().auto_aid()    // Retail SKU / demand counts (with `postprocess` feature)
LaplaceForecaster::new().skaters()     // Full skaters-parity pool (30+ leaves, sticky lattice,
                                       //   CRPS terminal, fast_slow, larger compute)
```

## Committing to a seasonal period

Two builders that both add a seasonal leaf; behaviour differs on batch init:

```rust
// User-committed period. Since v0.15.2, also defaults `with_seasonal_batch_init()` on.
// Recommended when you know the period.
LaplaceForecaster::new().auto().with_seasonal(12);

// Auto-detect fallback period. Does NOT default batch init (measured fev-27 regression).
// The .auto() / .skaters() harnesses call this internally.
LaplaceForecaster::new().skaters().auto_with_seasonal_period(12);

// Opt out of batch init if the amplitude is growing or the phase is shifting
LaplaceForecaster::new().auto().with_seasonal(12).no_seasonal_batch_init();

// Multiple seasonal periods (e.g. weekly + annual for daily data)
LaplaceForecaster::new().auto().with_seasonal_multi(&[7, 365]);
```

## v0.15.3 scoring knobs (opt-in)

These target the multi-step objective mismatch. Substantial wins on fev-27
(~−5 % MASE, −4 % WQL on `.auto()`). Compose freely.

```rust
// Match softmax scoring depth to your forecast horizon
LaplaceForecaster::new().auto().with_scoring_horizon(12);

// Sliding-window log-lik (last w observations) instead of cumulative
LaplaceForecaster::new().auto().with_scoring_window(14);

// Both compound — this is the v0.15.3 recommended recipe:
LaplaceForecaster::new()
    .auto()
    .with_seasonal(12)
    .with_scoring_horizon(12)
    .with_scoring_window(14);
```

## v0.15.4 MultiScaleLaplace — best-known fev-27 config

Wraps `.skaters()` at multiple decimation strides ({1, period, k}). Each scale
contributes its coarse-step prediction weighted by softmax over per-scale
training log-lik. **Biggest single-change improvement in project history:
−11.3 % MASE on leaderboard-comparable fev-27 subset.**

```rust
use anofox_forecast::models::laplace::{MultiScaleLaplace, DistributionalForecaster};

let H = 24;                                  // your forecast horizon
let mut f = MultiScaleLaplace::skaters(H)
    .with_scoring_horizon()                  // pass scH to each scale's sub
    .with_scoring_window(14);                // pass scW=14 to each scale's sub
if period >= 2 {
    f = f.with_period(period);               // period-aligned decimation stride
}
f.fit(&ts)?;
let dists = f.forecast_dist(H)?;
```

Compute cost: 2-3× fit time vs plain `.skaters()` on longer panels (each
activated scale runs a full skaters pool). Sub-forecasters at scale s are
dropped when `N_train / s < 50` (avoids fitting a 30-leaf pool on 29 obs).

## v0.15.4 parade + GPD tails — extreme-quantile calibration (opt-in)

Only relevant for **extreme quantiles (q < 0.02 or q > 0.98)** — anomaly
detection, VaR estimation, rare-event forecasting. **Measured neutral on
fev-27 WQL** which uses q ∈ [0.1, 0.9] (metric-shape mismatch).

```rust
use anofox_forecast::models::laplace::{LaplaceForecaster, GpdTailsForecaster};
use anofox_forecast::models::Forecaster;

// Per-horizon parade PIT tracking during fit + GPD splice on top
let mut base = LaplaceForecaster::new().skaters().with_parade(H);
let mut f = GpdTailsForecaster::new(base);
f.fit(&ts)?;

// Standard forecast_dist gives the body predictive
let dists = f.forecast_dist(H)?;

// For extreme quantiles use the spliced version
let p99 = f.quantile_spliced(&dists, /*h=*/ 1, 0.99);
```

`with_parade(k)` alone (no GPD) is standalone useful for callers wanting
per-horizon PIT diagnostics:

```rust
let mut f = LaplaceForecaster::new().skaters().with_parade(24);
f.fit(&ts)?;
let pit_by_h = f.parade_pit();               // Option<&[Vec<f64>]>
// pit_by_h[h-1] is roughly Uniform(0, 1) when the h-step predictive is calibrated
```

Fit-time cost: ~17× on m4-hourly-scale panels. Storage: `O(N × k × 8 bytes)`.

## Additional leaf-pool builders (opt-in)

```rust
LaplaceForecaster::new()
    .with_holt(0.1, 0.02)                    // Holt's linear trend leaf
    .with_ar2(0.1)                           // AR(2) leaf
    .with_theta_alphas(&[0.05, 0.1, 0.3])    // Theta family
    .with_ou(0.1)                            // Ornstein-Uhlenbeck mean-reverting
    .with_frac_diff(0.4)                     // Fractional differencing
    .with_yeo_johnson_grid(&[0.0, 0.5, 1.0]) // Coordinate priors
    .with_stl(12)                            // STL decomposition leaf (NOT auto-enabled)
    .with_multi_h_scoring()                  // Add h-step LL contributions
    .with_calibration()                      // Quantile-matched calibration factor
    .with_per_horizon_calibration(24);       // Per-h calibration
```

## Inspecting a fitted model

```rust
use anofox_forecast::models::inspect::{Explanation, Inspectable};

if let Ok(Explanation::Laplace(ex)) = Inspectable::explanation(&f) {
    for (name, weight) in ex.leaf_names.iter().zip(ex.leaf_weights.iter()) {
        if *weight > 0.01 {
            println!("  {name}: {weight:.3}");
        }
    }
}
```

## SOTA position (fev-27, 23-set leaderboard-comparable subset)

| Config | geomean MASE |
|---|---:|
| `LaplaceForecaster::new().auto()` | 1.6457 |
| `.auto().with_seasonal(p).with_scoring_horizon(H).with_scoring_window(14)` | 1.5153 |
| **`MultiScaleLaplace::skaters(H).with_scoring_horizon().with_scoring_window(14)`** | **1.4602** |

Places this crate at ~rank 7-8 on the SOTA classical panel — competitive with
Nixtla `auto_ets` (1.440), ahead of our own `AutoETS` (1.525) and `Seasonal
Naive` (1.665). Full progression + comparisons in `docs/SOTA_POSITIONING.md`.

## When NOT to use LaplaceForecaster

- **Short-history panels (N < 100)** — streaming leaves need warmup. Use
  `AutoTheta` / `AutoETS` from `crate::models::theta` / `crate::models::exponential`.
- **You only need a point forecast** — `AutoTheta` / `AutoETS` are simpler.
- **You need extreme-quantile calibration for anomaly detection** on a small
  dataset — see the module note about mismatch with fev-style WQL.
