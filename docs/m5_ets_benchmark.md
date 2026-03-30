# AutoETS Model Pool Benchmark: Full M5 Dataset

Comparison of AutoETS **Complete** (19 models) vs **Reduced** (8 models) pool
on the full M5 retail forecasting dataset (30,490 series), based on Petropoulos et al. (2023)
"Wielding Occam's razor: Fast and frugal retail forecasting" ([arXiv:2102.13209](https://arxiv.org/abs/2102.13209)).

## Setup

| Parameter | Value |
|---|---|
| Dataset | M5 (all 30,490 item-store series, daily Walmart sales) |
| Series evaluated | 13,766 (filtered: >30% nonzero days) |
| Training window | 1,941 days (2011-01-29 to 2016-05-22) |
| Test horizon | 28 days (4 weeks, matching M5 competition) |
| Seasonal period | 7 (weekly) |
| Selection criterion | AICc |
| Platform | Linux, Rust release build, rayon parallel (multi-core) |

## Results

### Accuracy

| Metric | Complete (19 models) | Reduced (8 models) | Difference |
|---|---|---|---|
| Avg RMSE | 2.1998 | 2.1999 | +0.0001 (+0.005%) |
| Median RMSE | 1.4569 | 1.4571 | +0.0002 (+0.01%) |
| Avg MAPE | 56.83% | 56.80% | -0.03pp |
| Avg sMAPE | 111.03% | 111.01% | -0.02pp |
| Success rate | 13,766/13,766 (100%) | 13,766/13,766 (100%) |

The Reduced pool achieves **virtually identical accuracy** to the Complete pool
across all 13,766 evaluated series.

### Speed

| Metric | Complete | Reduced | Speedup |
|---|---|---|---|
| Wall-clock time | 206.6 s | 119.7 s | **1.73x** |
| Avg CPU time per series | 311.6 ms | 175.8 ms | **1.77x** |
| Total CPU time | 4,290 s | 2,420 s | **1.77x** |
| Throughput | 67 series/s | 115 series/s | **1.72x** |

The Reduced pool is **1.7x faster** at scale. With 13,766 series, this saves
**87 seconds** wall-clock time (207s vs 120s) on a multi-core machine.

### Model Selection Distribution

| Model | Complete | Reduced | Notes |
|---|---|---|---|
| ETS(A,N,A) | 12,215 (88.7%) | 12,355 (89.8%) | Seasonal, no trend |
| ETS(A,Ad,A) | 669 (4.9%) | 696 (5.1%) | Seasonal, damped trend |
| ETS(A,N,N) | 550 (4.0%) | 676 (4.9%) | Simple exponential smoothing |
| ETS(A,A,A) | 167 (1.2%) | — | Undamped (Complete only) |
| ETS(A,A,N) | 160 (1.2%) | — | Undamped (Complete only) |
| ETS(A,Ad,N) | 5 (0.0%) | 39 (0.3%) | Damped trend, no season |

Key observations:
- **89%** of M5 series select ETS(A,N,A) — additive seasonal without trend
- The 327 series using undamped trend (A,A,A / A,A,N) in Complete switch to
  damped variants or simpler models in Reduced — more robust for retail
- All dominant models (A,N,A, A,Ad,A, A,N,N) are in both pools
- Multiplicative error models are never selected (M5 data contains zeros)

### Reduced Pool Models

The 8 models in the Reduced pool (Petropoulos et al., 2023):

| Category | Additive Error | Multiplicative Error |
|---|---|---|
| Level only | ANN | MNN |
| Trend only (damped) | AAdN | MAdN |
| Seasonal only | ANA | MNM |
| Trend + Seasonal | AAdA | MAdM |

Design principles:
1. **Damped trends only** — undamped trends produce positively biased long-horizon forecasts
2. **Matched error/seasonal** — additive error pairs with additive seasonal, multiplicative with multiplicative
3. **Balanced coverage** — 2 models per forecast profile (level, trend, seasonal, trend+seasonal)

## Conclusion

On the full M5 dataset (13,766 eligible series of 30,490 total):

| | Complete | Reduced |
|---|---|---|
| **Accuracy** | RMSE 2.1998 | RMSE 2.1999 |
| **Speed** | 207 s | **120 s (1.7x faster)** |
| **Models evaluated** | ~9 per series | ~4 per series |

The Reduced pool delivers **identical accuracy** with **1.7x speedup** at scale.
For high-volume retail forecasting, this translates to significant compute savings
without any loss in forecast quality.

### Usage

```rust
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, ModelPool};

let config = AutoETSConfig::with_period(7)
    .with_model_pool(ModelPool::Reduced);
let mut model = AutoETS::with_config(config);
model.fit(&series).unwrap();
```

### All Available Pools

```rust
ModelPool::Complete               // 19 models (default)
ModelPool::NoMultiplicativeTrend  // 15 models
ModelPool::DampedTrendOnly        // 12 models
ModelPool::MatchErrorSeasonal     // 16 models
ModelPool::Reduced                // 8 models (recommended for scale)
```

## Reference

Petropoulos, F., Grushka-Cockayne, Y., Siemsen, E., & Spiliotis, E. (2023).
Wielding Occam's razor: Fast and frugal retail forecasting. *arXiv:2102.13209*.
