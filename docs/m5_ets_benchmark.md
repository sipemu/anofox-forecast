# AutoETS Model Pool Benchmark: Full M5 Dataset

Comparison of AutoETS **Complete** (19 models) vs **Reduced** (8 models) pool
on the full M5 retail forecasting dataset (30,490 series), based on Petropoulos et al. (2023)
"Wielding Occam's razor: Fast and frugal retail forecasting" ([arXiv:2102.13209](https://arxiv.org/abs/2102.13209)).

## Setup

| Parameter | Value |
|---|---|
| Dataset | M5 — all 30,490 item-store combinations (daily Walmart sales) |
| Series evaluated | **30,490** (100%, no filtering) |
| Training window | 1,941 days (2011-01-29 to 2016-05-22) |
| Test horizon | 28 days (4 weeks, matching M5 competition) |
| Seasonal period | 7 (weekly) |
| Selection criterion | AICc |
| Platform | Linux, Rust release build, rayon parallel (multi-core) |

## Results

### Accuracy

| Metric | Complete (19 models) | Reduced (8 models) | Difference |
|---|---|---|---|
| Avg RMSE | 1.4332 | 1.4331 | -0.0001 (-0.007%) |
| Median RMSE | 0.9257 | 0.9257 | 0.0000 |
| Avg MAPE | 63.72% | 63.70% | -0.02pp |
| Avg sMAPE | 141.40% | 141.39% | -0.01pp |
| Success rate | 30,490/30,490 (100%) | 30,490/30,490 (100%) |

**Identical accuracy.** Across all 30,490 series, RMSE differs by less than 0.01%.

### Speed

| Metric | Complete | Reduced | Speedup |
|---|---|---|---|
| Wall-clock time | 370.8 s | **184.9 s** | **2.01x** |
| Avg CPU time per series | 250.7 ms | **121.4 ms** | **2.06x** |
| Total CPU time | 7,644 s | **3,702 s** | **2.07x** |
| Throughput | 82 series/s | **165 series/s** | **2.01x** |

The Reduced pool is **2x faster** across all 30,490 series, saving **186 seconds** wall-clock.

### Model Selection Distribution

| Model | Complete | Reduced | Notes |
|---|---|---|---|
| ETS(A,N,A) | 20,851 (68.4%) | 21,363 (70.1%) | Seasonal, no trend |
| ETS(A,N,N) | 6,317 (20.7%) | 7,260 (23.8%) | Simple exponential smoothing |
| ETS(A,A,N) | 1,452 (4.8%) | — | Undamped trend (Complete only) |
| ETS(A,Ad,A) | 1,183 (3.9%) | 1,269 (4.2%) | Seasonal, damped trend |
| ETS(A,A,A) | 464 (1.5%) | — | Undamped seasonal (Complete only) |
| ETS(A,Ad,N) | 223 (0.7%) | 598 (2.0%) | Damped trend, no season |

Key observations:
- **68-70%** of M5 series select ETS(A,N,A) — additive seasonal without trend
- **21-24%** select ETS(A,N,N) — simple exponential smoothing (highly intermittent SKUs)
- The ~1,900 series selecting undamped trend (A,A,N / A,A,A) in Complete switch to
  damped variants or simpler models in Reduced — more robust for retail forecasting
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

On the **complete M5 dataset** (30,490 series, no filtering):

| | Complete | Reduced |
|---|---|---|
| **RMSE** | 1.4332 | **1.4331** |
| **Wall-clock** | 371 s | **185 s (2.0x faster)** |
| **CPU/series** | 251 ms | **121 ms (2.1x faster)** |
| **Failure rate** | 0% | 0% |

The Reduced pool delivers **identical accuracy with 2x speedup**. At 30,490 series,
this saves **3 minutes** wall-clock on a multi-core machine, or **65 minutes** of
total CPU time (7,644s vs 3,702s).

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
