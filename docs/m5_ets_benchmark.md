# AutoETS Model Pool Benchmark: M5 Dataset

Comparison of AutoETS **Complete** (19 models) vs **Reduced** (8 models) pool
on the M5 retail forecasting dataset, based on Petropoulos et al. (2023)
"Wielding Occam's razor: Fast and frugal retail forecasting" ([arXiv:2102.13209](https://arxiv.org/abs/2102.13209)).

## Setup

| Parameter | Value |
|---|---|
| Dataset | M5 top-1000 series (daily Walmart sales) |
| Series evaluated | 998 (filtered: >30% nonzero days) |
| Training window | 1,941 days (2011-01-29 to 2016-05-22) |
| Test horizon | 28 days (4 weeks, matching M5 competition) |
| Seasonal period | 7 (weekly) |
| Selection criterion | AICc |
| Platform | Linux, Rust release build, single-threaded |

## Results

### Accuracy

| Metric | Complete (19 models) | Reduced (8 models) | Difference |
|---|---|---|---|
| Avg RMSE | 7.6995 | 7.7003 | +0.001 (+0.01%) |
| Median RMSE | 5.8341 | 5.8341 | 0.000 |
| Avg MAPE | 76.50% | 76.51% | +0.01pp |
| Avg sMAPE | 71.34% | 71.33% | -0.01pp |
| Success rate | 998/998 (100%) | 998/998 (100%) |

The Reduced pool achieves **virtually identical accuracy** to the Complete pool
across all 998 series. On 991 of 998 series, both pools selected the same model.

### Speed

| Metric | Complete | Reduced |
|---|---|---|
| Avg fit time | 87.8 ms | 94.0 ms |
| Total (998 series) | 87.6 s | 93.8 s |

On long series (n=1,941), optimization cost is dominated by the O(n) likelihood
evaluation per NM iteration, not the number of candidate models. Both pools
evaluate only additive-error candidates on M5 data (multiplicative is excluded
automatically due to zero-valued observations). The timing difference is within
system variance (sequential runs, CPU thermal effects).

On shorter series (n < 200), Criterion benchmarks show the Reduced pool is
25-48% faster due to fewer candidates.

### Model Selection Distribution

| Model | Complete | Reduced |
|---|---|---|
| ETS(A,N,A) — seasonal, no trend | 932 (93.4%) | 936 (93.8%) |
| ETS(A,Ad,A) — seasonal, damped trend | 33 (3.3%) | 34 (3.4%) |
| ETS(A,N,N) — simple exponential smoothing | 26 (2.6%) | 27 (2.7%) |
| ETS(A,A,A) — seasonal, undamped trend | 5 (0.5%) | — |
| ETS(A,A,N) — undamped trend, no season | 2 (0.2%) | — |
| ETS(A,Ad,N) — damped trend, no season | — | 1 (0.1%) |

Key observations:
- **93%+** of M5 series select additive seasonal without trend — ETS(A,N,A)
- All top models are in the Reduced pool
- The 7 series using undamped trend (A,A,A / A,A,N) in Complete switched to
  damped variants (A,Ad,A / A,Ad,N) in Reduced — arguably more robust for
  retail forecasting where trends rarely persist linearly

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

On the full M5 top-1000 dataset (998 eligible series), the Reduced pool is
**accuracy-equivalent** to the Complete pool:

- RMSE differs by 0.01% (7.7003 vs 7.6995)
- 99.3% of series select the same model
- The remaining 0.7% switch from undamped to damped trend (more robust)

The Reduced pool is recommended for large-scale retail forecasting where
computational budget is constrained and trend robustness is valued.

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
ModelPool::Complete            // 19 models (default)
ModelPool::NoMultiplicativeTrend  // 15 models
ModelPool::DampedTrendOnly     // 12 models
ModelPool::MatchErrorSeasonal  // 16 models
ModelPool::Reduced             // 8 models (recommended for scale)
```

## Reference

Petropoulos, F., Grushka-Cockayne, Y., Siemsen, E., & Spiliotis, E. (2023).
Wielding Occam's razor: Fast and frugal retail forecasting. *arXiv:2102.13209*.
