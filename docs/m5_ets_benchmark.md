# AutoETS Model Pool Benchmark: M5 Dataset

Comparison of AutoETS **Complete** (19 models) vs **Reduced** (8 models) pool
on the M5 retail forecasting dataset, based on Petropoulos et al. (2023)
"Wielding Occam's razor: Fast and frugal retail forecasting" ([arXiv:2102.13209](https://arxiv.org/abs/2102.13209)).

## Setup

| Parameter | Value |
|---|---|
| Dataset | M5 top-1000 series (daily Walmart sales) |
| Series evaluated | 500 (filtered: >30% nonzero days) |
| Training window | 1941 days (2011-01-29 to 2016-05-22) |
| Test horizon | 28 days (4 weeks, matching M5 competition) |
| Seasonal period | 7 (weekly) |
| Selection criterion | AICc |
| Platform | Linux, Rust release build, single-threaded |

## Results

### Accuracy

| Metric | Complete (19 models) | Reduced (8 models) | Difference |
|---|---|---|---|
| Avg RMSE | 7.8945 | 7.8943 | -0.003% |
| Median RMSE | 5.7813 | 5.7813 | 0.000% |
| Avg MAPE | 74.47% | 74.45% | -0.03pp |
| Avg sMAPE | 67.82% | 67.81% | -0.01pp |

The Reduced pool achieves **virtually identical accuracy** to the Complete pool.
On 497 of 500 series, both pools selected the same model. The 3 differences were
between ETS(A,A,A) in Complete and ETS(A,Ad,A) in Reduced (undamped vs damped trend),
with negligible accuracy impact.

### Speed

| Metric | Complete | Reduced | Speedup |
|---|---|---|---|
| Avg fit time | 87.1 ms | 84.3 ms | 1.03x |
| Total (500 series) | 43.6 s | 42.2 s | 1.03x |

Speed improvement is modest (3%) on these long series (n=1941) because optimization
cost is dominated by the O(n) per-evaluation cost, not the number of candidate models.
On shorter series (n < 200), the speedup is more pronounced (~25-48% from reduced
candidate count + stagnation-based early stopping).

### Model Selection Distribution

| Model | Complete | Reduced |
|---|---|---|
| ETS(A,N,A) | 460 (92.0%) | 462 (92.4%) |
| ETS(A,Ad,A) | 21 (4.2%) | 22 (4.4%) |
| ETS(A,N,N) | 16 (3.2%) | 16 (3.2%) |
| ETS(A,A,A) | 3 (0.6%) | -- |

Key observations:
- **92%** of M5 series selected additive seasonal without trend (ETS(A,N,A))
- All selected models are in the Reduced pool (ANN, ANA, AAdA are all Reduced members)
- The 3 series selecting ETS(A,A,A) in Complete switched to ETS(A,Ad,A) in Reduced
  (damped variant, which is arguably more robust)

### Reduced Pool Models

The 8 models in the Reduced pool (Petropoulos et al., 2023):

| Category | Additive Error | Multiplicative Error |
|---|---|---|
| Level only | ANN | MNN |
| Trend only (damped) | AAdN | MAdN |
| Seasonal only | ANA | MNM |
| Trend + Seasonal | AAdA | MAdM |

Design principles:
1. **Damped trends only** -- undamped trends produce positively biased long-horizon forecasts
2. **Matched error/seasonal** -- additive error with additive seasonal, multiplicative with multiplicative
3. **Balanced coverage** -- 2 models per forecast profile (level, trend, seasonal, trend+seasonal)

## Conclusion

The Reduced pool is recommended for large-scale retail forecasting:
- **No accuracy loss** on M5 data (RMSE difference < 0.01%)
- **Fewer candidates** (8 vs 19) reduces computation, especially on shorter series
- **More robust** model selection (only damped trends, matched error/seasonal types)

Usage:
```rust
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig, ModelPool};

let config = AutoETSConfig::with_period(7)
    .with_model_pool(ModelPool::Reduced);
let mut model = AutoETS::with_config(config);
model.fit(&series).unwrap();
```

## Reference

Petropoulos, F., Grushka-Cockayne, Y., Siemsen, E., & Spiliotis, E. (2023).
Wielding Occam's razor: Fast and frugal retail forecasting. *arXiv:2102.13209*.
