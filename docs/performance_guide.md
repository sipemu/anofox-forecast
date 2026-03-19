# Performance Guide

Practical reference for choosing and tuning `anofox-forecast` models based on data size, speed requirements, and resource constraints.

---

## Model Complexity Overview

### Baseline Models

| Model | `fit()` Complexity | `predict()` Complexity | Memory | Parallel |
|-------|-------------------|----------------------|--------|----------|
| Naive | O(1) | O(h) | O(n) fitted values | No |
| SeasonalNaive | O(n) | O(h) | O(n + m) | No |
| RandomWalkWithDrift | O(n) | O(h) | O(n) | No |
| HistoricAverage | O(n) | O(h) | O(n) | No |
| SimpleMovingAverage | O(n) | O(h) | O(n) | No |
| WindowAverage | O(n) | O(h) | O(n) | No |
| SeasonalWindowAverage | O(n) | O(h) | O(n + m) | No |

*n = series length, h = forecast horizon, m = seasonal period*

### Exponential Smoothing Family

| Model | `fit()` Complexity | `predict()` Complexity | Memory | Parallel |
|-------|-------------------|----------------------|--------|----------|
| SES | O(n) fixed alpha; O(n * k) auto | O(h) | O(n) | No |
| HoltLinearTrend | O(n * k) auto | O(h) | O(n) | No |
| HoltWinters | O(n * m) fixed; O(n * m * k) auto | O(h) | O(n + m) | No |
| SeasonalES | O(n * m) | O(h) | O(n + m) | No |
| ETS | O(n * m) per specification | O(h) | O(n + m) | No |
| AutoETS | O(n * m * C) where C = candidates | O(h) | O(n + m) per candidate | Yes |

*k = optimizer iterations (~50-200), C = number of candidate models (up to 30, ~15 with additive_only)*

### ARIMA Family

| Model | `fit()` Complexity | `predict()` Complexity | Memory | Parallel |
|-------|-------------------|----------------------|--------|----------|
| ARIMA(p,d,q) | O(n * (p+q) * k) | O(h * (p+q)) | O(n + p + q) | No |
| SARIMA | O(n * (p+q+P+Q) * k) | O(h * (p+q+P+Q)) | O(n + p+q + (P+Q)*s) | No |
| AutoARIMA (stepwise) | O(n * max(p,q) * k * S) | O(h) | O(n) per candidate | Yes |
| AutoARIMA (exhaustive) | O(n * max_p * max_q * max_d * k) | O(h) | O(n) per candidate | Yes |

*k = optimizer iterations, S = stepwise search steps (~15-30), s = seasonal period*

### Theta Family

| Model | `fit()` Complexity | `predict()` Complexity | Memory | Parallel |
|-------|-------------------|----------------------|--------|----------|
| Theta (STM) | O(n) | O(h) | O(n) | No |
| OptimizedTheta (OTM) | O(n * k) | O(h) | O(n) | No |
| DynamicTheta (DSTM) | O(n) | O(h) | O(n) | No |
| DynamicOptimizedTheta (DOTM) | O(n * k) | O(h) | O(n) | No |
| AutoTheta | O(n * k * 4) tests all 4 variants | O(h) | O(n) per variant | No |

### Advanced Models

| Model | `fit()` Complexity | `predict()` Complexity | Memory | Parallel |
|-------|-------------------|----------------------|--------|----------|
| MFLES | O(n * R * F) | O(h * F) | O(n * R) | No |
| MSTLForecaster | O(n * m * I) + trend model fit | O(h) | O(n * S) | No |
| TBATS | O(n * H * k) | O(h * H) | O(n + H * s) | No |
| AutoTBATS | O(n * H * k * C) | O(h) | O(n) per candidate | No |
| GARCH(p,q) | O(n * (p+q) * k) | O(h) | O(n + p + q) | No |
| Ensemble | Sum of constituent models | O(h * M) | Sum of constituents | No |
| AutoEnsemble | Sum of constituent models | O(h * M) | Sum of constituents | No |
| AutoForecast | AutoARIMA + AutoETS + AutoTheta | O(h) | Keeps best only | No |

*R = boosting rounds (default 10), F = Fourier order, I = MSTL iterations, H = number of harmonics, S = number of seasonal components, M = number of ensemble members, C = candidate count*

### Intermittent Demand Models

| Model | `fit()` Complexity | `predict()` Complexity | Memory | Parallel |
|-------|-------------------|----------------------|--------|----------|
| Croston | O(n) | O(h) | O(n) | No |
| Croston (SBA) | O(n) | O(h) | O(n) | No |
| TSB | O(n) | O(h) | O(n) | No |
| ADIDA | O(n) | O(h) | O(n) | No |
| IMAPA | O(n * A) | O(h) | O(n) | No |

*A = number of aggregation levels*

---

## Quick Selection Guide

### By Data Size

| Data Size | Observations | Recommended Models | Why |
|-----------|-------------|-------------------|-----|
| **Tiny** | < 20 | Naive, HistoricAverage, SES | Too few data points for complex models |
| **Small** | 20-50 | SES, Theta, Naive, Croston (intermittent) | Simple models avoid overfitting |
| **Medium** | 50-500 | ARIMA, ETS, HoltWinters, OptimizedTheta | Enough data for parameter estimation |
| **Large** | 500-10k | AutoARIMA, AutoETS, MFLES, MSTLForecaster | Auto selection benefits from larger samples |
| **Very Large** | 10k+ | MFLES, MSTLForecaster | Gradient-boosted decomposition scales well |

### By Use Case

| Scenario | Recommended | Avoid |
|----------|-------------|-------|
| Real-time / low latency | Naive, SES, Theta | AutoARIMA, AutoETS |
| Best accuracy (single series) | AutoForecast, AutoEnsemble | Naive, HistoricAverage |
| Intermittent / sparse demand | Croston, TSB, ADIDA, IMAPA | ARIMA, HoltWinters |
| Multiple seasonalities | MSTLForecaster, TBATS | SES, basic ARIMA |
| Volatility forecasting | GARCH | ETS, Theta |
| WASM / browser | SES, Theta, ARIMA (fixed order) | AutoARIMA exhaustive, large ensembles |

---

## Performance Tips

### 1. Use the `parallel` Feature for Batch Operations

Enable rayon-based parallelism in `Cargo.toml`:

```toml
anofox-forecast = { version = "0.4", features = ["parallel"] }
```

AutoARIMA and AutoETS parallelize candidate evaluation internally when this feature is enabled.

### 2. Constrain AutoARIMA Search Space

Default search space: `max_p=5, max_q=5, max_d=2`. That means hundreds of candidate models in exhaustive mode. Reduce it:

```rust
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};

// Tight search space: 3x faster than defaults
let config = AutoARIMAConfig::default()
    .with_max_orders(3, 1, 3);    // max_p=3, max_d=1, max_q=3

// True stepwise: hill-climbing instead of grid, even faster
let config = AutoARIMAConfig::default()
    .with_true_stepwise();

// For non-seasonal data, skip seasonal orders entirely
let config = AutoARIMAConfig::default()
    .with_seasonal_period(0);
```

### 3. Use `additive_only()` with AutoETS

Restricts the candidate set to additive error and seasonal types, roughly halving the number of models evaluated:

```rust
use anofox_forecast::models::exponential::{AutoETS, AutoETSConfig};

// ~15 candidates instead of ~30
let config = AutoETSConfig::with_period(12).additive_only();
let mut model = AutoETS::with_config(config);
```

Use this when your data has no obvious multiplicative patterns (constant variance, no amplitude growth with level).

### 4. Tune MFLES for Your Needs

MFLES uses gradient-boosted decomposition. Each boosting round stores intermediate state:

- **Speed**: Reduce `max_rounds` (default 10). Fewer rounds = faster fit, less memory.
- **Memory**: Each round adds O(n) storage for seasonal/trend components.
- **Accuracy**: More rounds capture more complex patterns but risk overfitting short series.

### 5. Bootstrap: Trade Speed for Accuracy

The `bootstrap_forecast` function re-fits residual paths `n_samples` times:

```rust
use anofox_forecast::utils::bootstrap::BootstrapConfig;

// Fast: 50 samples (rough intervals)
let fast = BootstrapConfig::new(50);

// Accurate: 500 samples (smooth intervals, 10x slower)
let accurate = BootstrapConfig::new(500);

// Block bootstrap for autocorrelated residuals
let block = BootstrapConfig::new(100).with_block_size(10);
```

Cost is O(n_samples * predict_cost). For ARIMA, each sample requires a full predict pass.

### 6. SIMD Acceleration

The library uses Trueno for SIMD-accelerated vector operations (`src/simd.rs`). These are used internally for:

- Sum, mean, variance calculations
- Dot products and distance metrics
- Z-score normalization

SIMD operations use f32 internally for maximum throughput (~7 digits precision). No user action required -- the library uses these automatically on supported hardware (x86 SSE2/AVX2, ARM NEON).

### 7. STL/MSTL Decomposition

STL decomposition is used by MSTLForecaster and internally by Theta models for seasonal adjustment. Tips:

- Cost scales with `period * n * iterations`. Large seasonal periods (e.g., 365 for daily data) are expensive.
- Robust mode (`STL::new(12).robust()`) adds extra iterations and is ~2x slower.
- MSTLForecaster with multiple periods runs STL for each period sequentially.

### 8. Release Mode and LTO

The `Cargo.toml` is configured with:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

Always benchmark and deploy with `--release`. Debug builds are 10-50x slower for numerical code.

---

## Benchmark Reference

Run all benchmarks:

```bash
cargo bench
```

Run a specific benchmark:

```bash
cargo bench --bench arima_benchmark
```

### Benchmark Files

| File | Models Covered | Series Sizes |
|------|---------------|-------------|
| `benches/arima_benchmark.rs` | ARIMA(1,1,0), ARIMA(1,1,1), SARIMA, AutoARIMA (stepwise + true_stepwise) | n=120, 200, 500 |
| `benches/ets_benchmark.rs` | SES, Holt, HoltWinters, AutoETS (full + additive_only) | n=200, 500 |
| `benches/prediction_benchmark.rs` | ARIMA predict, SARIMA predict, SES predict, HoltWinters predict | h=10, 24, 50 |
| `benches/simd_benchmark.rs` | sum, sum_of_squares, dot, mean, variance, euclidean_distance, l1_distance, zscore | n=100 to 100k |
| `benches/stl_benchmark.rs` | STL decompose (standard + robust) | n=200, 500, 1000; p=12, 52 |
| `benches/bootstrap_benchmark.rs` | Residual bootstrap, block bootstrap | n=100, samples=100, h=10 |

### What to Expect

Rough order-of-magnitude fit times on modern hardware (release mode, single core):

| Model | n=200 | n=500 | n=1000 |
|-------|-------|-------|--------|
| Naive / SES (fixed) | < 1 us | < 1 us | < 1 us |
| SES (auto) | ~10 us | ~25 us | ~50 us |
| Holt (auto) | ~50 us | ~100 us | ~200 us |
| HoltWinters (auto) | ~200 us | ~500 us | ~1 ms |
| ARIMA(1,1,1) | ~100 us | ~200 us | ~500 us |
| AutoETS (period=12) | ~5 ms | ~15 ms | ~30 ms |
| AutoARIMA (stepwise) | ~5 ms | ~15 ms | ~40 ms |
| Theta | ~10 us | ~25 us | ~50 us |
| MFLES | ~500 us | ~1 ms | ~3 ms |
| STL decompose | ~50 us | ~100 us | ~300 us |

Prediction times are typically O(h) and sub-microsecond for all models.

---

## WASM Performance

The `anofox-forecast-js` crate compiles the library to WebAssembly for browser and Node.js use.

### Key Differences from Native

| Aspect | Native | WASM |
|--------|--------|------|
| SIMD | AVX2/SSE2/NEON via Trueno | Not available (f32 SIMD fallback) |
| Parallelism | rayon (`parallel` feature) | Not available (single-threaded) |
| Floating-point | Full hardware FPU | WASM FP (spec-compliant, slightly slower) |
| Startup cost | None | Module instantiation (~5-50ms) |
| Typical overhead | Baseline | 2-5x slower than native |

### WASM Build

```bash
cd crates/anofox-forecast-js
wasm-pack build --release --target web
```

The WASM crate uses `default-features = false` and enables only the `js` feature (for `getrandom` browser support). The `parallel` and `postprocess` features are not available in WASM.

### WASM Recommendations

- Prefer simple models (SES, Theta, fixed-order ARIMA) for interactive/real-time use.
- AutoARIMA and AutoETS work but take noticeably longer without parallelism.
- Consider pre-fitting models server-side and shipping only the predict step to the browser.
- The wasm-opt pass (`-O3`) is enabled in the release profile for maximum optimization.

---

## DuckDB Extension

For production workloads processing large volumes of time series data, consider the [forecast-extension](https://github.com/sipemu/forecast-extension) for DuckDB. It exposes `anofox-forecast` models as SQL functions, enabling:

- Forecasting directly inside analytical queries
- Batch processing thousands of series with DuckDB's parallelism
- Zero data movement between storage and forecasting
- Integration with existing data pipelines

```sql
-- Example: forecast each product's sales
SELECT product_id, forecast(sales, 12) AS predictions
FROM sales_data
GROUP BY product_id;
```

This is the recommended approach when forecasting is part of a larger data pipeline rather than a standalone Rust application.

---

## Feature Flags Summary

| Feature | Default | Effect on Performance |
|---------|---------|----------------------|
| `parallel` | No | Enables rayon for auto model search, CV, bootstrap |
| `postprocess` | Yes | Adds faer + anofox-regression for conformal prediction |
| `js` | No | Enables getrandom/js for WASM builds |
| `serde` | No | Adds serialization support (minimal performance impact) |

Enable only what you need. For maximum speed with parallel workloads:

```toml
anofox-forecast = { version = "0.4", features = ["parallel"] }
```

For minimal binary size (e.g., WASM):

```toml
anofox-forecast = { version = "0.4", default-features = false }
```
