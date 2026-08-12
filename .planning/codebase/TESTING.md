# Testing Patterns

**Analysis Date:** 2026-08-09

## Test Framework

**Runner:**
- Rust built-in test harness via `#[test]` attribute
- No external test framework (no pytest, jest, etc.)
- Criterion for benchmarks (harness = false for custom benchmarks)
- Proptest for property-based testing

**Assertion Library:**
- Rust std `assert!()`, `assert_eq!()`, `assert_ne!()` macros
- Proptest assertions via `prop_assert!()`, `prop_assert_eq!()`
- Approx comparison: `assert!(value.abs() < tolerance)` with explicit tolerance
- Custom assertion helpers for slice comparison: `assert_slices_approx_eq(a, b, tol, context)`

**Run Commands:**
```bash
cargo test                          # Run all tests (unit + integration)
cargo test --lib                    # Unit tests only (src/**/*.rs)
cargo test --test '*'               # Integration tests only (tests/*.rs)
cargo test --doc                    # Doc tests only
cargo test -- --nocapture          # Show println! output during test run
cargo test -- --test-threads=1     # Run tests serially (useful for debugging)
cargo test --all-features          # Test all features enabled
make test                           # Makefile target (runs cargo test)
cargo clippy --all-targets --all-features -- -D warnings  # Lint tests
```

## Test File Organization

**Location:**
- **Unit tests:** Inline within source files under `#[cfg(test)] mod tests`
- **Integration tests:** Separate files in `tests/` directory (not under src)
- **Examples:** In `examples/` directory (documented usage, not tests)
- **Benchmarks:** In `benches/` directory with `harness = false`

**Naming:**
- Unit test modules: `#[cfg(test)] mod tests { ... }`
- Integration test files: `tests/{feature}_*.rs` or `tests/{model}_validation.rs`
  - Examples: `auto_arima_validation.rs`, `ets_validation.rs`, `property_tests.rs`
  - Descriptive names indicating what is being validated
- Test functions: `#[test] fn {operation}_{condition}_produces_{expected}()`
  - Examples: `naive_forecast_length_matches_horizon()`, `ses_json_round_trip_predictions_match()`

**Structure:**
```
tests/
├── auto_arima_validation.rs        # ARIMA order selection against reference
├── ets_validation.rs                # ETS model validation against statsforecast
├── property_tests.rs                # Property-based tests for invariants
├── persistence_integration.rs       # Serialization/deserialization round-trips
├── pipeline_integration.rs          # End-to-end pipeline tests
├── exog_integration.rs              # Exogenous regressor handling
├── interval_calibration.rs          # Prediction interval calibration
└── data/                            # Test fixtures
    └── reference/                   # Comparison reference values
```

## Test Structure

**Suite Organization:**
```rust
// Integration test file structure
//! Detailed description of what is being tested.
//!
//! Explains testing strategy and what invariants are verified.

#[cfg(feature = "serde")]
mod serde_tests {
    use anofox_forecast::core::{Forecast, TimeSeries};
    use anofox_forecast::models::Forecaster;
    // ... more imports
    
    fn make_test_data(n: usize) -> TimeSeries {
        // Shared test data factory
    }
    
    #[test]
    fn test_name() {
        // Test implementation
    }
}
```

**Patterns:**
- **Setup:** Helper functions at top of test module (e.g., `make_ts()`, `make_timestamps()`)
- **Arrange:** Create test data and model in test function
- **Act:** Call the operation being tested
- **Assert:** Verify results with tolerance-based comparisons
- **Teardown:** Automatic (Rust ownership)

**Setup Pattern:**
```rust
fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

fn make_test_data(n: usize) -> (Vec<chrono::DateTime<Utc>>, Vec<f64>) {
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let season = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let noise = ((42u64.wrapping_mul(i as u64 + 1) % 1000) as f64 - 500.0) / 500.0;
            trend + season + noise
        })
        .collect();
    (timestamps, values)
}
```

**Assertion Pattern:**
```rust
fn assert_slices_approx_eq(a: &[f64], b: &[f64], tol: f64, context: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", context);
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff < tol,
            "{}: mismatch at index {}: {} vs {} (diff {})",
            context, i, va, vb, diff
        );
    }
}

// Usage
assert_slices_approx_eq(
    original_forecast.primary(),
    restored_forecast.primary(),
    1e-10,
    "Naive JSON round-trip",
);
```

## Mocking

**Framework:** Manual implementation via helper test functions

**Patterns:**
- No mocking library (mockito, mocktopus, etc.)
- Test doubles created by hand: factories, stubs, dummy data generators
- All external dependencies (chrono, statrs) are deterministic and testable
- Models tested with deterministic synthetic time series

**What to Mock:**
- Time: Use `Utc.with_ymd_and_hms()` with fixed dates (not real time)
- Random data: Use deterministic generators with fixed seeds
- External APIs: Not applicable (no external API calls in library)

**What NOT to Mock:**
- Model behavior (test actual algorithms)
- Math operations (use real statrs, trueno, lbfgs)
- Serialization (test real serde round-trips)

**Example from `tests/property_tests.rs`:**
```rust
fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

/// Strategy for generating seasonal time series.
fn seasonal_values_strategy(
    min_len: usize,
    max_len: usize,
    period: usize,
) -> impl Strategy<Value = Vec<f64>> {
    (min_len..max_len).prop_flat_map(move |len| {
        (50.0..100.0_f64, 5.0..20.0_f64).prop_map(move |(base, amplitude)| {
            (0..len)
                .map(|i| {
                    base + amplitude * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                })
                .collect()
        })
    })
}
```

## Fixtures and Factories

**Test Data:**
```rust
// Trend + seasonality + noise
fn make_test_data(n: usize) -> (Vec<chrono::DateTime<Utc>>, Vec<f64>) {
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.3 * i as f64;
            let season = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let noise = ((42u64.wrapping_mul(i as u64 + 1) % 1000) as f64 - 500.0) / 500.0;
            trend + season + noise
        })
        .collect();
    (timestamps, values)
}

// Constant series with small noise
let series: Vec<f64> = (0..200)
    .map(|i| 5.0 + ((i * 7 + 3) % 11) as f64 * 0.2 - 1.0)
    .collect();

// Level shift pattern (three segments)
let mut series = Vec::new();
for _ in 0..100 { series.push(0.0 + noise); }
for _ in 0..100 { series.push(10.0 + noise); }
for _ in 0..100 { series.push(5.0 + noise); }
```

**Location:**
- Factories defined at top of test module (before individual tests)
- Shared across multiple tests within same file
- Factory functions follow naming convention: `make_{data_type}()` or `{data_type}_strategy()` (for proptest)

**Examples from `tests/`:**
- `make_ts()` — Generic TimeSeries builder
- `make_timestamps()` — Vec<DateTime> builder
- `make_test_data()` — Tuple of (timestamps, values) with trend + season + noise
- `valid_values_strategy()` — Proptest strategy for valid f64 ranges
- `trending_values_strategy()` — Proptest strategy for trending time series
- `seasonal_values_strategy()` — Proptest strategy for seasonal patterns

## Coverage

**Requirements:** 
- No automated minimum coverage enforced
- Manual review of critical paths (model fitting, prediction)
- Edge cases validated: empty data, NaN/Inf handling, dimension mismatches

**View Coverage:**
```bash
# Generate coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html --output-dir coverage

# OR with llvm-cov
cargo llvm-cov --html

# OR with kcov
cargo kcov --output coverage
```

## Test Types

**Unit Tests:**
- **Scope:** Individual functions, error handling, boundary conditions
- **Approach:** Inline `#[cfg(test)] mod tests` within source files
- **Examples:** `src/error.rs::tests`, `src/models/baseline/naive.rs::tests`
- **Coverage:** Error message formatting, struct construction, field access

**Integration Tests:**
- **Scope:** End-to-end workflows, model accuracy against reference values, round-trips
- **Approach:** Separate files in `tests/` directory
- **Examples:** `tests/ets_validation.rs`, `tests/persistence_integration.rs`
- **Coverage:** Model fit → predict → validation, serialization round-trips, exogenous handling

**Property-Based Tests:**
- **Framework:** Proptest
- **Scope:** Invariants that must hold for all valid inputs
- **Approach:** `proptest!` macro with strategy generators
- **File:** `tests/property_tests.rs`
- **Invariants tested:**
  - Forecast length matches horizon
  - All forecast values are finite (not NaN/Inf)
  - Model predictions remain stable across many inputs
  - Residuals have expected statistical properties

**Property Test Example from `tests/property_tests.rs`:**
```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn naive_forecast_length_matches_horizon(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        prop_assert_eq!(forecast.horizon(), horizon);
    }

    #[test]
    fn naive_forecasts_are_finite(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        for val in forecast.primary() {
            prop_assert!(val.is_finite(), "Forecast contains non-finite value: {}", val);
        }
    }
}
```

**E2E Tests:**
- **Framework:** None (integration tests serve this purpose)
- **Scope:** Real-world workflows documented in examples
- **Approach:** Executable examples in `examples/` that demonstrate complete use cases
- **Coverage:** Quickstart, model selection, exogenous handling, serialization

## Inline Module Tests

**Pattern:**
Located at end of modules using `#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn operation_works_correctly() {
        // Test implementation
    }
}
```

**Example from `src/error.rs`:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_descriptive() {
        let err = ForecastError::EmptyData;
        assert_eq!(err.to_string(), "empty input data");
        
        let err = ForecastError::InsufficientData {
            needed: 10,
            got: 5,
            hint: None,
        };
        assert_eq!(err.to_string(), "insufficient data: need at least 10, got 5");
    }

    #[test]
    fn errors_are_clonable_and_comparable() {
        let err1 = ForecastError::EmptyData;
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }
}
```

## Test Patterns for Common Scenarios

**Model Fit + Predict Validation:**
```rust
#[test]
fn ses_on_near_constant_data_produces_flat_forecast() {
    let values: Vec<f64> = (0..60)
        .map(|i| 42.0 + ((i * 7 + 3) % 11) as f64 * 0.05)
        .collect();
    let ts = TimeSeries::univariate(make_timestamps(60), values.clone()).unwrap();

    let mut model = SimpleExponentialSmoothing::auto();
    model.fit(&ts).unwrap();
    let fc = model.predict(10).unwrap();

    // Assert on model parameters
    let alpha = model.alpha().expect("alpha should be available");
    assert!(alpha < 0.5, "Alpha should be small for near-constant data");
    
    // Assert on forecast shape and values
    let first = fc.primary()[0];
    for (i, &pred) in fc.primary().iter().enumerate() {
        assert!(
            (pred - first).abs() < 1e-10,
            "h={}: SES forecasts should be flat",
            i + 1
        );
    }
}
```

**Serialization Round-Trip:**
```rust
#[test]
fn naive_json_round_trip_predictions_match() {
    let ts = make_ts(50);
    let horizon = 5;

    let mut model = Naive::new();
    model.fit(&ts).unwrap();
    let original_forecast = model.predict(horizon).unwrap();

    let json = to_json(&model).unwrap();
    assert!(json.len() > 10, "JSON should be non-empty");

    let restored: Naive = from_json(&json).unwrap();
    let restored_forecast = restored.predict(horizon).unwrap();

    assert_slices_approx_eq(
        original_forecast.primary(),
        restored_forecast.primary(),
        1e-10,
        "Naive JSON",
    );
}
```

**Dimension/Validation Error Handling:**
```rust
#[test]
fn insufficient_data_returns_error() {
    let ts = TimeSeries::univariate(
        make_timestamps(2),
        vec![1.0, 2.0],
    ).unwrap();
    
    let mut model = ARIMA::new(2, 0, 2);
    let result = model.fit(&ts);
    
    assert!(matches!(
        result,
        Err(ForecastError::InsufficientData { .. })
    ));
}
```

## Benchmark Structure

**Files:**
- Located in `benches/` directory with `harness = false` in Cargo.toml
- Examples: `simd_benchmark.rs`, `arima_benchmark.rs`, `ets_benchmark.rs`

**Framework:** Criterion.rs for microbenchmarks

**Pattern:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_model_fit(c: &mut Criterion) {
    c.bench_function("arima_fit_m3_daily", |b| {
        b.iter(|| {
            let mut model = ARIMA::new(1, 0, 1);
            model.fit(black_box(&ts)).unwrap();
        })
    });
}

criterion_group!(benches, benchmark_model_fit);
criterion_main!(benches);
```

---

*Testing analysis: 2026-08-09*
