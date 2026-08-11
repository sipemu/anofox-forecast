//! Edge-case robustness integration suite — ROBUST-01.
//!
//! One representative model per family is driven through the full ROADMAP
//! edge-case input set: constant, n=2, all-zeros/intermittent, NaN/Inf,
//! zero-length, extreme-scale. Each test asserts the exact `ForecastError`
//! variant where the outcome is deterministic, or `is_err()` + no-panic.
//! No test may trigger a panic.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::error::ForecastError;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

/// Build a `TimeSeries` from a slice of values with hourly timestamps.
///
/// The `.unwrap()` here is safe — we are constructing valid-shaped inputs;
/// the constructor invariant is that timestamps and values have equal length.
/// The unwrap is on the constructor, never on `fit()` or `predict()`.
fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

// ── Tracer: Naive + NaN → MissingValues ──────────────────────────────────────

#[test]
fn naive_nan_returns_missing_values() {
    let mut values = vec![1.0f64; 30];
    values[14] = f64::NAN;
    let ts = make_ts(&values);
    let result = Naive::new().fit(&ts);
    assert!(
        matches!(result, Err(ForecastError::MissingValues)),
        "expected MissingValues for NaN-containing series, got: {:?}",
        result
    );
}
