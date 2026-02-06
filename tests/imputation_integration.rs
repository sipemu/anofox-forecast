//! Integration tests for missing value imputation.
//!
//! Verifies that imputed time series can be successfully fitted by models.

use anofox_forecast::core::{MissingValuePolicy, TimeSeries};
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use chrono::{TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<chrono::Utc>> {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::days(i as i64))
        .collect()
}

#[test]
fn fit_after_backward_fill() {
    let timestamps = make_timestamps(30);
    let mut values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
    values[0] = f64::NAN; // Leading NaN
    values[1] = f64::NAN;

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.sanitized(MissingValuePolicy::BackwardFill).unwrap();
    assert!(!clean.has_missing_values());

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
}

#[test]
fn fit_after_fill_mean() {
    let timestamps = make_timestamps(30);
    let mut values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
    values[5] = f64::NAN;
    values[15] = f64::NAN;

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.sanitized(MissingValuePolicy::FillMean).unwrap();
    assert!(!clean.has_missing_values());

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
}

#[test]
fn fit_after_fill_median() {
    let timestamps = make_timestamps(30);
    let mut values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
    values[10] = f64::INFINITY;

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.sanitized(MissingValuePolicy::FillMedian).unwrap();
    assert!(!clean.has_missing_values());

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
}

#[test]
fn fit_after_interpolate_policy() {
    let timestamps = make_timestamps(30);
    let mut values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
    values[3] = f64::NAN;
    values[4] = f64::NAN;
    values[20] = f64::NAN;

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.sanitized(MissingValuePolicy::Interpolate).unwrap();
    assert!(!clean.has_missing_values());

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
}

#[test]
fn fit_after_forward_backward() {
    let timestamps = make_timestamps(30);
    let mut values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
    values[0] = f64::NAN; // leading
    values[29] = f64::NAN; // trailing
    values[15] = f64::NAN; // interior

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.imputed_forward_backward();
    assert!(!clean.has_missing_values());

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
}

#[test]
fn fit_after_moving_average_imputation() {
    let timestamps = make_timestamps(50);
    let mut values: Vec<f64> = (0..50)
        .map(|i| 10.0 + (i as f64 * 0.3).sin() * 5.0)
        .collect();
    values[10] = f64::NAN;
    values[25] = f64::NAN;
    values[26] = f64::NAN;

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.imputed_moving_average(5).unwrap();
    assert!(!clean.has_missing_values());

    let mut model = Naive::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.primary().len(), 5);
}

#[test]
fn fit_after_seasonal_imputation() {
    let timestamps = make_timestamps(42);
    // Weekly pattern: 6 full weeks
    let mut values: Vec<f64> = (0..42)
        .map(|i| {
            let day = i % 7;
            50.0 + match day {
                0 => 10.0,
                1 => 8.0,
                2 => 6.0,
                3 => 4.0,
                4 => 2.0,
                5 => -5.0,
                6 => -8.0,
                _ => 0.0,
            }
        })
        .collect();
    // Remove a few points
    values[7] = f64::NAN; // Monday of week 2
    values[21] = f64::NAN; // Monday of week 4

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.imputed_seasonal(7).unwrap();
    assert!(!clean.has_missing_values());

    // The imputed Monday values should be close to 60.0
    let monday_val = clean.primary_values()[7];
    assert!((monday_val - 60.0).abs() < 1.0);
}

#[test]
fn ets_fit_after_imputation() {
    let timestamps = make_timestamps(50);
    let mut values: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
    values[5] = f64::NAN;
    values[25] = f64::INFINITY;

    let ts = TimeSeries::univariate(timestamps, values).unwrap();
    let clean = ts.sanitized(MissingValuePolicy::FillMean).unwrap();

    let mut model = AutoETS::new();
    model.fit(&clean).unwrap();
    let forecast = model.predict(10).unwrap();
    assert_eq!(forecast.primary().len(), 10);
}
