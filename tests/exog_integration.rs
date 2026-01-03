//! Integration tests for exogenous variable support.
//!
//! These tests verify that the Rust implementation correctly handles exogenous regressors
//! and can be compared against Nixtla statsforecast reference outputs.

use anofox_forecast::core::{CalendarAnnotations, TimeSeries};
use anofox_forecast::models::arima::{AutoARIMA, ARIMA, SARIMA};
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::theta::{AutoTheta, Theta};
use anofox_forecast::models::Forecaster;
use chrono::{TimeZone, Utc};
use std::collections::HashMap;

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::days(i as i64))
        .collect()
}

/// Generate test data with known exogenous effects.
/// y = 50 + 5*sin(2*pi*t/7) + 10*(t/n) + AR(1) noise
fn generate_exog_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);

    // Coefficients
    let beta0 = 50.0;
    let beta1 = 5.0;
    let beta2 = 10.0;
    let ar_coef = 0.6;

    let mut ar_component = 0.0;
    let seed: u64 = 42;

    for i in 0..n {
        // Weekly pattern regressor
        let x1_val = (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
        x1.push(x1_val);

        // Linear trend regressor
        let x2_val = i as f64 / n as f64;
        x2.push(x2_val);

        // Exogenous contribution
        let exog_effect = beta0 + beta1 * x1_val + beta2 * x2_val;

        // Simple deterministic noise based on seed
        let noise = ((seed.wrapping_mul(i as u64 + 1) % 1000) as f64 - 500.0) / 500.0;
        ar_component = ar_coef * ar_component + noise;

        y.push(exog_effect + ar_component);
    }

    (y, x1, x2)
}

fn create_ts_with_regressors(n: usize, y: Vec<f64>, x1: Vec<f64>, x2: Vec<f64>) -> TimeSeries {
    let timestamps = make_timestamps(n);
    let calendar = CalendarAnnotations::new()
        .with_regressor("x1".to_string(), x1)
        .with_regressor("x2".to_string(), x2);

    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    ts.set_calendar(calendar);
    ts
}

fn create_future_regressors(
    start_idx: usize,
    horizon: usize,
    n: usize,
) -> HashMap<String, Vec<f64>> {
    let mut future = HashMap::new();

    let x1: Vec<f64> = (start_idx..start_idx + horizon)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
        .collect();

    let x2: Vec<f64> = (start_idx..start_idx + horizon)
        .map(|i| i as f64 / n as f64)
        .collect();

    future.insert("x1".to_string(), x1);
    future.insert("x2".to_string(), x2);
    future
}

#[test]
fn arima_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = ARIMA::new(1, 0, 1);
    model.fit(&ts).unwrap();

    // Model should report it has exogenous regressors
    assert!(model.supports_exog());
    assert!(model.has_exog());
    assert_eq!(model.exog_names().unwrap().len(), 2);

    // predict() should fail when model has exogenous
    assert!(model.predict(horizon).is_err());

    // predict_with_exog() should work
    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);

    // Forecasts should be reasonable (not NaN or infinite)
    for val in forecast.primary() {
        assert!(val.is_finite());
    }
}

#[test]
fn sarima_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = SARIMA::new(1, 0, 1, 1, 0, 0, 7);
    model.fit(&ts).unwrap();

    assert!(model.supports_exog());
    assert!(model.has_exog());

    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn auto_arima_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = AutoARIMA::new();
    model.fit(&ts).unwrap();

    assert!(model.supports_exog());
    assert!(model.has_exog());

    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn mfles_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = MFLES::new(vec![7]);
    model.fit(&ts).unwrap();

    assert!(model.supports_exog());
    assert!(model.has_exog());

    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn naive_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = Naive::new();
    model.fit(&ts).unwrap();

    assert!(model.supports_exog());
    assert!(model.has_exog());

    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn arima_without_exogenous_still_works() {
    // Verify that models without exogenous still work normally
    let n = 100;
    let horizon = 10;
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
        .collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = ARIMA::new(1, 1, 1);
    model.fit(&ts).unwrap();

    // Model should report it supports exog but doesn't have any
    assert!(model.supports_exog());
    assert!(!model.has_exog());
    assert!(model.exog_names().is_none());

    // predict() should work without exog
    let forecast = model.predict(horizon).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn missing_future_regressor_errors() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = ARIMA::new(1, 0, 1);
    model.fit(&ts).unwrap();

    // Provide only x1, missing x2
    let mut incomplete_future = HashMap::new();
    incomplete_future.insert(
        "x1".to_string(),
        (0..horizon).map(|i| (i as f64).sin()).collect(),
    );

    let result = model.predict_with_exog(horizon, &incomplete_future);
    assert!(result.is_err());
}

#[test]
fn wrong_regressor_length_errors() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = ARIMA::new(1, 0, 1);
    model.fit(&ts).unwrap();

    // Provide wrong length for x1
    let mut wrong_length_future = HashMap::new();
    wrong_length_future.insert("x1".to_string(), vec![1.0, 2.0, 3.0]); // Only 3 values
    wrong_length_future.insert(
        "x2".to_string(),
        (0..horizon).map(|i| i as f64 / n as f64).collect(),
    );

    let result = model.predict_with_exog(horizon, &wrong_length_future);
    assert!(result.is_err());
}

#[test]
fn exog_intervals_work() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = ARIMA::new(1, 0, 1);
    model.fit(&ts).unwrap();

    let future = create_future_regressors(n, horizon, n);
    let forecast = model
        .predict_with_exog_intervals(horizon, &future, 0.95)
        .unwrap();

    assert_eq!(forecast.horizon(), horizon);
    assert!(forecast.lower_series(0).is_ok());
    assert!(forecast.upper_series(0).is_ok());

    // Intervals should be valid
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    let point = forecast.primary();

    for i in 0..horizon {
        assert!(lower[i] <= point[i]);
        assert!(point[i] <= upper[i]);
    }
}

#[test]
fn exogenous_effect_visible_in_forecast() {
    // Test that exogenous regressors actually affect the forecast
    let n = 100;
    let horizon = 10;

    // Create data where y is strongly influenced by x1
    let timestamps = make_timestamps(n);
    let x1: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
        .collect();
    let y: Vec<f64> = x1.iter().map(|&x| 50.0 + 20.0 * x).collect();

    let calendar = CalendarAnnotations::new().with_regressor("x1".to_string(), x1);
    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    ts.set_calendar(calendar);

    let mut model = ARIMA::new(0, 0, 0); // Pure regression, no ARIMA
    model.fit(&ts).unwrap();

    // Create two different future scenarios
    let mut future_high = HashMap::new();
    future_high.insert("x1".to_string(), vec![1.0; horizon]); // x1 = 1 (high)

    let mut future_low = HashMap::new();
    future_low.insert("x1".to_string(), vec![-1.0; horizon]); // x1 = -1 (low)

    let forecast_high = model.predict_with_exog(horizon, &future_high).unwrap();
    let forecast_low = model.predict_with_exog(horizon, &future_low).unwrap();

    // High x1 should give higher forecasts
    for i in 0..horizon {
        assert!(
            forecast_high.primary()[i] > forecast_low.primary()[i],
            "Exogenous effect not visible at horizon {}",
            i + 1
        );
    }
}

#[test]
fn theta_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = Theta::new();
    model.fit(&ts).unwrap();

    // Model should report it has exogenous regressors
    assert!(model.supports_exog());
    assert!(model.has_exog());
    assert_eq!(model.exog_names().unwrap().len(), 2);

    // predict() should fail when model has exogenous
    assert!(model.predict(horizon).is_err());

    // predict_with_exog() should work
    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);

    // Forecasts should be reasonable (not NaN or infinite)
    for val in forecast.primary() {
        assert!(val.is_finite());
    }
}

#[test]
fn auto_theta_with_exogenous_basic() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = AutoTheta::new();
    model.fit(&ts).unwrap();

    assert!(model.supports_exog());
    assert!(model.has_exog());

    let future = create_future_regressors(n, horizon, n);
    let forecast = model.predict_with_exog(horizon, &future).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn theta_without_exogenous_still_works() {
    // Verify that Theta without exogenous still works normally
    let n = 100;
    let horizon = 10;
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
        .collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = Theta::new();
    model.fit(&ts).unwrap();

    // Model should report it supports exog but doesn't have any
    assert!(model.supports_exog());
    assert!(!model.has_exog());
    assert!(model.exog_names().is_none());

    // predict() should work without exog
    let forecast = model.predict(horizon).unwrap();
    assert_eq!(forecast.horizon(), horizon);
}

#[test]
fn theta_exog_intervals_work() {
    let n = 100;
    let horizon = 10;
    let (y, x1, x2) = generate_exog_test_data(n);
    let ts = create_ts_with_regressors(n, y, x1, x2);

    let mut model = Theta::new();
    model.fit(&ts).unwrap();

    let future = create_future_regressors(n, horizon, n);
    let forecast = model
        .predict_with_exog_intervals(horizon, &future, 0.95)
        .unwrap();

    assert_eq!(forecast.horizon(), horizon);
    assert!(forecast.lower_series(0).is_ok());
    assert!(forecast.upper_series(0).is_ok());

    // Intervals should be valid
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    let point = forecast.primary();

    for i in 0..horizon {
        assert!(lower[i] <= point[i]);
        assert!(point[i] <= upper[i]);
    }
}
