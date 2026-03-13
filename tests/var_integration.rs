//! Integration tests for the VAR multivariate forecasting pipeline.
#![allow(clippy::needless_range_loop)]
//!
//! Tests cover:
//! - VAR model on correlated time series
//! - VARForecaster with TimeSeries + regressors
//! - Granger causality on known causal structure
//! - Forecast accuracy on simple VAR(1) generated data
//! - VAR with different lag orders

use anofox_forecast::core::{CalendarAnnotations, TimeSeries};
use anofox_forecast::models::var::VAR;
use anofox_forecast::models::var_forecaster::VARForecaster;
use anofox_forecast::models::Forecaster;
use chrono::{TimeZone, Utc};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::days(i as i64))
        .collect()
}

/// Generate a VAR(1) system with known coefficients:
///   y1(t) = c1 + a11*y1(t-1) + a12*y2(t-1) + e1
///   y2(t) = c2 + a21*y1(t-1) + a22*y2(t-1) + e2
fn generate_var1(
    n: usize,
    c: [f64; 2],
    a: [[f64; 2]; 2],
    noise_scale: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];

    y1[0] = rng.gen_range(-1.0..1.0);
    y2[0] = rng.gen_range(-1.0..1.0);

    for t in 1..n {
        let e1: f64 = rng.gen_range(-noise_scale..noise_scale);
        let e2: f64 = rng.gen_range(-noise_scale..noise_scale);
        y1[t] = c[0] + a[0][0] * y1[t - 1] + a[0][1] * y2[t - 1] + e1;
        y2[t] = c[1] + a[1][0] * y1[t - 1] + a[1][1] * y2[t - 1] + e2;
    }

    vec![y1, y2]
}

/// Generate a 3-variable VAR(1) chain: y1 -> y2 -> y3
fn generate_chain_var1(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut y = vec![vec![0.0; n]; 3];

    for var in 0..3 {
        y[var][0] = rng.gen_range(-0.5..0.5);
    }

    for t in 1..n {
        y[0][t] = 0.5 * y[0][t - 1] + rng.gen_range(-0.01..0.01);
        y[1][t] = 0.3 * y[0][t - 1] + 0.4 * y[1][t - 1] + rng.gen_range(-0.01..0.01);
        y[2][t] = 0.25 * y[1][t - 1] + 0.3 * y[2][t - 1] + rng.gen_range(-0.01..0.01);
    }

    y
}

// ---------------------------------------------------------------------------
// VAR with 2 correlated time series
// ---------------------------------------------------------------------------

#[test]
fn var_two_correlated_series_recovers_coefficients() {
    let c = [0.5, 0.3];
    let a = [[0.6, 0.1], [0.05, 0.7]];
    let data = generate_var1(500, c, a, 0.01, 42);

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();

    let coefs = model.coefficients().unwrap();
    let intercepts = model.intercepts().unwrap();

    // Intercepts should be close to the true values
    assert!(
        (intercepts[0] - c[0]).abs() < 0.1,
        "intercept[0]: expected ~{}, got {}",
        c[0],
        intercepts[0]
    );
    assert!(
        (intercepts[1] - c[1]).abs() < 0.1,
        "intercept[1]: expected ~{}, got {}",
        c[1],
        intercepts[1]
    );

    // Coefficient matrix should be close to the true values
    assert!(
        (coefs[0][0][0] - a[0][0]).abs() < 0.05,
        "a11: expected ~{}, got {}",
        a[0][0],
        coefs[0][0][0]
    );
    assert!(
        (coefs[0][1][0] - a[0][1]).abs() < 0.05,
        "a12: expected ~{}, got {}",
        a[0][1],
        coefs[0][1][0]
    );
    assert!(
        (coefs[1][0][0] - a[1][0]).abs() < 0.05,
        "a21: expected ~{}, got {}",
        a[1][0],
        coefs[1][0][0]
    );
    assert!(
        (coefs[1][1][0] - a[1][1]).abs() < 0.05,
        "a22: expected ~{}, got {}",
        a[1][1],
        coefs[1][1][0]
    );
}

#[test]
fn var_three_correlated_series_dimensions_correct() {
    let data = generate_chain_var1(200, 77);

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();

    assert_eq!(model.n_vars(), 3);

    let forecasts = model.predict(10).unwrap();
    assert_eq!(forecasts.len(), 3);
    for var in 0..3 {
        assert_eq!(forecasts[var].len(), 10);
        for &val in &forecasts[var] {
            assert!(val.is_finite(), "forecast contains non-finite value");
        }
    }
}

// ---------------------------------------------------------------------------
// VARForecaster with TimeSeries + regressors
// ---------------------------------------------------------------------------

#[test]
fn var_forecaster_univariate_fit_predict() {
    let n = 100;
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| 10.0 + (i as f64 * 0.2).sin() * 3.0)
        .collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = VARForecaster::new(2);
    model.fit(&ts).unwrap();

    let forecast = model.predict(10).unwrap();
    assert_eq!(forecast.horizon(), 10);

    for &val in forecast.primary() {
        assert!(val.is_finite(), "forecast contains non-finite value");
    }
}

#[test]
fn var_forecaster_with_regressors_uses_all_variables() {
    // Create a time series where the primary values are driven by a regressor
    let n = 200;
    let timestamps = make_timestamps(n);

    let mut rng = StdRng::seed_from_u64(123);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];

    x[0] = 1.0;
    y[0] = 2.0;

    for t in 1..n {
        let e1: f64 = rng.gen_range(-0.01..0.01);
        let e2: f64 = rng.gen_range(-0.01..0.01);
        x[t] = 0.8 * x[t - 1] + e1;
        y[t] = 0.3 * x[t - 1] + 0.5 * y[t - 1] + e2;
    }

    let calendar = CalendarAnnotations::new().with_regressor("x_driver".to_string(), x.clone());
    let mut ts = TimeSeries::univariate(timestamps, y).unwrap();
    ts.set_calendar(calendar);

    let mut model = VARForecaster::new(1);
    model.fit(&ts).unwrap();

    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.horizon(), 5);

    // Fitted values should be available and match the series length
    let fitted = model.fitted_values().unwrap();
    assert_eq!(fitted.len(), n);

    // First value is NaN padding for lag-1
    assert!(fitted[0].is_nan());
    // Remaining fitted values should be finite
    for &val in &fitted[1..] {
        assert!(val.is_finite(), "fitted value is not finite");
    }
}

#[test]
fn var_forecaster_prediction_intervals_are_ordered() {
    let n = 150;
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| 50.0 + (i as f64 * 0.1).sin() * 10.0)
        .collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = VARForecaster::new(1);
    model.fit(&ts).unwrap();

    let forecast = model.predict_with_intervals(10, 0.95).unwrap();
    assert_eq!(forecast.horizon(), 10);
    assert!(forecast.has_lower());
    assert!(forecast.has_upper());

    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    let point = forecast.primary();

    for i in 0..10 {
        assert!(
            lower[i] <= point[i],
            "lower[{}] = {} > point[{}] = {}",
            i,
            lower[i],
            i,
            point[i]
        );
        assert!(
            point[i] <= upper[i],
            "point[{}] = {} > upper[{}] = {}",
            i,
            point[i],
            i,
            upper[i]
        );
    }

    // Intervals should widen with horizon
    for i in 1..10 {
        let prev_width = upper[i - 1] - lower[i - 1];
        let curr_width = upper[i] - lower[i];
        assert!(
            curr_width >= prev_width - 1e-10,
            "interval width decreased at step {}: {} -> {}",
            i,
            prev_width,
            curr_width
        );
    }
}

// ---------------------------------------------------------------------------
// Granger causality on known causal structure
// ---------------------------------------------------------------------------

#[test]
fn granger_causality_detects_unidirectional_cause() {
    // y1 drives y2 but y2 does NOT drive y1
    // y1(t) = 0.5*y1(t-1)
    // y2(t) = 0.3*y1(t-1) + 0.5*y2(t-1)
    let data = generate_var1(500, [0.0, 0.0], [[0.5, 0.0], [0.3, 0.5]], 0.01, 42);

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();

    let f_y1_causes_y2 = model.granger_causality_test(0, 1).unwrap();
    let f_y2_causes_y1 = model.granger_causality_test(1, 0).unwrap();

    // y1->y2 should be significant
    assert!(
        f_y1_causes_y2 > 4.0,
        "y1->y2 F-stat ({}) should be significant (> 4.0)",
        f_y1_causes_y2
    );

    // y2->y1 should be much smaller (no true causation)
    assert!(
        f_y1_causes_y2 > f_y2_causes_y1 * 5.0,
        "y1->y2 ({:.1}) should be much larger than y2->y1 ({:.1})",
        f_y1_causes_y2,
        f_y2_causes_y1
    );
}

#[test]
fn granger_causality_on_chain_structure() {
    // Chain: y0 -> y1 -> y2
    let data = generate_chain_var1(500, 99);

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();

    let f_0_to_1 = model.granger_causality_test(0, 1).unwrap();
    let f_1_to_2 = model.granger_causality_test(1, 2).unwrap();
    let f_2_to_0 = model.granger_causality_test(2, 0).unwrap();

    // Direct causal links should be significant
    assert!(
        f_0_to_1 > 4.0,
        "y0->y1 F-stat ({:.1}) should be significant",
        f_0_to_1
    );
    assert!(
        f_1_to_2 > 4.0,
        "y1->y2 F-stat ({:.1}) should be significant",
        f_1_to_2
    );

    // Reverse direction (y2 -> y0) should be non-significant
    assert!(
        f_0_to_1 > f_2_to_0,
        "y0->y1 ({:.1}) should exceed y2->y0 ({:.1})",
        f_0_to_1,
        f_2_to_0
    );
}

#[test]
fn granger_causality_bidirectional_feedback() {
    // Both directions have influence
    let data = generate_var1(500, [0.0, 0.0], [[0.5, 0.2], [0.2, 0.5]], 0.01, 55);

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();

    let f_01 = model.granger_causality_test(0, 1).unwrap();
    let f_10 = model.granger_causality_test(1, 0).unwrap();

    // Both directions should be significant
    assert!(
        f_01 > 4.0,
        "y0->y1 F-stat ({:.1}) should be significant with bidirectional feedback",
        f_01
    );
    assert!(
        f_10 > 4.0,
        "y1->y0 F-stat ({:.1}) should be significant with bidirectional feedback",
        f_10
    );
}

// ---------------------------------------------------------------------------
// VAR forecast accuracy on simple VAR(1) generated data
// ---------------------------------------------------------------------------

#[test]
fn var1_forecast_accuracy_on_generated_data() {
    let c = [1.0, 0.5];
    let a = [[0.6, 0.1], [0.05, 0.7]];
    let n = 300;
    let horizon = 10;

    // Generate training data
    let data = generate_var1(n, c, a, 0.01, 42);

    // Generate the "true" future by continuing the same DGP
    let mut rng = StdRng::seed_from_u64(9999);
    let mut true_future = vec![vec![0.0; horizon]; 2];
    let mut prev = [data[0][n - 1], data[1][n - 1]];
    for h in 0..horizon {
        let y1_new = c[0] + a[0][0] * prev[0] + a[0][1] * prev[1] + rng.gen_range(-0.01..0.01);
        let y2_new = c[1] + a[1][0] * prev[0] + a[1][1] * prev[1] + rng.gen_range(-0.01..0.01);
        true_future[0][h] = y1_new;
        true_future[1][h] = y2_new;
        prev = [y1_new, y2_new];
    }

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();
    let forecasts = model.predict(horizon).unwrap();

    // Since noise is very small, forecasts should be close to true future
    for var in 0..2 {
        for h in 0..horizon {
            let err = (forecasts[var][h] - true_future[var][h]).abs();
            assert!(
                err < 0.5,
                "var {} step {}: forecast {:.4} vs true {:.4} (error {:.4})",
                var,
                h,
                forecasts[var][h],
                true_future[var][h],
                err
            );
        }
    }
}

#[test]
fn var1_fitted_plus_residuals_reconstruct_original() {
    let data = generate_var1(100, [0.5, 0.3], [[0.6, 0.1], [0.05, 0.7]], 0.1, 42);

    let mut model = VAR::new(1);
    model.fit(&data).unwrap();

    let fitted = model.fitted_values().unwrap();
    let residuals = model.residuals().unwrap();

    let p = model.order();

    for var in 0..2 {
        for t in 0..fitted[var].len() {
            let reconstructed = fitted[var][t] + residuals[var][t];
            let original = data[var][t + p];
            let diff = (reconstructed - original).abs();
            assert!(
                diff < 1e-10,
                "reconstruction error at var={}, t={}: {:.2e}",
                var,
                t,
                diff
            );
        }
    }

    // Residuals should have small variance when noise scale is small
    let residual_data = generate_var1(500, [0.5, 0.3], [[0.6, 0.1], [0.05, 0.7]], 0.01, 42);
    let mut m2 = VAR::new(1);
    m2.fit(&residual_data).unwrap();
    let resid = m2.residuals().unwrap();
    for var in 0..2 {
        let var_r: f64 = resid[var].iter().map(|r| r * r).sum::<f64>() / resid[var].len() as f64;
        assert!(
            var_r < 0.001,
            "var {} residual variance ({:.6}) should be small with low-noise data",
            var,
            var_r
        );
    }
}

// ---------------------------------------------------------------------------
// VAR with different lag orders
// ---------------------------------------------------------------------------

#[test]
fn var2_recovers_two_lag_coefficients() {
    let mut rng = StdRng::seed_from_u64(88);
    let n = 500;
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];

    y1[0] = 0.1;
    y1[1] = 0.2;
    y2[0] = -0.1;
    y2[1] = 0.0;

    // True DGP: VAR(2)
    // y1(t) = 0.4*y1(t-1) + 0.1*y1(t-2) + 0.05*y2(t-1)
    // y2(t) = 0.3*y2(t-1) + 0.15*y2(t-2) + 0.1*y1(t-1)
    for t in 2..n {
        y1[t] = 0.4 * y1[t - 1] + 0.1 * y1[t - 2] + 0.05 * y2[t - 1] + rng.gen_range(-0.01..0.01);
        y2[t] = 0.3 * y2[t - 1] + 0.15 * y2[t - 2] + 0.1 * y1[t - 1] + rng.gen_range(-0.01..0.01);
    }

    let data = vec![y1, y2];
    let mut model = VAR::new(2);
    model.fit(&data).unwrap();

    let coefs = model.coefficients().unwrap();

    // Check eq0: y1 at lag1 ~ 0.4
    assert!(
        (coefs[0][0][0] - 0.4).abs() < 0.05,
        "y1 lag1 in eq0: expected ~0.4, got {:.4}",
        coefs[0][0][0]
    );
    // Check eq0: y1 at lag2 ~ 0.1
    assert!(
        (coefs[0][0][1] - 0.1).abs() < 0.05,
        "y1 lag2 in eq0: expected ~0.1, got {:.4}",
        coefs[0][0][1]
    );
    // Check eq0: y2 at lag1 ~ 0.05
    assert!(
        (coefs[0][1][0] - 0.05).abs() < 0.05,
        "y2 lag1 in eq0: expected ~0.05, got {:.4}",
        coefs[0][1][0]
    );
}

#[test]
fn var_lag_order_affects_forecast_quality() {
    // Generate VAR(2) data
    let mut rng = StdRng::seed_from_u64(77);
    let n = 300;
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];
    y1[0] = 0.5;
    y1[1] = 0.3;
    y2[0] = -0.2;
    y2[1] = 0.1;

    for t in 2..n {
        y1[t] = 0.4 * y1[t - 1] + 0.15 * y1[t - 2] + 0.05 * y2[t - 1] + rng.gen_range(-0.01..0.01);
        y2[t] = 0.3 * y2[t - 1] + 0.1 * y2[t - 2] + 0.1 * y1[t - 1] + rng.gen_range(-0.01..0.01);
    }

    let data = vec![y1.clone(), y2.clone()];

    // Fit with correct order (p=2)
    let mut model_correct = VAR::new(2);
    model_correct.fit(&data).unwrap();
    let residuals_correct = model_correct.residuals().unwrap();
    let rss_correct: f64 = residuals_correct[0].iter().map(|r| r * r).sum();

    // Fit with wrong order (p=1) -- should have larger residuals
    let mut model_wrong = VAR::new(1);
    model_wrong.fit(&data).unwrap();
    let residuals_wrong = model_wrong.residuals().unwrap();
    let rss_wrong: f64 = residuals_wrong[0].iter().map(|r| r * r).sum();

    // The correctly specified model should have lower RSS
    assert!(
        rss_correct < rss_wrong,
        "VAR(2) RSS ({:.6}) should be lower than VAR(1) RSS ({:.6}) for VAR(2) data",
        rss_correct,
        rss_wrong
    );
}

#[test]
fn var_high_lag_order_on_small_data_still_works() {
    // VAR(3) on data with just enough observations
    let data = generate_var1(20, [0.0, 0.0], [[0.3, 0.1], [0.1, 0.3]], 0.5, 42);

    let mut model = VAR::new(3);
    model.fit(&data).unwrap();

    let forecasts = model.predict(5).unwrap();
    assert_eq!(forecasts.len(), 2);
    assert_eq!(forecasts[0].len(), 5);

    for var in 0..2 {
        for &val in &forecasts[var] {
            assert!(val.is_finite(), "forecast contains non-finite value");
        }
    }
}
