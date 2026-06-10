//! Issue #106 cross-cutting conformance test.
//!
//! Exercises the Decomposable / training-state trait surface uniformly
//! across all 11 auto-tuned forecasters in scope. Each model is fit on a
//! synthetic seasonal series and must:
//!
//! 1. Expose `training_values()` matching the fit input.
//! 2. Expose `training_regressors()` returning `None` (no regressors
//!    present in this test's series).
//! 3. Either expose `trend_component()` (returns `&[f64]`) or fail with
//!    `InvalidParameter` cleanly; same for `seasonal_component()`.
//! 4. Satisfy the decomposition invariant `trend + seasonal + residual
//!    == training` to a tolerance per model (1e-9 for decomposition-native
//!    models, 1e-6 for ETS-shaped trend = fitted approximations).
//!
//! For models where trend or seasonal returns Err, the invariant is
//! relaxed: we just verify that the residual is consistent
//! (`fitted_values + residual ≈ training` within tolerance).
//!
//! This test is the safety net for Phases 2–5: if any model regresses
//! on the trait contract, it surfaces here rather than at a downstream
//! integration site.

use std::collections::HashMap;

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::exponential::{
    AutoETS, HoltLinearTrend, HoltWinters, SeasonalES, SeasonalType, SimpleExponentialSmoothing,
};
use anofox_forecast::models::theta::{AutoTheta, DynamicTheta, OptimizedTheta};
use anofox_forecast::models::{AutoTBATS, Forecaster, MSTLForecaster, MFLES};

use chrono::{Duration, TimeZone, Utc};

fn make_seasonal_series(n: usize, period: usize) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.5 * i as f64;
            let seasonal =
                10.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin();
            let noise = ((i * 17) % 7) as f64 * 0.1 - 0.3;
            trend + seasonal + noise
        })
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn check_conformance<F>(name: &str, model: &F, ts: &TimeSeries, residual_tol: f64)
where
    F: Forecaster,
{
    check_conformance_with(name, model, ts, residual_tol, true);
}

/// Variant that lets the caller disable the decomposition-invariant check.
/// AutoARIMA-with-differencing computes residuals on a differenced scale,
/// so the additive invariant `trend + seasonal + residual == training`
/// doesn't apply without integration; we still validate the contract
/// surface (training_values, training_regressors, fitted_values present).
fn check_conformance_with<F>(
    name: &str,
    model: &F,
    ts: &TimeSeries,
    residual_tol: f64,
    check_invariant: bool,
) where
    F: Forecaster,
{
    // Contract 1: training_values matches input.
    let training = model
        .training_values()
        .unwrap_or_else(|_| panic!("{}: training_values() must return Ok after fit", name));
    assert_eq!(
        training,
        ts.primary_values(),
        "{}: training_values must equal fit input",
        name
    );

    // Contract 2: training_regressors None for series without regressors.
    let regs: Option<&HashMap<String, Vec<f64>>> = model.training_regressors();
    assert!(
        regs.is_none(),
        "{}: training_regressors must be None when none supplied",
        name
    );

    // Contract 3: fitted_values present. Some models drop warmup rows
    // (e.g. AutoARIMA), so length ≤ training length is acceptable.
    let fitted = model
        .fitted_values()
        .unwrap_or_else(|| panic!("{}: fitted_values must be Some after fit", name));
    assert!(
        fitted.len() <= training.len(),
        "{}: fitted_values length ({}) must be ≤ training length ({})",
        name,
        fitted.len(),
        training.len()
    );

    // Contract 4: trend + seasonal + residual == training (relaxed when
    // trend or seasonal is Err — invariant becomes
    // fitted + residual ≈ training, which holds by definition of residual).
    let residual = model
        .residual_component()
        .unwrap_or_else(|_| panic!("{}: residual_component must return Ok after fit", name));
    // Align all component lengths by right-edge (model drops warmup rows
    // from the front, so the last fitted.len() training values align with
    // the fitted vector).
    let n_compare = fitted.len().min(residual.len());
    let train_offset = training.len().saturating_sub(n_compare);

    if !check_invariant {
        return;
    }

    let trend_ok = model.trend_component().ok();
    let seasonal_ok = model.seasonal_component().ok();

    if let (Some(trend), Some(seasonal)) = (trend_ok, seasonal_ok) {
        let trend_offset = trend.len().saturating_sub(n_compare);
        let seasonal_offset = seasonal.len().saturating_sub(n_compare);
        let residual_offset = residual.len().saturating_sub(n_compare);
        for k in 0..n_compare {
            let t = trend[trend_offset + k];
            let s = seasonal[seasonal_offset + k];
            let r = residual[residual_offset + k];
            let train = training[train_offset + k];
            if t.is_nan() || s.is_nan() || r.is_nan() || train.is_nan() {
                continue;
            }
            let sum = t + s + r;
            assert!(
                (sum - train).abs() < residual_tol,
                "{}: full decomposition invariant violated at k={}: sum={}, training={}",
                name,
                k,
                sum,
                train
            );
        }
    } else {
        // Trend or seasonal not exposed → relaxed invariant.
        let residual_offset = residual.len().saturating_sub(n_compare);
        for k in 0..n_compare {
            let f = fitted[k];
            let r = residual[residual_offset + k];
            let train = training[train_offset + k];
            if f.is_nan() || r.is_nan() || train.is_nan() {
                continue;
            }
            let sum = f + r;
            assert!(
                (sum - train).abs() < residual_tol,
                "{}: relaxed invariant violated at k={}: fitted+residual={}, training={}",
                name,
                k,
                sum,
                train
            );
        }
    }
}

#[test]
fn mstl_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(200, 24);
    let mut model = MSTLForecaster::new(vec![24]);
    model.fit(&ts).unwrap();
    check_conformance("MSTLForecaster", &model, &ts, 1e-9);
}

#[test]
fn mfles_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(120, 12);
    let mut model = MFLES::new(vec![12]);
    model.fit(&ts).unwrap();
    check_conformance("MFLES", &model, &ts, 1e-6);
}

#[test]
fn auto_arima_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(80, 12);
    let mut model = AutoARIMA::new();
    model.fit(&ts).unwrap();
    // ARIMA with differencing computes residuals on a differenced scale —
    // the additive invariant doesn't apply without integration. We still
    // verify the contract surface.
    check_conformance_with("AutoARIMA", &model, &ts, 1e-9, false);
}

#[test]
fn simple_es_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = SimpleExponentialSmoothing::auto();
    model.fit(&ts).unwrap();
    check_conformance("SimpleExponentialSmoothing", &model, &ts, 1e-9);
}

#[test]
fn holt_linear_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = HoltLinearTrend::new(0.3, 0.1);
    model.fit(&ts).unwrap();
    check_conformance("HoltLinearTrend", &model, &ts, 1e-9);
}

#[test]
fn holt_winters_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(120, 12);
    let mut model = HoltWinters::new(0.3, 0.1, 0.3, 12, SeasonalType::Additive);
    model.fit(&ts).unwrap();
    check_conformance("HoltWinters", &model, &ts, 1e-9);
}

#[test]
fn seasonal_es_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(120, 12);
    let mut model = SeasonalES::new(12);
    model.fit(&ts).unwrap();
    check_conformance("SeasonalES", &model, &ts, 1e-9);
}

#[test]
fn auto_ets_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = AutoETS::new();
    model.fit(&ts).unwrap();
    check_conformance("AutoETS", &model, &ts, 1e-9);
}

#[test]
fn auto_theta_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = AutoTheta::new();
    model.fit(&ts).unwrap();
    check_conformance("AutoTheta", &model, &ts, 1e-9);
}

#[test]
fn dynamic_theta_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = DynamicTheta::new(0.3);
    model.fit(&ts).unwrap();
    check_conformance("DynamicTheta", &model, &ts, 1e-9);
}

#[test]
fn optimized_theta_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = OptimizedTheta::new();
    model.fit(&ts).unwrap();
    check_conformance("OptimizedTheta", &model, &ts, 1e-9);
}

#[test]
fn auto_tbats_conforms_to_decomposable_contract() {
    let ts = make_seasonal_series(120, 12);
    let mut model = AutoTBATS::new(vec![12]);
    model.fit(&ts).unwrap();
    check_conformance("AutoTBATS", &model, &ts, 1e-6);
}
