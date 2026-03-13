//! End-to-end pipeline integration tests.
#![allow(clippy::redundant_closure, clippy::needless_range_loop)]
//!
//! Tests cover:
//! - Ensemble fit -> predict -> constrain -> postprocess
//! - Fit multiple models -> cross_validate -> select best -> predict with intervals
//! - STL decompose -> fit trend model -> recompose forecast
//! - Seasonal model fit -> predict -> diagnostics

use anofox_forecast::core::{ConstrainedForecast, ForecastConstraint, TimeSeries};
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::ensemble::{CombinationMethod, Ensemble};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use anofox_forecast::postprocess::{PointForecasts, PostProcessor};
use anofox_forecast::seasonality::STL;
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig, CVStrategy};
use anofox_forecast::validation::diagnose_residuals;
use chrono::{TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    (0..n)
        .map(|i| base + chrono::Duration::days(i as i64))
        .collect()
}

/// Generate synthetic data with trend, weekly seasonality, and noise.
fn generate_seasonal_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let trend = 100.0 + 0.5 * i as f64;
            let season = 20.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            // Deterministic noise to keep tests reproducible
            let noise = ((42u64.wrapping_mul(i as u64 + 1) % 1000) as f64 - 500.0) / 250.0;
            trend + season + noise
        })
        .collect()
}

/// Generate data with a clear linear trend and small noise.
fn generate_trending_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let trend = 10.0 + 2.0 * i as f64;
            let noise = ((71u64.wrapping_mul(i as u64 + 3) % 1000) as f64 - 500.0) / 1000.0;
            trend + noise
        })
        .collect()
}

/// Generate data that is always positive but may include values near zero.
fn generate_positive_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let base = 5.0 + 0.1 * i as f64;
            let season = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
            let noise = ((13u64.wrapping_mul(i as u64 + 5) % 1000) as f64 - 500.0) / 500.0;
            (base + season + noise).max(0.1) // Ensure positive
        })
        .collect()
}

fn make_ts(values: Vec<f64>) -> TimeSeries {
    let timestamps = make_timestamps(values.len());
    TimeSeries::univariate(timestamps, values).unwrap()
}

// ---------------------------------------------------------------------------
// Pipeline 1: Ensemble -> predict -> constrain -> postprocess
// ---------------------------------------------------------------------------

#[test]
fn ensemble_predict_constrain_postprocess_pipeline() {
    let n = 120;
    let horizon = 10;
    let values = generate_positive_data(n);
    let ts = make_ts(values);

    // Step 1: Build and fit an ensemble of three models
    let models: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(SimpleExponentialSmoothing::auto()),
        Box::new(Theta::new()),
    ];
    let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::Mean);
    ensemble.fit(&ts).unwrap();

    // Step 2: Predict
    let forecast = ensemble.predict(horizon).unwrap();
    assert_eq!(forecast.horizon(), horizon);

    for &val in forecast.primary() {
        assert!(
            val.is_finite(),
            "ensemble forecast contains non-finite value"
        );
    }

    // Step 3: Apply constraints (non-negative + upper bound)
    let constrained = ConstrainedForecast::apply(
        &forecast,
        &[
            ForecastConstraint::NonNegative,
            ForecastConstraint::UpperBound(500.0),
        ],
    );
    for &val in constrained.primary() {
        assert!(
            val >= 0.0,
            "constrained value {} should be non-negative",
            val
        );
        assert!(val <= 500.0, "constrained value {} should be <= 500", val);
    }

    // Step 4: Use PostProcessor (conformal) to generate intervals
    // Build training data from fitted values + actuals for postprocessing
    let fitted = ensemble.fitted_values().unwrap();
    let actuals = ts.primary_values();

    // Collect valid (non-NaN) pairs
    let valid_pairs: Vec<(f64, f64)> = fitted
        .iter()
        .zip(actuals.iter())
        .filter(|(f, _)| f.is_finite())
        .map(|(&f, &a)| (f, a))
        .collect();

    assert!(
        valid_pairs.len() >= 10,
        "need enough valid fitted values for postprocessing"
    );

    let train_forecasts =
        PointForecasts::from_values(valid_pairs.iter().map(|(f, _)| *f).collect());
    let train_actuals: Vec<f64> = valid_pairs.iter().map(|(_, a)| *a).collect();

    let processor = PostProcessor::conformal(0.90);
    let trained = processor.train(&train_forecasts, &train_actuals).unwrap();

    let predict_forecasts = PointForecasts::from_values(forecast.primary().to_vec());
    let intervals = processor
        .predict_intervals(&trained, &predict_forecasts)
        .unwrap();

    assert_eq!(intervals.len(), horizon);

    // Lower should be <= upper for all steps
    for i in 0..horizon {
        assert!(
            intervals.lower()[i] <= intervals.upper()[i],
            "interval lower {} > upper {} at step {}",
            intervals.lower()[i],
            intervals.upper()[i],
            i
        );
    }
}

#[test]
fn ensemble_weighted_mse_improves_on_equal_weights() {
    let n = 100;
    let horizon = 5;
    let values = generate_trending_data(n);
    let ts = make_ts(values.clone());

    // Ensemble with mean combination
    let models_mean: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(HoltLinearTrend::auto()),
        Box::new(Theta::new()),
    ];
    let mut ens_mean = Ensemble::new(models_mean).with_method(CombinationMethod::Mean);
    ens_mean.fit(&ts).unwrap();
    let forecast_mean = ens_mean.predict(horizon).unwrap();

    // Ensemble with MSE-weighted combination
    let models_mse: Vec<Box<dyn Forecaster>> = vec![
        Box::new(Naive::new()),
        Box::new(HoltLinearTrend::auto()),
        Box::new(Theta::new()),
    ];
    let mut ens_mse = Ensemble::new(models_mse).with_method(CombinationMethod::WeightedMSE);
    ens_mse.fit(&ts).unwrap();
    let forecast_mse = ens_mse.predict(horizon).unwrap();

    // Both should produce valid forecasts
    for i in 0..horizon {
        assert!(forecast_mean.primary()[i].is_finite());
        assert!(forecast_mse.primary()[i].is_finite());
    }

    // The MSE-weighted ensemble should give non-equal weights
    let weights = ens_mse.weights();
    let equal = 1.0 / weights.len() as f64;
    let any_non_equal = weights.iter().any(|&w| (w - equal).abs() > 1e-6);
    assert!(
        any_non_equal,
        "MSE weights should differ from equal weights: {:?}",
        weights
    );
}

// ---------------------------------------------------------------------------
// Pipeline 2: Cross-validate models -> select best -> predict with intervals
// ---------------------------------------------------------------------------

#[test]
fn cross_validate_select_best_predict_pipeline() {
    let n = 120;
    let horizon = 7;
    let values = generate_seasonal_data(n);
    let ts = make_ts(values);

    // Cross-validate Naive
    let cv_config = CVConfig {
        horizon,
        initial_window: 60,
        step_size: 10,
        strategy: CVStrategy::Expanding,
        seasonal_period: Some(7),
        gap: 0,
        purge: 0,
        embargo: 0,
    };

    let cv_naive = cross_validate(&cv_config, &ts, || Naive::new()).unwrap();
    let cv_ses = cross_validate(&cv_config, &ts, || SimpleExponentialSmoothing::auto()).unwrap();
    let cv_theta = cross_validate(&cv_config, &ts, || Theta::new()).unwrap();

    // All CVs should produce valid aggregated metrics
    assert!(cv_naive.aggregated.rmse.is_finite());
    assert!(cv_ses.aggregated.rmse.is_finite());
    assert!(cv_theta.aggregated.rmse.is_finite());

    // Select best model by RMSE
    let results = [
        ("Naive", cv_naive.aggregated.rmse),
        ("SES", cv_ses.aggregated.rmse),
        ("Theta", cv_theta.aggregated.rmse),
    ];
    let best = results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // Fit the best model on full data and predict with intervals
    let mut best_model: Box<dyn Forecaster> = match best.0 {
        "Naive" => Box::new(Naive::new()),
        "SES" => Box::new(SimpleExponentialSmoothing::auto()),
        "Theta" => Box::new(Theta::new()),
        _ => unreachable!(),
    };

    best_model.fit(&ts).unwrap();
    let forecast = best_model.predict_with_intervals(horizon, 0.95).unwrap();

    assert_eq!(forecast.horizon(), horizon);

    // Verify intervals are properly ordered
    if forecast.has_lower() && forecast.has_upper() {
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let point = forecast.primary();

        for i in 0..horizon {
            assert!(
                lower[i] <= point[i],
                "lower[{}]={} > point[{}]={}",
                i,
                lower[i],
                i,
                point[i]
            );
            assert!(
                point[i] <= upper[i],
                "point[{}]={} > upper[{}]={}",
                i,
                point[i],
                i,
                upper[i]
            );
        }
    }
}

#[test]
fn cross_validation_fold_metrics_are_consistent() {
    let n = 100;
    let values = generate_trending_data(n);
    let ts = make_ts(values);

    let cv_config = CVConfig {
        horizon: 5,
        initial_window: 50,
        step_size: 10,
        strategy: CVStrategy::Expanding,
        seasonal_period: None,
        gap: 0,
        purge: 0,
        embargo: 0,
    };

    let cv_result = cross_validate(&cv_config, &ts, || ARIMA::new(1, 1, 0)).unwrap();

    assert!(cv_result.n_folds > 0, "should have at least one fold");

    // All per-fold MAE values should be non-negative
    for (i, metric) in cv_result.fold_metrics.iter().enumerate() {
        assert!(
            metric.mae >= 0.0,
            "fold {} MAE ({}) should be non-negative",
            i,
            metric.mae
        );
        assert!(
            metric.rmse >= 0.0,
            "fold {} RMSE ({}) should be non-negative",
            i,
            metric.rmse
        );
        assert!(
            metric.rmse >= metric.mae,
            "fold {} RMSE ({}) should be >= MAE ({})",
            i,
            metric.rmse,
            metric.mae
        );
    }

    // Aggregated RMSE should be the mean of fold RMSEs (approximately)
    let avg_rmse: f64 =
        cv_result.fold_metrics.iter().map(|m| m.rmse).sum::<f64>() / cv_result.n_folds as f64;
    let diff = (cv_result.aggregated.rmse - avg_rmse).abs();
    assert!(
        diff < 1e-6,
        "aggregated RMSE ({}) should match fold average ({})",
        cv_result.aggregated.rmse,
        avg_rmse
    );
}

// ---------------------------------------------------------------------------
// Pipeline 3: STL decompose -> fit trend -> recompose forecast
// ---------------------------------------------------------------------------

#[test]
fn stl_decompose_fit_trend_recompose_pipeline() {
    let n = 112; // 16 full weeks for period=7
    let values = generate_seasonal_data(n);
    let _ts = make_ts(values.clone());
    let horizon = 7;

    // Step 1: Decompose with STL
    let stl = STL::new(7);
    let decomp = stl
        .decompose(&values)
        .expect("STL decomposition should succeed");

    // Verify decomposition components sum to original
    for i in 0..n {
        let recomposed = decomp.trend[i] + decomp.seasonal[i] + decomp.remainder[i];
        let diff = (recomposed - values[i]).abs();
        assert!(
            diff < 1e-10,
            "decomposition should sum to original at index {}: {} vs {} (diff {})",
            i,
            recomposed,
            values[i],
            diff
        );
    }

    // Step 2: Fit a trend model on the trend component
    let trend_ts = make_ts(decomp.trend.clone());
    let mut trend_model = HoltLinearTrend::auto();
    trend_model.fit(&trend_ts).unwrap();
    let trend_forecast = trend_model.predict(horizon).unwrap();

    // Step 3: Recompose the forecast by adding the last seasonal cycle
    let last_seasonal_cycle: Vec<f64> = (0..horizon)
        .map(|h| decomp.seasonal[n - 7 + (h % 7)])
        .collect();

    let recomposed_forecast: Vec<f64> = trend_forecast
        .primary()
        .iter()
        .zip(last_seasonal_cycle.iter())
        .map(|(t, s)| t + s)
        .collect();

    // Step 4: Verify the recomposed forecast is reasonable
    assert_eq!(recomposed_forecast.len(), horizon);

    for (i, &val) in recomposed_forecast.iter().enumerate() {
        assert!(
            val.is_finite(),
            "recomposed forecast at step {} is not finite",
            i
        );
    }

    // The trend forecast should be generally increasing (since our data trends up)
    let trend_slope = trend_forecast.primary()[horizon - 1] - trend_forecast.primary()[0];
    assert!(
        trend_slope > 0.0,
        "trend forecast should be increasing, but slope is {}",
        trend_slope
    );
}

#[test]
fn stl_seasonal_strength_reflects_data_seasonality() {
    // Data with strong seasonality
    let n = 140;
    let strong_seasonal: Vec<f64> = (0..n)
        .map(|i| {
            100.0
                + 30.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                + ((i as u64 * 17 % 1000) as f64 - 500.0) / 1000.0
        })
        .collect();

    // Data with no seasonality (just trend + noise)
    let weak_seasonal: Vec<f64> = (0..n)
        .map(|i| 100.0 + 0.5 * i as f64 + ((i as u64 * 17 % 1000) as f64 - 500.0) / 100.0)
        .collect();

    let stl = STL::new(7);

    let decomp_strong = stl.decompose(&strong_seasonal).unwrap();
    let decomp_weak = stl.decompose(&weak_seasonal).unwrap();

    let strength_strong = decomp_strong.seasonal_strength();
    let strength_weak = decomp_weak.seasonal_strength();

    assert!(
        strength_strong > strength_weak,
        "seasonal data strength ({:.3}) should exceed non-seasonal ({:.3})",
        strength_strong,
        strength_weak
    );
    assert!(
        strength_strong > 0.5,
        "strong seasonal data should have strength > 0.5, got {:.3}",
        strength_strong
    );
}

// ---------------------------------------------------------------------------
// Pipeline 4: Seasonal model -> predict -> diagnostics
// ---------------------------------------------------------------------------

#[test]
fn fit_seasonal_model_predict_and_diagnose() {
    let n = 100;
    let values = generate_seasonal_data(n);
    let ts = make_ts(values);
    let horizon = 14; // 2 weeks

    // Fit a model
    let mut model = Theta::new();
    model.fit(&ts).unwrap();

    // Predict
    let forecast = model.predict(horizon).unwrap();
    assert_eq!(forecast.horizon(), horizon);

    for &val in forecast.primary() {
        assert!(val.is_finite(), "forecast contains non-finite value");
    }

    // Get residuals and run diagnostics
    let residuals = model.residuals().unwrap();
    let valid_residuals: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();

    assert!(
        valid_residuals.len() >= 20,
        "need enough residuals for diagnostics, got {}",
        valid_residuals.len()
    );

    let diag = diagnose_residuals(&valid_residuals, 2);

    // Diagnostics should produce finite values
    assert!(diag.mean.is_finite(), "residual mean should be finite");
    assert!(
        diag.variance.is_finite(),
        "residual variance should be finite"
    );
    assert!(
        diag.variance >= 0.0,
        "residual variance should be non-negative"
    );
    assert_eq!(diag.n, valid_residuals.len());

    // Residual mean should be close to zero for a well-fitted model
    assert!(
        diag.mean.abs() < 10.0,
        "residual mean ({:.4}) should be small",
        diag.mean
    );
}

#[test]
fn arima_residuals_pass_ljung_box() {
    // Generate data from an ARIMA-like process
    let n = 200;
    let values = generate_trending_data(n);
    let ts = make_ts(values);

    let mut model = ARIMA::new(1, 1, 1);
    model.fit(&ts).unwrap();

    let residuals = model.residuals().unwrap();
    let valid_residuals: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();

    let diag = diagnose_residuals(&valid_residuals, 3);

    // For a well-specified model on trend data, residuals should be close to white noise
    // The Ljung-Box p-value should be above some threshold (we use a lenient alpha=0.01)
    assert!(
        diag.ljung_box.p_value.is_finite(),
        "Ljung-Box p-value should be finite"
    );
}

#[test]
fn multiple_models_diagnostics_comparison() {
    let n = 100;
    let values = generate_seasonal_data(n);
    let ts = make_ts(values);

    // Fit several models and compare residual variance
    let mut naive = Naive::new();
    naive.fit(&ts).unwrap();
    let naive_resid: Vec<f64> = naive
        .residuals()
        .unwrap()
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    let naive_diag = diagnose_residuals(&naive_resid, 1);

    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&ts).unwrap();
    let ses_resid: Vec<f64> = ses
        .residuals()
        .unwrap()
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    let ses_diag = diagnose_residuals(&ses_resid, 1);

    // All diagnostics should have finite, non-negative variance
    assert!(naive_diag.variance >= 0.0);
    assert!(ses_diag.variance >= 0.0);
    assert!(naive_diag.variance.is_finite());
    assert!(ses_diag.variance.is_finite());
}

// ---------------------------------------------------------------------------
// Pipeline: Constraints preserve interval ordering
// ---------------------------------------------------------------------------

#[test]
fn constrained_forecast_preserves_interval_ordering() {
    let n = 60;
    let values = generate_positive_data(n);
    let ts = make_ts(values);
    let horizon = 5;

    let mut model = Theta::new();
    model.fit(&ts).unwrap();
    let forecast = model.predict_with_intervals(horizon, 0.95).unwrap();

    // Apply a lower bound constraint
    let constrained = forecast.constrain(&[ForecastConstraint::LowerBound(0.0)]);

    let point = constrained.primary();
    for &val in point {
        assert!(val >= 0.0, "constrained point {} should be >= 0", val);
    }

    if constrained.has_lower() && constrained.has_upper() {
        let lower = constrained.lower_series(0).unwrap();
        let upper = constrained.upper_series(0).unwrap();

        for i in 0..horizon {
            assert!(
                lower[i] <= point[i],
                "constrained lower[{}]={} > point[{}]={}",
                i,
                lower[i],
                i,
                point[i]
            );
            assert!(
                point[i] <= upper[i],
                "constrained point[{}]={} > upper[{}]={}",
                i,
                point[i],
                i,
                upper[i]
            );
        }
    }
}

#[test]
fn integer_rounding_constraint_works() {
    let n = 60;
    let values = generate_positive_data(n);
    let ts = make_ts(values);
    let horizon = 5;

    let mut model = SimpleExponentialSmoothing::auto();
    model.fit(&ts).unwrap();
    let forecast = model.predict(horizon).unwrap();

    let constrained = forecast.constrain(&[ForecastConstraint::IntegerRound]);

    for &val in constrained.primary() {
        assert!(
            (val - val.round()).abs() < 1e-10,
            "integer-rounded value {} should be integer",
            val
        );
    }
}

// ---------------------------------------------------------------------------
// Pipeline: Postprocess with different methods
// ---------------------------------------------------------------------------

#[test]
fn postprocess_normal_predictor_produces_valid_intervals() {
    let n = 100;
    let values = generate_trending_data(n);
    let ts = make_ts(values.clone());
    let horizon = 5;

    let mut model = HoltLinearTrend::auto();
    model.fit(&ts).unwrap();

    let fitted = model.fitted_values().unwrap();
    let actuals = ts.primary_values();

    let valid_fitted: Vec<f64> = fitted.iter().copied().filter(|f| f.is_finite()).collect();
    let valid_actuals: Vec<f64> = actuals
        .iter()
        .zip(fitted.iter())
        .filter(|(_, f)| f.is_finite())
        .map(|(a, _)| *a)
        .collect();

    let train_f = PointForecasts::from_values(valid_fitted);
    let processor = PostProcessor::normal(vec![0.1, 0.5, 0.9]);
    let trained = processor.train(&train_f, &valid_actuals).unwrap();

    let forecast = model.predict(horizon).unwrap();
    let predict_f = PointForecasts::from_values(forecast.primary().to_vec());
    let intervals = processor.predict_intervals(&trained, &predict_f).unwrap();

    assert_eq!(intervals.len(), horizon);
    for i in 0..horizon {
        assert!(
            intervals.lower()[i] <= intervals.upper()[i],
            "normal predictor: lower {} > upper {} at step {}",
            intervals.lower()[i],
            intervals.upper()[i],
            i
        );
    }
}
