//! Convenience functions for common forecast combination patterns.
//!
//! This module provides high-level functions for comparing models, cross-validating
//! across a registry, and automatically building ensembles from top performers.

use crate::core::TimeSeries;
use crate::error::{ForecastError, Result};
use crate::models::ensemble::Ensemble;
use crate::models::traits::{Forecaster, ModelRegistry};
use crate::utils::cross_validation::CvFoldGenerator;
use crate::utils::metrics::{calculate_metrics, mae, rmse};
use std::fmt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// fit_all_and_compare
// ---------------------------------------------------------------------------

/// Result for a single model in a holdout comparison.
#[derive(Debug, Clone)]
pub struct ModelResult {
    /// Display name of the model.
    pub name: String,
    /// Mean Absolute Error on the holdout set.
    pub mae: f64,
    /// Root Mean Squared Error on the holdout set.
    pub rmse: f64,
    /// Mean Absolute Percentage Error on the holdout set.
    pub mape: f64,
    /// Whether the model was successfully fitted and predicted.
    pub fit_succeeded: bool,
}

/// Comparison of all models in a registry using holdout evaluation.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Per-model results, sorted by MAE ascending.
    pub results: Vec<ModelResult>,
}

impl fmt::Display for ModelComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<25} {:>10} {:>10} {:>10}  Status",
            "Model", "MAE", "RMSE", "MAPE"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;
        for r in &self.results {
            let status = if r.fit_succeeded { "OK" } else { "FAILED" };
            if r.fit_succeeded {
                writeln!(
                    f,
                    "{:<25} {:>10.4} {:>10.4} {:>9.2}%  {}",
                    r.name, r.mae, r.rmse, r.mape, status
                )?;
            } else {
                writeln!(
                    f,
                    "{:<25} {:>10} {:>10} {:>10}  {}",
                    r.name, "N/A", "N/A", "N/A", status
                )?;
            }
        }
        Ok(())
    }
}

/// Fit all models in a registry against a time series and compare using holdout.
///
/// The last `holdout` observations are reserved for evaluation. Each model is
/// fitted on the training portion and predicted over the holdout horizon.
/// Failed models are included in the results with `fit_succeeded = false` and
/// NaN metrics. Results are sorted by MAE ascending (failed models last).
///
/// When the `parallel` feature is enabled, models are evaluated in parallel.
///
/// # Arguments
/// * `registry` - A [`ModelRegistry`] containing model specifications to compare.
/// * `ts` - The full time series.
/// * `holdout` - Number of trailing observations to withhold for evaluation.
///
/// # Returns
/// A [`ModelComparison`] with one entry per model, sorted by MAE.
pub fn fit_all_and_compare(
    registry: &ModelRegistry,
    ts: &TimeSeries,
    holdout: usize,
) -> ModelComparison {
    if registry.is_empty() {
        return ModelComparison {
            results: Vec::new(),
        };
    }

    let n = ts.len();
    if holdout >= n || holdout == 0 {
        return all_models_failed(registry);
    }

    let split = n - holdout;
    let train = match ts.slice(0, split) {
        Ok(t) => t,
        Err(_) => return all_models_failed(registry),
    };
    let actual = &ts.primary_values()[split..];

    let specs: Vec<_> = registry.iter().collect();
    let evaluate = |spec: &&crate::models::traits::ModelSpec| -> ModelResult {
        evaluate_single_model(spec, &train, actual, holdout)
    };

    #[cfg(feature = "parallel")]
    let mut results: Vec<ModelResult> = specs.par_iter().map(evaluate).collect();

    #[cfg(not(feature = "parallel"))]
    let mut results: Vec<ModelResult> = specs.iter().map(evaluate).collect();

    results.sort_by(cmp_model_result);
    ModelComparison { results }
}

/// Mark every model in the registry as failed and return a sorted comparison.
fn all_models_failed(registry: &ModelRegistry) -> ModelComparison {
    let mut results: Vec<ModelResult> = registry
        .iter()
        .map(|spec| failed_model_result(&spec.name))
        .collect();
    results.sort_by(cmp_model_result);
    ModelComparison { results }
}

/// Construct a `ModelResult` with NaN metrics and `fit_succeeded = false`.
fn failed_model_result(name: &str) -> ModelResult {
    ModelResult {
        name: name.to_string(),
        mae: f64::NAN,
        rmse: f64::NAN,
        mape: f64::NAN,
        fit_succeeded: false,
    }
}

/// Evaluate a single model: fit on `train`, predict `horizon` steps, compare to `actual`.
fn evaluate_single_model(
    spec: &crate::models::traits::ModelSpec,
    train: &TimeSeries,
    actual: &[f64],
    horizon: usize,
) -> ModelResult {
    let mut model = spec.create();
    let name = spec.name.to_string();

    if model.fit(train).is_err() {
        return failed_model_result(&name);
    }
    let forecast = match model.predict(horizon) {
        Ok(f) => f,
        Err(_) => return failed_model_result(&name),
    };

    let predicted = forecast.primary();
    match calculate_metrics(actual, predicted, None) {
        Ok(metrics) => ModelResult {
            name,
            mae: metrics.mae,
            rmse: metrics.rmse,
            mape: metrics.mape.unwrap_or(f64::NAN),
            fit_succeeded: true,
        },
        Err(_) => ModelResult {
            name,
            mae: mae(actual, predicted),
            rmse: rmse(actual, predicted),
            mape: f64::NAN,
            fit_succeeded: true,
        },
    }
}

/// Sort helper: succeeded models first (by MAE ascending), then failed models.
fn cmp_model_result(a: &ModelResult, b: &ModelResult) -> std::cmp::Ordering {
    match (a.fit_succeeded, b.fit_succeeded) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.name.cmp(&b.name),
        (true, true) => a
            .mae
            .partial_cmp(&b.mae)
            .unwrap_or(std::cmp::Ordering::Equal),
    }
}

// ---------------------------------------------------------------------------
// cross_validate_all
// ---------------------------------------------------------------------------

/// Result for a single model in a cross-validation comparison.
#[derive(Debug, Clone)]
pub struct CVModelResult {
    /// Display name of the model.
    pub name: String,
    /// Mean MAE across folds.
    pub mean_mae: f64,
    /// Mean RMSE across folds.
    pub mean_rmse: f64,
    /// Standard deviation of MAE across folds.
    pub std_mae: f64,
}

/// Comparison of all models in a registry using cross-validation.
#[derive(Debug, Clone)]
pub struct CVComparison {
    /// Per-model results, sorted by mean MAE ascending.
    pub results: Vec<CVModelResult>,
}

impl fmt::Display for CVComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<25} {:>10} {:>10} {:>10}",
            "Model", "Mean MAE", "Mean RMSE", "Std MAE"
        )?;
        writeln!(f, "{}", "-".repeat(58))?;
        for r in &self.results {
            if r.mean_mae.is_finite() {
                writeln!(
                    f,
                    "{:<25} {:>10.4} {:>10.4} {:>10.4}",
                    r.name, r.mean_mae, r.mean_rmse, r.std_mae
                )?;
            } else {
                writeln!(
                    f,
                    "{:<25} {:>10} {:>10} {:>10}",
                    r.name, "N/A", "N/A", "N/A"
                )?;
            }
        }
        Ok(())
    }
}

/// Cross-validate all models in a registry.
///
/// Uses an expanding-window strategy with the given number of folds and
/// forecast horizon. The initial window is chosen so that the requested
/// number of folds fit within the series. Failed models are included with
/// NaN metrics. Results are sorted by mean MAE ascending.
///
/// When the `parallel` feature is enabled, individual model evaluations run
/// in parallel.
///
/// # Arguments
/// * `registry` - A [`ModelRegistry`] containing model specifications.
/// * `ts` - The time series to cross-validate on.
/// * `n_folds` - Desired number of cross-validation folds.
/// * `horizon` - Forecast horizon per fold.
///
/// # Returns
/// A [`CVComparison`] with one entry per model, sorted by mean MAE.
pub fn cross_validate_all(
    registry: &ModelRegistry,
    ts: &TimeSeries,
    n_folds: usize,
    horizon: usize,
) -> CVComparison {
    if registry.is_empty() {
        return CVComparison {
            results: Vec::new(),
        };
    }

    let (initial_window, step_size) = compute_cv_fold_params(ts.len(), n_folds, horizon);

    let generator = CvFoldGenerator::new()
        .initial_window(initial_window)
        .horizon(horizon)
        .step_size(step_size);
    let folds = generator.generate(ts.len());

    let specs: Vec<_> = registry.iter().collect();
    let evaluate = |spec: &&crate::models::traits::ModelSpec| -> CVModelResult {
        cv_single_model(spec, ts, &folds)
    };

    #[cfg(feature = "parallel")]
    let mut results: Vec<CVModelResult> = specs.par_iter().map(evaluate).collect();

    #[cfg(not(feature = "parallel"))]
    let mut results: Vec<CVModelResult> = specs.iter().map(evaluate).collect();

    results.sort_by(cmp_cv_result);
    CVComparison { results }
}

/// Compute the initial window size and step size for cross-validation folds.
///
/// Given the series length, desired number of folds, and forecast horizon,
/// returns `(initial_window, step_size)` that fit approximately `n_folds`
/// folds within the series.
fn compute_cv_fold_params(series_len: usize, n_folds: usize, horizon: usize) -> (usize, usize) {
    if n_folds == 0 || horizon == 0 || series_len < horizon + 2 {
        return (series_len, 1); // Will produce 0 folds
    }

    let needed_for_folds = n_folds * horizon;
    if needed_for_folds >= series_len {
        let iw = series_len.saturating_sub(n_folds + horizon - 1).max(2);
        (iw, 1)
    } else {
        let iw = series_len.saturating_sub(needed_for_folds).max(2);
        (iw, horizon)
    }
}

/// Construct a `CVModelResult` with NaN metrics (used when no folds succeed).
fn failed_cv_result(name: String) -> CVModelResult {
    CVModelResult {
        name,
        mean_mae: f64::NAN,
        mean_rmse: f64::NAN,
        std_mae: f64::NAN,
    }
}

/// Run cross-validation for a single model across all folds.
fn cv_single_model(
    spec: &crate::models::traits::ModelSpec,
    ts: &TimeSeries,
    folds: &[crate::utils::cross_validation::Fold],
) -> CVModelResult {
    let name = spec.name.to_string();

    if folds.is_empty() {
        return failed_cv_result(name);
    }

    let (fold_maes, fold_rmses) = collect_fold_metrics(spec, ts, folds);

    if fold_maes.is_empty() {
        return failed_cv_result(name);
    }

    aggregate_fold_metrics(name, &fold_maes, &fold_rmses)
}

/// Evaluate a model on each fold and collect per-fold MAE and RMSE values.
///
/// Folds where the model fails to fit/predict or produces non-finite metrics
/// are silently skipped.
fn collect_fold_metrics(
    spec: &crate::models::traits::ModelSpec,
    ts: &TimeSeries,
    folds: &[crate::utils::cross_validation::Fold],
) -> (Vec<f64>, Vec<f64>) {
    let mut fold_maes = Vec::with_capacity(folds.len());
    let mut fold_rmses = Vec::with_capacity(folds.len());

    for fold in folds {
        let train = match ts.slice(fold.train_start, fold.train_end) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let mut model = spec.create();
        if model.fit(&train).is_err() {
            continue;
        }
        let forecast = match model.predict(fold.test_size()) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let actual_slice = &ts.primary_values()[fold.test_start..fold.test_end];
        let predicted = forecast.primary();
        let fold_mae = mae(actual_slice, predicted);
        let fold_rmse = rmse(actual_slice, predicted);
        if fold_mae.is_finite() && fold_rmse.is_finite() {
            fold_maes.push(fold_mae);
            fold_rmses.push(fold_rmse);
        }
    }

    (fold_maes, fold_rmses)
}

/// Aggregate per-fold MAE and RMSE into mean and standard deviation.
///
/// Assumes `fold_maes` is non-empty.
fn aggregate_fold_metrics(name: String, fold_maes: &[f64], fold_rmses: &[f64]) -> CVModelResult {
    let n = fold_maes.len() as f64;
    let mean_mae = fold_maes.iter().sum::<f64>() / n;
    let mean_rmse = fold_rmses.iter().sum::<f64>() / n;
    let std_mae = if fold_maes.len() > 1 {
        let variance = fold_maes
            .iter()
            .map(|v| (v - mean_mae).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    CVModelResult {
        name,
        mean_mae,
        mean_rmse,
        std_mae,
    }
}

/// Sort helper: finite-metric models first (by mean MAE ascending), then NaN models.
fn cmp_cv_result(a: &CVModelResult, b: &CVModelResult) -> std::cmp::Ordering {
    let fa = a.mean_mae.is_finite();
    let fb = b.mean_mae.is_finite();
    match (fa, fb) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.name.cmp(&b.name),
        (true, true) => a
            .mean_mae
            .partial_cmp(&b.mean_mae)
            .unwrap_or(std::cmp::Ordering::Equal),
    }
}

// ---------------------------------------------------------------------------
// ensemble_best_k
// ---------------------------------------------------------------------------

/// Auto-select the top-k models by holdout performance and combine into an ensemble.
///
/// Evaluates all models in the registry on a holdout set, picks the `k` best by
/// MAE, then builds an [`Ensemble`] with those models (using mean combination).
/// The ensemble is fitted on the **full** time series before being returned.
///
/// # Arguments
/// * `registry` - A [`ModelRegistry`] containing model specifications.
/// * `ts` - The full time series.
/// * `k` - Number of top models to include in the ensemble.
/// * `holdout` - Number of trailing observations used for model selection.
///
/// # Errors
/// Returns [`ForecastError::EmptyData`] if no models succeeded or the registry is empty.
///
/// # Returns
/// A fitted `Box<dyn Forecaster>` backed by an [`Ensemble`] of the top-k models.
pub fn ensemble_best_k(
    registry: &ModelRegistry,
    ts: &TimeSeries,
    k: usize,
    holdout: usize,
) -> Result<Box<dyn Forecaster>> {
    if registry.is_empty() || k == 0 {
        return Err(ForecastError::EmptyData);
    }

    let comparison = fit_all_and_compare(registry, ts, holdout);

    // Collect names of top-k succeeded models.
    let top_names: Vec<String> = comparison
        .results
        .iter()
        .filter(|r| r.fit_succeeded)
        .take(k)
        .map(|r| r.name.clone())
        .collect();

    if top_names.is_empty() {
        return Err(ForecastError::EmptyData);
    }

    // Create fresh instances of those models for the ensemble.
    let models: Vec<Box<dyn Forecaster>> = registry
        .iter()
        .filter(|spec| top_names.contains(&spec.name.to_string()))
        .map(|spec| spec.create())
        .collect();

    if models.is_empty() {
        return Err(ForecastError::EmptyData);
    }

    let mut ensemble = Ensemble::new(models);
    ensemble.fit(ts)?;

    Ok(Box::new(ensemble))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::models::baseline::{Naive, RandomWalkWithDrift, WindowAverage};
    use crate::models::{ModelRegistry, ModelSpec};
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }

    fn make_test_series(n: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    fn make_registry() -> ModelRegistry {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RWD",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));
        registry.register(ModelSpec::with_period(
            "WindowAvg5",
            |p| Box::new(WindowAverage::new(p)),
            5,
            false,
        ));
        registry
    }

    // -----------------------------------------------------------------------
    // fit_all_and_compare
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_all_and_compare_basic() {
        let registry = make_registry();
        let ts = make_test_series(50);
        let comparison = fit_all_and_compare(&registry, &ts, 10);

        // All three models should succeed.
        assert_eq!(comparison.results.len(), 3);
        for r in &comparison.results {
            assert!(r.fit_succeeded, "model {} should succeed", r.name);
            assert!(r.mae.is_finite(), "model {} MAE should be finite", r.name);
            assert!(r.rmse.is_finite(), "model {} RMSE should be finite", r.name);
        }

        // Results should be sorted by MAE ascending.
        for w in comparison.results.windows(2) {
            assert!(
                w[0].mae <= w[1].mae,
                "results should be sorted by MAE: {} ({}) <= {} ({})",
                w[0].name,
                w[0].mae,
                w[1].name,
                w[1].mae
            );
        }
    }

    #[test]
    fn test_fit_all_and_compare_display() {
        let registry = make_registry();
        let ts = make_test_series(50);
        let comparison = fit_all_and_compare(&registry, &ts, 10);
        let display = format!("{}", comparison);
        assert!(display.contains("Model"));
        assert!(display.contains("MAE"));
        assert!(display.contains("RMSE"));
    }

    #[test]
    fn test_fit_all_and_compare_empty_registry() {
        let registry = ModelRegistry::new();
        let ts = make_test_series(50);
        let comparison = fit_all_and_compare(&registry, &ts, 10);
        assert!(comparison.results.is_empty());
    }

    #[test]
    fn test_fit_all_and_compare_bad_holdout() {
        let registry = make_registry();
        let ts = make_test_series(50);
        // holdout >= series length, all models should be marked failed
        let comparison = fit_all_and_compare(&registry, &ts, 50);
        assert_eq!(comparison.results.len(), 3);
        for r in &comparison.results {
            assert!(!r.fit_succeeded);
        }
    }

    // -----------------------------------------------------------------------
    // cross_validate_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_validate_all_basic() {
        let registry = make_registry();
        let ts = make_test_series(60);
        let cv = cross_validate_all(&registry, &ts, 3, 5);

        // Should produce a result for each model.
        assert_eq!(cv.results.len(), 3);
        for r in &cv.results {
            assert!(
                r.mean_mae.is_finite(),
                "model {} mean_mae should be finite",
                r.name
            );
            assert!(
                r.mean_rmse.is_finite(),
                "model {} mean_rmse should be finite",
                r.name
            );
            assert!(
                r.std_mae.is_finite(),
                "model {} std_mae should be finite",
                r.name
            );
        }

        // Sorted by mean_mae ascending.
        for w in cv.results.windows(2) {
            assert!(
                w[0].mean_mae <= w[1].mean_mae,
                "results should be sorted by mean MAE"
            );
        }
    }

    #[test]
    fn test_cross_validate_all_display() {
        let registry = make_registry();
        let ts = make_test_series(60);
        let cv = cross_validate_all(&registry, &ts, 3, 5);
        let display = format!("{}", cv);
        assert!(display.contains("Model"));
        assert!(display.contains("Mean MAE"));
    }

    #[test]
    fn test_cross_validate_all_empty_registry() {
        let registry = ModelRegistry::new();
        let ts = make_test_series(60);
        let cv = cross_validate_all(&registry, &ts, 3, 5);
        assert!(cv.results.is_empty());
    }

    // -----------------------------------------------------------------------
    // ensemble_best_k
    // -----------------------------------------------------------------------

    #[test]
    fn test_ensemble_best_k_basic() {
        let registry = make_registry();
        let ts = make_test_series(50);
        let ensemble = ensemble_best_k(&registry, &ts, 2, 10);
        assert!(ensemble.is_ok(), "ensemble_best_k should succeed");

        let model = ensemble.unwrap();
        assert!(model.is_fitted());
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        // Predictions should be finite
        for &v in forecast.primary() {
            assert!(v.is_finite(), "forecast value should be finite");
        }
    }

    #[test]
    fn test_ensemble_best_k_single() {
        let registry = make_registry();
        let ts = make_test_series(50);
        let ensemble = ensemble_best_k(&registry, &ts, 1, 10);
        assert!(ensemble.is_ok());
        let model = ensemble.unwrap();
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn test_ensemble_best_k_empty_registry() {
        let registry = ModelRegistry::new();
        let ts = make_test_series(50);
        let result = ensemble_best_k(&registry, &ts, 2, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_best_k_k_zero() {
        let registry = make_registry();
        let ts = make_test_series(50);
        let result = ensemble_best_k(&registry, &ts, 0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_best_k_k_larger_than_registry() {
        let registry = make_registry();
        let ts = make_test_series(50);
        // k=10 but only 3 models in registry; should use all 3
        let ensemble = ensemble_best_k(&registry, &ts, 10, 10);
        assert!(ensemble.is_ok());
        let model = ensemble.unwrap();
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }
}
