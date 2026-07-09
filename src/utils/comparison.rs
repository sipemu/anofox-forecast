//! Model comparison utilities for evaluating multiple forecasters on the same data.
//!
//! This module provides a convenient API for comparing forecasting models:
//! - In-sample accuracy metrics from fitted values
//! - Optional cross-validation metrics
//! - Timing information for each model fit
//! - Results sorted by in-sample RMSE
//!
//! # Example
//!
//! ```
//! use anofox_forecast::models::{BoxedForecaster, ModelRegistry, ModelSpec};
//! use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift};
//! use anofox_forecast::utils::comparison::{compare_models, ComparisonConfig};
//! # use anofox_forecast::core::TimeSeries;
//! # use chrono::{TimeZone, Utc};
//! # let ts = TimeSeries::univariate(
//! #     (0..30).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
//! #     (1..=30).map(|i| i as f64).collect(),
//! # ).unwrap();
//!
//! let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
//!     ("Naive", Box::new(|| Box::new(Naive::new()))),
//!     ("RWD", Box::new(|| Box::new(RandomWalkWithDrift::new()))),
//! ];
//!
//! let config = ComparisonConfig::default();
//! let results = compare_models(&factories, &ts, &config).unwrap();
//!
//! for r in &results {
//!     println!("{}: RMSE={:.4}", r.model_name, r.in_sample.rmse);
//! }
//! ```

use crate::core::TimeSeries;
use crate::error::Result;
use crate::models::{BoxedForecaster, ModelRegistry};
use crate::utils::cross_validation::CVConfig;
use crate::utils::metrics::{calculate_metrics, AccuracyMetrics};
use std::fmt;
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of comparing a single model against a time series.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Name of the model.
    pub model_name: String,
    /// In-sample accuracy metrics (fitted values vs actuals).
    pub in_sample: AccuracyMetrics,
    /// Cross-validation metrics, if CV was enabled.
    pub cv_metrics: Option<AccuracyMetrics>,
    /// Time taken to fit the model, in microseconds.
    pub fit_time_us: u64,
}

/// Configuration for model comparison.
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Whether to run cross-validation for each model.
    pub run_cv: bool,
    /// Cross-validation configuration (used only if `run_cv` is true).
    pub cv_config: CVConfig,
    /// Forecast horizon for prediction.
    pub horizon: usize,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            run_cv: false,
            cv_config: CVConfig::default(),
            horizon: 1,
        }
    }
}

impl ComparisonConfig {
    /// Create a new comparison config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable cross-validation with the given config.
    pub fn with_cv(mut self, cv_config: CVConfig) -> Self {
        self.run_cv = true;
        self.cv_config = cv_config;
        self
    }

    /// Set the forecast horizon.
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    /// Enable or disable cross-validation.
    pub fn run_cv(mut self, run: bool) -> Self {
        self.run_cv = run;
        self
    }
}

/// Wrapper around `Vec<ComparisonResult>` that provides a formatted table display.
pub struct ComparisonTable(pub Vec<ComparisonResult>);

impl fmt::Display for ComparisonTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "(no results)");
        }

        let has_cv = self.0.iter().any(|r| r.cv_metrics.is_some());

        // Header
        if has_cv {
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>10} {:>10} {:>12}",
                "Model", "RMSE", "MAE", "SMAPE", "CV RMSE", "Fit (us)"
            )?;
            writeln!(f, "{}", "-".repeat(74))?;
        } else {
            writeln!(
                f,
                "{:<20} {:>10} {:>10} {:>10} {:>12}",
                "Model", "RMSE", "MAE", "SMAPE", "Fit (us)"
            )?;
            writeln!(f, "{}", "-".repeat(64))?;
        }

        // Rows
        for r in &self.0 {
            if has_cv {
                let cv_rmse = r
                    .cv_metrics
                    .as_ref()
                    .map(|m| format!("{:.4}", m.rmse))
                    .unwrap_or_else(|| "N/A".to_string());
                writeln!(
                    f,
                    "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>12}",
                    r.model_name,
                    r.in_sample.rmse,
                    r.in_sample.mae,
                    r.in_sample.smape,
                    cv_rmse,
                    r.fit_time_us
                )?;
            } else {
                writeln!(
                    f,
                    "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>12}",
                    r.model_name,
                    r.in_sample.rmse,
                    r.in_sample.mae,
                    r.in_sample.smape,
                    r.fit_time_us
                )?;
            }
        }

        Ok(())
    }
}

/// Compute in-sample accuracy metrics from a fitted model's fitted values and the actuals.
///
/// Filters out non-finite pairs (some models produce NaN for initial fitted values).
fn compute_in_sample(actual: &[f64], fitted: &[f64]) -> Option<AccuracyMetrics> {
    let (a, p): (Vec<f64>, Vec<f64>) = actual
        .iter()
        .zip(fitted.iter())
        .filter(|(av, fv)| av.is_finite() && fv.is_finite())
        .map(|(&av, &fv)| (av, fv))
        .unzip();
    if a.is_empty() {
        return None;
    }
    calculate_metrics(&a, &p, None).ok()
}

/// Run cross-validation for a single model factory using the fold generator.
///
/// Manually iterates over folds (since `BoxedForecaster` does not implement the `Forecaster`
/// trait required by the generic `cross_validate` function).
fn run_cv_for_factory(
    factory: &dyn Fn() -> BoxedForecaster,
    series: &TimeSeries,
    cv_config: &CVConfig,
) -> Option<AccuracyMetrics> {
    let generator = cv_config.to_fold_generator();
    let folds = match generator.generate(series.len()) {
        Ok(f) if !f.is_empty() => f,
        _ => return None,
    };

    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();

    for fold in &folds {
        // Create training subseries
        let train = match series.slice(fold.train_start, fold.train_end) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Fit model on training data
        let mut model = factory();
        if model.fit(&train).is_err() {
            continue;
        }

        // Predict for the test horizon
        let test_len = fold.test_end - fold.test_start;
        let forecast = match model.predict(test_len) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Collect actual test values and predictions
        let test_actual = &series.primary_values()[fold.test_start..fold.test_end];
        let predicted = forecast.primary();

        let len = test_actual.len().min(predicted.len());
        all_actual.extend_from_slice(&test_actual[..len]);
        all_predicted.extend_from_slice(&predicted[..len]);
    }

    if all_actual.is_empty() {
        return None;
    }

    calculate_metrics(&all_actual, &all_predicted, cv_config.seasonal_period).ok()
}

/// Compare multiple models on the same time series.
///
/// For each model factory, creates an instance, fits it, computes in-sample metrics,
/// and optionally runs cross-validation. Results are sorted by in-sample RMSE (ascending).
///
/// Models that fail to fit are silently skipped. This allows robust comparison even when
/// some models cannot handle the data.
///
/// # Arguments
/// * `factories` - Slice of `(name, factory_fn)` pairs. Each factory creates a new
///   `BoxedForecaster` instance.
/// * `series` - The time series to compare models on.
/// * `config` - Comparison configuration (CV, horizon, etc.).
///
/// # Returns
/// A sorted `Vec<ComparisonResult>` with the best model (lowest in-sample RMSE) first.
///
/// # Example
///
/// ```
/// use anofox_forecast::models::BoxedForecaster;
/// use anofox_forecast::models::baseline::Naive;
/// use anofox_forecast::utils::comparison::{compare_models, ComparisonConfig};
/// # use anofox_forecast::core::TimeSeries;
/// # use chrono::{TimeZone, Utc};
/// # let ts = TimeSeries::univariate(
/// #     (0..30).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
/// #     (1..=30).map(|i| i as f64).collect(),
/// # ).unwrap();
///
/// let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
///     ("Naive", Box::new(|| Box::new(Naive::new()))),
/// ];
/// let results = compare_models(&factories, &ts, &ComparisonConfig::default()).unwrap();
/// assert_eq!(results.len(), 1);
/// ```
pub fn compare_models(
    factories: &[(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)],
    series: &TimeSeries,
    config: &ComparisonConfig,
) -> Result<Vec<ComparisonResult>> {
    let actual = series.primary_values();

    let process = |(name, factory): &(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)| -> Option<ComparisonResult> {
        let mut model = factory();

        let start = Instant::now();
        if model.fit(series).is_err() {
            return None;
        }
        let fit_time_us = start.elapsed().as_micros() as u64;

        let in_sample = compute_in_sample(actual, model.fitted_values()?)?;

        let cv_metrics = if config.run_cv {
            run_cv_for_factory(factory.as_ref(), series, &config.cv_config)
        } else {
            None
        };

        Some(ComparisonResult {
            model_name: name.to_string(),
            in_sample,
            cv_metrics,
            fit_time_us,
        })
    };

    #[cfg(feature = "parallel")]
    let mut results: Vec<ComparisonResult> = factories.par_iter().filter_map(process).collect();

    #[cfg(not(feature = "parallel"))]
    let mut results: Vec<ComparisonResult> = factories.iter().filter_map(process).collect();

    results.sort_by(|a, b| {
        a.in_sample
            .rmse
            .partial_cmp(&b.in_sample.rmse)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Compare all models in a registry on the same time series.
///
/// This is a convenience wrapper that uses the model specs from a [`ModelRegistry`]
/// to create factories and delegate to the core comparison logic.
///
/// # Arguments
/// * `registry` - A model registry containing model specifications.
/// * `series` - The time series to compare models on.
/// * `config` - Comparison configuration.
///
/// # Returns
/// A sorted `Vec<ComparisonResult>` with the best model first.
///
/// # Example
///
/// ```
/// use anofox_forecast::models::{ModelRegistry, ModelSpec};
/// use anofox_forecast::models::baseline::Naive;
/// use anofox_forecast::utils::comparison::{compare_registry, ComparisonConfig};
/// # use anofox_forecast::core::TimeSeries;
/// # use chrono::{TimeZone, Utc};
/// # let ts = TimeSeries::univariate(
/// #     (0..30).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
/// #     (1..=30).map(|i| i as f64).collect(),
/// # ).unwrap();
///
/// let mut registry = ModelRegistry::new();
/// registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
///
/// let results = compare_registry(&registry, &ts, &ComparisonConfig::default()).unwrap();
/// assert_eq!(results.len(), 1);
/// ```
pub fn compare_registry(
    registry: &ModelRegistry,
    series: &TimeSeries,
    config: &ComparisonConfig,
) -> Result<Vec<ComparisonResult>> {
    let actual = series.primary_values();
    let specs: Vec<_> = registry.iter().collect();

    let process = |spec: &&crate::models::ModelSpec| -> Option<ComparisonResult> {
        let mut model = spec.create();

        let start = Instant::now();
        if model.fit(series).is_err() {
            return None;
        }
        let fit_time_us = start.elapsed().as_micros() as u64;

        let in_sample = compute_in_sample(actual, model.fitted_values()?)?;

        let cv_metrics = if config.run_cv {
            let factory = || spec.create();
            run_cv_for_factory(&factory, series, &config.cv_config)
        } else {
            None
        };

        Some(ComparisonResult {
            model_name: spec.name.to_string(),
            in_sample,
            cv_metrics,
            fit_time_us,
        })
    };

    #[cfg(feature = "parallel")]
    let mut results: Vec<ComparisonResult> = specs.par_iter().filter_map(process).collect();

    #[cfg(not(feature = "parallel"))]
    let mut results: Vec<ComparisonResult> = specs.iter().filter_map(process).collect();

    results.sort_by(|a, b| {
        a.in_sample
            .rmse
            .partial_cmp(&b.in_sample.rmse)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
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

    // -----------------------------------------------------------------------
    // compare_models: basic usage with 2-3 baseline models
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_two_models() {
        let ts = make_test_series(30);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
            ("Naive", Box::new(|| Box::new(Naive::new()))),
            ("RWD", Box::new(|| Box::new(RandomWalkWithDrift::new()))),
        ];

        let config = ComparisonConfig::default();
        let results = compare_models(&factories, &ts, &config).unwrap();

        assert_eq!(results.len(), 2);
        // Results should be sorted by RMSE ascending
        assert!(results[0].in_sample.rmse <= results[1].in_sample.rmse);
        // Both should have valid metrics
        for r in &results {
            assert!(r.in_sample.rmse >= 0.0);
            assert!(r.in_sample.mae >= 0.0);
        }
    }

    #[test]
    fn test_compare_three_models() {
        let ts = make_test_series(30);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
            ("Naive", Box::new(|| Box::new(Naive::new()))),
            ("RWD", Box::new(|| Box::new(RandomWalkWithDrift::new()))),
            ("WindowAvg", Box::new(|| Box::new(WindowAverage::new(5)))),
        ];

        let config = ComparisonConfig::default();
        let results = compare_models(&factories, &ts, &config).unwrap();

        assert_eq!(results.len(), 3);
        // Verify sort order
        for w in results.windows(2) {
            assert!(w[0].in_sample.rmse <= w[1].in_sample.rmse);
        }
    }

    // -----------------------------------------------------------------------
    // compare_models: with CV enabled
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_with_cv() {
        let ts = make_test_series(50);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
            ("Naive", Box::new(|| Box::new(Naive::new()))),
            ("RWD", Box::new(|| Box::new(RandomWalkWithDrift::new()))),
        ];

        let cv_config = CVConfig::expanding(20, 1).with_step_size(5);
        let config = ComparisonConfig::new().with_cv(cv_config);

        let results = compare_models(&factories, &ts, &config).unwrap();
        assert_eq!(results.len(), 2);

        // Both should have CV metrics
        for r in &results {
            assert!(
                r.cv_metrics.is_some(),
                "model {} should have CV metrics",
                r.model_name
            );
            let cv = r.cv_metrics.as_ref().unwrap();
            assert!(cv.rmse >= 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // compare_registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_via_registry() {
        let ts = make_test_series(30);

        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RWD",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));

        let config = ComparisonConfig::default();
        let results = compare_registry(&registry, &ts, &config).unwrap();

        assert_eq!(results.len(), 2);
        // Should be sorted by RMSE
        assert!(results[0].in_sample.rmse <= results[1].in_sample.rmse);
    }

    // -----------------------------------------------------------------------
    // Empty factories list
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_empty_factories() {
        let ts = make_test_series(30);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![];

        let config = ComparisonConfig::default();
        let results = compare_models(&factories, &ts, &config).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_compare_empty_registry() {
        let ts = make_test_series(30);
        let registry = ModelRegistry::new();

        let config = ComparisonConfig::default();
        let results = compare_registry(&registry, &ts, &config).unwrap();

        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Display output
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_table_without_cv() {
        let ts = make_test_series(30);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
            ("Naive", Box::new(|| Box::new(Naive::new()))),
            ("RWD", Box::new(|| Box::new(RandomWalkWithDrift::new()))),
        ];

        let config = ComparisonConfig::default();
        let results = compare_models(&factories, &ts, &config).unwrap();
        let table = ComparisonTable(results);
        let output = format!("{}", table);

        assert!(output.contains("Model"));
        assert!(output.contains("RMSE"));
        assert!(output.contains("MAE"));
        assert!(output.contains("SMAPE"));
        assert!(output.contains("Naive"));
        assert!(output.contains("RWD"));
        // Should NOT contain CV RMSE column when CV is disabled
        assert!(!output.contains("CV RMSE"));
    }

    #[test]
    fn test_display_table_with_cv() {
        let ts = make_test_series(50);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> =
            vec![("Naive", Box::new(|| Box::new(Naive::new())))];

        let cv_config = CVConfig::expanding(20, 1).with_step_size(5);
        let config = ComparisonConfig::new().with_cv(cv_config);
        let results = compare_models(&factories, &ts, &config).unwrap();
        let table = ComparisonTable(results);
        let output = format!("{}", table);

        assert!(output.contains("CV RMSE"));
    }

    #[test]
    fn test_display_empty_results() {
        let table = ComparisonTable(vec![]);
        let output = format!("{}", table);
        assert_eq!(output, "(no results)");
    }

    // -----------------------------------------------------------------------
    // Config builder
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = ComparisonConfig::default();
        assert!(!config.run_cv);
        assert_eq!(config.horizon, 1);
    }

    #[test]
    fn test_config_builder() {
        let cv_config = CVConfig::expanding(15, 3).with_step_size(2);
        let config = ComparisonConfig::new().with_cv(cv_config).with_horizon(5);

        assert!(config.run_cv);
        assert_eq!(config.horizon, 5);
        assert_eq!(config.cv_config.min_initial_window, 15);
        assert_eq!(config.cv_config.horizon, 3);
        assert_eq!(config.cv_config.step_size, 2);
    }

    #[test]
    fn test_config_run_cv_toggle() {
        let config = ComparisonConfig::new().run_cv(true).run_cv(false);
        assert!(!config.run_cv);
    }

    // -----------------------------------------------------------------------
    // Model failure is gracefully skipped
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_skips_failed_models() {
        // Use a very short series so SeasonalNaive(12) fails but Naive succeeds
        let ts = make_test_series(10);
        let factories: Vec<(&str, Box<dyn Fn() -> BoxedForecaster + Send + Sync>)> = vec![
            ("Naive", Box::new(|| Box::new(Naive::new()))),
            (
                "SeasonalNaive",
                Box::new(|| Box::new(crate::models::baseline::SeasonalNaive::new(12))),
            ),
        ];

        let config = ComparisonConfig::default();
        let results = compare_models(&factories, &ts, &config).unwrap();

        // Only Naive should succeed; SeasonalNaive(12) needs >= 12 points
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model_name, "Naive");
    }
}
