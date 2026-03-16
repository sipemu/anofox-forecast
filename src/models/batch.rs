//! Batch forecasting operations for fitting and predicting across multiple series or models.
//!
//! This module provides functions for:
//! - Fitting one model configuration to many time series ([`fit_many`])
//! - Generating predictions from multiple fitted models ([`predict_many`])
//! - Combined fit-and-predict across many series ([`fit_predict_many`])
//! - Model comparison by fitting all models in a registry to one series ([`fit_registry`])
//!
//! All batch operations support optional parallelism via the `parallel` feature flag (rayon).

use crate::core::{Forecast, TimeSeries};
use crate::error::Result;
use crate::models::traits::{Forecaster, ModelRegistry};
use crate::orchestration::pipeline::PipelineResult;
use crate::utils::metrics::{calculate_metrics, AccuracyMetrics};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of a batch fit operation, pairing a fitted model with its series index.
#[derive(Debug)]
pub struct BatchResult<M> {
    /// The fitted model (or unfitted if an error occurred).
    pub model: M,
    /// Index of the series this model was fit to.
    pub series_index: usize,
    /// Error message if fitting failed.
    pub error: Option<String>,
}

/// Result of a combined fit-and-predict operation for a single series.
#[derive(Debug)]
pub struct FitPredictResult {
    /// The forecast, if fitting and prediction both succeeded.
    pub forecast: Option<Forecast>,
    /// Name of the model used.
    pub model_name: String,
    /// Index of the series that was forecast.
    pub series_index: usize,
    /// Error message if fitting or prediction failed.
    pub error: Option<String>,
}

/// Result of fitting a single model from a registry to a series.
#[derive(Debug)]
pub struct RegistryResult {
    /// Name of the model.
    pub model_name: String,
    /// The forecast, if the model was successfully fit and predicted.
    pub forecast: Option<Forecast>,
    /// In-sample accuracy metrics, if fitted values were available.
    pub metrics: Option<AccuracyMetrics>,
    /// Error message if fitting or prediction failed.
    pub error: Option<String>,
}

/// Fit the same model configuration to many time series.
///
/// Creates a fresh model instance via `factory` for each series and fits it.
/// With the `parallel` feature enabled, series are processed in parallel using rayon.
///
/// # Arguments
/// * `factory` - A function that creates a new model instance.
/// * `series` - Slice of time series references to fit against.
///
/// # Returns
/// A `Vec<Result<M>>` with one entry per input series, in the same order.
///
/// # Example
/// ```
/// use anofox_forecast::models::batch::fit_many;
/// use anofox_forecast::models::baseline::Naive;
/// # use anofox_forecast::core::TimeSeries;
/// # use chrono::{TimeZone, Utc};
/// # let ts1 = TimeSeries::univariate(
/// #     (0..20).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
/// #     (1..=20).map(|i| i as f64).collect(),
/// # ).unwrap();
/// # let ts2 = ts1.clone();
///
/// let all_series = vec![&ts1, &ts2];
/// let results = fit_many(Naive::new, &all_series);
/// assert_eq!(results.len(), 2);
/// assert!(results[0].is_ok());
/// ```
pub fn fit_many<F, M>(factory: F, series: &[&TimeSeries]) -> Vec<Result<M>>
where
    F: Fn() -> M + Send + Sync,
    M: Forecaster + Send,
{
    #[cfg(feature = "parallel")]
    {
        series
            .par_iter()
            .map(|ts| {
                let mut model = factory();
                model.fit(ts)?;
                Ok(model)
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        series
            .iter()
            .map(|ts| {
                let mut model = factory();
                model.fit(ts)?;
                Ok(model)
            })
            .collect()
    }
}

/// Generate predictions from multiple fitted models.
///
/// Calls `predict(horizon)` on each model. Models that have not been fitted will
/// produce an error result.
///
/// # Arguments
/// * `models` - Slice of fitted models to predict from.
/// * `horizon` - Number of future time steps to forecast.
///
/// # Returns
/// A `Vec<Result<Forecast>>` with one entry per model, in the same order.
///
/// # Example
/// ```
/// use anofox_forecast::models::batch::{fit_many, predict_many};
/// use anofox_forecast::models::baseline::Naive;
/// # use anofox_forecast::core::TimeSeries;
/// # use chrono::{TimeZone, Utc};
/// # let ts = TimeSeries::univariate(
/// #     (0..20).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
/// #     (1..=20).map(|i| i as f64).collect(),
/// # ).unwrap();
///
/// let fitted: Vec<_> = fit_many(Naive::new, &[&ts])
///     .into_iter()
///     .filter_map(|r| r.ok())
///     .collect();
/// let forecasts = predict_many(&fitted, 5);
/// assert_eq!(forecasts.len(), 1);
/// ```
pub fn predict_many(models: &[impl Forecaster], horizon: usize) -> Vec<Result<Forecast>> {
    models.iter().map(|m| m.predict(horizon)).collect()
}

/// Fit a model to many series and immediately predict, returning structured results.
///
/// For each series, creates a fresh model via `factory`, fits it, and predicts `horizon`
/// steps. With the `parallel` feature enabled, series are processed in parallel.
///
/// # Arguments
/// * `factory` - A function that creates a new model instance.
/// * `series` - Slice of time series references.
/// * `horizon` - Number of future time steps to forecast.
///
/// # Returns
/// A `Vec<FitPredictResult>` with one entry per series, preserving index ordering.
///
/// # Example
/// ```
/// use anofox_forecast::models::batch::fit_predict_many;
/// use anofox_forecast::models::baseline::Naive;
/// # use anofox_forecast::core::TimeSeries;
/// # use chrono::{TimeZone, Utc};
/// # let ts = TimeSeries::univariate(
/// #     (0..20).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
/// #     (1..=20).map(|i| i as f64).collect(),
/// # ).unwrap();
///
/// let results = fit_predict_many(Naive::new, &[&ts], 5);
/// assert_eq!(results.len(), 1);
/// assert!(results[0].forecast.is_some());
/// assert!(results[0].error.is_none());
/// ```
pub fn fit_predict_many<F, M>(
    factory: F,
    series: &[&TimeSeries],
    horizon: usize,
) -> Vec<FitPredictResult>
where
    F: Fn() -> M + Send + Sync,
    M: Forecaster + Send,
{
    let process = |idx: usize, ts: &&TimeSeries| -> FitPredictResult {
        let mut model = factory();
        let model_name = model.name().to_string();

        match model.fit(ts) {
            Ok(()) => match model.predict(horizon) {
                Ok(forecast) => FitPredictResult {
                    forecast: Some(forecast),
                    model_name,
                    series_index: idx,
                    error: None,
                },
                Err(e) => FitPredictResult {
                    forecast: None,
                    model_name,
                    series_index: idx,
                    error: Some(format!("predict failed: {e}")),
                },
            },
            Err(e) => FitPredictResult {
                forecast: None,
                model_name,
                series_index: idx,
                error: Some(format!("fit failed: {e}")),
            },
        }
    };

    #[cfg(feature = "parallel")]
    {
        series
            .par_iter()
            .enumerate()
            .map(|(idx, ts)| process(idx, ts))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        series
            .iter()
            .enumerate()
            .map(|(idx, ts)| process(idx, ts))
            .collect()
    }
}

/// Fit all models in a registry to the same time series and compare results.
///
/// For each model specification in the registry, creates an instance, fits it to the
/// series, predicts one step ahead (for metric comparison), and computes in-sample
/// accuracy metrics when fitted values are available.
///
/// With the `parallel` feature enabled, models are fitted in parallel using rayon.
///
/// # Arguments
/// * `registry` - A [`ModelRegistry`] containing model specifications to compare.
/// * `series` - The time series to fit all models against.
///
/// # Returns
/// A `Vec<RegistryResult>` with one entry per model in the registry.
///
/// # Example
/// ```
/// use anofox_forecast::models::batch::fit_registry;
/// use anofox_forecast::models::{ModelRegistry, ModelSpec};
/// use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift};
/// # use anofox_forecast::core::TimeSeries;
/// # use chrono::{TimeZone, Utc};
/// # let ts = TimeSeries::univariate(
/// #     (0..30).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect(),
/// #     (1..=30).map(|i| i as f64).collect(),
/// # ).unwrap();
///
/// let mut registry = ModelRegistry::new();
/// registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
/// registry.register(ModelSpec::new("RWD", || Box::new(RandomWalkWithDrift::new()), true));
///
/// let results = fit_registry(&registry, &ts);
/// assert_eq!(results.len(), 2);
/// for r in &results {
///     assert!(r.error.is_none(), "model {} failed: {:?}", r.model_name, r.error);
/// }
/// ```
pub fn fit_registry(registry: &ModelRegistry, series: &TimeSeries) -> Vec<RegistryResult> {
    let actual = series.primary_values();
    let specs: Vec<_> = registry.iter().collect();

    let process = |spec: &&crate::models::traits::ModelSpec| -> RegistryResult {
        let model_name = spec.name.to_string();
        let mut model = spec.create();

        match model.fit(series) {
            Ok(()) => {
                let metrics = model.fitted_values().and_then(|fitted| {
                    let (a, p): (Vec<f64>, Vec<f64>) = actual
                        .iter()
                        .zip(fitted.iter())
                        .filter(|(a, f)| a.is_finite() && f.is_finite())
                        .map(|(&a, &f)| (a, f))
                        .unzip();
                    if a.is_empty() {
                        return None;
                    }
                    calculate_metrics(&a, &p, None).ok()
                });

                match model.predict(1) {
                    Ok(forecast) => RegistryResult {
                        model_name,
                        forecast: Some(forecast),
                        metrics,
                        error: None,
                    },
                    Err(e) => RegistryResult {
                        model_name,
                        forecast: None,
                        metrics,
                        error: Some(format!("predict failed: {e}")),
                    },
                }
            }
            Err(e) => RegistryResult {
                model_name,
                forecast: None,
                metrics: None,
                error: Some(format!("fit failed: {e}")),
            },
        }
    };

    #[cfg(feature = "parallel")]
    {
        specs.par_iter().map(process).collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        specs.iter().map(process).collect()
    }
}

/// Partition pipeline results into clean and flagged sets by structural-break status.
///
/// Returns `(clean, flagged)` where each vector contains indices into `results`.
/// Flagged series had a structural break detected in the holdout period and
/// should be excluded from aggregated accuracy summaries.
pub fn partition_by_structural_break(results: &[PipelineResult]) -> (Vec<usize>, Vec<usize>) {
    let mut clean = Vec::new();
    let mut flagged = Vec::new();
    for (i, r) in results.iter().enumerate() {
        if r.structural_break_in_holdout {
            flagged.push(i);
        } else {
            clean.push(i);
        }
    }
    (clean, flagged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::models::baseline::{Naive, RandomWalkWithDrift, SeasonalNaive, WindowAverage};
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

    fn make_seasonal_series(n: usize, period: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| 10.0 + 5.0 * ((2.0 * std::f64::consts::PI * i as f64) / period as f64).sin())
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    // -----------------------------------------------------------------------
    // fit_many
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_many_single_series() {
        let ts = make_test_series(30);
        let results = fit_many(Naive::new, &[&ts]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        assert!(results[0].as_ref().unwrap().is_fitted());
    }

    #[test]
    fn test_fit_many_multiple_series() {
        let ts1 = make_test_series(30);
        let ts2 = make_test_series(50);
        let ts3 = make_seasonal_series(40, 7);
        let results = fit_many(Naive::new, &[&ts1, &ts2, &ts3]);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_fit_many_preserves_order() {
        let ts1 = make_test_series(20);
        let ts2 = make_test_series(40);
        let results = fit_many(Naive::new, &[&ts1, &ts2]);

        // Both fitted; the first series has 20 residuals, the second 40.
        let m1 = results[0].as_ref().unwrap();
        let m2 = results[1].as_ref().unwrap();
        assert_eq!(m1.residuals().unwrap().len(), 20);
        assert_eq!(m2.residuals().unwrap().len(), 40);
    }

    #[test]
    fn test_fit_many_empty_series_errors() {
        let ts = make_test_series(30);
        // SeasonalNaive with period 12 needs >= 12 points.
        let short = make_test_series(5);
        let results = fit_many(|| SeasonalNaive::new(12), &[&ts, &short]);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    #[test]
    fn test_fit_many_empty_slice() {
        let results = fit_many(Naive::new, &[] as &[&TimeSeries]);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // predict_many
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_many_basic() {
        let ts = make_test_series(30);
        let fitted: Vec<Naive> = fit_many(Naive::new, &[&ts])
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let forecasts = predict_many(&fitted, 5);
        assert_eq!(forecasts.len(), 1);
        let fc = forecasts[0].as_ref().unwrap();
        assert_eq!(fc.horizon(), 5);
    }

    #[test]
    fn test_predict_many_multiple_models() {
        let ts1 = make_test_series(30);
        let ts2 = make_test_series(40);
        let fitted: Vec<Naive> = fit_many(Naive::new, &[&ts1, &ts2])
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let forecasts = predict_many(&fitted, 3);
        assert_eq!(forecasts.len(), 2);
        for fc in &forecasts {
            assert!(fc.is_ok());
            assert_eq!(fc.as_ref().unwrap().horizon(), 3);
        }
    }

    #[test]
    fn test_predict_many_unfitted_model_errors() {
        let unfitted = Naive::new();
        let forecasts = predict_many(&[unfitted], 5);
        assert_eq!(forecasts.len(), 1);
        assert!(forecasts[0].is_err());
    }

    // -----------------------------------------------------------------------
    // fit_predict_many
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_predict_many_basic() {
        let ts = make_test_series(30);
        let results = fit_predict_many(Naive::new, &[&ts], 5);
        assert_eq!(results.len(), 1);

        let r = &results[0];
        assert!(r.forecast.is_some());
        assert!(r.error.is_none());
        assert_eq!(r.series_index, 0);
        assert_eq!(r.model_name, "Naive");
        assert_eq!(r.forecast.as_ref().unwrap().horizon(), 5);
    }

    #[test]
    fn test_fit_predict_many_multiple() {
        let ts1 = make_test_series(30);
        let ts2 = make_test_series(50);
        let results = fit_predict_many(Naive::new, &[&ts1, &ts2], 3);
        assert_eq!(results.len(), 2);

        for (idx, r) in results.iter().enumerate() {
            assert!(r.forecast.is_some(), "series {idx} should succeed");
            assert!(r.error.is_none());
            assert_eq!(r.series_index, idx);
        }
    }

    #[test]
    fn test_fit_predict_many_records_fit_error() {
        let short = make_test_series(5);
        let results = fit_predict_many(|| SeasonalNaive::new(12), &[&short], 3);
        assert_eq!(results.len(), 1);

        let r = &results[0];
        assert!(r.forecast.is_none());
        assert!(r.error.is_some());
        assert!(r.error.as_ref().unwrap().contains("fit failed"));
    }

    #[test]
    fn test_fit_predict_many_mixed_success_failure() {
        let good = make_test_series(30);
        let bad = make_test_series(3);
        let results = fit_predict_many(|| SeasonalNaive::new(12), &[&good, &bad], 5);
        assert_eq!(results.len(), 2);

        assert!(results[0].forecast.is_some());
        assert!(results[0].error.is_none());

        assert!(results[1].forecast.is_none());
        assert!(results[1].error.is_some());
    }

    #[test]
    fn test_fit_predict_many_empty_slice() {
        let results = fit_predict_many(Naive::new, &[] as &[&TimeSeries], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fit_predict_many_model_name_preserved() {
        let ts = make_test_series(30);
        let results = fit_predict_many(|| WindowAverage::new(5), &[&ts], 3);
        assert_eq!(results[0].model_name, "WindowAverage");
    }

    // -----------------------------------------------------------------------
    // fit_registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_registry_basic() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RWD",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));

        let ts = make_test_series(30);
        let results = fit_registry(&registry, &ts);
        assert_eq!(results.len(), 2);

        for r in &results {
            assert!(r.error.is_none(), "{} failed: {:?}", r.model_name, r.error);
            assert!(r.forecast.is_some());
            assert!(r.metrics.is_some());
        }
    }

    #[test]
    fn test_fit_registry_has_metrics() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));

        let ts = make_test_series(30);
        let results = fit_registry(&registry, &ts);

        let metrics = results[0].metrics.as_ref().unwrap();
        assert!(metrics.mae >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.smape >= 0.0);
    }

    #[test]
    fn test_fit_registry_model_names() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RWD",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));

        let ts = make_test_series(30);
        let results = fit_registry(&registry, &ts);

        let names: Vec<&str> = results.iter().map(|r| r.model_name.as_str()).collect();
        assert_eq!(names, vec!["Naive", "RWD"]);
    }

    #[test]
    fn test_fit_registry_empty_registry() {
        let registry = ModelRegistry::new();
        let ts = make_test_series(30);
        let results = fit_registry(&registry, &ts);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fit_registry_records_error_for_bad_model() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::with_period(
            "SeasonalNaive",
            |p| Box::new(SeasonalNaive::new(p)),
            12,
            true,
        ));

        // Only 5 data points - Naive works, SeasonalNaive(12) should fail.
        let ts = make_test_series(5);
        let results = fit_registry(&registry, &ts);
        assert_eq!(results.len(), 2);

        assert!(results[0].error.is_none(), "Naive should succeed");
        assert!(
            results[1].error.is_some(),
            "SeasonalNaive(12) should fail with 5 points"
        );
    }

    #[test]
    fn test_fit_registry_forecast_horizon_is_one() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));

        let ts = make_test_series(30);
        let results = fit_registry(&registry, &ts);
        let fc = results[0].forecast.as_ref().unwrap();
        assert_eq!(fc.horizon(), 1);
    }

    #[test]
    fn test_fit_registry_metrics_reasonable_for_trend() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        registry.register(ModelSpec::new(
            "RWD",
            || Box::new(RandomWalkWithDrift::new()),
            true,
        ));

        // Linear trend: 1, 2, 3, ..., 30
        let ts = make_test_series(30);
        let results = fit_registry(&registry, &ts);

        let naive_rmse = results[0].metrics.as_ref().unwrap().rmse;
        let rwd_rmse = results[1].metrics.as_ref().unwrap().rmse;

        // RandomWalkWithDrift should capture the trend better than Naive on a linear series.
        // Both have MAE=1 for a unit-step series, but RWD should be at least as good.
        assert!(
            rwd_rmse <= naive_rmse + 0.01,
            "RWD rmse={rwd_rmse} should be <= Naive rmse={naive_rmse}"
        );
    }

    // -----------------------------------------------------------------------
    // BatchResult
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_result_success() {
        let ts = make_test_series(30);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let result = BatchResult {
            model,
            series_index: 0,
            error: None,
        };
        assert!(result.error.is_none());
        assert!(result.model.is_fitted());
        assert_eq!(result.series_index, 0);
    }

    #[test]
    fn test_batch_result_failure() {
        let model = Naive::new();
        let result = BatchResult {
            model,
            series_index: 3,
            error: Some("insufficient data".to_string()),
        };
        assert!(result.error.is_some());
        assert!(!result.model.is_fitted());
    }

    #[test]
    fn partition_by_structural_break_filters() {
        use crate::core::Forecast;
        use crate::orchestration::decision_log::DecisionLog;
        use crate::orchestration::pipeline::PipelineResult;

        let make = |flagged: bool| PipelineResult {
            forecast: Forecast::from_values(vec![1.0]),
            model_name: "Naive".into(),
            profile: None,
            log: DecisionLog::new(),
            model_metadata: vec![],
            horizon_analysis: None,
            selection_confidence: None,
            model_confidence_set: None,
            quality_floor: None,
            preprocess: None,
            ensemble_weights: None,
            metric_scores: None,
            trend_selection: None,
            seasonal_selection: None,
            structural_break_in_holdout: flagged,
        };

        let results = vec![make(false), make(true), make(false), make(true), make(false)];
        let (clean, flagged) = partition_by_structural_break(&results);
        assert_eq!(clean, vec![0, 2, 4]);
        assert_eq!(flagged, vec![1, 3]);
    }
}
