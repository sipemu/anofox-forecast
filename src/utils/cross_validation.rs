//! Cross-validation utilities for time series forecasting.

use crate::core::TimeSeries;
use crate::error::Result;
use crate::models::Forecaster;
use crate::utils::metrics::{calculate_metrics, AccuracyMetrics};

/// Cross-validation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CVStrategy {
    /// Rolling window: fixed training window size, slides forward.
    Rolling,
    /// Expanding window: training window grows, starts from initial_window.
    Expanding,
}

impl Default for CVStrategy {
    fn default() -> Self {
        Self::Expanding
    }
}

/// Configuration for time series cross-validation.
#[derive(Debug, Clone)]
pub struct CVConfig {
    /// Forecast horizon for each fold.
    pub horizon: usize,
    /// Initial training window size.
    pub initial_window: usize,
    /// Step size between folds.
    pub step_size: usize,
    /// Cross-validation strategy.
    pub strategy: CVStrategy,
    /// Optional seasonal period for MASE calculation.
    pub seasonal_period: Option<usize>,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            horizon: 1,
            initial_window: 10,
            step_size: 1,
            strategy: CVStrategy::Expanding,
            seasonal_period: None,
        }
    }
}

impl CVConfig {
    /// Create a new CV configuration with expanding window strategy.
    pub fn expanding(initial_window: usize, horizon: usize) -> Self {
        Self {
            initial_window,
            horizon,
            step_size: 1,
            strategy: CVStrategy::Expanding,
            seasonal_period: None,
        }
    }

    /// Create a new CV configuration with rolling window strategy.
    pub fn rolling(window_size: usize, horizon: usize) -> Self {
        Self {
            initial_window: window_size,
            horizon,
            step_size: 1,
            strategy: CVStrategy::Rolling,
            seasonal_period: None,
        }
    }

    /// Set the step size between folds.
    pub fn with_step_size(mut self, step_size: usize) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set the seasonal period for MASE calculation.
    pub fn with_seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = Some(period);
        self
    }
}

/// Results from cross-validation.
#[derive(Debug, Clone)]
pub struct CVResults {
    /// Number of folds evaluated.
    pub n_folds: usize,
    /// Aggregated metrics across all folds.
    pub aggregated: AggregatedMetrics,
    /// Per-fold metrics.
    pub fold_metrics: Vec<AccuracyMetrics>,
    /// Per-fold actual values (flattened).
    pub actual_values: Vec<f64>,
    /// Per-fold predicted values (flattened).
    pub predicted_values: Vec<f64>,
}

/// Aggregated metrics from cross-validation.
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Mean MAE across folds.
    pub mae: f64,
    /// Mean RMSE across folds.
    pub rmse: f64,
    /// Mean SMAPE across folds.
    pub smape: f64,
    /// Mean MAPE across folds (None if any fold had zeros).
    pub mape: Option<f64>,
    /// Standard deviation of MAE across folds.
    pub mae_std: f64,
    /// Standard deviation of RMSE across folds.
    pub rmse_std: f64,
}

/// Perform time series cross-validation.
///
/// # Arguments
/// * `config` - Cross-validation configuration
/// * `series` - The time series to validate on
/// * `model_factory` - Function that creates a fresh model instance for each fold
///
/// # Returns
/// `CVResults` containing aggregated and per-fold metrics.
///
/// # Example
/// ```
/// use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};
/// use anofox_forecast::models::baseline::Naive;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..20)
///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i as u32 % 24, 0, 0).unwrap())
///     .collect();
/// let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let config = CVConfig::expanding(10, 1).with_step_size(2);
/// let results = cross_validate(&config, &ts, Naive::new).unwrap();
///
/// assert!(results.n_folds > 0);
/// assert!(results.aggregated.mae >= 0.0);
/// ```
pub fn cross_validate<F, Factory>(
    config: &CVConfig,
    series: &TimeSeries,
    model_factory: Factory,
) -> Result<CVResults>
where
    F: Forecaster,
    Factory: Fn() -> F,
{
    let n = series.len();
    let mut fold_metrics = Vec::new();
    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();

    // Calculate number of folds
    let mut origin = config.initial_window;
    while origin + config.horizon <= n {
        // Determine training window
        let train_start = match config.strategy {
            CVStrategy::Rolling => {
                if origin > config.initial_window {
                    origin - config.initial_window
                } else {
                    0
                }
            }
            CVStrategy::Expanding => 0,
        };
        let train_end = origin;

        // Create training subset
        let train_series = series.slice(train_start, train_end)?;

        // Create and fit model
        let mut model = model_factory();
        model.fit(&train_series)?;

        // Generate forecast
        let forecast = model.predict(config.horizon)?;
        let predictions = forecast.primary();

        // Get actual values for this fold
        let actual: Vec<f64> = (origin..origin + config.horizon)
            .map(|i| series.primary_values()[i])
            .collect();

        // Calculate metrics for this fold
        let metrics = calculate_metrics(&actual, predictions, config.seasonal_period)?;
        fold_metrics.push(metrics);

        // Store values for overall metrics
        all_actual.extend_from_slice(&actual);
        all_predicted.extend_from_slice(predictions);

        origin += config.step_size;
    }

    let n_folds = fold_metrics.len();
    if n_folds == 0 {
        return Ok(CVResults {
            n_folds: 0,
            aggregated: AggregatedMetrics {
                mae: f64::NAN,
                rmse: f64::NAN,
                smape: f64::NAN,
                mape: None,
                mae_std: f64::NAN,
                rmse_std: f64::NAN,
            },
            fold_metrics: vec![],
            actual_values: vec![],
            predicted_values: vec![],
        });
    }

    // Aggregate metrics
    let mae_values: Vec<f64> = fold_metrics.iter().map(|m| m.mae).collect();
    let rmse_values: Vec<f64> = fold_metrics.iter().map(|m| m.rmse).collect();
    let smape_values: Vec<f64> = fold_metrics.iter().map(|m| m.smape).collect();

    let mae_mean = mae_values.iter().sum::<f64>() / n_folds as f64;
    let rmse_mean = rmse_values.iter().sum::<f64>() / n_folds as f64;
    let smape_mean = smape_values.iter().sum::<f64>() / n_folds as f64;

    let mae_std = std_dev(&mae_values);
    let rmse_std = std_dev(&rmse_values);

    // MAPE is only valid if all folds have it
    let mape = if fold_metrics.iter().all(|m| m.mape.is_some()) {
        let mape_values: Vec<f64> = fold_metrics.iter().filter_map(|m| m.mape).collect();
        Some(mape_values.iter().sum::<f64>() / n_folds as f64)
    } else {
        None
    };

    Ok(CVResults {
        n_folds,
        aggregated: AggregatedMetrics {
            mae: mae_mean,
            rmse: rmse_mean,
            smape: smape_mean,
            mape,
            mae_std,
            rmse_std,
        },
        fold_metrics,
        actual_values: all_actual,
        predicted_values: all_predicted,
    })
}

/// Calculate sample standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::baseline::{Naive, SimpleMovingAverage};
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        use chrono::Duration;
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n)
            .map(|i| base + Duration::hours(i as i64))
            .collect()
    }

    #[test]
    fn cv_expanding_window_basic() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // With step_size=1, horizon=1, starting from origin=10
        // Folds: 10->11, 11->12, ..., 19->20 = 10 folds
        assert_eq!(results.n_folds, 10);
        assert!(results.aggregated.mae.is_finite());
    }

    #[test]
    fn cv_rolling_window_basic() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::rolling(10, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        assert_eq!(results.n_folds, 10);
        assert!(results.aggregated.mae.is_finite());
    }

    #[test]
    fn cv_with_step_size() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1).with_step_size(2);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Folds: 10->11, 12->13, 14->15, 16->17, 18->19 = 5 folds
        assert_eq!(results.n_folds, 5);
    }

    #[test]
    fn cv_multi_step_horizon() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 3);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Folds: 10->13, 11->14, ..., 17->20 = 8 folds
        assert_eq!(results.n_folds, 8);
        // Each fold has 3 predictions
        assert_eq!(results.actual_values.len(), 8 * 3);
        assert_eq!(results.predicted_values.len(), 8 * 3);
    }

    #[test]
    fn cv_insufficient_data_returns_zero_folds() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // initial_window=10 but only 5 data points
        let config = CVConfig::expanding(10, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        assert_eq!(results.n_folds, 0);
    }

    #[test]
    fn cv_naive_perfect_on_constant() {
        let timestamps = make_timestamps(20);
        let values = vec![5.0; 20]; // Constant series
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Naive should have zero error on constant series
        assert_relative_eq!(results.aggregated.mae, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cv_sma_on_linear_trend() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(15, 1);
        let results = cross_validate(&config, &ts, || SimpleMovingAverage::new(5)).unwrap();

        // SMA will lag behind a linear trend
        assert!(results.aggregated.mae > 0.0);
        assert!(results.aggregated.rmse >= results.aggregated.mae);
    }

    #[test]
    fn cv_metrics_are_consistent() {
        let timestamps = make_timestamps(25);
        let values: Vec<f64> = (0..25).map(|i| (i as f64).sin() * 10.0 + 50.0).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(15, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Basic sanity checks
        assert!(results.aggregated.rmse >= results.aggregated.mae);
        assert!(results.aggregated.smape >= 0.0);
        assert!(results.aggregated.smape <= 200.0);
        assert!(results.aggregated.mae_std >= 0.0);
    }

    #[test]
    fn cv_fold_metrics_match_aggregated() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64 + 0.1 * (i as f64).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Aggregated MAE should be mean of fold MAEs
        let manual_mae_mean: f64 = results.fold_metrics.iter().map(|m| m.mae).sum::<f64>()
            / results.n_folds as f64;
        assert_relative_eq!(results.aggregated.mae, manual_mae_mean, epsilon = 1e-10);
    }

    #[test]
    fn cv_with_seasonal_period() {
        let timestamps = make_timestamps(30);
        // Seasonal pattern with slight variation so naive MAE is non-zero
        let values: Vec<f64> = (0..30)
            .map(|i| ((i % 4) as f64) * 10.0 + 5.0 + 0.5 * (i as f64))
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Use horizon >= seasonal_period for MASE to be computable
        let config = CVConfig::expanding(12, 5).with_seasonal_period(4).with_step_size(3);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // MASE should be computed for folds where horizon > period
        let mase_count = results.fold_metrics.iter().filter(|m| m.mase.is_some()).count();
        assert!(mase_count > 0);
    }

    #[test]
    fn cv_config_builders() {
        let expanding = CVConfig::expanding(10, 3);
        assert_eq!(expanding.initial_window, 10);
        assert_eq!(expanding.horizon, 3);
        assert_eq!(expanding.strategy, CVStrategy::Expanding);

        let rolling = CVConfig::rolling(15, 2);
        assert_eq!(rolling.initial_window, 15);
        assert_eq!(rolling.horizon, 2);
        assert_eq!(rolling.strategy, CVStrategy::Rolling);

        let with_step = CVConfig::expanding(10, 1).with_step_size(5);
        assert_eq!(with_step.step_size, 5);

        let with_seasonal = CVConfig::expanding(10, 1).with_seasonal_period(12);
        assert_eq!(with_seasonal.seasonal_period, Some(12));
    }

    #[test]
    fn cv_default_config() {
        let config = CVConfig::default();
        assert_eq!(config.horizon, 1);
        assert_eq!(config.initial_window, 10);
        assert_eq!(config.step_size, 1);
        assert_eq!(config.strategy, CVStrategy::Expanding);
        assert_eq!(config.seasonal_period, None);
    }

    #[test]
    fn cv_values_stored_correctly() {
        let timestamps = make_timestamps(15);
        let values: Vec<f64> = (0..15).map(|i| i as f64 * 2.0).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let config = CVConfig::expanding(10, 2).with_step_size(2);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Verify actual values are from the series
        for &actual in &results.actual_values {
            assert!(values.iter().any(|&v| (v - actual).abs() < 1e-10));
        }
    }
}
