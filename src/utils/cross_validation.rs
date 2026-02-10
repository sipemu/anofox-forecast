//! Cross-validation utilities for time series forecasting.
//!
//! This module provides comprehensive cross-validation functionality including:
//! - Standalone fold generation via [`CvFoldGenerator`]
//! - Expanding and rolling window strategies
//! - Gap and purge parameters for preventing data leakage
//! - Fill strategies for unknown future features
//! - Grouped cross-validation for multi-series data
//! - Simple train/test splitting utilities

use crate::core::TimeSeries;
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::metrics::{calculate_metrics, AccuracyMetrics};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Cross-validation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CVStrategy {
    /// Rolling window: fixed training window size, slides forward.
    Rolling,
    /// Expanding window: training window grows, starts from initial_window.
    #[default]
    Expanding,
}

/// A single fold specification with train/test indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// Start index of training data (inclusive).
    pub train_start: usize,
    /// End index of training data (exclusive).
    pub train_end: usize,
    /// Start index of test data (inclusive).
    pub test_start: usize,
    /// End index of test data (exclusive).
    pub test_end: usize,
}

impl Fold {
    /// Returns the number of training samples.
    pub fn train_size(&self) -> usize {
        self.train_end - self.train_start
    }

    /// Returns the number of test samples.
    pub fn test_size(&self) -> usize {
        self.test_end - self.test_start
    }
}

/// Standalone fold generator for reusable cross-validation indices.
///
/// This struct generates fold indices without running models, enabling:
/// - Custom workflows where fold indices are needed separately
/// - Multi-series scenarios where folds must be consistent
/// - Integration with external training pipelines
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::{CvFoldGenerator, CVStrategy};
///
/// let generator = CvFoldGenerator::new()
///     .initial_window(100)
///     .horizon(7)
///     .step_size(7)
///     .gap(1)
///     .strategy(CVStrategy::Expanding);
///
/// let folds = generator.generate(365);
/// for fold in &folds {
///     println!("Train: {}..{}, Test: {}..{}",
///         fold.train_start, fold.train_end,
///         fold.test_start, fold.test_end);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CvFoldGenerator {
    /// Initial training window size.
    pub initial_window: usize,
    /// Forecast horizon (test size per fold).
    pub horizon: usize,
    /// Step size between consecutive folds.
    pub step_size: usize,
    /// Gap between training end and test start (prevents leakage from lagged features).
    pub gap: usize,
    /// Purge window: observations to remove before training end (prevents lookahead bias).
    pub purge: usize,
    /// Cross-validation strategy (expanding or rolling).
    pub strategy: CVStrategy,
}

impl Default for CvFoldGenerator {
    fn default() -> Self {
        Self {
            initial_window: 10,
            horizon: 1,
            step_size: 1,
            gap: 0,
            purge: 0,
            strategy: CVStrategy::Expanding,
        }
    }
}

impl CvFoldGenerator {
    /// Create a new fold generator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial training window size.
    pub fn initial_window(mut self, size: usize) -> Self {
        self.initial_window = size;
        self
    }

    /// Set the forecast horizon (test size per fold).
    pub fn horizon(mut self, h: usize) -> Self {
        self.horizon = h;
        self
    }

    /// Set the step size between folds.
    pub fn step_size(mut self, step: usize) -> Self {
        self.step_size = step;
        self
    }

    /// Set the gap between training end and test start.
    ///
    /// Use this to prevent data leakage when features have lag effects.
    /// For example, if you use lagged features up to 3 periods, set gap=3.
    pub fn gap(mut self, g: usize) -> Self {
        self.gap = g;
        self
    }

    /// Set the purge window size.
    ///
    /// Purging removes observations from the end of the training set to prevent
    /// lookahead bias. Common in financial applications where autocorrelation
    /// in features could leak information.
    pub fn purge(mut self, p: usize) -> Self {
        self.purge = p;
        self
    }

    /// Set the cross-validation strategy.
    pub fn strategy(mut self, s: CVStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Generate fold indices for a series of given length.
    ///
    /// Returns a vector of [`Fold`] structs containing train/test indices.
    /// Returns an empty vector if the series is too short for any folds.
    pub fn generate(&self, series_len: usize) -> Vec<Fold> {
        let mut folds = Vec::new();
        let mut origin = self.initial_window;

        while origin + self.gap + self.horizon <= series_len {
            let train_start = match self.strategy {
                CVStrategy::Rolling => origin.saturating_sub(self.initial_window),
                CVStrategy::Expanding => 0,
            };

            // Apply purge to training end
            let train_end = origin.saturating_sub(self.purge);

            // Skip if purge makes training window too small
            if train_end <= train_start {
                origin += self.step_size;
                continue;
            }

            let test_start = origin + self.gap;
            let test_end = test_start + self.horizon;

            folds.push(Fold {
                train_start,
                train_end,
                test_start,
                test_end,
            });

            origin += self.step_size;
        }

        folds
    }

    /// Returns the number of folds that would be generated for a given series length.
    pub fn n_folds(&self, series_len: usize) -> usize {
        self.generate(series_len).len()
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
    /// Gap between training end and test start.
    pub gap: usize,
    /// Purge window: observations to remove before training end.
    pub purge: usize,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            horizon: 1,
            initial_window: 10,
            step_size: 1,
            strategy: CVStrategy::Expanding,
            seasonal_period: None,
            gap: 0,
            purge: 0,
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
            gap: 0,
            purge: 0,
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
            gap: 0,
            purge: 0,
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

    /// Set the gap between training end and test start.
    ///
    /// Use this to prevent data leakage when features have lag effects.
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Set the purge window size.
    ///
    /// Purging removes observations from the end of the training set to prevent
    /// lookahead bias from autocorrelated features.
    pub fn with_purge(mut self, purge: usize) -> Self {
        self.purge = purge;
        self
    }

    /// Convert to a CvFoldGenerator.
    pub fn to_fold_generator(&self) -> CvFoldGenerator {
        CvFoldGenerator {
            initial_window: self.initial_window,
            horizon: self.horizon,
            step_size: self.step_size,
            gap: self.gap,
            purge: self.purge,
            strategy: self.strategy,
        }
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
    /// Fold specifications used.
    pub folds: Vec<Fold>,
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

/// Strategy for filling unknown future feature values during cross-validation.
///
/// When performing CV, test periods may have features whose true values are
/// unknown at prediction time (e.g., stockout indicators, promotional flags).
/// This trait defines how to fill these values.
pub trait FillStrategy: Clone {
    /// Fill unknown values given training data.
    ///
    /// # Arguments
    /// * `train_values` - The feature values from the training period
    /// * `test_len` - Number of values to generate for the test period
    ///
    /// # Returns
    /// A vector of filled values for the test period.
    fn fill(&self, train_values: &[f64], test_len: usize) -> Vec<f64>;
}

/// Fill with the last observed value (carry forward).
#[derive(Debug, Clone, Default)]
pub struct LastValueFill;

impl FillStrategy for LastValueFill {
    fn fill(&self, train_values: &[f64], test_len: usize) -> Vec<f64> {
        let last = train_values.last().copied().unwrap_or(0.0);
        vec![last; test_len]
    }
}

/// Fill with the mean of training values.
#[derive(Debug, Clone, Default)]
pub struct MeanFill;

impl FillStrategy for MeanFill {
    fn fill(&self, train_values: &[f64], test_len: usize) -> Vec<f64> {
        if train_values.is_empty() {
            return vec![0.0; test_len];
        }
        let mean = train_values.iter().sum::<f64>() / train_values.len() as f64;
        vec![mean; test_len]
    }
}

/// Fill with the median of training values.
#[derive(Debug, Clone, Default)]
pub struct MedianFill;

impl FillStrategy for MedianFill {
    fn fill(&self, train_values: &[f64], test_len: usize) -> Vec<f64> {
        if train_values.is_empty() {
            return vec![0.0; test_len];
        }
        let mut sorted: Vec<f64> = train_values
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .collect();
        if sorted.is_empty() {
            return vec![0.0; test_len];
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        vec![median; test_len]
    }
}

/// Fill with zeros.
#[derive(Debug, Clone, Default)]
pub struct ZeroFill;

impl FillStrategy for ZeroFill {
    fn fill(&self, _train_values: &[f64], test_len: usize) -> Vec<f64> {
        vec![0.0; test_len]
    }
}

/// Fill with a constant value.
#[derive(Debug, Clone)]
pub struct ConstantFill(pub f64);

impl FillStrategy for ConstantFill {
    fn fill(&self, _train_values: &[f64], test_len: usize) -> Vec<f64> {
        vec![self.0; test_len]
    }
}

/// Fill with the mode (most frequent value) - useful for categorical indicators.
#[derive(Debug, Clone, Default)]
pub struct ModeFill;

impl FillStrategy for ModeFill {
    fn fill(&self, train_values: &[f64], test_len: usize) -> Vec<f64> {
        if train_values.is_empty() {
            return vec![0.0; test_len];
        }

        // Count occurrences (using integer keys for floating point)
        use std::collections::HashMap;
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &v in train_values {
            if v.is_finite() {
                // Use fixed-point representation for counting
                let key = (v * 1_000_000.0).round() as i64;
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        let mode = counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(key, _)| key as f64 / 1_000_000.0)
            .unwrap_or(0.0);

        vec![mode; test_len]
    }
}

/// Simple train/test split for time series.
///
/// Splits a time series at a given index or ratio, respecting temporal order.
///
/// # Arguments
/// * `series` - The time series to split
/// * `split_point` - Either an absolute index or a ratio (0.0 to 1.0)
///
/// # Returns
/// A tuple of (train_series, test_series).
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::train_test_split;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..100)
///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
///         + chrono::Duration::hours(i))
///     .collect();
/// let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// // Split at 80% for training
/// let (train, test) = train_test_split(&ts, 0.8).unwrap();
/// assert_eq!(train.len(), 80);
/// assert_eq!(test.len(), 20);
/// ```
pub fn train_test_split(series: &TimeSeries, split_point: f64) -> Result<(TimeSeries, TimeSeries)> {
    let n = series.len();
    if n < 2 {
        return Err(ForecastError::InvalidParameter(
            "Series must have at least 2 observations for train/test split".to_string(),
        ));
    }

    let split_idx = if split_point > 0.0 && split_point < 1.0 {
        // Ratio-based split
        (n as f64 * split_point).round() as usize
    } else if split_point >= 1.0 && split_point < n as f64 {
        // Index-based split
        split_point as usize
    } else {
        return Err(ForecastError::InvalidParameter(format!(
            "split_point must be a ratio (0.0-1.0) or index (1 to {}), got {}",
            n - 1,
            split_point
        )));
    };

    // Ensure both splits have at least 1 observation
    let split_idx = split_idx.clamp(1, n - 1);

    let train = series.slice(0, split_idx)?;
    let test = series.slice(split_idx, n)?;

    Ok((train, test))
}

/// Split at an absolute index.
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::train_test_split_at;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..100)
///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
///         + chrono::Duration::hours(i))
///     .collect();
/// let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let (train, test) = train_test_split_at(&ts, 70).unwrap();
/// assert_eq!(train.len(), 70);
/// assert_eq!(test.len(), 30);
/// ```
pub fn train_test_split_at(series: &TimeSeries, index: usize) -> Result<(TimeSeries, TimeSeries)> {
    let n = series.len();
    if index == 0 || index >= n {
        return Err(ForecastError::InvalidParameter(format!(
            "split index must be between 1 and {}, got {}",
            n - 1,
            index
        )));
    }

    let train = series.slice(0, index)?;
    let test = series.slice(index, n)?;

    Ok((train, test))
}

/// Evaluate a single CV fold: slice, fit, predict, compute metrics.
fn evaluate_fold<F: Forecaster>(
    series: &TimeSeries,
    fold: &Fold,
    model_factory: &dyn Fn() -> F,
    seasonal_period: Option<usize>,
) -> Result<(AccuracyMetrics, Vec<f64>, Vec<f64>)> {
    let train_series = series.slice(fold.train_start, fold.train_end)?;
    let mut model = model_factory();
    model.fit(&train_series)?;
    let forecast = model.predict(fold.test_size())?;
    let predicted: Vec<f64> = forecast.primary().to_vec();
    let actual: Vec<f64> = (fold.test_start..fold.test_end)
        .map(|i| series.primary_values()[i])
        .collect();
    let metrics = calculate_metrics(&actual, &predicted, seasonal_period)?;
    Ok((metrics, actual, predicted))
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
    F: Forecaster + Send,
    Factory: Fn() -> F + Sync,
{
    let generator = config.to_fold_generator();
    let folds = generator.generate(series.len());

    if folds.is_empty() {
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
            folds: vec![],
        });
    }

    let seasonal_period = config.seasonal_period;

    // Evaluate each fold (parallel when feature enabled)
    #[cfg(feature = "parallel")]
    let fold_results: Vec<Result<(AccuracyMetrics, Vec<f64>, Vec<f64>)>> = folds
        .par_iter()
        .map(|fold| evaluate_fold(series, fold, &model_factory, seasonal_period))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let fold_results: Vec<Result<(AccuracyMetrics, Vec<f64>, Vec<f64>)>> = folds
        .iter()
        .map(|fold| evaluate_fold(series, fold, &model_factory, seasonal_period))
        .collect();

    let mut fold_metrics = Vec::with_capacity(folds.len());
    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();

    for result in fold_results {
        let (metrics, actual, predicted) = result?;
        fold_metrics.push(metrics);
        all_actual.extend_from_slice(&actual);
        all_predicted.extend_from_slice(&predicted);
    }

    let n_folds = fold_metrics.len();

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
        folds,
    })
}

/// Results from grouped cross-validation.
#[derive(Debug, Clone)]
pub struct GroupedCVResults {
    /// Results per group (keyed by group identifier).
    pub group_results: Vec<(String, CVResults)>,
    /// Aggregated metrics across all groups.
    pub aggregated: AggregatedMetrics,
    /// Fold specifications used (consistent across all groups).
    pub folds: Vec<Fold>,
}

/// Perform grouped cross-validation across multiple series.
///
/// All series use the same fold boundaries, ensuring consistent evaluation.
/// This is essential for panel/hierarchical time series where you want
/// comparable metrics across series.
///
/// # Arguments
/// * `config` - Cross-validation configuration
/// * `series_map` - Iterator of (group_id, TimeSeries) pairs
/// * `model_factory` - Function that creates a fresh model for each fold and group
///
/// # Returns
/// `GroupedCVResults` containing per-group and aggregated metrics.
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::{grouped_cross_validate, CVConfig};
/// use anofox_forecast::models::baseline::Naive;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// // Create multiple series
/// let timestamps: Vec<_> = (0..30)
///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
///         + chrono::Duration::hours(i))
///     .collect();
///
/// let series_a = TimeSeries::univariate(
///     timestamps.clone(),
///     (0..30).map(|i| i as f64).collect()
/// ).unwrap();
///
/// let series_b = TimeSeries::univariate(
///     timestamps.clone(),
///     (0..30).map(|i| (i as f64) * 2.0).collect()
/// ).unwrap();
///
/// let series_map = vec![
///     ("product_a".to_string(), series_a),
///     ("product_b".to_string(), series_b),
/// ];
///
/// let config = CVConfig::expanding(15, 3).with_step_size(3);
/// let results = grouped_cross_validate(&config, series_map.into_iter(), Naive::new).unwrap();
///
/// assert_eq!(results.group_results.len(), 2);
/// ```
pub fn grouped_cross_validate<F, Factory, I>(
    config: &CVConfig,
    series_map: I,
    model_factory: Factory,
) -> Result<GroupedCVResults>
where
    F: Forecaster + Send,
    Factory: Fn() -> F + Sync,
    I: IntoIterator<Item = (String, TimeSeries)>,
{
    let series_vec: Vec<(String, TimeSeries)> = series_map.into_iter().collect();

    if series_vec.is_empty() {
        return Err(ForecastError::InvalidParameter(
            "No series provided for grouped cross-validation".to_string(),
        ));
    }

    // Use the minimum length across all series for fold generation
    let min_len = series_vec.iter().map(|(_, s)| s.len()).min().unwrap_or(0);

    let generator = config.to_fold_generator();
    let folds = generator.generate(min_len);

    if folds.is_empty() {
        return Err(ForecastError::InvalidParameter(
            "Not enough data for any CV folds".to_string(),
        ));
    }

    let mut group_results = Vec::with_capacity(series_vec.len());
    let mut all_mae = Vec::new();
    let mut all_rmse = Vec::new();
    let mut all_smape = Vec::new();
    let mut all_mape = Vec::new();

    for (group_id, series) in series_vec {
        // Run CV using the shared folds (not series-specific folds)
        let cv_result = cross_validate_with_folds(config, &series, &folds, &model_factory)?;

        all_mae.push(cv_result.aggregated.mae);
        all_rmse.push(cv_result.aggregated.rmse);
        all_smape.push(cv_result.aggregated.smape);
        if let Some(mape) = cv_result.aggregated.mape {
            all_mape.push(mape);
        }

        group_results.push((group_id, cv_result));
    }

    let n_groups = group_results.len() as f64;
    let aggregated = AggregatedMetrics {
        mae: all_mae.iter().sum::<f64>() / n_groups,
        rmse: all_rmse.iter().sum::<f64>() / n_groups,
        smape: all_smape.iter().sum::<f64>() / n_groups,
        mape: if all_mape.len() == group_results.len() {
            Some(all_mape.iter().sum::<f64>() / n_groups)
        } else {
            None
        },
        mae_std: std_dev(&all_mae),
        rmse_std: std_dev(&all_rmse),
    };

    Ok(GroupedCVResults {
        group_results,
        aggregated,
        folds,
    })
}

/// Internal helper to run cross-validation with pre-computed folds.
fn cross_validate_with_folds<F, Factory>(
    config: &CVConfig,
    series: &TimeSeries,
    folds: &[Fold],
    model_factory: &Factory,
) -> Result<CVResults>
where
    F: Forecaster,
    Factory: Fn() -> F,
{
    if folds.is_empty() {
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
            folds: vec![],
        });
    }

    let mut fold_metrics = Vec::with_capacity(folds.len());
    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();

    for fold in folds {
        // Create training subset
        let train_series = series.slice(fold.train_start, fold.train_end)?;

        // Create and fit model
        let mut model = model_factory();
        model.fit(&train_series)?;

        // Generate forecast
        let forecast = model.predict(fold.test_size())?;
        let predictions = forecast.primary();

        // Get actual values for this fold
        let actual: Vec<f64> = (fold.test_start..fold.test_end)
            .map(|i| series.primary_values()[i])
            .collect();

        // Calculate metrics for this fold
        let metrics = calculate_metrics(&actual, predictions, config.seasonal_period)?;
        fold_metrics.push(metrics);

        // Store values for overall metrics
        all_actual.extend_from_slice(&actual);
        all_predicted.extend_from_slice(predictions);
    }

    let n_folds = fold_metrics.len();

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
        folds: folds.to_vec(),
    })
}

/// Calculate sample standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
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
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    // ==================== CvFoldGenerator Tests ====================

    #[test]
    fn fold_generator_basic() {
        let gen = CvFoldGenerator::new()
            .initial_window(10)
            .horizon(1)
            .step_size(1);

        let folds = gen.generate(20);

        // 10 folds: origin 10..19, each with 1 test point
        assert_eq!(folds.len(), 10);
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 10);
        assert_eq!(folds[0].test_start, 10);
        assert_eq!(folds[0].test_end, 11);
    }

    #[test]
    fn fold_generator_with_gap() {
        let gen = CvFoldGenerator::new()
            .initial_window(10)
            .horizon(1)
            .step_size(1)
            .gap(2);

        let folds = gen.generate(20);

        // With gap=2, test starts at origin+2
        assert_eq!(folds[0].train_end, 10);
        assert_eq!(folds[0].test_start, 12); // 10 + gap(2)
        assert_eq!(folds[0].test_end, 13);

        // Fewer folds because of gap
        assert_eq!(folds.len(), 8);
    }

    #[test]
    fn fold_generator_with_purge() {
        let gen = CvFoldGenerator::new()
            .initial_window(10)
            .horizon(1)
            .step_size(1)
            .purge(2);

        let folds = gen.generate(20);

        // With purge=2, train_end is reduced by 2
        assert_eq!(folds[0].train_end, 8); // 10 - purge(2)
        assert_eq!(folds[0].test_start, 10);
    }

    #[test]
    fn fold_generator_rolling() {
        let gen = CvFoldGenerator::new()
            .initial_window(5)
            .horizon(1)
            .step_size(1)
            .strategy(CVStrategy::Rolling);

        let folds = gen.generate(15);

        // Rolling: train window stays fixed size
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 5);

        assert_eq!(folds[5].train_start, 5);
        assert_eq!(folds[5].train_end, 10);
    }

    #[test]
    fn fold_generator_multi_step_horizon() {
        let gen = CvFoldGenerator::new()
            .initial_window(10)
            .horizon(3)
            .step_size(2);

        let folds = gen.generate(20);

        assert_eq!(folds[0].test_start, 10);
        assert_eq!(folds[0].test_end, 13);
        assert_eq!(folds[0].test_size(), 3);
    }

    #[test]
    fn fold_generator_insufficient_data() {
        let gen = CvFoldGenerator::new().initial_window(10).horizon(5);

        let folds = gen.generate(10); // Not enough for any fold
        assert!(folds.is_empty());
    }

    // ==================== Train/Test Split Tests ====================

    #[test]
    fn train_test_split_by_ratio() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let (train, test) = train_test_split(&ts, 0.8).unwrap();

        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);

        // Verify values are correct
        assert_relative_eq!(train.primary_values()[0], 0.0);
        assert_relative_eq!(train.primary_values()[79], 79.0);
        assert_relative_eq!(test.primary_values()[0], 80.0);
    }

    #[test]
    fn train_test_split_at_index() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let (train, test) = train_test_split_at(&ts, 70).unwrap();

        assert_eq!(train.len(), 70);
        assert_eq!(test.len(), 30);
    }

    #[test]
    fn train_test_split_edge_cases() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Very small ratio should give at least 1 training sample
        let (train, test) = train_test_split(&ts, 0.1).unwrap();
        assert!(train.len() >= 1);
        assert!(test.len() >= 1);

        // Very large ratio should give at least 1 test sample
        let (train, test) = train_test_split(&ts, 0.99).unwrap();
        assert!(train.len() >= 1);
        assert!(test.len() >= 1);
    }

    #[test]
    fn train_test_split_invalid() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        assert!(train_test_split_at(&ts, 0).is_err());
        assert!(train_test_split_at(&ts, 10).is_err());
    }

    // ==================== FillStrategy Tests ====================

    #[test]
    fn fill_strategy_last_value() {
        let train = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fill = LastValueFill;
        let result = fill.fill(&train, 3);

        assert_eq!(result, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn fill_strategy_mean() {
        let train = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fill = MeanFill;
        let result = fill.fill(&train, 3);

        assert_eq!(result, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn fill_strategy_median() {
        let train = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fill = MedianFill;
        let result = fill.fill(&train, 3);

        assert_eq!(result, vec![3.0, 3.0, 3.0]);

        // Even number of elements
        let train_even = vec![1.0, 2.0, 3.0, 4.0];
        let result_even = fill.fill(&train_even, 2);
        assert_eq!(result_even, vec![2.5, 2.5]);
    }

    #[test]
    fn fill_strategy_zero() {
        let train = vec![1.0, 2.0, 3.0];
        let fill = ZeroFill;
        let result = fill.fill(&train, 5);

        assert_eq!(result, vec![0.0; 5]);
    }

    #[test]
    fn fill_strategy_constant() {
        let train = vec![1.0, 2.0, 3.0];
        let fill = ConstantFill(42.0);
        let result = fill.fill(&train, 3);

        assert_eq!(result, vec![42.0, 42.0, 42.0]);
    }

    #[test]
    fn fill_strategy_mode() {
        let train = vec![1.0, 2.0, 2.0, 3.0, 2.0, 4.0];
        let fill = ModeFill;
        let result = fill.fill(&train, 3);

        assert_eq!(result, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn fill_strategy_empty_input() {
        let train: Vec<f64> = vec![];

        assert_eq!(LastValueFill.fill(&train, 2), vec![0.0, 0.0]);
        assert_eq!(MeanFill.fill(&train, 2), vec![0.0, 0.0]);
        assert_eq!(MedianFill.fill(&train, 2), vec![0.0, 0.0]);
        assert_eq!(ModeFill.fill(&train, 2), vec![0.0, 0.0]);
    }

    // ==================== CVConfig Tests ====================

    #[test]
    fn cv_config_with_gap() {
        let config = CVConfig::expanding(10, 1).with_gap(3);
        assert_eq!(config.gap, 3);

        let gen = config.to_fold_generator();
        assert_eq!(gen.gap, 3);
    }

    #[test]
    fn cv_config_with_purge() {
        let config = CVConfig::expanding(10, 1).with_purge(2);
        assert_eq!(config.purge, 2);

        let gen = config.to_fold_generator();
        assert_eq!(gen.purge, 2);
    }

    // ==================== Cross-Validation Tests ====================

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
        assert_eq!(results.folds.len(), 10);
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
    fn cv_with_gap_reduces_folds() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config_no_gap = CVConfig::expanding(10, 1);
        let config_gap = CVConfig::expanding(10, 1).with_gap(3);

        let results_no_gap = cross_validate(&config_no_gap, &ts, Naive::new).unwrap();
        let results_gap = cross_validate(&config_gap, &ts, Naive::new).unwrap();

        // Gap should reduce number of folds
        assert!(results_gap.n_folds < results_no_gap.n_folds);
    }

    #[test]
    fn cv_with_purge() {
        let timestamps = make_timestamps(25);
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1).with_purge(2);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Verify purge is applied - train_end should be origin - purge
        let first_fold = &results.folds[0];
        assert_eq!(first_fold.train_end, 8); // 10 - 2
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
        let manual_mae_mean: f64 =
            results.fold_metrics.iter().map(|m| m.mae).sum::<f64>() / results.n_folds as f64;
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
        let config = CVConfig::expanding(12, 5)
            .with_seasonal_period(4)
            .with_step_size(3);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // MASE should be computed for folds where horizon > period
        let mase_count = results
            .fold_metrics
            .iter()
            .filter(|m| m.mase.is_some())
            .count();
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
        assert_eq!(config.gap, 0);
        assert_eq!(config.purge, 0);
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

    // ==================== Grouped CV Tests ====================

    #[test]
    fn grouped_cv_basic() {
        let timestamps = make_timestamps(30);

        let series_a =
            TimeSeries::univariate(timestamps.clone(), (0..30).map(|i| i as f64).collect())
                .unwrap();

        let series_b = TimeSeries::univariate(
            timestamps.clone(),
            (0..30).map(|i| (i as f64) * 2.0).collect(),
        )
        .unwrap();

        let series_map = vec![
            ("product_a".to_string(), series_a),
            ("product_b".to_string(), series_b),
        ];

        let config = CVConfig::expanding(15, 3).with_step_size(3);
        let results = grouped_cross_validate(&config, series_map.into_iter(), Naive::new).unwrap();

        assert_eq!(results.group_results.len(), 2);
        assert!(results.aggregated.mae.is_finite());

        // Each group should have same number of folds
        let n_folds_a = results.group_results[0].1.n_folds;
        let n_folds_b = results.group_results[1].1.n_folds;
        assert_eq!(n_folds_a, n_folds_b);
    }

    #[test]
    fn grouped_cv_uses_min_length() {
        let timestamps_short = make_timestamps(20);
        let timestamps_long = make_timestamps(30);

        let series_short =
            TimeSeries::univariate(timestamps_short, (0..20).map(|i| i as f64).collect()).unwrap();

        let series_long =
            TimeSeries::univariate(timestamps_long, (0..30).map(|i| i as f64).collect()).unwrap();

        let series_map = vec![
            ("short".to_string(), series_short),
            ("long".to_string(), series_long),
        ];

        let config = CVConfig::expanding(10, 1);
        let results = grouped_cross_validate(&config, series_map.into_iter(), Naive::new).unwrap();

        // Folds should be based on min length (20)
        for (_, cv_result) in &results.group_results {
            assert_eq!(cv_result.n_folds, 10); // 20 - 10 = 10 folds
        }
    }

    #[test]
    fn grouped_cv_empty_input() {
        let series_map: Vec<(String, TimeSeries)> = vec![];
        let config = CVConfig::expanding(10, 1);

        let result = grouped_cross_validate(&config, series_map.into_iter(), Naive::new);
        assert!(result.is_err());
    }
}
