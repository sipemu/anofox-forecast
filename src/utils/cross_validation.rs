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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CVStrategy {
    /// Rolling window: fixed training window size, slides forward.
    Rolling,
    /// Expanding window: training window grows, starts from min_initial_window.
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
/// Fold generation is driven by a target number of folds (`n_folds`). The generator
/// computes step sizes and origins to produce exactly that many folds, subject to the
/// constraint that every fold's training set has at least `min_initial_window` observations.
///
/// If the constraint cannot be satisfied, the behavior depends on `on_constraint_violation`:
/// - `Error` (default): return an error
/// - `ReduceFolds`: silently reduce the number of folds until feasible
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::{CvFoldGenerator, CVStrategy};
///
/// let folds = CvFoldGenerator::new()
///     .n_folds(5)
///     .horizon(7)
///     .min_initial_window(30)
///     .strategy(CVStrategy::Expanding)
///     .generate(365)
///     .unwrap();
///
/// assert_eq!(folds.len(), 5);
/// for fold in &folds {
///     assert!(fold.train_size() >= 30);
///     println!("Train: {}..{}, Test: {}..{}",
///         fold.train_start, fold.train_end,
///         fold.test_start, fold.test_end);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CvFoldGenerator {
    /// Target number of folds to generate.
    pub target_n_folds: usize,
    /// Minimum number of observations in any training window.
    pub min_initial_window: usize,
    /// Forecast horizon (test size per fold).
    pub horizon: usize,
    /// Gap between training end and test start (prevents leakage from lagged features).
    pub gap: usize,
    /// Purge window: observations to remove before training end (prevents lookahead bias).
    pub purge: usize,
    /// Embargo: observations to exclude after each test set.
    pub embargo: usize,
    /// Cross-validation strategy (expanding or rolling).
    pub strategy: CVStrategy,
    /// What to do when `min_initial_window` cannot be satisfied for the requested folds.
    pub on_constraint_violation: ConstraintViolation,
}

/// Behavior when `min_initial_window` cannot be satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintViolation {
    /// Return an error (default).
    Error,
    /// Silently reduce the number of folds until feasible.
    ReduceFolds,
}

impl Default for CvFoldGenerator {
    fn default() -> Self {
        Self {
            target_n_folds: 5,
            min_initial_window: 10,
            horizon: 1,
            gap: 0,
            purge: 0,
            embargo: 0,
            strategy: CVStrategy::Expanding,
            on_constraint_violation: ConstraintViolation::Error,
        }
    }
}

impl CvFoldGenerator {
    /// Create a new fold generator with default settings (5 folds).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target number of folds.
    pub fn n_folds(mut self, n: usize) -> Self {
        self.target_n_folds = n;
        self
    }

    /// Set the minimum training window size constraint.
    ///
    /// Every fold's training set will have at least this many observations.
    /// If the series is too short to satisfy this for the requested number of folds,
    /// the behavior depends on `on_constraint_violation`.
    pub fn min_initial_window(mut self, size: usize) -> Self {
        self.min_initial_window = size;
        self
    }

    /// Set the forecast horizon (test size per fold).
    pub fn horizon(mut self, h: usize) -> Self {
        self.horizon = h;
        self
    }

    /// Set the step size between folds.
    ///
    /// **Deprecated**: step size is now computed automatically from `n_folds`.
    /// This method is kept for backwards compatibility but is ignored when
    /// `target_n_folds > 0`.
    pub fn step_size(mut self, _step: usize) -> Self {
        // Ignored — step is computed from n_folds and series_len.
        // Kept for API compatibility during transition.
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

    /// Set the embargo window size.
    pub fn embargo(mut self, e: usize) -> Self {
        self.embargo = e;
        self
    }

    /// Set the cross-validation strategy.
    pub fn strategy(mut self, s: CVStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Set behavior when `min_initial_window` cannot be satisfied.
    ///
    /// - `ConstraintViolation::Error` (default): return an error
    /// - `ConstraintViolation::ReduceFolds`: silently reduce the number of folds
    pub fn on_constraint_violation(mut self, behavior: ConstraintViolation) -> Self {
        self.on_constraint_violation = behavior;
        self
    }

    /// Ensure the last fold's test window reaches the series end.
    ///
    /// This is now always true — the last fold is always anchored at the series end.
    /// Kept for API compatibility.
    pub fn ensure_end_coverage(self, _enable: bool) -> Self {
        self
    }

    /// Generate fold indices for a series of given length.
    ///
    /// Places `target_n_folds` folds working backwards from the series end,
    /// ensuring every fold's training set has at least `min_initial_window`
    /// observations. The last fold always covers the series end.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The series is too short for even one fold with the given constraints
    /// - `min_initial_window` cannot be satisfied and `on_constraint_violation` is `Error`
    pub fn generate(&self, series_len: usize) -> crate::error::Result<Vec<Fold>> {
        use crate::error::ForecastError;

        let horizon = self.horizon.max(1);
        let gap = self.gap;
        let purge = self.purge;
        let min_train = self.min_initial_window.max(1);

        // Minimum series length: min_train + purge + gap + horizon
        let min_series = min_train + purge + gap + horizon;
        if series_len < min_series {
            return Err(ForecastError::InsufficientData {
                needed: min_series,
                got: series_len,
                hint: Some(format!(
                    "need at least min_initial_window({}) + purge({}) + gap({}) + horizon({})",
                    min_train, purge, gap, horizon
                )),
            });
        }

        // Available space for placing fold origins:
        // Last fold: test_end = series_len, test_start = series_len - horizon
        // First fold origin must be >= min_train + purge
        let last_origin = series_len - gap - horizon;
        let first_origin = min_train + purge;

        if last_origin < first_origin {
            return Err(ForecastError::InsufficientData {
                needed: min_series,
                got: series_len,
                hint: Some("series too short for constraints".into()),
            });
        }

        let available_range = last_origin - first_origin;
        let mut n_folds = self.target_n_folds.max(1);

        // Compute step size to spread folds evenly
        let step = if n_folds <= 1 {
            1
        } else {
            let step = available_range / (n_folds - 1);
            if step == 0 {
                // Not enough room for requested folds
                match self.on_constraint_violation {
                    ConstraintViolation::Error => {
                        return Err(ForecastError::InvalidParameter(format!(
                            "cannot fit {} folds: only {} positions available with \
                             min_initial_window={}, horizon={}, gap={}, purge={}",
                            n_folds,
                            available_range + 1,
                            min_train,
                            horizon,
                            gap,
                            purge
                        )));
                    }
                    ConstraintViolation::ReduceFolds => {
                        // Reduce to max feasible: one fold per available position
                        n_folds = (available_range + 1).min(n_folds);
                        if n_folds <= 1 {
                            1
                        } else {
                            available_range / (n_folds - 1)
                        }
                    }
                }
            } else {
                step
            }
        };

        let mut folds = Vec::with_capacity(n_folds);
        let mut max_embargo_end: usize = 0;

        for i in 0..n_folds {
            let origin = if n_folds <= 1 || i == n_folds - 1 {
                // Last fold is always anchored at the series end
                last_origin
            } else {
                first_origin + i * step
            };

            let base_train_start = match self.strategy {
                CVStrategy::Rolling => origin.saturating_sub(min_train + purge),
                CVStrategy::Expanding => 0,
            };
            let train_end = origin.saturating_sub(purge);
            let train_start = if self.embargo > 0 && max_embargo_end > base_train_start {
                max_embargo_end.min(train_end)
            } else {
                base_train_start
            };

            if train_end <= train_start {
                // Skip this fold — embargo ate the training set
                let test_end = (origin + gap + horizon).min(series_len);
                let embargo_end = (test_end + self.embargo).min(series_len);
                if embargo_end > max_embargo_end {
                    max_embargo_end = embargo_end;
                }
                continue;
            }

            let test_start = origin + gap;
            let test_end = (test_start + horizon).min(series_len);

            folds.push(Fold {
                train_start,
                train_end,
                test_start,
                test_end,
            });

            if self.embargo > 0 {
                let embargo_end = (test_end + self.embargo).min(series_len);
                if embargo_end > max_embargo_end {
                    max_embargo_end = embargo_end;
                }
            }
        }

        Ok(folds)
    }
}

/// Configuration for time series cross-validation.
#[derive(Debug, Clone)]
pub struct CVConfig {
    /// Forecast horizon for each fold.
    pub horizon: usize,
    /// Minimum number of observations in the first training window.
    pub min_initial_window: usize,
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
    /// Embargo: observations to exclude after each test set.
    pub embargo: usize,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            horizon: 1,
            min_initial_window: 10,
            step_size: 1,
            strategy: CVStrategy::Expanding,
            seasonal_period: None,
            gap: 0,
            purge: 0,
            embargo: 0,
        }
    }
}

impl CVConfig {
    /// Create a new CV configuration with expanding window strategy.
    pub fn expanding(min_initial_window: usize, horizon: usize) -> Self {
        Self {
            min_initial_window,
            horizon,
            step_size: 1,
            strategy: CVStrategy::Expanding,
            seasonal_period: None,
            gap: 0,
            purge: 0,
            embargo: 0,
        }
    }

    /// Create a new CV configuration with rolling window strategy.
    pub fn rolling(window_size: usize, horizon: usize) -> Self {
        Self {
            min_initial_window: window_size,
            horizon,
            step_size: 1,
            strategy: CVStrategy::Rolling,
            seasonal_period: None,
            gap: 0,
            purge: 0,
            embargo: 0,
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

    /// Set the embargo window size.
    pub fn with_embargo(mut self, embargo: usize) -> Self {
        self.embargo = embargo;
        self
    }

    /// Convert to a CvFoldGenerator.
    ///
    /// The step_size from CVConfig is used to estimate the target number of folds.
    pub fn to_fold_generator(&self) -> CvFoldGenerator {
        CvFoldGenerator {
            target_n_folds: 5, // default; callers can override
            min_initial_window: self.min_initial_window,
            horizon: self.horizon,
            gap: self.gap,
            purge: self.purge,
            embargo: self.embargo,
            strategy: self.strategy,
            on_constraint_violation: ConstraintViolation::ReduceFolds,
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

    let horizon = fold.test_size();
    let forecast = if model.has_exog() {
        // Extract future regressor values from the test portion of the series
        let test_series = series.slice(fold.test_start, fold.test_end)?;
        let future_regs = test_series.all_regressors();
        model.predict_with_exog(horizon, &future_regs)?
    } else {
        model.predict(horizon)?
    };

    let predicted = forecast.primary();
    // Use direct slice to avoid collecting a Vec just for metrics computation
    let actual_slice = &series.primary_values()[fold.test_start..fold.test_end];
    let metrics = calculate_metrics(actual_slice, predicted, seasonal_period)?;
    Ok((metrics, actual_slice.to_vec(), predicted.to_vec()))
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
    let folds = generator.generate(series.len())?;

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
/// let results = grouped_cross_validate(&config, series_map, Naive::new).unwrap();
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
    let folds = generator.generate(min_len)?;

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
    F: Forecaster + Send,
    Factory: Fn() -> F + Sync,
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

    let seasonal_period = config.seasonal_period;

    // Evaluate each fold (parallel when feature enabled)
    #[cfg(feature = "parallel")]
    let fold_results: Vec<Result<(AccuracyMetrics, Vec<f64>, Vec<f64>)>> = folds
        .par_iter()
        .map(|fold| evaluate_fold(series, fold, model_factory, seasonal_period))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let fold_results: Vec<Result<(AccuracyMetrics, Vec<f64>, Vec<f64>)>> = folds
        .iter()
        .map(|fold| evaluate_fold(series, fold, model_factory, seasonal_period))
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
        folds: folds.to_vec(),
    })
}

/// Streaming cross-validation aggregator using Welford's online algorithm.
///
/// Accumulates fold metrics incrementally without storing all individual results,
/// enabling early stopping and convergence monitoring for large-scale CV.
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::StreamingCVAggregator;
/// use anofox_forecast::utils::metrics::AccuracyMetrics;
///
/// let mut agg = StreamingCVAggregator::new();
///
/// // Add fold results one at a time
/// let metrics = AccuracyMetrics {
///     mae: 1.5, mse: 4.0, rmse: 2.0, smape: 10.0, mape: Some(8.0), mase: None, r_squared: 0.9,
/// };
/// agg.update(&metrics);
///
/// assert_eq!(agg.n_folds(), 1);
/// assert!((agg.mean_mae() - 1.5).abs() < 1e-10);
///
/// // Check convergence: has the running mean stabilized?
/// assert!(!agg.has_converged(0.01)); // need at least 3 folds
/// ```
#[derive(Debug, Clone)]
pub struct StreamingCVAggregator {
    count: usize,
    // Welford accumulators: (mean, M2) for online variance
    mae_mean: f64,
    mae_m2: f64,
    rmse_mean: f64,
    rmse_m2: f64,
    smape_mean: f64,
    smape_m2: f64,
    mape_mean: f64,
    mape_m2: f64,
    mape_count: usize,
    // Track last mean for convergence check
    prev_mae_mean: f64,
}

impl StreamingCVAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mae_mean: 0.0,
            mae_m2: 0.0,
            rmse_mean: 0.0,
            rmse_m2: 0.0,
            smape_mean: 0.0,
            smape_m2: 0.0,
            mape_mean: 0.0,
            mape_m2: 0.0,
            mape_count: 0,
            prev_mae_mean: f64::NAN,
        }
    }

    /// Add a fold's metrics to the running aggregation.
    pub fn update(&mut self, metrics: &AccuracyMetrics) {
        self.prev_mae_mean = self.mae_mean;
        self.count += 1;
        let n = self.count as f64;

        // Welford update for MAE
        let delta = metrics.mae - self.mae_mean;
        self.mae_mean += delta / n;
        let delta2 = metrics.mae - self.mae_mean;
        self.mae_m2 += delta * delta2;

        // Welford update for RMSE
        let delta = metrics.rmse - self.rmse_mean;
        self.rmse_mean += delta / n;
        let delta2 = metrics.rmse - self.rmse_mean;
        self.rmse_m2 += delta * delta2;

        // Welford update for SMAPE
        let delta = metrics.smape - self.smape_mean;
        self.smape_mean += delta / n;
        let delta2 = metrics.smape - self.smape_mean;
        self.smape_m2 += delta * delta2;

        // Welford update for MAPE (if available)
        if let Some(mape) = metrics.mape {
            self.mape_count += 1;
            let mn = self.mape_count as f64;
            let delta = mape - self.mape_mean;
            self.mape_mean += delta / mn;
            let delta2 = mape - self.mape_mean;
            self.mape_m2 += delta * delta2;
        }
    }

    /// Number of folds accumulated so far.
    pub fn n_folds(&self) -> usize {
        self.count
    }

    /// Running mean MAE.
    pub fn mean_mae(&self) -> f64 {
        self.mae_mean
    }

    /// Running mean RMSE.
    pub fn mean_rmse(&self) -> f64 {
        self.rmse_mean
    }

    /// Running mean SMAPE.
    pub fn mean_smape(&self) -> f64 {
        self.smape_mean
    }

    /// Running mean MAPE (None if no fold had MAPE).
    pub fn mean_mape(&self) -> Option<f64> {
        if self.mape_count > 0 {
            Some(self.mape_mean)
        } else {
            None
        }
    }

    /// Running sample standard deviation of MAE.
    pub fn std_mae(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.mae_m2 / (self.count - 1) as f64).sqrt()
    }

    /// Running sample standard deviation of RMSE.
    pub fn std_rmse(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.rmse_m2 / (self.count - 1) as f64).sqrt()
    }

    /// Check whether the running MAE mean has converged within `tolerance`.
    ///
    /// Returns `true` if at least 3 folds have been processed and the
    /// relative change in mean MAE from the previous fold is below `tolerance`.
    pub fn has_converged(&self, tolerance: f64) -> bool {
        if self.count < 3 || self.prev_mae_mean.is_nan() {
            return false;
        }
        let change = (self.mae_mean - self.prev_mae_mean).abs();
        let scale = self.mae_mean.abs().max(1e-10);
        change / scale < tolerance
    }

    /// Convert the accumulated state into [`AggregatedMetrics`].
    pub fn finalize(&self) -> AggregatedMetrics {
        AggregatedMetrics {
            mae: self.mae_mean,
            rmse: self.rmse_mean,
            smape: self.smape_mean,
            mape: self.mean_mape(),
            mae_std: self.std_mae(),
            rmse_std: self.std_rmse(),
        }
    }
}

impl Default for StreamingCVAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform cross-validation with early stopping based on convergence.
///
/// Like [`cross_validate`], but stops adding folds once the running MAE
/// mean has stabilized within `tolerance`. Returns results from however
/// many folds were evaluated.
///
/// # Arguments
/// * `config` - Cross-validation configuration
/// * `series` - The time series to validate on
/// * `model_factory` - Function that creates a fresh model for each fold
/// * `tolerance` - Relative change threshold for convergence (e.g., 0.01 for 1%)
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::{cross_validate_early_stop, CVConfig};
/// use anofox_forecast::models::baseline::Naive;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..50)
///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
///         + chrono::Duration::hours(i))
///     .collect();
/// let values = vec![5.0; 50]; // constant series
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let config = CVConfig::expanding(10, 1);
/// let results = cross_validate_early_stop(&config, &ts, Naive::new, 0.01).unwrap();
/// // May stop before all 40 folds if MAE converges quickly
/// assert!(results.n_folds >= 3);
/// ```
pub fn cross_validate_early_stop<F, Factory>(
    config: &CVConfig,
    series: &TimeSeries,
    model_factory: Factory,
    tolerance: f64,
) -> Result<CVResults>
where
    F: Forecaster,
    Factory: Fn() -> F,
{
    let generator = config.to_fold_generator();
    let folds = generator.generate(series.len())?;

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

    let mut aggregator = StreamingCVAggregator::new();
    let mut fold_metrics = Vec::new();
    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();
    let mut used_folds = Vec::new();

    for fold in &folds {
        let (metrics, actual, predicted) =
            evaluate_fold(series, fold, &model_factory, config.seasonal_period)?;

        aggregator.update(&metrics);
        fold_metrics.push(metrics);
        all_actual.extend_from_slice(&actual);
        all_predicted.extend_from_slice(&predicted);
        used_folds.push(fold.clone());

        if aggregator.has_converged(tolerance) {
            break;
        }
    }

    Ok(CVResults {
        n_folds: fold_metrics.len(),
        aggregated: aggregator.finalize(),
        fold_metrics,
        actual_values: all_actual,
        predicted_values: all_predicted,
        folds: used_folds,
    })
}

/// Configuration for rolling/expanding window forecast evaluation.
///
/// Walk-forward evaluation: train on a window, predict the next `horizon` steps,
/// step forward, and repeat. Supports both rolling (fixed-size) and expanding
/// (growing) training windows.
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::RollingForecastConfig;
///
/// let config = RollingForecastConfig::new(50, 7)
///     .step_size(7)
///     .expanding(false); // rolling window
///
/// assert_eq!(config.initial_train_size, 50);
/// assert_eq!(config.horizon, 7);
/// assert_eq!(config.step_size, 7);
/// assert!(!config.expanding);
/// ```
#[derive(Debug, Clone)]
pub struct RollingForecastConfig {
    /// Minimum number of observations for the first training window.
    pub initial_train_size: usize,
    /// Number of steps to forecast at each window position.
    pub horizon: usize,
    /// Number of steps to advance the window origin between iterations.
    pub step_size: usize,
    /// If `true`, the training window grows from the start of the series (expanding).
    /// If `false`, the training window has a fixed size equal to `initial_train_size` (rolling).
    pub expanding: bool,
}

impl RollingForecastConfig {
    /// Create a new configuration with expanding window (default).
    ///
    /// Step size defaults to `horizon` so that forecast windows are non-overlapping.
    pub fn new(initial_train_size: usize, horizon: usize) -> Self {
        Self {
            initial_train_size,
            horizon,
            step_size: horizon,
            expanding: true,
        }
    }

    /// Set the step size between successive forecast origins.
    pub fn step_size(mut self, step: usize) -> Self {
        self.step_size = step;
        self
    }

    /// Set the window mode: `true` for expanding, `false` for rolling.
    pub fn expanding(mut self, expanding: bool) -> Self {
        self.expanding = expanding;
        self
    }
}

/// A single window's predictions and actuals from rolling forecast evaluation.
#[derive(Debug, Clone)]
pub struct RollingForecastWindow {
    /// Start index of the training data (inclusive).
    pub train_start: usize,
    /// End index of the training data (exclusive).
    pub train_end: usize,
    /// Predicted values for this window.
    pub predictions: Vec<f64>,
    /// Actual values for this window.
    pub actuals: Vec<f64>,
}

/// Results from a rolling/expanding window forecast evaluation.
#[derive(Debug, Clone)]
pub struct RollingForecastResult {
    /// Per-window results in chronological order.
    pub windows: Vec<RollingForecastWindow>,
    /// All predictions concatenated in order.
    pub all_predictions: Vec<f64>,
    /// All actuals concatenated in order.
    pub all_actuals: Vec<f64>,
    /// Per-window accuracy metrics.
    pub window_metrics: Vec<AccuracyMetrics>,
    /// Aggregated metrics across all windows.
    pub aggregated: AggregatedMetrics,
}

/// Perform rolling or expanding window forecast evaluation.
///
/// Walk-forward evaluation trains a model on historical data, generates a forecast
/// of length `horizon`, records predictions vs actuals, then steps forward and
/// repeats. This produces realistic out-of-sample accuracy estimates.
///
/// # Arguments
/// * `series` - The full time series to evaluate on
/// * `config` - Rolling forecast configuration
/// * `model_factory` - Function that creates a fresh model for each window
///
/// # Returns
/// `RollingForecastResult` containing per-window and aggregated metrics.
///
/// # Example
///
/// ```
/// use anofox_forecast::utils::cross_validation::{rolling_forecast, RollingForecastConfig};
/// use anofox_forecast::models::baseline::Naive;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..30)
///     .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
///         + chrono::Duration::hours(i))
///     .collect();
/// let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let config = RollingForecastConfig::new(20, 3).step_size(3);
/// let result = rolling_forecast(&ts, &config, Naive::new).unwrap();
///
/// assert!(result.windows.len() > 0);
/// assert_eq!(result.all_predictions.len(), result.all_actuals.len());
/// ```
pub fn rolling_forecast<F, Factory>(
    series: &TimeSeries,
    config: &RollingForecastConfig,
    model_factory: Factory,
) -> Result<RollingForecastResult>
where
    F: Forecaster + Send,
    Factory: Fn() -> F + Sync,
{
    let n = series.len();

    if config.initial_train_size == 0 {
        return Err(ForecastError::InvalidParameter(
            "initial_train_size must be at least 1".to_string(),
        ));
    }
    if config.horizon == 0 {
        return Err(ForecastError::InvalidParameter(
            "horizon must be at least 1".to_string(),
        ));
    }
    if config.step_size == 0 {
        return Err(ForecastError::InvalidParameter(
            "step_size must be at least 1".to_string(),
        ));
    }
    if config.initial_train_size + config.horizon > n {
        return Err(ForecastError::InvalidParameter(format!(
            "Series length ({}) is too short for initial_train_size ({}) + horizon ({})",
            n, config.initial_train_size, config.horizon
        )));
    }

    // Build the list of (train_start, train_end) pairs for each window
    let mut window_specs: Vec<(usize, usize)> = Vec::new();
    let mut origin = config.initial_train_size;
    while origin + config.horizon <= n {
        let train_start = if config.expanding {
            0
        } else {
            origin.saturating_sub(config.initial_train_size)
        };
        let train_end = origin;
        window_specs.push((train_start, train_end));
        origin += config.step_size;
    }

    if window_specs.is_empty() {
        return Err(ForecastError::InvalidParameter(
            "Not enough data for any forecast window".to_string(),
        ));
    }

    let horizon = config.horizon;
    let values = series.primary_values();

    // Evaluate each window
    let evaluate_window =
        |&(train_start, train_end): &(usize, usize)| -> Result<(RollingForecastWindow, AccuracyMetrics)> {
            let train_series = series.slice(train_start, train_end)?;
            let mut model = model_factory();
            model.fit(&train_series)?;
            let forecast = model.predict(horizon)?;
            let predictions: Vec<f64> = forecast.primary().to_vec();

            let test_end = train_end + horizon;
            let actuals: Vec<f64> = (train_end..test_end).map(|i| values[i]).collect();

            let metrics = calculate_metrics(&actuals, &predictions, None)?;

            let window = RollingForecastWindow {
                train_start,
                train_end,
                predictions,
                actuals,
            };
            Ok((window, metrics))
        };

    #[cfg(feature = "parallel")]
    let results: Vec<Result<(RollingForecastWindow, AccuracyMetrics)>> =
        window_specs.par_iter().map(evaluate_window).collect();

    #[cfg(not(feature = "parallel"))]
    let results: Vec<Result<(RollingForecastWindow, AccuracyMetrics)>> =
        window_specs.iter().map(evaluate_window).collect();

    let mut windows = Vec::with_capacity(results.len());
    let mut window_metrics = Vec::with_capacity(results.len());
    let mut all_predictions = Vec::new();
    let mut all_actuals = Vec::new();

    for result in results {
        let (window, metrics) = result?;
        all_predictions.extend_from_slice(&window.predictions);
        all_actuals.extend_from_slice(&window.actuals);
        windows.push(window);
        window_metrics.push(metrics);
    }

    let n_windows = window_metrics.len();

    // Aggregate metrics
    let mae_values: Vec<f64> = window_metrics.iter().map(|m| m.mae).collect();
    let rmse_values: Vec<f64> = window_metrics.iter().map(|m| m.rmse).collect();
    let smape_values: Vec<f64> = window_metrics.iter().map(|m| m.smape).collect();

    let mae_mean = mae_values.iter().sum::<f64>() / n_windows as f64;
    let rmse_mean = rmse_values.iter().sum::<f64>() / n_windows as f64;
    let smape_mean = smape_values.iter().sum::<f64>() / n_windows as f64;

    let mae_std = std_dev(&mae_values);
    let rmse_std = std_dev(&rmse_values);

    let mape = if window_metrics.iter().all(|m| m.mape.is_some()) {
        let mape_values: Vec<f64> = window_metrics.iter().filter_map(|m| m.mape).collect();
        Some(mape_values.iter().sum::<f64>() / n_windows as f64)
    } else {
        None
    };

    let aggregated = AggregatedMetrics {
        mae: mae_mean,
        rmse: rmse_mean,
        smape: smape_mean,
        mape,
        mae_std,
        rmse_std,
    };

    Ok(RollingForecastResult {
        windows,
        all_predictions,
        all_actuals,
        window_metrics,
        aggregated,
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
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .min_initial_window(10)
            .horizon(1)
            .generate(50)
            .unwrap();

        assert_eq!(folds.len(), 5);
        // First fold starts at min_initial_window
        assert!(folds[0].train_size() >= 10);
        // Last fold reaches the end
        assert_eq!(folds.last().unwrap().test_end, 50);
    }

    #[test]
    fn fold_generator_with_gap() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .min_initial_window(10)
            .horizon(1)
            .gap(2)
            .generate(50)
            .unwrap();

        // With gap=2, test starts at train_end + 2
        for fold in &folds {
            assert!(fold.test_start >= fold.train_end + 2);
        }
    }

    #[test]
    fn fold_generator_with_purge() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .min_initial_window(10)
            .horizon(1)
            .purge(2)
            .generate(50)
            .unwrap();

        // With purge=2, train_end < origin (origin = train_end + purge)
        // Test that folds are valid
        for fold in &folds {
            assert!(fold.train_end < fold.test_start);
        }
    }

    #[test]
    fn fold_generator_rolling() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .min_initial_window(10)
            .horizon(1)
            .strategy(CVStrategy::Rolling)
            .generate(50)
            .unwrap();

        // Rolling: all folds have same train size
        for fold in &folds {
            assert_eq!(fold.train_size(), 10);
        }
    }

    #[test]
    fn fold_generator_multi_step_horizon() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .min_initial_window(10)
            .horizon(3)
            .generate(50)
            .unwrap();

        for fold in &folds {
            assert_eq!(fold.test_size(), 3);
        }
    }

    #[test]
    fn fold_generator_insufficient_data() {
        // Series too short for constraints
        let result = CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(5)
            .generate(10);
        assert!(result.is_err());
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
        assert!(!train.is_empty());
        assert!(!test.is_empty());

        // Very large ratio should give at least 1 test sample
        let (train, test) = train_test_split(&ts, 0.99).unwrap();
        assert!(!train.is_empty());
        assert!(!test.is_empty());
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

        // Default 5 folds (or fewer if constrained)
        assert!(results.n_folds > 0);
        assert!(results.aggregated.mae.is_finite());
        assert_eq!(results.folds.len(), results.n_folds);
    }

    #[test]
    fn cv_rolling_window_basic() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::rolling(10, 1);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        assert!(results.n_folds > 0);
        assert!(results.aggregated.mae.is_finite());
    }

    #[test]
    fn cv_with_gap() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1).with_gap(3);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Gap should be respected in all folds
        for fold in &results.folds {
            assert!(fold.test_start >= fold.train_end + 3);
        }
    }

    #[test]
    fn cv_with_purge() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1).with_purge(2);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        // Purge should create separation between train_end and test_start
        for fold in &results.folds {
            assert!(fold.test_start > fold.train_end);
        }
    }

    #[test]
    fn cv_multi_step_horizon() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 3);
        let results = cross_validate(&config, &ts, Naive::new).unwrap();

        assert!(results.n_folds > 0);
        // Each fold should have 3 predictions
        assert_eq!(results.actual_values.len(), results.n_folds * 3);
        assert_eq!(results.predicted_values.len(), results.n_folds * 3);
    }

    #[test]
    fn cv_insufficient_data_returns_error_or_zero_folds() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // min_initial_window=10 but only 5 data points
        let config = CVConfig::expanding(10, 1);
        // Should either error or return 0 folds
        match cross_validate(&config, &ts, Naive::new) {
            Ok(results) => assert_eq!(results.n_folds, 0),
            Err(_) => {} // also acceptable
        }
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
        assert_eq!(expanding.min_initial_window, 10);
        assert_eq!(expanding.horizon, 3);
        assert_eq!(expanding.strategy, CVStrategy::Expanding);

        let rolling = CVConfig::rolling(15, 2);
        assert_eq!(rolling.min_initial_window, 15);
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
        assert_eq!(config.min_initial_window, 10);
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
        let results = grouped_cross_validate(&config, series_map, Naive::new).unwrap();

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
        let results = grouped_cross_validate(&config, series_map, Naive::new).unwrap();

        // All groups should have the same number of folds (based on min length)
        let fold_counts: Vec<_> = results
            .group_results
            .iter()
            .map(|(_, r)| r.n_folds)
            .collect();
        assert!(fold_counts.iter().all(|&n| n == fold_counts[0]));
        assert!(fold_counts[0] > 0);
    }

    #[test]
    fn grouped_cv_empty_input() {
        let series_map: Vec<(String, TimeSeries)> = vec![];
        let config = CVConfig::expanding(10, 1);

        let result = grouped_cross_validate(&config, series_map, Naive::new);
        assert!(result.is_err());
    }

    // ==================== StreamingCVAggregator Tests ====================

    #[test]
    fn streaming_aggregator_single_fold() {
        let mut agg = StreamingCVAggregator::new();
        let metrics = AccuracyMetrics {
            mae: 2.0,
            mse: 0.0,
            rmse: 3.0,
            smape: 15.0,
            mape: Some(10.0),
            mase: None,
            r_squared: 0.0,
        };
        agg.update(&metrics);

        assert_eq!(agg.n_folds(), 1);
        assert_relative_eq!(agg.mean_mae(), 2.0);
        assert_relative_eq!(agg.mean_rmse(), 3.0);
        assert_relative_eq!(agg.mean_smape(), 15.0);
        assert_relative_eq!(agg.mean_mape().unwrap(), 10.0);
        assert_relative_eq!(agg.std_mae(), 0.0);
    }

    #[test]
    fn streaming_aggregator_matches_batch() {
        let fold_metrics = vec![
            AccuracyMetrics {
                mae: 1.0,
                mse: 0.0,
                rmse: 1.5,
                smape: 10.0,
                mape: Some(8.0),
                mase: None,
                r_squared: 0.0,
            },
            AccuracyMetrics {
                mae: 2.0,
                mse: 0.0,
                rmse: 2.5,
                smape: 12.0,
                mape: Some(9.0),
                mase: None,
                r_squared: 0.0,
            },
            AccuracyMetrics {
                mae: 3.0,
                mse: 0.0,
                rmse: 3.5,
                smape: 14.0,
                mape: Some(11.0),
                mase: None,
                r_squared: 0.0,
            },
        ];

        let mut agg = StreamingCVAggregator::new();
        for m in &fold_metrics {
            agg.update(m);
        }

        // Compare with batch computation
        let mae_vals: Vec<f64> = fold_metrics.iter().map(|m| m.mae).collect();
        let batch_mean = mae_vals.iter().sum::<f64>() / mae_vals.len() as f64;
        let batch_std = std_dev(&mae_vals);

        assert_relative_eq!(agg.mean_mae(), batch_mean, epsilon = 1e-10);
        assert_relative_eq!(agg.std_mae(), batch_std, epsilon = 1e-10);
        assert_eq!(agg.n_folds(), 3);
    }

    #[test]
    fn streaming_aggregator_convergence() {
        let mut agg = StreamingCVAggregator::new();

        // First two folds: not enough for convergence
        agg.update(&AccuracyMetrics {
            mae: 1.0,
            mse: 0.0,
            rmse: 1.0,
            smape: 5.0,
            mape: None,
            mase: None,
            r_squared: 0.0,
        });
        assert!(!agg.has_converged(0.01));

        agg.update(&AccuracyMetrics {
            mae: 1.0,
            mse: 0.0,
            rmse: 1.0,
            smape: 5.0,
            mape: None,
            mase: None,
            r_squared: 0.0,
        });
        assert!(!agg.has_converged(0.01));

        // Third fold with same value: should converge
        agg.update(&AccuracyMetrics {
            mae: 1.0,
            mse: 0.0,
            rmse: 1.0,
            smape: 5.0,
            mape: None,
            mase: None,
            r_squared: 0.0,
        });
        assert!(agg.has_converged(0.01));
    }

    #[test]
    fn streaming_aggregator_no_mape() {
        let mut agg = StreamingCVAggregator::new();
        agg.update(&AccuracyMetrics {
            mae: 1.0,
            mse: 0.0,
            rmse: 1.0,
            smape: 5.0,
            mape: None,
            mase: None,
            r_squared: 0.0,
        });
        assert!(agg.mean_mape().is_none());
    }

    #[test]
    fn streaming_aggregator_finalize() {
        let mut agg = StreamingCVAggregator::new();
        agg.update(&AccuracyMetrics {
            mae: 2.0,
            mse: 0.0,
            rmse: 3.0,
            smape: 10.0,
            mape: Some(5.0),
            mase: None,
            r_squared: 0.0,
        });
        agg.update(&AccuracyMetrics {
            mae: 4.0,
            mse: 0.0,
            rmse: 5.0,
            smape: 20.0,
            mape: Some(15.0),
            mase: None,
            r_squared: 0.0,
        });

        let result = agg.finalize();
        assert_relative_eq!(result.mae, 3.0);
        assert_relative_eq!(result.rmse, 4.0);
        assert_relative_eq!(result.smape, 15.0);
        assert_relative_eq!(result.mape.unwrap(), 10.0);
    }

    // ==================== Early Stop CV Tests ====================

    #[test]
    fn cv_early_stop_constant_series() {
        let timestamps = make_timestamps(50);
        let values = vec![5.0; 50];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1);
        let results = cross_validate_early_stop(&config, &ts, Naive::new, 0.01).unwrap();

        // Naive on constant series has 0 MAE from the start,
        // so it should stop at minimum folds (3)
        assert!(results.n_folds >= 3);
        assert!(results.n_folds < 40); // should stop well before all folds
        assert_relative_eq!(results.aggregated.mae, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cv_early_stop_runs_all_if_needed() {
        let timestamps = make_timestamps(20);
        // Highly variable series that won't converge quickly
        let values: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 100.0 } else { 0.0 })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = CVConfig::expanding(10, 1);
        // Very tight tolerance: may run all folds
        let results = cross_validate_early_stop(&config, &ts, Naive::new, 1e-15).unwrap();

        // Should have evaluated all available folds
        assert!(results.n_folds > 0);
    }

    // ==================== Rolling Forecast Tests ====================

    #[test]
    fn rolling_forecast_expanding_basic() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = RollingForecastConfig::new(20, 3).step_size(3);
        let result = rolling_forecast(&ts, &config, Naive::new).unwrap();

        // With 30 points, initial=20, horizon=3, step=3:
        // Window 0: train [0..20], test [20..23]
        // Window 1: train [0..23], test [23..26]
        // Window 2: train [0..26], test [26..29]
        // Window 3: train [0..29], test [29..32] -> exceeds, so 3 windows
        assert_eq!(result.windows.len(), 3);
        assert_eq!(result.all_predictions.len(), 9);
        assert_eq!(result.all_actuals.len(), 9);

        // Expanding: train_start should always be 0
        for w in &result.windows {
            assert_eq!(w.train_start, 0);
        }

        // Train end should grow
        assert_eq!(result.windows[0].train_end, 20);
        assert_eq!(result.windows[1].train_end, 23);
        assert_eq!(result.windows[2].train_end, 26);
    }

    #[test]
    fn rolling_forecast_fixed_window() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = RollingForecastConfig::new(15, 3)
            .step_size(3)
            .expanding(false);
        let result = rolling_forecast(&ts, &config, Naive::new).unwrap();

        // Rolling: train window stays fixed at 15
        for w in &result.windows {
            assert_eq!(w.train_end - w.train_start, 15);
        }

        // train_start should slide forward
        assert_eq!(result.windows[0].train_start, 0);
        assert_eq!(result.windows[1].train_start, 3);
    }

    #[test]
    fn rolling_forecast_step_size_one() {
        let timestamps = make_timestamps(25);
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = RollingForecastConfig::new(20, 3).step_size(1);
        let result = rolling_forecast(&ts, &config, Naive::new).unwrap();

        // origin goes 20, 21, 22 (22+3=25 is ok), 23 (23+3=26 > 25, stop)
        assert_eq!(result.windows.len(), 3);
    }

    #[test]
    fn rolling_forecast_constant_series() {
        let timestamps = make_timestamps(30);
        let values = vec![5.0; 30];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = RollingForecastConfig::new(20, 3).step_size(3);
        let result = rolling_forecast(&ts, &config, Naive::new).unwrap();

        // Naive on constant series should have zero error
        assert!(result.aggregated.mae.abs() < 1e-10);
        assert!(result.aggregated.rmse.abs() < 1e-10);

        // Predictions should equal actuals
        for (p, a) in result.all_predictions.iter().zip(result.all_actuals.iter()) {
            assert!((p - a).abs() < 1e-10);
        }
    }

    #[test]
    fn rolling_forecast_actuals_match_series() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| i as f64 * 2.0 + 1.0).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let config = RollingForecastConfig::new(20, 5).step_size(5);
        let result = rolling_forecast(&ts, &config, Naive::new).unwrap();

        // Verify actuals come from the original series
        for w in &result.windows {
            for (j, &actual) in w.actuals.iter().enumerate() {
                let idx = w.train_end + j;
                assert!((actual - values[idx]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rolling_forecast_insufficient_data() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = RollingForecastConfig::new(10, 5);
        let result = rolling_forecast(&ts, &config, Naive::new);

        // 10 + 5 > 10, not enough data
        assert!(result.is_err());
    }

    #[test]
    fn rolling_forecast_invalid_params() {
        let timestamps = make_timestamps(30);
        let values = vec![1.0; 30];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Zero horizon
        let config = RollingForecastConfig {
            initial_train_size: 20,
            horizon: 0,
            step_size: 1,
            expanding: true,
        };
        assert!(rolling_forecast(&ts, &config, Naive::new).is_err());

        // Zero step_size
        let config = RollingForecastConfig {
            initial_train_size: 20,
            horizon: 3,
            step_size: 0,
            expanding: true,
        };
        assert!(rolling_forecast(&ts, &config, Naive::new).is_err());

        // Zero initial_train_size
        let config = RollingForecastConfig {
            initial_train_size: 0,
            horizon: 3,
            step_size: 1,
            expanding: true,
        };
        assert!(rolling_forecast(&ts, &config, Naive::new).is_err());
    }

    #[test]
    fn rolling_forecast_config_builder() {
        let config = RollingForecastConfig::new(50, 7)
            .step_size(3)
            .expanding(false);

        assert_eq!(config.initial_train_size, 50);
        assert_eq!(config.horizon, 7);
        assert_eq!(config.step_size, 3);
        assert!(!config.expanding);
    }

    #[test]
    fn rolling_forecast_config_defaults() {
        let config = RollingForecastConfig::new(100, 12);

        assert_eq!(config.initial_train_size, 100);
        assert_eq!(config.horizon, 12);
        assert_eq!(config.step_size, 12); // defaults to horizon
        assert!(config.expanding); // defaults to true
    }

    #[test]
    fn rolling_forecast_metrics_consistent() {
        let timestamps = make_timestamps(40);
        let values: Vec<f64> = (0..40).map(|i| (i as f64).sin() * 10.0 + 50.0).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let config = RollingForecastConfig::new(25, 3).step_size(3);
        let result = rolling_forecast(&ts, &config, Naive::new).unwrap();

        // RMSE >= MAE always
        assert!(result.aggregated.rmse >= result.aggregated.mae);
        // SMAPE in valid range
        assert!(result.aggregated.smape >= 0.0);
        assert!(result.aggregated.smape <= 200.0);
        // Std devs non-negative
        assert!(result.aggregated.mae_std >= 0.0);
        assert!(result.aggregated.rmse_std >= 0.0);
        // Per-window metrics match count
        assert_eq!(result.window_metrics.len(), result.windows.len());
    }

    // ==================== Parallel CV Tests ====================

    #[test]
    fn grouped_cv_parallel_matches_sequential_results() {
        // Grouped CV uses cross_validate_with_folds internally which now
        // parallelizes when the feature is enabled. Verify results are valid.
        let timestamps = make_timestamps(30);

        let series_a =
            TimeSeries::univariate(timestamps.clone(), (0..30).map(|i| i as f64).collect())
                .unwrap();

        let series_b = TimeSeries::univariate(
            timestamps.clone(),
            (0..30).map(|i| (i as f64) * 2.0).collect(),
        )
        .unwrap();

        let series_map = vec![("a".to_string(), series_a), ("b".to_string(), series_b)];

        let config = CVConfig::expanding(15, 3).with_step_size(3);
        let results = grouped_cross_validate(&config, series_map, Naive::new).unwrap();

        assert_eq!(results.group_results.len(), 2);
        assert!(results.aggregated.mae.is_finite());
        assert!(results.aggregated.rmse.is_finite());

        // Both groups should have same number of folds
        let n_a = results.group_results[0].1.n_folds;
        let n_b = results.group_results[1].1.n_folds;
        assert_eq!(n_a, n_b);
        assert!(n_a > 0);
    }

    #[test]
    fn fold_generator_embargo_zero_matches_no_embargo() {
        let g1 = CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(3)
            .step_size(3);
        let g2 = CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(3)
            .step_size(3)
            .embargo(0);
        assert_eq!(g1.generate(50), g2.generate(50));
    }
    #[test]
    fn fold_generator_embargo_shrinks_training() {
        let folds = CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(3)
            .step_size(3)
            .embargo(5)
            .generate(50)
            .unwrap();
        assert!(!folds.is_empty());
        if folds.len() > 1 {
            assert!(folds[1].train_start > 0);
        }
    }
    #[test]
    fn fold_generator_embargo_with_gap_and_purge() {
        let folds = CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(3)
            .step_size(3)
            .gap(1)
            .purge(1)
            .embargo(3)
            .generate(50)
            .unwrap();
        assert!(!folds.is_empty());
        for fold in &folds {
            assert!(fold.train_end > fold.train_start);
        }
    }
    #[test]
    fn fold_generator_embargo_beyond_series_clamps() {
        let folds = CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(3)
            .step_size(3)
            .embargo(1000)
            .generate(30)
            .unwrap();
        assert!(folds.len() <= 2, "got {}", folds.len());
    }
    #[test]
    fn fold_generator_embargo_expanding_vs_rolling() {
        assert!(!CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(2)
            .step_size(2)
            .strategy(CVStrategy::Expanding)
            .embargo(3)
            .generate(40)
            .unwrap()
            .is_empty());
        for fold in &CvFoldGenerator::new()
            .min_initial_window(10)
            .horizon(2)
            .strategy(CVStrategy::Rolling)
            .embargo(3)
            .generate(40)
            .unwrap()
        {
            assert!(fold.train_end > fold.train_start);
        }
    }
    #[test]
    fn fold_generator_embargo_cvconfig_integration() {
        let gen = CVConfig::expanding(10, 3)
            .with_step_size(3)
            .with_embargo(5)
            .to_fold_generator();
        assert_eq!(gen.embargo, 5);
        assert!(!gen.generate(50).unwrap().is_empty());
    }

    // ══════════════════════════════════════════════════════════════════
    // Comprehensive CvFoldGenerator tests (n_folds-driven)
    // ══════════════════════════════════════════════════════════════════

    // ── A. Basic correctness ─────────────────────────────────────────

    #[test]
    fn a1_exact_5_folds_expanding_matches_sklearn() {
        // sklearn TimeSeriesSplit(n_splits=5, test_size=1) on range(10)
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(1)
            .min_initial_window(5)
            .generate(10)
            .unwrap();
        assert_eq!(folds.len(), 5);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 5, test_start: 5, test_end: 6 });
        assert_eq!(folds[1], Fold { train_start: 0, train_end: 6, test_start: 6, test_end: 7 });
        assert_eq!(folds[2], Fold { train_start: 0, train_end: 7, test_start: 7, test_end: 8 });
        assert_eq!(folds[3], Fold { train_start: 0, train_end: 8, test_start: 8, test_end: 9 });
        assert_eq!(folds[4], Fold { train_start: 0, train_end: 9, test_start: 9, test_end: 10 });
    }

    #[test]
    fn a2_exact_3_folds_larger_horizon() {
        // n=20, n_folds=3, h=3, min=5: first_origin=5, last_origin=17, step=6
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(3)
            .min_initial_window(5)
            .generate(20)
            .unwrap();
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 5, test_start: 5, test_end: 8 });
        assert_eq!(folds[1], Fold { train_start: 0, train_end: 11, test_start: 11, test_end: 14 });
        assert_eq!(folds[2], Fold { train_start: 0, train_end: 17, test_start: 17, test_end: 20 });
    }

    #[test]
    fn a3_all_indices_within_bounds() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(7)
            .min_initial_window(30)
            .generate(200)
            .unwrap();
        for fold in &folds {
            assert!(fold.train_start < fold.train_end);
            assert!(fold.train_end <= 200);
            assert!(fold.test_start < fold.test_end);
            assert!(fold.test_end <= 200);
        }
    }

    #[test]
    fn a4_no_train_test_overlap() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(7)
            .min_initial_window(30)
            .generate(200)
            .unwrap();
        for fold in &folds {
            assert!(fold.train_end <= fold.test_start);
        }
    }

    #[test]
    fn a5_folds_chronologically_ordered() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(5)
            .min_initial_window(20)
            .generate(200)
            .unwrap();
        for i in 1..folds.len() {
            assert!(folds[i].test_start > folds[i - 1].test_start);
        }
    }

    // ── B. Expanding window ──────────────────────────────────────────

    #[test]
    fn b1_expanding_all_start_at_zero() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(5)
            .min_initial_window(20)
            .generate(200)
            .unwrap();
        for fold in &folds {
            assert_eq!(fold.train_start, 0);
        }
    }

    #[test]
    fn b2_expanding_train_size_non_decreasing() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(5)
            .min_initial_window(20)
            .generate(200)
            .unwrap();
        for i in 1..folds.len() {
            assert!(folds[i].train_size() >= folds[i - 1].train_size());
        }
    }

    #[test]
    fn b3_expanding_first_fold_train_size_equals_min() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(1)
            .min_initial_window(5)
            .generate(10)
            .unwrap();
        assert_eq!(folds[0].train_size(), 5);
    }

    // ── C. Rolling window ────────────────────────────────────────────

    #[test]
    fn c1_rolling_exact_indices() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(1)
            .min_initial_window(5)
            .strategy(CVStrategy::Rolling)
            .generate(10)
            .unwrap();
        assert_eq!(folds.len(), 5);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 5, test_start: 5, test_end: 6 });
        assert_eq!(folds[1], Fold { train_start: 1, train_end: 6, test_start: 6, test_end: 7 });
        assert_eq!(folds[2], Fold { train_start: 2, train_end: 7, test_start: 7, test_end: 8 });
        assert_eq!(folds[3], Fold { train_start: 3, train_end: 8, test_start: 8, test_end: 9 });
        assert_eq!(folds[4], Fold { train_start: 4, train_end: 9, test_start: 9, test_end: 10 });
    }

    #[test]
    fn c2_rolling_all_same_train_size() {
        let folds = CvFoldGenerator::new()
            .n_folds(10)
            .horizon(1)
            .min_initial_window(50)
            .strategy(CVStrategy::Rolling)
            .generate(500)
            .unwrap();
        for fold in &folds {
            assert_eq!(fold.train_size(), 50);
        }
    }

    #[test]
    fn c3_rolling_train_start_increases() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(5)
            .min_initial_window(20)
            .strategy(CVStrategy::Rolling)
            .generate(200)
            .unwrap();
        for i in 1..folds.len() {
            assert!(folds[i].train_start > folds[i - 1].train_start);
        }
    }

    // ── D. Last fold anchored at series end ──────────────────────────

    #[test]
    fn d1_last_fold_expanding() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(12)
            .min_initial_window(20)
            .generate(144)
            .unwrap();
        assert_eq!(folds.last().unwrap().test_end, 144);
    }

    #[test]
    fn d2_last_fold_rolling() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(5)
            .min_initial_window(20)
            .strategy(CVStrategy::Rolling)
            .generate(100)
            .unwrap();
        assert_eq!(folds.last().unwrap().test_end, 100);
    }

    #[test]
    fn d3_last_fold_with_gap() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(5)
            .min_initial_window(10)
            .gap(3)
            .generate(80)
            .unwrap();
        assert_eq!(folds.last().unwrap().test_end, 80);
    }

    #[test]
    fn d4_last_fold_with_purge() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(5)
            .min_initial_window(10)
            .purge(3)
            .generate(80)
            .unwrap();
        assert_eq!(folds.last().unwrap().test_end, 80);
    }

    #[test]
    fn d5_single_fold_anchored() {
        let folds = CvFoldGenerator::new()
            .n_folds(1)
            .horizon(5)
            .min_initial_window(10)
            .generate(50)
            .unwrap();
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].test_end, 50);
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 45);
    }

    // ── E. min_initial_window constraint ─────────────────────────────

    #[test]
    fn e1_expanding_respects_min() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(7)
            .min_initial_window(50)
            .generate(200)
            .unwrap();
        for (i, fold) in folds.iter().enumerate() {
            assert!(fold.train_size() >= 50, "fold {} train_size {}", i, fold.train_size());
        }
    }

    #[test]
    fn e2_rolling_respects_min() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(7)
            .min_initial_window(50)
            .strategy(CVStrategy::Rolling)
            .generate(200)
            .unwrap();
        for fold in &folds {
            assert_eq!(fold.train_size(), 50);
        }
    }

    #[test]
    fn e3_series_exactly_min_plus_horizon() {
        // Only room for 1 fold
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(3)
            .min_initial_window(7)
            .on_constraint_violation(ConstraintViolation::ReduceFolds)
            .generate(10)
            .unwrap();
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 7, test_start: 7, test_end: 10 });
    }

    // ── F. Gap ───────────────────────────────────────────────────────

    #[test]
    fn f1_gap_respected_all_folds() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(5)
            .min_initial_window(10)
            .gap(3)
            .generate(80)
            .unwrap();
        for fold in &folds {
            assert!(fold.test_start >= fold.train_end + 3);
        }
    }

    #[test]
    fn f2_gap_exact_indices() {
        // n=30, n_folds=3, h=2, min=5, gap=3: first_origin=5, last_origin=25, step=10
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(2)
            .min_initial_window(5)
            .gap(3)
            .generate(30)
            .unwrap();
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 5, test_start: 8, test_end: 10 });
        assert_eq!(folds[1], Fold { train_start: 0, train_end: 15, test_start: 18, test_end: 20 });
        assert_eq!(folds[2], Fold { train_start: 0, train_end: 25, test_start: 28, test_end: 30 });
    }

    #[test]
    fn f3_gap_zero_means_adjacent() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(1)
            .min_initial_window(5)
            .gap(0)
            .generate(20)
            .unwrap();
        for fold in &folds {
            assert_eq!(fold.test_start, fold.train_end);
        }
    }

    // ── G. Purge ─────────────────────────────────────────────────────

    #[test]
    fn g1_purge_exact_indices() {
        // n=30, n_folds=3, h=2, min=5, purge=3: first_origin=8, last_origin=28, step=10
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(2)
            .min_initial_window(5)
            .purge(3)
            .generate(30)
            .unwrap();
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 5, test_start: 8, test_end: 10 });
        assert_eq!(folds[1], Fold { train_start: 0, train_end: 15, test_start: 18, test_end: 20 });
        assert_eq!(folds[2], Fold { train_start: 0, train_end: 25, test_start: 28, test_end: 30 });
    }

    #[test]
    fn g2_purge_plus_gap() {
        // n=30, n_folds=3, h=2, min=5, gap=3, purge=2: first_origin=7, last_origin=25, step=9
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(2)
            .min_initial_window(5)
            .gap(3)
            .purge(2)
            .generate(30)
            .unwrap();
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 5, test_start: 10, test_end: 12 });
        assert_eq!(folds[2].test_end, 30);
        // All folds: test_start - train_end >= gap + purge = 5
        for fold in &folds {
            assert!(fold.test_start - fold.train_end >= 5);
        }
    }

    // ── H. Embargo ───────────────────────────────────────────────────

    #[test]
    fn h1_embargo_shifts_subsequent_train_start() {
        // n=30, n_folds=3, h=2, min=5, embargo=4
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(2)
            .min_initial_window(5)
            .embargo(4)
            .generate(30)
            .unwrap();
        // Fold 0: train_start=0 (embargo doesn't affect first fold)
        assert_eq!(folds[0].train_start, 0);
        // Subsequent folds should have train_start shifted forward
        if folds.len() > 1 {
            assert!(folds[1].train_start > 0, "embargo should shift fold 1 train_start");
        }
    }

    #[test]
    fn h2_large_embargo_skips_folds() {
        // embargo so large it eats subsequent folds
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(2)
            .min_initial_window(3)
            .embargo(10)
            .generate(20)
            .unwrap();
        // Some folds should be skipped
        assert!(folds.len() < 3, "large embargo should skip folds, got {}", folds.len());
    }

    // ── I. ConstraintViolation::Error ────────────────────────────────

    #[test]
    fn i1_error_series_too_short() {
        let result = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(5)
            .min_initial_window(10)
            .generate(10); // need 15
        assert!(result.is_err());
    }

    #[test]
    fn i2_error_with_gap_purge() {
        let result = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(3)
            .min_initial_window(5)
            .gap(3)
            .purge(2)
            .generate(12); // need 5+2+3+3=13
        assert!(result.is_err());
    }

    #[test]
    fn i3_error_step_zero() {
        // Enough for 1 fold but not 10
        let result = CvFoldGenerator::new()
            .n_folds(10)
            .horizon(3)
            .min_initial_window(5)
            .generate(9); // first_origin=5, last_origin=6, range=1, step=0
        assert!(result.is_err());
    }

    // ── J. ConstraintViolation::ReduceFolds ──────────────────────────

    #[test]
    fn j1_reduces_to_1_fold() {
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(3)
            .min_initial_window(7)
            .on_constraint_violation(ConstraintViolation::ReduceFolds)
            .generate(10)
            .unwrap();
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 7, test_start: 7, test_end: 10 });
    }

    #[test]
    fn j2_reduces_to_feasible_count() {
        let folds = CvFoldGenerator::new()
            .n_folds(10)
            .horizon(3)
            .min_initial_window(5)
            .on_constraint_violation(ConstraintViolation::ReduceFolds)
            .generate(15)
            .unwrap();
        // first_origin=5, last_origin=12, range=7, step=0 → reduce to 8 folds
        assert_eq!(folds.len(), 8);
        assert_eq!(folds.last().unwrap().test_end, 15);
    }

    #[test]
    fn j3_no_reduction_when_fits() {
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(5)
            .min_initial_window(10)
            .on_constraint_violation(ConstraintViolation::ReduceFolds)
            .generate(100)
            .unwrap();
        assert_eq!(folds.len(), 3);
    }

    // ── K. Edge cases ────────────────────────────────────────────────

    #[test]
    fn k1_horizon_larger_than_half_series() {
        let folds = CvFoldGenerator::new()
            .n_folds(2)
            .horizon(15)
            .min_initial_window(5)
            .generate(25)
            .unwrap();
        assert_eq!(folds.len(), 2);
        for fold in &folds {
            assert_eq!(fold.test_size(), 15);
        }
        assert_eq!(folds.last().unwrap().test_end, 25);
    }

    #[test]
    fn k2_default_n_folds_is_5() {
        assert_eq!(CvFoldGenerator::new().target_n_folds, 5);
    }

    #[test]
    fn k3_default_constraint_violation_is_error() {
        assert_eq!(CvFoldGenerator::new().on_constraint_violation, ConstraintViolation::Error);
    }

    #[test]
    fn k4_uneven_step_last_fold_jumps() {
        // step=1 but last fold jumps to end: origins 2,3,4,5 then 9
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(1)
            .min_initial_window(2)
            .generate(10)
            .unwrap();
        assert_eq!(folds.len(), 5);
        assert_eq!(folds[3].test_start, 5); // origin=5
        assert_eq!(folds[4].test_start, 9); // jump to last_origin
    }

    #[test]
    fn k5_perfectly_even_distribution() {
        // available_range=8, 4 intervals → step=2, no remainder
        let folds = CvFoldGenerator::new()
            .n_folds(5)
            .horizon(1)
            .min_initial_window(1)
            .generate(10)
            .unwrap();
        assert_eq!(folds.len(), 5);
        assert_eq!(folds[0].test_start, 1);
        assert_eq!(folds[1].test_start, 3);
        assert_eq!(folds[2].test_start, 5);
        assert_eq!(folds[3].test_start, 7);
        assert_eq!(folds[4].test_start, 9);
    }

    // ── L. Reference: match sklearn TimeSeriesSplit ───────────────────

    #[test]
    fn l1_sklearn_n_splits_3_test_size_2_on_20() {
        // sklearn: train=[0..14), test=[14..16) / train=[0..16), test=[16..18) / train=[0..18), test=[18..20)
        let folds = CvFoldGenerator::new()
            .n_folds(3)
            .horizon(2)
            .min_initial_window(14)
            .generate(20)
            .unwrap();
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 14, test_start: 14, test_end: 16 });
        assert_eq!(folds[1], Fold { train_start: 0, train_end: 16, test_start: 16, test_end: 18 });
        assert_eq!(folds[2], Fold { train_start: 0, train_end: 18, test_start: 18, test_end: 20 });
    }

    #[test]
    fn l2_sklearn_n_splits_4_test_size_1_on_20() {
        let folds = CvFoldGenerator::new()
            .n_folds(4)
            .horizon(1)
            .min_initial_window(16)
            .generate(20)
            .unwrap();
        assert_eq!(folds.len(), 4);
        assert_eq!(folds[0], Fold { train_start: 0, train_end: 16, test_start: 16, test_end: 17 });
        assert_eq!(folds[1], Fold { train_start: 0, train_end: 17, test_start: 17, test_end: 18 });
        assert_eq!(folds[2], Fold { train_start: 0, train_end: 18, test_start: 18, test_end: 19 });
        assert_eq!(folds[3], Fold { train_start: 0, train_end: 19, test_start: 19, test_end: 20 });
    }
}
