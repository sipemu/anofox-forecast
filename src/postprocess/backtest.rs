//! Backtesting support for postprocessing methods.
//!
//! This module provides rolling/expanding window backtesting with
//! horizon-aware calibration for evaluating postprocessor performance.
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::{PostProcessor, BacktestConfig};
//!
//! let config = BacktestConfig::new()
//!     .initial_window(100)
//!     .step(10)
//!     .horizon(7)
//!     .expanding(true);
//!
//! let results = processor.backtest(&forecasts, &actuals, config)?;
//! println!("Coverage: {:.1}%", results.coverage() * 100.0);
//! ```

use std::collections::HashMap;

use crate::error::{ForecastError, Result};
use crate::postprocess::{PointForecasts, PostProcessor, PredictionIntervals, TrainedModel};

/// Configuration for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Minimum number of samples for initial training window.
    pub initial_window: usize,
    /// Number of samples to step forward between folds.
    pub step: usize,
    /// Forecast horizon (number of steps to predict per fold).
    pub horizon: usize,
    /// If true, use expanding window; if false, use rolling window.
    pub expanding: bool,
    /// If true, train separate model per horizon.
    pub horizon_aware: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_window: 50,
            step: 1,
            horizon: 1,
            expanding: true,
            horizon_aware: false,
        }
    }
}

impl BacktestConfig {
    /// Create a new backtest configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial training window size.
    pub fn initial_window(mut self, size: usize) -> Self {
        self.initial_window = size;
        self
    }

    /// Set the step size between folds.
    pub fn step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Set the forecast horizon.
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    /// Set whether to use expanding (true) or rolling (false) window.
    pub fn expanding(mut self, expanding: bool) -> Self {
        self.expanding = expanding;
        self
    }

    /// Set whether to use horizon-aware calibration.
    pub fn horizon_aware(mut self, aware: bool) -> Self {
        self.horizon_aware = aware;
        self
    }
}

/// Results from a single backtest fold.
#[derive(Debug, Clone)]
pub struct BacktestFold {
    /// Index of this fold.
    pub fold_idx: usize,
    /// Start index of training data.
    pub train_start: usize,
    /// End index of training data (exclusive).
    pub train_end: usize,
    /// Start index of test data.
    pub test_start: usize,
    /// End index of test data (exclusive).
    pub test_end: usize,
    /// Predicted intervals for this fold.
    pub intervals: PredictionIntervals,
    /// Actual values for this fold.
    pub actuals: Vec<f64>,
    /// Coverage achieved (fraction of actuals within intervals).
    pub coverage: f64,
    /// Average interval width.
    pub avg_width: f64,
}

impl BacktestFold {
    /// Get the training set size.
    pub fn train_size(&self) -> usize {
        self.train_end - self.train_start
    }

    /// Get the test set size.
    pub fn test_size(&self) -> usize {
        self.test_end - self.test_start
    }

    /// Get the predicted intervals.
    pub fn intervals(&self) -> &PredictionIntervals {
        &self.intervals
    }

    /// Get the actual values.
    pub fn actuals(&self) -> &[f64] {
        &self.actuals
    }

    /// Get the coverage for this fold.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Get the average interval width for this fold.
    pub fn avg_width(&self) -> f64 {
        self.avg_width
    }
}

/// Calibrated model trained on all backtest data, per horizon.
#[derive(Debug, Clone)]
pub struct CalibratedModelByHorizon {
    /// Trained models indexed by horizon (1-indexed).
    models: HashMap<usize, TrainedModel>,
}

impl CalibratedModelByHorizon {
    /// Get the model for a specific horizon.
    pub fn get(&self, horizon: usize) -> Option<&TrainedModel> {
        self.models.get(&horizon)
    }

    /// Get all horizons that have trained models.
    pub fn horizons(&self) -> Vec<usize> {
        let mut h: Vec<_> = self.models.keys().copied().collect();
        h.sort();
        h
    }

    /// Get the number of horizon-specific models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if there are no models.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Results from backtesting a postprocessor.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Configuration used for backtesting.
    config: BacktestConfig,
    /// Results for each fold.
    folds: Vec<BacktestFold>,
    /// Pooled forecasts (all folds concatenated).
    pooled_forecasts: Vec<f64>,
    /// Pooled actuals (all folds concatenated).
    pooled_actuals: Vec<f64>,
    /// Per-horizon forecasts (horizon -> Vec<forecast>).
    forecasts_by_horizon: HashMap<usize, Vec<f64>>,
    /// Per-horizon actuals (horizon -> Vec<actual>).
    actuals_by_horizon: HashMap<usize, Vec<f64>>,
    /// Per-horizon coverage (computed lazily).
    coverage_by_horizon: HashMap<usize, f64>,
    /// Per-horizon interval widths (computed lazily).
    widths_by_horizon: HashMap<usize, f64>,
}

impl BacktestResult {
    /// Get the number of folds.
    pub fn n_folds(&self) -> usize {
        self.folds.len()
    }

    /// Get the configuration used.
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }

    /// Iterate over fold results.
    pub fn folds(&self) -> impl Iterator<Item = &BacktestFold> {
        self.folds.iter()
    }

    /// Get a specific fold by index.
    pub fn fold(&self, idx: usize) -> Option<&BacktestFold> {
        self.folds.get(idx)
    }

    /// Get overall coverage across all folds.
    pub fn coverage(&self) -> f64 {
        if self.folds.is_empty() {
            return 0.0;
        }

        let total_covered: usize = self
            .folds
            .iter()
            .map(|f| {
                let n = f.intervals.len();
                (f.coverage * n as f64).round() as usize
            })
            .sum();

        let total_samples: usize = self.folds.iter().map(|f| f.intervals.len()).sum();

        if total_samples == 0 {
            0.0
        } else {
            total_covered as f64 / total_samples as f64
        }
    }

    /// Get calibration error (absolute deviation from target coverage).
    pub fn calibration_error(&self, target_coverage: f64) -> f64 {
        (self.coverage() - target_coverage).abs()
    }

    /// Get average interval width across all folds.
    pub fn interval_widths(&self) -> f64 {
        if self.folds.is_empty() {
            return 0.0;
        }

        let total_width: f64 = self
            .folds
            .iter()
            .map(|f| f.avg_width * f.intervals.len() as f64)
            .sum();

        let total_samples: usize = self.folds.iter().map(|f| f.intervals.len()).sum();

        if total_samples == 0 {
            0.0
        } else {
            total_width / total_samples as f64
        }
    }

    /// Get coverage by horizon (only available if horizon_aware was true).
    pub fn coverage_by_horizon(&self) -> &HashMap<usize, f64> {
        &self.coverage_by_horizon
    }

    /// Get interval widths by horizon (only available if horizon_aware was true).
    pub fn widths_by_horizon(&self) -> &HashMap<usize, f64> {
        &self.widths_by_horizon
    }

    /// Get pooled forecasts from all folds.
    pub fn pooled_forecasts(&self) -> &[f64] {
        &self.pooled_forecasts
    }

    /// Get pooled actuals from all folds.
    pub fn pooled_actuals(&self) -> &[f64] {
        &self.pooled_actuals
    }

    /// Train a calibrated model on all backtest data (pooled).
    pub fn calibrated_model(&self, processor: &PostProcessor) -> Result<TrainedModel> {
        if self.pooled_forecasts.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let forecasts = PointForecasts::from_values(self.pooled_forecasts.clone());
        processor.train(&forecasts, &self.pooled_actuals)
    }

    /// Train horizon-specific calibrated models.
    pub fn calibrated_model_by_horizon(
        &self,
        processor: &PostProcessor,
    ) -> Result<CalibratedModelByHorizon> {
        if self.forecasts_by_horizon.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "horizon-aware data not available; set horizon_aware=true in config".to_string(),
            ));
        }

        let mut models = HashMap::new();

        for (&horizon, forecasts) in &self.forecasts_by_horizon {
            let actuals = self
                .actuals_by_horizon
                .get(&horizon)
                .ok_or_else(|| ForecastError::EmptyData)?;

            if forecasts.is_empty() {
                continue;
            }

            let point_forecasts = PointForecasts::from_values(forecasts.clone());
            let trained = processor.train(&point_forecasts, actuals)?;
            models.insert(horizon, trained);
        }

        Ok(CalibratedModelByHorizon { models })
    }
}

impl PostProcessor {
    /// Run backtesting on historical data.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - All historical point forecasts
    /// * `actuals` - All historical actual values
    /// * `config` - Backtest configuration
    ///
    /// # Returns
    ///
    /// A `BacktestResult` containing fold-level and aggregate metrics.
    pub fn backtest(
        &self,
        forecasts: &PointForecasts,
        actuals: &[f64],
        config: BacktestConfig,
    ) -> Result<BacktestResult> {
        let n = forecasts.len();

        if n != actuals.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: n,
                got: actuals.len(),
            });
        }

        if n < config.initial_window + config.horizon {
            return Err(ForecastError::InsufficientData {
                needed: config.initial_window + config.horizon,
                got: n,
            });
        }

        let forecast_values = forecasts.values();
        let mut folds = Vec::new();
        let mut pooled_forecasts = Vec::new();
        let mut pooled_actuals = Vec::new();
        let mut forecasts_by_horizon: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut actuals_by_horizon: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut coverage_by_horizon: HashMap<usize, (usize, usize)> = HashMap::new(); // (covered, total)
        let mut widths_by_horizon: HashMap<usize, (f64, usize)> = HashMap::new(); // (sum, count)

        let mut fold_idx = 0;
        let mut test_start = config.initial_window;

        while test_start + config.horizon <= n {
            let train_start = if config.expanding {
                0
            } else {
                test_start.saturating_sub(config.initial_window)
            };
            let train_end = test_start;
            let test_end = (test_start + config.horizon).min(n);

            // Extract train data
            let train_forecasts: Vec<f64> = forecast_values[train_start..train_end].to_vec();
            let train_actuals: Vec<f64> = actuals[train_start..train_end].to_vec();

            // Extract test data
            let test_forecasts_vec: Vec<f64> = forecast_values[test_start..test_end].to_vec();
            let test_actuals: Vec<f64> = actuals[test_start..test_end].to_vec();

            // Train and predict
            let train_f = PointForecasts::from_values(train_forecasts);
            let test_f = PointForecasts::from_values(test_forecasts_vec.clone());

            let trained = self.train(&train_f, &train_actuals)?;
            let intervals = self.predict_intervals(&trained, &test_f)?;

            // Calculate coverage and width for this fold
            let mut covered = 0;
            let mut total_width = 0.0;

            for i in 0..intervals.len() {
                let lower = intervals.lower()[i];
                let upper = intervals.upper()[i];
                let actual = test_actuals[i];

                if actual >= lower && actual <= upper {
                    covered += 1;
                }
                total_width += upper - lower;

                // Track by horizon
                let h = i + 1; // 1-indexed horizon
                pooled_forecasts.push(test_forecasts_vec[i]);
                pooled_actuals.push(actual);

                if config.horizon_aware {
                    forecasts_by_horizon
                        .entry(h)
                        .or_default()
                        .push(test_forecasts_vec[i]);
                    actuals_by_horizon.entry(h).or_default().push(actual);

                    let cov_entry = coverage_by_horizon.entry(h).or_insert((0, 0));
                    if actual >= lower && actual <= upper {
                        cov_entry.0 += 1;
                    }
                    cov_entry.1 += 1;

                    let width_entry = widths_by_horizon.entry(h).or_insert((0.0, 0));
                    width_entry.0 += upper - lower;
                    width_entry.1 += 1;
                }
            }

            let n_test = intervals.len();
            let fold_coverage = if n_test > 0 {
                covered as f64 / n_test as f64
            } else {
                0.0
            };
            let fold_avg_width = if n_test > 0 {
                total_width / n_test as f64
            } else {
                0.0
            };

            folds.push(BacktestFold {
                fold_idx,
                train_start,
                train_end,
                test_start,
                test_end,
                intervals,
                actuals: test_actuals,
                coverage: fold_coverage,
                avg_width: fold_avg_width,
            });

            fold_idx += 1;
            test_start += config.step;
        }

        // Convert coverage and width tracking to final values
        let coverage_by_horizon_final: HashMap<usize, f64> = coverage_by_horizon
            .into_iter()
            .map(|(h, (covered, total))| {
                let cov = if total > 0 {
                    covered as f64 / total as f64
                } else {
                    0.0
                };
                (h, cov)
            })
            .collect();

        let widths_by_horizon_final: HashMap<usize, f64> = widths_by_horizon
            .into_iter()
            .map(|(h, (sum, count))| {
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                (h, avg)
            })
            .collect();

        Ok(BacktestResult {
            config,
            folds,
            pooled_forecasts,
            pooled_actuals,
            forecasts_by_horizon,
            actuals_by_horizon,
            coverage_by_horizon: coverage_by_horizon_final,
            widths_by_horizon: widths_by_horizon_final,
        })
    }

    /// Predict intervals using horizon-specific models.
    ///
    /// # Arguments
    ///
    /// * `models` - Horizon-specific trained models
    /// * `forecasts` - Forecasts for each horizon (length should match max horizon)
    ///
    /// # Returns
    ///
    /// Prediction intervals with horizon-specific calibration applied.
    pub fn predict_intervals_by_horizon(
        &self,
        models: &CalibratedModelByHorizon,
        forecasts: &PointForecasts,
    ) -> Result<PredictionIntervals> {
        let n = forecasts.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }

        let mut all_lower = Vec::with_capacity(n);
        let mut all_upper = Vec::with_capacity(n);

        for i in 0..n {
            let h = i + 1; // 1-indexed horizon

            // Get model for this horizon, or fall back to closest available
            let model = models
                .get(h)
                .or_else(|| {
                    // Fall back to max available horizon if requested horizon not available
                    models.horizons().iter().max().and_then(|&max_h| models.get(max_h))
                })
                .ok_or_else(|| {
                    ForecastError::InvalidParameter(format!("no model available for horizon {}", h))
                })?;

            // Predict for single point
            let single_forecast = PointForecasts::from_values(vec![forecasts.values()[i]]);
            let intervals = self.predict_intervals(model, &single_forecast)?;

            all_lower.push(intervals.lower()[0]);
            all_upper.push(intervals.upper()[0]);
        }

        // Compute coverage from first model (they should all have same coverage)
        let coverage = if let Some(&h) = models.horizons().first() {
            if let Some(_model) = models.get(h) {
                0.90 // Default; ideally would get from model
            } else {
                0.90
            }
        } else {
            0.90
        };

        PredictionIntervals::from_bounds(all_lower, all_upper, coverage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize) -> (PointForecasts, Vec<f64>) {
        let forecasts: Vec<f64> = (0..n).map(|i| 50.0 + 0.2 * i as f64).collect();
        let actuals: Vec<f64> = forecasts
            .iter()
            .enumerate()
            .map(|(i, &f)| f + 1.0 * ((i as f64 * 0.1).sin()))
            .collect();
        (PointForecasts::from_values(forecasts), actuals)
    }

    mod backtest_config {
        use super::*;

        #[test]
        fn default_values() {
            let config = BacktestConfig::default();
            assert_eq!(config.initial_window, 50);
            assert_eq!(config.step, 1);
            assert_eq!(config.horizon, 1);
            assert!(config.expanding);
            assert!(!config.horizon_aware);
        }

        #[test]
        fn builder_pattern() {
            let config = BacktestConfig::new()
                .initial_window(100)
                .step(10)
                .horizon(7)
                .expanding(false)
                .horizon_aware(true);

            assert_eq!(config.initial_window, 100);
            assert_eq!(config.step, 10);
            assert_eq!(config.horizon, 7);
            assert!(!config.expanding);
            assert!(config.horizon_aware);
        }
    }

    mod backtest_result {
        use super::*;

        #[test]
        fn basic_backtest() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .step(10)
                .horizon(5);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();

            assert!(result.n_folds() > 0);
            assert!(result.coverage() >= 0.0 && result.coverage() <= 1.0);
            assert!(result.interval_widths() >= 0.0);
        }

        #[test]
        fn expanding_window() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(30)
                .step(10)
                .horizon(5)
                .expanding(true);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();

            // Check that training window grows
            let fold0 = result.fold(0).unwrap();
            let fold1 = result.fold(1).unwrap();

            assert_eq!(fold0.train_start, 0);
            assert_eq!(fold1.train_start, 0);
            assert!(fold1.train_size() > fold0.train_size());
        }

        #[test]
        fn rolling_window() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(30)
                .step(10)
                .horizon(5)
                .expanding(false);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();

            // Check that training window slides
            let fold0 = result.fold(0).unwrap();
            let fold1 = result.fold(1).unwrap();

            assert_eq!(fold0.train_size(), 30);
            assert_eq!(fold1.train_size(), 30);
            assert!(fold1.train_start > fold0.train_start);
        }

        #[test]
        fn horizon_aware() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .step(10)
                .horizon(5)
                .horizon_aware(true);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();

            let coverage_by_h = result.coverage_by_horizon();
            let widths_by_h = result.widths_by_horizon();

            assert!(!coverage_by_h.is_empty());
            assert!(!widths_by_h.is_empty());

            // Should have entries for horizons 1-5
            for h in 1..=5 {
                assert!(coverage_by_h.contains_key(&h));
                assert!(widths_by_h.contains_key(&h));
            }
        }

        #[test]
        fn calibrated_model_pooled() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .step(10)
                .horizon(5);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();

            let calibrated = result.calibrated_model(&processor);
            assert!(calibrated.is_ok());
        }

        #[test]
        fn calibrated_model_by_horizon() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .step(10)
                .horizon(5)
                .horizon_aware(true);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();

            let calibrated_by_h = result.calibrated_model_by_horizon(&processor);
            assert!(calibrated_by_h.is_ok());

            let models = calibrated_by_h.unwrap();
            assert_eq!(models.len(), 5);
        }
    }

    mod backtest_fold {
        use super::*;

        #[test]
        fn fold_accessors() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .step(10)
                .horizon(5);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();
            let fold = result.fold(0).unwrap();

            assert_eq!(fold.train_size(), 50);
            assert_eq!(fold.test_size(), 5);
            assert!(fold.coverage() >= 0.0 && fold.coverage() <= 1.0);
            assert!(fold.avg_width() >= 0.0);
            assert_eq!(fold.actuals().len(), 5);
            assert_eq!(fold.intervals().len(), 5);
        }
    }

    mod predict_by_horizon {
        use super::*;

        #[test]
        fn predict_with_horizon_models() {
            let (forecasts, actuals) = make_data(100);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .step(10)
                .horizon(5)
                .horizon_aware(true);

            let result = processor.backtest(&forecasts, &actuals, config).unwrap();
            let models = result.calibrated_model_by_horizon(&processor).unwrap();

            // New forecasts for 5-step horizon
            let new_forecasts = PointForecasts::from_values(vec![70.0, 70.5, 71.0, 71.5, 72.0]);

            let intervals = processor
                .predict_intervals_by_horizon(&models, &new_forecasts)
                .unwrap();

            assert_eq!(intervals.len(), 5);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn insufficient_data() {
            let (forecasts, actuals) = make_data(30);
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::new()
                .initial_window(50)
                .horizon(5);

            let result = processor.backtest(&forecasts, &actuals, config);
            assert!(result.is_err());
        }

        #[test]
        fn mismatched_lengths() {
            let forecasts = PointForecasts::from_values(vec![1.0, 2.0, 3.0]);
            let actuals = vec![1.0, 2.0];
            let processor = PostProcessor::conformal(0.90);

            let config = BacktestConfig::default();
            let result = processor.backtest(&forecasts, &actuals, config);

            assert!(result.is_err());
        }
    }
}
