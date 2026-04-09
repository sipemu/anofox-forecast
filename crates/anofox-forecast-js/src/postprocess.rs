//! PostProcessor wrappers for JavaScript.
//!
//! This module provides wasm-bindgen wrappers for postprocessing methods
//! that convert point forecasts into calibrated prediction intervals.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::postprocess::{
    BacktestConfig as InnerBacktestConfig, BacktestResult as InnerBacktestResult,
    BinnedConformalPredictor as InnerBinnedConformalPredictor,
    BinnedConformalResult as InnerBinnedConformalResult,
    BootstrapPredictor as InnerBootstrapPredictor, BootstrapResult as InnerBootstrapResult,
    ConformalPredictor as InnerConformalPredictor, ConformalResult as InnerConformalResult,
    HistoricalSimResult as InnerHistoricalSimResult,
    HistoricalSimulator as InnerHistoricalSimulator, IDRPredictor as InnerIDRPredictor,
    IDRResult as InnerIDRResult, NormalPredictor as InnerNormalPredictor,
    NormalResult as InnerNormalResult, PerStepConformalResult as InnerPerStepConformalResult,
    PointForecasts as InnerPointForecasts, PostProcessor as InnerPostProcessor,
    PredictionIntervals as InnerPredictionIntervals, QRAPredictor as InnerQRAPredictor,
    QRAResult as InnerQRAResult, QuantileForecasts as InnerQuantileForecasts,
    TrainedModel as InnerTrainedModel,
};

// =============================================================================
// POINT FORECASTS
// =============================================================================

/// Point forecasts — a sequence of predicted values.
///
/// Used as input for postprocessing methods.
#[wasm_bindgen]
pub struct JsPointForecasts {
    inner: InnerPointForecasts,
}

#[wasm_bindgen]
impl JsPointForecasts {
    /// Create point forecasts from an array of values.
    #[wasm_bindgen(constructor)]
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            inner: InnerPointForecasts::from_values(values),
        }
    }

    /// Get the number of forecast points.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Get the forecast values.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Check if empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl JsPointForecasts {
    pub(crate) fn inner(&self) -> &InnerPointForecasts {
        &self.inner
    }
}

// =============================================================================
// PREDICTION INTERVALS
// =============================================================================

/// Prediction intervals with lower bounds, upper bounds, and a coverage level.
#[wasm_bindgen]
pub struct JsPredictionIntervals {
    inner: InnerPredictionIntervals,
}

#[wasm_bindgen]
impl JsPredictionIntervals {
    /// Get the lower bounds.
    #[wasm_bindgen(getter)]
    pub fn lower(&self) -> Vec<f64> {
        self.inner.lower().to_vec()
    }

    /// Get the upper bounds.
    #[wasm_bindgen(getter)]
    pub fn upper(&self) -> Vec<f64> {
        self.inner.upper().to_vec()
    }

    /// Get the coverage level (e.g. 0.90).
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    /// Get the number of intervals.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Get the interval widths.
    pub fn widths(&self) -> Vec<f64> {
        self.inner.widths()
    }

    /// Get the interval midpoints.
    pub fn midpoints(&self) -> Vec<f64> {
        self.inner.midpoints()
    }

    /// Compute empirical coverage given actual values.
    #[wasm_bindgen(js_name = empiricalCoverage)]
    pub fn empirical_coverage(&self, actuals: Vec<f64>) -> Option<f64> {
        self.inner.empirical_coverage(&actuals)
    }
}

impl JsPredictionIntervals {
    fn from_inner(inner: InnerPredictionIntervals) -> Self {
        Self { inner }
    }
}

// =============================================================================
// CONFORMAL PREDICTOR
// =============================================================================

/// Conformal prediction result — stores calibration data from fitting.
#[wasm_bindgen]
pub struct JsConformalResult {
    inner: InnerConformalResult,
}

#[wasm_bindgen]
impl JsConformalResult {
    /// Get the quantile value (interval half-width).
    #[wasm_bindgen(js_name = quantileValue)]
    pub fn quantile_value(&self) -> f64 {
        self.inner.quantile_value()
    }

    /// Get the coverage level.
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    /// Get the nonconformity scores.
    pub fn scores(&self) -> Vec<f64> {
        self.inner.scores().to_vec()
    }
}

/// Conformal predictor for distribution-free prediction intervals.
///
/// Provides coverage-guaranteed intervals without distributional assumptions.
#[wasm_bindgen]
pub struct JsConformalPredictor {
    inner: InnerConformalPredictor,
}

#[wasm_bindgen]
impl JsConformalPredictor {
    /// Create a split conformal predictor.
    ///
    /// @param coverage - Target coverage level in (0, 1), e.g. 0.90
    #[wasm_bindgen(constructor)]
    pub fn new(coverage: f64) -> Self {
        Self {
            inner: InnerConformalPredictor::split(coverage),
        }
    }

    /// Create a conformal predictor with the cross-validation method.
    ///
    /// @param coverage - Target coverage level in (0, 1)
    /// @param nFolds - Number of cross-validation folds
    #[wasm_bindgen(js_name = crossVal)]
    pub fn cross_val(coverage: f64, n_folds: usize) -> Self {
        Self {
            inner: InnerConformalPredictor::cross_val(coverage, n_folds),
        }
    }

    /// Create a conformal predictor with the Jackknife+ method.
    ///
    /// @param coverage - Target coverage level in (0, 1)
    #[wasm_bindgen(js_name = jackknifePlus)]
    pub fn jackknife_plus(coverage: f64) -> Self {
        Self {
            inner: InnerConformalPredictor::jackknife_plus(coverage),
        }
    }

    /// Calibrate (fit) the predictor on historical forecasts and actuals.
    ///
    /// @param forecasts - Historical point forecast values
    /// @param actuals - Corresponding actual observed values
    pub fn calibrate(
        &self,
        forecasts: Vec<f64>,
        actuals: Vec<f64>,
    ) -> Result<JsConformalResult, JsError> {
        self.inner
            .fit(&forecasts, &actuals)
            .map(|r| JsConformalResult { inner: r })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate prediction intervals for new point forecasts.
    ///
    /// @param result - A fitted JsConformalResult from calibrate()
    /// @param pointForecasts - New point forecast values
    #[wasm_bindgen(js_name = predictIntervals)]
    pub fn predict_intervals(
        &self,
        result: &JsConformalResult,
        point_forecasts: Vec<f64>,
    ) -> JsPredictionIntervals {
        let intervals = self.inner.predict_values(&result.inner, &point_forecasts);
        JsPredictionIntervals::from_inner(intervals)
    }

    /// Generate quantile forecasts at multiple quantile levels.
    ///
    /// Returns a serialized object with `quantiles` (levels) and `values` (matrix).
    ///
    /// @param result - A fitted JsConformalResult from calibrate()
    /// @param pointForecasts - New point forecast values
    /// @param levels - Quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9]
    #[wasm_bindgen(js_name = predictQuantiles)]
    pub fn predict_quantiles(
        &self,
        result: &JsConformalResult,
        point_forecasts: Vec<f64>,
        levels: Vec<f64>,
    ) -> Result<JsValue, JsError> {
        let qf = self
            .inner
            .predict_quantiles(&result.inner, &point_forecasts, &levels);
        let js_qf = QuantileForecastsJs::from_inner(&qf);
        serde_wasm_bindgen::to_value(&js_qf).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Fit per-horizon-step conformal intervals.
    ///
    /// Each horizon step gets its own interval half-width, allowing tighter
    /// intervals for near-horizon steps and wider for far-horizon steps.
    ///
    /// @param foldForecasts - Per-fold forecast arrays (array of arrays), each of length `horizon`
    /// @param foldActuals - Per-fold actual arrays (array of arrays), each of length `horizon`
    #[wasm_bindgen(js_name = fitPerStep)]
    pub fn fit_per_step(
        &self,
        fold_forecasts: JsValue,
        fold_actuals: JsValue,
    ) -> Result<JsPerStepConformalResult, JsError> {
        let fc: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(fold_forecasts)
            .map_err(|e| JsError::new(&format!("invalid foldForecasts: {}", e)))?;
        let ac: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(fold_actuals)
            .map_err(|e| JsError::new(&format!("invalid foldActuals: {}", e)))?;
        self.inner
            .fit_per_step(&fc, &ac)
            .map(|r| JsPerStepConformalResult { inner: r })
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

// =============================================================================
// NORMAL PREDICTOR
// =============================================================================

/// Normal prediction result — stores Gaussian fit parameters.
#[wasm_bindgen]
pub struct JsNormalResult {
    inner: InnerNormalResult,
}

#[wasm_bindgen]
impl JsNormalResult {
    /// Get the mean error (bias).
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Get the standard deviation of errors.
    #[wasm_bindgen(js_name = stdDev)]
    pub fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }
}

/// Normal predictor — assumes Gaussian forecast errors.
///
/// Generates quantile-based prediction intervals from the estimated
/// error mean and standard deviation.
#[wasm_bindgen]
pub struct JsNormalPredictor {
    inner: InnerNormalPredictor,
}

#[wasm_bindgen]
impl JsNormalPredictor {
    /// Create a new normal predictor.
    ///
    /// @param quantiles - Sorted quantile levels in (0,1), e.g. [0.1, 0.5, 0.9]
    #[wasm_bindgen(constructor)]
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self {
            inner: InnerNormalPredictor::new(quantiles),
        }
    }

    /// Fit the predictor on historical forecasts and actuals.
    ///
    /// @param forecasts - Historical point forecast values
    /// @param actuals - Corresponding actual observed values
    pub fn fit(&self, forecasts: Vec<f64>, actuals: Vec<f64>) -> Result<JsNormalResult, JsError> {
        self.inner
            .fit(&forecasts, &actuals)
            .map(|r| JsNormalResult { inner: r })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate prediction intervals for new point forecasts.
    ///
    /// Returns intervals using the first and last quantile as bounds.
    ///
    /// @param result - A fitted JsNormalResult from fit()
    /// @param pointForecasts - New point forecast values
    #[wasm_bindgen(js_name = predictIntervals)]
    pub fn predict_intervals(
        &self,
        result: &JsNormalResult,
        point_forecasts: Vec<f64>,
    ) -> Result<JsPredictionIntervals, JsError> {
        let pf = InnerPointForecasts::from_values(point_forecasts);
        let qf = self
            .inner
            .predict(&result.inner, &pf)
            .map_err(|e| JsError::new(&e.to_string()))?;
        // Extract intervals from the outer quantiles
        let n_q = qf.n_quantiles();
        if n_q < 2 {
            return Err(JsError::new("need at least 2 quantiles for intervals"));
        }
        let lower: Vec<f64> = (0..qf.n_times())
            .map(|t| qf.at_time(t).unwrap()[0])
            .collect();
        let upper: Vec<f64> = (0..qf.n_times())
            .map(|t| qf.at_time(t).unwrap()[n_q - 1])
            .collect();
        let qs = qf.quantiles();
        let coverage = qs[n_q - 1] - qs[0];
        let intervals = InnerPredictionIntervals::from_bounds(lower, upper, coverage)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsPredictionIntervals::from_inner(intervals))
    }
}

// =============================================================================
// HISTORICAL SIMULATOR
// =============================================================================

/// Historical simulation result — stores the empirical error distribution.
#[wasm_bindgen]
pub struct JsHistoricalSimResult {
    inner: InnerHistoricalSimResult,
}

#[wasm_bindgen]
impl JsHistoricalSimResult {
    /// Get the sorted errors.
    pub fn errors(&self) -> Vec<f64> {
        self.inner.errors().to_vec()
    }

    /// Get the quantile values.
    #[wasm_bindgen(js_name = quantileValues)]
    pub fn quantile_values(&self) -> Vec<f64> {
        self.inner.quantile_values().to_vec()
    }
}

/// Historical simulator — non-parametric prediction intervals
/// based on the empirical error distribution.
#[wasm_bindgen]
pub struct JsHistoricalSimulator {
    inner: InnerHistoricalSimulator,
}

#[wasm_bindgen]
impl JsHistoricalSimulator {
    /// Create a new historical simulator.
    ///
    /// @param quantiles - Sorted quantile levels in (0,1), e.g. [0.1, 0.5, 0.9]
    #[wasm_bindgen(constructor)]
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self {
            inner: InnerHistoricalSimulator::new(quantiles),
        }
    }

    /// Create a simulator with a rolling window.
    ///
    /// @param quantiles - Sorted quantile levels in (0,1)
    /// @param window - Number of recent observations to use
    #[wasm_bindgen(js_name = withWindow)]
    pub fn with_window(quantiles: Vec<f64>, window: usize) -> Self {
        Self {
            inner: InnerHistoricalSimulator::with_window(quantiles, window),
        }
    }

    /// Fit the simulator on historical forecasts and actuals.
    ///
    /// @param forecasts - Historical point forecast values
    /// @param actuals - Corresponding actual observed values
    pub fn simulate(
        &self,
        forecasts: Vec<f64>,
        actuals: Vec<f64>,
    ) -> Result<JsHistoricalSimResult, JsError> {
        self.inner
            .fit(&forecasts, &actuals)
            .map(|r| JsHistoricalSimResult { inner: r })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate prediction intervals for new point forecasts.
    ///
    /// @param result - A fitted JsHistoricalSimResult from simulate()
    /// @param pointForecasts - New point forecast values
    #[wasm_bindgen(js_name = predictIntervals)]
    pub fn predict_intervals(
        &self,
        result: &JsHistoricalSimResult,
        point_forecasts: Vec<f64>,
    ) -> Result<JsPredictionIntervals, JsError> {
        let qf = self
            .inner
            .predict_values(&result.inner, &point_forecasts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n_q = qf.n_quantiles();
        if n_q < 2 {
            return Err(JsError::new("need at least 2 quantiles for intervals"));
        }
        let lower: Vec<f64> = (0..qf.n_times())
            .map(|t| qf.at_time(t).unwrap()[0])
            .collect();
        let upper: Vec<f64> = (0..qf.n_times())
            .map(|t| qf.at_time(t).unwrap()[n_q - 1])
            .collect();
        let qs = qf.quantiles();
        let coverage = qs[n_q - 1] - qs[0];
        let intervals = InnerPredictionIntervals::from_bounds(lower, upper, coverage)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsPredictionIntervals::from_inner(intervals))
    }
}

// =============================================================================
// TRAINED MODEL (opaque handle)
// =============================================================================

/// An opaque handle to a trained postprocessing model.
///
/// Returned by `JsPostProcessor.train()` and consumed by prediction methods.
#[wasm_bindgen]
pub struct JsTrainedModel {
    inner: InnerTrainedModel,
}

// =============================================================================
// POST PROCESSOR — unified API
// =============================================================================

/// Unified postprocessor that wraps conformal, normal, and historical simulation
/// methods behind a single API.
#[wasm_bindgen]
pub struct JsPostProcessor {
    inner: InnerPostProcessor,
}

#[wasm_bindgen]
impl JsPostProcessor {
    /// Create a conformal prediction postprocessor.
    ///
    /// @param coverage - Target coverage level in (0, 1), e.g. 0.90
    pub fn conformal(coverage: f64) -> Self {
        Self {
            inner: InnerPostProcessor::conformal(coverage),
        }
    }

    /// Create a normal prediction postprocessor.
    ///
    /// @param quantiles - Sorted quantile levels in (0,1)
    pub fn normal(quantiles: Vec<f64>) -> Self {
        Self {
            inner: InnerPostProcessor::normal(quantiles),
        }
    }

    /// Create a historical simulation postprocessor.
    ///
    /// @param quantiles - Sorted quantile levels in (0,1)
    #[wasm_bindgen(js_name = historicalSim)]
    pub fn historical_sim(quantiles: Vec<f64>) -> Self {
        Self {
            inner: InnerPostProcessor::historical_sim(quantiles),
        }
    }

    /// Train the postprocessor on historical data.
    ///
    /// @param forecasts - JsPointForecasts with historical predictions
    /// @param actuals - Corresponding actual observed values
    pub fn train(
        &self,
        forecasts: &JsPointForecasts,
        actuals: Vec<f64>,
    ) -> Result<JsTrainedModel, JsError> {
        self.inner
            .train(forecasts.inner(), &actuals)
            .map(|m| JsTrainedModel { inner: m })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate prediction intervals from a trained model.
    ///
    /// @param trained - A JsTrainedModel from train()
    /// @param forecasts - New point forecasts
    #[wasm_bindgen(js_name = predictIntervals)]
    pub fn predict_intervals(
        &self,
        trained: &JsTrainedModel,
        forecasts: &JsPointForecasts,
    ) -> Result<JsPredictionIntervals, JsError> {
        self.inner
            .predict_intervals(&trained.inner, forecasts.inner())
            .map(JsPredictionIntervals::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

// =============================================================================
// BACKTEST CONFIG AND RESULT
// =============================================================================

/// Configuration for backtesting a postprocessor.
#[wasm_bindgen]
pub struct JsBacktestConfig {
    inner: InnerBacktestConfig,
}

#[wasm_bindgen]
impl JsBacktestConfig {
    /// Create a new backtest configuration with default settings.
    ///
    /// Defaults: initial_window=50, step=1, horizon=1, expanding=true.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: InnerBacktestConfig::default(),
        }
    }

    /// Set the initial training window size.
    #[wasm_bindgen(js_name = initialWindow)]
    pub fn initial_window(mut self, size: usize) -> Self {
        self.inner = self.inner.initial_window(size);
        self
    }

    /// Set the step size between folds.
    pub fn step(mut self, step: usize) -> Self {
        self.inner = self.inner.step(step);
        self
    }

    /// Set the forecast horizon.
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.inner = self.inner.horizon(horizon);
        self
    }

    /// Set whether to use expanding (true) or rolling (false) window.
    pub fn expanding(mut self, expanding: bool) -> Self {
        self.inner = self.inner.expanding(expanding);
        self
    }
}

impl Default for JsBacktestConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from backtesting a postprocessor.
#[wasm_bindgen]
pub struct JsBacktestResult {
    inner: InnerBacktestResult,
}

#[wasm_bindgen]
impl JsBacktestResult {
    /// Get the number of backtest folds.
    #[wasm_bindgen(js_name = numFolds)]
    pub fn n_folds(&self) -> usize {
        self.inner.n_folds()
    }

    /// Get overall coverage across all folds.
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    /// Get average interval width across all folds.
    #[wasm_bindgen(js_name = intervalWidths)]
    pub fn interval_widths(&self) -> f64 {
        self.inner.interval_widths()
    }

    /// Get calibration error (absolute deviation from target coverage).
    #[wasm_bindgen(js_name = calibrationError)]
    pub fn calibration_error(&self, target_coverage: f64) -> f64 {
        self.inner.calibration_error(target_coverage)
    }
}

/// Run a backtest on a postprocessor.
///
/// @param processor - The postprocessor to evaluate
/// @param forecasts - All historical point forecasts
/// @param actuals - All historical actual values
/// @param config - Backtest configuration
#[wasm_bindgen(js_name = backtestPostProcessor)]
pub fn backtest_post_processor(
    processor: &JsPostProcessor,
    forecasts: &JsPointForecasts,
    actuals: Vec<f64>,
    config: JsBacktestConfig,
) -> Result<JsBacktestResult, JsError> {
    processor
        .inner
        .backtest(forecasts.inner(), &actuals, config.inner)
        .map(|r| JsBacktestResult { inner: r })
        .map_err(|e| JsError::new(&e.to_string()))
}

// =============================================================================
// SERIALIZABLE QUANTILE FORECASTS (shared helper)
// =============================================================================

/// Serialized representation of QuantileForecasts for JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct QuantileForecastsJs {
    /// Quantile levels.
    quantiles: Vec<f64>,
    /// Values matrix: values[time_idx][quantile_idx].
    values: Vec<Vec<f64>>,
}

impl QuantileForecastsJs {
    fn from_inner(qf: &InnerQuantileForecasts) -> Self {
        Self {
            quantiles: qf.quantiles().to_vec(),
            values: qf.values().to_vec(),
        }
    }
}

// =============================================================================
// BOOTSTRAP PREDICTOR (postprocess)
// =============================================================================

/// Bootstrap result — stores residuals and configuration from fitting.
#[wasm_bindgen]
pub struct JsBootstrapResult {
    inner: InnerBootstrapResult,
}

#[wasm_bindgen]
impl JsBootstrapResult {
    /// Get the residuals used for resampling.
    pub fn residuals(&self) -> Vec<f64> {
        self.inner.residuals().to_vec()
    }

    /// Get the coverage level.
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    /// Get the number of replicates.
    #[wasm_bindgen(js_name = nReplicates, getter)]
    pub fn n_replicates(&self) -> usize {
        self.inner.n_replicates()
    }
}

/// Bootstrap predictor for distribution-free prediction intervals.
///
/// Resamples forecast residuals to simulate future error paths, producing
/// per-step prediction intervals where uncertainty grows with horizon.
#[wasm_bindgen]
pub struct JsBootstrapPredictor {
    inner: InnerBootstrapPredictor,
}

#[wasm_bindgen]
impl JsBootstrapPredictor {
    /// Create a bootstrap predictor with the given coverage level.
    ///
    /// @param coverage - Target coverage level in (0, 1), e.g. 0.90
    #[wasm_bindgen(constructor)]
    pub fn new(coverage: f64) -> Self {
        Self {
            inner: InnerBootstrapPredictor::new(coverage),
        }
    }

    /// Set the number of bootstrap replicates (default: 1000).
    ///
    /// @param n - Number of replicates (minimum 10)
    #[wasm_bindgen(js_name = nReplicates)]
    pub fn n_replicates(self, n: usize) -> Self {
        Self {
            inner: self.inner.n_replicates(n),
        }
    }

    /// Use block bootstrap with the given block size.
    ///
    /// Preserves autocorrelation in residuals.
    ///
    /// @param size - Block size (0 falls back to IID)
    #[wasm_bindgen(js_name = blockSize)]
    pub fn block_size(self, size: usize) -> Self {
        Self {
            inner: self.inner.block_size(size),
        }
    }

    /// Set random seed for reproducibility.
    ///
    /// @param seed - Random seed value
    pub fn seed(self, seed: u64) -> Self {
        Self {
            inner: self.inner.seed(seed),
        }
    }

    /// Fit the predictor on historical forecasts and actuals.
    ///
    /// Computes residuals and stores them for resampling.
    ///
    /// @param forecasts - Historical point forecast values
    /// @param actuals - Corresponding actual observed values
    pub fn fit(
        &self,
        forecasts: Vec<f64>,
        actuals: Vec<f64>,
    ) -> Result<JsBootstrapResult, JsError> {
        self.inner
            .fit(&forecasts, &actuals)
            .map(|r| JsBootstrapResult { inner: r })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate prediction intervals for new point forecasts.
    ///
    /// Returns a JsPredictionIntervals with lower and upper bounds.
    ///
    /// @param result - A fitted JsBootstrapResult from fit()
    /// @param pointForecasts - New point forecast values
    #[wasm_bindgen(js_name = predictIntervals)]
    pub fn predict_intervals(
        &self,
        result: &JsBootstrapResult,
        point_forecasts: Vec<f64>,
    ) -> JsPredictionIntervals {
        let intervals = self.inner.predict(&result.inner, &point_forecasts);
        JsPredictionIntervals::from_inner(intervals)
    }

    /// Generate quantile forecasts at multiple quantile levels.
    ///
    /// Returns a serialized object with `quantiles` (levels) and `values` (matrix).
    ///
    /// @param result - A fitted JsBootstrapResult from fit()
    /// @param pointForecasts - New point forecast values
    /// @param levels - Quantile levels in (0, 1), e.g. [0.1, 0.25, 0.5, 0.75, 0.9]
    #[wasm_bindgen(js_name = predictQuantiles)]
    pub fn predict_quantiles(
        &self,
        result: &JsBootstrapResult,
        point_forecasts: Vec<f64>,
        levels: Vec<f64>,
    ) -> Result<JsValue, JsError> {
        let qf = self
            .inner
            .predict_quantiles(&result.inner, &point_forecasts, &levels);
        let js_qf = QuantileForecastsJs::from_inner(&qf);
        serde_wasm_bindgen::to_value(&js_qf).map_err(|e| JsError::new(&e.to_string()))
    }
}

// =============================================================================
// PER-STEP CONFORMAL RESULT
// =============================================================================

/// Per-step conformal result — each horizon step has its own interval half-width.
///
/// Step h=1 gets a tighter interval than step h=12 when forecast error grows
/// with horizon.
#[wasm_bindgen]
pub struct JsPerStepConformalResult {
    inner: InnerPerStepConformalResult,
}

#[wasm_bindgen]
impl JsPerStepConformalResult {
    /// Get the per-step interval half-widths.
    #[wasm_bindgen(js_name = halfWidths, getter)]
    pub fn half_widths(&self) -> Vec<f64> {
        self.inner.half_widths().to_vec()
    }

    /// Get the coverage level.
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    /// Get the number of horizon steps.
    #[wasm_bindgen(getter)]
    pub fn horizon(&self) -> usize {
        self.inner.horizon()
    }

    /// Apply per-step intervals to a point forecast.
    ///
    /// Returns a JsPredictionIntervals with per-step lower and upper bounds.
    ///
    /// @param pointForecasts - Point forecast values (length must match horizon)
    #[wasm_bindgen(js_name = predictIntervals)]
    pub fn predict_intervals(&self, point_forecasts: Vec<f64>) -> JsPredictionIntervals {
        let intervals = self.inner.predict_intervals(&point_forecasts);
        JsPredictionIntervals::from_inner(intervals)
    }

    /// Apply per-step intervals and return lower/upper as a plain object.
    ///
    /// Returns a serialized object with `lower` and `upper` arrays.
    ///
    /// @param pointForecasts - Point forecast values (length must match horizon)
    pub fn predict(&self, point_forecasts: Vec<f64>) -> Result<JsValue, JsError> {
        let (lower, upper) = self.inner.predict(&point_forecasts);
        let result = PerStepPredictJs { lower, upper };
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }
}

/// Serialized per-step predict result.
#[derive(Serialize)]
struct PerStepPredictJs {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

// =============================================================================
// BINNED CONFORMAL PREDICTOR
// =============================================================================

/// Binned conformal predictor for heteroscedastic prediction intervals.
///
/// Bins calibration residuals by predicted magnitude so that intervals widen
/// where errors are systematically larger.
///
/// ```javascript
/// const binned = new JsBinnedConformalPredictor(0.9, 3);
/// const result = binned.fit(trainForecasts, trainActuals);
/// const intervals = binned.predict(result, newForecasts);
/// ```
#[wasm_bindgen]
pub struct JsBinnedConformalPredictor {
    inner: InnerBinnedConformalPredictor,
}

#[wasm_bindgen]
pub struct JsBinnedConformalResult {
    inner: InnerBinnedConformalResult,
}

#[wasm_bindgen]
impl JsBinnedConformalResult {
    #[wasm_bindgen(getter, js_name = binEdges)]
    pub fn bin_edges(&self) -> Vec<f64> {
        self.inner.bin_edges().to_vec()
    }

    #[wasm_bindgen(getter, js_name = binQuantiles)]
    pub fn bin_quantiles(&self) -> Vec<f64> {
        self.inner.bin_quantiles().to_vec()
    }

    #[wasm_bindgen(getter, js_name = globalQuantile)]
    pub fn global_quantile(&self) -> f64 {
        self.inner.global_quantile()
    }

    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> f64 {
        self.inner.coverage()
    }

    #[wasm_bindgen(getter, js_name = nBins)]
    pub fn n_bins(&self) -> usize {
        self.inner.n_bins()
    }
}

#[wasm_bindgen]
impl JsBinnedConformalPredictor {
    /// Create a binned conformal predictor with a target coverage and
    /// number of bins.
    #[wasm_bindgen(constructor)]
    pub fn new(coverage: f64, n_bins: usize) -> Self {
        Self {
            inner: InnerBinnedConformalPredictor::new(coverage, n_bins),
        }
    }

    /// Fit the predictor on historical forecasts and actuals.
    pub fn fit(
        &self,
        forecasts: &[f64],
        actuals: &[f64],
    ) -> Result<JsBinnedConformalResult, JsError> {
        let res = self
            .inner
            .fit(forecasts, actuals)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsBinnedConformalResult { inner: res })
    }

    /// Apply the fitted result to new point forecasts.
    pub fn predict(
        &self,
        result: &JsBinnedConformalResult,
        point_forecasts: &[f64],
    ) -> Result<JsValue, JsError> {
        let intervals = self.inner.predict(&result.inner, point_forecasts);
        let out = BinnedIntervalsJs {
            lower: intervals.lower().to_vec(),
            upper: intervals.upper().to_vec(),
            coverage: intervals.coverage(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BinnedIntervalsJs {
    lower: Vec<f64>,
    upper: Vec<f64>,
    coverage: f64,
}

// =============================================================================
// QRA PREDICTOR
// =============================================================================

/// Quantile Regression Averaging predictor — combines forecasts from multiple
/// models into calibrated quantile forecasts.
#[wasm_bindgen]
pub struct JsQRAPredictor {
    inner: InnerQRAPredictor,
}

#[wasm_bindgen]
pub struct JsQRAResult {
    inner: InnerQRAResult,
}

#[wasm_bindgen]
impl JsQRAResult {
    #[wasm_bindgen(getter)]
    pub fn quantiles(&self) -> Vec<f64> {
        self.inner.quantiles().to_vec()
    }

    #[wasm_bindgen(getter, js_name = nForecasters)]
    pub fn n_forecasters(&self) -> usize {
        self.inner.n_forecasters()
    }

    #[wasm_bindgen(getter, js_name = hasIntercept)]
    pub fn has_intercept(&self) -> bool {
        self.inner.has_intercept()
    }
}

#[wasm_bindgen]
impl JsQRAPredictor {
    /// Create a QRA predictor with the given target quantiles (unregularised).
    #[wasm_bindgen(constructor)]
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self {
            inner: InnerQRAPredictor::standard(quantiles),
        }
    }

    /// Create a Lasso-regularised QRA predictor.
    pub fn lasso(quantiles: Vec<f64>, lambda: f64) -> Self {
        Self {
            inner: InnerQRAPredictor::lasso(quantiles, lambda),
        }
    }

    /// Fit the predictor on a matrix of forecasts from multiple models and
    /// the corresponding actual values.
    ///
    /// @param forecastsMatrix - Vec<Vec<f64>>: rows = time, cols = models
    /// @param actuals - Actual values (length == rows of matrix)
    pub fn fit(&self, forecasts_matrix: JsValue, actuals: &[f64]) -> Result<JsQRAResult, JsError> {
        let matrix: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(forecasts_matrix)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let res = self
            .inner
            .fit(&matrix, actuals)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsQRAResult { inner: res })
    }

    /// Produce quantile forecasts for a new matrix of model forecasts.
    pub fn predict(
        &self,
        result: &JsQRAResult,
        forecasts_matrix: JsValue,
    ) -> Result<JsValue, JsError> {
        let matrix: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(forecasts_matrix)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let quantile_forecasts = self
            .inner
            .predict(&result.inner, &matrix)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let out = QuantileForecastsJs::from_inner(&quantile_forecasts);
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
    }
}

// =============================================================================
// IDR PREDICTOR
// =============================================================================

/// Isotonic Distributional Regression predictor — state-of-the-art
/// calibration via monotonic rank-based mapping.
#[wasm_bindgen]
pub struct JsIDRPredictor {
    inner: InnerIDRPredictor,
}

#[wasm_bindgen]
pub struct JsIDRResult {
    inner: InnerIDRResult,
}

#[wasm_bindgen]
impl JsIDRResult {
    #[wasm_bindgen(getter)]
    pub fn quantiles(&self) -> Vec<f64> {
        self.inner.quantiles().to_vec()
    }
}

#[wasm_bindgen]
impl JsIDRPredictor {
    /// Create an IDR predictor with the given target quantiles.
    #[wasm_bindgen(constructor)]
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self {
            inner: InnerIDRPredictor::new(quantiles),
        }
    }

    /// Fit on historical forecasts and actuals.
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<JsIDRResult, JsError> {
        let res = self
            .inner
            .fit(forecasts, actuals)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(JsIDRResult { inner: res })
    }

    /// Produce quantile forecasts for new point forecasts.
    pub fn predict(
        &self,
        result: &JsIDRResult,
        point_forecasts: &JsPointForecasts,
    ) -> Result<JsValue, JsError> {
        let quantile_forecasts = self
            .inner
            .predict(&result.inner, point_forecasts.inner())
            .map_err(|e| JsError::new(&e.to_string()))?;
        let out = QuantileForecastsJs::from_inner(&quantile_forecasts);
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
    }
}
