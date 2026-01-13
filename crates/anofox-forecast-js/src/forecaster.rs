//! Forecaster wrappers for JavaScript.
//!
//! This module provides wasm-bindgen wrappers for all forecasting models
//! available in the anofox-forecast crate.

use crate::time_series::{Forecast, TimeSeries};
use wasm_bindgen::prelude::*;

// Import all models from anofox-forecast
use anofox_forecast::models::Forecaster as ForecasterTrait;

// Baseline models
use anofox_forecast::models::baseline::{
    HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive, SeasonalWindowAverage,
    SimpleMovingAverage, WindowAverage,
};

// Exponential smoothing models
use anofox_forecast::models::exponential::{
    AutoETS, HoltLinearTrend, HoltWinters, SeasonalES, SimpleExponentialSmoothing, ETS,
};

// Theta models
use anofox_forecast::models::theta::{AutoTheta, DynamicTheta, OptimizedTheta, Theta};

// ARIMA models
use anofox_forecast::models::arima::{AutoARIMA, ARIMA, SARIMA};

// Intermittent demand models
use anofox_forecast::models::intermittent::{Croston, ADIDA, IMAPA, TSB};

// Advanced models
use anofox_forecast::models::garch::GARCH;
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
use anofox_forecast::models::tbats::{AutoTBATS, TBATS};

// =============================================================================
// BASELINE MODELS
// =============================================================================

/// Naive forecaster - uses the last observation as forecast.
#[wasm_bindgen]
pub struct NaiveForecaster {
    model: Naive,
}

#[wasm_bindgen]
impl NaiveForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: Naive::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for NaiveForecaster {
    fn default() -> Self { Self::new() }
}

/// Mean (Historic Average) forecaster - uses the historical mean as forecast.
#[wasm_bindgen]
pub struct MeanForecaster {
    model: HistoricAverage,
}

#[wasm_bindgen]
impl MeanForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: HistoricAverage::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for MeanForecaster {
    fn default() -> Self { Self::new() }
}

/// Seasonal Naive forecaster - uses observations from the same season.
#[wasm_bindgen]
pub struct SeasonalNaiveForecaster {
    model: SeasonalNaive,
}

#[wasm_bindgen]
impl SeasonalNaiveForecaster {
    /// @param period - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Self {
        Self { model: SeasonalNaive::new(period) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Random Walk with Drift forecaster.
#[wasm_bindgen]
pub struct RandomWalkDriftForecaster {
    model: RandomWalkWithDrift,
}

#[wasm_bindgen]
impl RandomWalkDriftForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: RandomWalkWithDrift::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for RandomWalkDriftForecaster {
    fn default() -> Self { Self::new() }
}

/// Simple Moving Average forecaster.
#[wasm_bindgen]
pub struct SMAForecaster {
    model: SimpleMovingAverage,
}

#[wasm_bindgen]
impl SMAForecaster {
    /// @param window - Window size for the moving average
    #[wasm_bindgen(constructor)]
    pub fn new(window: usize) -> Self {
        Self { model: SimpleMovingAverage::new(window) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Window Average forecaster - uses the last N observations.
#[wasm_bindgen]
pub struct WindowAverageForecaster {
    model: WindowAverage,
}

#[wasm_bindgen]
impl WindowAverageForecaster {
    /// @param window_size - Size of the rolling window
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Self {
        Self { model: WindowAverage::new(window_size) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Seasonal Window Average forecaster.
#[wasm_bindgen]
pub struct SeasonalWindowAverageForecaster {
    model: SeasonalWindowAverage,
}

#[wasm_bindgen]
impl SeasonalWindowAverageForecaster {
    /// @param period - Seasonal period
    /// @param window - Number of seasonal cycles to average
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize, window: usize) -> Self {
        Self { model: SeasonalWindowAverage::new(period, window) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

// =============================================================================
// EXPONENTIAL SMOOTHING MODELS
// =============================================================================

/// Simple Exponential Smoothing forecaster.
#[wasm_bindgen]
pub struct SESForecaster {
    model: SimpleExponentialSmoothing,
}

#[wasm_bindgen]
impl SESForecaster {
    /// @param alpha - Smoothing parameter (0 < alpha <= 1)
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64) -> Self {
        Self { model: SimpleExponentialSmoothing::new(alpha) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Holt Linear Trend (Double Exponential Smoothing) forecaster.
#[wasm_bindgen]
pub struct HoltForecaster {
    model: HoltLinearTrend,
}

#[wasm_bindgen]
impl HoltForecaster {
    /// @param alpha - Level smoothing parameter (0 < alpha <= 1)
    /// @param beta - Trend smoothing parameter (0 < beta <= 1)
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { model: HoltLinearTrend::new(alpha, beta) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Holt-Winters (Triple Exponential Smoothing) forecaster.
#[wasm_bindgen]
pub struct HoltWintersForecaster {
    model: HoltWinters,
}

#[wasm_bindgen]
impl HoltWintersForecaster {
    /// Create with additive seasonality.
    /// @param alpha - Level smoothing parameter
    /// @param beta - Trend smoothing parameter
    /// @param gamma - Seasonal smoothing parameter
    /// @param period - Seasonal period
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        Self { model: HoltWinters::additive(alpha, beta, gamma, period) }
    }

    /// Create with multiplicative seasonality.
    #[wasm_bindgen(js_name = multiplicative)]
    pub fn multiplicative(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        Self { model: HoltWinters::multiplicative(alpha, beta, gamma, period) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Seasonal Exponential Smoothing forecaster.
#[wasm_bindgen]
pub struct SeasonalESForecaster {
    model: SeasonalES,
}

#[wasm_bindgen]
impl SeasonalESForecaster {
    /// @param period - Seasonal period
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Self {
        Self { model: SeasonalES::new(period) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// ETS (Error-Trend-Seasonal) state-space model.
/// Use string codes: "A" = Additive, "M" = Multiplicative, "N" = None
#[wasm_bindgen]
pub struct ETSForecaster {
    model: ETS,
}

#[wasm_bindgen]
impl ETSForecaster {
    /// Create an ETS model with specified components.
    /// @param error - Error type: "A" (additive) or "M" (multiplicative)
    /// @param trend - Trend type: "N" (none), "A" (additive), or "Ad" (additive damped)
    /// @param seasonal - Seasonal type: "N" (none), "A" (additive), or "M" (multiplicative)
    /// @param period - Seasonal period (ignored if seasonal is "N")
    #[wasm_bindgen(constructor)]
    pub fn new(error: &str, trend: &str, seasonal: &str, period: usize) -> Result<ETSForecaster, JsError> {
        use anofox_forecast::models::exponential::{ErrorType, TrendType, ETSSeasonalType, ETSSpec};

        let error_type = match error.to_uppercase().as_str() {
            "A" => ErrorType::Additive,
            "M" => ErrorType::Multiplicative,
            _ => return Err(JsError::new("Error type must be 'A' or 'M'")),
        };

        let trend_type = match trend.to_uppercase().as_str() {
            "N" => TrendType::None,
            "A" => TrendType::Additive,
            "AD" => TrendType::AdditiveDamped,
            _ => return Err(JsError::new("Trend type must be 'N', 'A', or 'Ad'")),
        };

        let seasonal_type = match seasonal.to_uppercase().as_str() {
            "N" => ETSSeasonalType::None,
            "A" => ETSSeasonalType::Additive,
            "M" => ETSSeasonalType::Multiplicative,
            _ => return Err(JsError::new("Seasonal type must be 'N', 'A', or 'M'")),
        };

        let spec = ETSSpec::new(error_type, trend_type, seasonal_type);
        Ok(Self { model: ETS::new(spec, period) })
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// AutoETS - Automatic ETS model selection.
#[wasm_bindgen]
pub struct AutoETSForecaster {
    model: AutoETS,
}

#[wasm_bindgen]
impl AutoETSForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: AutoETS::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoETSForecaster {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// THETA MODELS
// =============================================================================

/// Theta forecaster - the standard Theta method.
#[wasm_bindgen]
pub struct ThetaForecaster {
    model: Theta,
}

#[wasm_bindgen]
impl ThetaForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: Theta::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for ThetaForecaster {
    fn default() -> Self { Self::new() }
}

/// Optimized Theta forecaster - automatically optimizes parameters.
#[wasm_bindgen]
pub struct OptimizedThetaForecaster {
    model: OptimizedTheta,
}

#[wasm_bindgen]
impl OptimizedThetaForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: OptimizedTheta::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for OptimizedThetaForecaster {
    fn default() -> Self { Self::new() }
}

/// Dynamic Theta forecaster - updates coefficients dynamically.
#[wasm_bindgen]
pub struct DynamicThetaForecaster {
    model: DynamicTheta,
}

#[wasm_bindgen]
impl DynamicThetaForecaster {
    /// @param alpha - Smoothing parameter for the forecast
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64) -> Self {
        Self { model: DynamicTheta::new(alpha) }
    }

    /// Create an optimized Dynamic Theta model.
    pub fn optimized() -> Self {
        Self { model: DynamicTheta::optimized() }
    }

    /// Create a seasonal Dynamic Theta model.
    /// @param period - Seasonal period
    pub fn seasonal(period: usize) -> Self {
        Self { model: DynamicTheta::seasonal(period) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// AutoTheta - Automatic Theta model selection.
#[wasm_bindgen]
pub struct AutoThetaForecaster {
    model: AutoTheta,
}

#[wasm_bindgen]
impl AutoThetaForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: AutoTheta::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoThetaForecaster {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// ARIMA MODELS
// =============================================================================

/// ARIMA forecaster - Autoregressive Integrated Moving Average.
#[wasm_bindgen]
pub struct ARIMAForecaster {
    model: ARIMA,
}

#[wasm_bindgen]
impl ARIMAForecaster {
    /// @param p - AR order (autoregressive)
    /// @param d - Differencing order
    /// @param q - MA order (moving average)
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { model: ARIMA::new(p, d, q) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// SARIMA forecaster - Seasonal ARIMA.
#[wasm_bindgen]
pub struct SARIMAForecaster {
    model: SARIMA,
}

#[wasm_bindgen]
impl SARIMAForecaster {
    /// @param p - AR order
    /// @param d - Differencing order
    /// @param q - MA order
    /// @param seasonal_p - Seasonal AR order
    /// @param seasonal_d - Seasonal differencing order
    /// @param seasonal_q - Seasonal MA order
    /// @param period - Seasonal period
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize, seasonal_p: usize, seasonal_d: usize, seasonal_q: usize, period: usize) -> Self {
        Self { model: SARIMA::new(p, d, q, seasonal_p, seasonal_d, seasonal_q, period) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// AutoARIMA - Automatic ARIMA order selection.
#[wasm_bindgen]
pub struct AutoARIMAForecaster {
    model: AutoARIMA,
}

#[wasm_bindgen]
impl AutoARIMAForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: AutoARIMA::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model.predict_with_intervals(horizon, level).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoARIMAForecaster {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// INTERMITTENT DEMAND MODELS
// =============================================================================

/// Croston's method for intermittent demand forecasting.
#[wasm_bindgen]
pub struct CrostonForecaster {
    model: Croston,
}

#[wasm_bindgen]
impl CrostonForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: Croston::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for CrostonForecaster {
    fn default() -> Self { Self::new() }
}

/// TSB (Teunter-Syntetos-Babai) method for intermittent demand.
#[wasm_bindgen]
pub struct TSBForecaster {
    model: TSB,
}

#[wasm_bindgen]
impl TSBForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: TSB::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for TSBForecaster {
    fn default() -> Self { Self::new() }
}

/// ADIDA (Aggregate-Disaggregate Intermittent Demand Approach).
#[wasm_bindgen]
pub struct ADIDAForecaster {
    model: ADIDA,
}

#[wasm_bindgen]
impl ADIDAForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: ADIDA::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for ADIDAForecaster {
    fn default() -> Self { Self::new() }
}

/// IMAPA (Intermittent Multiple Aggregation Prediction Algorithm).
#[wasm_bindgen]
pub struct IMAPAForecaster {
    model: IMAPA,
}

#[wasm_bindgen]
impl IMAPAForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model: IMAPA::new() }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for IMAPAForecaster {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// ADVANCED MODELS
// =============================================================================

/// TBATS - Trigonometric seasonality, Box-Cox, ARMA errors, Trend, Seasonal.
#[wasm_bindgen]
pub struct TBATSForecaster {
    model: TBATS,
}

#[wasm_bindgen]
impl TBATSForecaster {
    /// @param seasonal_periods - Array of seasonal periods (e.g., [7, 365] for daily data)
    #[wasm_bindgen(constructor)]
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        Self { model: TBATS::new(seasonal_periods) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// AutoTBATS - Automatic TBATS model selection.
#[wasm_bindgen]
pub struct AutoTBATSForecaster {
    model: AutoTBATS,
}

#[wasm_bindgen]
impl AutoTBATSForecaster {
    /// @param seasonal_periods - Array of seasonal periods
    #[wasm_bindgen(constructor)]
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        Self { model: AutoTBATS::new(seasonal_periods) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// MFLES - Multiple Frequency Locally Estimated Scatterplot Smoothing.
#[wasm_bindgen]
pub struct MFLESForecaster {
    model: MFLES,
}

#[wasm_bindgen]
impl MFLESForecaster {
    /// @param seasonal_periods - Array of seasonal periods
    #[wasm_bindgen(constructor)]
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        Self { model: MFLES::new(seasonal_periods) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// MSTL Forecaster - Multiple Seasonal-Trend decomposition using LOESS.
#[wasm_bindgen]
pub struct MSTLForecasterWrapper {
    model: MSTLForecaster,
}

#[wasm_bindgen]
impl MSTLForecasterWrapper {
    /// @param seasonal_periods - Array of seasonal periods
    #[wasm_bindgen(constructor)]
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        Self { model: MSTLForecaster::new(seasonal_periods) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// GARCH - Generalized Autoregressive Conditional Heteroskedasticity.
#[wasm_bindgen]
pub struct GARCHForecaster {
    model: GARCH,
}

#[wasm_bindgen]
impl GARCHForecaster {
    /// @param p - GARCH order (lagged variance terms)
    /// @param q - ARCH order (lagged squared residuals)
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, q: usize) -> Self {
        Self { model: GARCH::new(p, q) }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model.fit(series.inner()).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model.predict(horizon).map(Forecast::from_inner).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}
