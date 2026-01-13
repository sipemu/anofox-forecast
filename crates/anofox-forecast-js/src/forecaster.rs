//! Forecaster wrappers for JavaScript.

use crate::time_series::{Forecast, TimeSeries};
use anofox_forecast::models::baseline::{
    HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive, WindowAverage,
};
use anofox_forecast::models::exponential::{HoltLinearTrend, HoltWinters, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::{DynamicTheta, OptimizedTheta, Theta};
use anofox_forecast::models::Forecaster as ForecasterTrait;
use wasm_bindgen::prelude::*;

/// Naive forecaster - uses the last observation as forecast.
#[wasm_bindgen]
pub struct NaiveForecaster {
    model: Naive,
}

#[wasm_bindgen]
impl NaiveForecaster {
    /// Create a new Naive forecaster.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        NaiveForecaster {
            model: Naive::new(),
        }
    }

    /// Fit the model to the time series.
    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate predictions.
    ///
    /// @param horizon - Number of steps to forecast
    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    /// Generate predictions with confidence intervals.
    ///
    /// @param horizon - Number of steps to forecast
    /// @param level - Confidence level (e.g., 0.95 for 95%)
    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    /// Get the model name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for NaiveForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean forecaster - uses the historical mean as forecast.
#[wasm_bindgen]
pub struct MeanForecaster {
    model: HistoricAverage,
}

#[wasm_bindgen]
impl MeanForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        MeanForecaster {
            model: HistoricAverage::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for MeanForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Seasonal Naive forecaster - uses observations from the same season.
#[wasm_bindgen]
pub struct SeasonalNaiveForecaster {
    model: SeasonalNaive,
}

#[wasm_bindgen]
impl SeasonalNaiveForecaster {
    /// Create a new Seasonal Naive forecaster.
    ///
    /// @param period - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Self {
        SeasonalNaiveForecaster {
            model: SeasonalNaive::new(period),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
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
        RandomWalkDriftForecaster {
            model: RandomWalkWithDrift::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for RandomWalkDriftForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Window Average forecaster - uses a rolling window mean.
#[wasm_bindgen]
pub struct WindowAverageForecaster {
    model: WindowAverage,
}

#[wasm_bindgen]
impl WindowAverageForecaster {
    /// Create a new Window Average forecaster.
    ///
    /// @param window_size - Size of the rolling window
    #[wasm_bindgen(constructor)]
    pub fn new(window_size: usize) -> Self {
        WindowAverageForecaster {
            model: WindowAverage::new(window_size),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Simple Exponential Smoothing forecaster.
#[wasm_bindgen]
pub struct SESForecaster {
    model: SimpleExponentialSmoothing,
}

#[wasm_bindgen]
impl SESForecaster {
    /// Create a new Simple Exponential Smoothing forecaster.
    ///
    /// @param alpha - Smoothing parameter (0 < alpha <= 1)
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64) -> Self {
        SESForecaster {
            model: SimpleExponentialSmoothing::new(alpha),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
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
    /// Create a new Holt Linear Trend forecaster.
    ///
    /// @param alpha - Level smoothing parameter (0 < alpha <= 1)
    /// @param beta - Trend smoothing parameter (0 < beta <= 1)
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, beta: f64) -> Self {
        HoltForecaster {
            model: HoltLinearTrend::new(alpha, beta),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
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
    /// Create a new Holt-Winters forecaster with additive seasonality.
    ///
    /// @param alpha - Level smoothing parameter
    /// @param beta - Trend smoothing parameter
    /// @param gamma - Seasonal smoothing parameter
    /// @param period - Seasonal period
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        HoltWintersForecaster {
            model: HoltWinters::additive(alpha, beta, gamma, period),
        }
    }

    /// Create a new Holt-Winters forecaster with multiplicative seasonality.
    #[wasm_bindgen(js_name = multiplicative)]
    pub fn multiplicative(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        HoltWintersForecaster {
            model: HoltWinters::multiplicative(alpha, beta, gamma, period),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Theta forecaster - the standard Theta method.
#[wasm_bindgen]
pub struct ThetaForecaster {
    model: Theta,
}

#[wasm_bindgen]
impl ThetaForecaster {
    /// Create a new Theta forecaster (uses theta=2.0).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        ThetaForecaster {
            model: Theta::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for ThetaForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic Theta forecaster - allows specifying a custom theta parameter.
#[wasm_bindgen]
pub struct DynamicThetaForecaster {
    model: DynamicTheta,
}

#[wasm_bindgen]
impl DynamicThetaForecaster {
    /// Create a new Dynamic Theta forecaster.
    ///
    /// @param alpha - Smoothing parameter for the forecast
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64) -> Self {
        DynamicThetaForecaster {
            model: DynamicTheta::new(alpha),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// Optimized Theta forecaster - automatically optimizes theta parameter.
#[wasm_bindgen]
pub struct OptimizedThetaForecaster {
    model: OptimizedTheta,
}

#[wasm_bindgen]
impl OptimizedThetaForecaster {
    /// Create a new Optimized Theta forecaster.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        OptimizedThetaForecaster {
            model: OptimizedTheta::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict(horizon)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        let forecast = self
            .model
            .predict_with_intervals(horizon, level)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Forecast::from_inner(forecast))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for OptimizedThetaForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn make_test_series() -> TimeSeries {
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        TimeSeries::new(&values).unwrap()
    }

    #[wasm_bindgen_test]
    fn test_naive_forecaster() {
        let ts = make_test_series();
        let mut model = NaiveForecaster::new();

        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        assert_eq!(forecast.horizon, 5);
        assert_eq!(forecast.values.len(), 5);
        // Naive forecast should be the last value repeated
        for val in forecast.values.iter() {
            assert_eq!(*val, 20.0);
        }
    }

    #[wasm_bindgen_test]
    fn test_theta_forecaster() {
        let ts = make_test_series();
        let mut model = ThetaForecaster::new();

        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        assert_eq!(forecast.horizon, 5);
        assert!(!forecast.values.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_ses_forecaster() {
        let ts = make_test_series();
        let mut model = SESForecaster::new(0.3);

        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        assert_eq!(forecast.horizon, 5);
    }
}
