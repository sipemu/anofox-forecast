//! AutoForecast and AutoEnsemble wrappers for JavaScript.
//!
//! Provides wasm-bindgen wrappers for automatic model selection and
//! ensemble forecasting models.

use crate::time_series::{Forecast, TimeSeries};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::models::auto_forecast::{AutoForecast, AutoForecastConfig};
use anofox_forecast::models::ensemble::{AutoEnsemble, AutoEnsembleConfig};
use anofox_forecast::models::Forecaster;

// =============================================================================
// AutoForecast
// =============================================================================

/// Automatic model selection across ARIMA, ETS, and Theta families.
///
/// Fits all enabled auto models and selects the best one based on
/// cross-validation error.
#[wasm_bindgen]
pub struct AutoForecaster {
    model: AutoForecast,
}

#[wasm_bindgen]
impl AutoForecaster {
    /// Create a new AutoForecaster with default configuration.
    ///
    /// By default, includes AutoARIMA, AutoETS, and AutoTheta candidates
    /// and uses cross-validation for model selection.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: AutoForecast::new(),
        }
    }

    /// Create a seasonal AutoForecaster.
    ///
    /// @param period - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    #[wasm_bindgen]
    pub fn seasonal(period: usize) -> Self {
        Self {
            model: AutoForecast::seasonal(period),
        }
    }

    /// Create an AutoForecaster with custom configuration.
    ///
    /// @param seasonalPeriod - Seasonal period (0 or undefined for non-seasonal)
    /// @param includeArima - Include AutoARIMA candidate (default: true)
    /// @param includeEts - Include AutoETS candidate (default: true)
    /// @param includeTheta - Include AutoTheta candidate (default: true)
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(
        seasonal_period: Option<usize>,
        include_arima: Option<bool>,
        include_ets: Option<bool>,
        include_theta: Option<bool>,
    ) -> Self {
        let mut config = match seasonal_period {
            Some(p) if p > 1 => AutoForecastConfig::with_period(p),
            _ => AutoForecastConfig::default(),
        };

        if let Some(false) = include_arima {
            config = config.without_arima();
        }
        if let Some(false) = include_ets {
            config = config.without_ets();
        }
        if let Some(false) = include_theta {
            config = config.without_theta();
        }

        Self {
            model: AutoForecast::with_config(config),
        }
    }

    /// Fit the model to a time series.
    ///
    /// Fits all candidate models and selects the best one.
    ///
    /// @param series - TimeSeries to fit
    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict future values.
    ///
    /// @param horizon - Number of steps to forecast
    /// @returns Forecast with point predictions
    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict with prediction intervals.
    ///
    /// @param horizon - Number of steps to forecast
    /// @param level - Confidence level (e.g., 0.95 for 95% intervals)
    /// @returns Forecast with point predictions and intervals
    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the name of the selected model.
    ///
    /// @returns Name of the best model, or undefined if not yet fitted
    #[wasm_bindgen(js_name = selectedModelName)]
    pub fn selected_model_name(&self) -> Option<String> {
        self.model.selected_model_name().map(|s| s.to_string())
    }

    /// Get all candidate scores as a JSON object.
    ///
    /// Returns an array of `{ name, score }` objects sorted by score (ascending).
    #[wasm_bindgen(js_name = allScores)]
    pub fn all_scores(&self) -> Result<JsValue, JsError> {
        let scores: Vec<ModelScoreJs> = self
            .model
            .all_scores()
            .iter()
            .map(|(name, score)| ModelScoreJs {
                name: name.clone(),
                score: *score,
            })
            .collect();
        serde_wasm_bindgen::to_value(&scores).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the model name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoForecaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Score entry for model comparison.
#[derive(Serialize)]
struct ModelScoreJs {
    name: String,
    score: f64,
}

// =============================================================================
// AutoForecastBuilder
// =============================================================================

/// Builder for constructing an AutoForecaster with custom parameters.
///
/// Uses a fluent API: chain setter methods and call `build()` to produce
/// an `AutoForecaster`.
#[wasm_bindgen]
pub struct AutoForecastBuilder {
    seasonal_period: Option<usize>,
    include_arima: Option<bool>,
    include_ets: Option<bool>,
    include_theta: Option<bool>,
}

#[wasm_bindgen]
impl AutoForecastBuilder {
    /// Create a new builder with all defaults enabled.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            seasonal_period: None,
            include_arima: None,
            include_ets: None,
            include_theta: None,
        }
    }

    /// Set the seasonal period.
    ///
    /// @param period - Seasonal period (e.g., 12 for monthly data)
    #[wasm_bindgen(js_name = seasonalPeriod)]
    pub fn seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = Some(period);
        self
    }

    /// Include or exclude AutoARIMA from the candidate set.
    ///
    /// @param include - Whether to include AutoARIMA (default: true)
    #[wasm_bindgen(js_name = includeArima)]
    pub fn include_arima(mut self, include: bool) -> Self {
        self.include_arima = Some(include);
        self
    }

    /// Include or exclude AutoETS from the candidate set.
    ///
    /// @param include - Whether to include AutoETS (default: true)
    #[wasm_bindgen(js_name = includeEts)]
    pub fn include_ets(mut self, include: bool) -> Self {
        self.include_ets = Some(include);
        self
    }

    /// Include or exclude AutoTheta from the candidate set.
    ///
    /// @param include - Whether to include AutoTheta (default: true)
    #[wasm_bindgen(js_name = includeTheta)]
    pub fn include_theta(mut self, include: bool) -> Self {
        self.include_theta = Some(include);
        self
    }

    /// Build the AutoForecaster with the configured parameters.
    ///
    /// @returns A new AutoForecaster ready to fit
    pub fn build(self) -> AutoForecaster {
        let mut builder = AutoForecast::builder();

        if let Some(p) = self.seasonal_period {
            builder = builder.seasonal_period(p);
        }
        if let Some(v) = self.include_arima {
            builder = builder.include_arima(v);
        }
        if let Some(v) = self.include_ets {
            builder = builder.include_ets(v);
        }
        if let Some(v) = self.include_theta {
            builder = builder.include_theta(v);
        }

        AutoForecaster {
            model: builder.build(),
        }
    }
}

impl Default for AutoForecastBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// AutoEnsemble
// =============================================================================

/// Automatic ensemble that selects top-K models across families.
///
/// Fits AutoARIMA, AutoETS, and AutoTheta, ranks them by cross-validation error,
/// and combines the top-K into a weighted ensemble forecast.
#[wasm_bindgen]
pub struct AutoEnsembleForecaster {
    model: AutoEnsemble,
}

#[wasm_bindgen]
impl AutoEnsembleForecaster {
    /// Create a new AutoEnsembleForecaster with default configuration.
    ///
    /// By default, uses top-3 models with weighted MSE combination.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: AutoEnsemble::new(),
        }
    }

    /// Create a seasonal AutoEnsembleForecaster.
    ///
    /// @param period - Seasonal period
    #[wasm_bindgen]
    pub fn seasonal(period: usize) -> Self {
        Self {
            model: AutoEnsemble::seasonal(period),
        }
    }

    /// Create with custom configuration.
    ///
    /// @param topK - Number of top models to include in ensemble (default: 3)
    /// @param seasonalPeriod - Seasonal period (0 or undefined for non-seasonal)
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(top_k: Option<usize>, seasonal_period: Option<usize>) -> Self {
        let mut config = match seasonal_period {
            Some(p) if p > 1 => AutoEnsembleConfig::with_period(p),
            _ => AutoEnsembleConfig::default(),
        };

        if let Some(k) = top_k {
            config = config.with_top_k(k);
        }

        Self {
            model: AutoEnsemble::with_config(config),
        }
    }

    /// Fit the ensemble to a time series.
    ///
    /// Fits all candidate models, selects the top-K, and builds the ensemble.
    ///
    /// @param series - TimeSeries to fit
    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict future values.
    ///
    /// @param horizon - Number of steps to forecast
    /// @returns Forecast with combined point predictions
    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict with prediction intervals.
    ///
    /// @param horizon - Number of steps to forecast
    /// @param level - Confidence level (e.g., 0.95 for 95% intervals)
    /// @returns Forecast with combined predictions and intervals
    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the number of models in the ensemble.
    #[wasm_bindgen(js_name = modelCount)]
    pub fn model_count(&self) -> usize {
        self.model.model_count()
    }

    /// Get all candidate scores as a JSON object.
    ///
    /// Returns an array of `{ name, score }` objects sorted by score (ascending).
    #[wasm_bindgen(js_name = allScores)]
    pub fn all_scores(&self) -> Result<JsValue, JsError> {
        let scores: Vec<ModelScoreJs> = self
            .model
            .all_scores()
            .iter()
            .map(|(name, score)| ModelScoreJs {
                name: name.clone(),
                score: *score,
            })
            .collect();
        serde_wasm_bindgen::to_value(&scores).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the model name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoEnsembleForecaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn make_test_values() -> Vec<f64> {
        (0..60)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect()
    }

    #[wasm_bindgen_test]
    fn test_auto_forecaster_basic() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoForecaster::new();
        model.fit(&ts).unwrap();

        assert!(model.selected_model_name().is_some());
        assert_eq!(model.name(), "AutoForecast");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[wasm_bindgen_test]
    fn test_auto_forecaster_with_intervals() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert_eq!(forecast.horizon(), 5);
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[wasm_bindgen_test]
    fn test_auto_forecaster_seasonal() {
        let values: Vec<f64> = (0..100)
            .map(|i| {
                50.0 + 0.3 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
            })
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoForecaster::seasonal(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[wasm_bindgen_test]
    fn test_auto_forecaster_scores() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoForecaster::new();
        model.fit(&ts).unwrap();

        let scores = model.all_scores();
        assert!(scores.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_auto_ensemble_basic() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoEnsembleForecaster::new();
        model.fit(&ts).unwrap();

        assert!(model.model_count() > 0);
        assert_eq!(model.name(), "AutoEnsemble");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[wasm_bindgen_test]
    fn test_auto_ensemble_with_intervals() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoEnsembleForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[wasm_bindgen_test]
    fn test_auto_ensemble_custom_top_k() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoEnsembleForecaster::with_config(Some(2), None);
        model.fit(&ts).unwrap();

        assert!(model.model_count() <= 2);
    }

    #[wasm_bindgen_test]
    fn test_auto_ensemble_scores() {
        let values = make_test_values();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = AutoEnsembleForecaster::new();
        model.fit(&ts).unwrap();

        let scores = model.all_scores();
        assert!(scores.is_ok());
    }
}
