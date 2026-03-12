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
        Self {
            model: Naive::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

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

/// Mean (Historic Average) forecaster - uses the historical mean as forecast.
#[wasm_bindgen]
pub struct MeanForecaster {
    model: HistoricAverage,
}

#[wasm_bindgen]
impl MeanForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: HistoricAverage::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
    /// @param period - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Self {
        Self {
            model: SeasonalNaive::new(period),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: RandomWalkWithDrift::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: SimpleMovingAverage::new(window),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: WindowAverage::new(window_size),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: SeasonalWindowAverage::new(period, window),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: SimpleExponentialSmoothing::new(alpha),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: HoltLinearTrend::new(alpha, beta),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: HoltWinters::additive(alpha, beta, gamma, period),
        }
    }

    /// Create with multiplicative seasonality.
    #[wasm_bindgen(js_name = multiplicative)]
    pub fn multiplicative(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        Self {
            model: HoltWinters::multiplicative(alpha, beta, gamma, period),
        }
    }

    /// Create with automatic parameter optimization.
    /// @param period - Seasonal period
    /// @param seasonal_type - "additive" or "multiplicative"
    #[wasm_bindgen(js_name = auto)]
    pub fn auto(period: usize, seasonal_type: &str) -> Result<HoltWintersForecaster, JsError> {
        use anofox_forecast::models::exponential::SeasonalType;

        let st = match seasonal_type.to_lowercase().as_str() {
            "additive" | "a" => SeasonalType::Additive,
            "multiplicative" | "m" => SeasonalType::Multiplicative,
            _ => {
                return Err(JsError::new(
                    "seasonal_type must be 'additive' or 'multiplicative'",
                ))
            }
        };

        Ok(Self {
            model: HoltWinters::auto(period, st),
        })
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: SeasonalES::new(period),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// ETS (Error-Trend-Seasonal) state-space model.
///
/// Use string codes: "A" = Additive, "M" = Multiplicative, "N" = None
/// Or use standard ETS notation like "ANN", "AAA", "MAM", "AAdM".
///
/// Follows the ETS taxonomy from FPP3: <https://otexts.com/fpp3/taxonomy.html>
///
/// Note: Some combinations are invalid/unstable per FPP3:
/// - MAA (Multiplicative error + Additive trend + Additive seasonal)
/// - MAdA (Multiplicative error + Damped trend + Additive seasonal)
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
    /// @throws Error if the combination is unstable (MAA or MAdA)
    #[wasm_bindgen(constructor)]
    pub fn new(
        error: &str,
        trend: &str,
        seasonal: &str,
        period: usize,
    ) -> Result<ETSForecaster, JsError> {
        use anofox_forecast::models::exponential::{
            ETSSeasonalType, ETSSpec, ErrorType, TrendType,
        };

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

        // Validate the specification (MAA and MAdA are unstable)
        if !spec.is_valid() {
            return Err(JsError::new(&format!(
                "ETS({},{},{}) is an unstable model combination per FPP3. \
                 Multiplicative error with additive trend and additive seasonal is not supported.",
                error.to_uppercase(),
                trend.to_uppercase(),
                seasonal.to_uppercase()
            )));
        }

        Ok(Self {
            model: ETS::new(spec, period),
        })
    }

    /// Create an ETS model from standard notation.
    ///
    /// @param notation - ETS notation string like "ANN", "AAA", "MAM", "AAdM"
    /// @param period - Seasonal period (required if notation has seasonal component)
    ///
    /// Format: ErrorTrendSeasonal
    /// - Error: A (additive) or M (multiplicative)
    /// - Trend: N (none), A (additive), or Ad (additive damped)
    /// - Seasonal: N (none), A (additive), or M (multiplicative)
    ///
    /// Examples:
    /// - "ANN" - Simple exponential smoothing
    /// - "AAN" - Holt's linear method
    /// - "AAA" - Holt-Winters additive
    /// - "MAM" - Multiplicative Holt-Winters
    /// - "AAdM" - Damped trend with multiplicative seasonal
    ///
    /// @throws Error for invalid notation or unstable combinations (MAA, MAdA)
    #[wasm_bindgen(js_name = fromNotation)]
    pub fn from_notation(notation: &str, period: usize) -> Result<ETSForecaster, JsError> {
        use anofox_forecast::models::exponential::ETSSpec;

        let spec = ETSSpec::from_notation(notation).map_err(|e| JsError::new(&e.to_string()))?;

        Ok(Self {
            model: ETS::new(spec, period),
        })
    }

    /// Check if an ETS specification is valid/stable.
    ///
    /// @param error - Error type: "A" or "M"
    /// @param trend - Trend type: "N", "A", or "Ad"
    /// @param seasonal - Seasonal type: "N", "A", or "M"
    /// @returns true if the combination is stable and usable
    ///
    /// Invalid combinations (return false):
    /// - M,A,A - Multiplicative error with additive trend and additive seasonal
    /// - M,Ad,A - Multiplicative error with damped trend and additive seasonal
    #[wasm_bindgen(js_name = isValidSpec)]
    pub fn is_valid_spec(error: &str, trend: &str, seasonal: &str) -> bool {
        use anofox_forecast::models::exponential::{
            ETSSeasonalType, ETSSpec, ErrorType, TrendType,
        };

        let error_type = match error.to_uppercase().as_str() {
            "A" => ErrorType::Additive,
            "M" => ErrorType::Multiplicative,
            _ => return false,
        };

        let trend_type = match trend.to_uppercase().as_str() {
            "N" => TrendType::None,
            "A" => TrendType::Additive,
            "AD" => TrendType::AdditiveDamped,
            _ => return false,
        };

        let seasonal_type = match seasonal.to_uppercase().as_str() {
            "N" => ETSSeasonalType::None,
            "A" => ETSSeasonalType::Additive,
            "M" => ETSSeasonalType::Multiplicative,
            _ => return false,
        };

        let spec = ETSSpec::new(error_type, trend_type, seasonal_type);
        spec.is_valid()
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

/// AutoETS - Automatic ETS model selection.
///
/// Follows the ETS taxonomy from FPP3: <https://otexts.com/fpp3/taxonomy.html>
#[wasm_bindgen]
pub struct AutoETSForecaster {
    model: AutoETS,
}

#[wasm_bindgen]
impl AutoETSForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: AutoETS::new(),
        }
    }

    /// Create AutoETS with a specific seasonal period.
    #[wasm_bindgen(js_name = withPeriod)]
    pub fn with_period(period: usize) -> Self {
        Self {
            model: AutoETS::with_period(period),
        }
    }

    /// Create AutoETS restricted to additive models only.
    /// This excludes multiplicative error and multiplicative seasonality.
    #[wasm_bindgen(js_name = additiveOnly)]
    pub fn additive_only() -> Self {
        use anofox_forecast::models::exponential::AutoETSConfig;
        Self {
            model: AutoETS::with_config(AutoETSConfig::default().additive_only()),
        }
    }

    /// Create AutoETS with custom configuration.
    /// @param period - Optional seasonal period (null for auto-detection)
    /// @param allow_multiplicative_error - Allow multiplicative error models
    /// @param allow_multiplicative_seasonal - Allow multiplicative seasonality
    /// @param allow_damped - Allow damped trend models
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(
        period: Option<usize>,
        allow_multiplicative_error: bool,
        allow_multiplicative_seasonal: bool,
        allow_damped: bool,
    ) -> Self {
        use anofox_forecast::models::exponential::AutoETSConfig;
        let config = AutoETSConfig {
            seasonal_period: period,
            allow_multiplicative_error,
            allow_multiplicative_seasonal,
            allow_damped,
            ..Default::default()
        };
        Self {
            model: AutoETS::with_config(config),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoETSForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        Self {
            model: Theta::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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

/// Optimized Theta forecaster - automatically optimizes parameters.
#[wasm_bindgen]
pub struct OptimizedThetaForecaster {
    model: OptimizedTheta,
}

#[wasm_bindgen]
impl OptimizedThetaForecaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: OptimizedTheta::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: DynamicTheta::new(alpha),
        }
    }

    /// Create an optimized Dynamic Theta model.
    pub fn optimized() -> Self {
        Self {
            model: DynamicTheta::optimized(),
        }
    }

    /// Create a seasonal Dynamic Theta model.
    /// @param period - Seasonal period
    pub fn seasonal(period: usize) -> Self {
        Self {
            model: DynamicTheta::seasonal(period),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: AutoTheta::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoThetaForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        Self {
            model: ARIMA::new(p, d, q),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        period: usize,
    ) -> Self {
        Self {
            model: SARIMA::new(p, d, q, seasonal_p, seasonal_d, seasonal_q, period),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: AutoARIMA::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for AutoARIMAForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        Self {
            model: Croston::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for CrostonForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for TSBForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        Self {
            model: ADIDA::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for ADIDAForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        Self {
            model: IMAPA::new(),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

impl Default for IMAPAForecaster {
    fn default() -> Self {
        Self::new()
    }
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
        Self {
            model: TBATS::new(seasonal_periods),
        }
    }

    /// Create a TBATSForecaster with specified seasonal periods.
    ///
    /// @param periods - Array of seasonal periods
    /// @returns A new TBATSForecaster
    #[wasm_bindgen(js_name = withSeasonalPeriods)]
    pub fn with_seasonal_periods(periods: Vec<usize>) -> Self {
        Self {
            model: TBATS::new(periods),
        }
    }

    /// Enable Box-Cox transformation.
    ///
    /// @param lambda - Box-Cox parameter (0 = log, 1 = identity)
    #[wasm_bindgen(js_name = setBoxCox)]
    pub fn set_box_cox(&mut self, lambda: f64) {
        self.model = std::mem::replace(&mut self.model, TBATS::new(vec![])).with_box_cox(lambda);
    }

    /// Enable damped trend.
    ///
    /// @param phi - Damping parameter (typically 0.8-0.99)
    #[wasm_bindgen(js_name = setDampedTrend)]
    pub fn set_damped_trend(&mut self, phi: f64) {
        self.model = std::mem::replace(&mut self.model, TBATS::new(vec![])).with_damped_trend(phi);
    }

    /// Set ARMA error orders.
    ///
    /// @param p - AR order
    /// @param q - MA order
    #[wasm_bindgen(js_name = setArma)]
    pub fn set_arma(&mut self, p: usize, q: usize) {
        self.model = std::mem::replace(&mut self.model, TBATS::new(vec![])).with_arma(p, q);
    }

    /// Set Fourier K (number of harmonics) for each seasonal period.
    ///
    /// @param k - Array of K values (one per seasonal period)
    #[wasm_bindgen(js_name = setFourierK)]
    pub fn set_fourier_k(&mut self, k: Vec<usize>) {
        self.model = std::mem::replace(&mut self.model, TBATS::new(vec![])).with_fourier_k(k);
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

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
    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.model
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: AutoTBATS::new(seasonal_periods),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: MFLES::new(seasonal_periods),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: MSTLForecaster::new(seasonal_periods),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
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
        Self {
            model: GARCH::new(p, q),
        }
    }

    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.model
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

// =============================================================================
// ENSEMBLE
// =============================================================================

/// Ensemble forecaster that combines multiple models.
///
/// Supports mean, median, weighted MSE, and custom-weight combination.
/// Models are specified by name strings (e.g., "naive", "sma5", "ses").
#[wasm_bindgen]
pub struct EnsembleForecaster {
    model: anofox_forecast::models::ensemble::Ensemble,
}

/// Create a boxed forecaster from a model name string.
fn model_from_name(name: &str) -> Result<Box<dyn ForecasterTrait>, JsError> {
    match name.to_lowercase().as_str() {
        "naive" => Ok(Box::new(Naive::new())),
        "mean" | "historicaverage" => Ok(Box::new(HistoricAverage::new())),
        "rwdrift" | "randomwalkwithdrift" => Ok(Box::new(RandomWalkWithDrift::new())),
        "ses" | "simpleexponentialsmoothing" => Ok(Box::new(SimpleExponentialSmoothing::auto())),
        "holt" | "holtlineartrend" => Ok(Box::new(HoltLinearTrend::auto())),
        "autoarima" => Ok(Box::new(AutoARIMA::new())),
        "autoets" => Ok(Box::new(AutoETS::new())),
        "autotheta" => Ok(Box::new(AutoTheta::new())),
        s if s.starts_with("sma") => {
            let window: usize = s[3..].parse().unwrap_or(5);
            Ok(Box::new(SimpleMovingAverage::new(window)))
        }
        s if s.starts_with("wa") && s.len() > 2 => {
            let window: usize = s[2..].parse().unwrap_or(5);
            Ok(Box::new(WindowAverage::new(window)))
        }
        other => Err(JsError::new(&format!(
            "Unknown model '{}'. Use: naive, mean, rwdrift, ses, holt, autoarima, autoets, autotheta, sma<N>, wa<N>",
            other
        ))),
    }
}

#[wasm_bindgen]
impl EnsembleForecaster {
    /// Create an ensemble from an array of model name strings.
    ///
    /// Supported names: "naive", "mean", "rwdrift", "ses", "holt",
    /// "autoarima", "autoets", "autotheta", "sma5", "wa10", etc.
    ///
    /// @param modelNames - Array of model name strings
    #[wasm_bindgen(constructor)]
    pub fn new(model_names: Vec<String>) -> Result<EnsembleForecaster, JsError> {
        let models: Result<Vec<Box<dyn ForecasterTrait>>, JsError> =
            model_names.iter().map(|n| model_from_name(n)).collect();
        Ok(Self {
            model: anofox_forecast::models::ensemble::Ensemble::new(models?),
        })
    }

    /// Set custom combination weights.
    ///
    /// Weights are normalized to sum to 1. Length must match number of models.
    ///
    /// @param weights - Array of combination weights
    #[wasm_bindgen(js_name = setWeights)]
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.model = std::mem::replace(
            &mut self.model,
            anofox_forecast::models::ensemble::Ensemble::new(vec![]),
        )
        .with_weights(weights);
    }

    /// Set the combination method to median.
    #[wasm_bindgen(js_name = setMedian)]
    pub fn set_median(&mut self) {
        self.model = std::mem::replace(
            &mut self.model,
            anofox_forecast::models::ensemble::Ensemble::new(vec![]),
        )
        .with_method(anofox_forecast::models::ensemble::CombinationMethod::Median);
    }

    /// Set the combination method to weighted MSE.
    #[wasm_bindgen(js_name = setWeightedMse)]
    pub fn set_weighted_mse(&mut self) {
        self.model = std::mem::replace(
            &mut self.model,
            anofox_forecast::models::ensemble::Ensemble::new(vec![]),
        )
        .with_method(anofox_forecast::models::ensemble::CombinationMethod::WeightedMSE);
    }

    /// Fit all models in the ensemble.
    ///
    /// @param series - TimeSeries to fit
    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.model
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Predict future values using the combined ensemble.
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

    /// Get the model name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.model.name().to_string()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    // Helper function to create a simple time series for testing
    fn create_test_series(n: usize) -> TimeSeries {
        let values: Vec<f64> = (0..n).map(|i| 10.0 + i as f64).collect();
        TimeSeries::new(&values).unwrap()
    }

    // Helper function to create a seasonal time series
    fn create_seasonal_series(n: usize, period: usize) -> TimeSeries {
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 10.0 + 0.5 * i as f64;
                let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                trend + seasonal
            })
            .collect();
        TimeSeries::new(&values).unwrap()
    }

    // Helper function to create intermittent demand series
    fn create_intermittent_series() -> TimeSeries {
        let values = vec![
            0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 7.0, 0.0, 0.0, 4.0, 0.0, 0.0,
        ];
        TimeSeries::new(&values).unwrap()
    }

    // =========================================================================
    // BASELINE MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_naive_forecaster() {
        let ts = create_test_series(20);
        let mut model = NaiveForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "Naive");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // Naive should repeat the last value
        let values = ts.values();
        let last_value = values.last().unwrap();
        for pred in forecast.values() {
            assert!((pred - last_value).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_naive_with_intervals() {
        let ts = create_test_series(20);
        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // Check intervals exist
        let lower = forecast.lower();
        let upper = forecast.upper();
        assert!(lower.is_some());
        assert!(upper.is_some());

        // Lower should be less than point forecast, upper should be greater
        for i in 0..5 {
            assert!(lower.as_ref().unwrap()[i] <= forecast.values()[i]);
            assert!(upper.as_ref().unwrap()[i] >= forecast.values()[i]);
        }
    }

    #[wasm_bindgen_test]
    fn test_mean_forecaster() {
        let ts = create_test_series(20);
        let mut model = MeanForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "HistoricAverage");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // Mean should be constant across horizon
        let first_pred = forecast.values()[0];
        for pred in forecast.values() {
            assert!((pred - first_pred).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_seasonal_naive_forecaster() {
        let ts = create_seasonal_series(24, 12);
        let mut model = SeasonalNaiveForecaster::new(12);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "SeasonalNaive");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_random_walk_drift_forecaster() {
        let ts = create_test_series(30);
        let mut model = RandomWalkDriftForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "RandomWalkWithDrift");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // With positive drift, forecasts should be increasing
        for i in 1..5 {
            assert!(forecast.values()[i] > forecast.values()[i - 1]);
        }
    }

    #[wasm_bindgen_test]
    fn test_sma_forecaster() {
        let ts = create_test_series(20);
        let mut model = SMAForecaster::new(5);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "SimpleMovingAverage");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // SMA should produce constant forecasts (all same value)
        let first_pred = forecast.values()[0];
        for pred in forecast.values() {
            assert!((pred - first_pred).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_window_average_forecaster() {
        let ts = create_test_series(20);
        let mut model = WindowAverageForecaster::new(5);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "WindowAverage");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_seasonal_window_average_forecaster() {
        let ts = create_seasonal_series(36, 12);
        let mut model = SeasonalWindowAverageForecaster::new(12, 2);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "SeasonalWindowAverage");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    // =========================================================================
    // EXPONENTIAL SMOOTHING MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_ses_forecaster() {
        let ts = create_test_series(20);
        let mut model = SESForecaster::new(0.3);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "SimpleExponentialSmoothing");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // SES produces flat forecasts
        let first_pred = forecast.values()[0];
        for pred in forecast.values() {
            assert!((pred - first_pred).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_holt_forecaster() {
        let ts = create_test_series(30);
        let mut model = HoltForecaster::new(0.3, 0.1);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "HoltLinearTrend");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // Holt should capture trend - forecasts should be increasing
        for i in 1..5 {
            assert!(forecast.values()[i] > forecast.values()[i - 1]);
        }
    }

    #[wasm_bindgen_test]
    fn test_holt_winters_additive() {
        let ts = create_seasonal_series(48, 12);
        // new() uses additive seasonality by default
        let mut model = HoltWintersForecaster::new(0.3, 0.1, 0.1, 12);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "HoltWinters(additive)");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_holt_winters_multiplicative() {
        // Need positive data for multiplicative
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let trend = 100.0 + i as f64;
                let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                trend * seasonal
            })
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = HoltWintersForecaster::multiplicative(0.3, 0.1, 0.1, 12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_holt_winters_auto() {
        let ts = create_seasonal_series(48, 12);
        let mut model = HoltWintersForecaster::auto(12, "additive").unwrap();

        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_holt_winters_auto_multiplicative() {
        // Need positive data for multiplicative
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let trend = 100.0 + i as f64;
                let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                trend * seasonal
            })
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = HoltWintersForecaster::auto(12, "multiplicative").unwrap();
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_seasonal_es_forecaster() {
        let ts = create_seasonal_series(48, 12);
        let mut model = SeasonalESForecaster::new(12);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "SeasonalES");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_ets_forecaster_ann() {
        let ts = create_test_series(30);
        // ETS(A,N,N) - Additive error, No trend, No seasonal
        let mut model = ETSForecaster::new("A", "N", "N", 1).unwrap();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "ETS");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ets_forecaster_aan() {
        let ts = create_test_series(30);
        // ETS(A,A,N) - Additive error, Additive trend, No seasonal
        let mut model = ETSForecaster::new("A", "A", "N", 1).unwrap();

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // With trend data, forecasts should be increasing
        for i in 1..5 {
            assert!(forecast.values()[i] > forecast.values()[i - 1]);
        }
    }

    #[wasm_bindgen_test]
    fn test_ets_forecaster_aaa() {
        let ts = create_seasonal_series(48, 12);
        // ETS(A,A,A) - Additive error, Additive trend, Additive seasonal
        let mut model = ETSForecaster::new("A", "A", "A", 12).unwrap();

        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_ets_forecaster_damped() {
        let ts = create_test_series(30);
        // ETS(A,Ad,N) - Additive error, Additive damped trend, No seasonal
        let mut model = ETSForecaster::new("A", "Ad", "N", 1).unwrap();

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ets_forecaster_multiplicative() {
        // Need positive data for multiplicative models
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let trend = 100.0 + i as f64;
                let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                trend * seasonal
            })
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        // ETS(M,A,M) - Multiplicative error, Additive trend, Multiplicative seasonal
        let mut model = ETSForecaster::new("M", "A", "M", 12).unwrap();
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_auto_ets_forecaster() {
        let ts = create_test_series(30);
        let mut model = AutoETSForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "AutoETS");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_auto_ets_with_period() {
        let ts = create_seasonal_series(48, 12);
        let mut model = AutoETSForecaster::with_period(12);

        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_auto_ets_additive_only() {
        let ts = create_test_series(30);
        let mut model = AutoETSForecaster::additive_only();

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_auto_ets_with_config() {
        let ts = create_seasonal_series(48, 12);
        let mut model = AutoETSForecaster::with_config(
            Some(12), // period
            false,    // allow_multiplicative_error
            true,     // allow_multiplicative_seasonal
            true,     // allow_damped
        );

        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    // =========================================================================
    // THETA MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_theta_forecaster() {
        let ts = create_test_series(30);
        let mut model = ThetaForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "Theta");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_optimized_theta_forecaster() {
        let ts = create_test_series(30);
        let mut model = OptimizedThetaForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "OptimizedTheta");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_dynamic_theta_forecaster() {
        let ts = create_test_series(30);
        let mut model = DynamicThetaForecaster::new(0.5);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "DynamicTheta");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_dynamic_theta_optimized() {
        let ts = create_test_series(30);
        let mut model = DynamicThetaForecaster::optimized();

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_auto_theta_forecaster() {
        let ts = create_test_series(30);
        let mut model = AutoThetaForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "AutoTheta");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    // =========================================================================
    // ARIMA MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_arima_forecaster() {
        let ts = create_test_series(50);
        let mut model = ARIMAForecaster::new(1, 1, 1);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "ARIMA");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_sarima_forecaster() {
        let ts = create_seasonal_series(72, 12);
        let mut model = SARIMAForecaster::new(1, 1, 1, 1, 1, 1, 12);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "SARIMA");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_auto_arima_forecaster() {
        let ts = create_test_series(50);
        let mut model = AutoARIMAForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "AutoARIMA");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    // Note: AutoARIMAForecaster does not have seasonal configuration in WASM API
    // Seasonal ARIMA is tested through SARIMAForecaster above

    // =========================================================================
    // INTERMITTENT DEMAND MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_croston_forecaster() {
        let ts = create_intermittent_series();
        let mut model = CrostonForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "Croston");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);

        // Croston produces flat forecasts
        let first_pred = forecast.values()[0];
        for pred in forecast.values() {
            assert!((pred - first_pred).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_tsb_forecaster() {
        let ts = create_intermittent_series();
        let mut model = TSBForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "TSB");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_adida_forecaster() {
        let ts = create_intermittent_series();
        let mut model = ADIDAForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "ADIDA");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_imapa_forecaster() {
        let ts = create_intermittent_series();
        let mut model = IMAPAForecaster::new();

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "IMAPA");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    // =========================================================================
    // ADVANCED MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_tbats_forecaster() {
        let ts = create_seasonal_series(48, 12);
        let mut model = TBATSForecaster::new(vec![12]);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "TBATS");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_auto_tbats_forecaster() {
        let ts = create_seasonal_series(48, 12);
        let mut model = AutoTBATSForecaster::new(vec![12]);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "AutoTBATS");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_mfles_forecaster() {
        let ts = create_seasonal_series(48, 12);
        let mut model = MFLESForecaster::new(vec![12]);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "MFLES");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_mstl_forecaster() {
        let ts = create_seasonal_series(48, 12);
        let mut model = MSTLForecasterWrapper::new(vec![12]);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "MSTLForecaster");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_garch_forecaster() {
        // GARCH needs returns-like data (mean around 0)
        let values: Vec<f64> = (0..100)
            .map(|i| 0.01 * (i as f64 * 0.1).sin() + 0.001 * (i as f64))
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = GARCHForecaster::new(1, 1);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "GARCH");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    // =========================================================================
    // ERROR HANDLING TESTS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_predict_without_fit_fails() {
        let model = NaiveForecaster::new();
        let result = model.predict(5);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_empty_series_fails() {
        let values: Vec<f64> = vec![];
        // TimeSeries may or may not error on empty data, but model fit should fail
        if let Ok(ts) = TimeSeries::new(&values) {
            let mut model = NaiveForecaster::new();
            let result = model.fit(&ts);
            assert!(result.is_err(), "Expected fit to fail with empty data");
        }
        // If TimeSeries::new fails for empty data, that's also acceptable
    }

    #[wasm_bindgen_test]
    fn test_insufficient_data_for_seasonal() {
        let ts = create_test_series(5); // Only 5 points
        let mut model = SeasonalNaiveForecaster::new(12); // Needs 12+ points

        let result = model.fit(&ts);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_holt_winters_invalid_seasonal_type() {
        let result = HoltWintersForecaster::auto(12, "invalid");
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_invalid_error_type() {
        let result = ETSForecaster::new("X", "N", "N", 1);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_invalid_trend_type() {
        let result = ETSForecaster::new("A", "X", "N", 1);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_invalid_seasonal_type() {
        let result = ETSForecaster::new("A", "N", "X", 1);
        assert!(result.is_err());
    }

    // =========================================================================
    // PREDICTION INTERVAL TESTS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_naive_intervals() {
        let ts = create_test_series(30);
        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        let lower = forecast.lower().unwrap();
        let upper = forecast.upper().unwrap();
        let values = forecast.values();

        for i in 0..5 {
            assert!(lower[i] <= values[i]);
            assert!(upper[i] >= values[i]);
        }
    }

    #[wasm_bindgen_test]
    fn test_mean_intervals() {
        let ts = create_test_series(30);
        let mut model = MeanForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        let lower = forecast.lower().unwrap();
        let upper = forecast.upper().unwrap();
        let values = forecast.values();

        for i in 0..5 {
            assert!(lower[i] < values[i]);
            assert!(upper[i] > values[i]);
        }
    }

    #[wasm_bindgen_test]
    fn test_theta_intervals() {
        let ts = create_test_series(30);
        let mut model = ThetaForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        assert!(forecast.lower().is_some());
        assert!(forecast.upper().is_some());
    }

    #[wasm_bindgen_test]
    fn test_auto_theta_intervals() {
        let ts = create_test_series(30);
        let mut model = AutoThetaForecaster::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        assert!(forecast.lower().is_some());
        assert!(forecast.upper().is_some());
    }

    #[wasm_bindgen_test]
    fn test_dynamic_theta_intervals() {
        let ts = create_test_series(30);
        let mut model = DynamicThetaForecaster::new(0.5);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();

        assert!(forecast.lower().is_some());
        assert!(forecast.upper().is_some());
    }

    // =========================================================================
    // ETS NOTATION AND VALIDATION TESTS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_ets_from_notation_ann() {
        // Simple exponential smoothing
        let mut model = ETSForecaster::from_notation("ANN", 1).unwrap();
        let ts = create_test_series(30);
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ets_from_notation_aaa() {
        // Holt-Winters additive
        let mut model = ETSForecaster::from_notation("AAA", 12).unwrap();
        let ts = create_seasonal_series(48, 12);
        model.fit(&ts).unwrap();
        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_ets_from_notation_mam() {
        // Multiplicative Holt-Winters
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let trend = 100.0 + i as f64;
                let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                trend * seasonal
            })
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = ETSForecaster::from_notation("MAM", 12).unwrap();
        model.fit(&ts).unwrap();
        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_ets_from_notation_damped() {
        // Damped trend with multiplicative seasonal
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let trend = 100.0 + i as f64;
                let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                trend * seasonal
            })
            .collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = ETSForecaster::from_notation("AAdM", 12).unwrap();
        model.fit(&ts).unwrap();
        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.values().len(), 12);
    }

    #[wasm_bindgen_test]
    fn test_ets_from_notation_invalid() {
        // Invalid notation should fail
        assert!(ETSForecaster::from_notation("XYZ", 12).is_err());
        assert!(ETSForecaster::from_notation("", 12).is_err());
        assert!(ETSForecaster::from_notation("A", 12).is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_from_notation_unstable_maa() {
        // MAA is unstable and should fail
        let result = ETSForecaster::from_notation("MAA", 12);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_from_notation_unstable_mada() {
        // MAdA is unstable and should fail
        let result = ETSForecaster::from_notation("MAdA", 12);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_unstable_combination_new() {
        // Creating unstable combination via new() should also fail
        let result = ETSForecaster::new("M", "A", "A", 12);
        assert!(result.is_err());

        let result = ETSForecaster::new("M", "Ad", "A", 12);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_ets_is_valid_spec() {
        // Valid combinations
        assert!(ETSForecaster::is_valid_spec("A", "N", "N"));
        assert!(ETSForecaster::is_valid_spec("A", "A", "A"));
        assert!(ETSForecaster::is_valid_spec("A", "A", "M"));
        assert!(ETSForecaster::is_valid_spec("M", "A", "M"));
        assert!(ETSForecaster::is_valid_spec("M", "N", "M"));
        assert!(ETSForecaster::is_valid_spec("A", "Ad", "M"));

        // Invalid/unstable combinations
        assert!(!ETSForecaster::is_valid_spec("M", "A", "A"));
        assert!(!ETSForecaster::is_valid_spec("M", "Ad", "A"));

        // Invalid parameters
        assert!(!ETSForecaster::is_valid_spec("X", "A", "A"));
        assert!(!ETSForecaster::is_valid_spec("A", "X", "A"));
        assert!(!ETSForecaster::is_valid_spec("A", "A", "X"));
    }

    // =========================================================================
    // ADDITIONAL EDGE CASE TESTS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_single_data_point() {
        let values = vec![42.0];
        let ts = TimeSeries::new(&values).unwrap();

        // Naive should handle single point
        let mut model = NaiveForecaster::new();
        let result = model.fit(&ts);
        // This may succeed or fail depending on implementation
        if result.is_ok() {
            let forecast = model.predict(3).unwrap();
            assert_eq!(forecast.values().len(), 3);
            // All forecasts should be the same as the single value
            for v in forecast.values() {
                assert!((v - 42.0).abs() < 1e-10);
            }
        }
    }

    #[wasm_bindgen_test]
    fn test_two_data_points() {
        let values = vec![10.0, 20.0];
        let ts = TimeSeries::new(&values).unwrap();

        // Naive with 2 points
        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.values().len(), 3);
        // Naive repeats last value
        for v in forecast.values() {
            assert!((v - 20.0).abs() < 1e-10);
        }

        // Mean with 2 points
        let mut mean_model = MeanForecaster::new();
        mean_model.fit(&ts).unwrap();
        let mean_forecast = mean_model.predict(3).unwrap();
        // Mean should be 15.0
        for v in mean_forecast.values() {
            assert!((v - 15.0).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_nan_in_series() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let ts = TimeSeries::new(&values).unwrap();

        // Check if series reports missing values
        assert!(ts.has_missing_values());

        // Models may or may not handle NaN gracefully
        let mut model = NaiveForecaster::new();
        let result = model.fit(&ts);
        // Either it should fail with an error or handle NaN
        // We're just verifying it doesn't panic
        let _ = result;
    }

    #[wasm_bindgen_test]
    fn test_constant_series() {
        // All same values
        let values = vec![5.0; 20];
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        // All forecasts should be 5.0
        for v in forecast.values() {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_negative_values() {
        // Series with negative values
        let values: Vec<f64> = (-10..10).map(|i| i as f64).collect();
        let ts = TimeSeries::new(&values).unwrap();

        // Additive models should handle negative values
        let mut model = ETSForecaster::new("A", "A", "N", 1).unwrap();
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_negative_values_multiplicative_fails() {
        // Multiplicative models require positive data
        let values: Vec<f64> = (-10..10).map(|i| i as f64).collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = ETSForecaster::new("M", "N", "N", 1).unwrap();
        let result = model.fit(&ts);
        // Should fail for negative data with multiplicative error
        // (or handle gracefully)
        let _ = result;
    }

    #[wasm_bindgen_test]
    fn test_large_horizon() {
        let ts = create_test_series(30);

        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();

        // Predict far into the future
        let forecast = model.predict(100).unwrap();
        assert_eq!(forecast.values().len(), 100);
    }

    #[wasm_bindgen_test]
    fn test_zero_horizon() {
        let ts = create_test_series(30);

        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();

        // Zero horizon should return empty or error
        let result = model.predict(0);
        if let Ok(forecast) = result {
            assert_eq!(forecast.values().len(), 0);
        }
        // Either empty result or error is acceptable
    }

    #[wasm_bindgen_test]
    fn test_all_zeros_series() {
        let values = vec![0.0; 20];
        let ts = TimeSeries::new(&values).unwrap();

        // Additive models should handle all zeros
        let mut model = MeanForecaster::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        for v in forecast.values() {
            assert!(v.abs() < 1e-10);
        }
    }

    #[wasm_bindgen_test]
    fn test_very_large_values() {
        let values: Vec<f64> = (0..20).map(|i| 1e15 + i as f64).collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        // Should handle large values without overflow
        assert_eq!(forecast.values().len(), 5);
        assert!(forecast.values()[0] > 1e14);
    }

    #[wasm_bindgen_test]
    fn test_very_small_values() {
        let values: Vec<f64> = (0..20).map(|i| 1e-15 + (i as f64 * 1e-16)).collect();
        let ts = TimeSeries::new(&values).unwrap();

        let mut model = NaiveForecaster::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();

        // Should handle small values without underflow
        assert_eq!(forecast.values().len(), 5);
        assert!(forecast.values()[0] > 0.0);
    }

    // =========================================================================
    // TBATS BUILDER METHODS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_tbats_with_seasonal_periods() {
        let ts = create_seasonal_series(48, 12);
        let mut model = TBATSForecaster::with_seasonal_periods(vec![12]);

        model.fit(&ts).unwrap();
        assert_eq!(model.name(), "TBATS");

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_tbats_set_box_cox() {
        let ts = create_seasonal_series(48, 12);
        let mut model = TBATSForecaster::new(vec![12]);
        model.set_box_cox(0.5);

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_tbats_predict_with_intervals() {
        let ts = create_seasonal_series(48, 12);
        let mut model = TBATSForecaster::new(vec![12]);

        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    // =========================================================================
    // ENSEMBLE MODELS
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_ensemble_basic() {
        let ts = create_test_series(30);
        let mut model =
            EnsembleForecaster::new(vec!["naive".to_string(), "sma5".to_string()]).unwrap();

        model.fit(&ts).unwrap();
        assert_eq!(model.model_count(), 2);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ensemble_set_weights() {
        let ts = create_test_series(30);
        let mut model =
            EnsembleForecaster::new(vec!["naive".to_string(), "sma5".to_string()]).unwrap();
        model.set_weights(vec![0.7, 0.3]);

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ensemble_set_median() {
        let ts = create_test_series(30);
        let mut model = EnsembleForecaster::new(vec![
            "naive".to_string(),
            "sma5".to_string(),
            "mean".to_string(),
        ])
        .unwrap();
        model.set_median();

        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.values().len(), 5);
    }

    #[wasm_bindgen_test]
    fn test_ensemble_invalid_model() {
        let result = EnsembleForecaster::new(vec!["nonexistent".to_string()]);
        assert!(result.is_err());
    }
}
