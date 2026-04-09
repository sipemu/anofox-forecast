//! Regression forecaster bindings for JavaScript.
//!
//! Exposes `RegressionForecaster` with configurable features (trend, lags,
//! Fourier, rolling statistics, changepoints) and the full set of regression
//! backends (OLS, Ridge, ElasticNet, Quantile, WLS, RLS, Tweedie, Poisson,
//! BLS, Dynamic).

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use anofox_forecast::models::regression::{
    ChangepointEncoding as InnerChangepointEncoding, ChangepointFeature as InnerChangepointFeature,
    RegressionBackend as InnerRegressionBackend, RegressionFeatures as InnerRegressionFeatures,
    RegressionForecaster as InnerRegressionForecaster, RollingStatKind as InnerRollingStatKind,
    SeasonalSpec as InnerSeasonalSpec, TrendType as InnerTrendType,
    WeightStrategy as InnerWeightStrategy,
};
use anofox_forecast::models::Forecaster as ForecasterTrait;

use crate::time_series::{Forecast, TimeSeries};

// ---------------------------------------------------------------------------
// Small parse helpers
// ---------------------------------------------------------------------------

fn parse_rolling_kind(kind: &str, alpha: Option<f64>) -> Result<InnerRollingStatKind, JsError> {
    let a = alpha.unwrap_or(0.3);
    match kind {
        "mean" | "Mean" => Ok(InnerRollingStatKind::Mean),
        "std" | "Std" => Ok(InnerRollingStatKind::Std),
        "var" | "Var" => Ok(InnerRollingStatKind::Var),
        "min" | "Min" => Ok(InnerRollingStatKind::Min),
        "max" | "Max" => Ok(InnerRollingStatKind::Max),
        "median" | "Median" => Ok(InnerRollingStatKind::Median),
        "sum" | "Sum" => Ok(InnerRollingStatKind::Sum),
        "ewmMean" | "EwmMean" | "ewm_mean" => Ok(InnerRollingStatKind::EwmMean { alpha: a }),
        "ewmStd" | "EwmStd" | "ewm_std" => Ok(InnerRollingStatKind::EwmStd { alpha: a }),
        other => Err(JsError::new(&format!(
            "Unknown rolling kind '{}'. Use: mean, std, var, min, max, median, sum, ewmMean, ewmStd",
            other
        ))),
    }
}

fn parse_trend_type(t: &str) -> Result<InnerTrendType, JsError> {
    match t {
        "linear" | "Linear" => Ok(InnerTrendType::Linear),
        "quadratic" | "Quadratic" => Ok(InnerTrendType::Quadratic),
        "cubic" | "Cubic" => Ok(InnerTrendType::Cubic),
        "exponential" | "Exponential" => Ok(InnerTrendType::Exponential),
        "theilSen" | "TheilSen" | "theil_sen" => Ok(InnerTrendType::TheilSen),
        other => Err(JsError::new(&format!(
            "Unknown trend type '{}'. Use: linear, quadratic, cubic, exponential, theilSen",
            other
        ))),
    }
}

fn parse_changepoint_encoding(enc: &str) -> Result<InnerChangepointEncoding, JsError> {
    match enc {
        "stepFunctions" | "StepFunctions" | "step" => Ok(InnerChangepointEncoding::StepFunctions),
        "regimeIndex" | "RegimeIndex" | "regime" => Ok(InnerChangepointEncoding::RegimeIndex),
        "cumulativeCount" | "CumulativeCount" | "count" => {
            Ok(InnerChangepointEncoding::CumulativeCount)
        }
        other => Err(JsError::new(&format!(
            "Unknown changepoint encoding '{}'. Use: stepFunctions, regimeIndex, cumulativeCount",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// RegressionFeatures builder wrapper
// ---------------------------------------------------------------------------

/// Feature configuration for [`RegressionForecaster`].
///
/// All methods mutate in place and return `void` — chain via multiple calls.
///
/// ```javascript
/// const features = new RegressionFeatures();
/// features.noTrend();
/// features.lags(3);
/// features.withRollingMean(7);
/// features.withEwmMean(20, 0.3);
/// const model = new RegressionForecaster(features);
/// ```
#[wasm_bindgen]
pub struct RegressionFeatures {
    inner: InnerRegressionFeatures,
}

impl Default for RegressionFeatures {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl RegressionFeatures {
    /// Create default features: trend enabled, no lags, exog enabled.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: InnerRegressionFeatures::new(),
        }
    }

    // ── Trend / lags / exog toggles ──────────────────────────────

    pub fn trend(&mut self) {
        self.inner = std::mem::take(&mut self.inner).trend();
    }

    #[wasm_bindgen(js_name = noTrend)]
    pub fn no_trend(&mut self) {
        self.inner = std::mem::take(&mut self.inner).no_trend();
    }

    pub fn lags(&mut self, max_lag: usize) {
        self.inner = std::mem::take(&mut self.inner).lags(max_lag);
    }

    #[wasm_bindgen(js_name = specificLags)]
    pub fn specific_lags(&mut self, lags: Vec<usize>) {
        self.inner = std::mem::take(&mut self.inner).specific_lags(&lags);
    }

    #[wasm_bindgen(js_name = autoLags)]
    pub fn auto_lags(&mut self, max_lag: usize) {
        self.inner = std::mem::take(&mut self.inner).auto_lags(max_lag);
    }

    pub fn exog(&mut self) {
        self.inner = std::mem::take(&mut self.inner).exog();
    }

    #[wasm_bindgen(js_name = noExog)]
    pub fn no_exog(&mut self) {
        self.inner = std::mem::take(&mut self.inner).no_exog();
    }

    // ── Trend / seasonal components ──────────────────────────────

    /// Add a trend component column.
    ///
    /// `trend` ∈ "linear" | "quadratic" | "cubic" | "exponential" | "theilSen"
    #[wasm_bindgen(js_name = withTrendComponent)]
    pub fn with_trend_component(&mut self, trend: &str) -> Result<(), JsError> {
        let t = parse_trend_type(trend)?;
        self.inner = std::mem::take(&mut self.inner).with_trend_component(t);
        Ok(())
    }

    /// Add Fourier seasonality features.
    pub fn fourier(&mut self, period: usize, order: usize) {
        self.inner = std::mem::take(&mut self.inner).fourier(period, order);
    }

    /// Add dummy seasonal encoding.
    #[wasm_bindgen(js_name = dummySeasonal)]
    pub fn dummy_seasonal(&mut self, period: usize) {
        self.inner = std::mem::take(&mut self.inner).dummy_seasonal(period);
    }

    // ── Changepoint features ─────────────────────────────────────

    /// Add step-function changepoint indicators at the given indices.
    #[wasm_bindgen(js_name = withChangepointSteps)]
    pub fn with_changepoint_steps(&mut self, indices: Vec<usize>) {
        self.inner = std::mem::take(&mut self.inner).with_changepoint_steps(indices);
    }

    /// Add changepoint features with a specific encoding.
    ///
    /// `encoding` ∈ "stepFunctions" | "regimeIndex" | "cumulativeCount"
    #[wasm_bindgen(js_name = withChangepoints)]
    pub fn with_changepoints(
        &mut self,
        indices: Vec<usize>,
        encoding: &str,
    ) -> Result<(), JsError> {
        let enc = parse_changepoint_encoding(encoding)?;
        let feat = Arc::new(InnerChangepointFeature::new(indices, enc));
        self.inner = std::mem::take(&mut self.inner).with_structural(feat);
        Ok(())
    }

    // ── Rolling / recursive features ─────────────────────────────

    /// Add a rolling statistic with default lag = 1.
    ///
    /// `kind` ∈ "mean" | "std" | "var" | "min" | "max" | "median" | "sum" |
    ///          "ewmMean" | "ewmStd" (EWM kinds require `alpha`)
    #[wasm_bindgen(js_name = withRolling)]
    pub fn with_rolling(
        &mut self,
        window: usize,
        kind: &str,
        alpha: Option<f64>,
    ) -> Result<(), JsError> {
        let k = parse_rolling_kind(kind, alpha)?;
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling(window, k)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    /// Add a rolling statistic with an explicit lag (≥ 1).
    #[wasm_bindgen(js_name = withRollingLagged)]
    pub fn with_rolling_lagged(
        &mut self,
        window: usize,
        lag: usize,
        kind: &str,
        alpha: Option<f64>,
    ) -> Result<(), JsError> {
        let k = parse_rolling_kind(kind, alpha)?;
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_lagged(window, lag, k)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingMean)]
    pub fn with_rolling_mean(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_mean(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingStd)]
    pub fn with_rolling_std(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_std(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingVar)]
    pub fn with_rolling_var(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_var(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingMin)]
    pub fn with_rolling_min(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_min(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingMax)]
    pub fn with_rolling_max(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_max(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingMedian)]
    pub fn with_rolling_median(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_median(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withRollingSum)]
    pub fn with_rolling_sum(&mut self, window: usize) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_rolling_sum(window)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withEwmMean)]
    pub fn with_ewm_mean(&mut self, window: usize, alpha: f64) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_ewm_mean(window, alpha)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = withEwmStd)]
    pub fn with_ewm_std(&mut self, window: usize, alpha: f64) -> Result<(), JsError> {
        self.inner = std::mem::take(&mut self.inner)
            .with_ewm_std(window, alpha)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    // ── Differencing ─────────────────────────────────────────────

    /// Apply regular differencing of order `d` before fitting.
    pub fn differencing(&mut self, d: usize) {
        self.inner = std::mem::take(&mut self.inner).differencing(d);
    }

    /// Apply seasonal differencing.
    #[wasm_bindgen(js_name = seasonalDifferencing)]
    pub fn seasonal_differencing(&mut self, d: usize, period: usize) {
        self.inner = std::mem::take(&mut self.inner).seasonal_differencing(d, period);
    }

    /// Apply fractional differencing of order `d` ∈ (0, 1).
    #[wasm_bindgen(js_name = fractionalDifferencing)]
    pub fn fractional_differencing(&mut self, d: f64) {
        self.inner = std::mem::take(&mut self.inner).fractional_differencing(d);
    }
}

// ---------------------------------------------------------------------------
// RegressionForecaster wrapper
// ---------------------------------------------------------------------------

/// Multi-backend regression forecaster.
///
/// Supported backends (constructor arg): `"ols"` (default), `"ridge"`,
/// `"elasticNet"`, `"quantile"`, `"wls"`, `"rls"`, `"tweedie"`, `"poisson"`,
/// `"bls"`.
///
/// ```javascript
/// const features = new RegressionFeatures();
/// features.noTrend();
/// features.lags(3);
/// features.withRollingMean(7);
///
/// const model = RegressionForecaster.ols(features);
/// model.fit(ts);
/// const forecast = model.predict(12);
/// ```
#[wasm_bindgen]
pub struct RegressionForecaster {
    inner: InnerRegressionForecaster,
}

#[wasm_bindgen]
impl RegressionForecaster {
    /// Create an OLS regression forecaster with the given features.
    pub fn ols(features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(InnerRegressionBackend::Ols, features.inner),
        }
    }

    /// Create a Ridge regression forecaster (L2 regularization).
    pub fn ridge(lambda: f64, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::Ridge { lambda },
                features.inner,
            ),
        }
    }

    /// Create an ElasticNet regression forecaster (L1+L2 regularization).
    #[wasm_bindgen(js_name = elasticNet)]
    pub fn elastic_net(lambda: f64, alpha: f64, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::ElasticNet { lambda, alpha },
                features.inner,
            ),
        }
    }

    /// Create a Quantile regression forecaster (τ ∈ (0, 1)).
    pub fn quantile(tau: f64, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::Quantile { tau },
                features.inner,
            ),
        }
    }

    /// Create a Weighted Least Squares forecaster with exponential-decay weights.
    #[wasm_bindgen(js_name = wlsDecay)]
    pub fn wls_decay(decay: f64, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::Wls {
                    strategy: InnerWeightStrategy::ExponentialDecay(decay),
                },
                features.inner,
            ),
        }
    }

    /// Create a Recursive Least Squares forecaster with a forgetting factor.
    pub fn rls(forgetting_factor: f64, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::Rls { forgetting_factor },
                features.inner,
            ),
        }
    }

    /// Create a Tweedie GLM forecaster.
    pub fn tweedie(var_power: f64, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::Tweedie {
                    var_power,
                    link_power: None,
                },
                features.inner,
            ),
        }
    }

    /// Create a Poisson regression forecaster (count data).
    pub fn poisson(features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(InnerRegressionBackend::Poisson, features.inner),
        }
    }

    /// Create a Box-constrained least-squares forecaster.
    ///
    /// Pass `None` for either bound to leave it unconstrained.
    pub fn bls(lower: Option<f64>, upper: Option<f64>, features: RegressionFeatures) -> Self {
        Self {
            inner: InnerRegressionForecaster::new(
                InnerRegressionBackend::Bls { lower, upper },
                features.inner,
            ),
        }
    }

    /// Fit the model on the supplied time series.
    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.inner
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Generate forecasts for `horizon` steps.
    pub fn predict(&self, horizon: usize) -> Result<Forecast, JsError> {
        self.inner
            .predict(horizon)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Forecast with confidence intervals.
    #[wasm_bindgen(js_name = predictWithIntervals)]
    pub fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast, JsError> {
        self.inner
            .predict_with_intervals(horizon, level)
            .map(Forecast::from_inner)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

// Silence unused import warnings for variants that are only accessed
// indirectly through builder methods.
#[allow(dead_code)]
type _Unused = (InnerSeasonalSpec,);
