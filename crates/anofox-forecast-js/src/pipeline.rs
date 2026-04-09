//! Pipeline + transform bindings for JavaScript.
//!
//! Exposes `Pipeline` and a builder API for chaining reversible transforms
//! (BoxCox, Difference, SeasonalDifference, Scale, Log) around any of the
//! built-in forecasting models. The pipeline itself implements the
//! `Forecaster` interface, so `fit` / `predict` work just like any other
//! model.

use wasm_bindgen::prelude::*;

use anofox_forecast::models::arima::{AutoARIMA, ARIMA, SARIMA};
use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift, SeasonalNaive};
use anofox_forecast::models::exponential::{
    AutoETS, HoltWinters, SeasonalType, SimpleExponentialSmoothing, ETS,
};
use anofox_forecast::models::theta::{AutoTheta, Theta};
use anofox_forecast::models::{BoxedForecaster, Forecaster as ForecasterTrait};
use anofox_forecast::transform::pipeline::{Pipeline as InnerPipeline, Transform};
use anofox_forecast::transform::transforms::{
    BoxCoxTransform, DifferenceTransform, LogTransform, ScaleMethod, ScaleTransform,
    SeasonalDifferenceTransform,
};

use crate::time_series::{Forecast, TimeSeries};

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Builder for a composable transform → model pipeline.
///
/// ```javascript
/// import { PipelineBuilder } from '@sipemu/anofox-forecast';
///
/// const builder = new PipelineBuilder();
/// builder.boxCoxAuto();        // Box-Cox with automatic lambda
/// builder.difference(1);        // first-order differencing
/// const model = builder.buildAutoArima();
///
/// model.fit(ts);
/// const forecast = model.predict(12);
/// ```
#[wasm_bindgen]
pub struct PipelineBuilder {
    transforms: Vec<Box<dyn Transform>>,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl PipelineBuilder {
    /// Create an empty pipeline builder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    // ── Box-Cox ──────────────────────────────────────────────────

    /// Append a Box-Cox transform with an automatically selected lambda.
    #[wasm_bindgen(js_name = boxCoxAuto)]
    pub fn box_cox_auto(&mut self) {
        self.transforms.push(Box::new(BoxCoxTransform::auto()));
    }

    /// Append a Box-Cox transform with a fixed lambda (e.g. 0.0 = log,
    /// 1.0 = identity).
    #[wasm_bindgen(js_name = boxCox)]
    pub fn box_cox(&mut self, lambda: f64) {
        self.transforms
            .push(Box::new(BoxCoxTransform::with_lambda(lambda)));
    }

    // ── Differencing ─────────────────────────────────────────────

    /// Append a differencing transform of order `d` (typically 1 or 2).
    pub fn difference(&mut self, d: usize) {
        self.transforms.push(Box::new(DifferenceTransform::new(d)));
    }

    /// Append a seasonal differencing transform with the given period.
    #[wasm_bindgen(js_name = seasonalDifference)]
    pub fn seasonal_difference(&mut self, period: usize) {
        self.transforms
            .push(Box::new(SeasonalDifferenceTransform::new(period)));
    }

    // ── Scaling ──────────────────────────────────────────────────

    /// Append a Z-score standardisation transform ((x − μ) / σ).
    pub fn standardize(&mut self) {
        self.transforms
            .push(Box::new(ScaleTransform::new(ScaleMethod::Standardize)));
    }

    /// Append a min-max normalisation transform ([0, 1]).
    pub fn normalize(&mut self) {
        self.transforms
            .push(Box::new(ScaleTransform::new(ScaleMethod::Normalize)));
    }

    /// Append a robust (median / IQR) scaling transform.
    #[wasm_bindgen(js_name = robustScale)]
    pub fn robust_scale(&mut self) {
        self.transforms
            .push(Box::new(ScaleTransform::new(ScaleMethod::RobustScale)));
    }

    // ── Log ──────────────────────────────────────────────────────

    /// Append a natural-log transform.
    pub fn log(&mut self) {
        self.transforms.push(Box::new(LogTransform::new()));
    }

    // ── Terminal builders ────────────────────────────────────────

    fn build_with(self, model: BoxedForecaster) -> Pipeline {
        let mut builder = InnerPipeline::builder().model(model);
        for t in self.transforms {
            builder = builder.transform_boxed(t);
        }
        Pipeline {
            inner: builder.build(),
        }
    }

    /// Build the pipeline with a `Naive` forecaster as the inner model.
    #[wasm_bindgen(js_name = buildNaive)]
    pub fn build_naive(self) -> Pipeline {
        self.build_with(Box::new(Naive::new()))
    }

    /// Build with a `SeasonalNaive(period)` inner model.
    #[wasm_bindgen(js_name = buildSeasonalNaive)]
    pub fn build_seasonal_naive(self, period: usize) -> Pipeline {
        self.build_with(Box::new(SeasonalNaive::new(period)))
    }

    /// Build with a `RandomWalkWithDrift` inner model.
    #[wasm_bindgen(js_name = buildRandomWalkDrift)]
    pub fn build_random_walk_drift(self) -> Pipeline {
        self.build_with(Box::new(RandomWalkWithDrift::new()))
    }

    /// Build with a `SimpleExponentialSmoothing(alpha)` inner model.
    #[wasm_bindgen(js_name = buildSes)]
    pub fn build_ses(self, alpha: f64) -> Pipeline {
        self.build_with(Box::new(SimpleExponentialSmoothing::new(alpha)))
    }

    /// Build with an auto-fitted additive `HoltWinters` inner model.
    #[wasm_bindgen(js_name = buildHoltWintersAdditive)]
    pub fn build_holt_winters_additive(self, period: usize) -> Pipeline {
        self.build_with(Box::new(HoltWinters::auto(period, SeasonalType::Additive)))
    }

    /// Build with an auto-fitted multiplicative `HoltWinters` inner model.
    #[wasm_bindgen(js_name = buildHoltWintersMultiplicative)]
    pub fn build_holt_winters_multiplicative(self, period: usize) -> Pipeline {
        self.build_with(Box::new(HoltWinters::auto(
            period,
            SeasonalType::Multiplicative,
        )))
    }

    /// Build with an `ETS::default()` inner model.
    #[wasm_bindgen(js_name = buildEts)]
    pub fn build_ets(self) -> Pipeline {
        self.build_with(Box::new(ETS::default()))
    }

    /// Build with an `AutoETS` inner model (period is inferred at fit time).
    #[wasm_bindgen(js_name = buildAutoEts)]
    pub fn build_auto_ets(self) -> Pipeline {
        self.build_with(Box::new(AutoETS::new()))
    }

    /// Build with a `Theta` inner model.
    #[wasm_bindgen(js_name = buildTheta)]
    pub fn build_theta(self) -> Pipeline {
        self.build_with(Box::new(Theta::new()))
    }

    /// Build with an `AutoTheta` inner model.
    #[wasm_bindgen(js_name = buildAutoTheta)]
    pub fn build_auto_theta(self) -> Pipeline {
        self.build_with(Box::new(AutoTheta::new()))
    }

    /// Build with an `ARIMA(p, d, q)` inner model.
    #[wasm_bindgen(js_name = buildArima)]
    pub fn build_arima(self, p: usize, d: usize, q: usize) -> Pipeline {
        self.build_with(Box::new(ARIMA::new(p, d, q)))
    }

    /// Build with a `SARIMA(p, d, q)(P, D, Q, period)` inner model.
    #[wasm_bindgen(js_name = buildSarima)]
    #[allow(clippy::too_many_arguments)]
    pub fn build_sarima(
        self,
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        period: usize,
    ) -> Pipeline {
        self.build_with(Box::new(SARIMA::new(
            p, d, q, seasonal_p, seasonal_d, seasonal_q, period,
        )))
    }

    /// Build with an `AutoARIMA` inner model.
    #[wasm_bindgen(js_name = buildAutoArima)]
    pub fn build_auto_arima(self) -> Pipeline {
        self.build_with(Box::new(AutoARIMA::new()))
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A composable transform → model pipeline that implements the `Forecaster`
/// interface.
///
/// Fit and predict calls are forwarded through the chain of transforms and
/// automatically inverse-transformed on output.
#[wasm_bindgen]
pub struct Pipeline {
    inner: InnerPipeline,
}

#[wasm_bindgen]
impl Pipeline {
    /// Fit the pipeline to a time series.
    pub fn fit(&mut self, series: &TimeSeries) -> Result<(), JsError> {
        self.inner
            .fit(series.inner())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Forecast `horizon` steps ahead.
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
