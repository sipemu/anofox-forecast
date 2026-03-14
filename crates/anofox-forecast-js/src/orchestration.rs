//! Orchestration module wrappers for JavaScript.
//!
//! Provides wasm-bindgen wrappers for the orchestration layer: data profiling,
//! declarative pipeline construction, multi-metric selection, preprocessing,
//! ensemble modes, structured reports, and MCP-ready tool functions.

use crate::time_series::{Forecast, TimeSeries};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::models::baseline::{Naive, SimpleMovingAverage};
use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
use anofox_forecast::models::{ModelRegistry, ModelSpec};
use anofox_forecast::orchestration::metric_strategy::{
    Metric as InnerMetric, MetricStrategy as InnerMetricStrategy,
};
use anofox_forecast::orchestration::pipeline::{
    EnsembleMode as InnerEnsembleMode, PipelineBuilder as InnerPipelineBuilder,
    PipelineResult as InnerPipelineResult,
};
use anofox_forecast::orchestration::preprocess::PreprocessMode as InnerPreprocessMode;
use anofox_forecast::orchestration::profile::DataProfile as InnerDataProfile;
use anofox_forecast::orchestration::report::PipelineReport as InnerPipelineReport;
use anofox_forecast::orchestration::tools;

// =============================================================================
// DATA PROFILE
// =============================================================================

/// Automated data profile — stationarity, trend, seasonality, quality score.
///
/// Profile a time series to understand its characteristics before model selection.
#[wasm_bindgen]
pub struct JsDataProfile {
    inner: InnerDataProfile,
}

#[wasm_bindgen]
impl JsDataProfile {
    /// Profile a time series.
    ///
    /// @param series - TimeSeries to profile
    /// @returns A comprehensive data profile
    #[wasm_bindgen(js_name = fromSeries)]
    pub fn from_series(series: &TimeSeries) -> Self {
        Self {
            inner: InnerDataProfile::from_series(series.inner()),
        }
    }

    /// Profile raw values (without timestamps).
    ///
    /// @param values - Array of numeric values
    /// @returns A data profile
    #[wasm_bindgen(js_name = fromValues)]
    pub fn from_values(values: &[f64]) -> Self {
        Self {
            inner: InnerDataProfile::from_values(values),
        }
    }

    /// Number of observations.
    #[wasm_bindgen(getter, js_name = nObservations)]
    pub fn n_observations(&self) -> usize {
        self.inner.n_observations
    }

    /// Arithmetic mean.
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        self.inner.mean
    }

    /// Standard deviation.
    #[wasm_bindgen(getter, js_name = stdDev)]
    pub fn std_dev(&self) -> f64 {
        self.inner.std_dev
    }

    /// Minimum value.
    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f64 {
        self.inner.min
    }

    /// Maximum value.
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        self.inner.max
    }

    /// Count of NaN or infinite values.
    #[wasm_bindgen(getter, js_name = missingCount)]
    pub fn missing_count(&self) -> usize {
        self.inner.missing_count
    }

    /// Whether any value is negative.
    #[wasm_bindgen(getter, js_name = hasNegatives)]
    pub fn has_negatives(&self) -> bool {
        self.inner.has_negatives
    }

    /// Whether every finite value is integer-valued.
    #[wasm_bindgen(getter, js_name = isInteger)]
    pub fn is_integer(&self) -> bool {
        self.inner.is_integer
    }

    /// ADF test statistic.
    #[wasm_bindgen(getter, js_name = adfStatistic)]
    pub fn adf_statistic(&self) -> f64 {
        self.inner.adf_statistic
    }

    /// ADF p-value.
    #[wasm_bindgen(getter, js_name = adfPValue)]
    pub fn adf_p_value(&self) -> f64 {
        self.inner.adf_p_value
    }

    /// Whether ADF concludes stationarity at 5%.
    #[wasm_bindgen(getter, js_name = adfIsStationary)]
    pub fn adf_is_stationary(&self) -> bool {
        self.inner.adf_is_stationary
    }

    /// KPSS test statistic.
    #[wasm_bindgen(getter, js_name = kpssStatistic)]
    pub fn kpss_statistic(&self) -> f64 {
        self.inner.kpss_statistic
    }

    /// KPSS p-value.
    #[wasm_bindgen(getter, js_name = kpssPValue)]
    pub fn kpss_p_value(&self) -> f64 {
        self.inner.kpss_p_value
    }

    /// Whether KPSS concludes stationarity at 5%.
    #[wasm_bindgen(getter, js_name = kpssIsStationary)]
    pub fn kpss_is_stationary(&self) -> bool {
        self.inner.kpss_is_stationary
    }

    /// Combined stationarity (ADF stationary AND KPSS stationary).
    #[wasm_bindgen(getter, js_name = isStationary)]
    pub fn is_stationary(&self) -> bool {
        self.inner.is_stationary()
    }

    /// Trend strength (R-squared, 0.0 to 1.0).
    #[wasm_bindgen(getter, js_name = trendStrength)]
    pub fn trend_strength(&self) -> f64 {
        self.inner.trend_strength
    }

    /// Slope of the linear trend.
    #[wasm_bindgen(getter, js_name = trendSlope)]
    pub fn trend_slope(&self) -> f64 {
        self.inner.trend_slope
    }

    /// Trend direction: "Rising", "Falling", or "Flat".
    #[wasm_bindgen(getter, js_name = trendDirection)]
    pub fn trend_direction(&self) -> String {
        format!("{}", self.inner.trend_direction)
    }

    /// Autocorrelation at lag 1.
    #[wasm_bindgen(getter, js_name = acfLag1)]
    pub fn acf_lag1(&self) -> f64 {
        self.inner.acf_lag1
    }

    /// Autocorrelation at lag 2.
    #[wasm_bindgen(getter, js_name = acfLag2)]
    pub fn acf_lag2(&self) -> f64 {
        self.inner.acf_lag2
    }

    /// Partial autocorrelation at lag 1.
    #[wasm_bindgen(getter, js_name = partialAcfLag1)]
    pub fn partial_acf_lag1(&self) -> f64 {
        self.inner.partial_acf_lag1
    }

    /// Skewness.
    #[wasm_bindgen(getter)]
    pub fn skewness(&self) -> f64 {
        self.inner.skewness
    }

    /// Excess kurtosis.
    #[wasm_bindgen(getter)]
    pub fn kurtosis(&self) -> f64 {
        self.inner.kurtosis
    }

    /// Approximate entropy (undefined if series is too short).
    #[wasm_bindgen(getter, js_name = approximateEntropy)]
    pub fn approximate_entropy(&self) -> Option<f64> {
        self.inner.approximate_entropy
    }

    /// Lempel-Ziv complexity (normalized).
    #[wasm_bindgen(getter, js_name = lempelZiv)]
    pub fn lempel_ziv(&self) -> f64 {
        self.inner.lempel_ziv
    }

    /// Fraction of values that are exactly zero.
    #[wasm_bindgen(getter, js_name = zeroFraction)]
    pub fn zero_fraction(&self) -> f64 {
        self.inner.zero_fraction
    }

    /// Whether the series is classified as intermittent.
    #[wasm_bindgen(getter, js_name = isIntermittent)]
    pub fn is_intermittent(&self) -> bool {
        self.inner.is_intermittent
    }

    /// Heuristic data-quality score in [0.0, 1.0].
    #[wasm_bindgen(getter, js_name = qualityScore)]
    pub fn quality_score(&self) -> f64 {
        self.inner.quality_score
    }

    /// Human-readable summary string.
    #[wasm_bindgen]
    pub fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Full profile as a formatted string.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }

    /// Profile as a JSON-serializable object.
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<JsValue, JsError> {
        let p = &self.inner;
        let obj = DataProfileJson {
            n_observations: p.n_observations,
            mean: p.mean,
            std_dev: p.std_dev,
            min: p.min,
            max: p.max,
            missing_count: p.missing_count,
            missing_fraction: p.missing_fraction,
            has_negatives: p.has_negatives,
            has_zeros: p.has_zeros,
            is_integer: p.is_integer,
            adf_statistic: p.adf_statistic,
            adf_p_value: p.adf_p_value,
            adf_is_stationary: p.adf_is_stationary,
            kpss_statistic: p.kpss_statistic,
            kpss_p_value: p.kpss_p_value,
            kpss_is_stationary: p.kpss_is_stationary,
            is_stationary: p.is_stationary(),
            trend_strength: p.trend_strength,
            trend_slope: p.trend_slope,
            trend_direction: format!("{}", p.trend_direction),
            acf_lag1: p.acf_lag1,
            acf_lag2: p.acf_lag2,
            partial_acf_lag1: p.partial_acf_lag1,
            skewness: p.skewness,
            kurtosis: p.kurtosis,
            approximate_entropy: p.approximate_entropy,
            lempel_ziv: p.lempel_ziv,
            zero_fraction: p.zero_fraction,
            is_intermittent: p.is_intermittent,
            quality_score: p.quality_score,
        };
        serde_wasm_bindgen::to_value(&obj).map_err(|e| JsError::new(&e.to_string()))
    }
}

impl JsDataProfile {
    pub(crate) fn inner(&self) -> &InnerDataProfile {
        &self.inner
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DataProfileJson {
    n_observations: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    missing_count: usize,
    missing_fraction: f64,
    has_negatives: bool,
    has_zeros: bool,
    is_integer: bool,
    adf_statistic: f64,
    adf_p_value: f64,
    adf_is_stationary: bool,
    kpss_statistic: f64,
    kpss_p_value: f64,
    kpss_is_stationary: bool,
    is_stationary: bool,
    trend_strength: f64,
    trend_slope: f64,
    trend_direction: String,
    acf_lag1: f64,
    acf_lag2: f64,
    partial_acf_lag1: f64,
    skewness: f64,
    kurtosis: f64,
    approximate_entropy: Option<f64>,
    lempel_ziv: f64,
    zero_fraction: f64,
    is_intermittent: bool,
    quality_score: f64,
}

// =============================================================================
// MODEL SELECTION TOOL
// =============================================================================

/// Recommend models based on data profile characteristics.
///
/// Returns an object with `recommended` (model names) and `reasoning` (explanations).
///
/// @param profile - A JsDataProfile from profiling
/// @param availableModels - Optional array of model names to filter by
#[wasm_bindgen(js_name = selectModels)]
pub fn select_models(
    profile: &JsDataProfile,
    available_models: Option<Vec<String>>,
) -> Result<JsValue, JsError> {
    let available = available_models.unwrap_or_default();
    let output = tools::select_models(tools::SelectModelsInput {
        profile: profile.inner(),
        available_models: &available,
    });

    let result = SelectModelsResultJson {
        recommended: output.recommended,
        reasoning: output.reasoning,
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectModelsResultJson {
    recommended: Vec<String>,
    reasoning: Vec<String>,
}

// =============================================================================
// PIPELINE BUILDER
// =============================================================================

/// Declarative pipeline builder for composing forecasting workflows.
///
/// Chain configuration methods and call `build()` to produce a pipeline,
/// then `execute()` to run it.
///
/// ```js
/// const result = new JsPipelineBuilder()
///   .profile()
///   .preprocess("auto")
///   .metric("auto")
///   .ensemble("auto")
///   .addModel("Naive")
///   .addModel("SES")
///   .withFallback()
///   .nonNegative()
///   .build()
///   .execute(ts, 12);
/// ```
#[wasm_bindgen]
pub struct JsPipelineBuilder {
    builder: InnerPipelineBuilder,
    models: Vec<(String, usize)>, // (name, seasonal_period) — 0 means non-seasonal
}

#[wasm_bindgen]
impl JsPipelineBuilder {
    /// Create a new pipeline builder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: InnerPipelineBuilder::new(),
            models: Vec::new(),
        }
    }

    /// Enable data profiling.
    pub fn profile(mut self) -> Self {
        self.builder = self.builder.profile();
        self
    }

    /// Set preprocessing mode.
    ///
    /// @param mode - "auto", "none", or "manual"
    pub fn preprocess(mut self, mode: &str) -> Self {
        let pp = match mode {
            "auto" => InnerPreprocessMode::Auto,
            "none" => InnerPreprocessMode::None,
            _ => InnerPreprocessMode::Auto,
        };
        self.builder = self.builder.preprocess(pp);
        self
    }

    /// Set the metric strategy for model selection.
    ///
    /// @param strategy - "auto", "mae", "mse", "rmse", "smape", "wape", or "mda"
    pub fn metric(mut self, strategy: &str) -> Self {
        let ms = match strategy {
            "auto" => InnerMetricStrategy::Auto,
            "mae" => InnerMetricStrategy::Single(InnerMetric::MAE),
            "mse" => InnerMetricStrategy::Single(InnerMetric::MSE),
            "rmse" => InnerMetricStrategy::Single(InnerMetric::RMSE),
            "smape" => InnerMetricStrategy::Single(InnerMetric::SMAPE),
            "wape" => InnerMetricStrategy::Single(InnerMetric::WAPE),
            "mda" => InnerMetricStrategy::Single(InnerMetric::MDA),
            _ => InnerMetricStrategy::Auto,
        };
        self.builder = self.builder.metric(ms);
        self
    }

    /// Set the ensemble mode.
    ///
    /// @param mode - "auto", "none", "mean", "median", or "weighted"
    pub fn ensemble(mut self, mode: &str) -> Self {
        use anofox_forecast::models::ensemble::CombinationMethod;
        let em = match mode {
            "auto" => InnerEnsembleMode::Auto,
            "none" => InnerEnsembleMode::None,
            "mean" => InnerEnsembleMode::Fixed(CombinationMethod::Mean),
            "median" => InnerEnsembleMode::Fixed(CombinationMethod::Median),
            "weighted" => InnerEnsembleMode::Fixed(CombinationMethod::WeightedMSE),
            _ => InnerEnsembleMode::Auto,
        };
        self.builder = self.builder.ensemble(em);
        self
    }

    /// Add a built-in model to the pipeline.
    ///
    /// Supported models: "Naive", "SES", "SMA", "SMA5", "SMA10"
    ///
    /// @param name - Model name
    #[wasm_bindgen(js_name = addModel)]
    pub fn add_model(mut self, name: &str) -> Self {
        self.models.push((name.to_string(), 0));
        self
    }

    /// Add a seasonal model to the pipeline.
    ///
    /// @param name - Model name (e.g., "SeasonalNaive")
    /// @param period - Seasonal period
    #[wasm_bindgen(js_name = addSeasonalModel)]
    pub fn add_seasonal_model(mut self, name: &str, period: usize) -> Self {
        self.models.push((name.to_string(), period));
        self
    }

    /// Select top-K models for evaluation.
    ///
    /// @param k - Number of models to select
    #[wasm_bindgen(js_name = selectModels)]
    pub fn select_models(mut self, k: usize) -> Self {
        self.builder = self.builder.select_models(k);
        self
    }

    /// Enable cross-validation.
    ///
    /// @param folds - Number of CV folds
    /// @param horizon - Forecast horizon for each fold
    #[wasm_bindgen(js_name = crossValidate)]
    pub fn cross_validate(mut self, folds: usize, horizon: usize) -> Self {
        self.builder = self.builder.cross_validate(folds, horizon);
        self
    }

    /// Enable fallback chain (Naive → SMA).
    #[wasm_bindgen(js_name = withFallback)]
    pub fn with_fallback(mut self) -> Self {
        self.builder = self.builder.with_fallback();
        self
    }

    /// Apply non-negative constraint to forecasts.
    #[wasm_bindgen(js_name = nonNegative)]
    pub fn non_negative(mut self) -> Self {
        self.builder = self.builder.non_negative();
        self
    }

    /// Set the seasonal period hint.
    ///
    /// @param period - Seasonal period (e.g., 12 for monthly data)
    #[wasm_bindgen(js_name = seasonalPeriod)]
    pub fn seasonal_period(mut self, period: usize) -> Self {
        self.builder = self.builder.seasonal_period(period);
        self
    }

    /// Build and execute the pipeline.
    ///
    /// @param series - TimeSeries to forecast
    /// @param horizon - Number of steps to forecast
    /// @returns JsPipelineResult with forecast, profile, and diagnostics
    pub fn execute(self, series: &TimeSeries, horizon: usize) -> Result<JsPipelineResult, JsError> {
        let mut registry = ModelRegistry::new();

        // Register requested models
        for (name, period) in &self.models {
            register_model(&mut registry, name, *period);
        }

        // If no models specified, add defaults
        if self.models.is_empty() {
            registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
            registry.register(ModelSpec::new(
                "SMA(5)",
                || Box::new(SimpleMovingAverage::new(5)),
                false,
            ));
            registry.register(ModelSpec::new(
                "SES",
                || Box::new(SimpleExponentialSmoothing::auto()),
                true,
            ));
        }

        let pipeline = self.builder.registry(registry).build();

        pipeline
            .execute(series.inner(), horizon)
            .map(|r| JsPipelineResult { inner: r })
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

impl Default for JsPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Register a built-in model by name.
fn register_model(registry: &mut ModelRegistry, name: &str, period: usize) {
    use anofox_forecast::models::baseline::SeasonalNaive;

    match name.to_lowercase().as_str() {
        "naive" => {
            registry.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        }
        "ses" => {
            registry.register(ModelSpec::new(
                "SES",
                || Box::new(SimpleExponentialSmoothing::auto()),
                true,
            ));
        }
        "sma" | "sma5" => {
            registry.register(ModelSpec::new(
                "SMA(5)",
                || Box::new(SimpleMovingAverage::new(5)),
                false,
            ));
        }
        "sma10" => {
            registry.register(ModelSpec::new(
                "SMA(10)",
                || Box::new(SimpleMovingAverage::new(10)),
                false,
            ));
        }
        "sma3" => {
            registry.register(ModelSpec::new(
                "SMA(3)",
                || Box::new(SimpleMovingAverage::new(3)),
                false,
            ));
        }
        "seasonalnaive" => {
            let p = if period > 0 { period } else { 7 };
            registry.register(ModelSpec::with_period(
                "SeasonalNaive",
                |p| Box::new(SeasonalNaive::new(p)),
                p,
                true,
            ));
        }
        _ => {
            // Unknown model — skip silently
        }
    }
}

// =============================================================================
// PIPELINE RESULT
// =============================================================================

/// Result of executing a forecasting pipeline.
///
/// Contains the forecast, selected model name, data profile, decision log,
/// preprocessing info, ensemble weights, and metric scores.
#[wasm_bindgen]
pub struct JsPipelineResult {
    inner: InnerPipelineResult,
}

#[wasm_bindgen]
impl JsPipelineResult {
    /// Get the forecast.
    #[wasm_bindgen(getter)]
    pub fn forecast(&self) -> Forecast {
        Forecast::from_inner(self.inner.forecast.clone())
    }

    /// Get the name of the selected model.
    #[wasm_bindgen(getter, js_name = modelName)]
    pub fn model_name(&self) -> String {
        self.inner.model_name.clone()
    }

    /// Get the data profile (undefined if profiling was not enabled).
    #[wasm_bindgen(getter)]
    pub fn profile(&self) -> Option<JsDataProfile> {
        self.inner
            .profile
            .as_ref()
            .map(|p| JsDataProfile { inner: p.clone() })
    }

    /// Get the decision log as a formatted string.
    #[wasm_bindgen(getter, js_name = decisionLog)]
    pub fn decision_log(&self) -> String {
        format!("{}", self.inner.log)
    }

    /// Get the number of decisions in the log.
    #[wasm_bindgen(getter, js_name = decisionCount)]
    pub fn decision_count(&self) -> usize {
        self.inner.log.len()
    }

    /// Get quality floor result (undefined if not computed).
    #[wasm_bindgen(getter, js_name = qualityFloor)]
    pub fn quality_floor(&self) -> Option<String> {
        self.inner
            .quality_floor
            .as_ref()
            .map(|qf| format!("{}", qf))
    }

    /// Get model confidence set (undefined if not computed).
    #[wasm_bindgen(getter, js_name = modelConfidenceSet)]
    pub fn model_confidence_set(&self) -> Result<JsValue, JsError> {
        match &self.inner.model_confidence_set {
            Some(mcs) => {
                let obj = McsJson {
                    included: mcs.included.clone(),
                    p_value: mcs.mcs_p_value,
                    single_winner: mcs.has_single_winner(),
                };
                serde_wasm_bindgen::to_value(&obj).map_err(|e| JsError::new(&e.to_string()))
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Get selection confidence (undefined if not computed).
    #[wasm_bindgen(getter, js_name = selectionConfidence)]
    pub fn selection_confidence(&self) -> Option<String> {
        self.inner
            .selection_confidence
            .as_ref()
            .map(|c| format!("{}", c))
    }

    /// Get ensemble weights as JSON (undefined if not an ensemble).
    #[wasm_bindgen(getter, js_name = ensembleWeights)]
    pub fn ensemble_weights(&self) -> Result<JsValue, JsError> {
        match &self.inner.ensemble_weights {
            Some(weights) => {
                let obj: Vec<WeightJson> = weights
                    .iter()
                    .map(|(name, w)| WeightJson {
                        model: name.clone(),
                        weight: *w,
                    })
                    .collect();
                serde_wasm_bindgen::to_value(&obj).map_err(|e| JsError::new(&e.to_string()))
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Get metric scores as JSON (undefined if not computed).
    #[wasm_bindgen(getter, js_name = metricScores)]
    pub fn metric_scores(&self) -> Result<JsValue, JsError> {
        match &self.inner.metric_scores {
            Some(scores) => {
                let obj: Vec<MetricScoreJson> = scores
                    .iter()
                    .map(|(name, ms)| MetricScoreJson {
                        model: name.clone(),
                        score: ms.primary,
                        components: ms
                            .components
                            .iter()
                            .map(|(m, v)| MetricComponentJson {
                                metric: format!("{}", m),
                                value: *v,
                            })
                            .collect(),
                    })
                    .collect();
                serde_wasm_bindgen::to_value(&obj).map_err(|e| JsError::new(&e.to_string()))
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Get preprocessing info (undefined if not applied).
    #[wasm_bindgen(getter, js_name = preprocessInfo)]
    pub fn preprocess_info(&self) -> Result<JsValue, JsError> {
        match &self.inner.preprocess {
            Some(pp) => {
                let obj = PreprocessJson {
                    boxcox_lambda: pp.boxcox_lambda,
                    outliers_replaced: pp.outliers_replaced,
                    steps_applied: pp.steps_applied.clone(),
                };
                serde_wasm_bindgen::to_value(&obj).map_err(|e| JsError::new(&e.to_string()))
            }
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Generate a full structured report.
    pub fn report(&self) -> JsPipelineReport {
        JsPipelineReport {
            inner: InnerPipelineReport::from_result(&self.inner),
        }
    }

    /// Get a human-readable summary of the result.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct McsJson {
    included: Vec<String>,
    p_value: f64,
    single_winner: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WeightJson {
    model: String,
    weight: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MetricScoreJson {
    model: String,
    score: f64,
    components: Vec<MetricComponentJson>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MetricComponentJson {
    metric: String,
    value: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PreprocessJson {
    boxcox_lambda: Option<f64>,
    outliers_replaced: usize,
    steps_applied: Vec<String>,
}

// =============================================================================
// PIPELINE REPORT
// =============================================================================

/// Structured multi-section pipeline report.
///
/// Contains summary, data profile, preprocessing, model selection,
/// ensemble, forecast, horizon analysis, decision log, and execution metadata.
#[wasm_bindgen]
pub struct JsPipelineReport {
    inner: InnerPipelineReport,
}

#[wasm_bindgen]
impl JsPipelineReport {
    /// Get the report title.
    #[wasm_bindgen(getter)]
    pub fn title(&self) -> String {
        self.inner.title.clone()
    }

    /// Get the number of sections.
    #[wasm_bindgen(getter, js_name = sectionCount)]
    pub fn section_count(&self) -> usize {
        self.inner.sections.len()
    }

    /// Get a section heading by index.
    #[wasm_bindgen(js_name = sectionHeading)]
    pub fn section_heading(&self, index: usize) -> Option<String> {
        self.inner.sections.get(index).map(|s| s.heading.clone())
    }

    /// Get the full report as formatted text.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}", self.inner)
    }

    /// Get the report as a JSON object.
    ///
    /// Returns `{ title, sections: [{ heading, content }] }`.
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<JsValue, JsError> {
        use anofox_forecast::orchestration::report::ReportContent;

        let sections: Vec<ReportSectionJson> = self
            .inner
            .sections
            .iter()
            .map(|s| {
                let content = match &s.content {
                    ReportContent::Text(text) => ReportContentJson::Text { text: text.clone() },
                    ReportContent::KeyValue(pairs) => ReportContentJson::KeyValue {
                        pairs: pairs
                            .iter()
                            .map(|(k, v)| KeyValueJson {
                                key: k.clone(),
                                value: v.clone(),
                            })
                            .collect(),
                    },
                    ReportContent::Table { headers, rows } => ReportContentJson::Table {
                        headers: headers.clone(),
                        rows: rows.clone(),
                    },
                };
                ReportSectionJson {
                    heading: s.heading.clone(),
                    content,
                }
            })
            .collect();

        let report = ReportJson {
            title: self.inner.title.clone(),
            sections,
        };
        serde_wasm_bindgen::to_value(&report).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportJson {
    title: String,
    sections: Vec<ReportSectionJson>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReportSectionJson {
    heading: String,
    content: ReportContentJson,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "type")]
enum ReportContentJson {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "keyValue")]
    KeyValue { pairs: Vec<KeyValueJson> },
    #[serde(rename = "table")]
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KeyValueJson {
    key: String,
    value: String,
}

// =============================================================================
// EXPLAIN RESULT TOOL
// =============================================================================

/// Generate a human-readable explanation of a pipeline result.
///
/// @param result - A JsPipelineResult to explain
/// @param verbosity - "brief", "normal", or "detailed"
/// @returns Object with `summary` and `sections`
#[wasm_bindgen(js_name = explainResult)]
pub fn explain_result(result: &JsPipelineResult, verbosity: &str) -> Result<JsValue, JsError> {
    let v = match verbosity {
        "brief" => tools::ExplainVerbosity::Brief,
        "detailed" => tools::ExplainVerbosity::Detailed,
        _ => tools::ExplainVerbosity::Normal,
    };

    let output = tools::explain_result(tools::ExplainResultInput {
        result: &result.inner,
        verbosity: v,
    });

    let obj = ExplainResultJson {
        summary: output.summary,
        sections: output
            .sections
            .into_iter()
            .map(|(h, c)| ExplainSectionJson {
                heading: h,
                content: c,
            })
            .collect(),
    };
    serde_wasm_bindgen::to_value(&obj).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExplainResultJson {
    summary: String,
    sections: Vec<ExplainSectionJson>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExplainSectionJson {
    heading: String,
    content: String,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn make_test_values(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                50.0 + 0.3 * i as f64 + 10.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin()
            })
            .collect()
    }

    #[wasm_bindgen_test]
    fn test_data_profile_from_values() {
        let values = make_test_values(100);
        let profile = JsDataProfile::from_values(&values);
        assert_eq!(profile.n_observations(), 100);
        assert!(profile.quality_score() > 0.0);
        assert!(!profile.is_intermittent());
    }

    #[wasm_bindgen_test]
    fn test_data_profile_from_series() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();
        let profile = JsDataProfile::from_series(&ts);
        assert_eq!(profile.n_observations(), 60);
    }

    #[wasm_bindgen_test]
    fn test_data_profile_to_json() {
        let values = make_test_values(50);
        let profile = JsDataProfile::from_values(&values);
        let json = profile.to_json();
        assert!(json.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_data_profile_summary() {
        let values = make_test_values(50);
        let profile = JsDataProfile::from_values(&values);
        let summary = profile.summary();
        assert!(!summary.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_select_models_tool() {
        let values = make_test_values(100);
        let profile = JsDataProfile::from_values(&values);
        let result = select_models(&profile, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_select_models_with_filter() {
        let values = make_test_values(100);
        let profile = JsDataProfile::from_values(&values);
        let available = vec!["Naive".to_string(), "SES".to_string()];
        let result = select_models(&profile, Some(available));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_basic() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .profile()
            .add_model("Naive")
            .add_model("SES")
            .with_fallback()
            .execute(&ts, 7);

        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(!r.model_name().is_empty());
        assert_eq!(r.forecast().horizon(), 7);
        assert!(r.profile().is_some());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_with_preprocessing() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .profile()
            .preprocess("auto")
            .metric("auto")
            .add_model("Naive")
            .add_model("SES")
            .with_fallback()
            .execute(&ts, 5);

        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_with_ensemble() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .profile()
            .ensemble("auto")
            .add_model("Naive")
            .add_model("SES")
            .add_model("SMA")
            .with_fallback()
            .execute(&ts, 5);

        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_default_models() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        // No models added — should use defaults
        let result = JsPipelineBuilder::new()
            .profile()
            .with_fallback()
            .execute(&ts, 5);

        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_report() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .add_model("Naive")
            .with_fallback()
            .execute(&ts, 5)
            .unwrap();

        let report = result.report();
        assert!(!report.title().is_empty());
        assert!(report.section_count() > 0);
        let text = report.to_string_js();
        assert!(text.contains("Pipeline Report"));
    }

    #[wasm_bindgen_test]
    fn test_pipeline_report_json() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .add_model("Naive")
            .with_fallback()
            .execute(&ts, 5)
            .unwrap();

        let report = result.report();
        let json = report.to_json();
        assert!(json.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_explain_result_brief() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .add_model("Naive")
            .with_fallback()
            .execute(&ts, 5)
            .unwrap();

        let explanation = explain_result(&result, "brief");
        assert!(explanation.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_explain_result_detailed() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .profile()
            .add_model("Naive")
            .with_fallback()
            .execute(&ts, 5)
            .unwrap();

        let explanation = explain_result(&result, "detailed");
        assert!(explanation.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_pipeline_result_accessors() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .profile()
            .add_model("Naive")
            .add_model("SES")
            .with_fallback()
            .execute(&ts, 5)
            .unwrap();

        // Accessors should not panic
        let _ = result.decision_log();
        assert!(result.decision_count() > 0);
        let _ = result.quality_floor();
        let _ = result.model_confidence_set();
        let _ = result.selection_confidence();
        let _ = result.ensemble_weights();
        let _ = result.metric_scores();
        let _ = result.preprocess_info();
        let _ = result.to_string_js();
    }

    #[wasm_bindgen_test]
    fn test_pipeline_non_negative() {
        let values = make_test_values(60);
        let ts = TimeSeries::new(&values).unwrap();

        let result = JsPipelineBuilder::new()
            .add_model("Naive")
            .with_fallback()
            .non_negative()
            .execute(&ts, 5)
            .unwrap();

        for v in result.forecast().values() {
            assert!(v >= 0.0);
        }
    }
}
