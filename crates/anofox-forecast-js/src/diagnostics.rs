//! Diagnostic bindings for JavaScript: AID (Automatic Identification of
//! Demand) and intermittent demand diagnostics.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::validation::aid::{
    AidAnalyzer as InnerAidAnalyzer, AidAnomalyLabel, AidResult as InnerAidResult,
};
use anofox_forecast::validation::intermittent_diagnostics::{
    DemandClassification, IntermittentDiagnostics as InnerIntermittent,
};

// ---------------------------------------------------------------------------
// AID
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AidSummaryJs {
    demand_type: String,
    distribution: String,
    is_fractional: bool,
    mean: f64,
    variance: f64,
    shape: Option<f64>,
    scale: Option<f64>,
    zero_prob: Option<f64>,
    zero_proportion: f64,
    n_observations: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AidResultJs {
    summary: AidSummaryJs,
    anomalies: Vec<String>,
}

fn anomaly_label_str(label: AidAnomalyLabel) -> &'static str {
    match label {
        AidAnomalyLabel::Normal => "normal",
        AidAnomalyLabel::Stockout => "stockout",
        AidAnomalyLabel::NewProduct => "newProduct",
        AidAnomalyLabel::ObsoleteProduct => "obsoleteProduct",
        AidAnomalyLabel::HighOutlier => "highOutlier",
        AidAnomalyLabel::LowOutlier => "lowOutlier",
    }
}

fn aid_to_js(result: &InnerAidResult) -> AidResultJs {
    let summary = result.summary();
    let features = result.features();
    AidResultJs {
        summary: AidSummaryJs {
            demand_type: format!("{:?}", summary.demand_type),
            distribution: format!("{:?}", summary.distribution),
            is_fractional: summary.is_fractional,
            mean: summary.mean,
            variance: summary.variance,
            shape: summary.shape,
            scale: summary.scale,
            zero_prob: summary.zero_prob,
            zero_proportion: summary.zero_proportion,
            n_observations: summary.n_observations,
        },
        anomalies: features
            .labels
            .iter()
            .map(|&l| anomaly_label_str(l).to_string())
            .collect(),
    }
}

/// Automatic Identification of Demand: distribution fitting, demand type
/// classification, and per-observation anomaly detection (stockouts,
/// lifecycle events, outliers).
///
/// @param values - Array of demand observations
/// @param anomalyAlpha - Significance level for anomaly detection (default: 0.05)
/// @param intermittentThreshold - Zero-proportion threshold for "intermittent" (default: 0.3)
/// @returns Object with `summary` (demand type, distribution, params) and
///          `anomalies` (per-observation labels)
#[wasm_bindgen(js_name = analyzeDemand)]
pub fn analyze_demand(
    values: &[f64],
    anomaly_alpha: Option<f64>,
    intermittent_threshold: Option<f64>,
) -> Result<JsValue, JsError> {
    let mut analyzer = InnerAidAnalyzer::new();
    if let Some(a) = anomaly_alpha {
        analyzer = analyzer.anomaly_alpha(a);
    }
    if let Some(t) = intermittent_threshold {
        analyzer = analyzer.intermittent_threshold(t);
    }
    let result = analyzer.analyze(values);
    let js_result = aid_to_js(&result);
    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Intermittent diagnostics
// ---------------------------------------------------------------------------

fn classification_str(c: DemandClassification) -> &'static str {
    match c {
        DemandClassification::Smooth => "smooth",
        DemandClassification::Erratic => "erratic",
        DemandClassification::Intermittent => "intermittent",
        DemandClassification::Lumpy => "lumpy",
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct IntermittentResultJs {
    adi: f64,
    cv_squared: f64,
    classification: String,
    recommended_model: String,
    zero_fraction: f64,
    bias: f64,
    coverage_rate: Option<f64>,
    periods_in_stock: Vec<f64>,
}

fn intermittent_to_js(inner: &InnerIntermittent) -> IntermittentResultJs {
    IntermittentResultJs {
        adi: inner.adi,
        cv_squared: inner.cv_squared,
        classification: classification_str(inner.classification).to_string(),
        recommended_model: inner.recommended_model().to_string(),
        zero_fraction: inner.zero_fraction,
        bias: inner.bias,
        coverage_rate: inner.coverage_rate,
        periods_in_stock: inner.periods_in_stock.clone(),
    }
}

/// Intermittent demand diagnostics: Syntetos-Boylan classification +
/// recommended model.
///
/// @param actuals - Actual demand values
/// @returns Object with `adi`, `cvSquared`, `classification`, `recommendedModel`, `zeroFraction`
#[wasm_bindgen(js_name = intermittentDiagnostics)]
pub fn intermittent_diagnostics(actuals: &[f64]) -> Result<JsValue, JsError> {
    let diag = InnerIntermittent::from_data(actuals);
    let out = intermittent_to_js(&diag);
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Intermittent demand diagnostics with a forecast — adds bias and
/// periods-in-stock tracking.
///
/// @param actuals - Actual demand values
/// @param forecast - Forecast values (length ≤ actuals)
/// @returns Object with full diagnostics including `bias` and `periodsInStock`
#[wasm_bindgen(js_name = intermittentDiagnosticsWithForecast)]
pub fn intermittent_diagnostics_with_forecast(
    actuals: &[f64],
    forecast: &[f64],
) -> Result<JsValue, JsError> {
    let diag = InnerIntermittent::with_forecast(actuals, forecast);
    let out = intermittent_to_js(&diag);
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Intermittent diagnostics with a forecast and prediction intervals —
/// additionally computes coverage rate.
#[wasm_bindgen(js_name = intermittentDiagnosticsWithIntervals)]
pub fn intermittent_diagnostics_with_intervals(
    actuals: &[f64],
    forecast: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<JsValue, JsError> {
    let diag = InnerIntermittent::with_intervals(actuals, forecast, lower, upper);
    let out = intermittent_to_js(&diag);
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}
