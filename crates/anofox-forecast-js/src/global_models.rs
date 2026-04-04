//! Global/batch model bindings for JavaScript.
//!
//! Exposes GlobalETS, GlobalAutoETS, GlobalCroston for processing
//! many series with shared parameters.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::models::exponential::{
    ETSSeasonalType, ETSSpec, ErrorType, GlobalAutoETS as InnerGlobalAutoETS,
    GlobalETS as InnerGlobalETS, ModelPool as InnerModelPool, TrendType,
};
use anofox_forecast::models::intermittent::GlobalCroston as InnerGlobalCroston;

/// Parse a ModelPool string into the Rust enum.
fn parse_model_pool(pool: &str) -> Result<InnerModelPool, JsError> {
    match pool {
        "complete" | "Complete" => Ok(InnerModelPool::Complete),
        "reduced" | "Reduced" => Ok(InnerModelPool::Reduced),
        "dampedTrendOnly" | "DampedTrendOnly" => Ok(InnerModelPool::DampedTrendOnly),
        "matchErrorSeasonal" | "MatchErrorSeasonal" => Ok(InnerModelPool::MatchErrorSeasonal),
        "noMultiplicativeTrend" | "NoMultiplicativeTrend" => {
            Ok(InnerModelPool::NoMultiplicativeTrend)
        }
        other => Err(JsError::new(&format!(
            "Unknown model pool '{}'. Use: complete, reduced, dampedTrendOnly, matchErrorSeasonal, noMultiplicativeTrend",
            other
        ))),
    }
}

/// Parse an ETS spec string like "ANA", "AAdA", "MNM" etc.
fn parse_ets_spec(spec: &str) -> Result<ETSSpec, JsError> {
    let chars: Vec<char> = spec.chars().collect();
    if chars.len() < 3 {
        return Err(JsError::new(
            "ETS spec must be at least 3 chars (e.g., 'ANA')",
        ));
    }

    let error = match chars[0] {
        'A' => ErrorType::Additive,
        'M' => ErrorType::Multiplicative,
        _ => return Err(JsError::new("Error type must be 'A' or 'M'")),
    };

    let (trend, trend_end) = if chars.len() >= 4 && chars[1] == 'A' && chars[2] == 'd' {
        (TrendType::AdditiveDamped, 3)
    } else {
        match chars[1] {
            'N' => (TrendType::None, 2),
            'A' => (TrendType::Additive, 2),
            _ => return Err(JsError::new("Trend must be 'N', 'A', or 'Ad'")),
        }
    };

    let seasonal = if trend_end >= chars.len() {
        ETSSeasonalType::None
    } else {
        match chars[trend_end] {
            'N' => ETSSeasonalType::None,
            'A' => ETSSeasonalType::Additive,
            'M' => ETSSeasonalType::Multiplicative,
            _ => return Err(JsError::new("Seasonal must be 'N', 'A', or 'M'")),
        }
    };

    Ok(ETSSpec::new(error, trend, seasonal))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GlobalETSResult {
    forecasts: Vec<Vec<f64>>,
    alpha: f64,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
}

/// Fit a GlobalETS model: shared smoothing parameters across many series.
///
/// @param series - Array of arrays (each inner array is one series)
/// @param spec - ETS specification string (e.g., "ANA", "AAdA", "MNM")
/// @param period - Seasonal period
/// @param horizon - Forecast horizon
/// @returns Object with forecasts array and fitted parameters
#[wasm_bindgen(js_name = globalEts)]
pub fn global_ets(
    series: JsValue,
    spec: &str,
    period: usize,
    horizon: usize,
) -> Result<JsValue, JsError> {
    let all_series: Vec<Vec<f64>> =
        serde_wasm_bindgen::from_value(series).map_err(|e| JsError::new(&e.to_string()))?;

    let ets_spec = parse_ets_spec(spec)?;
    let mut model = InnerGlobalETS::new(ets_spec, period);
    model
        .fit(&all_series)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let forecasts = model.predict(horizon);
    let (alpha, beta, gamma, phi) = model.params();

    let result = GlobalETSResult {
        forecasts,
        alpha,
        beta,
        gamma,
        phi,
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GlobalAutoETSResult {
    forecasts: Vec<Vec<f64>>,
    selected_specs: Vec<String>,
}

/// Fit a GlobalAutoETS model: automatic model selection across many series.
///
/// @param series - Array of arrays (each inner array is one series)
/// @param period - Seasonal period
/// @param horizon - Forecast horizon
/// @param pool - Model pool: "complete", "reduced", "dampedTrendOnly", "matchErrorSeasonal" (default: "complete")
/// @returns Object with forecasts and selectedSpecs arrays
#[wasm_bindgen(js_name = globalAutoEts)]
pub fn global_auto_ets(
    series: JsValue,
    period: usize,
    horizon: usize,
    pool: Option<String>,
) -> Result<JsValue, JsError> {
    let all_series: Vec<Vec<f64>> =
        serde_wasm_bindgen::from_value(series).map_err(|e| JsError::new(&e.to_string()))?;

    let model_pool = match pool.as_deref() {
        Some(p) => parse_model_pool(p)?,
        None => InnerModelPool::Complete,
    };

    let mut model = InnerGlobalAutoETS::new(period, model_pool);
    model
        .fit(&all_series)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let forecasts = model.predict(horizon);
    let specs: Vec<String> = model
        .selected_specs()
        .iter()
        .map(|s| format!("{:?}", s))
        .collect();

    let result = GlobalAutoETSResult {
        forecasts,
        selected_specs: specs,
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GlobalCrostonResult {
    forecasts: Vec<Vec<f64>>,
    alpha: f64,
}

/// Fit a GlobalCroston model: shared α across many intermittent demand series.
///
/// @param series - Array of arrays (each inner array is one series)
/// @param horizon - Forecast horizon
/// @param variant - "classic" or "sba" (default: "classic")
/// @returns Object with forecasts array and alpha
#[wasm_bindgen(js_name = globalCroston)]
pub fn global_croston(
    series: JsValue,
    horizon: usize,
    variant: Option<String>,
) -> Result<JsValue, JsError> {
    let all_series: Vec<Vec<f64>> =
        serde_wasm_bindgen::from_value(series).map_err(|e| JsError::new(&e.to_string()))?;

    let mut model = match variant.as_deref() {
        Some("sba") | Some("SBA") => InnerGlobalCroston::sba(),
        _ => InnerGlobalCroston::new(),
    };

    model
        .fit(&all_series)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let forecasts = model.predict(horizon);

    let result = GlobalCrostonResult {
        forecasts,
        alpha: model.alpha(),
    };
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
}
