//! Anomaly detection + spectral analysis bindings for JavaScript.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::detection::{
    detect_dominant_period as inner_detect_dominant_period,
    detect_outliers as inner_detect_outliers, detect_outliers_auto as inner_detect_outliers_auto,
    welch_periodogram as inner_welch_periodogram, OutlierConfig as InnerOutlierConfig,
    OutlierMethod as InnerOutlierMethod,
};

fn parse_outlier_method(m: &str) -> Result<InnerOutlierMethod, JsError> {
    match m {
        "iqr" | "IQR" => Ok(InnerOutlierMethod::IQR),
        "zScore" | "z_score" | "ZScore" => Ok(InnerOutlierMethod::ZScore),
        "modifiedZScore" | "modified_z_score" | "ModifiedZScore" => {
            Ok(InnerOutlierMethod::ModifiedZScore)
        }
        other => Err(JsError::new(&format!(
            "Unknown outlier method '{}'. Use: iqr, zScore, modifiedZScore",
            other
        ))),
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutlierResultJs {
    outlier_indices: Vec<usize>,
    scores: Vec<f64>,
    threshold: f64,
    method: String,
    n_outliers: usize,
    outlier_percentage: f64,
}

/// Detect outliers in a time series.
///
/// @param values - Array of numeric values
/// @param method - "iqr" (default), "zScore", or "modifiedZScore"
/// @param threshold - Method-specific threshold (default: 1.5 IQR / 3.0 Z / 3.5 MZ)
/// @returns Object with `outlierIndices`, `scores`, `threshold`, `method`, `nOutliers`, `outlierPercentage`
#[wasm_bindgen(js_name = detectOutliers)]
pub fn detect_outliers(
    values: &[f64],
    method: Option<String>,
    threshold: Option<f64>,
) -> Result<JsValue, JsError> {
    let cfg = match method.as_deref() {
        Some(m) => {
            let meth = parse_outlier_method(m)?;
            let default_thr = match meth {
                InnerOutlierMethod::IQR => 1.5,
                InnerOutlierMethod::ZScore => 3.0,
                InnerOutlierMethod::ModifiedZScore => 3.5,
            };
            InnerOutlierConfig {
                method: meth,
                threshold: threshold.unwrap_or(default_thr),
            }
        }
        None => InnerOutlierConfig::default(),
    };

    let result = inner_detect_outliers(values, &cfg);
    let out = OutlierResultJs {
        n_outliers: result.outlier_count(),
        outlier_percentage: result.outlier_percentage(),
        outlier_indices: result.outlier_indices,
        scores: result.scores,
        threshold: result.threshold,
        method: format!("{:?}", result.method),
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Detect outliers with default configuration (IQR method, 1.5 multiplier).
#[wasm_bindgen(js_name = detectOutliersAuto)]
pub fn detect_outliers_auto(values: &[f64]) -> Result<JsValue, JsError> {
    let result = inner_detect_outliers_auto(values);
    let out = OutlierResultJs {
        n_outliers: result.outlier_count(),
        outlier_percentage: result.outlier_percentage(),
        outlier_indices: result.outlier_indices,
        scores: result.scores,
        threshold: result.threshold,
        method: format!("{:?}", result.method),
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PeriodogramEntry {
    period: usize,
    power: f64,
}

/// Welch's periodogram for spectral density estimation.
///
/// @param values - Array of numeric values
/// @param windowSize - Window size for the segmented FFT
/// @param overlap - Fractional overlap between windows in [0, 1) (default: 0.5)
/// @returns Array of {period, power} entries sorted by frequency
#[wasm_bindgen(js_name = welchPeriodogram)]
pub fn welch_periodogram(
    values: &[f64],
    window_size: usize,
    overlap: Option<f64>,
) -> Result<JsValue, JsError> {
    let result = inner_welch_periodogram(values, window_size, overlap.unwrap_or(0.5));
    let out: Vec<PeriodogramEntry> = result
        .into_iter()
        .map(|(period, power)| PeriodogramEntry { period, power })
        .collect();
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Detect the dominant seasonal period in a time series.
///
/// @param values - Array of numeric values
/// @returns The detected period, or `undefined` if no strong period is found
#[wasm_bindgen(js_name = detectDominantPeriod)]
pub fn detect_dominant_period(values: &[f64]) -> Option<usize> {
    inner_detect_dominant_period(values)
}
