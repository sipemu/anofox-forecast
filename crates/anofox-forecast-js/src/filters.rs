//! Band-pass / trend filter bindings for JavaScript.
//!
//! Hodrick-Prescott, Christiano-Fitzgerald, Baxter-King, Hamilton, and
//! fractional differencing — each returns trend/cycle/differenced series as
//! plain JS arrays.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::models::arima::{
    find_min_fractional_d as inner_find_min_frac_d, fractional_difference as inner_frac_diff,
};
use anofox_forecast::seasonality::bandpass::{bk_filter as inner_bk, cf_filter as inner_cf};
use anofox_forecast::seasonality::hamilton::hamilton_filter as inner_hamilton;
use anofox_forecast::seasonality::hp_filter::HodrickPrescottFilter;
use anofox_forecast::seasonality::traits::TrendComponent;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CycleDecompositionJs {
    trend: Vec<f64>,
    cycle: Vec<f64>,
    low_period: usize,
    high_period: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HamiltonDecompositionJs {
    trend: Vec<f64>,
    cycle: Vec<f64>,
    h: usize,
    p: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct HpResultJs {
    trend: Vec<f64>,
    cycle: Vec<f64>,
    lambda: f64,
}

/// Apply the Hodrick-Prescott filter to a series.
///
/// @param values - Array of numeric values
/// @param lambda - Smoothing parameter (default: 1600 for quarterly data).
///                 Common values: 1600 (quarterly), 129600 (monthly), 6.25 (annual).
/// @returns Object with `trend`, `cycle`, `lambda`
#[wasm_bindgen(js_name = hpFilter)]
pub fn hp_filter(values: &[f64], lambda: Option<f64>) -> Result<JsValue, JsError> {
    let lam = lambda.unwrap_or(1600.0);
    let mut hp = HodrickPrescottFilter::new(lam).map_err(|e| JsError::new(&e.to_string()))?;
    hp.fit_trend(values)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let trend = hp.fitted_trend().to_vec();
    let cycle = hp.cycle().to_vec();
    let out = HpResultJs {
        trend,
        cycle,
        lambda: lam,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Christiano-Fitzgerald band-pass filter — asymmetric, preserves series length.
///
/// @param values - Array of numeric values
/// @param lowPeriod - Lower bound of the passband in periods (e.g., 6 quarters)
/// @param highPeriod - Upper bound of the passband in periods (e.g., 32 quarters)
/// @param drift - If `true`, remove linear drift before filtering (default: true)
/// @returns Object with `trend`, `cycle`, `lowPeriod`, `highPeriod`
#[wasm_bindgen(js_name = cfFilter)]
pub fn cf_filter(
    values: &[f64],
    low_period: usize,
    high_period: usize,
    drift: Option<bool>,
) -> Result<JsValue, JsError> {
    let decomposition = inner_cf(values, low_period, high_period, drift.unwrap_or(true))
        .map_err(|e| JsError::new(&e.to_string()))?;
    let out = CycleDecompositionJs {
        trend: decomposition.trend,
        cycle: decomposition.cycle,
        low_period: decomposition.low_period,
        high_period: decomposition.high_period,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Baxter-King band-pass filter — symmetric, loses `2k` observations at each edge.
///
/// @param values - Array of numeric values
/// @param lowPeriod - Lower bound of the passband in periods
/// @param highPeriod - Upper bound of the passband in periods
/// @param k - Half-length of the filter (default: 12 for quarterly data)
/// @returns Object with `trend`, `cycle`, `lowPeriod`, `highPeriod`
#[wasm_bindgen(js_name = bkFilter)]
pub fn bk_filter(
    values: &[f64],
    low_period: usize,
    high_period: usize,
    k: Option<usize>,
) -> Result<JsValue, JsError> {
    let decomposition = inner_bk(values, low_period, high_period, k.unwrap_or(12))
        .map_err(|e| JsError::new(&e.to_string()))?;
    let out = CycleDecompositionJs {
        trend: decomposition.trend,
        cycle: decomposition.cycle,
        low_period: decomposition.low_period,
        high_period: decomposition.high_period,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Hamilton filter — regression-based trend-cycle decomposition that avoids
/// the HP filter's endpoint bias.
///
/// @param values - Array of numeric values
/// @param h - Forecast horizon used in the regression (default: 8)
/// @param p - Number of lags in the regression (default: 4)
/// @returns Object with `trend`, `cycle`, `h`, `p`
#[wasm_bindgen(js_name = hamiltonFilter)]
pub fn hamilton_filter(
    values: &[f64],
    h: Option<usize>,
    p: Option<usize>,
) -> Result<JsValue, JsError> {
    let h = h.unwrap_or(8);
    let p = p.unwrap_or(4);
    let decomposition = inner_hamilton(values, h, p).map_err(|e| JsError::new(&e.to_string()))?;
    let out = HamiltonDecompositionJs {
        trend: decomposition.trend,
        cycle: decomposition.cycle,
        h,
        p,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}

/// Fractional differencing — removes just enough memory to achieve
/// stationarity while preserving predictive signal.
///
/// @param values - Array of numeric values
/// @param d - Differencing order, typically `0 < d < 1`
/// @param threshold - Weight truncation threshold (default: 1e-4)
/// @returns Array of differenced values (shorter than input)
#[wasm_bindgen(js_name = fractionalDifference)]
pub fn fractional_difference(values: &[f64], d: f64, threshold: Option<f64>) -> Vec<f64> {
    inner_frac_diff(values, d, threshold.unwrap_or(1e-4))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MinFracDJs {
    d: f64,
    p_value: f64,
}

/// Find the minimum fractional differencing order `d` that makes the series
/// stationary according to the Augmented Dickey-Fuller test.
///
/// @param values - Array of numeric values
/// @param significance - ADF p-value threshold (default: 0.05)
/// @param threshold - Weight truncation threshold for the inner differencing (default: 1e-4)
/// @returns Object with `d` (the minimum differencing order) and `pValue` (the ADF p-value at that d)
#[wasm_bindgen(js_name = findMinFractionalD)]
pub fn find_min_fractional_d(
    values: &[f64],
    significance: Option<f64>,
    threshold: Option<f64>,
) -> Result<JsValue, JsError> {
    let (d, p_value) = inner_find_min_frac_d(
        values,
        significance.unwrap_or(0.05),
        threshold.unwrap_or(1e-4),
    );
    let out = MinFracDJs { d, p_value };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
}
