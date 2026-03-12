//! STL and MSTL decomposition wrappers for JavaScript.
//!
//! Provides wasm-bindgen wrappers for seasonal-trend decomposition methods.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::seasonality::{MSTL, STL};

/// Result of STL decomposition, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StlResultJs {
    /// Trend component.
    trend: Vec<f64>,
    /// Seasonal component.
    seasonal: Vec<f64>,
    /// Remainder component.
    remainder: Vec<f64>,
    /// Seasonal strength (0 to 1).
    seasonal_strength: f64,
    /// Trend strength (0 to 1).
    trend_strength: f64,
}

/// Result of MSTL decomposition, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MstlResultJs {
    /// Trend component.
    trend: Vec<f64>,
    /// Seasonal components (one array per period).
    seasonals: Vec<Vec<f64>>,
    /// The seasonal periods corresponding to each component.
    seasonal_periods: Vec<usize>,
    /// Remainder component.
    remainder: Vec<f64>,
    /// Trend strength (0 to 1).
    trend_strength: f64,
}

/// Perform STL (Seasonal-Trend decomposition using LOESS).
///
/// Decomposes a time series into trend, seasonal, and remainder components.
///
/// @param values - Array of numeric values
/// @param period - Seasonal period (e.g., 12 for monthly data)
/// @param robust - Enable robust fitting to reduce outlier influence (default: false)
/// @returns Object with `trend`, `seasonal`, `remainder`, `seasonalStrength`, and `trendStrength`
#[wasm_bindgen(js_name = stlDecompose)]
pub fn stl_decompose(
    values: &[f64],
    period: usize,
    robust: Option<bool>,
) -> Result<JsValue, JsError> {
    let stl = if robust.unwrap_or(false) {
        STL::new(period).robust()
    } else {
        STL::new(period)
    };

    let result = stl.decompose(values).ok_or_else(|| {
        JsError::new("STL decomposition failed. Ensure data length >= 2 * period")
    })?;

    let js_result = StlResultJs {
        seasonal_strength: result.seasonal_strength(),
        trend_strength: result.trend_strength(),
        trend: result.trend,
        seasonal: result.seasonal,
        remainder: result.remainder,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Perform MSTL (Multiple Seasonal-Trend decomposition using LOESS).
///
/// Decomposes a time series with multiple seasonal periods into trend,
/// multiple seasonal components, and remainder.
///
/// @param values - Array of numeric values
/// @param periods - Array of seasonal periods (e.g., [7, 365] for daily data)
/// @returns Object with `trend`, `seasonals` (array of arrays), `seasonalPeriods`, `remainder`, and `trendStrength`
#[wasm_bindgen(js_name = mstlDecompose)]
pub fn mstl_decompose(values: &[f64], periods: Vec<usize>) -> Result<JsValue, JsError> {
    let mstl = MSTL::new(periods);

    let result = mstl.decompose(values).ok_or_else(|| {
        JsError::new("MSTL decomposition failed. Ensure data length >= 2 * max(periods)")
    })?;

    let js_result = MstlResultJs {
        trend_strength: result.trend_strength(),
        trend: result.trend,
        seasonals: result.seasonal_components,
        seasonal_periods: result.seasonal_periods,
        remainder: result.remainder,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn generate_seasonal_series(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal =
                    10.0 * ((2.0 * std::f64::consts::PI * i as f64 / period as f64).sin());
                trend + seasonal
            })
            .collect()
    }

    #[wasm_bindgen_test]
    fn test_stl_decompose() {
        let series = generate_seasonal_series(120, 12);
        let result = stl_decompose(&series, 12, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_stl_decompose_robust() {
        let mut series = generate_seasonal_series(120, 12);
        series[30] = 100.0; // outlier
        let result = stl_decompose(&series, 12, Some(true));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_stl_decompose_insufficient_data() {
        let series = vec![1.0; 10];
        let result = stl_decompose(&series, 12, None);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_mstl_decompose() {
        let series: Vec<f64> = (0..200)
            .map(|i| {
                let trend = 0.05 * i as f64;
                let s1 = 5.0 * ((2.0 * std::f64::consts::PI * i as f64 / 7.0).sin());
                let s2 = 3.0 * ((2.0 * std::f64::consts::PI * i as f64 / 24.0).sin());
                trend + s1 + s2
            })
            .collect();

        let result = mstl_decompose(&series, vec![7, 24]);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_mstl_decompose_single_period() {
        let series = generate_seasonal_series(120, 12);
        let result = mstl_decompose(&series, vec![12]);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_mstl_decompose_insufficient_data() {
        let series = vec![1.0; 30];
        let result = mstl_decompose(&series, vec![7, 24]);
        assert!(result.is_err());
    }
}
