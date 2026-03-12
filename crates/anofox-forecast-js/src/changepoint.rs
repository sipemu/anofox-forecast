//! Changepoint detection bindings for JavaScript.
//!
//! Exposes the PELT (Pruned Exact Linear Time) algorithm for detecting
//! structural changes in time series data.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::changepoint::{
    pelt_detect, CostFunction as InnerCostFunction, PeltConfig as InnerPeltConfig,
};

/// Result of changepoint detection, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PeltResultJs {
    /// Detected changepoint indices.
    changepoints: Vec<usize>,
    /// Segment boundaries as [start, end] pairs.
    segments: Vec<(usize, usize)>,
    /// Total cost (excluding penalty).
    cost: f64,
    /// Number of changepoints detected.
    n_changepoints: usize,
    /// Segment means.
    segment_means: Vec<f64>,
}

/// Detect changepoints in a time series using the PELT algorithm.
///
/// Returns an object with `changepoints` (indices), `segments` (boundary pairs),
/// `cost`, `nChangepoints`, and `segmentMeans`.
///
/// @param values - Array of numeric values
/// @param penalty - Penalty for each changepoint (controls number of changepoints; higher = fewer)
/// @param costFunction - Cost function: "l1", "l2" (default), "normal", "poisson", "linearTrend", "meanVariance", or "cusum"
/// @param minSegmentLength - Minimum segment length (default: 2)
/// @returns Object with changepoint detection results
#[wasm_bindgen(js_name = detectChangepoints)]
pub fn detect_changepoints(
    values: &[f64],
    penalty: f64,
    cost_function: Option<String>,
    min_segment_length: Option<usize>,
) -> Result<JsValue, JsError> {
    let cost_fn = match cost_function.as_deref() {
        Some("l1") | Some("L1") => InnerCostFunction::L1,
        Some("l2") | Some("L2") | None => InnerCostFunction::L2,
        Some("normal") | Some("Normal") => InnerCostFunction::Normal,
        Some("poisson") | Some("Poisson") => InnerCostFunction::Poisson,
        Some("linearTrend") | Some("LinearTrend") => InnerCostFunction::LinearTrend,
        Some("meanVariance") | Some("MeanVariance") => InnerCostFunction::MeanVariance,
        Some("cusum") | Some("Cusum") | Some("CUSUM") => InnerCostFunction::Cusum,
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown cost function '{}'. Use: l1, l2, normal, poisson, linearTrend, meanVariance, cusum",
                other
            )));
        }
    };

    let config = InnerPeltConfig::default()
        .cost_function(cost_fn)
        .penalty(penalty)
        .min_segment_length(min_segment_length.unwrap_or(2));

    let result = pelt_detect(values, &config);

    let js_result = PeltResultJs {
        segment_means: result.segment_means(values),
        changepoints: result.changepoints,
        segments: result.segments,
        cost: result.cost,
        n_changepoints: result.n_changepoints,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Detect changepoints using BIC penalty (penalty = log(n)).
///
/// BIC penalty automatically adapts to series length, providing a good
/// default for most use cases.
///
/// @param values - Array of numeric values
/// @param costFunction - Cost function (default: "l2")
/// @returns Object with changepoint detection results
#[wasm_bindgen(js_name = detectChangepointsBic)]
pub fn detect_changepoints_bic(
    values: &[f64],
    cost_function: Option<String>,
) -> Result<JsValue, JsError> {
    let penalty = (values.len() as f64).ln();
    detect_changepoints(values, penalty, cost_function, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_detect_changepoints_clear_shift() {
        let mut series = vec![0.0; 10];
        series.extend(vec![10.0; 10]);

        let result = detect_changepoints(&series, 2.0, None, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_detect_changepoints_no_change() {
        let series = vec![5.0; 20];

        let result = detect_changepoints(&series, 10.0, None, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_detect_changepoints_l1() {
        let mut series = vec![0.0; 10];
        series.extend(vec![10.0; 10]);

        let result = detect_changepoints(&series, 2.0, Some("l1".to_string()), None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_detect_changepoints_linear_trend() {
        let mut series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        series.extend((0..50).map(|i| 100.0 - i as f64));

        let result = detect_changepoints(&series, 100.0, Some("linearTrend".to_string()), None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_detect_changepoints_bic() {
        let mut series = vec![0.0; 20];
        series.extend(vec![10.0; 20]);

        let result = detect_changepoints_bic(&series, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_detect_changepoints_invalid_cost() {
        let series = vec![1.0; 20];
        let result = detect_changepoints(&series, 2.0, Some("invalid".to_string()), None);
        assert!(result.is_err());
    }
}
