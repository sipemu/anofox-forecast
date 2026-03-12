//! Cross-validation wrappers for JavaScript.
//!
//! Provides wasm-bindgen wrappers for time series cross-validation.

use crate::time_series::TimeSeries;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::baseline::{Naive, SimpleMovingAverage};
use anofox_forecast::models::exponential::{AutoETS, HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::utils::cross_validation::{cross_validate, CVConfig};

/// Result of cross-validation, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CrossValidationResultJs {
    /// Mean RMSE across folds.
    rmse: f64,
    /// Mean MAE across folds.
    mae: f64,
    /// Mean MAPE across folds (NaN if unavailable).
    mape: f64,
    /// Mean SMAPE across folds.
    smape: f64,
    /// Number of folds evaluated.
    folds: usize,
    /// Standard deviation of MAE across folds.
    mae_std: f64,
    /// Standard deviation of RMSE across folds.
    rmse_std: f64,
}

/// Perform time series cross-validation with expanding window.
///
/// Evaluates a forecasting model using multiple train/test splits where
/// the training window grows with each fold.
///
/// @param values - Array of numeric values
/// @param timestamps - Optional array of timestamps as milliseconds since epoch
/// @param modelType - Model type: "naive", "ses", "holt", "autoarima", "autoets", "autotheta", "sma5", etc.
/// @param horizon - Forecast horizon for each fold
/// @param initialWindow - Initial training window size (default: max(10, length/3))
/// @returns Object with `rmse`, `mae`, `mape`, `smape`, `folds`, `maeStd`, `rmseStd`
#[wasm_bindgen(js_name = crossValidate)]
pub fn cross_validate_js(
    values: &[f64],
    timestamps: Option<Vec<f64>>,
    model_type: &str,
    horizon: usize,
    initial_window: Option<usize>,
) -> Result<JsValue, JsError> {
    // Build time series
    let ts = match timestamps {
        Some(ref ts_ms) => {
            let inner_ts = TimeSeries::with_timestamps(values, ts_ms)?;
            inner_ts
        }
        None => TimeSeries::new(values)?,
    };

    let default_window = (values.len() / 3).max(10);
    let iw = initial_window.unwrap_or(default_window);

    let config = CVConfig::expanding(iw, horizon).with_step_size(horizon);

    let results = run_cv_for_model(&config, &ts, model_type)?;

    let js_result = CrossValidationResultJs {
        rmse: results.aggregated.rmse,
        mae: results.aggregated.mae,
        mape: results.aggregated.mape.unwrap_or(f64::NAN),
        smape: results.aggregated.smape,
        folds: results.n_folds,
        mae_std: results.aggregated.mae_std,
        rmse_std: results.aggregated.rmse_std,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Internal helper to run cross-validation for a specific model type.
fn run_cv_for_model(
    config: &CVConfig,
    ts: &TimeSeries,
    model_type: &str,
) -> Result<anofox_forecast::utils::cross_validation::CVResults, JsError> {
    match model_type.to_lowercase().as_str() {
        "naive" => cross_validate(config, ts.inner(), || Naive::new())
            .map_err(|e| JsError::new(&e.to_string())),
        "ses" | "simpleexponentialsmoothing" => {
            cross_validate(config, ts.inner(), || SimpleExponentialSmoothing::auto())
                .map_err(|e| JsError::new(&e.to_string()))
        }
        "holt" | "holtlineartrend" => {
            cross_validate(config, ts.inner(), || HoltLinearTrend::auto())
                .map_err(|e| JsError::new(&e.to_string()))
        }
        "autoarima" => cross_validate(config, ts.inner(), || AutoARIMA::new())
            .map_err(|e| JsError::new(&e.to_string())),
        "autoets" => cross_validate(config, ts.inner(), || AutoETS::new())
            .map_err(|e| JsError::new(&e.to_string())),
        "autotheta" => cross_validate(config, ts.inner(), || AutoTheta::new())
            .map_err(|e| JsError::new(&e.to_string())),
        s if s.starts_with("sma") => {
            let window: usize = s[3..].parse().unwrap_or(5);
            cross_validate(config, ts.inner(), || SimpleMovingAverage::new(window))
                .map_err(|e| JsError::new(&e.to_string()))
        }
        other => Err(JsError::new(&format!(
            "Unknown model type '{}'. Use: naive, ses, holt, autoarima, autoets, autotheta, sma<N>",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    fn make_test_values(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect()
    }

    #[wasm_bindgen_test]
    fn test_cross_validate_naive() {
        let values = make_test_values(60);
        let result = cross_validate_js(&values, None, "naive", 5, Some(20));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_cross_validate_ses() {
        let values = make_test_values(60);
        let result = cross_validate_js(&values, None, "ses", 5, Some(20));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_cross_validate_sma() {
        let values = make_test_values(60);
        let result = cross_validate_js(&values, None, "sma5", 5, Some(20));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_cross_validate_default_window() {
        let values = make_test_values(60);
        let result = cross_validate_js(&values, None, "naive", 5, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_cross_validate_invalid_model() {
        let values = make_test_values(60);
        let result = cross_validate_js(&values, None, "nonexistent", 5, Some(20));
        assert!(result.is_err());
    }
}
