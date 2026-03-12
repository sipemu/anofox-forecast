//! Bootstrap forecast interval wrappers for JavaScript.
//!
//! Provides wasm-bindgen wrappers for residual bootstrap forecast intervals.

use crate::time_series::{Forecast, TimeSeries};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::baseline::{Naive, SimpleMovingAverage};
use anofox_forecast::models::exponential::{AutoETS, HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster as ForecasterTrait;
use anofox_forecast::utils::bootstrap::{bootstrap_forecast as inner_bootstrap, BootstrapConfig};

/// Result of bootstrap forecast, serialized to JavaScript.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BootstrapResultJs {
    /// Point forecast values.
    point: Vec<f64>,
    /// Lower prediction interval bounds.
    lower: Vec<f64>,
    /// Upper prediction interval bounds.
    upper: Vec<f64>,
    /// Confidence level used.
    level: f64,
    /// Number of bootstrap samples used.
    n_samples: usize,
}

/// Generate a bootstrap forecast with empirical prediction intervals.
///
/// Uses residual bootstrap: resamples fitted residuals, generates synthetic
/// series, re-fits the model, and collects forecast distributions to
/// compute confidence intervals.
///
/// @param values - Array of numeric values
/// @param timestamps - Optional array of timestamps as milliseconds since epoch
/// @param modelType - Model type: "naive", "ses", "holt", "autoarima", "autoets", "autotheta", "sma5", etc.
/// @param horizon - Number of steps to forecast
/// @param level - Confidence level (e.g., 0.95 for 95% intervals)
/// @param nSamples - Number of bootstrap samples (default: 200)
/// @returns Object with `point`, `lower`, `upper`, `level`, `nSamples`
#[wasm_bindgen(js_name = bootstrapForecast)]
pub fn bootstrap_forecast_js(
    values: &[f64],
    timestamps: Option<Vec<f64>>,
    model_type: &str,
    horizon: usize,
    level: f64,
    n_samples: Option<usize>,
) -> Result<JsValue, JsError> {
    let ts = match timestamps {
        Some(ref ts_ms) => TimeSeries::with_timestamps(values, ts_ms)?,
        None => TimeSeries::new(values)?,
    };

    let config = BootstrapConfig::new(n_samples.unwrap_or(200)).with_seed(42);

    let forecast = run_bootstrap_for_model(&ts, model_type, horizon, level, &config)?;

    let point = forecast.values();
    let lower = forecast.lower().unwrap_or_else(|| point.clone());
    let upper = forecast.upper().unwrap_or_else(|| point.clone());

    let js_result = BootstrapResultJs {
        point,
        lower,
        upper,
        level,
        n_samples: config.n_samples,
    };

    serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
}

/// Internal helper to run bootstrap for a specific model type.
fn run_bootstrap_for_model(
    ts: &TimeSeries,
    model_type: &str,
    horizon: usize,
    level: f64,
    config: &BootstrapConfig,
) -> Result<Forecast, JsError> {
    match model_type.to_lowercase().as_str() {
        "naive" => run_bootstrap_typed::<Naive>(Naive::new(), ts, horizon, level, config),
        "ses" | "simpleexponentialsmoothing" => run_bootstrap_typed::<SimpleExponentialSmoothing>(
            SimpleExponentialSmoothing::auto(),
            ts,
            horizon,
            level,
            config,
        ),
        "holt" | "holtlineartrend" => run_bootstrap_typed::<HoltLinearTrend>(
            HoltLinearTrend::auto(),
            ts,
            horizon,
            level,
            config,
        ),
        "autoarima" => {
            run_bootstrap_typed::<AutoARIMA>(AutoARIMA::new(), ts, horizon, level, config)
        }
        "autoets" => run_bootstrap_typed::<AutoETS>(AutoETS::new(), ts, horizon, level, config),
        "autotheta" => {
            run_bootstrap_typed::<AutoTheta>(AutoTheta::new(), ts, horizon, level, config)
        }
        s if s.starts_with("sma") => {
            let window: usize = s[3..].parse().unwrap_or(5);
            run_bootstrap_typed::<SimpleMovingAverage>(
                SimpleMovingAverage::new(window),
                ts,
                horizon,
                level,
                config,
            )
        }
        other => Err(JsError::new(&format!(
            "Unknown model type '{}'. Use: naive, ses, holt, autoarima, autoets, autotheta, sma<N>",
            other
        ))),
    }
}

/// Run bootstrap forecast for a typed model.
fn run_bootstrap_typed<M>(
    mut model: M,
    ts: &TimeSeries,
    horizon: usize,
    level: f64,
    config: &BootstrapConfig,
) -> Result<Forecast, JsError>
where
    M: ForecasterTrait + Clone + Send + Sync,
{
    model
        .fit(ts.inner())
        .map_err(|e| JsError::new(&e.to_string()))?;

    let forecast = inner_bootstrap(&model, ts.inner(), horizon, level, config)
        .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(Forecast::from_inner(forecast))
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
    fn test_bootstrap_naive() {
        let values = make_test_values(50);
        let result = bootstrap_forecast_js(&values, None, "naive", 5, 0.95, Some(50));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_bootstrap_ses() {
        let values = make_test_values(50);
        let result = bootstrap_forecast_js(&values, None, "ses", 5, 0.95, Some(50));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_bootstrap_sma() {
        let values = make_test_values(50);
        let result = bootstrap_forecast_js(&values, None, "sma5", 5, 0.90, Some(50));
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_bootstrap_default_samples() {
        let values = make_test_values(50);
        let result = bootstrap_forecast_js(&values, None, "naive", 5, 0.95, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_bootstrap_invalid_model() {
        let values = make_test_values(50);
        let result = bootstrap_forecast_js(&values, None, "nonexistent", 5, 0.95, Some(50));
        assert!(result.is_err());
    }
}
