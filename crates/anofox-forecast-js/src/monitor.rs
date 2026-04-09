//! Sequential monitoring bindings for JavaScript.
//!
//! Exposes the [`SequentialDetector`](anofox_forecast::monitor::SequentialDetector)
//! API for online changepoint detection on forecast errors. The detector's
//! state is round-tripped through JS as a plain object, so callers can keep
//! it in a `let state = ...` variable and pass it back to
//! [`updateForecastMonitor`] each time new errors arrive.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use anofox_forecast::monitor::{
    CriticalValue as InnerCriticalValue, Detector as InnerDetector,
    ForecastErrorType as InnerErrorType, SequentialConfig as InnerConfig,
    SequentialDetector as InnerDetector_, StreamState as InnerStream,
};

// ---------------------------------------------------------------------------
// JS-side mirror types (camelCase for ergonomic JS interop)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct StreamStateJs {
    train_mean: f64,
    sigma2: f64,
    crit_value: f64,
    cusum_a: f64,
    cusum_b: f64,
    cusum: Vec<f64>,
    threshold: Vec<f64>,
    tau: Option<usize>,
}

impl From<&InnerStream> for StreamStateJs {
    fn from(s: &InnerStream) -> Self {
        Self {
            train_mean: s.train_mean,
            sigma2: s.sigma2,
            crit_value: s.crit_value,
            cusum_a: s.cusum_a,
            cusum_b: s.cusum_b,
            cusum: s.cusum.clone(),
            threshold: s.threshold.clone(),
            tau: s.tau,
        }
    }
}

impl From<StreamStateJs> for InnerStream {
    fn from(s: StreamStateJs) -> Self {
        Self {
            train_mean: s.train_mean,
            sigma2: s.sigma2,
            crit_value: s.crit_value,
            cusum_a: s.cusum_a,
            cusum_b: s.cusum_b,
            cusum: s.cusum,
            threshold: s.threshold,
            tau: s.tau,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ConfigJs {
    m: usize,
    detector: String,
    error_type: String,
    gamma: f64,
    alpha: f64,
    sigma2_override: Option<f64>,
    crit_value_override: Option<f64>,
}

impl ConfigJs {
    fn from_inner(c: &InnerConfig) -> Self {
        Self {
            m: c.m,
            detector: detector_to_str(c.detector).to_string(),
            error_type: error_type_to_str(c.error_type).to_string(),
            gamma: c.gamma,
            alpha: c.alpha,
            sigma2_override: c.sigma2,
            crit_value_override: match c.critical_value {
                InnerCriticalValue::Fixed(v) => Some(v),
                _ => None,
            },
        }
    }

    fn into_inner(self) -> Result<InnerConfig, JsError> {
        let detector = parse_detector(&self.detector)?;
        let error_type = parse_error_type(&self.error_type)?;
        let mut cfg = InnerConfig::new(self.m)
            .detector(detector)
            .error_type(error_type)
            .gamma(self.gamma)
            .alpha(self.alpha);
        if let Some(s2) = self.sigma2_override {
            cfg = cfg.with_sigma2(s2);
        }
        if let Some(cv) = self.crit_value_override {
            cfg = cfg.critical_value(InnerCriticalValue::Fixed(cv));
        }
        Ok(cfg)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DetectorJs {
    config: ConfigJs,
    train_mean_raw: f64,
    raw: Option<StreamStateJs>,
    squared: Option<StreamStateJs>,
}

impl DetectorJs {
    fn from_inner(d: &InnerDetector_) -> Self {
        Self {
            config: ConfigJs::from_inner(&d.config),
            train_mean_raw: d.train_mean_raw,
            raw: d.raw.as_ref().map(StreamStateJs::from),
            squared: d.squared.as_ref().map(StreamStateJs::from),
        }
    }

    fn into_inner(self) -> Result<InnerDetector_, JsError> {
        Ok(InnerDetector_ {
            config: self.config.into_inner()?,
            train_mean_raw: self.train_mean_raw,
            raw: self.raw.map(InnerStream::from),
            squared: self.squared.map(InnerStream::from),
        })
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_detector(s: &str) -> Result<InnerDetector, JsError> {
    match s {
        "pageCusum" | "PageCusum" | "PageCUSUM" => Ok(InnerDetector::PageCusum),
        "pageCusum1" | "PageCusum1" | "PageCUSUM1" => Ok(InnerDetector::PageCusum1),
        "cusum" | "Cusum" | "CUSUM" => Ok(InnerDetector::Cusum),
        "cusum1" | "Cusum1" | "CUSUM1" => Ok(InnerDetector::Cusum1),
        other => Err(JsError::new(&format!(
            "Unknown detector '{}'. Use: pageCusum, pageCusum1, cusum, cusum1",
            other
        ))),
    }
}

fn detector_to_str(d: InnerDetector) -> &'static str {
    match d {
        InnerDetector::PageCusum => "pageCusum",
        InnerDetector::PageCusum1 => "pageCusum1",
        InnerDetector::Cusum => "cusum",
        InnerDetector::Cusum1 => "cusum1",
    }
}

fn parse_error_type(s: &str) -> Result<InnerErrorType, JsError> {
    match s {
        "raw" | "Raw" => Ok(InnerErrorType::Raw),
        "squared" | "Squared" => Ok(InnerErrorType::Squared),
        "both" | "Both" => Ok(InnerErrorType::Both),
        other => Err(JsError::new(&format!(
            "Unknown errorType '{}'. Use: raw, squared, both",
            other
        ))),
    }
}

fn error_type_to_str(e: InnerErrorType) -> &'static str {
    match e {
        InnerErrorType::Raw => "raw",
        InnerErrorType::Squared => "squared",
        InnerErrorType::Both => "both",
    }
}

// ---------------------------------------------------------------------------
// Public WASM functions
// ---------------------------------------------------------------------------

/// Run sequential CUSUM monitoring on a vector of forecast errors.
///
/// Returns an object containing the full detector state. Pass it back to
/// [`updateForecastMonitor`] (alongside any new errors) to continue
/// monitoring without recomputing the training window.
///
/// @param errors - Vector of forecast errors (residuals)
/// @param m - Length of training window (must satisfy 2 ≤ m < errors.length)
/// @param detector - Detector variant: `"pageCusum"` (default), `"pageCusum1"`, `"cusum"`, `"cusum1"`
/// @param errorType - Error transformation: `"both"` (default), `"raw"`, `"squared"`
/// @param gamma - Weight tuning parameter, `0 ≤ γ < 0.5` (default: `0`)
/// @param alpha - Nominal type-I error rate (default: `0.05`)
/// @param sigma2 - Optional override for the training-window variance
/// @param critValue - Optional fixed critical value (skips lookup/simulation)
/// @returns Detector state object with `tau`, `tauSquared`, `cusum`, `threshold`, etc.
#[wasm_bindgen(js_name = monitorForecastErrors)]
#[allow(clippy::too_many_arguments)]
pub fn monitor_forecast_errors(
    errors: &[f64],
    m: usize,
    detector: Option<String>,
    error_type: Option<String>,
    gamma: Option<f64>,
    alpha: Option<f64>,
    sigma2: Option<f64>,
    crit_value: Option<f64>,
) -> Result<JsValue, JsError> {
    let det = detector
        .as_deref()
        .map(parse_detector)
        .transpose()?
        .unwrap_or(InnerDetector::PageCusum);
    let etype = error_type
        .as_deref()
        .map(parse_error_type)
        .transpose()?
        .unwrap_or(InnerErrorType::Both);

    let mut cfg = InnerConfig::new(m)
        .detector(det)
        .error_type(etype)
        .gamma(gamma.unwrap_or(0.0))
        .alpha(alpha.unwrap_or(0.05));
    if let Some(s2) = sigma2 {
        cfg = cfg.with_sigma2(s2);
    }
    if let Some(cv) = crit_value {
        cfg = cfg.critical_value(InnerCriticalValue::Fixed(cv));
    }

    let detector = InnerDetector_::fit(errors, cfg).map_err(|e| JsError::new(&e.to_string()))?;
    let js_detector = DetectorJs::from_inner(&detector);
    serde_wasm_bindgen::to_value(&js_detector).map_err(|e| JsError::new(&e.to_string()))
}

/// Continue monitoring with new errors using a previously returned state.
///
/// Pass the `state` object exactly as returned from
/// [`monitorForecastErrors`] (or a previous `updateForecastMonitor` call).
/// The new errors are processed in order, the CUSUM stream is extended, and
/// the updated state is returned.
///
/// @param state - Detector state from a previous call
/// @param newErrors - Additional forecast errors to monitor
/// @returns Updated detector state object
#[wasm_bindgen(js_name = updateForecastMonitor)]
pub fn update_forecast_monitor(state: JsValue, new_errors: &[f64]) -> Result<JsValue, JsError> {
    let js_state: DetectorJs =
        serde_wasm_bindgen::from_value(state).map_err(|e| JsError::new(&e.to_string()))?;
    let mut detector = js_state.into_inner()?;
    detector
        .update(new_errors)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let js_detector = DetectorJs::from_inner(&detector);
    serde_wasm_bindgen::to_value(&js_detector).map_err(|e| JsError::new(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests (run via `wasm-pack test --node`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn monitor_basic_no_change() {
        let errors: Vec<f64> = (0..400).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let result = monitor_forecast_errors(&errors, 200, None, None, None, None, None, None);
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn monitor_with_invalid_detector() {
        let errors = vec![0.0; 100];
        let result = monitor_forecast_errors(
            &errors,
            50,
            Some("nonsense".to_string()),
            None,
            None,
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn round_trip_state_via_update() {
        let errors: Vec<f64> = (0..200).map(|i| (i as f64 * 0.01).sin()).collect();
        let state = monitor_forecast_errors(
            &errors,
            100,
            None,
            Some("raw".to_string()),
            None,
            None,
            None,
            Some(2.5),
        )
        .unwrap();

        // Feed in 20 more errors via update.
        let new_errors = vec![0.05; 20];
        let updated = update_forecast_monitor(state, &new_errors);
        assert!(updated.is_ok());
    }
}
