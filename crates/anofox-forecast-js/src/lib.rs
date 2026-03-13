//! WebAssembly bindings for anofox-forecast.
//!
//! This crate provides JavaScript/TypeScript bindings for the anofox-forecast
//! time series forecasting library using wasm-bindgen.

use wasm_bindgen::prelude::*;

pub mod auto_models;
pub mod bootstrap;
pub mod calendar;
pub mod changepoint;
pub mod cross_validation;
pub mod decomposition;
pub mod features;
pub mod forecaster;
pub mod postprocess;
pub mod time_series;
pub mod validation;

// Re-export all public items for flat WASM module access.
pub use auto_models::{AutoEnsembleForecaster, AutoForecastBuilder, AutoForecaster};
pub use bootstrap::bootstrap_forecast_js;
pub use calendar::*;
pub use changepoint::{detect_changepoints, detect_changepoints_bic};
pub use cross_validation::cross_validate_js;
pub use decomposition::{mstl_decompose, stl_decompose};
pub use features::*;
pub use forecaster::*;
pub use postprocess::*;
pub use time_series::*;
pub use validation::*;

/// Initialize the WASM module.
///
/// This function is called automatically when the module is loaded.
#[wasm_bindgen(start)]
pub fn init() {
    // Future: Set up panic hook for better error messages
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
