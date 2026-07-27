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
pub mod detection;
pub mod diagnostics;
pub mod features;
pub mod filters;
pub mod forecaster;
pub mod global_models;
pub mod hierarchy;
pub mod laplace_playground;
pub mod monitor;
pub mod pipeline;
pub mod postprocess;
pub mod regression;
pub mod time_series;
pub mod validation;

// Re-export all public items for flat WASM module access.
pub use auto_models::{AutoEnsembleForecaster, AutoForecastBuilder, AutoForecaster};
pub use bootstrap::bootstrap_forecast_js;
pub use calendar::*;
pub use changepoint::{auto_detect_changepoints, detect_changepoints, detect_changepoints_bic};
pub use cross_validation::{cross_validate_js, rolling_forecast_js};
pub use decomposition::{mstl_decompose, stl_decompose};
pub use detection::{
    detect_dominant_period, detect_outliers, detect_outliers_auto, welch_periodogram,
};
pub use diagnostics::{
    analyze_demand, intermittent_diagnostics, intermittent_diagnostics_with_forecast,
    intermittent_diagnostics_with_intervals,
};
pub use features::*;
pub use filters::{
    bk_filter, cf_filter, find_min_fractional_d, fractional_difference, hamilton_filter, hp_filter,
};
pub use forecaster::*;
pub use global_models::{global_auto_ets, global_croston, global_ets, global_theta};
pub use hierarchy::HierarchyTree;
pub use laplace_playground::LaplacePlayground;
pub use monitor::{monitor_forecast_errors, update_forecast_monitor};
pub use pipeline::{Pipeline, PipelineBuilder};
pub use postprocess::*;
pub use regression::{RegressionFeatures, RegressionForecaster};
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
