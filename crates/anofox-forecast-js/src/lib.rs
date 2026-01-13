//! WebAssembly bindings for anofox-forecast.
//!
//! This crate provides JavaScript/TypeScript bindings for the anofox-forecast
//! time series forecasting library using wasm-bindgen.

use wasm_bindgen::prelude::*;

pub mod forecaster;
pub mod time_series;

pub use forecaster::*;
pub use time_series::*;

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
