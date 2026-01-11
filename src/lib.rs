//! # anofox-forecast
//!
//! Time series forecasting library for Rust.
//!
//! Provides 35+ forecasting models including ARIMA, ETS, Theta,
//! and baseline methods, along with seasonality decomposition (STL/MSTL),
//! changepoint detection, and outlier detection.
//!
//! For comprehensive periodicity detection, see the
//! [fdars](https://crates.io/crates/fdars-core) crate.

// Allow some clippy warnings for cleaner code in specific cases
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]

// Prevent use of parallel feature on WASM targets (rayon requires OS threads)
#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
compile_error!(
    "The 'parallel' feature is not supported on WASM targets. Build without --features parallel"
);

pub mod changepoint;
pub mod core;
pub mod detection;
pub mod error;
pub mod features;
pub mod models;
pub mod postprocess;
pub mod seasonality;
pub mod simd;
pub mod transform;
pub mod utils;
pub mod validation;

pub use error::{ForecastError, Result};

pub mod prelude {
    pub use crate::core::{Forecast, TimeSeries};
    pub use crate::error::{ForecastError, Result};
    pub use crate::models::Forecaster;
    pub use crate::utils::{calculate_metrics, quantile_normal, AccuracyMetrics};
}
