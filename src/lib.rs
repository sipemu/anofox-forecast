//! # anofox-forecast
//!
//! Time series forecasting library for Rust.
//!
//! Provides 35+ forecasting models including ARIMA, ETS, Theta,
//! and baseline methods, along with automatic model selection (`AutoForecast`,
//! `AutoEnsemble`), seasonality decomposition (STL/MSTL), changepoint detection,
//! outlier detection, and model serialization.
//!
//! Cross-validation is available both as a Rust-native API ([`utils::cross_validate`])
//! and as a DuckDB extension ([forecast-extension](https://github.com/DataZooDE/forecast-extension))
//! for multi-series datasets at scale.
//!
//! For comprehensive periodicity detection, see the
//! [fdars](https://crates.io/crates/fdars-core) crate.

// Allow some clippy warnings for cleaner code in specific cases
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_is_multiple_of)] // is_multiple_of is unstable on WASM

// Prevent use of parallel feature on WASM targets (rayon requires OS threads)
#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
compile_error!(
    "The 'parallel' feature is not supported on WASM targets. Build without --features parallel"
);

pub mod batch;
pub mod changepoint;
pub mod core;
pub mod detection;
pub mod error;
pub mod features;
pub mod hierarchy;
pub mod models;
pub mod monitor;
#[cfg(feature = "postprocess")]
pub mod postprocess;
pub mod seasonality;
pub mod simd;
pub mod transform;
pub mod utils;
pub mod validation;

pub use error::{ForecastError, Result};

pub mod prelude {
    pub use crate::core::{ConstrainedForecast, Forecast, ForecastConstraint, TimeSeries};
    pub use crate::error::{ForecastError, Result};
    pub use crate::models::Forecaster;
    pub use crate::utils::{
        bootstrap_forecast, calculate_metrics, cross_validate, quantile_normal, AccuracyMetrics,
        BootstrapConfig, CVConfig, CVResults,
    };
    pub use crate::validation::{diagnose_residuals, ResidualDiagnostics};

    #[cfg(feature = "postprocess")]
    pub use crate::postprocess::{
        ConformalMethod, ConformalPredictor, ConformalResult, HistoricalSimResult,
        HistoricalSimulator, IDRPredictor, IDRResult, NormalPredictor, NormalResult,
        PointForecasts, PostProcessor, PredictionIntervals, QRAPredictor, QuantileForecasts,
    };
}
