//! # anofox-forecast
//!
//! Time series forecasting library - Rust port of anofox-time C++.
//!
//! Provides 35+ forecasting models including ARIMA, ETS, Theta, TBATS,
//! and baseline methods, along with seasonality analysis, changepoint
//! detection, outlier detection, and clustering capabilities.

// Allow some clippy warnings for cleaner code in specific cases
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]

pub mod changepoint;
pub mod clustering;
pub mod core;
pub mod detection;
pub mod error;
pub mod features;
pub mod models;
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
