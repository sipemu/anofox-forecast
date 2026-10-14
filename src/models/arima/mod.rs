//! ARIMA and SARIMA (Autoregressive Integrated Moving Average) models.
//!
//! This module provides:
//! - ARIMA models with various (p, d, q) specifications
//! - SARIMA models with seasonal components (P, D, Q)\[s\]
//! - AutoARIMA for automatic order selection

mod auto_arima;
mod diff;
mod exact_likelihood;
mod model;
mod r_arima;

pub use auto_arima::{AutoARIMA, AutoARIMAConfig, ModelOrder};
pub use diff::{
    difference, find_min_fractional_d, fractional_difference, fractional_weights, integrate,
    ndiffs_kpss, nsdiffs_seas, seasonal_difference, seasonal_integrate, suggest_differencing,
};
pub use model::{
    fit_exact_ml_for_tests, r_arima_aicc_for_tests, ARIMASpec, SARIMASpec, ARIMA, SARIMA,
};
