//! ARIMA and SARIMA (Autoregressive Integrated Moving Average) models.
//!
//! This module provides:
//! - ARIMA models with various (p, d, q) specifications
//! - SARIMA models with seasonal components (P, D, Q)\[s\]
//! - AutoARIMA for automatic order selection

mod auto_arima;
mod diff;
mod model;

pub use auto_arima::{AutoARIMA, AutoARIMAConfig, ModelOrder};
pub use diff::{difference, integrate, seasonal_difference};
pub use model::{ARIMASpec, SARIMASpec, ARIMA, SARIMA};
