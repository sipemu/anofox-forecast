//! ARIMA (Autoregressive Integrated Moving Average) models.
//!
//! This module provides:
//! - ARIMA models with various (p, d, q) specifications
//! - AutoARIMA for automatic order selection

mod arima;
mod auto_arima;
mod diff;

pub use arima::ARIMA;
pub use auto_arima::AutoARIMA;
pub use diff::{difference, integrate, seasonal_difference};
