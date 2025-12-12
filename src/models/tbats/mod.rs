//! TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal).
//!
//! TBATS is a powerful forecasting model for complex seasonal patterns:
//! - Multiple seasonal periods (e.g., daily + weekly patterns)
//! - Non-integer seasonality
//! - Box-Cox variance stabilization
//! - ARMA errors for residual autocorrelation
//!
//! Reference: De Livera, Hyndman & Snyder (2011) "Forecasting time series with
//! complex seasonal patterns using exponential smoothing"

mod model;
mod auto;

pub use model::TBATS;
pub use auto::AutoTBATS;
