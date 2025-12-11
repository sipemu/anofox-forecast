//! Forecasting models.

mod traits;

pub mod arima;
pub mod baseline;
pub mod ensemble;
pub mod exponential;
pub mod intermittent;
pub mod theta;

pub use traits::Forecaster;
