//! Ensemble forecasting methods.
//!
//! Combines multiple forecasting models for improved accuracy.

mod ensemble;

pub use ensemble::{CombinationMethod, Ensemble};
