//! Ensemble forecasting methods.
//!
//! Combines multiple forecasting models for improved accuracy.

mod model;

pub use model::{CombinationMethod, Ensemble};
