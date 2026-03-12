//! Ensemble forecasting methods.
//!
//! Combines multiple forecasting models for improved accuracy.

pub mod auto;
mod model;

pub use auto::{AutoEnsemble, AutoEnsembleConfig};
pub use model::{CombinationMethod, Ensemble};
