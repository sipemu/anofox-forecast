//! Utility functions for forecasting models.

pub mod cross_validation;
pub mod metrics;
pub mod optimization;
pub mod stats;

pub use cross_validation::{cross_validate, CVConfig, CVResults, CVStrategy};
pub use metrics::{calculate_metrics, AccuracyMetrics};
pub use optimization::{nelder_mead, NelderMeadConfig, NelderMeadResult};
pub use stats::quantile_normal;
