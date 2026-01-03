//! Utility functions for forecasting models.

pub mod bootstrap;
pub mod cross_validation;
pub mod metrics;
pub mod ols;
pub mod optimization;
pub mod stats;

pub use bootstrap::{bootstrap_forecast, bootstrap_intervals, BootstrapConfig, BootstrapResult};
pub use cross_validation::{cross_validate, CVConfig, CVResults, CVStrategy};
pub use metrics::{calculate_metrics, AccuracyMetrics};
pub use ols::{ols_fit, ols_residuals, OLSResult};
pub use optimization::{nelder_mead, NelderMeadConfig, NelderMeadResult};
pub use stats::quantile_normal;
