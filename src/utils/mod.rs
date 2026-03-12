//! Utility functions for forecasting models.

pub mod bootstrap;
pub mod comparison;
pub mod cross_validation;
pub mod metrics;
pub mod ols;
pub mod optimization;
#[cfg(feature = "serde")]
pub mod persistence;
pub mod stats;

pub use bootstrap::{bootstrap_forecast, bootstrap_intervals, BootstrapConfig, BootstrapResult};
pub use cross_validation::{
    cross_validate, grouped_cross_validate, train_test_split, train_test_split_at,
    AggregatedMetrics, CVConfig, CVResults, CVStrategy, ConstantFill, CvFoldGenerator,
    FillStrategy, Fold, GroupedCVResults, LastValueFill, MeanFill, MedianFill, ModeFill, ZeroFill,
};
pub use metrics::{calculate_metrics, AccuracyMetrics};
pub use ols::{ols_fit, ols_residuals, OLSResult};
pub use optimization::{nelder_mead, NelderMeadConfig, NelderMeadResult};
pub use comparison::{compare_models, compare_registry, ComparisonConfig, ComparisonResult, ComparisonTable};
pub use stats::quantile_normal;
