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
pub use comparison::{
    compare_models, compare_registry, ComparisonConfig, ComparisonResult, ComparisonTable,
};
pub use cross_validation::{
    cross_validate, grouped_cross_validate, rolling_forecast, train_test_split,
    train_test_split_at, AggregatedMetrics, CVConfig, CVResults, CVStrategy, ConstantFill,
    CvFoldGenerator, FillStrategy, Fold, GroupedCVResults, LastValueFill, MeanFill, MedianFill,
    ModeFill, RollingForecastConfig, RollingForecastResult, RollingForecastWindow, ZeroFill,
};
pub use metrics::{
    bias, calculate_metrics, coverage, mda, msis, periods_in_stock, skill_score, theils_u1,
    theils_u2, wape, AccuracyMetrics, ForecastMetrics,
};
pub use ols::{ols_fit, ols_residuals, OLSResult};
pub use optimization::{nelder_mead, NelderMeadConfig, NelderMeadResult};
pub use stats::quantile_normal;
