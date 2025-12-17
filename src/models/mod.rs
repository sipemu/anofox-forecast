//! Forecasting models.

mod traits;

pub mod arima;
pub mod baseline;
pub mod ensemble;
pub mod exponential;
pub mod garch;
pub mod intermittent;
pub mod mfles;
pub mod mstl_forecaster;
pub mod tbats;
pub mod theta;

pub use garch::GARCH;
pub use mfles::MFLES;
pub use mstl_forecaster::{MSTLForecaster, SeasonalForecastMethod, TrendForecastMethod};
pub use tbats::{AutoTBATS, TBATS};
pub use traits::{BoxedForecaster, Forecaster, ModelRegistry, ModelSpec};
