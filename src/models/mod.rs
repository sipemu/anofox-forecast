//! Forecasting models.

mod traits;

pub mod arima;
pub mod auto_forecast;
pub mod baseline;
pub mod batch;
pub mod ensemble;
pub mod exponential;
pub mod garch;
pub mod intermittent;
pub mod kalman;
pub mod mfles;
pub mod mstl_forecaster;
pub mod tbats;
pub mod theta;
pub mod var;

pub use garch::GARCH;
pub use mfles::MFLES;
pub use mstl_forecaster::{MSTLForecaster, SeasonalForecastMethod, TrendForecastMethod};
pub use tbats::{AutoTBATS, TBATS};
pub use traits::{validate_series_complete, BoxedForecaster, Forecaster, ModelRegistry, ModelSpec};
pub use var::VAR;
