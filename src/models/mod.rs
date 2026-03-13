//! Forecasting models.

pub mod explain;
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
pub mod kalman_forecaster;
pub mod mfles;
pub mod mstl_forecaster;
pub mod tbats;
pub mod theta;
pub mod var;
pub mod var_forecaster;

pub use garch::GARCH;
pub use kalman_forecaster::KalmanForecaster;
pub use mfles::MFLES;
pub use mstl_forecaster::{MSTLForecaster, SeasonalForecastMethod, TrendForecastMethod};
pub use tbats::{AutoTBATS, TBATS};
pub use traits::{validate_series_complete, BoxedForecaster, Forecaster, ModelRegistry, ModelSpec};
pub use var::VAR;
pub use var_forecaster::VARForecaster;
