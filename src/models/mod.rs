//! Forecasting models.

pub mod explain;
pub mod inspect;
mod traits;

pub mod arima;
pub mod auto_forecast;
pub mod baseline;
pub mod convenience;
pub mod cv_select;
pub mod ensemble;
pub mod exponential;
pub mod garch;
pub mod intermittent;
pub mod kalman;
pub mod kalman_forecaster;
#[cfg(feature = "distributional")]
pub mod laplace;
pub mod mfles;
pub mod mstl_forecaster;
pub mod regression;
pub mod smart;
pub mod tbats;
pub mod theta;
pub mod var;
pub mod var_forecaster;

pub use cv_select::{Candidate, CvSelectForecaster};
pub use garch::GARCH;
#[cfg(feature = "distributional")]
pub use inspect::LaplaceExplanation;
pub use inspect::{
    ArimaExplanation, EtsExplanation, Explanation, Inspectable, MflesExplanation, MstlExplanation,
    RegressionExplanation, TbatsExplanation, ThetaExplanation,
};
pub use kalman_forecaster::KalmanForecaster;
#[cfg(feature = "distributional")]
pub use laplace::{DistributionalForecaster, Gaussian, GaussianMixture, LaplaceForecaster};
pub use mfles::MFLES;
pub use mstl_forecaster::{MSTLForecaster, SeasonalForecastMethod, TrendForecastMethod};
pub use smart::{SelectedFamily, SmartForecaster};
pub use tbats::{AutoTBATS, TBATS};
pub use traits::{
    validate_series_complete, BoxedForecaster, FittedParams, Forecaster, ModelRegistry, ModelSpec,
};
pub use var::VAR;
pub use var_forecaster::VARForecaster;
