//! Incremental model updates and compact, serializable model state.
//!
//! [`IncrementalForecaster`] extends [`Forecaster`] for models whose fitted
//! state can be filtered forward with new observations at fixed parameters,
//! instead of refitting on the full history:
//!
//! - [`IncrementalForecaster::update`] runs the model's state recursion over
//!   the new observations only (O(k) for k new points). Smoothing parameters
//!   are **not** re-estimated; deciding when a full refit is due (e.g. every
//!   N updates) is left to the caller.
//! - [`IncrementalForecaster::state`] / [`IncrementalForecaster::from_state`]
//!   export and restore a [`ModelState`]: parameters, the final state vector
//!   and the running error sums needed for prediction intervals. Its size is
//!   O(seasonal period), independent of the history length, and it is
//!   serde-serializable with the `serde` feature.
//!
//! Updating a model fitted on `n` points with `k` more yields exactly the
//! state, point forecasts and intervals of fitting the first `n + k` points
//! with the same (fixed) parameters and initial states.
//!
//! A restored model can predict and be updated further, but holds no fitted
//! values, residuals or training history (`fitted_values()` returns `None`).
//!
//! Supported models: [`SimpleExponentialSmoothing`], [`HoltLinearTrend`],
//! [`HoltWinters`], [`ETS`] and [`AutoETS`] (which exports the state of its
//! selected [`ETS`] model).
//!
//! [`SimpleExponentialSmoothing`]: crate::models::exponential::SimpleExponentialSmoothing
//! [`HoltLinearTrend`]: crate::models::exponential::HoltLinearTrend
//! [`HoltWinters`]: crate::models::exponential::HoltWinters
//! [`ETS`]: crate::models::exponential::ETS
//! [`AutoETS`]: crate::models::exponential::AutoETS
//!
//! # Example
//! ```
//! use anofox_forecast::core::TimeSeries;
//! use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
//! use anofox_forecast::models::{Forecaster, IncrementalForecaster};
//! use chrono::{Duration, TimeZone, Utc};
//!
//! let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
//! let values = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];
//! let timestamps: Vec<_> = (0..values.len()).map(|i| base + Duration::days(i as i64)).collect();
//! let ts = TimeSeries::univariate(timestamps, values).unwrap();
//!
//! let mut model = SimpleExponentialSmoothing::auto();
//! model.fit(&ts).unwrap();
//!
//! // Store the compact state, restore it later and feed new observations.
//! let state = model.state().unwrap();
//! let mut restored = SimpleExponentialSmoothing::from_state(state).unwrap();
//! restored.update(&[14.0, 16.0]).unwrap();
//! let forecast = restored.predict(3).unwrap();
//! assert_eq!(forecast.horizon(), 3);
//! ```

use crate::error::{ForecastError, Result};
use crate::models::exponential::{
    ETSSpec, HoltLinearTrend, HoltWinters, SeasonalType, SimpleExponentialSmoothing, ETS,
};
use crate::models::Forecaster;

/// A forecaster that can be updated with new observations at fixed
/// parameters and exported to / restored from a compact [`ModelState`].
///
/// See the [module documentation](self) for the exact semantics.
pub trait IncrementalForecaster: Forecaster {
    /// Filter the fitted state forward over `new_obs` without re-estimating
    /// parameters.
    ///
    /// Returns [`ForecastError::FitRequired`] if the model has not been
    /// fitted (or restored), [`ForecastError::MissingValues`] if `new_obs`
    /// contains a non-finite value, and [`ForecastError::InvalidParameter`]
    /// if the model was fit with exogenous regressors (their future values
    /// are not part of an update). An empty slice is a no-op.
    fn update(&mut self, new_obs: &[f64]) -> Result<()>;

    /// Export the compact state of a fitted model.
    ///
    /// Returns [`ForecastError::FitRequired`] if the model has not been
    /// fitted or restored.
    fn state(&self) -> Result<ModelState>;

    /// Restore a model from a state exported by [`state`](Self::state).
    ///
    /// Returns [`ForecastError::InvalidParameter`] if the state belongs to a
    /// different model or is internally inconsistent (non-finite values,
    /// wrong seasonal-vector length).
    fn from_state(state: ModelState) -> Result<Self>
    where
        Self: Sized;
}

/// Compact state of a fitted [`IncrementalForecaster`].
///
/// Size is O(seasonal period) and independent of the training-history
/// length. Serializable with the `serde` feature (see
/// [`crate::utils::persistence`] for JSON / bincode helpers).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ModelState {
    /// [`SimpleExponentialSmoothing`] state.
    SimpleExponentialSmoothing(SesState),
    /// [`HoltLinearTrend`] state.
    HoltLinearTrend(HoltState),
    /// [`HoltWinters`] state.
    HoltWinters(HoltWintersState),
    /// [`ETS`] state (also exported by `AutoETS` for its selected model).
    ETS(EtsState),
}

impl ModelState {
    /// Name of the model this state belongs to (matches [`Forecaster::name`]).
    pub fn model_name(&self) -> &'static str {
        match self {
            ModelState::SimpleExponentialSmoothing(_) => "SimpleExponentialSmoothing",
            ModelState::HoltLinearTrend(_) => "HoltLinearTrend",
            ModelState::HoltWinters(_) => "HoltWinters",
            ModelState::ETS(_) => "ETS",
        }
    }

    /// Number of observations the state has absorbed (fit + updates).
    pub fn n_obs(&self) -> usize {
        match self {
            ModelState::SimpleExponentialSmoothing(s) => s.n,
            ModelState::HoltLinearTrend(s) => s.n,
            ModelState::HoltWinters(s) => s.n,
            ModelState::ETS(s) => s.n,
        }
    }

    /// Restore the model this state belongs to as a boxed
    /// [`IncrementalForecaster`], without knowing its concrete type.
    pub fn restore(self) -> Result<Box<dyn IncrementalForecaster>> {
        Ok(match self {
            ModelState::SimpleExponentialSmoothing(_) => {
                Box::new(SimpleExponentialSmoothing::from_state(self)?)
            }
            ModelState::HoltLinearTrend(_) => Box::new(HoltLinearTrend::from_state(self)?),
            ModelState::HoltWinters(_) => Box::new(HoltWinters::from_state(self)?),
            ModelState::ETS(_) => Box::new(ETS::from_state(self)?),
        })
    }
}

/// State of a fitted [`SimpleExponentialSmoothing`] model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SesState {
    /// Smoothing parameter.
    pub alpha: f64,
    /// Final level.
    pub level: f64,
    /// Observations absorbed.
    pub n: usize,
    /// Sum of squared one-step residuals (first observation excluded).
    pub sse: f64,
    /// Whether `alpha` is re-estimated on the next `fit()`.
    pub optimize: bool,
    /// Whether `alpha` was supplied as a warm start (affects the interval
    /// degrees of freedom, see `SimpleExponentialSmoothing::with_alpha`).
    pub warm_start: bool,
}

/// State of a fitted [`HoltLinearTrend`] model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HoltState {
    /// Level smoothing parameter.
    pub alpha: f64,
    /// Trend smoothing parameter.
    pub beta: f64,
    /// Damping parameter (`None` = undamped).
    pub phi: Option<f64>,
    /// Final level.
    pub level: f64,
    /// Final trend.
    pub trend: f64,
    /// Observations absorbed.
    pub n: usize,
    /// Sum of squared one-step residuals (first observation excluded).
    pub sse: f64,
    /// Whether the parameters are re-estimated on the next `fit()`.
    pub optimize: bool,
}

/// State of a fitted [`HoltWinters`] model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HoltWintersState {
    /// Level smoothing parameter.
    pub alpha: f64,
    /// Trend smoothing parameter.
    pub beta: f64,
    /// Seasonal smoothing parameter.
    pub gamma: f64,
    /// Seasonal period.
    pub seasonal_period: usize,
    /// Seasonal type.
    pub seasonal_type: SeasonalType,
    /// Final level.
    pub level: f64,
    /// Final trend.
    pub trend: f64,
    /// Final seasonal indices, indexed by `t % seasonal_period`.
    pub seasonals: Vec<f64>,
    /// Observations absorbed.
    pub n: usize,
    /// Sum of squared one-step residuals (first season excluded).
    pub sse: f64,
    /// Whether the parameters are re-estimated on the next `fit()`.
    pub optimize: bool,
}

/// State of a fitted [`ETS`] model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EtsState {
    /// Model specification.
    pub spec: ETSSpec,
    /// Seasonal period (1 for non-seasonal models).
    pub seasonal_period: usize,
    /// Level smoothing parameter.
    pub alpha: f64,
    /// Trend smoothing parameter.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter.
    pub gamma: Option<f64>,
    /// Damping parameter.
    pub phi: Option<f64>,
    /// Final level.
    pub level: f64,
    /// Final trend (0 for trendless models).
    pub trend: f64,
    /// Final seasonal states, indexed by `t % seasonal_period` (empty for
    /// non-seasonal models).
    pub seasonals: Vec<f64>,
    /// Observations absorbed.
    pub n: usize,
    /// Sum of squared raw residuals `y - forecast`.
    pub sse: f64,
    /// Sum of squared likelihood errors (relative errors for
    /// multiplicative-error models, raw residuals otherwise).
    pub sse_lik: f64,
    /// Sum of `ln|forecast|` (multiplicative-error likelihood term).
    pub sum_log_fc: f64,
    /// Whether the parameters are re-estimated on the next `fit()`.
    pub optimize: bool,
}

/// Validate the observations passed to [`IncrementalForecaster::update`].
pub(crate) fn validate_update_obs(new_obs: &[f64]) -> Result<()> {
    if new_obs.iter().any(|v| !v.is_finite()) {
        return Err(ForecastError::MissingValues);
    }
    Ok(())
}

/// Error for an `update()` on a model fit with exogenous regressors.
pub(crate) fn exog_update_error(model: &str) -> ForecastError {
    ForecastError::InvalidParameter(format!(
        "{model} was fit with exogenous regressors; update() cannot filter them forward, refit instead"
    ))
}

/// Error for a state that belongs to another model.
pub(crate) fn wrong_state_error(expected: &str, state: &ModelState) -> ForecastError {
    ForecastError::InvalidParameter(format!(
        "cannot restore {expected} from a {} state",
        state.model_name()
    ))
}

/// Check that every value of a restored state is finite.
pub(crate) fn check_finite_state(model: &str, values: &[f64]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ForecastError::InvalidParameter(format!(
            "{model} state contains non-finite values"
        )));
    }
    Ok(())
}
