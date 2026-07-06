//! Distributional forecasting shell (alpha, `distributional` feature).
//!
//! Inspired by [`microprediction/skaters`](https://github.com/microprediction/skaters):
//! streaming leaves, per-observation likelihood-weighted mixture, per-horizon
//! [`GaussianMixture`] output. Default leaf set: EMA, drift, AR(1);
//! [`LaplaceForecaster::with_holt`], [`LaplaceForecaster::with_ar2`], and
//! [`LaplaceForecaster::with_seasonal`] each add an opt-in leaf. The full
//! skaters ensemble (OU, fractional-differencing, Yeo-Johnson, CRPS-tuned
//! terminal leaf) is deferred.
//!
//! Attribution: skaters is MIT-licensed. See `THIRD_PARTY_NOTICES.md` at
//! the repository root for full attribution and license text.
//!
//! # Example
//! ```ignore
//! use anofox_forecast::models::laplace::{LaplaceForecaster, DistributionalForecaster};
//! use anofox_forecast::models::Forecaster;
//!
//! let mut f = LaplaceForecaster::new();
//! f.fit(&ts)?;
//! let dists = f.forecast_dist(5)?;                  // Vec<GaussianMixture>
//! let intervals = f.predict_with_intervals(5, 0.9)?; // point + P05/P95
//! ```
//!
//! The distributional surface is exposed via the [`DistributionalForecaster`]
//! trait; the point-forecast surface via the existing
//! [`Forecaster`](crate::models::Forecaster) trait. The mixture parameters
//! are also reachable via [`Explanation::Laplace`](crate::models::Explanation).

pub mod dist;
pub mod ensemble;
pub mod forecaster;
pub mod leaf;
pub mod leaves;

pub use dist::{Gaussian, GaussianMixture};
pub use forecaster::LaplaceForecaster;
pub use leaf::Leaf;

use crate::error::Result;

/// Trait for models that emit per-horizon predictive densities.
///
/// Sibling to [`Forecaster`](crate::models::Forecaster) — implementers must
/// also implement `Forecaster` (point forecast = mixture mean). Object-safe:
/// `Box<dyn DistributionalForecaster>` works.
pub trait DistributionalForecaster: crate::models::Forecaster {
    /// Predictive `GaussianMixture` for each step in `1..=horizon`.
    ///
    /// # Errors
    /// - `FitRequired` if the model has not been fit.
    fn forecast_dist(&self, horizon: usize) -> Result<Vec<GaussianMixture>>;
}
