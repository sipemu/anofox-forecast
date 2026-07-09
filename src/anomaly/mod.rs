//! Streaming multivariate anomaly detection on the prediction parade.
//!
//! Port of [microprediction/timemachines' `mahalanobis`](
//! https://github.com/microprediction/timemachines/blob/main/src/timemachines/heads/mahalanobis.py)
//! head. Wraps any distributional forecaster and emits, per tick,
//! a Mahalanobis distance and calibrated p-value of the forecast
//! surprise vector.
//!
//! ## Layers
//!
//! 1. [`chi2`], [`gpd`], [`linalg`], [`quantile`] — numerical primitives.
//! 2. `parade` (Phase 2) — PIT + z-vector bookkeeping on top of a base
//!    distributional forecaster.
//! 3. `mahalanobis` (Phase 3) — running μ / Σ of z, Satterthwaite bulk +
//!    GPD tail p-value.
//! 4. `zbank` (Phase 4, optional) — bank of engines at different
//!    `(scale_alpha, stride)` gridpoints.
//!
//! See `docs/ANOMALY_PLAN.md` for the full implementation plan and the
//! reference algorithm's derivation.

pub mod chi2;
pub mod gpd;
pub mod linalg;
pub mod parade;
pub mod quantile;

pub use parade::Parade;
