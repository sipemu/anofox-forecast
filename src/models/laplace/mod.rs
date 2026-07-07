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
//!
//! # Choosing a selector (empirical cross-panel guidance)
//!
//! The α-21/α-22 stack ships three zero-config selectors. Which one wins
//! depends on the panel type. Full benchmarks are in
//! `examples/skaters_m5_full_auto.rs`, `skaters_m4_daily_benchmark.rs`,
//! and `skaters_m3_monthly_benchmark.rs`.
//!
//! | panel | domain | best selector | vs. AutoETS median MAE gap |
//! |-------|--------|---------------|-----------------------------|
//! | M5 full 30k | retail counts (all intermittent) | [`LaplaceForecaster::auto_aid`] | **+0.8%**, 42× faster than AutoETS |
//! | M5 top-1000 | retail (non-intermittent only) | `Laplace + AR2 + S7 + FD + OU` | +2.9% |
//! | M4 daily | economic continuous | [`LaplaceForecaster::auto`] (or upstream `AutoTheta`) | +7.5% |
//! | M3 monthly | macroeconomic | [`LaplaceForecaster::auto`] (or upstream `AutoTheta`) | +6.2% |
//!
//! ## Rules of thumb
//!
//! - **Retail SKU / demand data (counts, intermittency)** → use
//!   [`LaplaceForecaster::new().auto_aid()`](LaplaceForecaster::auto_aid)
//!   or [`SmartForecaster`](crate::models::SmartForecaster). AID's
//!   distribution-family classification is designed for this segment;
//!   [Poisson](leaves::PoissonLeaf) / [Negative-Binomial](leaves::NegativeBinomialLeaf)
//!   / [seasonal-Croston](leaves::SeasonalIntermittentLeaf) are the right
//!   leaves.
//!
//! - **Economic / financial / continuous non-demand series** → use
//!   [`LaplaceForecaster::new().auto()`](LaplaceForecaster::auto) (no
//!   AID). On M3 monthly `auto_aid` **regresses ~7% median MAE vs plain
//!   auto** because AID picks distribution families (usually LogNormal)
//!   whose Gaussian moment-match doesn't fit smooth continuous data.
//!
//! - **Not sure which** → benchmark both on a held-out window of your
//!   own data before deciding. A single `.fit()` + `.predict()` per
//!   selector is a few milliseconds.
//!
//! **The [`SmartForecaster`](crate::models::SmartForecaster) route is
//! specifically demand-focused.** It commits to a single Laplace
//! distribution-family configuration based on AID's classification and
//! is not designed to be a general-purpose replacement for `AutoETS` /
//! `AutoTheta` on economic panels. On M3 monthly it regresses ~14% vs.
//! plain `auto()`.

pub mod dist;
pub mod ensemble;
pub mod forecaster;
pub mod global;
pub mod leaf;
pub mod leaves;

pub use dist::{Gaussian, GaussianMixture};
pub use forecaster::LaplaceForecaster;
pub use global::{GlobalLaplace, MetaLearnerScaffold};
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
