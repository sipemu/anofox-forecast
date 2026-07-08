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
//! The stack ships three zero-config selectors. Which one wins depends
//! on the panel type. Full benchmark in `examples/fev_benchmark.rs`
//! covers 27 fev / Chronos-benchmark classical datasets; the M5-specific
//! benchmark in `examples/skaters_m5_full_auto.rs` covers full retail
//! demand.
//!
//! | panel | domain | best selector |
//! |-------|--------|---------------|
//! | M5 full 30k | retail counts (all intermittent) | [`LaplaceForecaster::auto_aid`] — median MAE within 0.8 % of AutoETS, ~42× faster |
//! | M4 hourly, tourism monthly | seasonal with adequate history | [`LaplaceForecaster::auto`] — competitive with classical |
//! | m1/m3/m4 yearly, tourism_yearly | short-history (N < 50) | `AutoTheta` — Laplace's streaming leaves haven't warmed up |
//! | M3 monthly, M4 daily | economic continuous, adequate history | [`LaplaceForecaster::auto`] — close to `AutoTheta`, +5-15 % gap |
//! | dominick, m5-style retail | retail SKU counts | [`LaplaceForecaster::auto_aid`] — Croston-family family selection |
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
//!
//! # Warmup requirement — training length matters
//!
//! `LaplaceForecaster` is a **streaming per-observation** design.
//! Each leaf (`EmaLeaf`, `Ar1Leaf`, `SeasonalEmaLeaf`, `HoltLeaf`, ...)
//! maintains state that converges as it processes observations. The leaf
//! softmax also needs several observations to reweight from uniform
//! toward the leaves that fit the series.
//!
//! **Consequence:** on short-history panels the leaf state and softmax
//! weights are still warming up when the forecast is requested. Classical
//! closed-form fitters (`AutoETS`, `AutoTheta`) that directly optimize
//! their parameters over the full training window do not share this
//! penalty and are consistently better on panels where `N < 60`.
//!
//! Empirical rule of thumb from the fev-benchmark evaluation
//! (`examples/fev_benchmark.rs`):
//!
//! | training length | Laplace vs. AutoTheta MASE gap |
//! |-----------------|---------------------------------|
//! | `N > 300` (m3_monthly, m4_daily, hospital, nn5_weekly, fred_md) | within ±5 %, competitive |
//! | `N = 100–300` (m4_weekly, tourism_monthly, cif_2016) | +5 to +25 % |
//! | `N = 50–100` (m4_hourly, m3_quarterly, tourism_quarterly) | +25 to +80 % |
//! | `N < 50` (m1_yearly, m3_yearly, tourism_yearly) | +15 to +30 %, cold start dominates |
//!
//! This is a **fundamental architectural property**, not a bug or a
//! configuration issue. If your data has short histories, prefer
//! `AutoTheta` / `AutoETS` (available directly in
//! `crate::models::exponential`, `crate::models::theta`). Reach for
//! `LaplaceForecaster` when either:
//!
//! - The series is long enough for the streaming leaves to converge
//!   (`N ≥ 100`, ideally ≥ 300); or
//! - You need the distributional output (mixture density, quantiles,
//!   per-h calibration) that classical forecasters don't provide; or
//! - You're on retail / demand data where AID's family classification
//!   provides the largest win (via `.auto_aid()` / `SmartForecaster`).

pub mod dist;
pub mod ensemble;
pub mod forecaster;
pub mod global;
pub mod hierarchical;
pub mod leaf;
mod leaf_enum;
pub mod leaves;

pub use dist::{Gaussian, GaussianMixture};
pub use forecaster::LaplaceForecaster;
pub use global::{GlobalLaplace, MetaLearnerScaffold};
pub use hierarchical::HierarchicalLaplace;
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
