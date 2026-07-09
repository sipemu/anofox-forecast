//! Streaming multivariate anomaly detection on the prediction parade.
//!
//! Port of [microprediction/timemachines' `mahalanobis`](
//! https://github.com/microprediction/timemachines/blob/main/src/timemachines/heads/mahalanobis.py)
//! head. Wraps a fitted distributional forecaster and emits, per tick,
//! a Mahalanobis distance and calibrated p-value of the forecast
//! surprise vector.
//!
//! # Quick start
//!
//! ```no_run
//! use anofox_forecast::anomaly::{MahalanobisConfig, MahalanobisDetector};
//! use anofox_forecast::models::laplace::LaplaceForecaster;
//! use anofox_forecast::prelude::TimeSeries;
//!
//! # fn go(train: TimeSeries, live_stream: Vec<f64>) -> anofox_forecast::Result<()> {
//! let cfg = MahalanobisConfig::new(8);       // k = forecast horizon
//! let mut det = MahalanobisDetector::fit_and_wrap(
//!     LaplaceForecaster::new().auto(),
//!     &train,
//!     cfg,
//! )?;
//!
//! for &y in &live_stream {
//!     det.observe(y)?;
//!     let out = det.state();
//!     if let Some(p) = out.p_value {
//!         if p < 0.001 {
//!             println!("anomaly at y={y}, p={p:.2e}, run={}", out.run);
//!         }
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! See `examples/anomaly_detection.rs` for a runnable synthetic
//! demonstration (`cargo run --release --features anomaly --example anomaly_detection`).
//!
//! # Interpretation
//!
//! - `state().d2` — Mahalanobis distance² of the current k-vector of
//!   standardised surprises;
//! - `state().p_value` — calibrated tail probability
//!   (Uniform(0, 1) under a well-specified forecaster);
//! - `state().run` — consecutive ticks with `d² > q_guard`. A run of
//!   **1-2** ticks reads as a **point outlier**; a **growing run**
//!   reads as a **changepoint / regime shift**.
//!
//! `p_value` and `d2` are `None` while any horizon of the parade is
//! still warming up (the first `k` observations post-fit).
//!
//! # Layers
//!
//! 1. [`chi2`], [`gpd`], [`linalg`], [`quantile`] — numerical primitives.
//! 2. [`parade`] — PIT + z-vector bookkeeping on top of a base
//!    distributional forecaster.
//! 3. [`mahalanobis`] — running μ / Σ of z, Welch-Satterthwaite bulk +
//!    Peaks-Over-Threshold GPD tail, Huberized updates, changepoint
//!    escape, deep-evidence (`nlp`) channel.
//! 4. [`zbank`] — optional multi-engine bank at different
//!    `(scale_alpha, stride)` gridpoints for multi-scale detection.
//!
//! # Configuration
//!
//! [`MahalanobisConfig::new`] returns defaults matching the Python
//! reference exactly:
//!
//! | field | default | meaning |
//! |---|---|---|
//! | `alpha` | `0.02` | EWMA rate for μ, Σ, m2, v2 |
//! | `scatter` | `Factor { factors: 1, dfloor: 1e-3 }` | scatter regularisation |
//! | `guard_p` | `0.99` | Huberisation threshold quantile |
//! | `adapt_after` | `10` | changepoint escape after this many guarded ticks |
//! | `pot_level` | `0.98` | POT tail threshold quantile |
//! | `min_exc` | `30` | excesses required before GPD kicks in |
//!
//! # Reference
//!
//! - Efron, B. (2004). Large-scale simultaneous hypothesis testing:
//!   The choice of a null distribution.
//! - Hosking & Wallis (1987). Parameter and quantile estimation for
//!   the generalized Pareto distribution.
//! - Wichura, M.J. (1988). AS 241: The Percentage Points of the
//!   Normal Distribution.
//! - Cotton, P. (2024). microprediction/timemachines and
//!   microprediction/skaters.

pub mod chi2;
pub mod gpd;
pub mod linalg;
pub mod mahalanobis;
pub mod parade;
pub mod quantile;
pub mod zbank;

pub use mahalanobis::{
    AnomalyOutput, MahalanobisConfig, MahalanobisDetector, MahalanobisScorer, ScatterMode,
};
pub use parade::Parade;
pub use zbank::{ZBank, ZBankBuilder, ZBankDetector};
