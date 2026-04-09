//! Online monitoring of forecast errors via sequential CUSUM detectors.
//!
//! Unlike [`crate::changepoint`], which performs **offline retrospective**
//! segmentation of a completed series, this module monitors a stream of
//! forecast errors as new observations arrive and flags the moment a fitted
//! forecasting model becomes inaccurate.
//!
//! # Attribution
//!
//! The algorithms and API are a Rust port of the R package
//! [`changepoint.forecast`](https://github.com/grundy95/changepoint.forecast)
//! by Thomas Grundy (Lancaster University), released under the MIT License.
//!
//! # References
//!
//! - Fremdt, S. (2014). Page's sequential procedure for change-point detection
//!   in time series regression. *Statistics*, 49(1), 128–155.
//!   <https://doi.org/10.1080/02331888.2014.921899>
//! - Grundy, T., Killick, R., & Mihaylov, G. (2020). High-dimensional changepoint
//!   detection via a geometrically inspired mapping. *Statistics and Computing*,
//!   30, 1155–1166. <https://doi.org/10.1007/s11222-020-09940-y>
//! - Aue, A., & Horváth, L. (2004). Delay time in sequential detection of change.
//!   *Statistics & Probability Letters*, 67(3), 221–231.
//!   <https://doi.org/10.1016/j.spl.2004.01.001>
//!
//! # Typical workflow
//!
//! 1. Fit a forecaster on historical data.
//! 2. Feed the in-sample residuals (or rolling-origin CV residuals) into
//!    [`SequentialDetector::fit`] with a chosen training-window length `m`.
//! 3. As new observations arrive, compute the new forecast errors and call
//!    [`SequentialDetector::update`] to extend the CUSUM stream. The detector
//!    keeps a small constant-size state so each update is O(new errors).
//! 4. Check [`SequentialDetector::first_detection`] (or `has_detected`); once
//!    it returns `Some`, the model has drifted and should be refit.
//!
//! # Detectors
//!
//! Four CUSUM variants are supported:
//!
//! - **`Detector::PageCusum`** (default, recommended) — two-sided Page's CUSUM.
//! - **`Detector::PageCusum1`** — one-sided Page's CUSUM; use when you
//!   specifically expect a mean/variance *increase*.
//! - **`Detector::Cusum`** — two-sided original CUSUM.
//! - **`Detector::Cusum1`** — one-sided original CUSUM.
//!
//! The original CUSUM detectors grow linearly in the no-change regime and are
//! only recommended when the changepoint is expected to occur shortly after
//! monitoring begins.
//!
//! # Residual source: in-sample vs cross-validated
//!
//! Two helpers wire this module to the [`Forecaster`](crate::models::Forecaster)
//! trait:
//!
//! - [`monitor_forecaster`] uses the model's **in-sample residuals**. These
//!   are cheap but biased: because the model was fit to the same data, the
//!   sample variance of its residuals systematically underestimates the true
//!   innovation variance. The threshold is therefore slightly too tight and
//!   the detector slightly more sensitive than its nominal `alpha`.
//! - [`monitor_forecaster_cv`] uses **out-of-sample residuals** from a
//!   rolling-origin cross-validation. These give an unbiased variance estimate
//!   and the nominal false-alarm rate, at the cost of fitting the model at
//!   each origin (the existing [`rolling_forecast`](crate::utils::cross_validation::rolling_forecast)
//!   machinery is reused).
//!
//! **Recommendation:** use `monitor_forecaster_cv` for production monitoring
//! where alpha calibration matters. Use `monitor_forecaster` for quick
//! diagnostic passes where you already have a fitted model in hand.

pub mod sequential;
pub mod sequential_crit;
pub mod sequential_table;

pub use sequential::{
    monitor_forecaster, monitor_forecaster_cv, weight, Detector, ForecastErrorType,
    SequentialConfig, SequentialDetector, StreamState,
};
pub use sequential_crit::{simulate_critical_value, CriticalValue};
pub use sequential_table::lookup_critical_value;
