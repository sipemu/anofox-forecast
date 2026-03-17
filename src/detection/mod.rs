//! Detection utilities for time series analysis.
//!
//! This module provides tools for detecting:
//! - **Seasonal periods** — [`detect_periods`], [`detect_dominant_period`]
//! - Outliers and anomalies
//! - Spectral analysis (Welch's periodogram)
//!
//! # Period detection
//!
//! With the `seasonal-detection` feature enabled, period detection uses the
//! SAZED ensemble algorithm from [`fdars-core`](https://crates.io/crates/fdars-core)
//! for maximum robustness. Without it, a Welch-periodogram-based detector
//! with spectral-leakage suppression is used.
//!
//! ```
//! use anofox_forecast::detection::{detect_periods, detect_dominant_period, PeriodDetectionConfig};
//!
//! let signal: Vec<f64> = (0..144)
//!     .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
//!     .collect();
//!
//! assert_eq!(detect_dominant_period(&signal), Some(12));
//! ```

mod fft;
mod outlier;
pub mod period;

// Outlier detection
pub use outlier::{
    detect_outliers, detect_outliers_auto, OutlierConfig, OutlierMethod, OutlierResult,
};

// FFT utilities — low-level spectral estimation
pub use fft::welch_periodogram;

// Period detection — high-level API
pub use period::{detect_dominant_period, detect_periods, Period, PeriodDetectionConfig};
