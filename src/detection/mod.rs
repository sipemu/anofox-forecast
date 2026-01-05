//! Detection utilities for time series analysis.
//!
//! This module provides tools for detecting:
//! - Outliers and anomalies
//! - Spectral analysis (Welch's periodogram)
//!
//! For comprehensive periodicity and seasonality detection, see the
//! [fdars](https://crates.io/crates/fdars-core) crate which provides
//! ACF, FFT, Autoperiod, CFD-Autoperiod, and SAZED algorithms.

mod fft;
mod outlier;

// Outlier detection
pub use outlier::{
    detect_outliers, detect_outliers_auto, OutlierConfig, OutlierMethod, OutlierResult,
};

// FFT utilities - only Welch's periodogram (other FFT methods available in fdars)
pub use fft::welch_periodogram;
