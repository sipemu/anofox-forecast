//! Detection utilities for time series analysis.
//!
//! This module provides tools for detecting:
//! - Seasonality patterns
//! - Outliers and anomalies
//! - Periodicity (multiple detection algorithms)
//!
//! # Periodicity Detection
//!
//! Multiple algorithms are available for periodicity detection:
//!
//! - [`ACFPeriodicityDetector`]: Time-domain detection using autocorrelation
//! - [`FFTPeriodicityDetector`]: Frequency-domain detection using FFT
//! - [`Autoperiod`]: Hybrid FFT+ACF method (recommended for general use)
//! - [`CFDAutoperiod`]: Noise-resistant variant with clustering
//! - [`SAZED`]: Ensemble method combining multiple approaches
//!
//! ## Quick Start
//!
//! ```rust
//! use anofox_forecast::detection::{detect_period, detect_period_ensemble};
//!
//! let signal: Vec<f64> = (0..240)
//!     .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
//!     .collect();
//!
//! // Use Autoperiod (recommended)
//! let result = detect_period(&signal);
//! if let Some(period) = result.primary_period {
//!     println!("Detected period: {}", period);
//! }
//!
//! // Or use SAZED ensemble for more robustness
//! let result = detect_period_ensemble(&signal);
//! ```

mod fft;
mod outlier;
mod periodicity;
mod sazed;
mod seasonality;

// Outlier detection
pub use outlier::{
    detect_outliers, detect_outliers_auto, OutlierConfig, OutlierMethod, OutlierResult,
};

// Legacy seasonality detection (ACF-based)
pub use seasonality::{
    detect_seasonality, detect_seasonality_auto, seasonal_strength, SeasonalityConfig,
    SeasonalityResult,
};

// FFT utilities
pub use fft::{
    fft_real, frequency_index_to_period, period_to_frequency_index, periodogram, periodogram_peaks,
    welch_periodogram,
};

// Periodicity detectors
pub use periodicity::{
    detect_period, detect_period_range, ACFPeriodicityDetector, Autoperiod, CFDAutoperiod,
    DetectedPeriod, FFTPeriodicityDetector, PeriodSource, PeriodicityDetector, PeriodicityResult,
};

// SAZED ensemble
pub use sazed::{detect_period_ensemble, detect_period_ensemble_range, SAZED};
