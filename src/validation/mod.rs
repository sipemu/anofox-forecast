//! Statistical validation tests for time series models.
//!
//! Provides diagnostic tests for model residuals and stationarity tests.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::validation::{ljung_box, durbin_watson, adf_test, kpss_test};
//!
//! // Check if residuals are white noise
//! let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05, -0.08, 0.12, -0.15, 0.1, -0.05];
//! let lb_result = ljung_box(&residuals, Some(5), 0);
//! if lb_result.is_white_noise(0.05) {
//!     println!("Residuals pass Ljung-Box test");
//! }
//!
//! // Check for first-order autocorrelation
//! let dw_result = durbin_watson(&residuals);
//! println!("Durbin-Watson statistic: {}", dw_result.statistic);
//!
//! // Test stationarity
//! let series = vec![1.0, 1.2, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 1.1, 0.9];
//! let adf = adf_test(&series, None);
//! let kpss = kpss_test(&series, None);
//! ```

pub mod residual_tests;
pub mod stationarity;

// Re-export from residual_tests
pub use residual_tests::{
    box_pierce, durbin_watson, ljung_box, AutocorrelationType, DurbinWatsonResult, LjungBoxResult,
};

// Re-export from stationarity
pub use stationarity::{
    adf_test, kpss_test, test_stationarity, CriticalValues, StationarityResult,
};
