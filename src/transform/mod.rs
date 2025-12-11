//! Data transformations for time series.
//!
//! Provides scaling, normalization, Box-Cox transforms, and window functions.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::transform::{standardize, boxcox_auto, rolling_mean};
//!
//! let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Standardize to zero mean, unit variance
//! let scaled = standardize(&series);
//!
//! // Box-Cox transformation
//! let bc = boxcox_auto(&series);
//!
//! // Rolling mean with window 3
//! let rm = rolling_mean(&series, 3, false);
//! ```

pub mod boxcox;
pub mod scale;
pub mod window;

// Re-export from scale
pub use scale::{center, normalize, robust_scale, scale_to_range, standardize, ScaleResult};

// Re-export from boxcox
pub use boxcox::{
    boxcox, boxcox_auto, boxcox_lambda, boxcox_shifted, inv_boxcox, is_boxcox_suitable,
    BoxCoxResult,
};

// Re-export from window
pub use window::{
    ewm_mean, ewm_std, ewm_var, expanding_max, expanding_mean, expanding_min, expanding_sum,
    rolling_max, rolling_mean, rolling_median, rolling_min, rolling_std, rolling_sum, rolling_var,
};
