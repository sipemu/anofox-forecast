//! Detection utilities for time series analysis.
//!
//! This module provides tools for detecting:
//! - Seasonality patterns
//! - Outliers and anomalies

mod outlier;
mod seasonality;

pub use outlier::{detect_outliers, detect_outliers_auto, OutlierConfig, OutlierMethod, OutlierResult};
pub use seasonality::{
    detect_seasonality, detect_seasonality_auto, seasonal_strength, SeasonalityConfig,
    SeasonalityResult,
};
