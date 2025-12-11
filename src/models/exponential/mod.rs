//! Exponential smoothing models.
//!
//! This module provides exponential smoothing forecasting methods:
//! - Simple Exponential Smoothing (SES)
//! - Holt's Linear Trend
//! - Holt-Winters (additive and multiplicative seasonality)
//! - ETS (Error-Trend-Seasonal) state-space framework
//! - AutoETS (automatic model selection)

mod auto_ets;
mod ets;
mod holt;
mod holt_winters;
mod ses;

pub use auto_ets::{AutoETS, AutoETSConfig, SelectionCriterion};
pub use ets::{ETSSpec, ErrorType, SeasonalType as ETSSeasonalType, TrendType, ETS};
pub use holt::HoltLinearTrend;
pub use holt_winters::{HoltWinters, SeasonalType};
pub use ses::SimpleExponentialSmoothing;
