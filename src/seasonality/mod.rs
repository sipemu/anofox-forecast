//! Seasonality detection and decomposition.
//!
//! This module provides tools for analyzing seasonal patterns in time series:
//! - STL: Seasonal-Trend decomposition using LOESS
//! - MSTL: Multiple seasonal-trend decomposition for multiple periods
//! - Fourier: Prophet-style Fourier seasonality modeling
//! - Seasonal differencing: ARIMA-style seasonal differencing

pub mod auto_seasonal;
pub mod auto_trend;
pub mod convenience;
pub mod dummy;
pub mod exponential_trend;
pub mod fourier;
pub mod hp_filter;
pub mod logistic_trend;
mod mstl;
pub mod piecewise;
pub mod polynomial;
pub mod seasonal_diff;
mod stl;
pub mod theilsen;
pub mod traits;

pub use auto_seasonal::AutoSeasonal;
pub use auto_trend::AutoTrend;
pub use convenience::recompose as recompose_components;
pub use convenience::{
    deseasonalize, detrend, remainder_component, seasonal_adjust, seasonal_component,
    trend_component,
};
pub use dummy::{dummy_seasonal_amplitude, dummy_seasonal_strength, DummySeasonality};
pub use exponential_trend::ExponentialTrend;
pub use fourier::{fourier_terms, FourierSeasonality};
pub use hp_filter::{hp_cycle_variance_ratio, hp_trend_strength, HodrickPrescottFilter};
pub use logistic_trend::{CapacityMode, LogisticTrend};
pub use mstl::{MSTLResult, MSTL};
pub use piecewise::{piecewise_n_segments, piecewise_trend_features, PiecewiseLinearTrend};
pub use polynomial::PolynomialTrend;
pub use seasonal_diff::{
    seasonal_diff_strength, seasonal_diff_variance_reduction, SeasonalDifference,
};
pub use stl::{STLResult, StlBuilder, StlScratch, STL};
pub use theilsen::TheilSenTrend;
pub use traits::{AutoRecencyConfig, Recency, SeasonalComponent, TrendComponent};
