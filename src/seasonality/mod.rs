//! Seasonality detection and decomposition.
//!
//! This module provides tools for analyzing seasonal patterns in time series:
//! - STL: Seasonal-Trend decomposition using LOESS
//! - MSTL: Multiple seasonal-trend decomposition for multiple periods
//! - Fourier: Prophet-style Fourier seasonality modeling

pub mod convenience;
pub mod fourier;
mod mstl;
mod stl;

pub use convenience::recompose as recompose_components;
pub use convenience::{
    deseasonalize, detrend, remainder_component, seasonal_adjust, seasonal_component,
    trend_component,
};
pub use fourier::{fourier_terms, FourierSeasonality};
pub use mstl::{MSTLResult, MSTL};
pub use stl::{STLResult, StlBuilder, StlScratch, STL};
