//! Seasonality detection and decomposition.
//!
//! This module provides tools for analyzing seasonal patterns in time series:
//! - STL: Seasonal-Trend decomposition using LOESS
//! - MSTL: Multiple seasonal-trend decomposition for multiple periods

mod mstl;
mod stl;

pub use mstl::{MSTLResult, MSTL};
pub use stl::{STLResult, STL};
