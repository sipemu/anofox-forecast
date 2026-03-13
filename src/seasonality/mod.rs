//! Seasonality detection and decomposition.
//!
//! This module provides tools for analyzing seasonal patterns in time series:
//! - STL: Seasonal-Trend decomposition using LOESS
//! - MSTL: Multiple seasonal-trend decomposition for multiple periods
//! - Fourier: Prophet-style Fourier seasonality modeling

pub mod fourier;
mod mstl;
mod stl;

pub use fourier::{fourier_terms, FourierSeasonality};
pub use mstl::{MSTLResult, MSTL};
pub use stl::{STLResult, StlBuilder, StlScratch, STL};
