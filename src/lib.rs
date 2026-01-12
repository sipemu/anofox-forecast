//! # anofox-forecast
//!
//! Time series forecasting library for Rust.
//!
//! Provides 35+ forecasting models including ARIMA, ETS, Theta,
//! and baseline methods, along with seasonality decomposition (STL/MSTL),
//! changepoint detection, and outlier detection.
//!
//! For comprehensive periodicity detection, see the
//! [fdars](https://crates.io/crates/fdars-core) crate.
//!
//! # Architecture Decisions
//!
//! ## Cross-Validation Split (`ts_cv_split`)
//!
//! Time series cross-validation with data leakage prevention is implemented in the
//! [forecast-extension](https://github.com/DataZooDE/forecast-extension) DuckDB extension
//! rather than in this crate. This section documents the rationale.
//!
//! ### Why CV Split is Not Part of `TimeSeries`
//!
//! The [`TimeSeries`](crate::core::TimeSeries) struct represents a single time series with
//! its values, timestamps, and metadata. Cross-validation splitting was considered as a
//! method on `TimeSeries` but was intentionally kept separate for these reasons:
//!
//! 1. **Cross-series coordination**: Fold generation is a global operation across multiple
//!    series, not a per-series operation. CV requires consistent fold boundaries across all
//!    series in a dataset.
//!
//! 2. **External feature handling**: Unknown future features like `stockout` flags or
//!    `segment_id` changes are external columns that don't belong in the series data model.
//!    These require schema-aware handling at the data layer.
//!
//! 3. **Data manipulation efficiency**: DuckDB's vectorized execution is more efficient for
//!    the bulk data operations (filtering, joining, filling) that CV split requires.
//!
//! 4. **Schema flexibility**: SQL macros can handle arbitrary column schemas without
//!    requiring Rust to know the schema at compile time.
//!
//! ### Component Distribution
//!
//! | Component | Location | Rationale |
//! |-----------|----------|-----------|
//! | Fold generation | DuckDB extension | Cross-series coordination, global operation |
//! | Train/test assignment | SQL/DuckDB | Simple comparison, vectorized execution |
//! | Unknown feature filling | Rust UDF via DuckDB | Per-series state tracking |
//! | Orchestration | SQL macro | Flexible, schema-agnostic |
//!
//! ### Using CV Functionality
//!
//! For time series cross-validation with data leakage prevention, use the `ts_cv_split`
//! function from the [forecast-extension](https://github.com/DataZooDE/forecast-extension):
//!
//! ```sql
//! -- Example: Generate CV folds with unknown feature handling
//! SELECT * FROM ts_cv_split(
//!     my_data,
//!     n_splits := 3,
//!     horizon := 7,
//!     unknown_features := ['stockout', 'segment_id']
//! );
//! ```
//!
//! See [forecast-extension#54](https://github.com/DataZooDE/forecast-extension/issues/54)
//! for implementation details.
//!
//! ### Future Considerations
//!
//! If per-series CV semantics become necessary in Rust (e.g., for standalone use without
//! DuckDB), the fold generation logic could be extracted:
//!
//! ```rust,ignore
//! pub struct CvFoldGenerator {
//!     n_splits: usize,
//!     horizon: usize,
//!     gap: usize,
//! }
//!
//! impl CvFoldGenerator {
//!     pub fn folds(&self, series_len: usize) -> Vec<usize> {
//!         // Returns training end indices for each fold
//!     }
//! }
//! ```
//!
//! This would allow fold generation to be shared while keeping data manipulation
//! in the appropriate layer (SQL for multi-series datasets, Rust for single-series use).

// Allow some clippy warnings for cleaner code in specific cases
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_is_multiple_of)] // is_multiple_of is unstable on WASM

// Prevent use of parallel feature on WASM targets (rayon requires OS threads)
#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
compile_error!(
    "The 'parallel' feature is not supported on WASM targets. Build without --features parallel"
);

pub mod changepoint;
pub mod core;
pub mod detection;
pub mod error;
pub mod features;
pub mod models;
#[cfg(feature = "postprocess")]
pub mod postprocess;
pub mod seasonality;
pub mod simd;
pub mod transform;
pub mod utils;
pub mod validation;

pub use error::{ForecastError, Result};

pub mod prelude {
    pub use crate::core::{Forecast, TimeSeries};
    pub use crate::error::{ForecastError, Result};
    pub use crate::models::Forecaster;
    pub use crate::utils::{calculate_metrics, quantile_normal, AccuracyMetrics};
}
