//! Changepoint detection algorithms.
//!
//! Provides methods to detect structural changes in time series.
//!
//! # Available Algorithms
//!
//! - **PELT**: Pruned Exact Linear Time - exact method with O(n) average complexity
//!
//! # Example
//!
//! ```
//! use anofox_forecast::changepoint::{pelt_detect, PeltConfig};
//!
//! // Create series with a level shift
//! let mut series = vec![0.0; 50];
//! series.extend(vec![10.0; 50]);
//!
//! let config = PeltConfig::default().penalty(5.0);
//! let result = pelt_detect(&series, &config);
//!
//! // Should detect one changepoint around index 50
//! assert_eq!(result.n_changepoints, 1);
//! ```

pub mod cost;
pub mod pelt;

pub use cost::{l1_cost, l2_cost, normal_cost, poisson_cost, segment_cost, total_cost, CostFunction};
pub use pelt::{pelt_detect, PeltConfig, PeltResult};
