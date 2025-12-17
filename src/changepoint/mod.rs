//! Changepoint detection algorithms.
//!
//! Provides methods to detect structural changes in time series.
//!
//! # Available Algorithms
//!
//! - **PELT**: Pruned Exact Linear Time - exact method with O(n) average complexity
//!
//! # Cost Functions
//!
//! - **L1**: Robust to outliers, uses median
//! - **L2**: Standard mean-based (default)
//! - **Normal**: Log-likelihood for variance changes
//! - **Poisson**: For count data
//! - **LinearTrend**: Detects slope/trend changes
//! - **MeanVariance**: Joint mean and variance changes
//! - **Cusum**: Cumulative sum for sustained shifts
//! - **Periodicity**: Seasonal pattern changes
//!
//! # Example
//!
//! ```
//! use anofox_forecast::changepoint::{pelt_detect, PeltConfig, CostFunction};
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
//!
//! // Use LinearTrend cost for slope changes
//! let config = PeltConfig::default()
//!     .cost_function(CostFunction::LinearTrend)
//!     .penalty(10.0);
//! ```

pub mod cost;
pub mod pelt;

pub use cost::{
    cusum_cost, l1_cost, l2_cost, linear_trend_cost, mean_variance_cost, normal_cost,
    periodicity_cost, poisson_cost, segment_cost, total_cost, CostFunction,
};
pub use pelt::{pelt_detect, PeltConfig, PeltResult};
