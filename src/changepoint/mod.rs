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

pub mod algorithms;
pub mod cost;
pub mod costs;
pub mod detector;
pub mod metrics;
pub mod pelt;
pub mod signal;

pub use cost::{
    cusum_cost, l1_cost, l2_cost, linear_trend_cost, mean_variance_cost, normal_cost, poisson_cost,
    segment_cost, total_cost, CostFunction,
};
pub use pelt::{pelt_detect, AutoPeltResult, Pelt, PeltConfig, PeltResult};

// Trait-based detection surface (mirrors ruptures).
pub use algorithms::{
    BinsegDetector, BottomUpDetector, DynpDetector, KernelCpdDetector, KernelKind, PeltDetector,
    WindowDetector,
};
pub use costs::{
    CostAR, CostCLinear, CostCosine, CostCusum, CostL1, CostL2, CostLinear, CostLinearTrend,
    CostMahalanobis, CostMeanVariance, CostNormal, CostPoisson, CostRank, CostRbf,
};
pub use detector::{Cost, Detector, DetectorResult};
pub use metrics::{hausdorff, precision_recall, randindex, PrecisionRecall};
pub use signal::Signal;
