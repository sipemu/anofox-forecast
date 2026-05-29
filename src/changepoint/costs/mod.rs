//! Trait-based cost functions for changepoint detection.
//!
//! Each cost is a struct implementing [`Cost`](crate::changepoint::Cost).
//! Cumulative-sum-based costs (`CostL2`, `CostNormal`) precompute O(n)
//! tables during `fit` so subsequent `error` calls are O(1).
//!
//! Mirrors the cost-function surface of the
//! [`ruptures`](https://github.com/deepcharles/ruptures) Python library.

pub mod ar;
pub mod clinear;
pub mod cosine;
pub mod cusum;
pub mod l1;
pub mod l2;
pub mod linear;
pub mod linear_mv;
pub mod mahalanobis;
pub mod mean_variance;
pub mod normal;
pub mod poisson;
pub mod rank;
pub mod rbf;

pub use ar::CostAR;
pub use clinear::CostCLinear;
pub use cosine::CostCosine;
pub use cusum::CostCusum;
pub use l1::CostL1;
pub use l2::CostL2;
pub use linear::CostLinearTrend;
pub use linear_mv::CostLinear;
pub use mahalanobis::CostMahalanobis;
pub use mean_variance::CostMeanVariance;
pub use normal::CostNormal;
pub use poisson::CostPoisson;
pub use rank::CostRank;
pub use rbf::CostRbf;
