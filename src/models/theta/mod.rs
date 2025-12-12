//! Theta model family.
//!
//! The Theta method decomposes a time series using "theta lines" and
//! combines forecasts from different components.
//!
//! ## Model Variants
//!
//! - **Theta (STM)**: Standard Theta Model with fixed alpha=0.1, theta=2.0
//! - **OptimizedTheta (OTM)**: Optimizes alpha and theta parameters
//! - **DynamicTheta (DSTM)**: Updates linear coefficients dynamically
//! - **DynamicOptimizedTheta (DOTM)**: Combines dynamic updates with optimization
//! - **AutoTheta**: Automatically selects the best variant
//!
//! Supports both additive and multiplicative seasonal decomposition,
//! with multiplicative as the default to match NIXTLA statsforecast.

mod auto;
mod dynamic;
mod model;
mod optimized;

pub use auto::{AutoTheta, ThetaModelType};
pub use dynamic::{DynamicOptimizedTheta, DynamicTheta};
pub use model::{DecompositionType, Theta};
pub use optimized::OptimizedTheta;
