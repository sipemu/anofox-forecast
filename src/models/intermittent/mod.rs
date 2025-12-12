//! Intermittent demand forecasting models.
//!
//! This module provides models specifically designed for intermittent demand,
//! where demand is sporadic with many zero-demand periods interspersed with
//! occasional non-zero demands.
//!
//! Models included:
//! - Croston: Classic and SBA variants
//! - TSB: Teunter-Syntetos-Babai
//! - ADIDA: Aggregate-Disaggregate Intermittent Demand Approach
//! - IMAPA: Intermittent Multiple Aggregation Prediction Algorithm

mod adida;
mod croston;
mod imapa;
mod tsb;

pub use adida::ADIDA;
pub use croston::{Croston, CrostonVariant};
pub use imapa::IMAPA;
pub use tsb::TSB;
