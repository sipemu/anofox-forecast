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

mod adida;
mod croston;
mod tsb;

pub use adida::ADIDA;
pub use croston::{Croston, CrostonVariant};
pub use tsb::TSB;
