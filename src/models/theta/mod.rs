//! Theta model family.
//!
//! The Theta method decomposes a time series using "theta lines" and
//! combines forecasts from different components.

mod model;

pub use model::Theta;
