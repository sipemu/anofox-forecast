//! Theta model family.
//!
//! The Theta method decomposes a time series using "theta lines" and
//! combines forecasts from different components.

mod theta;

pub use theta::Theta;
