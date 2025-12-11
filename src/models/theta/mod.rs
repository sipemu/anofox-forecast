//! Theta model family.
//!
//! The Theta method decomposes a time series using "theta lines" and
//! combines forecasts from different components.
//!
//! Supports both additive and multiplicative seasonal decomposition,
//! with multiplicative as the default to match NIXTLA statsforecast.

mod model;

pub use model::{DecompositionType, Theta};
