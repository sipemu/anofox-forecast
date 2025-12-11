//! Baseline forecasting models.
//!
//! Simple methods that serve as benchmarks for more complex models.

mod naive;
mod random_walk;
mod seasonal_naive;
mod seasonal_window;
mod sma;

pub use naive::Naive;
pub use random_walk::RandomWalkWithDrift;
pub use seasonal_naive::SeasonalNaive;
pub use seasonal_window::SeasonalWindowAverage;
pub use sma::SimpleMovingAverage;
