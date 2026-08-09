//! Deterministic seeded time series fixtures for reproducible benchmarks (D-08).
//!
//! All benchmark suites (iai, dhat, criterion baseline) draw from this
//! single source of truth so results across dimensions are directly
//! comparable. The LCG generator matches the one in `benches/ets_benchmark.rs`
//! and is seed-parameterized so callers can produce independent series.

use anofox_forecast::core::TimeSeries;
use chrono::{Duration, TimeZone, Utc};

/// Build a deterministic synthetic time series with trend + seasonality + noise.
///
/// Uses the same LCG as existing project benches (multiplier 6364136223846793005),
/// parameterized by `seed` so callers can produce independent, reproducible series.
/// All values are positive (offset +20.0 to keep multiplicative models happy).
///
/// # Arguments
///
/// * `n` - Number of observations.
/// * `seed` - Initial LCG state; use fixed values (e.g. 42) for stability across runs.
pub fn make_seeded_series(n: usize, seed: u64) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
    let mut rng_state: u64 = seed;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
            let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            let trend = 0.05 * i as f64;
            trend + seasonal + noise + 20.0
        })
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}
