//! Shared measurement harness for anofox-forecast performance tracking.
//!
//! Provides:
//! - `baseline`: D-02 provenance schema structs for JSON baseline files
//! - `fixtures`: deterministic seeded time series for reproducible benchmarks (D-08)
//! - `loader`: Monash TSF + JSON dataset loader (ACCUR-01)
//! - `naive2`: ACF-gated seasonal/non-seasonal reference baseline (ACCUR-06, D-07)

pub mod baseline;
pub mod fixtures;
pub mod loader; // Monash TSF + JSON dataset loader (ACCUR-01)
pub mod naive2; // Naive2 baseline model (ACCUR-06, D-07)
