//! Shared measurement harness for anofox-forecast performance tracking.
//!
//! Provides:
//! - `baseline`: D-02 provenance schema structs for JSON baseline files
//! - `fixtures`: deterministic seeded time series for reproducible benchmarks (D-08)
//! - `loader`: Monash TSF + JSON dataset loader (ACCUR-01)
//! - `naive2`: ACF-gated seasonal/non-seasonal reference baseline (ACCUR-06, D-07)
//! - `dm_test`: Diebold-Mariano + HLN small-sample correction + HAC variance (BENCH-02, D-09)

pub mod baseline;
pub mod dm_test; // Diebold-Mariano + HLN + HAC significance gate (BENCH-02, D-09)
pub mod fixtures;
pub mod loader; // Monash TSF + JSON dataset loader (ACCUR-01)
pub mod naive2; // Naive2 baseline model (ACCUR-06, D-07)
