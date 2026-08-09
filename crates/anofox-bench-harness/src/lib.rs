//! Shared measurement harness for anofox-forecast performance tracking.
//!
//! Provides:
//! - `baseline`: D-02 provenance schema structs for JSON baseline files
//! - `fixtures`: deterministic seeded time series for reproducible benchmarks (D-08)

pub mod baseline;
pub mod fixtures;
