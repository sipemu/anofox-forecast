//! D-02 provenance baseline schema.
//!
//! Typed serde structs that define the JSON format for every dimension
//! under `.planning/baselines/`. All four dimension files (criterion.json,
//! iai.json, dhat.json, wasm_size.json) carry a `ProvenanceFingerprint`
//! so baselines are self-describing and auditable.

use serde::{Deserialize, Serialize};

/// Full provenance fingerprint attached to every committed baseline record.
///
/// Six fields capture the exact environment so any delta can be
/// attributed to code changes rather than toolchain or hardware drift.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProvenanceFingerprint {
    /// Short git SHA of the commit at capture time.
    pub git_sha: String,
    /// ISO-8601 UTC timestamp of the capture run.
    pub timestamp_iso: String,
    /// Full `rustc --version` string (e.g. "rustc 1.80.0 (051478957 2024-07-21)").
    pub rustc_version: String,
    /// CPU model string from /proc/cpuinfo or "unknown".
    pub host_cpu: String,
    /// Operating system and kernel version from `uname -sr`.
    pub host_os: String,
    /// Cargo feature flags active during the capture run.
    pub active_features: Vec<String>,
}

/// One criterion wall-clock entry (median + MAD, robust to noise).
///
/// Stored as informational only — CI reports drift but never fails on
/// criterion numbers (D-03). Hard gate uses iai instruction counts.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CriterionEntry {
    /// Benchmark function name (e.g. "auto_ets_fit_n200_parallel").
    pub name: String,
    /// Feature profile: "parallel" or "no_parallel" (D-09).
    pub profile: String,
    /// Median wall-clock time in nanoseconds.
    pub median_ns: f64,
    /// Median absolute deviation in nanoseconds (robust spread measure).
    pub mad_ns: f64,
}

/// One iai-callgrind instruction-count entry.
///
/// CI gate fails if `instruction_count` rises > 1% vs baseline (D-10).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IaiEntry {
    /// Benchmark function name.
    pub name: String,
    /// Total Valgrind instruction count for the measured function body.
    pub instruction_count: u64,
}

/// One dhat peak-memory entry.
///
/// Gate: assert `max_bytes <= baseline_bytes * 1.15` (D-12).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhatEntry {
    /// Model family or scenario name (e.g. "auto_ets_n1000").
    pub name: String,
    /// Peak heap bytes recorded by dhat during the run.
    pub peak_bytes: u64,
}

/// WASM binary size baseline (D-11, PERF-05).
///
/// CI gate fails when the current release `.wasm` grows strictly
/// greater than `bytes * 1.01` (> 1% relative growth).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WasmSizeBaseline {
    /// Full provenance of the environment that produced this baseline.
    pub provenance: ProvenanceFingerprint,
    /// Filename of the measured WASM artifact (e.g. "anofox_forecast_js_bg.wasm").
    pub filename: String,
    /// File size in bytes of the release-optimised WASM binary.
    pub bytes: u64,
}
