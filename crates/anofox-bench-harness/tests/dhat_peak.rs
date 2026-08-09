//! Peak-memory gate: assert each major model family stays within baseline × 1.15 (D-12).
//!
//! GLOBAL ALLOCATOR NOTE (Pitfall 3):
//!   dhat::Alloc must be the ONLY global allocator in this test binary.
//!   Do not add a second global allocator anywhere else in the harness test suite.
//!   wee_alloc is also banned (archived, memory leaks); only dhat::Alloc is used here.
//!
//! SINGLE-PROFILER CONSTRAINT:
//!   dhat panics if more than one Profiler is live simultaneously. All families are
//!   measured inside one #[test] function to guarantee sequential, non-overlapping
//!   profiler sessions. Run with: cargo test -p anofox-bench-harness --test dhat_peak
//!
//! FIRST-RUN HANDLING (MEAS-04):
//!   If .planning/baselines/dhat.json is missing or an entry is absent, the assertion
//!   is skipped with a printed notice — the baseline is written by update_dhat.sh,
//!   not asserted into existence.
//!
//! BOUNDARY (D-12):
//!   assert!(stats.max_bytes <= (baseline_bytes as f64 * 1.15) as usize)
//!   The float→usize truncation is intentional: truncating (not rounding) the threshold
//!   is conservative — it cannot accidentally widen the allowed range.

use anofox_bench_harness::fixtures::make_seeded_series;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::ensemble::AutoEnsemble;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::intermittent::Croston;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;
use std::path::PathBuf;

// NOTE: dhat::Alloc is the only global allocator in this test binary (see Pitfall 3 above).
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// ---------------------------------------------------------------------------
// Baseline loader
// ---------------------------------------------------------------------------

/// Load peak_bytes for a named entry from .planning/baselines/dhat.json.
/// Returns None if the file is missing, malformed, or the entry is absent.
fn load_dhat_baseline(name: &str) -> Option<u64> {
    // Resolve repo root relative to Cargo.toml of the root crate.
    // CARGO_MANIFEST_DIR points to the harness crate; go up two levels.
    let harness_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = harness_dir.parent()?.parent()?;
    let baseline_path = repo_root
        .join(".planning")
        .join("baselines")
        .join("dhat.json");

    if !baseline_path.exists() {
        return None;
    }

    let content = std::fs::read_to_string(&baseline_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let benchmarks = json.get("benchmarks")?.as_array()?;

    for entry in benchmarks {
        if entry.get("name")?.as_str()? == name {
            return entry.get("peak_bytes")?.as_u64();
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helper: assert or skip
// ---------------------------------------------------------------------------

/// Assert peak_bytes <= baseline × 1.15, or skip with a notice if baseline absent.
/// Returns (measured_peak_bytes, skipped: bool) for the capture mode used by update_dhat.sh.
fn check_peak(name: &str, peak_bytes: usize) -> bool {
    match load_dhat_baseline(name) {
        None => {
            println!(
                "NOTICE [{name}]: no baseline in dhat.json — skipping assertion. \
                 Run bash scripts/update_dhat.sh to capture the baseline."
            );
            false // skipped
        }
        Some(baseline_bytes) => {
            // D-12: peak strictly under or equal to 1.15×.
            // float→usize truncation is intentional (conservative — never widens the range).
            let threshold = (baseline_bytes as f64 * 1.15) as usize;
            assert!(
                peak_bytes <= threshold,
                "[{name}] peak memory {peak_bytes} bytes exceeds baseline {baseline_bytes} × 1.15 = {threshold} bytes"
            );
            println!(
                "PASS [{name}]: peak={peak_bytes}B  baseline={baseline_bytes}B  threshold={threshold}B"
            );
            true // asserted
        }
    }
}

// ---------------------------------------------------------------------------
// Peak-memory test — all families in one #[test] to avoid overlapping profilers
// ---------------------------------------------------------------------------

#[test]
fn peak_memory_all_families() {
    // Determine if we are in capture mode (CAPTURE_DHAT=1).
    // In capture mode, print "name peak_bytes" lines instead of asserting,
    // so update_dhat.sh can harvest the numbers.
    let capture_mode = std::env::var("CAPTURE_DHAT").unwrap_or_default() == "1";

    // n=1000 as specified in the plan (D-12 representative load).
    let n = 1000usize;

    // -----------------------------------------------------------------------
    // AutoETS
    // -----------------------------------------------------------------------
    {
        let _p = dhat::Profiler::builder().testing().build();
        let ts = make_seeded_series(n, 42);
        let mut model = AutoETS::with_period(12);
        model.fit(&ts).unwrap();
        model.predict(12).unwrap();
        let stats = dhat::HeapStats::get();
        let name = "auto_ets_n1000";
        if capture_mode {
            println!("CAPTURE {name} {}", stats.max_bytes);
        } else {
            check_peak(name, stats.max_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // AutoARIMA
    // -----------------------------------------------------------------------
    {
        let _p = dhat::Profiler::builder().testing().build();
        let ts = make_seeded_series(n, 7);
        let mut model = AutoARIMA::new();
        model.fit(&ts).unwrap();
        model.predict(12).unwrap();
        let stats = dhat::HeapStats::get();
        let name = "auto_arima_n1000";
        if capture_mode {
            println!("CAPTURE {name} {}", stats.max_bytes);
        } else {
            check_peak(name, stats.max_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // AutoTheta
    // -----------------------------------------------------------------------
    {
        let _p = dhat::Profiler::builder().testing().build();
        let ts = make_seeded_series(n, 13);
        let mut model = AutoTheta::new();
        model.fit(&ts).unwrap();
        model.predict(12).unwrap();
        let stats = dhat::HeapStats::get();
        let name = "auto_theta_n1000";
        if capture_mode {
            println!("CAPTURE {name} {}", stats.max_bytes);
        } else {
            check_peak(name, stats.max_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // Naive
    // -----------------------------------------------------------------------
    {
        let _p = dhat::Profiler::builder().testing().build();
        let ts = make_seeded_series(n, 17);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        model.predict(12).unwrap();
        let stats = dhat::HeapStats::get();
        let name = "naive_n1000";
        if capture_mode {
            println!("CAPTURE {name} {}", stats.max_bytes);
        } else {
            check_peak(name, stats.max_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // Croston
    // -----------------------------------------------------------------------
    {
        let _p = dhat::Profiler::builder().testing().build();
        let ts = make_seeded_series(n, 23);
        let mut model = Croston::new();
        model.fit(&ts).unwrap();
        model.predict(12).unwrap();
        let stats = dhat::HeapStats::get();
        let name = "croston_n1000";
        if capture_mode {
            println!("CAPTURE {name} {}", stats.max_bytes);
        } else {
            check_peak(name, stats.max_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // AutoEnsemble
    // -----------------------------------------------------------------------
    {
        let _p = dhat::Profiler::builder().testing().build();
        let ts = make_seeded_series(n, 31);
        let mut model = AutoEnsemble::new();
        model.fit(&ts).unwrap();
        model.predict(12).unwrap();
        let stats = dhat::HeapStats::get();
        let name = "auto_ensemble_n1000";
        if capture_mode {
            println!("CAPTURE {name} {}", stats.max_bytes);
        } else {
            check_peak(name, stats.max_bytes);
        }
    }
}
