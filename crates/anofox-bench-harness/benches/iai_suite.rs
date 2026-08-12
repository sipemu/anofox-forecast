//! iai-callgrind instruction-count benchmarks for the three hot paths (PERF-02).
//!
//! Measures CPU instruction counts (Ir) for AutoETS fit, AutoARIMA fit, and a 100-series
//! batch run. The 1% soft limit on EventKind::Ir causes CI to fail when instruction
//! count rises strictly greater than 1% vs the stored baseline (D-10).
//!
//! Run via:
//!   cargo bench -p anofox-bench-harness --bench iai_suite
//!
//! Requires valgrind to be installed and iai-callgrind-runner on PATH.
//! Install the runner at exactly 0.16.1 --locked (must match the crate version):
//!   cargo install iai-callgrind-runner --version 0.16.1 --locked

use anofox_bench_harness::fixtures::make_seeded_series;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use anofox_forecast::{batch, core::TimeSeries};
use iai_callgrind::{
    library_benchmark, library_benchmark_group, main, Callgrind, EventKind, LibraryBenchmarkConfig,
};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Setup functions — run outside the callgrind measurement window
// ---------------------------------------------------------------------------

fn setup_ts_n200_ets() -> TimeSeries {
    make_seeded_series(200, 42)
}

fn setup_ts_n200_arima() -> TimeSeries {
    // Different seed so ets and arima hot paths are independent
    make_seeded_series(200, 7)
}

// Build 100 raw value vectors for batch::auto_ets (&[Vec<f64>] form).
// Uses the same LCG as the project benches. Each series has length 100 with
// a different per-series seed offset so the batch sees heterogeneous input.
fn setup_batch_100() -> Vec<Vec<f64>> {
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(100);
    for s in 0..100u64 {
        let mut rng_state: u64 = s.wrapping_add(1).wrapping_mul(6364136223846793005);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
                let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                let trend = 0.05 * i as f64;
                trend + seasonal + noise + 20.0
            })
            .collect();
        series.push(values);
    }
    series
}

// ---------------------------------------------------------------------------
// Hot-path benchmark functions (measured under callgrind)
// Note: doc comments (///) cannot appear before #[library_benchmark] —
// the proc-macro rejects any attributes other than #[bench] and #[benches].
// ---------------------------------------------------------------------------

// AutoETS fit on a 200-observation series with period 12.
#[library_benchmark]
#[bench::n200(setup_ts_n200_ets())]
fn bench_auto_ets_fit(ts: TimeSeries) {
    let mut model = AutoETS::with_period(12);
    black_box(model.fit(black_box(&ts)).unwrap());
}

// AutoARIMA fit on a 200-observation series (auto order selection).
#[library_benchmark]
#[bench::n200(setup_ts_n200_arima())]
fn bench_auto_arima_fit(ts: TimeSeries) {
    let mut model = AutoARIMA::new();
    black_box(model.fit(black_box(&ts)).unwrap());
}

// Batch AutoETS: 100 series x length 100, period 12, horizon 12.
// Uses batch::auto_ets(&[Vec<f64>], period, horizon, pool) — the verified
// 4-argument signature from src/batch.rs.
#[library_benchmark]
#[bench::s100_n100(setup_batch_100())]
fn bench_batch_100(values: Vec<Vec<f64>>) {
    black_box(batch::auto_ets(black_box(&values), 12, Some(12), None));
}

// ---------------------------------------------------------------------------
// Benchmark group with 1% Ir soft limit (D-10: fail CI on >1% instruction growth)
// ---------------------------------------------------------------------------

library_benchmark_group!(
    name = hot_paths;
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::Ir, 1.0f64)]));
    benchmarks = bench_auto_ets_fit, bench_auto_arima_fit, bench_batch_100
);

main!(library_benchmark_groups = hot_paths);
