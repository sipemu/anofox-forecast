// benches/baseline_suite.rs — Tracked criterion baseline suite (D-06, D-07, D-09).
//
// Covers 7 representative model families × {single-series fit+predict, batch-100}:
//   AutoARIMA, AutoETS, AutoTheta, Naive, Croston, AutoEnsemble, LaplaceForecaster
//
// CIRCULAR DEPENDENCY NOTE (RESEARCH.md Open Question 3):
//   This file lives in the root crate. The root crate is already a dependency of
//   anofox-bench-harness, so adding anofox-bench-harness as a dev-dependency here
//   would create a cycle (root ← harness ← root). To break the cycle, the LCG
//   fixture is inlined directly below. The harness crate version is authoritative
//   for iai/dhat; this copy is identical code (same LCG, same seed convention).
//
// DUAL-PROFILE STRATEGY (D-09):
//   Bench IDs are profile-agnostic. update_criterion.sh runs the suite twice —
//   once with `--features parallel` (tagged "parallel") and once default/no rayon
//   (tagged "no_parallel") — and stamps each entry in criterion.json accordingly.
//   This file does not embed the profile in function names.
//
// SAMPLE SIZE (D-06):
//   criterion_group! uses Criterion::default().sample_size(20) for all baselines.
//
// RUN LOCALLY (D-03):
//   Use scripts/update_criterion.sh to capture baselines on a quiet local machine.
//   Never run this bench in CI for timing (wall-clock noise makes CI gates unreliable).

use criterion::{criterion_group, criterion_main, Criterion};

use anofox_forecast::batch;
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::baseline::Naive;
use anofox_forecast::models::ensemble::AutoEnsemble;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::intermittent::Croston;
#[cfg(feature = "distributional")]
use anofox_forecast::models::laplace::LaplaceForecaster;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

// ---------------------------------------------------------------------------
// Inlined fixture (identical to crates/anofox-bench-harness/src/fixtures.rs
// make_seeded_series — cannot import from harness to avoid the circular dep)
// ---------------------------------------------------------------------------

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

/// Deterministic LCG series: trend + seasonality + noise, all values positive.
/// Multiplier matches existing project benches (6364136223846793005).
fn make_seeded_series(n: usize, seed: u64) -> TimeSeries {
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
    TimeSeries::univariate(make_timestamps(n), values).unwrap()
}

/// Generate 100 seeded raw value vectors for batch benchmarks.
fn make_batch_100(n: usize) -> Vec<Vec<f64>> {
    (0..100u64)
        .map(|seed| {
            let mut rng_state: u64 = seed + 1;
            (0..n)
                .map(|i| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3;
                    let seasonal = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                    let trend = 0.05 * i as f64;
                    trend + seasonal + noise + 20.0
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AutoARIMA
// ---------------------------------------------------------------------------

fn bench_auto_arima(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("auto_arima_fit_predict_n200", |b| {
        b.iter(|| {
            let mut model = AutoARIMA::new();
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    // batch-100: no batch:: helper for ARIMA; loop over 100 seeded series
    let batch_series: Vec<TimeSeries> = (0..100u64)
        .map(|s| make_seeded_series(200, s + 100))
        .collect();
    c.bench_function("auto_arima_batch100_fit_predict_n200", |b| {
        b.iter(|| {
            for ts_i in &batch_series {
                let mut model = AutoARIMA::new();
                model.fit(ts_i).unwrap();
                model.predict(12).unwrap();
            }
        })
    });
}

// ---------------------------------------------------------------------------
// AutoETS
// ---------------------------------------------------------------------------

fn bench_auto_ets(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("auto_ets_fit_predict_n200_p12", |b| {
        b.iter(|| {
            let mut model = AutoETS::with_period(12);
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    // batch-100: use batch::auto_ets (verified 4-argument signature from src/batch.rs)
    let values = make_batch_100(200);
    c.bench_function("auto_ets_batch100_fit_predict_n200_p12", |b| {
        b.iter(|| {
            let _ = batch::auto_ets(&values, 12, Some(12), None);
        })
    });
}

// ---------------------------------------------------------------------------
// AutoTheta
// ---------------------------------------------------------------------------

fn bench_auto_theta(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("auto_theta_fit_predict_n200", |b| {
        b.iter(|| {
            let mut model = AutoTheta::new();
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    let batch_series: Vec<TimeSeries> = (0..100u64)
        .map(|s| make_seeded_series(200, s + 200))
        .collect();
    c.bench_function("auto_theta_batch100_fit_predict_n200", |b| {
        b.iter(|| {
            for ts_i in &batch_series {
                let mut model = AutoTheta::new();
                model.fit(ts_i).unwrap();
                model.predict(12).unwrap();
            }
        })
    });
}

// ---------------------------------------------------------------------------
// Naive
// ---------------------------------------------------------------------------

fn bench_naive(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("naive_fit_predict_n200", |b| {
        b.iter(|| {
            let mut model = Naive::new();
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    let batch_series: Vec<TimeSeries> = (0..100u64)
        .map(|s| make_seeded_series(200, s + 300))
        .collect();
    c.bench_function("naive_batch100_fit_predict_n200", |b| {
        b.iter(|| {
            for ts_i in &batch_series {
                let mut model = Naive::new();
                model.fit(ts_i).unwrap();
                model.predict(12).unwrap();
            }
        })
    });
}

// ---------------------------------------------------------------------------
// Croston (intermittent demand)
// ---------------------------------------------------------------------------

fn bench_croston(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("croston_fit_predict_n200", |b| {
        b.iter(|| {
            let mut model = Croston::new();
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    let batch_series: Vec<TimeSeries> = (0..100u64)
        .map(|s| make_seeded_series(200, s + 400))
        .collect();
    c.bench_function("croston_batch100_fit_predict_n200", |b| {
        b.iter(|| {
            for ts_i in &batch_series {
                let mut model = Croston::new();
                model.fit(ts_i).unwrap();
                model.predict(12).unwrap();
            }
        })
    });
}

// ---------------------------------------------------------------------------
// AutoEnsemble
// ---------------------------------------------------------------------------

fn bench_auto_ensemble(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("auto_ensemble_fit_predict_n200", |b| {
        b.iter(|| {
            let mut model = AutoEnsemble::new();
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    let batch_series: Vec<TimeSeries> = (0..100u64)
        .map(|s| make_seeded_series(200, s + 500))
        .collect();
    c.bench_function("auto_ensemble_batch100_fit_predict_n200", |b| {
        b.iter(|| {
            for ts_i in &batch_series {
                let mut model = AutoEnsemble::new();
                model.fit(ts_i).unwrap();
                model.predict(12).unwrap();
            }
        })
    });
}

// ---------------------------------------------------------------------------
// LaplaceForecaster (requires `distributional` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "distributional")]
fn bench_laplace(c: &mut Criterion) {
    let ts = make_seeded_series(200, 42);
    c.bench_function("laplace_fit_predict_n200", |b| {
        b.iter(|| {
            let mut model = LaplaceForecaster::new();
            model.fit(&ts).unwrap();
            model.predict(12).unwrap();
        })
    });

    let batch_series: Vec<TimeSeries> = (0..100u64)
        .map(|s| make_seeded_series(200, s + 600))
        .collect();
    c.bench_function("laplace_batch100_fit_predict_n200", |b| {
        b.iter(|| {
            for ts_i in &batch_series {
                let mut model = LaplaceForecaster::new();
                model.fit(ts_i).unwrap();
                model.predict(12).unwrap();
            }
        })
    });
}

// ---------------------------------------------------------------------------
// Registration — standardized sample_size per D-06
// Laplace bench requires `distributional` feature (gated module).
// ---------------------------------------------------------------------------

#[cfg(feature = "distributional")]
criterion_group!(
    name = baselines;
    config = Criterion::default().sample_size(20);
    targets = bench_auto_arima,
              bench_auto_ets,
              bench_auto_theta,
              bench_naive,
              bench_croston,
              bench_auto_ensemble,
              bench_laplace
);

#[cfg(not(feature = "distributional"))]
criterion_group!(
    name = baselines;
    config = Criterion::default().sample_size(20);
    targets = bench_auto_arima,
              bench_auto_ets,
              bench_auto_theta,
              bench_naive,
              bench_croston,
              bench_auto_ensemble
);

criterion_main!(baselines);
