//! M3 accuracy harness: tracer + AutoETS on M3 monthly.
//!
//! ## Data
//! Reads Monash TSF files from `$ANOFOX_DATASET_DIR`. Test skips cleanly
//! when the env var is not set (ACCUR-01).
//!
//! ## Tracer (Plan 01)
//! Proves one M3-monthly series flows end-to-end:
//!   loader → single expanding-window fold → temporal-integrity assertion
//!   → AutoETS predict → training-denominator MASE → finite positive result.
//!
//! ## Assertions (Plan 01 tracer)
//! 1. Temporal integrity: fold satisfies `train_end <= test_start` (ACCUR-02).
//! 2. MASE computed via `mase_scale` (training denominator), NOT via
//!    `ForecastMetrics::compute` (which scales on the test slice).
//! 3. MASE is finite and positive on a real M3-monthly series.
//!
//! Plans 02–04 will expand this file with full per-frequency loops,
//! Naive2 baseline, Diebold-Mariano gate, and the ACCUR-08 anchor.

use anofox_bench_harness::loader::{dataset_dir_from_env, mase_scale, parse_tsf_with_meta};
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::cross_validation::{CVStrategy, ConstraintViolation, CvFoldGenerator};
use chrono::{Duration, TimeZone, Utc};

/// Build a synthetic-timestamp `TimeSeries` from a value slice.
///
/// Uses monthly-spaced UTC timestamps starting 2000-01-01. This helper is
/// local for Plan 01; Plan 02 will extract it into a shared harness module.
fn make_ts_from_slice(values: &[f64]) -> anofox_forecast::error::Result<TimeSeries> {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec())
}

/// Tracer: one M3-monthly series through CV → temporal gate → training-denominator MASE.
///
/// This proves the full accuracy path end-to-end on a single series before any
/// horizontal expansion (Plan 02). The test skips cleanly when `ANOFOX_DATASET_DIR`
/// is not set (ACCUR-01).
#[test]
fn tracer_m3_monthly_autoets_one_series() {
    // ACCUR-01: env gate — skip cleanly when dataset corpus not available.
    let dir = match dataset_dir_from_env() {
        Some(d) => d,
        None => {
            eprintln!("ANOFOX_DATASET_DIR not set — skipping tracer");
            return;
        }
    };

    // Load M3-monthly only (not the full load_m3 which also reads Y/Q).
    let tsf_path = dir.join("m3_monthly.tsf");
    let (_freq, file_horizon, all_series) =
        parse_tsf_with_meta(&tsf_path).expect("failed to parse m3_monthly.tsf");

    // M3 monthly horizon is 18; period is 12 for monthly.
    let horizon = file_horizon;
    let period: usize = 12;

    // Pick the first series long enough for a single expanding-window fold.
    // Need: values.len() > horizon + period + 2 (at least one full season of
    // training data plus the test horizon).
    let series = all_series
        .iter()
        .find(|s| s.values.len() > horizon + period + 2)
        .expect("no M3-monthly series long enough for the tracer fold");

    let values = &series.values;
    let n = values.len();

    // Reuse the library's CvFoldGenerator (D-02): single fold, expanding window.
    // min_initial_window = n - horizon gives the competition single-origin split:
    // train = everything except the last H steps, test = last H steps.
    let folds = CvFoldGenerator::new()
        .n_folds(1)
        .horizon(horizon)
        .min_initial_window(n - horizon)
        .strategy(CVStrategy::Expanding)
        .on_constraint_violation(ConstraintViolation::ReduceFolds)
        .generate(n)
        .expect("fold generation failed");

    assert!(
        !folds.is_empty(),
        "expected at least one fold for series '{}' (len={})",
        series.id,
        n
    );

    let fold = &folds[0];

    // ACCUR-02: temporal integrity assertion — must never fire on well-formed input,
    // but its existence is required by the spec.
    assert!(
        fold.train_end <= fold.test_start,
        "temporal integrity violation: train_end={} > test_start={}",
        fold.train_end,
        fold.test_start
    );

    // Slice train and test.
    let train = &values[fold.train_start..fold.train_end];
    let test = &values[fold.test_start..fold.test_end];

    assert!(!train.is_empty(), "train slice must not be empty");
    assert!(!test.is_empty(), "test slice must not be empty");

    // Build TimeSeries and fit AutoETS.
    let train_ts = make_ts_from_slice(train).expect("TimeSeries construction failed");
    let mut model = AutoETS::new();
    model
        .fit(&train_ts)
        .unwrap_or_else(|e| panic!("AutoETS fit failed on '{}': {:?}", series.id, e));

    // Predict test horizon (not fitted_values — those are in-sample).
    let forecast = model
        .predict(horizon)
        .unwrap_or_else(|e| panic!("AutoETS predict failed on '{}': {:?}", series.id, e));
    let pred = forecast.primary();

    // Compute MASE with the TRAINING-slice denominator (Pitfall 1 — never
    // call ForecastMetrics::compute here, which scales on the test slice).
    let denom = mase_scale(train, period);
    let test_len = test.len().min(pred.len());
    let fmae = test
        .iter()
        .take(test_len)
        .zip(pred.iter().take(test_len))
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / test_len as f64;
    let mase = fmae / denom;

    // The tracer proves the pipeline produces a finite positive MASE.
    // The tight ≈0.93 whole-corpus anchor is landed in Plan 04, not here.
    assert!(
        mase.is_finite(),
        "MASE is NaN/Inf on series '{}' — check D-03 fix and training denominator",
        series.id
    );
    assert!(
        mase > 0.0,
        "MASE is non-positive ({}) on series '{}' — check prediction output",
        mase,
        series.id
    );

    eprintln!(
        "Tracer OK: series='{}' n={} train={} test={} horizon={} MASE={:.4}",
        series.id,
        n,
        train.len(),
        test.len(),
        horizon,
        mase
    );
}
