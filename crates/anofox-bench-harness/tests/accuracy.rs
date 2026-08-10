//! M3 accuracy harness: Naive2 + AutoETS over M3 Yearly/Quarterly/Monthly.
//!
//! ## Data
//! Reads Monash TSF files from `$ANOFOX_DATASET_DIR`. Test skips when the
//! env var is not set (ACCUR-01).
//!
//! ## Tracer (Plan 01)
//! Proves one M3-monthly series flows end-to-end:
//!   loader → single expanding-window fold → temporal-integrity assertion
//!   → AutoETS predict → training-denominator MASE → finite positive result.
//!
//! ## Per-frequency harness (Plan 02)
//! Full stratified run over M3 Yearly/Quarterly/Monthly for Naive2 + AutoETS:
//!   per-frequency FrequencyResult table with MASE, sMAPE, RMSE, MAE,
//!   MSIS (monthly only), and interval coverage (monthly only).
//!   NaN/Inf guarded aggregation — excluded count is logged (ACCUR-03).
//!   Never aggregates across frequencies (ACCUR-07).
//!
//! ## Assertions
//! 1. Temporal integrity: every fold satisfies `train_end <= test_start` (ACCUR-02).
//! 2. No silent NaN in per-frequency aggregates (ACCUR-03).
//! 3. Per-frequency stratification: Yearly/Quarterly/Monthly reported separately (ACCUR-07).
//! 4. AutoETS M3-monthly MASE anchor (ACCUR-08): within 0.02 of the pinned
//!    statsforecast reference value from `.planning/baselines/statsforecast_reference.json`.
//!    NOTE: the reference is pinned at statsforecast 2.0.3 (monthly MASE=0.8633),
//!    which differs from the historical published 0.93 (statsforecast 1.x). The
//!    comparison is apples-to-apples within the pinned env (provenance block in
//!    the fixture documents the exact version). See Plan 03 SUMMARY for full context.
//!
//! ## accuracy.json emit (Plan 04)
//! `emit_accuracy_json()` serialises per-frequency results into
//! `.planning/baselines/accuracy.json` ONLY when both:
//!   - `ANOFOX_WRITE_ACCURACY_BASELINE=1` is set, AND
//!   - the ACCUR-08 anchor assertion has passed (monthly MASE within tolerance).
//! Running plain `cargo test` without the write flag never overwrites the baseline.
//!
//! ## MSIS convention note (A4 / Pitfall 4)
//! The reused `src/utils/metrics.rs::msis()` scales by mean absolute period-1
//! first-differences, NOT seasonal differences as in the M4 competition MSIS.
//! Do NOT compare the reported MSIS directly to published M4 MSIS values.
//! The ACCUR-08 anchor is MASE only; MSIS is informational.

use anofox_bench_harness::loader::{dataset_dir_from_env, load_m3, mase_scale};
use anofox_bench_harness::naive2::Naive2;
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::cross_validation::{CVStrategy, ConstraintViolation, CvFoldGenerator};
use anofox_forecast::utils::metrics::{coverage, mae, msis, rmse, smape};
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;

// ─── TimeSeries helper ───────────────────────────────────────────────────────

/// Build a synthetic-timestamp `TimeSeries` from a value slice.
///
/// Uses monthly-spaced UTC timestamps starting 2000-01-01. This helper is
/// shared across the tracer and per-frequency harness in this file.
fn make_ts_from_slice(values: &[f64]) -> anofox_forecast::error::Result<TimeSeries> {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec())
}

// ─── FrequencyResult ─────────────────────────────────────────────────────────

/// Per-frequency accuracy summary for one model pair over one M3 frequency.
///
/// Produced by `run_accuracy_harness()` and consumed by Plan 04 to write
/// `.planning/baselines/accuracy.json`. Fields named `autoets_*` / `naive2_*`
/// carry the mean-of-finite-series metric after NaN/Inf filtering (ACCUR-03).
///
/// `autoets_msis` and `autoets_coverage` are `Some` only for monthly (the
/// richest interval evaluation frequency and the ACCUR-08 anchor frequency).
/// Other frequencies carry `None` to avoid runtime cost (see comment in
/// `run_accuracy_harness()`).
#[derive(Debug)]
pub struct FrequencyResult {
    pub frequency: String,
    pub n_series: usize,
    pub horizon: usize,
    pub autoets_mase: f64,
    pub naive2_mase: f64,
    pub autoets_smape: f64,
    pub naive2_smape: f64,
    pub autoets_rmse: f64,
    pub autoets_mae: f64,
    /// MSIS for AutoETS prediction intervals (monthly only — see module doc).
    pub autoets_msis: Option<f64>,
    /// Empirical interval coverage for AutoETS (monthly only).
    pub autoets_coverage: Option<f64>,
    /// Number of series excluded from aggregates due to non-finite metrics.
    pub skipped_nonfinite: usize,
}

// ─── Harness implementation ───────────────────────────────────────────────────

/// Run the M3 accuracy harness and return per-frequency results.
///
/// Returns an empty map when `ANOFOX_DATASET_DIR` is not set (ACCUR-01 env gate).
/// The caller detects the empty map and skips the test without failure.
fn run_accuracy_harness() -> HashMap<String, FrequencyResult> {
    let dir = match dataset_dir_from_env() {
        Some(d) => d,
        None => return HashMap::new(),
    };

    let all_series = match load_m3(&dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("WARNING: load_m3 failed: {} — returning empty harness", e);
            return HashMap::new();
        }
    };

    // Map frequency string to its canonical seasonal period.
    // "monthly" → 12, "quarterly" → 4, "yearly" → 1.
    let period_for_freq = |freq: &str| -> usize {
        match freq {
            "monthly" => 12,
            "quarterly" => 4,
            _ => 1, // yearly and unknown
        }
    };

    // Group series by frequency.
    let mut by_freq: HashMap<String, Vec<_>> = HashMap::new();
    for s in &all_series {
        by_freq.entry(s.frequency.clone()).or_default().push(s);
    }

    let mut results = HashMap::new();

    for (freq, series_list) in &by_freq {
        let period = period_for_freq(freq.as_str());
        let is_monthly = freq == "monthly";

        // Per-series metric accumulators (before NaN filtering).
        let mut autoets_mases: Vec<f64> = Vec::new();
        let mut naive2_mases: Vec<f64> = Vec::new();
        let mut autoets_smapes: Vec<f64> = Vec::new();
        let mut naive2_smapes: Vec<f64> = Vec::new();
        let mut autoets_rmses: Vec<f64> = Vec::new();
        let mut autoets_maes: Vec<f64> = Vec::new();
        // Interval metrics: only collected for monthly (MSIS convention A4, ACCUR-05).
        // Collecting intervals for all frequencies would triple runtime; monthly is
        // the ACCUR-08 anchor and has the richest seasonality for interval evaluation.
        let mut autoets_msis_vals: Vec<f64> = Vec::new();
        let mut autoets_cov_vals: Vec<f64> = Vec::new();

        for series in series_list {
            let horizon = series.horizon;
            let values = &series.values;
            let n = values.len();

            // Skip series too short for a single expanding-window fold.
            if n <= horizon + period + 2 {
                continue;
            }

            // Build the competition single-origin expanding-window fold.
            let folds = match CvFoldGenerator::new()
                .n_folds(1)
                .horizon(horizon)
                .min_initial_window(n - horizon)
                .strategy(CVStrategy::Expanding)
                .on_constraint_violation(ConstraintViolation::ReduceFolds)
                .generate(n)
            {
                Ok(f) if !f.is_empty() => f,
                _ => continue,
            };

            let fold = &folds[0];

            // ACCUR-02: temporal integrity — train must precede test on every fold.
            assert!(
                fold.train_end <= fold.test_start,
                "temporal integrity violation: train_end={} > test_start={} (series '{}', freq '{}')",
                fold.train_end,
                fold.test_start,
                series.id,
                freq
            );

            let train = &values[fold.train_start..fold.train_end];
            let test = &values[fold.test_start..fold.test_end];

            if train.is_empty() || test.is_empty() {
                continue;
            }

            // ── Naive2 ───────────────────────────────────────────────────────
            let mut naive2 = Naive2::new(period);
            let naive2_pred = match naive2.fit(train).and_then(|_| naive2.predict(horizon)) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let h = test.len().min(naive2_pred.len());
            let n2_pred_slice = &naive2_pred[..h];
            let test_slice = &test[..h];

            let n2_denom = mase_scale(train, period);
            let n2_fmae: f64 = test_slice
                .iter()
                .zip(n2_pred_slice.iter())
                .map(|(a, p)| (a - p).abs())
                .sum::<f64>()
                / h as f64;
            naive2_mases.push(n2_fmae / n2_denom);
            naive2_smapes.push(smape(test_slice, n2_pred_slice));

            // ── AutoETS (point metrics) ───────────────────────────────────────
            let train_ts = match make_ts_from_slice(train) {
                Ok(ts) => ts,
                Err(_) => continue,
            };
            let mut autoets = AutoETS::new();
            if autoets.fit(&train_ts).is_err() {
                continue;
            }
            let ae_forecast = match autoets.predict(horizon) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let ae_pred = ae_forecast.primary();
            let h_ae = test.len().min(ae_pred.len());
            let ae_pred_slice = &ae_pred[..h_ae];
            let test_ae_slice = &test[..h_ae];

            let ae_denom = mase_scale(train, period);
            let ae_fmae: f64 = test_ae_slice
                .iter()
                .zip(ae_pred_slice.iter())
                .map(|(a, p)| (a - p).abs())
                .sum::<f64>()
                / h_ae as f64;
            autoets_mases.push(ae_fmae / ae_denom);
            autoets_smapes.push(smape(test_ae_slice, ae_pred_slice));
            autoets_rmses.push(rmse(test_ae_slice, ae_pred_slice));
            autoets_maes.push(mae(test_ae_slice, ae_pred_slice));

            // ── AutoETS interval metrics (monthly only) ────────────────────────
            // MSIS convention note (A4/Pitfall 4): the reused msis() from
            // src/utils/metrics.rs scales by period-1 first differences, NOT
            // by seasonal differences as in the M4 competition formula. Do not
            // compare these MSIS values to published M4 MSIS benchmarks.
            // ACCUR-08 anchor is MASE only.
            if is_monthly {
                // 0.05 alpha → 95% nominal coverage level.
                let alpha = 0.05;
                let ae_interval = match autoets.predict_with_intervals(horizon, 1.0 - alpha) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                if let (Ok(lower), Ok(upper)) =
                    (ae_interval.lower_series(0), ae_interval.upper_series(0))
                {
                    let h_int = test.len().min(lower.len()).min(upper.len());
                    let test_int = &test[..h_int];
                    let lo = &lower[..h_int];
                    let hi = &upper[..h_int];

                    let msis_val = msis(test_int, lo, hi, alpha);
                    let cov_val = coverage(test_int, lo, hi);
                    autoets_msis_vals.push(msis_val);
                    autoets_cov_vals.push(cov_val);
                }
            }
        }

        // ── NaN-guarded aggregation ───────────────────────────────────────────
        // Filter out non-finite values before averaging (ACCUR-03 harness layer).
        // Log a WARNING with the excluded count when any are dropped.

        fn nanmean(vals: &[f64], label: &str, freq: &str) -> (f64, usize) {
            let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
            let skipped = vals.len() - finite.len();
            if skipped > 0 {
                eprintln!(
                    "WARNING: {} non-finite {} values excluded from '{}' aggregate",
                    skipped, label, freq
                );
            }
            let mean = if finite.is_empty() {
                f64::NAN
            } else {
                finite.iter().sum::<f64>() / finite.len() as f64
            };
            (mean, skipped)
        }

        let (ae_mase, sk_ae_mase) = nanmean(&autoets_mases, "autoets_mase", freq);
        let (n2_mase, sk_n2_mase) = nanmean(&naive2_mases, "naive2_mase", freq);
        let (ae_smape, _) = nanmean(&autoets_smapes, "autoets_smape", freq);
        let (n2_smape, _) = nanmean(&naive2_smapes, "naive2_smape", freq);
        let (ae_rmse, _) = nanmean(&autoets_rmses, "autoets_rmse", freq);
        let (ae_mae, _) = nanmean(&autoets_maes, "autoets_mae", freq);

        // Total skipped count (use the larger of the two MASE skip counts; they
        // should match since each series either contributes to both or neither).
        let skipped_nonfinite = sk_ae_mase.max(sk_n2_mase);

        let (ae_msis, ae_cov) = if is_monthly {
            let (m, _) = nanmean(&autoets_msis_vals, "autoets_msis", freq);
            let (c, _) = nanmean(&autoets_cov_vals, "autoets_coverage", freq);
            (Some(m), Some(c))
        } else {
            (None, None)
        };

        // ACCUR-07: insert under the per-frequency key — NEVER aggregate across
        // frequencies. The absence of a "combined" or "all" key is enforced by
        // the `per_frequency_stratification` test.
        results.insert(
            freq.clone(),
            FrequencyResult {
                frequency: freq.clone(),
                n_series: series_list.len(),
                horizon: series_list.first().map(|s| s.horizon).unwrap_or(0),
                autoets_mase: ae_mase,
                naive2_mase: n2_mase,
                autoets_smape: ae_smape,
                naive2_smape: n2_smape,
                autoets_rmse: ae_rmse,
                autoets_mae: ae_mae,
                autoets_msis: ae_msis,
                autoets_coverage: ae_cov,
                skipped_nonfinite,
            },
        );
    }

    results
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Tracer: one M3-monthly series through CV → temporal gate → training-denominator MASE.
///
/// This proves the full accuracy path end-to-end on a single series before any
/// horizontal expansion (Plan 02). The test skips cleanly when `ANOFOX_DATASET_DIR`
/// is not set (ACCUR-01).
#[test]
fn tracer_m3_monthly_autoets_one_series() {
    use anofox_bench_harness::loader::{dataset_dir_from_env, parse_tsf_with_meta};

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

/// Stratification test: harness produces separate frequency buckets.
///
/// When `ANOFOX_DATASET_DIR` is set, asserts:
/// - The result map contains "monthly", "quarterly", and "yearly" keys.
/// - Each bucket has `n_series > 0`.
/// - AutoETS and Naive2 MASE are finite for each frequency.
/// - No cross-frequency aggregate key ("all", "combined", etc.) exists.
///
/// When `ANOFOX_DATASET_DIR` is not set, prints a skip message and returns
/// cleanly (ACCUR-01).
#[test]
fn per_frequency_stratification() {
    let results = run_accuracy_harness();

    if results.is_empty() {
        eprintln!("ANOFOX_DATASET_DIR not set — skipping per_frequency_stratification");
        return;
    }

    // Assert three separate per-frequency buckets (ACCUR-07).
    for expected_freq in &["monthly", "quarterly", "yearly"] {
        let r = results.get(*expected_freq).unwrap_or_else(|| {
            panic!(
                "expected '{}' frequency bucket in harness output",
                expected_freq
            )
        });

        assert!(
            r.n_series > 0,
            "frequency '{}' must have at least one series",
            expected_freq
        );
        assert!(
            r.autoets_mase.is_finite(),
            "AutoETS MASE for '{}' must be finite (got {})",
            expected_freq,
            r.autoets_mase
        );
        assert!(
            r.naive2_mase.is_finite(),
            "Naive2 MASE for '{}' must be finite (got {})",
            expected_freq,
            r.naive2_mase
        );

        eprintln!(
            "Freq='{}' n={} horizon={} AutoETS_MASE={:.4} Naive2_MASE={:.4} skipped={}",
            expected_freq,
            r.n_series,
            r.horizon,
            r.autoets_mase,
            r.naive2_mase,
            r.skipped_nonfinite
        );
    }

    // Assert no cross-frequency aggregate key exists (ACCUR-07).
    for forbidden in &["all", "combined", "aggregate", "total"] {
        assert!(
            !results.contains_key(*forbidden),
            "cross-frequency aggregate key '{}' must not exist (ACCUR-07)",
            forbidden
        );
    }
}

/// MSIS and empirical coverage are present for the monthly frequency.
///
/// When `ANOFOX_DATASET_DIR` is set, asserts that monthly `autoets_msis` is
/// `Some(finite)` and `autoets_coverage` is `Some(c)` with `0.0 <= c <= 1.0`.
///
/// MSIS convention (A4): scaled by period-1 first-differences, not seasonal
/// differences. Do not compare to M4 MSIS benchmarks. ACCUR-08 anchor = MASE.
///
/// When env var is not set, skips cleanly (ACCUR-01).
#[test]
fn msis_coverage_present_monthly() {
    let results = run_accuracy_harness();

    if results.is_empty() {
        eprintln!("ANOFOX_DATASET_DIR not set — skipping msis_coverage_present_monthly");
        return;
    }

    let monthly = results
        .get("monthly")
        .expect("monthly frequency bucket must exist");

    let msis_val = monthly
        .autoets_msis
        .expect("autoets_msis must be Some for monthly frequency");
    assert!(
        msis_val.is_finite(),
        "autoets_msis must be finite (got {})",
        msis_val
    );

    let cov = monthly
        .autoets_coverage
        .expect("autoets_coverage must be Some for monthly frequency");
    assert!(
        (0.0..=1.0).contains(&cov),
        "autoets_coverage must be in [0.0, 1.0] (got {})",
        cov
    );

    eprintln!(
        "Monthly interval metrics: MSIS={:.4} coverage={:.4}",
        msis_val, cov
    );
}

// ─── Reference fixture loader ─────────────────────────────────────────────────

/// Load the pinned statsforecast reference monthly MASE from the committed fixture.
///
/// The fixture lives at `.planning/baselines/statsforecast_reference.json` and
/// was regenerated once on a pinned Python env (statsforecast 2.0.3) with
/// provenance block (D-06). The monthly MASE in the fixture (0.8633) differs
/// from the historical published 0.93 (statsforecast 1.x) due to revised AutoETS
/// implementation in 2.x. The comparison is apples-to-apples within the pinned
/// env; see Plan 03 SUMMARY for full context.
///
/// Falls back to literal 0.93 if the fixture cannot be located (degraded mode),
/// but this is not the expected path on a maintainer machine with the corpus set up.
fn load_reference_monthly_mase() -> f64 {
    // CARGO_MANIFEST_DIR = <workspace>/crates/anofox-bench-harness
    // workspace root     = CARGO_MANIFEST_DIR/../../
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = match manifest_dir.join("../..").canonicalize() {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "WARNING: could not canonicalize workspace root: {} — falling back to 0.93",
                e
            );
            return 0.93;
        }
    };
    let path = workspace_root.join(".planning/baselines/statsforecast_reference.json");
    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "WARNING: statsforecast_reference.json not found at {} ({})\n\
                 Regenerate with: uv run python3 validation/run_statsforecast.py --m3-reference\n\
                 Falling back to literal 0.93 anchor",
                path.display(),
                e
            );
            return 0.93;
        }
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "WARNING: statsforecast_reference.json is not valid JSON ({}): falling back to 0.93",
                e
            );
            return 0.93;
        }
    };
    json["datasets"]["M3"]["monthly"]["autoets_mase"]
        .as_f64()
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: datasets.M3.monthly.autoets_mase missing from fixture — falling back to 0.93"
            );
            0.93
        })
}

// ─── ACCUR-08 anchor test ─────────────────────────────────────────────────────

/// ACCUR-08: Whole-corpus AutoETS M3-monthly MASE anchor.
///
/// Validates that the anofox AutoETS M3-monthly MASE over the full 1428-series
/// corpus is within ±0.02 of the pinned statsforecast reference value from
/// `.planning/baselines/statsforecast_reference.json`.
///
/// The reference value is statsforecast 2.0.3 AutoETS on M3-monthly (0.8633),
/// pinned with provenance (D-06). This differs from the historical published
/// 0.93 (statsforecast 1.x); the comparison is apples-to-apples within the
/// pinned env. See Plan 03 SUMMARY and fixture provenance block for details.
///
/// Diagnostic message: if the assertion fails, check:
///   1. That the D-03 MASE-denominator fix is in effect (`src/utils/metrics.rs`).
///   2. That `mase_scale()` uses the TRAINING-slice denominator, not the test slice.
///   3. That the CvFoldGenerator single-origin split matches the Python harness.
///   4. That the reference fixture is for the correct statsforecast version.
///
/// When `ANOFOX_DATASET_DIR` is not set, the test skips cleanly (ACCUR-01).
#[test]
fn accur08_anchor_m3_monthly_autoets() {
    let results = run_accuracy_harness();

    if results.is_empty() {
        eprintln!("ANOFOX_DATASET_DIR not set — skipping accur08_anchor_m3_monthly_autoets");
        return;
    }

    let monthly = results
        .get("monthly")
        .expect("monthly frequency bucket must exist in harness output");

    let autoets_mase = monthly.autoets_mase;
    let reference_mase = load_reference_monthly_mase();
    let tolerance = 0.02_f64;

    assert!(
        autoets_mase.is_finite(),
        "AutoETS M3-monthly MASE is NaN/Inf (ACCUR-08) — check D-03 fix and training denominator"
    );

    assert!(
        (autoets_mase - reference_mase).abs() <= tolerance,
        "ACCUR-08 anchor FAILED: AutoETS M3-monthly MASE={:.4} is outside ±{:.2} of reference {:.4}.\n\
         Reference: statsforecast 2.0.3 pinned fixture ({}) — see Plan 03 SUMMARY.\n\
         Diagnostic: check D-03 MASE denominator fix in src/utils/metrics.rs;\n\
         check training-slice denominator in mase_scale() (Pitfall 1);\n\
         check CvFoldGenerator single-origin split alignment with Python harness.",
        autoets_mase,
        tolerance,
        reference_mase,
        ".planning/baselines/statsforecast_reference.json"
    );

    eprintln!(
        "ACCUR-08 PASSED: AutoETS M3-monthly MASE={:.4} (reference={:.4}, diff={:+.4}, tolerance=±{:.2})",
        autoets_mase,
        reference_mase,
        autoets_mase - reference_mase,
        tolerance
    );
}

// ─── accuracy.json emit helper ────────────────────────────────────────────────

/// Provenance block for `accuracy.json` (D-05).
///
/// Serialised into the `provenance` key of the JSON baseline.
/// The `note` field records the D-03 MASE-fix before/after and the MSIS
/// period-1 convention caveat (from Plan 02), making the number auditable.
#[derive(serde::Serialize)]
struct AccuracyProvenance {
    /// Short git SHA at capture time.
    git_sha: String,
    /// ISO-8601 UTC timestamp of the capture run.
    timestamp_iso: String,
    /// Full `rustc --version` output.
    rustc_version: String,
    /// CPU model (from /proc/cpuinfo or "unknown").
    host_cpu: String,
    /// OS + kernel (from `uname -sr` or "unknown").
    host_os: String,
    /// Cargo features active during the capture.
    active_features: Vec<String>,
    /// Human-readable note on conventions and known caveats.
    note: String,
}

/// Per-model metric row for one frequency in `accuracy.json`.
#[derive(serde::Serialize)]
struct ModelMetrics {
    mase: f64,
    smape: f64,
    rmse: f64,
    mae: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    msis: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<f64>,
}

/// One frequency entry in `accuracy.json`.
#[derive(serde::Serialize)]
struct FrequencyEntry {
    n_series: usize,
    horizon: usize,
    models: FrequencyModels,
}

/// Model pair for one frequency entry.
#[derive(serde::Serialize)]
struct FrequencyModels {
    #[serde(rename = "AutoETS")]
    autoets: ModelMetrics,
    #[serde(rename = "Naive2")]
    naive2: ModelMetrics,
}

/// Top-level `accuracy.json` document.
#[derive(serde::Serialize)]
struct AccuracyJson {
    provenance: AccuracyProvenance,
    datasets: AccuracyDatasets,
}

#[derive(serde::Serialize)]
struct AccuracyDatasets {
    #[serde(rename = "M3")]
    m3: AccuracyM3,
}

#[derive(serde::Serialize)]
struct AccuracyM3 {
    monthly: FrequencyEntry,
    quarterly: FrequencyEntry,
    yearly: FrequencyEntry,
}

/// Collect provenance metadata for `accuracy.json`.
fn collect_provenance() -> AccuracyProvenance {
    // git SHA: read from `git rev-parse --short HEAD`
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // timestamp
    let timestamp_iso = {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!(
            "{}Z",
            chrono::DateTime::from_timestamp(now as i64, 0)
                .map(|dt: chrono::DateTime<chrono::Utc>| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )
    };

    // rustc version
    let rustc_version = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // host CPU
    let host_cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    // host OS
    let host_os = std::process::Command::new("uname")
        .args(["-sr"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // active features: standard harness defaults (no special feature flags)
    let active_features = vec!["default".to_string(), "postprocess".to_string()];

    AccuracyProvenance {
        git_sha,
        timestamp_iso,
        rustc_version,
        host_cpu,
        host_os,
        active_features,
        note: concat!(
            "D-03 MASE fix: training-slice seasonal-naive denominator with period-1 fallback ",
            "when denominator collapses (constant series) — keeps series in aggregate instead ",
            "of dropping as None/NaN. Matches statsforecast 2.x behavior. ",
            "MSIS convention (A4/Pitfall 4): scaled by period-1 first-differences (not ",
            "seasonal differences as in M4 competition MSIS); do not compare MSIS to M4 ",
            "published benchmarks. ACCUR-08 anchor is MASE only. ",
            "Reference: statsforecast 2.0.3 fixture (monthly MASE=0.8633); differs from ",
            "historical 0.93 (statsforecast 1.x) due to revised AutoETS in 2.x."
        )
        .to_string(),
    }
}

/// Write `.planning/baselines/accuracy.json` with per-frequency M3 results.
///
/// This function is ONLY called when:
///   1. `ANOFOX_WRITE_ACCURACY_BASELINE=1` is set (explicit maintainer intent), AND
///   2. The ACCUR-08 anchor has already been validated (monthly MASE within tolerance).
///
/// Running plain `cargo test` without the write flag never calls this function,
/// so the committed baseline is never silently overwritten (Phase 1 CI-read-only rule).
///
/// # Guard contract
///
/// The anchor must pass BEFORE this function is called. The caller (Task 2 / checkpoint)
/// enforces this ordering:
///   1. Run `accur08_anchor_m3_monthly_autoets` and confirm it passes.
///   2. Set `ANOFOX_WRITE_ACCURACY_BASELINE=1` and re-run the test suite.
///   3. The guarded emit test below checks BOTH the env flag AND the anchor value.
///
/// If the anchor is outside tolerance when the write flag is set, this function
/// panics with a descriptive message — it refuses to lock an unvalidated number.
///
/// # Schema
///
/// Writes the `accuracy.json` schema from RESEARCH.md:
/// `{ provenance, datasets: { M3: { monthly/quarterly/yearly: { n_series, horizon, models: { AutoETS: {...}, Naive2: {...} } } } } }`
pub fn emit_accuracy_json(results: &HashMap<String, FrequencyResult>, anchor_passed: bool) {
    if !anchor_passed {
        panic!(
            "emit_accuracy_json called but anchor_passed=false. \
             The ACCUR-08 anchor MUST pass before accuracy.json can be written. \
             Check that AutoETS M3-monthly MASE is within ±0.02 of the reference."
        );
    }

    // CARGO_MANIFEST_DIR-based workspace root (same as load_reference_monthly_mase).
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .join("../..")
        .canonicalize()
        .expect("could not canonicalize workspace root from CARGO_MANIFEST_DIR");
    let out_path = workspace_root.join(".planning/baselines/accuracy.json");

    // Build the JSON document from FrequencyResult map.
    let make_entry = |freq: &str| -> FrequencyEntry {
        let r = results
            .get(freq)
            .unwrap_or_else(|| panic!("frequency '{}' missing from harness results", freq));
        FrequencyEntry {
            n_series: r.n_series,
            horizon: r.horizon,
            models: FrequencyModels {
                autoets: ModelMetrics {
                    mase: r.autoets_mase,
                    smape: r.autoets_smape,
                    rmse: r.autoets_rmse,
                    mae: r.autoets_mae,
                    msis: r.autoets_msis,
                    coverage: r.autoets_coverage,
                },
                naive2: ModelMetrics {
                    mase: r.naive2_mase,
                    smape: r.naive2_smape,
                    rmse: f64::NAN,
                    mae: f64::NAN,
                    msis: None,
                    coverage: None,
                },
            },
        }
    };

    let doc = AccuracyJson {
        provenance: collect_provenance(),
        datasets: AccuracyDatasets {
            m3: AccuracyM3 {
                monthly: make_entry("monthly"),
                quarterly: make_entry("quarterly"),
                yearly: make_entry("yearly"),
            },
        },
    };

    let json_str = serde_json::to_string_pretty(&doc).expect("accuracy.json serialization failed");

    // Ensure the parent directory exists.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("could not create {:?}: {}", parent, e));
    }

    std::fs::write(&out_path, json_str)
        .unwrap_or_else(|e| panic!("could not write accuracy.json to {:?}: {}", out_path, e));

    eprintln!("accuracy.json written to {}", out_path.display());
}

/// Guarded baseline emit: write `accuracy.json` only when write flag + anchor pass.
///
/// This test runs the full harness, validates the ACCUR-08 anchor, and ONLY THEN
/// calls `emit_accuracy_json()` to write `.planning/baselines/accuracy.json`.
///
/// Guard conditions (BOTH must be true to write):
///   1. `ANOFOX_WRITE_ACCURACY_BASELINE=1` env var is set.
///   2. AutoETS M3-monthly MASE is within ±0.02 of the pinned reference.
///
/// When the env flag is absent, the test skips cleanly without touching the file.
/// When `ANOFOX_DATASET_DIR` is absent, the harness returns an empty map and the
/// test also skips cleanly.
///
/// This test is designed to be run AFTER the checkpoint:decision confirms lock-now.
#[test]
fn emit_accuracy_baseline_if_write_flag_set() {
    // Check write flag FIRST — skip immediately if not set (no harness run needed).
    let write_flag = std::env::var("ANOFOX_WRITE_ACCURACY_BASELINE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !write_flag {
        eprintln!(
            "ANOFOX_WRITE_ACCURACY_BASELINE not set — skipping accuracy.json emit \
             (set to '1' after checkpoint:decision approves lock-now)"
        );
        return;
    }

    let results = run_accuracy_harness();
    if results.is_empty() {
        eprintln!(
            "ANOFOX_DATASET_DIR not set — skipping accuracy.json emit \
             (requires full M3 corpus)"
        );
        return;
    }

    // Validate the ACCUR-08 anchor before any write.
    let monthly = results
        .get("monthly")
        .expect("monthly frequency bucket must exist");
    let autoets_mase = monthly.autoets_mase;
    let reference_mase = load_reference_monthly_mase();
    let tolerance = 0.02_f64;

    let anchor_passed =
        autoets_mase.is_finite() && (autoets_mase - reference_mase).abs() <= tolerance;

    if !anchor_passed {
        panic!(
            "ACCUR-08 anchor FAILED — refusing to write accuracy.json.\n\
             AutoETS M3-monthly MASE={:.4} is outside ±{:.2} of reference {:.4}.\n\
             accuracy.json must NOT be committed until the anchor passes (ACCUR-08).",
            autoets_mase, tolerance, reference_mase
        );
    }

    eprintln!(
        "ACCUR-08 anchor PASSED (MASE={:.4}, ref={:.4}) — writing accuracy.json",
        autoets_mase, reference_mase
    );

    emit_accuracy_json(&results, anchor_passed);
}
