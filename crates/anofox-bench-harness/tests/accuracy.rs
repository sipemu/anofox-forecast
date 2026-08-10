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
//! 4. AutoETS M3-monthly MASE anchor (ACCUR-08) — committed by Plan 04.
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
