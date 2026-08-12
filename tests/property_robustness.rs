//! Property-based robustness tests for known-fragile numerical paths (ROBUST-03).
//!
//! Covers three areas identified in the ROADMAP as requiring property-test coverage:
//! - `changepoint::metrics` — reflexivity invariants (precision/recall/hausdorff/randindex)
//! - `seasonality::MSTL::decompose` — no-panic, no-NaN in components for bounded random slices
//! - `utils::CvFoldGenerator::generate` — no-panic, temporal integrity for random parameters
//!
//! All proptest blocks are bounded (`with_cases` ≤ 100) so the suite runs under ~30 s.
//! The `.proptest-regressions/` corpus directory is committed alongside this file so
//! any discovered shrunk counterexample replays deterministically on CI (ROBUST-03 / T-03-07).

use anofox_forecast::changepoint::{hausdorff, precision_recall, randindex};
use anofox_forecast::seasonality::MSTL;
use anofox_forecast::utils::CvFoldGenerator;
use proptest::prelude::*;

// =============================================================================
// Helper: generate strictly-increasing valid breakpoints with terminal == n
// =============================================================================

/// Build `n_bkps` strictly-increasing breakpoints with the terminal element equal to `n`.
///
/// Uses evenly-spaced internal breakpoints so every input is well-formed and
/// exercisable by the metric happy-path (malformed inputs are tested deterministically
/// in edge_case_robustness.rs, not via proptest).
fn make_bkps(n: usize, n_bkps: usize) -> Vec<usize> {
    let step = n / (n_bkps + 1);
    let mut bkps: Vec<usize> = (1..=n_bkps).map(|i| i * step).collect();
    bkps.push(n);
    bkps
}

// =============================================================================
// Changepoint metrics: reflexivity invariants (with_cases(50))
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Predicting the true breakpoints exactly yields perfect precision and recall.
    #[test]
    fn changepoint_precision_recall_self_match_is_perfect(
        n in 50usize..200,
        n_bkps in 1usize..5
    ) {
        let bkps = make_bkps(n, n_bkps);
        let pr = precision_recall(&bkps, &bkps, 0).unwrap();
        prop_assert!((pr.precision - 1.0).abs() < 1e-10,
            "precision should be 1.0, got {}", pr.precision);
        prop_assert!((pr.recall - 1.0).abs() < 1e-10,
            "recall should be 1.0, got {}", pr.recall);
        prop_assert!(pr.f1.is_finite(), "f1 is NaN/Inf: {}", pr.f1);
    }

    /// Hausdorff distance of any breakpoint set with itself is exactly 0.
    #[test]
    fn changepoint_hausdorff_reflexive(n in 10usize..200, n_bkps in 1usize..4) {
        let bkps = make_bkps(n, n_bkps);
        let h = hausdorff(&bkps, &bkps).unwrap();
        prop_assert!((h - 0.0).abs() < 1e-12,
            "hausdorff reflexive should be 0.0, got {}", h);
    }

    /// Rand index of any segmentation against itself is exactly 1.0.
    #[test]
    fn changepoint_randindex_self_is_one(n in 10usize..200, n_bkps in 1usize..4) {
        let bkps = make_bkps(n, n_bkps);
        let r = randindex(&bkps, &bkps, n).unwrap();
        prop_assert!((r - 1.0).abs() < 1e-10,
            "randindex self should be 1.0, got {}", r);
        prop_assert!(r.is_finite(), "randindex is NaN/Inf");
    }
}

// =============================================================================
// MSTL decomposition: no-panic, no-NaN (with_cases(30))
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// MSTL::decompose never panics for any bounded f64 slice; when it returns Some,
    /// all components are finite and lengths equal the input length.
    #[test]
    fn mstl_decompose_never_panics(
        values in prop::collection::vec(-1000.0f64..1000.0, 10..100),
        period in 2usize..8
    ) {
        let mstl = MSTL::new(vec![period]);
        // None is a valid return for short series (n < 2 * period); must never panic.
        let result = mstl.decompose(&values);
        if let Some(r) = result {
            // All trend values must be finite.
            for v in &r.trend {
                prop_assert!(v.is_finite(), "NaN/Inf in trend: {}", v);
            }
            // All seasonal component values must be finite.
            for seasonal in &r.seasonal_components {
                for v in seasonal {
                    prop_assert!(v.is_finite(), "NaN/Inf in seasonal component: {}", v);
                }
            }
            // All remainder values must be finite.
            for v in &r.remainder {
                prop_assert!(v.is_finite(), "NaN/Inf in remainder: {}", v);
            }
            // Length invariant.
            prop_assert_eq!(r.trend.len(), values.len(),
                "trend.len() != values.len()");
        }
    }

    /// MSTL on a constant series produces finite components (or None for short series).
    #[test]
    fn mstl_decompose_constant_series(
        n in 10usize..100,
        c in -1000.0f64..1000.0,
        period in 2usize..8
    ) {
        let values = vec![c; n];
        let mstl = MSTL::new(vec![period]);
        let result = mstl.decompose(&values);
        if let Some(r) = result {
            prop_assert!(r.trend.iter().all(|v| v.is_finite()),
                "NaN/Inf in constant-series trend");
            prop_assert!(r.remainder.iter().all(|v| v.is_finite()),
                "NaN/Inf in constant-series remainder");
        }
    }
}

// =============================================================================
// CV fold generator: no-panic, temporal integrity (with_cases(100))
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// CvFoldGenerator::generate never panics for any bounded parameter combination.
    /// When it returns Ok(folds), every fold satisfies temporal integrity:
    ///   - train_end <= test_start  (no leakage from test back into training)
    ///   - test_end <= series_len   (fold does not extend beyond the series)
    ///   - train_size() >= min_window  (minimum training window is respected)
    #[test]
    fn cv_generate_never_panics(
        series_len in 0usize..500,
        horizon in 1usize..50,
        n_folds in 1usize..10,
        min_window in 2usize..50,
        gap in 0usize..10
    ) {
        let result = CvFoldGenerator::new()
            .n_folds(n_folds)
            .horizon(horizon)
            .min_initial_window(min_window)
            .gap(gap)
            .generate(series_len);
        // Ok or Err are both valid outcomes; panic is not.
        if let Ok(folds) = result {
            for fold in &folds {
                prop_assert!(
                    fold.train_end <= fold.test_start,
                    "temporal leakage: train_end {} > test_start {}",
                    fold.train_end,
                    fold.test_start
                );
                prop_assert!(
                    fold.test_end <= series_len,
                    "test_end {} > series_len {}",
                    fold.test_end,
                    series_len
                );
                prop_assert!(
                    fold.train_size() >= min_window,
                    "train_size {} < min_window {}",
                    fold.train_size(),
                    min_window
                );
            }
        }
    }
}
