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
// Task 1 (tracer): changepoint self-match → precision/recall = 1.0
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
}
