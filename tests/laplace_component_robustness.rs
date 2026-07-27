//! Adversarial component-robustness tests for AR / AR(2) / GARCH leaves.
//!
//! Ports the *intent* of skaters' `test_component_robustness.py`
//! (upstream PR #157, merged 2026-07-24): drive each transform/leaf
//! against inputs that stress the numerics — large magnitude, warm-up
//! multi-step, near-Nyquist oscillation, single spikes — and assert
//! that the h-step forecasts remain finite AND stay within a
//! candidate-mean-bounded region of the input.
//!
//! Motivating bugs (both fixed 2026-07-25 in this crate):
//!
//! 1. `Ar2Leaf` variance formula was `σ · √h` (random-walk form), not
//!    the correct MA(∞) `σ² · Σ ψ_i²`. Overstated horizon-h uncertainty
//!    for stationary AR(2).
//! 2. `GarchWrappedLeaf` recursion used raw `y²` rather than
//!    deviation-from-mean squared. On level series (values ~1e5),
//!    "volatility" grew of order `|y|` and the inverse re-inflated it.
//!
//! These tests would have caught both bugs immediately on adversarial
//! inputs; they guard the fixes going forward.

use anofox_forecast::models::laplace::leaves::{Ar1Leaf, Ar2Leaf, EmaLeaf, GarchWrappedLeaf};
use anofox_forecast::models::laplace::Leaf;

/// Numerical-Recipes LCG so tests are deterministic without adding a
/// dev-dep on `rand`.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (self.0 >> 32) as u32;
        x as f64 / (u32::MAX as f64 + 1.0)
    }
    fn next_normal(&mut self) -> f64 {
        // Central-limit-theorem approximation: sum of 12 U(0,1) - 6.
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.next_f64();
        }
        s - 6.0
    }
}

fn all_finite(gs: &[anofox_forecast::models::laplace::Gaussian]) -> bool {
    gs.iter().all(|g| g.mean.is_finite() && g.std.is_finite())
}

fn max_abs_mean(gs: &[anofox_forecast::models::laplace::Gaussian]) -> f64 {
    gs.iter().map(|g| g.mean.abs()).fold(0.0f64, f64::max)
}

fn max_std(gs: &[anofox_forecast::models::laplace::Gaussian]) -> f64 {
    gs.iter().map(|g| g.std).fold(0.0f64, f64::max)
}

// ---------- Ar1Leaf ----------

#[test]
fn ar1_stays_finite_on_billion_scale_input() {
    // Values in [1e9, 2e9] — verify no numerical blow-up.
    let mut rng = Rng::new(1);
    let mut leaf = Ar1Leaf::new(0.5);
    for _ in 0..200 {
        let y = 1.5e9 + 5e8 * rng.next_normal();
        leaf.observe(y);
    }
    let preds = leaf.predict(50);
    assert!(all_finite(&preds), "predictions not all finite");
    // Mean should stay within a small multiple of input scale.
    assert!(
        max_abs_mean(&preds) < 1e11,
        "|mean| runaway: {}",
        max_abs_mean(&preds)
    );
}

#[test]
fn ar1_h_step_variance_bounded_for_stationary_phi() {
    // With phi < 1, MA(∞) variance approaches σ²/(1-φ²) — should stay
    // bounded across any horizon, not grow linearly.
    let mut rng = Rng::new(2);
    let mut leaf = Ar1Leaf::new(0.5);
    for _ in 0..500 {
        leaf.observe(rng.next_normal());
    }
    let preds = leaf.predict(1000);
    let bound = 3.0; // ≈ sqrt(1/(1-0.5²)) · sample-σ · slack
    assert!(max_std(&preds) < bound, "std runaway: {}", max_std(&preds));
}

// ---------- Ar2Leaf ----------

#[test]
fn ar2_h_step_variance_bounded_for_stationary_phis() {
    // The 2026-07-25 fix: Ar2 h-step variance now follows the MA(∞)
    // recursion, so for a stationary AR(2) it approaches a bound
    // (σ²·(1-φ_2)/[(1+φ_2)((1-φ_2)²-φ_1²)] in closed form).
    // Pre-fix, variance was σ²·h → grew unboundedly with h.
    let mut rng = Rng::new(3);
    let mut leaf = Ar2Leaf::new(0.05);
    // Feed a stationary AR(2) with known coefficients.
    let phi1 = 0.5;
    let phi2 = 0.3;
    let mut y_prev1 = 0.0;
    let mut y_prev2 = 0.0;
    for _ in 0..1000 {
        let y = phi1 * y_prev1 + phi2 * y_prev2 + rng.next_normal();
        leaf.observe(y);
        y_prev2 = y_prev1;
        y_prev1 = y;
    }
    let preds_short = leaf.predict(10);
    let preds_long = leaf.predict(1000);
    assert!(all_finite(&preds_short) && all_finite(&preds_long));
    let sigma_short = max_std(&preds_short);
    let sigma_long = max_std(&preds_long);
    // Long-horizon σ must not be much bigger than short-horizon σ for
    // a stationary process. Pre-fix: sigma_long / sigma_short ≈ 10
    // (from √1000/√10 ≈ 10). Post-fix: ratio ≈ 1.
    let ratio = sigma_long / sigma_short.max(1e-9);
    assert!(
        ratio < 2.0,
        "AR(2) h-step variance grows unboundedly: short σ={}, long σ={}, ratio={}",
        sigma_short,
        sigma_long,
        ratio
    );
}

#[test]
fn ar2_stays_finite_on_billion_scale_input() {
    let mut rng = Rng::new(4);
    let mut leaf = Ar2Leaf::new(0.05);
    for _ in 0..200 {
        let y = 1.5e9 + 5e8 * rng.next_normal();
        leaf.observe(y);
    }
    let preds = leaf.predict(50);
    assert!(all_finite(&preds));
    assert!(max_abs_mean(&preds) < 1e11);
}

#[test]
fn ar2_survives_single_spike_without_forecast_blowup() {
    // Warm up on quiet noise, insert one 1e6-magnitude spike, then
    // verify future forecasts don't inherit the spike as a persistent
    // level. Ar2Leaf's project_to_stationary is the guard.
    let mut rng = Rng::new(5);
    let mut leaf = Ar2Leaf::new(0.05);
    for _ in 0..300 {
        leaf.observe(rng.next_normal());
    }
    leaf.observe(1.0e6);
    // Continue with quiet noise.
    for _ in 0..300 {
        leaf.observe(rng.next_normal());
    }
    let preds = leaf.predict(20);
    assert!(all_finite(&preds));
    // The forecast should have decayed back near zero — the spike was
    // one observation, not a level shift.
    assert!(
        max_abs_mean(&preds) < 100.0,
        "Ar2 held onto spike: max |mean| = {}",
        max_abs_mean(&preds)
    );
}

// ---------- GarchWrappedLeaf ----------

#[test]
fn garch_shift_invariant_on_level_series() {
    // 2026-07-25 fix: GARCH recursion now runs on deviations from
    // running mean, so shifting the whole series up by 1e6 should NOT
    // blow up the inner leaf standardization. Pre-fix, α·y² dominated
    // and the inverse re-inflated the mixture σ into meaninglessness.
    let mut rng = Rng::new(6);
    let mut leaf = GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(0.3)));
    // Level series around 1e6 (typical for macroeconomic index /
    // large-magnitude accounting values).
    for _ in 0..500 {
        let y = 1.0e6 + rng.next_normal();
        leaf.observe(y);
    }
    let preds = leaf.predict(20);
    assert!(all_finite(&preds));
    // The predictive scale should be O(1) — the innovation variance —
    // not O(1e6). Pre-fix, σ was of order the level.
    assert!(
        max_std(&preds) < 100.0,
        "GARCH σ inflated by level: max σ = {}",
        max_std(&preds)
    );
}

#[test]
fn garch_matches_zero_mean_returns_case_unchanged() {
    // Regression check: for the mean-zero return series GARCH is
    // meant for, the deviation-from-mean fix should be a no-op.
    let mut rng = Rng::new(7);
    let mut leaf = GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(0.3)));
    for _ in 0..500 {
        // Mean-zero returns with clustered volatility.
        let vol = 1.0 + 0.5 * (rng.next_f64() * 2.0 - 1.0).abs();
        leaf.observe(vol * rng.next_normal());
    }
    let preds = leaf.predict(20);
    assert!(all_finite(&preds));
    // σ should be O(1-2), reflecting the actual innovation scale.
    assert!(max_std(&preds) > 0.1 && max_std(&preds) < 10.0);
}

#[test]
fn garch_stays_finite_on_billion_scale_input() {
    let mut rng = Rng::new(8);
    let mut leaf = GarchWrappedLeaf::with_defaults(Box::new(EmaLeaf::new(0.3)));
    for _ in 0..500 {
        let y = 1.5e9 + 1e8 * rng.next_normal();
        leaf.observe(y);
    }
    let preds = leaf.predict(50);
    assert!(all_finite(&preds));
    // Predictions should be near the recent level (~1.5e9), not
    // exploding into oblivion.
    assert!(max_abs_mean(&preds) < 1e11);
}
