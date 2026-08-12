//! Adaptive Conformal Inference (ACI).
//!
//! Gibbs, I., & Candès, E. J. (2021).
//! *Adaptive Conformal Inference under Distribution Shift.*
//! NeurIPS 2021.
//!
//! ACI maintains an online estimate of the working coverage level α_t and
//! adjusts it each step based on whether the most recent observation fell
//! inside the predicted interval:
//!
//! ```text
//! err_t = 1 if y_t ∉ C_t(x_t),  0 otherwise
//! α_{t+1} = α_t + γ * (α_target - err_t)
//! ```
//!
//! where γ > 0 is the learning rate. The interval radius is recomputed
//! each step from the (1 - α_t) quantile of recent absolute residuals.
//!
//! Under arbitrary distribution drift, ACI guarantees that the long-run
//! empirical miscoverage rate converges to the target α — without any
//! distributional assumptions.
//!
//! ## Usage
//!
//! ```ignore
//! let mut aci = AciPredictor::new(0.90, 0.01);   // target coverage 90%, γ=0.01
//! aci.fit(&initial_residuals)?;                  // seed the residual buffer
//! for (forecast, actual) in stream {
//!     let (lo, hi) = aci.predict_interval(forecast);
//!     aci.observe(forecast, actual);             // updates α_t and residuals
//! }
//! ```

use crate::error::{ForecastError, Result};
use crate::postprocess::PredictionIntervals;

/// Adaptive Conformal Inference predictor.
///
/// Stateful, online: the predictor tracks a sliding window of recent
/// absolute residuals and an online α_t that is updated after each
/// `observe()` call.
#[derive(Debug, Clone)]
pub struct AciPredictor {
    /// Target miscoverage rate α_target = 1 - target_coverage.
    alpha_target: f64,
    /// Learning rate for the α update.
    gamma: f64,
    /// Current online miscoverage rate α_t.
    alpha_t: f64,
    /// Maximum number of residuals to retain in the sliding window.
    max_window: usize,
    /// Buffer of recent absolute residuals (unsorted; sorted on demand).
    residuals: Vec<f64>,
    /// Number of `observe()` calls processed.
    n_observed: usize,
    /// Number of times the actual fell outside the interval.
    n_misses: usize,
}

impl AciPredictor {
    /// Create a new ACI predictor.
    ///
    /// # Arguments
    /// * `target_coverage` — desired long-run coverage (e.g. 0.90)
    /// * `gamma` — learning rate; typical values 0.005–0.05. Larger γ reacts
    ///   faster to drift but causes wider intervals.
    ///
    /// # Panics
    /// Panics if `target_coverage ∉ (0, 1)` or `gamma ≤ 0`.
    pub fn new(target_coverage: f64, gamma: f64) -> Self {
        assert!(
            target_coverage > 0.0 && target_coverage < 1.0,
            "target_coverage must be in (0, 1)"
        );
        assert!(gamma > 0.0, "gamma must be > 0");
        let alpha = 1.0 - target_coverage;
        Self {
            alpha_target: alpha,
            gamma,
            alpha_t: alpha,
            max_window: 500,
            residuals: Vec::new(),
            n_observed: 0,
            n_misses: 0,
        }
    }

    /// Set the maximum residual window size (default 500). Smaller windows
    /// adapt faster to drift; larger windows produce smoother intervals.
    pub fn with_max_window(mut self, max_window: usize) -> Self {
        assert!(max_window > 0, "max_window must be > 0");
        self.max_window = max_window;
        self
    }

    /// Target coverage level.
    pub fn target_coverage(&self) -> f64 {
        1.0 - self.alpha_target
    }

    /// Current online α_t (the working miscoverage level).
    pub fn alpha_t(&self) -> f64 {
        self.alpha_t
    }

    /// Learning rate γ.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Number of observations processed via `observe()`.
    pub fn n_observed(&self) -> usize {
        self.n_observed
    }

    /// Empirical miscoverage rate over all `observe()` calls so far.
    pub fn empirical_miscoverage(&self) -> f64 {
        if self.n_observed == 0 {
            0.0
        } else {
            self.n_misses as f64 / self.n_observed as f64
        }
    }

    /// Seed the residual buffer with prior absolute residuals.
    ///
    /// Required at least once before `predict_interval()` will return
    /// non-trivial bounds. The most recent `max_window` residuals are kept.
    pub fn fit(&mut self, abs_residuals: &[f64]) -> Result<()> {
        if abs_residuals.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        for &r in abs_residuals {
            if !r.is_finite() {
                return Err(ForecastError::InvalidParameter(
                    "residuals must be finite".to_string(),
                ));
            }
        }
        let start = abs_residuals.len().saturating_sub(self.max_window);
        self.residuals = abs_residuals[start..].to_vec();
        Ok(())
    }

    /// Compute the current interval radius from the residual buffer at level
    /// (1 - α_t). Returns 0.0 if the buffer is empty.
    pub fn current_radius(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        let mut sorted = self.residuals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        // α_t is clamped to (0, 1) in observe(), so coverage is in (0, 1).
        let coverage = 1.0 - self.alpha_t;
        let level = (coverage * (n as f64 + 1.0) / n as f64).min(1.0);
        let idx = ((n as f64) * level).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);
        sorted[idx]
    }

    /// Produce an interval `[forecast - r, forecast + r]` using the current
    /// radius `r` derived from α_t.
    pub fn predict_interval(&self, forecast: f64) -> (f64, f64) {
        let r = self.current_radius();
        (forecast - r, forecast + r)
    }

    /// Produce intervals for a slice of forecasts.
    pub fn predict_intervals(&self, forecasts: &[f64]) -> Result<PredictionIntervals> {
        let r = self.current_radius();
        let lower: Vec<f64> = forecasts.iter().map(|&f| f - r).collect();
        let upper: Vec<f64> = forecasts.iter().map(|&f| f + r).collect();
        PredictionIntervals::from_bounds(lower, upper, 1.0 - self.alpha_target)
    }

    /// Observe a (forecast, actual) pair and update α_t + the residual
    /// buffer.
    pub fn observe(&mut self, forecast: f64, actual: f64) {
        let radius = self.current_radius();
        let lower = forecast - radius;
        let upper = forecast + radius;
        let inside = actual >= lower && actual <= upper;
        let err: f64 = if inside { 0.0 } else { 1.0 };

        // ACI update: α_{t+1} = α_t + γ (α_target - err_t).
        // Note: when the actual is INSIDE (err=0), this pushes α_t up
        // (target=0.10 - 0 = +0.10), widening intervals to be more
        // conservative? No — wait: alpha is miscoverage. err=0 means
        // covered. We want α_t to *decrease* (less miscoverage budget) when
        // covered. The standard ACI formulation is:
        //   α_{t+1} = α_t + γ (α_target - err_t)
        // err=0 (covered): α_t increases by γ * α_target (e.g. +0.001 with
        //   α_target=0.10, γ=0.01). This gives BACK miscoverage budget,
        //   tightening intervals slightly.
        // err=1 (missed): α_t increases by γ * (α_target - 1) (e.g. -0.009),
        //   pulling α_t DOWN, widening intervals.
        // This matches Gibbs & Candès Algorithm 1.
        self.alpha_t += self.gamma * (self.alpha_target - err);
        // Clamp α_t to (0, 1) — the working coverage 1-α_t must be a valid
        // probability. Recommended in the original Gibbs & Candès paper to
        // avoid the radius oscillating between 0 (α_t ≥ 1) and the max
        // residual (α_t ≤ 0).
        self.alpha_t = self.alpha_t.clamp(0.001, 0.999);

        // Append the new absolute residual and slide the window.
        let abs_err = (forecast - actual).abs();
        self.residuals.push(abs_err);
        if self.residuals.len() > self.max_window {
            let drop = self.residuals.len() - self.max_window;
            self.residuals.drain(..drop);
        }

        self.n_observed += 1;
        if !inside {
            self.n_misses += 1;
        }
    }

    /// Reset the online state (α_t back to α_target, miss counter to 0)
    /// without clearing the residual buffer.
    pub fn reset_alpha(&mut self) {
        self.alpha_t = self.alpha_target;
        self.n_observed = 0;
        self.n_misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn fit_seeds_residual_buffer() {
        let mut aci = AciPredictor::new(0.90, 0.01);
        aci.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Radius is the (1-α_t) quantile of {1,2,3,4,5} ≈ 5
        let r = aci.current_radius();
        assert!((4.0..=5.0).contains(&r));
    }

    #[test]
    fn fit_empty_errors() {
        let mut aci = AciPredictor::new(0.90, 0.01);
        let err = aci.fit(&[]).unwrap_err();
        assert!(matches!(err, ForecastError::EmptyData));
    }

    #[test]
    fn predict_interval_uses_current_radius() {
        let mut aci = AciPredictor::new(0.90, 0.01);
        aci.fit(&[0.5; 50]).unwrap();
        let (lo, hi) = aci.predict_interval(10.0);
        assert!((lo - 9.5).abs() < 1e-9);
        assert!((hi - 10.5).abs() < 1e-9);
    }

    #[test]
    fn observe_inside_increases_alpha_t() {
        let mut aci = AciPredictor::new(0.90, 0.05);
        aci.fit(&[1.0; 30]).unwrap();
        let alpha_before = aci.alpha_t();
        // Forecast 0, actual 0 → inside → err=0 → α_t += γ * α_target
        aci.observe(0.0, 0.0);
        assert!(
            aci.alpha_t() > alpha_before,
            "covered observation should push α_t UP, but {} <= {}",
            aci.alpha_t(),
            alpha_before
        );
    }

    #[test]
    fn observe_outside_decreases_alpha_t() {
        let mut aci = AciPredictor::new(0.90, 0.05);
        aci.fit(&[0.1; 30]).unwrap();
        let alpha_before = aci.alpha_t();
        // Forecast 0, actual 1000 (well outside any reasonable interval)
        aci.observe(0.0, 1000.0);
        assert!(
            aci.alpha_t() < alpha_before,
            "missed observation should push α_t DOWN, but {} >= {}",
            aci.alpha_t(),
            alpha_before
        );
    }

    #[test]
    fn long_run_coverage_approaches_target() {
        // Stationary stream: forecast = 0, actual ~ N(0, 1).
        let mut aci = AciPredictor::new(0.90, 0.02).with_max_window(200);
        let mut rng = StdRng::seed_from_u64(42);

        // Seed with some initial residuals
        let init: Vec<f64> = (0..50).map(|_| rng.gen::<f64>() * 2.0).collect();
        aci.fit(&init).unwrap();

        // Run 2000 observations
        for _ in 0..2000 {
            // Standard-normal-ish via Box-Muller approximation
            let u1: f64 = rng.gen::<f64>().max(1e-300);
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            aci.observe(0.0, z);
        }

        let cov = 1.0 - aci.empirical_miscoverage();
        // Should converge near the 0.90 target — wide tolerance for finite n.
        assert!(
            (0.83..0.97).contains(&cov),
            "long-run coverage should approach target 0.90, got {:.3}",
            cov
        );
    }

    #[test]
    fn responds_to_drift() {
        // Distribution shifts at t=500: residuals jump from ±0.5 to ±5.0.
        let mut aci = AciPredictor::new(0.90, 0.05).with_max_window(100);
        aci.fit(&[0.5; 50]).unwrap();

        let radius_before_drift = aci.current_radius();

        // First 500 obs: small residuals (model well-calibrated).
        for i in 0..500 {
            let actual = if i % 2 == 0 { 0.4 } else { -0.4 };
            aci.observe(0.0, actual);
        }

        // Drift: observations now far from forecast.
        for i in 0..500 {
            let actual = if i % 2 == 0 { 5.0 } else { -5.0 };
            aci.observe(0.0, actual);
        }

        let radius_after_drift = aci.current_radius();
        assert!(
            radius_after_drift > radius_before_drift,
            "ACI should widen intervals after drift: before={:.3}, after={:.3}",
            radius_before_drift,
            radius_after_drift
        );
        // The radius should have grown substantially.
        assert!(
            radius_after_drift > 1.5,
            "radius after drift should reflect new residual scale, got {:.3}",
            radius_after_drift
        );
    }

    #[test]
    fn predict_intervals_returns_correct_coverage() {
        let mut aci = AciPredictor::new(0.95, 0.01);
        aci.fit(&[1.0; 100]).unwrap();
        let intervals = aci.predict_intervals(&[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(intervals.coverage(), 0.95);
        assert_eq!(intervals.len(), 3);
    }

    #[test]
    fn reset_alpha_clears_counters_only() {
        let mut aci = AciPredictor::new(0.90, 0.05);
        aci.fit(&[1.0; 30]).unwrap();
        aci.observe(0.0, 5.0); // miss
        aci.observe(0.0, 0.5); // hit
        assert_eq!(aci.n_observed(), 2);
        aci.reset_alpha();
        assert_eq!(aci.n_observed(), 0);
        assert!((aci.alpha_t() - 0.10).abs() < 1e-12);
        // Residuals should still be there (3 observations + 30 seed = 32)
        assert!(!aci.residuals.is_empty());
    }
}
