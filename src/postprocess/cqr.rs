//! Conformalized Quantile Regression (CQR).
//!
//! Romano, Y., Patterson, E., & Candès, E. J. (2019).
//! *Conformalized Quantile Regression.*
//! Advances in Neural Information Processing Systems, 32.
//!
//! CQR wraps a quantile-regression base learner: rather than computing
//! conformity scores from absolute residuals (as in standard split
//! conformal), it scores against the gap between predicted quantile bounds
//! and the actual value:
//!
//! ```text
//! E_i = max(q_lo_i - y_i, y_i - q_hi_i)
//! ```
//!
//! The (1-α)(n+1)/n quantile of these scores adjusts both bounds:
//!
//! ```text
//! C(x) = [q_lo(x) - Q, q_hi(x) + Q]
//! ```
//!
//! When the base quantile regressor is well-calibrated, `Q ≈ 0` and the
//! intervals stay tight. When it under- or over-covers, `Q` corrects the
//! gap. Tighter than absolute-residual conformal whenever heteroscedasticity
//! exists in the residuals.

use crate::error::{ForecastError, Result};
use crate::postprocess::PredictionIntervals;

/// Result of fitting a CQR predictor.
#[derive(Debug, Clone)]
pub struct CqrResult {
    /// Sorted conformity scores (one per calibration point).
    scores: Vec<f64>,
    /// The (1-α)(n+1)/n quantile — used to widen / tighten interval bounds.
    quantile_value: f64,
    /// Target coverage level (1-α).
    coverage: f64,
    /// Lower quantile τ used by the base learner (typically α/2).
    tau_lo: f64,
    /// Upper quantile τ used by the base learner (typically 1 − α/2).
    tau_hi: f64,
}

impl CqrResult {
    /// Sorted conformity scores from the calibration set.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Quantile adjustment Q applied symmetrically to both bounds.
    pub fn quantile_value(&self) -> f64 {
        self.quantile_value
    }

    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    pub fn tau_lo(&self) -> f64 {
        self.tau_lo
    }

    pub fn tau_hi(&self) -> f64 {
        self.tau_hi
    }

    /// Apply the calibration to new lower/upper quantile predictions.
    ///
    /// Returns conformalized intervals: `[q_lo - Q, q_hi + Q]`.
    pub fn predict_intervals(&self, q_lo: &[f64], q_hi: &[f64]) -> Result<PredictionIntervals> {
        if q_lo.len() != q_hi.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: q_lo.len(),
                got: q_hi.len(),
            });
        }
        let lower: Vec<f64> = q_lo.iter().map(|&q| q - self.quantile_value).collect();
        let upper: Vec<f64> = q_hi.iter().map(|&q| q + self.quantile_value).collect();
        PredictionIntervals::from_bounds(lower, upper, self.coverage)
    }
}

/// Conformalized Quantile Regression predictor.
///
/// Wraps any base learner that produces lower / upper quantile forecasts and
/// recalibrates them to achieve nominal coverage.
#[derive(Debug, Clone)]
pub struct CqrPredictor {
    coverage: f64,
}

impl CqrPredictor {
    /// Create a CQR predictor targeting the given coverage level.
    ///
    /// # Panics
    /// Panics if coverage is not in (0, 1).
    pub fn new(coverage: f64) -> Self {
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "coverage must be in (0, 1)"
        );
        Self { coverage }
    }

    /// Target coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Lower base-learner quantile τ used in fitting (= α/2 by default).
    pub fn tau_lo(&self) -> f64 {
        (1.0 - self.coverage) / 2.0
    }

    /// Upper base-learner quantile τ used in fitting (= 1 - α/2 by default).
    pub fn tau_hi(&self) -> f64 {
        1.0 - (1.0 - self.coverage) / 2.0
    }

    /// Fit the CQR predictor on calibration quantile predictions and actuals.
    ///
    /// # Arguments
    /// * `q_lo_calib` — lower quantile predictions on calibration set
    /// * `q_hi_calib` — upper quantile predictions on calibration set
    /// * `actuals` — actual values for the calibration set
    pub fn fit(
        &self,
        q_lo_calib: &[f64],
        q_hi_calib: &[f64],
        actuals: &[f64],
    ) -> Result<CqrResult> {
        let n = actuals.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }
        if q_lo_calib.len() != n || q_hi_calib.len() != n {
            return Err(ForecastError::DimensionMismatch {
                expected: n,
                got: q_lo_calib.len().min(q_hi_calib.len()),
            });
        }

        // Conformity score: E_i = max(q_lo - y, y - q_hi).
        // Positive E_i means the actual fell outside the [q_lo, q_hi] band.
        let mut scores: Vec<f64> = (0..n)
            .map(|i| (q_lo_calib[i] - actuals[i]).max(actuals[i] - q_hi_calib[i]))
            .collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Take the (1-α)(n+1)/n empirical quantile.
        // Equivalently: the ⌈(1-α)(n+1)⌉-th smallest score.
        let level = self.coverage * (n as f64 + 1.0) / n as f64;
        let level = level.min(1.0);
        let idx = ((n as f64) * level).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);
        let quantile_value = scores[idx];

        Ok(CqrResult {
            scores,
            quantile_value,
            coverage: self.coverage,
            tau_lo: self.tau_lo(),
            tau_hi: self.tau_hi(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calibrated quantile forecasts: q_lo and q_hi already cover roughly
    /// `coverage` of the actuals → CQR should report a near-zero adjustment.
    #[test]
    fn calibrated_base_gives_small_adjustment() {
        let n = 200;
        let actuals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        // Symmetric well-calibrated bounds: ±0.5 around actuals.
        let q_lo: Vec<f64> = actuals.iter().map(|&y| y - 0.5).collect();
        let q_hi: Vec<f64> = actuals.iter().map(|&y| y + 0.5).collect();

        let cqr = CqrPredictor::new(0.90);
        let result = cqr.fit(&q_lo, &q_hi, &actuals).unwrap();

        // Every actual is inside, so all conformity scores are ≤ 0
        // → quantile_value ≤ 0 (bands can be tightened).
        assert!(
            result.quantile_value() <= 0.0,
            "well-calibrated CQR should not need to widen, got Q={}",
            result.quantile_value()
        );
    }

    #[test]
    fn under_covering_base_widens_intervals() {
        // q_lo and q_hi are too tight: actuals fall outside the band.
        let n = 200;
        let actuals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let q_lo: Vec<f64> = actuals.iter().map(|&y| y - 0.05).collect();
        let q_hi: Vec<f64> = actuals.iter().map(|&y| y + 0.05).collect();

        // Inflate every other actual so the band misses it by 0.5
        let mut shifted_actuals = actuals.clone();
        for (i, a) in shifted_actuals.iter_mut().enumerate() {
            if i % 2 == 0 {
                *a += 0.6;
            }
        }

        let cqr = CqrPredictor::new(0.90);
        let result = cqr.fit(&q_lo, &q_hi, &shifted_actuals).unwrap();

        // Adjustment must widen (positive) to cover the under-covered points.
        assert!(
            result.quantile_value() > 0.3,
            "under-covering CQR should widen significantly, got Q={}",
            result.quantile_value()
        );
    }

    #[test]
    fn predict_intervals_applies_symmetric_adjustment() {
        let actuals = vec![0.0; 100];
        let q_lo = vec![-1.0; 100];
        let q_hi = vec![1.0; 100];

        let cqr = CqrPredictor::new(0.80);
        let result = cqr.fit(&q_lo, &q_hi, &actuals).unwrap();

        // Apply to new test forecasts.
        let test_lo = vec![5.0, 10.0];
        let test_hi = vec![7.0, 12.0];
        let intervals = result.predict_intervals(&test_lo, &test_hi).unwrap();

        // The adjustment should be applied symmetrically.
        let q = result.quantile_value();
        assert!((intervals.lower()[0] - (5.0 - q)).abs() < 1e-12);
        assert!((intervals.upper()[0] - (7.0 + q)).abs() < 1e-12);
        assert_eq!(intervals.coverage(), 0.80);
    }

    #[test]
    fn empty_calibration_errors() {
        let cqr = CqrPredictor::new(0.90);
        let err = cqr.fit(&[], &[], &[]).unwrap_err();
        assert!(matches!(err, ForecastError::EmptyData));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let cqr = CqrPredictor::new(0.90);
        let err = cqr.fit(&[1.0, 2.0], &[3.0], &[4.0, 5.0]).unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }

    #[test]
    fn tau_lo_tau_hi_match_coverage() {
        let cqr = CqrPredictor::new(0.90);
        assert!((cqr.tau_lo() - 0.05).abs() < 1e-12);
        assert!((cqr.tau_hi() - 0.95).abs() < 1e-12);
    }

    #[test]
    fn coverage_actually_achieved_on_holdout() {
        // Stationary synthetic experiment: same noise distribution on
        // calibration and test — CQR's exchangeability assumption holds.
        // The base learner under-covers (band too tight); CQR's
        // adjustment widens it to nominal.
        let n_per_set = 500;

        // Same noise distribution everywhere — uniform U(-1, 1).
        let synth = |n: usize, salt: u64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let h = (i as u64).wrapping_mul(2654435761).wrapping_add(salt);
                    ((h % 1000) as f64 / 1000.0 - 0.5) * 2.0
                })
                .collect()
        };
        let eps_calib = synth(n_per_set, 1);
        let eps_test = synth(n_per_set, 2);

        // Forecasts: zeros (so the residual = the noise).
        // Base "quantile" bounds: deliberately too tight at ±0.3, but the
        // true 90% quantile of U(-1, 1) is ±0.9. CQR should expand to
        // approximately ±0.9.
        let q_lo_calib = vec![-0.3; n_per_set];
        let q_hi_calib = vec![0.3; n_per_set];
        let q_lo_test = vec![-0.3; n_per_set];
        let q_hi_test = vec![0.3; n_per_set];

        let cqr = CqrPredictor::new(0.90);
        let result = cqr.fit(&q_lo_calib, &q_hi_calib, &eps_calib).unwrap();
        let intervals = result.predict_intervals(&q_lo_test, &q_hi_test).unwrap();

        let cov = intervals.empirical_coverage(&eps_test).unwrap();
        // Stationary case: CQR coverage should land near nominal 0.90.
        assert!(
            (0.85..0.95).contains(&cov),
            "CQR empirical coverage {:.3} should be near 0.90 (stationary case)",
            cov
        );
    }
}
