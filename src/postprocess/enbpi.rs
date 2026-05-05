//! Ensemble Bootstrap Prediction Intervals (EnbPI).
//!
//! Xu, C., & Xie, Y. (2021).
//! *Conformal prediction interval for dynamic time-series.*
//! ICML 2021.
//!
//! EnbPI builds prediction intervals from K bootstrap-trained model
//! ensembles, computing leave-one-out (LOO) residuals from the ensemble
//! prediction that **excludes** the bootstrap samples containing the
//! target point. The (1-α) quantile of these residuals is the interval
//! radius.
//!
//! Unlike Jackknife+, EnbPI does **not** require K full retraining passes
//! — the K models are trained once on bootstrap resamples (the natural
//! ensemble structure of bagging). The LOO is computed at *prediction*
//! time by ensemble masking.
//!
//! The `EnbPiPredictor` is agnostic to how the K bootstrap models are
//! produced: callers supply
//!
//! - `bootstrap_predictions: [n_train × K]` — prediction of each model on
//!   each training point
//! - `inclusion_mask: [n_train × K]` — `true` if training point i was
//!   included in bootstrap sample k (used to construct LOO ensembles)
//! - `actuals: [n_train]` — true values
//!
//! At prediction time, the model k's predictions on a new test set are
//! also supplied (length `n_test × K`); EnbPI averages all K models to
//! produce the point forecast and uses the LOO residual quantile for the
//! interval radius.
//!
//! ## Online updates
//!
//! `update()` accepts new (forecast, actual) pairs and slides the residual
//! window forward — adapts to distribution drift without retraining the
//! ensemble.

use crate::error::{ForecastError, Result};
use crate::postprocess::PredictionIntervals;

/// Result of fitting an EnbPI predictor.
#[derive(Debug, Clone)]
pub struct EnbPiResult {
    /// Sorted leave-one-out absolute residuals on the training set.
    residuals: Vec<f64>,
    /// (1-α) quantile of the residuals — interval half-width.
    quantile_value: f64,
    /// Target coverage level (1-α).
    coverage: f64,
    /// Number of bootstrap models in the ensemble.
    n_models: usize,
}

impl EnbPiResult {
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    pub fn quantile_value(&self) -> f64 {
        self.quantile_value
    }

    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    pub fn n_models(&self) -> usize {
        self.n_models
    }

    /// Apply the calibrated radius to point forecasts.
    pub fn predict_intervals(&self, point_forecasts: &[f64]) -> Result<PredictionIntervals> {
        let lower: Vec<f64> = point_forecasts
            .iter()
            .map(|&p| p - self.quantile_value)
            .collect();
        let upper: Vec<f64> = point_forecasts
            .iter()
            .map(|&p| p + self.quantile_value)
            .collect();
        PredictionIntervals::from_bounds(lower, upper, self.coverage)
    }

    /// Slide the residual window forward with new (forecast, actual)
    /// observations — keeps the most recent `max_window` residuals and
    /// recomputes the quantile. Use this for online operation under drift.
    pub fn update(
        &mut self,
        new_forecasts: &[f64],
        new_actuals: &[f64],
        max_window: usize,
    ) -> Result<()> {
        if new_forecasts.len() != new_actuals.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: new_forecasts.len(),
                got: new_actuals.len(),
            });
        }
        // Append new residuals (kept unsorted for cheap append).
        let mut all: Vec<f64> = self.residuals.clone();
        for (f, a) in new_forecasts.iter().zip(new_actuals.iter()) {
            all.push((f - a).abs());
        }
        // Trim to most recent max_window observations.
        if all.len() > max_window {
            let drop = all.len() - max_window;
            all.drain(..drop);
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }
        let level = self.coverage * (n as f64 + 1.0) / n as f64;
        let level = level.min(1.0);
        let idx = ((n as f64) * level).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);
        self.quantile_value = all[idx];
        self.residuals = all;
        Ok(())
    }
}

/// EnbPI predictor.
///
/// Coverage is the only configuration parameter at construction time —
/// the ensemble structure is supplied at fit time.
#[derive(Debug, Clone)]
pub struct EnbPiPredictor {
    coverage: f64,
}

impl EnbPiPredictor {
    /// # Panics
    /// Panics if coverage is not in (0, 1).
    pub fn new(coverage: f64) -> Self {
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "coverage must be in (0, 1)"
        );
        Self { coverage }
    }

    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Fit EnbPI on a bootstrap ensemble.
    ///
    /// # Arguments
    /// * `bootstrap_predictions` — flat `[n_train * n_models]` row-major
    ///   matrix: `predictions[i * n_models + k]` = model k's prediction on
    ///   training point i
    /// * `inclusion_mask` — flat `[n_train * n_models]`: `mask[i * K + k]`
    ///   is `true` iff training point i was included in bootstrap sample
    ///   k. Used to compute the LOO ensemble for point i.
    /// * `actuals` — true values for each training point
    /// * `n_models` — K, the number of bootstrap models in the ensemble
    pub fn fit(
        &self,
        bootstrap_predictions: &[f64],
        inclusion_mask: &[bool],
        actuals: &[f64],
        n_models: usize,
    ) -> Result<EnbPiResult> {
        if n_models == 0 {
            return Err(ForecastError::InvalidParameter(
                "n_models must be ≥ 1".to_string(),
            ));
        }
        let n = actuals.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }
        let expected = n * n_models;
        if bootstrap_predictions.len() != expected {
            return Err(ForecastError::DimensionMismatch {
                expected,
                got: bootstrap_predictions.len(),
            });
        }
        if inclusion_mask.len() != expected {
            return Err(ForecastError::DimensionMismatch {
                expected,
                got: inclusion_mask.len(),
            });
        }

        // For each training point i, compute the LOO ensemble prediction:
        // average over models k where i was NOT in bootstrap sample k.
        // Fall back to full ensemble average when no model excludes i.
        let mut residuals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let mut sum = 0.0;
            let mut count = 0usize;
            let mut full_sum = 0.0;
            for k in 0..n_models {
                let pred = bootstrap_predictions[i * n_models + k];
                full_sum += pred;
                if !inclusion_mask[i * n_models + k] {
                    sum += pred;
                    count += 1;
                }
            }
            let loo_pred = if count > 0 {
                sum / count as f64
            } else {
                full_sum / n_models as f64
            };
            residuals.push((loo_pred - actuals[i]).abs());
        }

        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let level = self.coverage * (n as f64 + 1.0) / n as f64;
        let level = level.min(1.0);
        let idx = ((n as f64) * level).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);
        let quantile_value = residuals[idx];

        Ok(EnbPiResult {
            residuals,
            quantile_value,
            coverage: self.coverage,
            n_models,
        })
    }

    /// Compute the ensemble point forecast for new test data by averaging
    /// the K bootstrap models' predictions.
    ///
    /// # Arguments
    /// * `test_predictions` — flat `[n_test * n_models]` row-major matrix
    /// * `n_models` — K
    pub fn ensemble_point_forecast(
        &self,
        test_predictions: &[f64],
        n_models: usize,
    ) -> Result<Vec<f64>> {
        if n_models == 0 {
            return Err(ForecastError::InvalidParameter(
                "n_models must be ≥ 1".to_string(),
            ));
        }
        if test_predictions.len() % n_models != 0 {
            return Err(ForecastError::DimensionMismatch {
                expected: test_predictions.len() / n_models * n_models,
                got: test_predictions.len(),
            });
        }
        let n_test = test_predictions.len() / n_models;
        let mut result = Vec::with_capacity(n_test);
        for i in 0..n_test {
            let mut sum = 0.0;
            for k in 0..n_models {
                sum += test_predictions[i * n_models + k];
            }
            result.push(sum / n_models as f64);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic ensemble where every model gives the same
    /// prediction `actuals + noise[k]` and check that the residuals
    /// match the expected magnitude.
    #[test]
    fn fit_recovers_noise_magnitude() {
        let n = 100;
        let n_models = 5;
        let actuals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        // Each model adds a different constant offset (no per-row noise).
        // The LOO ensemble for any given i averages 4 of 5 offsets.
        let model_offsets = [-0.2, -0.1, 0.0, 0.1, 0.2];
        let mut predictions: Vec<f64> = Vec::with_capacity(n * n_models);
        for i in 0..n {
            for k in 0..n_models {
                predictions.push(actuals[i] + model_offsets[k]);
            }
        }
        // Inclusion mask: model k included point i iff (i + k) is even.
        let mut mask: Vec<bool> = Vec::with_capacity(n * n_models);
        for i in 0..n {
            for k in 0..n_models {
                mask.push((i + k) % 2 == 0);
            }
        }

        let enbpi = EnbPiPredictor::new(0.90);
        let result = enbpi.fit(&predictions, &mask, &actuals, n_models).unwrap();

        // Residuals should be small and positive.
        assert_eq!(result.residuals().len(), n);
        assert!(result.quantile_value() >= 0.0);
        // The maximum possible LOO offset is bounded by max(|model_offsets|) ≈ 0.2.
        assert!(
            result.quantile_value() < 0.4,
            "quantile value {} too large for offsets ≤ 0.2",
            result.quantile_value()
        );
    }

    #[test]
    fn predict_intervals_widens_around_point_forecast() {
        // Trivial setup: every model predicts the actual exactly.
        let n = 50;
        let n_models = 3;
        let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut predictions: Vec<f64> = Vec::with_capacity(n * n_models);
        for i in 0..n {
            for _k in 0..n_models {
                predictions.push(actuals[i]);
            }
        }
        let mask: Vec<bool> = vec![false; n * n_models]; // all LOO

        // Add a known offset to make residuals non-zero
        let mut perturbed = predictions.clone();
        for p in perturbed.iter_mut() {
            *p += 0.5;
        }

        let enbpi = EnbPiPredictor::new(0.90);
        let result = enbpi.fit(&perturbed, &mask, &actuals, n_models).unwrap();

        // All residuals are exactly 0.5 → quantile = 0.5
        assert!((result.quantile_value() - 0.5).abs() < 1e-12);

        let test_pf = vec![10.0, 20.0, 30.0];
        let intervals = result.predict_intervals(&test_pf).unwrap();
        assert!((intervals.lower()[0] - 9.5).abs() < 1e-12);
        assert!((intervals.upper()[0] - 10.5).abs() < 1e-12);
    }

    #[test]
    fn ensemble_point_forecast_averages_models() {
        let n_test = 3;
        let n_models = 4;
        let test = vec![
            1.0, 2.0, 3.0, 4.0, // i=0: mean = 2.5
            10.0, 20.0, 30.0, 40.0, // i=1: mean = 25.0
            5.0, 5.0, 5.0, 5.0, // i=2: mean = 5.0
        ];
        let enbpi = EnbPiPredictor::new(0.90);
        let pf = enbpi.ensemble_point_forecast(&test, n_models).unwrap();
        assert_eq!(pf, vec![2.5, 25.0, 5.0]);
        let _ = n_test;
    }

    #[test]
    fn online_update_slides_window() {
        let actuals: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let preds: Vec<f64> = actuals.iter().map(|&a| a + 0.5).collect();
        let mask = vec![false; 50 * 1];

        let enbpi = EnbPiPredictor::new(0.90);
        let mut result = enbpi.fit(&preds, &mask, &actuals, 1).unwrap();

        // Initial radius = 0.5
        assert!((result.quantile_value() - 0.5).abs() < 1e-12);

        // Update with larger errors: forecasts off by 1.5
        let new_actuals = vec![100.0, 101.0, 102.0];
        let new_preds = vec![101.5, 102.5, 103.5];
        result.update(&new_preds, &new_actuals, 30).unwrap();

        // After sliding to a 30-window dominated by 0.5 errors,
        // the new larger errors should pull the quantile up but stay close.
        assert!(result.quantile_value() >= 0.5);
        assert!(result.residuals().len() <= 30);
    }

    #[test]
    fn empty_data_errors() {
        let enbpi = EnbPiPredictor::new(0.90);
        let err = enbpi.fit(&[], &[], &[], 1).unwrap_err();
        assert!(matches!(err, ForecastError::EmptyData));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let enbpi = EnbPiPredictor::new(0.90);
        // 2 actuals × 3 models = 6 expected slots, but only 5 supplied
        let err = enbpi
            .fit(&[1.0, 2.0, 3.0, 4.0, 5.0], &[false; 6], &[1.0, 2.0], 3)
            .unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }
}
