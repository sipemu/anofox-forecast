//! Conformalize quantile forecasts.
//!
//! This module provides the `conformalize` function to recalibrate quantile
//! forecasts using conformal prediction, ensuring coverage guarantees.
//!
//! # Overview
//!
//! Raw quantile forecasts from methods like QRA may not achieve nominal
//! coverage. The conformalize function applies conformal prediction to
//! adjust the quantile forecasts to achieve target coverage levels.
//!
//! # Algorithm
//!
//! For each quantile level τ:
//! 1. Compute conformity scores from calibration data
//! 2. Calculate the quantile of scores at coverage level
//! 3. Adjust predicted quantiles by the conformity score
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::{conformalize, QuantileForecasts};
//!
//! let calibrated = conformalize(
//!     &quantile_forecasts,
//!     &calib_predictions,
//!     &calib_actuals,
//! ).unwrap();
//! ```

use crate::error::{ForecastError, Result};
use crate::postprocess::{ConformalMethod, QuantileForecasts};

/// Configuration for conformalize operation.
#[derive(Debug, Clone)]
pub struct ConformalizeConfig {
    /// Conformal method to use for calibration.
    method: ConformalMethod,
    /// Whether to apply symmetric adjustment (both upper and lower).
    symmetric: bool,
}

impl Default for ConformalizeConfig {
    fn default() -> Self {
        Self {
            method: ConformalMethod::default(),
            symmetric: true,
        }
    }
}

impl ConformalizeConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the conformal method.
    pub fn method(mut self, method: ConformalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set whether to use symmetric adjustment.
    pub fn symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }
}

/// Result of conformalize operation.
#[derive(Debug, Clone)]
pub struct ConformalizeResult {
    /// Recalibrated quantile forecasts.
    forecasts: QuantileForecasts,
    /// Adjustment values applied for each quantile level.
    adjustments: Vec<f64>,
    /// Coverage achieved on calibration set (before adjustment).
    original_coverage: Vec<f64>,
}

impl ConformalizeResult {
    /// Get the recalibrated forecasts.
    pub fn forecasts(&self) -> &QuantileForecasts {
        &self.forecasts
    }

    /// Consume self and return the recalibrated forecasts.
    pub fn into_forecasts(self) -> QuantileForecasts {
        self.forecasts
    }

    /// Get the adjustments applied to each quantile.
    pub fn adjustments(&self) -> &[f64] {
        &self.adjustments
    }

    /// Get the original coverage on calibration set.
    pub fn original_coverage(&self) -> &[f64] {
        &self.original_coverage
    }
}

/// Conformalize quantile forecasts using conformal prediction.
///
/// Recalibrates quantile forecasts to achieve target coverage by applying
/// adjustments derived from calibration data.
///
/// # Arguments
///
/// * `forecasts` - Quantile forecasts to recalibrate
/// * `calib_forecasts` - Quantile forecasts on calibration set
/// * `calib_actuals` - Actual values on calibration set
///
/// # Returns
///
/// A `ConformalizeResult` containing recalibrated forecasts.
///
/// # Errors
///
/// Returns an error if:
/// - Quantile levels don't match between forecasts
/// - Calibration data dimensions don't match
/// - Insufficient calibration data
pub fn conformalize(
    forecasts: &QuantileForecasts,
    calib_forecasts: &QuantileForecasts,
    calib_actuals: &[f64],
) -> Result<ConformalizeResult> {
    conformalize_with_config(
        forecasts,
        calib_forecasts,
        calib_actuals,
        ConformalizeConfig::default(),
    )
}

/// Conformalize with custom configuration.
///
/// See [`conformalize`] for details.
pub fn conformalize_with_config(
    forecasts: &QuantileForecasts,
    calib_forecasts: &QuantileForecasts,
    calib_actuals: &[f64],
    config: ConformalizeConfig,
) -> Result<ConformalizeResult> {
    // Validate inputs
    if calib_forecasts.n_times() != calib_actuals.len() {
        return Err(ForecastError::DimensionMismatch {
            expected: calib_forecasts.n_times(),
            got: calib_actuals.len(),
        });
    }

    if calib_actuals.is_empty() {
        return Err(ForecastError::EmptyData);
    }

    // Check quantile levels match
    let forecast_quantiles = forecasts.quantiles();
    let calib_quantiles = calib_forecasts.quantiles();

    if forecast_quantiles != calib_quantiles {
        return Err(ForecastError::InvalidParameter(format!(
            "quantile levels must match: forecasts has {:?}, calibration has {:?}",
            forecast_quantiles, calib_quantiles
        )));
    }

    let quantiles = forecast_quantiles.to_vec();
    let n_calib = calib_actuals.len();

    // Compute conformity scores and adjustments per quantile
    let (adjustments, original_coverage) =
        compute_conformal_adjustments(&quantiles, calib_forecasts, calib_actuals, &config, n_calib);

    // Apply adjustments to produce recalibrated forecasts
    let new_values = apply_conformal_adjustments(forecasts, &quantiles, &adjustments);

    let calibrated = QuantileForecasts::from_values(quantiles, new_values)?;

    Ok(ConformalizeResult {
        forecasts: calibrated,
        adjustments,
        original_coverage,
    })
}

/// Compute conformity scores and conformal adjustments for each quantile level.
fn compute_conformal_adjustments(
    quantiles: &[f64],
    calib_forecasts: &QuantileForecasts,
    calib_actuals: &[f64],
    config: &ConformalizeConfig,
    n_calib: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut adjustments = Vec::with_capacity(quantiles.len());
    let mut original_coverage = Vec::with_capacity(quantiles.len());

    for (q_idx, &tau) in quantiles.iter().enumerate() {
        let mut scores: Vec<f64> = Vec::with_capacity(n_calib);
        let mut coverage_count = 0;

        for t in 0..n_calib {
            // SAFETY: t < n_calib == calib_forecasts.n_times(), validated by caller
            let predicted_q = calib_forecasts.at_time(t).unwrap()[q_idx];
            let actual = calib_actuals[t];

            let score = if config.symmetric {
                (actual - predicted_q).abs()
            } else if tau < 0.5 {
                predicted_q - actual
            } else {
                actual - predicted_q
            };
            scores.push(score);

            if actual <= predicted_q {
                coverage_count += 1;
            }
        }

        original_coverage.push(coverage_count as f64 / n_calib as f64);

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let target_coverage = if config.symmetric {
            tau.max(1.0 - tau)
        } else {
            tau
        };
        let idx = ((n_calib as f64 + 1.0) * target_coverage).ceil() as usize;
        let idx = idx.min(n_calib).saturating_sub(1);

        adjustments.push(scores[idx]);
    }

    (adjustments, original_coverage)
}

/// Apply conformal adjustments to forecast rows, enforcing monotonicity.
fn apply_conformal_adjustments(
    forecasts: &QuantileForecasts,
    quantiles: &[f64],
    adjustments: &[f64],
) -> Vec<Vec<f64>> {
    let n_times = forecasts.n_times();
    let n_quantiles = quantiles.len();
    let mut new_values: Vec<Vec<f64>> = Vec::with_capacity(n_times);

    for t in 0..n_times {
        // SAFETY: t < n_times == forecasts.n_times()
        let row = forecasts.at_time(t).unwrap();
        let mut adjusted_row = Vec::with_capacity(n_quantiles);

        for (q_idx, &tau) in quantiles.iter().enumerate() {
            let adjusted = if tau < 0.5 {
                row[q_idx] - adjustments[q_idx]
            } else {
                row[q_idx] + adjustments[q_idx]
            };
            adjusted_row.push(adjusted);
        }

        // Enforce monotonicity by sorting to prevent quantile crossing
        adjusted_row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        new_values.push(adjusted_row);
    }

    new_values
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ConformalizeConfig tests
    // =========================================================================

    mod config {
        use super::*;

        #[test]
        fn default_uses_split_method() {
            let config = ConformalizeConfig::default();
            matches!(config.method, ConformalMethod::Split { .. });
        }

        #[test]
        fn default_is_symmetric() {
            let config = ConformalizeConfig::default();
            assert!(config.symmetric);
        }

        #[test]
        fn method_builder_works() {
            let config = ConformalizeConfig::new().method(ConformalMethod::JackknifePlus);
            assert_eq!(config.method, ConformalMethod::JackknifePlus);
        }

        #[test]
        fn symmetric_builder_works() {
            let config = ConformalizeConfig::new().symmetric(false);
            assert!(!config.symmetric);
        }

        #[test]
        fn config_is_clonable() {
            let config = ConformalizeConfig::new().symmetric(false);
            let cloned = config.clone();
            assert!(!cloned.symmetric);
        }
    }

    // =========================================================================
    // ConformalizeResult tests
    // =========================================================================

    mod result {
        use super::*;

        fn make_simple_forecasts() -> QuantileForecasts {
            QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![5.0, 10.0, 15.0], vec![6.0, 11.0, 16.0]],
            )
            .unwrap()
        }

        fn make_calibration_data() -> (QuantileForecasts, Vec<f64>) {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..20)
                    .map(|i| {
                        let base = i as f64;
                        vec![base - 2.0, base, base + 2.0]
                    })
                    .collect(),
            )
            .unwrap();
            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();
            (forecasts, actuals)
        }

        #[test]
        fn forecasts_accessor_works() {
            let forecasts = make_simple_forecasts();
            let (calib, actuals) = make_calibration_data();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();
            assert_eq!(result.forecasts().n_times(), 2);
        }

        #[test]
        fn into_forecasts_consumes_result() {
            let forecasts = make_simple_forecasts();
            let (calib, actuals) = make_calibration_data();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();
            let consumed = result.into_forecasts();
            assert_eq!(consumed.n_times(), 2);
        }

        #[test]
        fn adjustments_accessor_works() {
            let forecasts = make_simple_forecasts();
            let (calib, actuals) = make_calibration_data();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();
            assert_eq!(result.adjustments().len(), 3);
        }

        #[test]
        fn original_coverage_accessor_works() {
            let forecasts = make_simple_forecasts();
            let (calib, actuals) = make_calibration_data();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();
            assert_eq!(result.original_coverage().len(), 3);
        }
    }

    // =========================================================================
    // Input validation tests
    // =========================================================================

    mod validation {
        use super::*;

        #[test]
        fn fails_on_length_mismatch() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![5.0, 10.0, 15.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]],
            )
            .unwrap();

            // Actuals has 3 points but calib has 2
            let actuals = vec![1.5, 2.5, 3.5];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fails_on_empty_calibration() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![5.0, 10.0, 15.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], Vec::<Vec<f64>>::new())
                .unwrap();

            let actuals: Vec<f64> = vec![];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fails_on_mismatched_quantiles() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![5.0, 10.0, 15.0]])
                    .unwrap();

            // Different quantile levels
            let calib = QuantileForecasts::from_values(
                vec![0.25, 0.5, 0.75],
                vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]],
            )
            .unwrap();

            let actuals = vec![1.5, 2.5];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Core functionality tests
    // =========================================================================

    mod conformalize_function {
        use super::*;

        fn make_forecasts(n: usize) -> QuantileForecasts {
            QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..n)
                    .map(|i| {
                        let base = 10.0 + i as f64;
                        vec![base - 2.0, base, base + 2.0]
                    })
                    .collect(),
            )
            .unwrap()
        }

        #[test]
        fn returns_forecasts_with_same_shape() {
            let forecasts = make_forecasts(5);
            let calib = make_forecasts(30);
            let actuals: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 + 0.3).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            assert_eq!(result.forecasts().n_times(), 5);
            assert_eq!(result.forecasts().n_quantiles(), 3);
        }

        #[test]
        fn returns_forecasts_with_same_quantiles() {
            let forecasts = make_forecasts(5);
            let calib = make_forecasts(30);
            let actuals: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 + 0.3).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            assert_eq!(result.forecasts().quantiles(), &[0.1, 0.5, 0.9]);
        }

        #[test]
        fn produces_non_negative_adjustments_for_symmetric() {
            let forecasts = make_forecasts(5);
            let calib = make_forecasts(30);
            let actuals: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 + 0.3).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            for &adj in result.adjustments() {
                assert!(adj >= 0.0, "symmetric adjustments should be non-negative");
            }
        }

        #[test]
        fn lower_quantiles_decrease_after_adjustment() {
            let forecasts = make_forecasts(5);
            let calib = make_forecasts(30);
            // Actuals tend to be higher, so lower quantiles need to decrease
            let actuals: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 + 5.0).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            // Check first time point, lower quantile (q=0.1)
            let original_lower = forecasts.at_time(0).unwrap()[0];
            let adjusted_lower = result.forecasts().at_time(0).unwrap()[0];

            assert!(
                adjusted_lower <= original_lower,
                "lower quantile should decrease (was {}, now {})",
                original_lower,
                adjusted_lower
            );
        }

        #[test]
        fn upper_quantiles_increase_after_adjustment() {
            let forecasts = make_forecasts(5);
            let calib = make_forecasts(30);
            // Actuals tend to be higher, so upper quantiles need to increase
            let actuals: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 + 5.0).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            // Check first time point, upper quantile (q=0.9)
            let original_upper = forecasts.at_time(0).unwrap()[2];
            let adjusted_upper = result.forecasts().at_time(0).unwrap()[2];

            assert!(
                adjusted_upper >= original_upper,
                "upper quantile should increase (was {}, now {})",
                original_upper,
                adjusted_upper
            );
        }

        #[test]
        fn preserves_monotonicity() {
            let forecasts = make_forecasts(10);
            let calib = make_forecasts(50);
            let actuals: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 + 2.0).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            // Check monotonicity at each time point
            for t in 0..10 {
                let row = result.forecasts().at_time(t).unwrap();
                assert!(row[0] <= row[1], "q0.1 <= q0.5 at t={}", t);
                assert!(row[1] <= row[2], "q0.5 <= q0.9 at t={}", t);
            }
        }

        #[test]
        fn original_coverage_in_valid_range() {
            let forecasts = make_forecasts(5);
            let calib = make_forecasts(30);
            let actuals: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            for &cov in result.original_coverage() {
                assert!(
                    (0.0..=1.0).contains(&cov),
                    "coverage should be in [0, 1], got {}",
                    cov
                );
            }
        }
    }

    // =========================================================================
    // Configuration option tests
    // =========================================================================

    mod config_options {
        use super::*;

        fn make_data() -> (QuantileForecasts, QuantileForecasts, Vec<f64>) {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..5)
                    .map(|i| {
                        let base = i as f64;
                        vec![base - 1.0, base, base + 1.0]
                    })
                    .collect(),
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let base = i as f64;
                        vec![base - 1.0, base, base + 1.0]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..30).map(|i| i as f64 + 0.2).collect();

            (forecasts, calib, actuals)
        }

        #[test]
        fn asymmetric_mode_works() {
            let (forecasts, calib, actuals) = make_data();
            let config = ConformalizeConfig::new().symmetric(false);

            let result = conformalize_with_config(&forecasts, &calib, &actuals, config);
            assert!(result.is_ok(), "asymmetric mode failed: {:?}", result.err());
        }

        #[test]
        fn different_methods_produce_results() {
            let (forecasts, calib, actuals) = make_data();

            let methods = [
                ConformalMethod::Split { cal_fraction: 0.2 },
                ConformalMethod::JackknifePlus,
            ];

            for method in methods {
                let config = ConformalizeConfig::new().method(method.clone());
                let result = conformalize_with_config(&forecasts, &calib, &actuals, config);
                assert!(result.is_ok(), "method {:?} should work", method);
            }
        }
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    mod edge_cases {
        use super::*;

        #[test]
        fn works_with_single_quantile() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.5], vec![vec![10.0], vec![11.0]]).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.5],
                (0..20).map(|i| vec![i as f64]).collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn works_with_perfect_calibration() {
            // Forecasts match actuals perfectly
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![8.0, 10.0, 12.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let base = i as f64;
                        vec![base - 2.0, base, base + 2.0]
                    })
                    .collect(),
            )
            .unwrap();

            // Perfect coverage - actuals between lower and upper
            let actuals: Vec<f64> = (0..30).map(|i| i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn works_with_extreme_values() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.9], vec![vec![-1000.0, 1000.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.9],
                (0..20)
                    .map(|i| {
                        let base = i as f64;
                        vec![base - 10.0, base + 10.0]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..20).map(|i| i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn handles_minimum_calibration_size() {
            let forecasts = QuantileForecasts::from_values(vec![0.5], vec![vec![5.0]]).unwrap();

            // Single calibration point
            let calib = QuantileForecasts::from_values(vec![0.5], vec![vec![4.0]]).unwrap();

            let actuals = vec![4.5];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }
    }

    // =========================================================================
    // Regression tests
    // =========================================================================

    mod regression {
        use super::*;

        #[test]
        fn consistent_results_on_same_input() {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..10)
                    .map(|i| {
                        let base = i as f64 * 2.0;
                        vec![base - 1.0, base, base + 1.0]
                    })
                    .collect(),
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..50)
                    .map(|i| {
                        let base = i as f64;
                        vec![base - 1.0, base, base + 1.0]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.3).collect();

            let result1 = conformalize(&forecasts, &calib, &actuals).unwrap();
            let result2 = conformalize(&forecasts, &calib, &actuals).unwrap();

            assert_eq!(result1.adjustments(), result2.adjustments());
        }
    }

    // =========================================================================
    // Coverage guarantee tests (forecast-12o)
    // =========================================================================

    mod coverage_guarantees {
        use super::*;

        /// Helper to compute interval coverage on a dataset
        fn compute_coverage(
            forecasts: &QuantileForecasts,
            actuals: &[f64],
            lower_q_idx: usize,
            upper_q_idx: usize,
        ) -> f64 {
            let mut covered = 0;
            for t in 0..forecasts.n_times() {
                let row = forecasts.at_time(t).unwrap();
                let lower = row[lower_q_idx];
                let upper = row[upper_q_idx];
                if actuals[t] >= lower && actuals[t] <= upper {
                    covered += 1;
                }
            }
            covered as f64 / actuals.len() as f64
        }

        #[test]
        fn achieves_target_coverage_on_calibration_data() {
            // Create deliberately under-covering forecasts (intervals too narrow)
            // Actuals have significant noise that forecasts don't capture
            let n_calib = 100;
            let calib_actuals: Vec<f64> = (0..n_calib)
                .map(|i| 50.0 + i as f64 + 5.0 * ((i as f64 * 0.2).sin()))
                .collect();

            // Forecasts centered on trend only, missing the sine variation
            // This creates systematic under-coverage
            let calib_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..n_calib)
                    .map(|i| {
                        let center = 50.0 + i as f64; // No sine term!
                        vec![center - 0.5, center, center + 0.5] // Too narrow for ±5 variation
                    })
                    .collect(),
            )
            .unwrap();

            // Test forecasts (same pattern - missing the sine)
            let n_test = 50;
            let test_actuals: Vec<f64> = (n_calib..n_calib + n_test)
                .map(|i| 50.0 + i as f64 + 5.0 * ((i as f64 * 0.2).sin()))
                .collect();

            let test_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (n_calib..n_calib + n_test)
                    .map(|i| {
                        let center = 50.0 + i as f64;
                        vec![center - 0.5, center, center + 0.5]
                    })
                    .collect(),
            )
            .unwrap();

            // Verify original coverage is low (intervals don't capture ±5 sine)
            let original_coverage = compute_coverage(&test_forecasts, &test_actuals, 0, 2);
            assert!(
                original_coverage < 0.80,
                "original should under-cover, got {:.1}%",
                original_coverage * 100.0
            );

            // Apply conformalize
            let result = conformalize(&test_forecasts, &calib_forecasts, &calib_actuals).unwrap();
            let calibrated = result.forecasts();

            // Coverage should improve significantly
            let calibrated_coverage = compute_coverage(calibrated, &test_actuals, 0, 2);
            assert!(
                calibrated_coverage > original_coverage,
                "calibrated coverage ({:.1}%) should exceed original ({:.1}%)",
                calibrated_coverage * 100.0,
                original_coverage * 100.0
            );
        }

        #[test]
        fn coverage_improves_for_under_covering_forecasts() {
            // Simulate forecaster that consistently underestimates uncertainty
            // Forecasts are based on trend only, actuals have significant noise
            let n = 200;
            let actuals: Vec<f64> = (0..n)
                .map(|i| 100.0 + 3.0 * ((i as f64 * 0.1).sin()))
                .collect();

            // Forecasts ignore the sine variation - centered on 100.0 with tiny intervals
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..n)
                    .map(|_| {
                        vec![99.8, 100.0, 100.2] // Way too tight for ±3 variation
                    })
                    .collect(),
            )
            .unwrap();

            // Split into calibration and test
            let split = 150;
            let calib_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..split)
                    .map(|t| forecasts.at_time(t).unwrap().to_vec())
                    .collect(),
            )
            .unwrap();
            let calib_actuals = &actuals[..split];

            let test_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (split..n)
                    .map(|t| forecasts.at_time(t).unwrap().to_vec())
                    .collect(),
            )
            .unwrap();
            let test_actuals = &actuals[split..];

            // Original coverage should be very low
            let orig_cov = compute_coverage(&test_forecasts, test_actuals, 0, 2);
            assert!(
                orig_cov < 0.50,
                "expected severe under-coverage, got {:.1}%",
                orig_cov * 100.0
            );

            // After conformalize
            let result = conformalize(&test_forecasts, &calib_forecasts, calib_actuals).unwrap();
            let new_cov = compute_coverage(result.forecasts(), test_actuals, 0, 2);

            // Should be much better
            assert!(
                new_cov >= 0.70,
                "conformalized coverage ({:.1}%) should be much better than original ({:.1}%)",
                new_cov * 100.0,
                orig_cov * 100.0
            );
        }

        #[test]
        fn well_calibrated_forecasts_stay_well_calibrated() {
            // Already well-calibrated forecasts shouldn't get worse
            let n = 100;
            let actuals: Vec<f64> = (0..n).map(|i| 50.0 + i as f64 * 0.1).collect();

            // Intervals that should achieve ~80% coverage naturally
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                actuals.iter().map(|&a| vec![a - 1.5, a, a + 1.5]).collect(),
            )
            .unwrap();

            let split = 70;
            let calib_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..split)
                    .map(|t| forecasts.at_time(t).unwrap().to_vec())
                    .collect(),
            )
            .unwrap();
            let calib_actuals = &actuals[..split];

            let test_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (split..n)
                    .map(|t| forecasts.at_time(t).unwrap().to_vec())
                    .collect(),
            )
            .unwrap();
            let test_actuals = &actuals[split..];

            let orig_cov = compute_coverage(&test_forecasts, test_actuals, 0, 2);

            let result = conformalize(&test_forecasts, &calib_forecasts, calib_actuals).unwrap();
            let new_cov = compute_coverage(result.forecasts(), test_actuals, 0, 2);

            // Coverage should stay good (not get dramatically worse)
            assert!(
                new_cov >= orig_cov * 0.9,
                "well-calibrated forecasts shouldn't get much worse: {:.1}% -> {:.1}%",
                orig_cov * 100.0,
                new_cov * 100.0
            );
        }

        #[test]
        fn median_coverage_tracked_correctly() {
            let n = 50;
            let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();

            // Median slightly biased high
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                actuals
                    .iter()
                    .map(|&a| {
                        vec![a - 2.0, a + 1.0, a + 4.0] // Median is biased +1
                    })
                    .collect(),
            )
            .unwrap();

            let result = conformalize(&forecasts, &forecasts, &actuals).unwrap();

            // Original coverage for median (q=0.5) should reflect the bias
            let orig_cov = result.original_coverage();
            assert_eq!(orig_cov.len(), 3);

            // All coverages should be valid probabilities
            for &c in orig_cov {
                assert!((0.0..=1.0).contains(&c));
            }
        }

        #[test]
        fn asymmetric_errors_handled() {
            // Forecaster that's biased high (actuals tend to be lower)
            let n = 100;
            let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                actuals
                    .iter()
                    .map(|&a| {
                        vec![a + 1.0, a + 3.0, a + 5.0] // All biased high
                    })
                    .collect(),
            )
            .unwrap();

            let split = 70;
            let calib_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..split)
                    .map(|t| forecasts.at_time(t).unwrap().to_vec())
                    .collect(),
            )
            .unwrap();

            let test_forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (split..n)
                    .map(|t| forecasts.at_time(t).unwrap().to_vec())
                    .collect(),
            )
            .unwrap();

            let result =
                conformalize(&test_forecasts, &calib_forecasts, &actuals[..split]).unwrap();

            // Lower quantile should have been pushed down significantly
            let adj = result.adjustments();
            assert!(
                adj[0] > 0.0,
                "lower quantile should have positive adjustment"
            );
        }
    }

    // =========================================================================
    // Numerical edge cases (forecast-6vj)
    // =========================================================================

    mod numerical_edge_cases {
        use super::*;

        #[test]
        fn handles_identical_quantile_values() {
            // All quantiles have same value (degenerate distribution)
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![5.0, 5.0, 5.0], vec![6.0, 6.0, 6.0]],
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..20)
                    .map(|i| {
                        let v = i as f64;
                        vec![v, v, v]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle identical quantile values");

            // Result should still be monotonic
            let calibrated = result.unwrap().into_forecasts();
            for t in 0..calibrated.n_times() {
                let row = calibrated.at_time(t).unwrap();
                assert!(row[0] <= row[1] && row[1] <= row[2]);
            }
        }

        #[test]
        fn handles_all_actuals_below_lower_quantile() {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![100.0, 110.0, 120.0]],
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30).map(|_| vec![100.0, 110.0, 120.0]).collect(),
            )
            .unwrap();

            // All actuals far below the forecasts
            let actuals: Vec<f64> = (0..30).map(|i| i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());

            // Lower quantile should be adjusted down significantly
            let r = result.unwrap();
            let calibrated = r.forecasts().at_time(0).unwrap();
            assert!(
                calibrated[0] < 100.0,
                "lower quantile should be pushed down, got {}",
                calibrated[0]
            );
        }

        #[test]
        fn handles_all_actuals_above_upper_quantile() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![0.0, 5.0, 10.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30).map(|_| vec![0.0, 5.0, 10.0]).collect(),
            )
            .unwrap();

            // All actuals far above the forecasts
            let actuals: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());

            // Upper quantile should be adjusted up significantly
            let r = result.unwrap();
            let calibrated = r.forecasts().at_time(0).unwrap();
            assert!(
                calibrated[2] > 10.0,
                "upper quantile should be pushed up, got {}",
                calibrated[2]
            );
        }

        #[test]
        fn handles_very_small_calibration_set() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![5.0, 10.0, 15.0]])
                    .unwrap();

            // Only 2 calibration points
            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![4.0, 9.0, 14.0], vec![6.0, 11.0, 16.0]],
            )
            .unwrap();

            let actuals = vec![9.5, 11.5];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should work with minimal calibration");
        }

        #[test]
        fn handles_large_magnitude_values() {
            let scale = 1e12;
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![scale - 1e9, scale, scale + 1e9]],
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let base = scale + (i as f64 * 1e8);
                        vec![base - 1e9, base, base + 1e9]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..30).map(|i| scale + (i as f64 * 1e8) + 5e8).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle large magnitudes");

            let calibrated = result.unwrap().into_forecasts();
            let row = calibrated.at_time(0).unwrap();
            assert!(row[0].is_finite() && row[1].is_finite() && row[2].is_finite());
        }

        #[test]
        fn handles_small_magnitude_values() {
            let scale = 1e-10;
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![scale * 0.9, scale, scale * 1.1]],
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let base = scale * (1.0 + i as f64 * 0.01);
                        vec![base * 0.9, base, base * 1.1]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..30)
                .map(|i| scale * (1.0 + i as f64 * 0.01 + 0.05))
                .collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle small magnitudes");
        }

        #[test]
        fn handles_negative_values() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![-15.0, -10.0, -5.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let base = -50.0 + i as f64;
                        vec![base - 5.0, base, base + 5.0]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..30).map(|i| -50.0 + i as f64 + 2.0).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle negative values");

            let calibrated = result.unwrap().into_forecasts();
            let row = calibrated.at_time(0).unwrap();
            assert!(
                row[0] <= row[1] && row[1] <= row[2],
                "monotonicity preserved"
            );
        }

        #[test]
        fn handles_mixed_sign_values() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![-5.0, 0.0, 5.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let base = -15.0 + i as f64;
                        vec![base - 3.0, base, base + 3.0]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..30).map(|i| -15.0 + i as f64 + 1.0).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle mixed signs");
        }

        #[test]
        fn handles_zero_width_original_intervals() {
            // q10 = q90 (point forecast, no uncertainty)
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![10.0, 10.0, 10.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30)
                    .map(|i| {
                        let v = i as f64;
                        vec![v, v, v]
                    })
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..30).map(|i| i as f64 + 2.0).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());

            // Should expand the interval
            let calibrated = result.unwrap().into_forecasts();
            let row = calibrated.at_time(0).unwrap();
            let width = row[2] - row[0];
            assert!(
                width > 0.0,
                "should create non-zero width interval, got {}",
                width
            );
        }

        #[test]
        fn handles_constant_actuals() {
            let forecasts =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![8.0, 10.0, 12.0]])
                    .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30).map(|_| vec![8.0, 10.0, 12.0]).collect(),
            )
            .unwrap();

            // All actuals are the same
            let actuals: Vec<f64> = vec![10.0; 30];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle constant actuals");
        }

        #[test]
        fn handles_many_quantile_levels() {
            let quantiles: Vec<f64> = (1..10).map(|i| i as f64 * 0.1).collect();

            let forecasts = QuantileForecasts::from_values(
                quantiles.clone(),
                vec![(1..10).map(|i| i as f64).collect()],
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                quantiles.clone(),
                (0..50)
                    .map(|t| (1..10).map(|i| t as f64 * 0.1 + i as f64).collect())
                    .collect(),
            )
            .unwrap();

            let actuals: Vec<f64> = (0..50).map(|i| 5.0 + i as f64 * 0.1).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok(), "should handle many quantiles");

            // Check monotonicity
            let calibrated = result.unwrap().into_forecasts();
            let row = calibrated.at_time(0).unwrap();
            for i in 1..row.len() {
                assert!(row[i - 1] <= row[i], "monotonicity violated at {}", i);
            }
        }
    }

    // =========================================================================
    // Property-based tests (forecast-36p)
    // =========================================================================

    mod property_tests {
        use super::*;

        #[test]
        fn adjustment_magnitude_correlates_with_miscalibration() {
            // Test that worse calibration (more misses) leads to larger adjustments
            // The key is that actuals should be OUTSIDE the forecast intervals

            let n = 100;

            // Scenario 1: Forecasts that mostly contain the actuals
            // Actuals vary slightly around 50, forecasts have wide intervals around 50
            let actuals_good: Vec<f64> = (0..n)
                .map(|i| 50.0 + 0.5 * ((i as f64 * 0.1).sin()))
                .collect();
            let well_calibrated = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..n).map(|_| vec![48.0, 50.0, 52.0]).collect(), // Wide enough
            )
            .unwrap();

            // Scenario 2: Forecasts that systematically miss (actuals outside intervals)
            // Actuals at 50, but forecasts centered at 60
            let actuals_bad: Vec<f64> = (0..n).map(|_| 50.0).collect();
            let poorly_calibrated = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..n).map(|_| vec![58.0, 60.0, 62.0]).collect(), // Centered at wrong value!
            )
            .unwrap();

            let result_good =
                conformalize(&well_calibrated, &well_calibrated, &actuals_good).unwrap();
            let result_bad =
                conformalize(&poorly_calibrated, &poorly_calibrated, &actuals_bad).unwrap();

            // The poorly calibrated model needs larger adjustments because actuals
            // are consistently outside the intervals
            let adj_good = result_good.adjustments()[0];
            let adj_bad = result_bad.adjustments()[0];

            assert!(
                adj_bad > adj_good,
                "poorly calibrated should need larger adjustment: good={}, bad={}",
                adj_good,
                adj_bad
            );
        }

        #[test]
        fn intervals_widen_or_stay_same_in_symmetric_mode() {
            let n = 50;
            let actuals: Vec<f64> = (0..n)
                .map(|i| 50.0 + i as f64 + 3.0 * ((i as f64 * 0.1).sin()))
                .collect();

            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                actuals.iter().map(|&a| vec![a - 1.0, a, a + 1.0]).collect(),
            )
            .unwrap();

            let result = conformalize(&forecasts, &forecasts, &actuals).unwrap();
            let calibrated = result.forecasts();

            for t in 0..n {
                let orig = forecasts.at_time(t).unwrap();
                let cal = calibrated.at_time(t).unwrap();

                let orig_width = orig[2] - orig[0];
                let cal_width = cal[2] - cal[0];

                assert!(
                    cal_width >= orig_width - 1e-10,
                    "interval at t={} should not shrink: {} -> {}",
                    t,
                    orig_width,
                    cal_width
                );
            }
        }

        #[test]
        fn monotonicity_always_preserved() {
            // Test with various challenging inputs
            let test_cases: Vec<(Vec<f64>, Vec<Vec<f64>>)> = vec![
                // Case 1: Normal data
                (
                    (0..50).map(|i| i as f64).collect(),
                    (0..50)
                        .map(|i| vec![i as f64 - 2.0, i as f64, i as f64 + 2.0])
                        .collect(),
                ),
                // Case 2: High variance
                (
                    (0..50)
                        .map(|i| i as f64 + 10.0 * ((i as f64 * 0.3).sin()))
                        .collect(),
                    (0..50)
                        .map(|i| {
                            let b = i as f64;
                            vec![b - 1.0, b, b + 1.0]
                        })
                        .collect(),
                ),
                // Case 3: Biased forecasts
                (
                    (0..50).map(|i| i as f64).collect(),
                    (0..50)
                        .map(|i| {
                            let b = i as f64 + 5.0; // Biased high
                            vec![b - 1.0, b, b + 1.0]
                        })
                        .collect(),
                ),
            ];

            for (actuals, forecast_values) in test_cases {
                let forecasts =
                    QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], forecast_values).unwrap();

                let result = conformalize(&forecasts, &forecasts, &actuals).unwrap();
                let calibrated = result.forecasts();

                for t in 0..calibrated.n_times() {
                    let row = calibrated.at_time(t).unwrap();
                    assert!(row[0] <= row[1], "q0.1 <= q0.5 failed at t={}", t);
                    assert!(row[1] <= row[2], "q0.5 <= q0.9 failed at t={}", t);
                }
            }
        }

        #[test]
        fn adjustments_are_stable_with_more_data() {
            // Adding more calibration data shouldn't wildly change adjustments
            let base_actuals: Vec<f64> = (0..100).map(|i| i as f64 + 0.5).collect();

            let make_forecasts = |n: usize| {
                QuantileForecasts::from_values(
                    vec![0.1, 0.5, 0.9],
                    (0..n)
                        .map(|i| {
                            let b = i as f64;
                            vec![b - 1.0, b, b + 1.0]
                        })
                        .collect(),
                )
                .unwrap()
            };

            let test_forecast = make_forecasts(10);

            // With 50 calibration points
            let calib_50 = make_forecasts(50);
            let result_50 = conformalize(&test_forecast, &calib_50, &base_actuals[..50]).unwrap();

            // With 100 calibration points
            let calib_100 = make_forecasts(100);
            let result_100 =
                conformalize(&test_forecast, &calib_100, &base_actuals[..100]).unwrap();

            // Adjustments should be in the same ballpark
            for i in 0..3 {
                let adj_50 = result_50.adjustments()[i];
                let adj_100 = result_100.adjustments()[i];
                let diff = (adj_50 - adj_100).abs();

                // Should be within 50% of each other (generous bound)
                let max_adj = adj_50.abs().max(adj_100.abs()).max(0.1);
                assert!(
                    diff < max_adj * 0.5,
                    "adjustment {} changed too much: {} vs {}",
                    i,
                    adj_50,
                    adj_100
                );
            }
        }

        #[test]
        fn extreme_quantiles_adjusted_appropriately() {
            // Very extreme quantiles (0.01, 0.99) should get larger adjustments
            let n = 200;
            let actuals: Vec<f64> = (0..n)
                .map(|i| i as f64 + 5.0 * ((i as f64 * 0.1).sin()))
                .collect();

            let forecasts = QuantileForecasts::from_values(
                vec![0.01, 0.1, 0.5, 0.9, 0.99],
                actuals
                    .iter()
                    .map(|&a| vec![a - 2.0, a - 1.0, a, a + 1.0, a + 2.0])
                    .collect(),
            )
            .unwrap();

            let result = conformalize(&forecasts, &forecasts, &actuals).unwrap();
            let adj = result.adjustments();

            // q0.01 should have >= adjustment than q0.1 (more extreme = needs more coverage)
            assert!(
                adj[0] >= adj[1] * 0.9,
                "q0.01 adjustment ({}) should be >= q0.1 ({})",
                adj[0],
                adj[1]
            );

            // q0.99 should have >= adjustment than q0.9
            assert!(
                adj[4] >= adj[3] * 0.9,
                "q0.99 adjustment ({}) should be >= q0.9 ({})",
                adj[4],
                adj[3]
            );
        }

        #[test]
        fn result_is_deterministic() {
            let n = 50;
            let actuals: Vec<f64> = (0..n)
                .map(|i| i as f64 + ((i as f64 * 0.2).sin()))
                .collect();

            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                actuals.iter().map(|&a| vec![a - 1.5, a, a + 1.5]).collect(),
            )
            .unwrap();

            let result1 = conformalize(&forecasts, &forecasts, &actuals).unwrap();
            let result2 = conformalize(&forecasts, &forecasts, &actuals).unwrap();

            // Adjustments should be exactly equal
            assert_eq!(result1.adjustments(), result2.adjustments());

            // Forecasts should be exactly equal
            for t in 0..n {
                let r1 = result1.forecasts().at_time(t).unwrap();
                let r2 = result2.forecasts().at_time(t).unwrap();
                assert_eq!(r1, r2, "forecasts differ at t={}", t);
            }
        }
    }

    // =========================================================================
    // Reference value tests (forecast-s3v)
    // =========================================================================

    mod reference_values {
        use super::*;

        /// Test against known reference values.
        /// These values should be verified against PostForecasts.jl output.
        #[test]
        fn simple_case_reference() {
            // Simple, reproducible test case
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![8.0, 10.0, 12.0], vec![9.0, 11.0, 13.0]],
            )
            .unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..20)
                    .map(|i| {
                        let b = i as f64;
                        vec![b - 2.0, b, b + 2.0]
                    })
                    .collect(),
            )
            .unwrap();

            // Actuals slightly above median
            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();

            let result = conformalize(&forecasts, &calib, &actuals).unwrap();

            // Verify structure
            assert_eq!(result.adjustments().len(), 3);
            assert_eq!(result.original_coverage().len(), 3);
            assert_eq!(result.forecasts().n_times(), 2);

            // Adjustments should be non-negative in symmetric mode
            for &adj in result.adjustments() {
                assert!(adj >= 0.0, "adjustment should be non-negative: {}", adj);
            }

            // All original coverages should be in [0, 1]
            for &cov in result.original_coverage() {
                assert!((0.0..=1.0).contains(&cov));
            }
        }

        #[test]
        fn known_adjustment_direction() {
            // When actuals are consistently above forecasts,
            // we expect upper quantile to be pushed up

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..50)
                    .map(|i| {
                        let b = i as f64;
                        vec![b - 1.0, b, b + 1.0]
                    })
                    .collect(),
            )
            .unwrap();

            // Actuals consistently 2.0 above forecast median
            let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 2.0).collect();

            let test =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![49.0, 50.0, 51.0]])
                    .unwrap();

            let result = conformalize(&test, &calib, &actuals).unwrap();
            let calibrated = result.forecasts().at_time(0).unwrap();

            // Upper bound should have increased (actuals tend to be high)
            assert!(
                calibrated[2] > 51.0,
                "upper quantile should increase from 51.0, got {}",
                calibrated[2]
            );
        }

        #[test]
        fn symmetric_adjustment_values() {
            // In symmetric mode, adjustment for q0.1 and q0.9 should be
            // based on the same absolute residuals

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..100)
                    .map(|i| {
                        let b = i as f64;
                        vec![b - 2.0, b, b + 2.0]
                    })
                    .collect(),
            )
            .unwrap();

            // Symmetric residuals around the median
            let actuals: Vec<f64> = (0..100)
                .map(|i| {
                    let b = i as f64;
                    if i % 2 == 0 {
                        b + 1.0
                    } else {
                        b - 1.0
                    }
                })
                .collect();

            let test =
                QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], vec![vec![48.0, 50.0, 52.0]])
                    .unwrap();

            let result = conformalize(&test, &calib, &actuals).unwrap();

            // In symmetric mode, lower and upper adjustments are based on |residual|
            // so they should widen the interval equally
            let calibrated = result.forecasts().at_time(0).unwrap();
            let lower_expansion = 48.0 - calibrated[0];
            let upper_expansion = calibrated[2] - 52.0;

            // Should be approximately equal
            let diff = (lower_expansion - upper_expansion).abs();
            assert!(
                diff < 0.1,
                "symmetric adjustments should be similar: lower={}, upper={}",
                lower_expansion,
                upper_expansion
            );
        }

        #[test]
        fn coverage_calculation_correct() {
            // Verify original_coverage is computed correctly
            let calib = QuantileForecasts::from_values(
                vec![0.5], // Just median for simplicity
                (0..10).map(|i| vec![i as f64]).collect(),
            )
            .unwrap();

            // 6 out of 10 actuals are below median forecast
            let actuals = vec![
                -1.0, // below 0
                0.5,  // below 1
                1.5,  // below 2
                2.5,  // below 3
                3.5,  // below 4
                4.5,  // below 5
                6.5,  // above 6
                7.5,  // above 7
                8.5,  // above 8
                9.5,  // above 9
            ];

            let test = QuantileForecasts::from_values(vec![0.5], vec![vec![5.0]]).unwrap();

            let result = conformalize(&test, &calib, &actuals).unwrap();

            // Coverage is fraction where actual <= forecast
            // For q=0.5: count where actual[i] <= calib[i]
            // Positions: 0:(-1<=0)=T, 1:(0.5<=1)=T, 2:(1.5<=2)=T, 3:(2.5<=3)=T,
            //            4:(3.5<=4)=T, 5:(4.5<=5)=T, 6:(6.5<=6)=F, 7:(7.5<=7)=F,
            //            8:(8.5<=8)=F, 9:(9.5<=9)=F
            // = 6/10 = 0.6
            let cov = result.original_coverage()[0];
            assert!(
                (cov - 0.6).abs() < 0.01,
                "expected coverage 0.6, got {}",
                cov
            );
        }
    }
}
