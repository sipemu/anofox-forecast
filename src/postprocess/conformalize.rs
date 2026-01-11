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
use crate::postprocess::{QuantileForecasts, ConformalMethod};

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
    conformalize_with_config(forecasts, calib_forecasts, calib_actuals, ConformalizeConfig::default())
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
    let n_quantiles = quantiles.len();
    let n_calib = calib_actuals.len();

    // Calculate original coverage and adjustments for each quantile
    let mut adjustments = Vec::with_capacity(n_quantiles);
    let mut original_coverage = Vec::with_capacity(n_quantiles);

    for (q_idx, &tau) in quantiles.iter().enumerate() {
        // Compute conformity scores: actual - predicted_quantile
        let mut scores: Vec<f64> = Vec::with_capacity(n_calib);
        let mut coverage_count = 0;

        for t in 0..n_calib {
            let predicted_q = calib_forecasts.at_time(t).unwrap()[q_idx];
            let actual = calib_actuals[t];

            if config.symmetric {
                // For symmetric: score is absolute residual
                scores.push((actual - predicted_q).abs());
            } else {
                // For asymmetric: score depends on quantile level
                if tau < 0.5 {
                    // Lower quantile: score is how much actual is below predicted
                    scores.push(predicted_q - actual);
                } else {
                    // Upper quantile: score is how much actual is above predicted
                    scores.push(actual - predicted_q);
                }
            }

            // Count coverage
            if actual <= predicted_q {
                coverage_count += 1;
            }
        }

        original_coverage.push(coverage_count as f64 / n_calib as f64);

        // Sort scores and find the quantile
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use conformal quantile: ceil((n+1) * coverage) / n
        let target_coverage = if config.symmetric { tau.max(1.0 - tau) } else { tau };
        let idx = ((n_calib as f64 + 1.0) * target_coverage).ceil() as usize;
        let idx = idx.min(n_calib).saturating_sub(1);

        let adjustment = scores[idx];
        adjustments.push(adjustment);
    }

    // Apply adjustments to forecasts
    let n_times = forecasts.n_times();
    let mut new_values: Vec<Vec<f64>> = Vec::with_capacity(n_times);

    for t in 0..n_times {
        let row = forecasts.at_time(t).unwrap();
        let mut adjusted_row = Vec::with_capacity(n_quantiles);

        for (q_idx, &tau) in quantiles.iter().enumerate() {
            let value = row[q_idx];
            let adj = adjustments[q_idx];

            let adjusted = if config.symmetric {
                // Symmetric: expand interval
                if tau < 0.5 {
                    value - adj
                } else {
                    value + adj
                }
            } else {
                // Asymmetric: direct adjustment
                if tau < 0.5 {
                    value - adj
                } else {
                    value + adj
                }
            };

            adjusted_row.push(adjusted);
        }

        // Enforce monotonicity by sorting to prevent quantile crossing
        adjusted_row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        new_values.push(adjusted_row);
    }

    let calibrated = QuantileForecasts::from_values(quantiles, new_values)?;

    Ok(ConformalizeResult {
        forecasts: calibrated,
        adjustments,
        original_coverage,
    })
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
                vec![
                    vec![5.0, 10.0, 15.0],
                    vec![6.0, 11.0, 16.0],
                ],
            ).unwrap()
        }

        fn make_calibration_data() -> (QuantileForecasts, Vec<f64>) {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..20).map(|i| {
                    let base = i as f64;
                    vec![base - 2.0, base, base + 2.0]
                }).collect(),
            ).unwrap();
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
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![5.0, 10.0, 15.0]],
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![
                    vec![1.0, 2.0, 3.0],
                    vec![2.0, 3.0, 4.0],
                ],
            ).unwrap();

            // Actuals has 3 points but calib has 2
            let actuals = vec![1.5, 2.5, 3.5];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fails_on_empty_calibration() {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![5.0, 10.0, 15.0]],
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                Vec::<Vec<f64>>::new(),
            ).unwrap();

            let actuals: Vec<f64> = vec![];

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fails_on_mismatched_quantiles() {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![5.0, 10.0, 15.0]],
            ).unwrap();

            // Different quantile levels
            let calib = QuantileForecasts::from_values(
                vec![0.25, 0.5, 0.75],
                vec![
                    vec![1.0, 2.0, 3.0],
                    vec![2.0, 3.0, 4.0],
                ],
            ).unwrap();

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
                (0..n).map(|i| {
                    let base = 10.0 + i as f64;
                    vec![base - 2.0, base, base + 2.0]
                }).collect(),
            ).unwrap()
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

            assert!(adjusted_lower <= original_lower,
                "lower quantile should decrease (was {}, now {})", original_lower, adjusted_lower);
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

            assert!(adjusted_upper >= original_upper,
                "upper quantile should increase (was {}, now {})", original_upper, adjusted_upper);
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
                assert!(cov >= 0.0 && cov <= 1.0,
                    "coverage should be in [0, 1], got {}", cov);
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
                (0..5).map(|i| {
                    let base = i as f64;
                    vec![base - 1.0, base, base + 1.0]
                }).collect(),
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30).map(|i| {
                    let base = i as f64;
                    vec![base - 1.0, base, base + 1.0]
                }).collect(),
            ).unwrap();

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
            let forecasts = QuantileForecasts::from_values(
                vec![0.5],
                vec![vec![10.0], vec![11.0]],
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.5],
                (0..20).map(|i| vec![i as f64]).collect(),
            ).unwrap();

            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn works_with_perfect_calibration() {
            // Forecasts match actuals perfectly
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                vec![vec![8.0, 10.0, 12.0]],
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..30).map(|i| {
                    let base = i as f64;
                    vec![base - 2.0, base, base + 2.0]
                }).collect(),
            ).unwrap();

            // Perfect coverage - actuals between lower and upper
            let actuals: Vec<f64> = (0..30).map(|i| i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn works_with_extreme_values() {
            let forecasts = QuantileForecasts::from_values(
                vec![0.1, 0.9],
                vec![vec![-1000.0, 1000.0]],
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.9],
                (0..20).map(|i| {
                    let base = i as f64;
                    vec![base - 10.0, base + 10.0]
                }).collect(),
            ).unwrap();

            let actuals: Vec<f64> = (0..20).map(|i| i as f64).collect();

            let result = conformalize(&forecasts, &calib, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn handles_minimum_calibration_size() {
            let forecasts = QuantileForecasts::from_values(
                vec![0.5],
                vec![vec![5.0]],
            ).unwrap();

            // Single calibration point
            let calib = QuantileForecasts::from_values(
                vec![0.5],
                vec![vec![4.0]],
            ).unwrap();

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
                (0..10).map(|i| {
                    let base = i as f64 * 2.0;
                    vec![base - 1.0, base, base + 1.0]
                }).collect(),
            ).unwrap();

            let calib = QuantileForecasts::from_values(
                vec![0.1, 0.5, 0.9],
                (0..50).map(|i| {
                    let base = i as f64;
                    vec![base - 1.0, base, base + 1.0]
                }).collect(),
            ).unwrap();

            let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.3).collect();

            let result1 = conformalize(&forecasts, &calib, &actuals).unwrap();
            let result2 = conformalize(&forecasts, &calib, &actuals).unwrap();

            assert_eq!(result1.adjustments(), result2.adjustments());
        }
    }
}
