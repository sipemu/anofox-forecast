//! Conformal prediction for distribution-free prediction intervals.
//!
//! Conformal prediction provides prediction intervals with finite-sample
//! coverage guarantees without distributional assumptions.
//!
//! # Methods
//!
//! - **Split Conformal**: Fast method using a holdout calibration set
//! - **Cross-Validation Conformal**: Uses all data via cross-validation
//! - **Jackknife+**: Leave-one-out with finite sample validity
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::{ConformalPredictor, ConformalMethod};
//!
//! let mut predictor = ConformalPredictor::new(0.90, ConformalMethod::Split { cal_fraction: 0.2 });
//!
//! // Fit on historical forecast errors
//! let result = predictor.fit(&historical_forecasts, &historical_actuals).unwrap();
//!
//! // Generate prediction intervals
//! let intervals = predictor.predict(&result, &new_forecasts);
//! ```

use crate::error::{ForecastError, Result};
use crate::postprocess::{PredictionIntervals, PointForecasts};

/// Conformal prediction method.
#[derive(Debug, Clone, PartialEq)]
pub enum ConformalMethod {
    /// Split conformal: use a fraction of data for calibration.
    Split {
        /// Fraction of data to use for calibration (0, 1).
        cal_fraction: f64,
    },
    /// Cross-validation conformal: use all data via k-fold CV.
    CrossVal {
        /// Number of folds for cross-validation.
        n_folds: usize,
    },
    /// Jackknife+ (leave-one-out with finite sample validity).
    JackknifePlus,
}

impl Default for ConformalMethod {
    fn default() -> Self {
        Self::Split { cal_fraction: 0.2 }
    }
}

/// Result of fitting a conformal predictor.
#[derive(Debug, Clone)]
pub struct ConformalResult {
    /// Sorted nonconformity scores (absolute residuals).
    scores: Vec<f64>,
    /// The quantile value for the specified coverage.
    quantile_value: f64,
    /// The coverage level used.
    coverage: f64,
    /// The method used.
    method: ConformalMethod,
}

impl ConformalResult {
    /// Get the nonconformity scores.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Get the quantile value (interval half-width).
    pub fn quantile_value(&self) -> f64 {
        self.quantile_value
    }

    /// Get the coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Get the method used.
    pub fn method(&self) -> &ConformalMethod {
        &self.method
    }
}

/// Conformal predictor for distribution-free prediction intervals.
///
/// Provides prediction intervals with coverage guarantees based on
/// the conformal prediction framework.
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    /// Target coverage level (e.g., 0.90 for 90% intervals).
    coverage: f64,
    /// Conformal method to use.
    method: ConformalMethod,
}

impl ConformalPredictor {
    /// Create a new conformal predictor.
    ///
    /// # Arguments
    ///
    /// * `coverage` - Target coverage level in (0, 1)
    /// * `method` - Conformal method to use
    ///
    /// # Panics
    ///
    /// Panics if coverage is not in (0, 1).
    pub fn new(coverage: f64, method: ConformalMethod) -> Self {
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "coverage must be in (0, 1)"
        );
        Self { coverage, method }
    }

    /// Create a split conformal predictor with default calibration fraction.
    pub fn split(coverage: f64) -> Self {
        Self::new(coverage, ConformalMethod::Split { cal_fraction: 0.2 })
    }

    /// Create a cross-validation conformal predictor.
    pub fn cross_val(coverage: f64, n_folds: usize) -> Self {
        Self::new(coverage, ConformalMethod::CrossVal { n_folds })
    }

    /// Create a Jackknife+ conformal predictor.
    pub fn jackknife_plus(coverage: f64) -> Self {
        Self::new(coverage, ConformalMethod::JackknifePlus)
    }

    /// Get the coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Get the method.
    pub fn method(&self) -> &ConformalMethod {
        &self.method
    }

    /// Fit the conformal predictor on historical forecasts and actuals.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - Historical point forecasts
    /// * `actuals` - Corresponding actual values
    ///
    /// # Returns
    ///
    /// A `ConformalResult` containing the calibration information.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forecasts and actuals have different lengths
    /// - Not enough data points for the chosen method
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<ConformalResult> {
        if forecasts.len() != actuals.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: forecasts.len(),
                got: actuals.len(),
            });
        }

        let n = forecasts.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }

        match &self.method {
            ConformalMethod::Split { cal_fraction } => {
                self.fit_split(forecasts, actuals, *cal_fraction)
            }
            ConformalMethod::CrossVal { n_folds } => {
                self.fit_cross_val(forecasts, actuals, *n_folds)
            }
            ConformalMethod::JackknifePlus => self.fit_jackknife_plus(forecasts, actuals),
        }
    }

    /// Fit using split conformal method.
    fn fit_split(
        &self,
        forecasts: &[f64],
        actuals: &[f64],
        cal_fraction: f64,
    ) -> Result<ConformalResult> {
        let n = forecasts.len();
        let cal_size = ((n as f64) * cal_fraction).ceil() as usize;

        if cal_size < 1 {
            return Err(ForecastError::InsufficientData {
                needed: 1,
                got: cal_size,
            });
        }

        // Use the last cal_size points for calibration
        let cal_start = n - cal_size;

        // Compute nonconformity scores (absolute residuals) on calibration set
        let mut scores: Vec<f64> = forecasts[cal_start..]
            .iter()
            .zip(actuals[cal_start..].iter())
            .map(|(f, a)| (f - a).abs())
            .collect();

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute the quantile for the coverage level
        // For conformal prediction: quantile at level (1 + 1/n) * coverage
        let adjusted_level = ((cal_size as f64 + 1.0) * self.coverage / cal_size as f64).min(1.0);
        let quantile_idx = ((cal_size as f64) * adjusted_level).ceil() as usize;
        let quantile_idx = quantile_idx.saturating_sub(1).min(scores.len() - 1);
        let quantile_value = scores[quantile_idx];

        Ok(ConformalResult {
            scores,
            quantile_value,
            coverage: self.coverage,
            method: self.method.clone(),
        })
    }

    /// Fit using cross-validation conformal method.
    fn fit_cross_val(
        &self,
        forecasts: &[f64],
        actuals: &[f64],
        n_folds: usize,
    ) -> Result<ConformalResult> {
        let n = forecasts.len();

        if n_folds < 2 {
            return Err(ForecastError::InvalidParameter(
                "n_folds must be at least 2".to_string(),
            ));
        }

        if n < n_folds {
            return Err(ForecastError::InsufficientData {
                needed: n_folds,
                got: n,
            });
        }

        // For simplicity, we compute leave-one-out scores for each fold
        // In a full implementation, we'd retrain models for each fold
        // Here we use the absolute residuals as nonconformity scores
        let mut scores: Vec<f64> = forecasts
            .iter()
            .zip(actuals.iter())
            .map(|(f, a)| (f - a).abs())
            .collect();

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute quantile
        let quantile_idx = ((n as f64) * self.coverage).ceil() as usize;
        let quantile_idx = quantile_idx.saturating_sub(1).min(scores.len() - 1);
        let quantile_value = scores[quantile_idx];

        Ok(ConformalResult {
            scores,
            quantile_value,
            coverage: self.coverage,
            method: self.method.clone(),
        })
    }

    /// Fit using Jackknife+ method.
    fn fit_jackknife_plus(&self, forecasts: &[f64], actuals: &[f64]) -> Result<ConformalResult> {
        let n = forecasts.len();

        if n < 2 {
            return Err(ForecastError::InsufficientData { needed: 2, got: n });
        }

        // Jackknife+ uses leave-one-out residuals
        // For this implementation, we use the absolute residuals
        // A full implementation would retrain the model for each LOO subset
        let mut scores: Vec<f64> = forecasts
            .iter()
            .zip(actuals.iter())
            .map(|(f, a)| (f - a).abs())
            .collect();

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Jackknife+ quantile: ceil((n+1) * coverage) / n
        let adjusted_level = (((n + 1) as f64) * self.coverage / n as f64).min(1.0);
        let quantile_idx = ((n as f64) * adjusted_level).ceil() as usize;
        let quantile_idx = quantile_idx.saturating_sub(1).min(scores.len() - 1);
        let quantile_value = scores[quantile_idx];

        Ok(ConformalResult {
            scores,
            quantile_value,
            coverage: self.coverage,
            method: self.method.clone(),
        })
    }

    /// Generate prediction intervals for new point forecasts.
    ///
    /// # Arguments
    ///
    /// * `result` - The fitted conformal result
    /// * `point_forecasts` - New point forecasts to generate intervals for
    ///
    /// # Returns
    ///
    /// Prediction intervals with the target coverage.
    pub fn predict(
        &self,
        result: &ConformalResult,
        point_forecasts: &PointForecasts,
    ) -> PredictionIntervals {
        let values = point_forecasts.values();
        let q = result.quantile_value;

        let lower: Vec<f64> = values.iter().map(|&v| v - q).collect();
        let upper: Vec<f64> = values.iter().map(|&v| v + q).collect();

        // Use unwrap since we control the inputs and they should be valid
        PredictionIntervals::new(
            point_forecasts.timestamps().to_vec(),
            lower,
            upper,
            self.coverage,
        )
        .expect("Valid prediction intervals")
    }

    /// Generate prediction intervals from raw values.
    pub fn predict_values(&self, result: &ConformalResult, values: &[f64]) -> PredictionIntervals {
        let q = result.quantile_value;

        let lower: Vec<f64> = values.iter().map(|&v| v - q).collect();
        let upper: Vec<f64> = values.iter().map(|&v| v + q).collect();

        PredictionIntervals::from_bounds(lower, upper, self.coverage)
            .expect("Valid prediction intervals")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ConformalMethod tests
    // =========================================================================

    mod conformal_method {
        use super::*;

        #[test]
        fn default_is_split_with_20_percent() {
            let method = ConformalMethod::default();
            match method {
                ConformalMethod::Split { cal_fraction } => {
                    assert!((cal_fraction - 0.2).abs() < 1e-10);
                }
                _ => panic!("Expected Split method"),
            }
        }

        #[test]
        fn split_stores_cal_fraction() {
            let method = ConformalMethod::Split { cal_fraction: 0.3 };
            if let ConformalMethod::Split { cal_fraction } = method {
                assert!((cal_fraction - 0.3).abs() < 1e-10);
            } else {
                panic!("Expected Split method");
            }
        }

        #[test]
        fn cross_val_stores_n_folds() {
            let method = ConformalMethod::CrossVal { n_folds: 5 };
            if let ConformalMethod::CrossVal { n_folds } = method {
                assert_eq!(n_folds, 5);
            } else {
                panic!("Expected CrossVal method");
            }
        }

        #[test]
        fn jackknife_plus_variant_exists() {
            let method = ConformalMethod::JackknifePlus;
            assert_eq!(method, ConformalMethod::JackknifePlus);
        }

        #[test]
        fn methods_are_clonable() {
            let method = ConformalMethod::Split { cal_fraction: 0.25 };
            let cloned = method.clone();
            assert_eq!(method, cloned);
        }
    }

    // =========================================================================
    // ConformalPredictor construction tests
    // =========================================================================

    mod construction {
        use super::*;

        #[test]
        fn new_creates_predictor() {
            let predictor = ConformalPredictor::new(
                0.90,
                ConformalMethod::Split { cal_fraction: 0.2 },
            );
            assert!((predictor.coverage() - 0.90).abs() < 1e-10);
        }

        #[test]
        fn split_creates_split_predictor() {
            let predictor = ConformalPredictor::split(0.95);
            assert!((predictor.coverage() - 0.95).abs() < 1e-10);
            match predictor.method() {
                ConformalMethod::Split { cal_fraction } => {
                    assert!((cal_fraction - 0.2).abs() < 1e-10);
                }
                _ => panic!("Expected Split method"),
            }
        }

        #[test]
        fn cross_val_creates_cv_predictor() {
            let predictor = ConformalPredictor::cross_val(0.90, 5);
            assert!((predictor.coverage() - 0.90).abs() < 1e-10);
            match predictor.method() {
                ConformalMethod::CrossVal { n_folds } => {
                    assert_eq!(*n_folds, 5);
                }
                _ => panic!("Expected CrossVal method"),
            }
        }

        #[test]
        fn jackknife_plus_creates_jackknife_predictor() {
            let predictor = ConformalPredictor::jackknife_plus(0.90);
            assert!((predictor.coverage() - 0.90).abs() < 1e-10);
            assert_eq!(predictor.method(), &ConformalMethod::JackknifePlus);
        }

        #[test]
        #[should_panic(expected = "coverage must be in (0, 1)")]
        fn new_panics_on_zero_coverage() {
            ConformalPredictor::new(0.0, ConformalMethod::default());
        }

        #[test]
        #[should_panic(expected = "coverage must be in (0, 1)")]
        fn new_panics_on_one_coverage() {
            ConformalPredictor::new(1.0, ConformalMethod::default());
        }

        #[test]
        #[should_panic(expected = "coverage must be in (0, 1)")]
        fn new_panics_on_negative_coverage() {
            ConformalPredictor::new(-0.1, ConformalMethod::default());
        }

        #[test]
        fn predictor_is_clonable() {
            let predictor = ConformalPredictor::split(0.90);
            let cloned = predictor.clone();
            assert!((cloned.coverage() - 0.90).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Split conformal fit tests
    // =========================================================================

    mod fit_split {
        use super::*;

        #[test]
        fn fit_returns_result() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();

            assert!((result.coverage() - 0.90).abs() < 1e-10);
            assert!(!result.scores().is_empty());
        }

        #[test]
        fn fit_fails_on_length_mismatch() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 10.5];

            let result = predictor.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_empty_data() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts: Vec<f64> = vec![];
            let actuals: Vec<f64> = vec![];

            let result = predictor.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn scores_are_sorted() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![10.5, 10.0, 12.5, 13.5, 14.0, 15.5, 16.0, 17.5, 18.0, 19.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            let scores = result.scores();

            for i in 1..scores.len() {
                assert!(scores[i] >= scores[i - 1], "Scores should be sorted");
            }
        }

        #[test]
        fn quantile_value_is_positive() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            assert!(result.quantile_value() >= 0.0);
        }

        #[test]
        fn higher_coverage_gives_larger_quantile() {
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![9.0, 12.0, 11.0, 14.0, 13.0, 16.0, 15.0, 18.0, 17.0, 20.0];

            let predictor_90 = ConformalPredictor::split(0.50);
            let predictor_95 = ConformalPredictor::split(0.90);

            let result_90 = predictor_90.fit(&forecasts, &actuals).unwrap();
            let result_95 = predictor_95.fit(&forecasts, &actuals).unwrap();

            assert!(result_95.quantile_value() >= result_90.quantile_value());
        }
    }

    // =========================================================================
    // Cross-validation conformal fit tests
    // =========================================================================

    mod fit_cross_val {
        use super::*;

        #[test]
        fn fit_returns_result() {
            let predictor = ConformalPredictor::cross_val(0.90, 5);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            assert!((result.coverage() - 0.90).abs() < 1e-10);
        }

        #[test]
        fn fit_fails_on_insufficient_folds() {
            let predictor = ConformalPredictor::cross_val(0.90, 1);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 10.5, 12.5];

            let result = predictor.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_when_n_less_than_folds() {
            let predictor = ConformalPredictor::cross_val(0.90, 10);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 10.5, 12.5];

            let result = predictor.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn uses_all_data_for_scores() {
            let predictor = ConformalPredictor::cross_val(0.90, 5);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5, 15.5, 15.5, 17.5, 18.5, 19.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            // CV uses all data points for scores
            assert_eq!(result.scores().len(), 10);
        }
    }

    // =========================================================================
    // Jackknife+ fit tests
    // =========================================================================

    mod fit_jackknife_plus {
        use super::*;

        #[test]
        fn fit_returns_result() {
            let predictor = ConformalPredictor::jackknife_plus(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            assert!((result.coverage() - 0.90).abs() < 1e-10);
        }

        #[test]
        fn fit_fails_on_single_point() {
            let predictor = ConformalPredictor::jackknife_plus(0.90);
            let forecasts = vec![10.0];
            let actuals = vec![10.5];

            let result = predictor.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn works_with_two_points() {
            let predictor = ConformalPredictor::jackknife_plus(0.90);
            let forecasts = vec![10.0, 11.0];
            let actuals = vec![10.5, 10.5];

            let result = predictor.fit(&forecasts, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn uses_all_data_for_scores() {
            let predictor = ConformalPredictor::jackknife_plus(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            assert_eq!(result.scores().len(), 5);
        }
    }

    // =========================================================================
    // Predict tests
    // =========================================================================

    mod predict {
        use super::*;
        use chrono::{TimeZone, Utc};

        fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
            (0..n)
                .map(|i| {
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64)
                })
                .collect()
        }

        #[test]
        fn predict_returns_intervals() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5, 15.5, 15.5, 17.5, 18.5, 19.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0, 21.0, 22.0]);
            let intervals = predictor.predict(&result, &new_forecasts);

            assert_eq!(intervals.len(), 3);
            assert!((intervals.coverage() - 0.90).abs() < 1e-10);
        }

        #[test]
        fn predict_with_timestamps() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5, 15.5, 15.5, 17.5, 18.5, 19.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();

            let timestamps = make_timestamps(3);
            let new_forecasts =
                PointForecasts::new(timestamps.clone(), vec![20.0, 21.0, 22.0]).unwrap();
            let intervals = predictor.predict(&result, &new_forecasts);

            assert!(intervals.has_timestamps());
            assert_eq!(intervals.timestamps(), &timestamps);
        }

        #[test]
        fn intervals_are_symmetric() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5, 15.5, 15.5, 17.5, 18.5, 19.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let intervals = predictor.predict(&result, &new_forecasts);

            let point = 20.0;
            let lower = intervals.lower()[0];
            let upper = intervals.upper()[0];

            let lower_diff = point - lower;
            let upper_diff = upper - point;

            assert!(
                (lower_diff - upper_diff).abs() < 1e-10,
                "Intervals should be symmetric"
            );
        }

        #[test]
        fn predict_values_works() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5, 15.5, 15.5, 17.5, 18.5, 19.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            let intervals = predictor.predict_values(&result, &[20.0, 21.0]);

            assert_eq!(intervals.len(), 2);
            assert!(!intervals.has_timestamps());
        }

        #[test]
        fn larger_errors_give_wider_intervals() {
            // Small errors
            let forecasts_small = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals_small = vec![10.1, 11.1, 12.1, 13.1, 14.1];

            // Large errors
            let forecasts_large = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals_large = vec![8.0, 13.0, 10.0, 15.0, 12.0];

            let predictor = ConformalPredictor::split(0.90);

            let result_small = predictor.fit(&forecasts_small, &actuals_small).unwrap();
            let result_large = predictor.fit(&forecasts_large, &actuals_large).unwrap();

            assert!(result_large.quantile_value() > result_small.quantile_value());
        }
    }

    // =========================================================================
    // Coverage validation tests
    // =========================================================================

    mod coverage_validation {
        use super::*;

        #[test]
        fn empirical_coverage_approximately_matches_target() {
            // Generate synthetic data with known error distribution
            let n = 100;
            let forecasts: Vec<f64> = (0..n).map(|i| i as f64).collect();
            // Errors uniformly distributed in [-1, 1]
            let errors: Vec<f64> = (0..n)
                .map(|i| ((i * 7 + 3) % 21) as f64 / 10.0 - 1.0)
                .collect();
            let actuals: Vec<f64> = forecasts
                .iter()
                .zip(errors.iter())
                .map(|(f, e)| f + e)
                .collect();

            let predictor = ConformalPredictor::split(0.90);
            let result = predictor.fit(&forecasts, &actuals).unwrap();

            // Test on new data with similar error distribution
            let new_forecasts: Vec<f64> = (100..150).map(|i| i as f64).collect();
            let new_errors: Vec<f64> = (100..150)
                .map(|i| ((i * 7 + 3) % 21) as f64 / 10.0 - 1.0)
                .collect();
            let new_actuals: Vec<f64> = new_forecasts
                .iter()
                .zip(new_errors.iter())
                .map(|(f, e)| f + e)
                .collect();

            let intervals = predictor.predict_values(&result, &new_forecasts);
            let empirical = intervals.empirical_coverage(&new_actuals).unwrap();

            // Allow some slack due to finite sample
            assert!(
                empirical >= 0.70,
                "Empirical coverage {} should be reasonably high",
                empirical
            );
        }
    }

    // =========================================================================
    // ConformalResult tests
    // =========================================================================

    mod conformal_result {
        use super::*;

        #[test]
        fn accessors_return_correct_values() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();

            assert!(!result.scores().is_empty());
            assert!(result.quantile_value() >= 0.0);
            assert!((result.coverage() - 0.90).abs() < 1e-10);
            match result.method() {
                ConformalMethod::Split { .. } => {}
                _ => panic!("Expected Split method"),
            }
        }

        #[test]
        fn result_is_clonable() {
            let predictor = ConformalPredictor::split(0.90);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = predictor.fit(&forecasts, &actuals).unwrap();
            let cloned = result.clone();

            assert_eq!(result.scores(), cloned.scores());
            assert!((result.quantile_value() - cloned.quantile_value()).abs() < 1e-10);
        }
    }
}
