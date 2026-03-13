//! Normal predictor for Gaussian-based probabilistic forecasting.
//!
//! The Normal predictor assumes forecast errors follow a Gaussian distribution.
//! It estimates the mean and standard deviation of errors from historical data
//! and uses these to compute quantile forecasts.
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::NormalPredictor;
//!
//! let mut predictor = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
//!
//! // Fit on historical forecast errors
//! let result = predictor.fit(&historical_forecasts, &historical_actuals).unwrap();
//!
//! // Generate quantile forecasts
//! let quantiles = predictor.predict(&result, &new_forecasts);
//! ```

use crate::error::{ForecastError, Result};
use crate::postprocess::{PointForecasts, QuantileForecasts};

/// Result of fitting a normal predictor.
#[derive(Debug, Clone)]
pub struct NormalResult {
    /// Mean of errors (bias).
    mean: f64,
    /// Standard deviation of errors.
    std_dev: f64,
    /// Quantile z-scores (standard normal quantiles).
    z_scores: Vec<f64>,
    /// The quantile levels.
    quantiles: Vec<f64>,
}

impl NormalResult {
    /// Get the mean error (bias).
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Get the z-scores for each quantile.
    pub fn z_scores(&self) -> &[f64] {
        &self.z_scores
    }

    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Compute the quantile adjustment for each quantile level.
    ///
    /// Returns: mean + z_score * std_dev for each quantile.
    pub fn quantile_adjustments(&self) -> Vec<f64> {
        self.z_scores
            .iter()
            .map(|&z| self.mean + z * self.std_dev)
            .collect()
    }
}

/// Normal predictor for Gaussian error assumption.
///
/// Assumes forecast errors are normally distributed and uses the
/// estimated mean and standard deviation to compute quantile forecasts.
#[derive(Debug, Clone)]
pub struct NormalPredictor {
    /// Target quantile levels.
    quantiles: Vec<f64>,
}

impl NormalPredictor {
    /// Create a new normal predictor.
    ///
    /// # Arguments
    ///
    /// * `quantiles` - Target quantile levels (must be in (0, 1) and sorted)
    ///
    /// # Panics
    ///
    /// Panics if quantiles are invalid or not sorted.
    pub fn new(quantiles: Vec<f64>) -> Self {
        // Validate quantiles
        for &q in &quantiles {
            assert!(q > 0.0 && q < 1.0, "quantiles must be in (0, 1)");
        }
        for w in quantiles.windows(2) {
            assert!(w[0] < w[1], "quantiles must be sorted in ascending order");
        }

        Self { quantiles }
    }

    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Fit the normal predictor on historical forecasts and actuals.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - Historical point forecasts
    /// * `actuals` - Corresponding actual values
    ///
    /// # Returns
    ///
    /// A `NormalResult` containing the Gaussian parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forecasts and actuals have different lengths
    /// - Not enough data points (need at least 2 for std dev)
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<NormalResult> {
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

        if n < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: n,
                hint: None,
            });
        }

        // Compute errors (actual - forecast)
        let errors: Vec<f64> = forecasts
            .iter()
            .zip(actuals.iter())
            .map(|(f, a)| a - f)
            .collect();

        // Compute mean
        let mean = errors.iter().sum::<f64>() / n as f64;

        // Compute standard deviation (sample std dev with n-1)
        let variance = errors.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();

        // Compute z-scores for each quantile using the inverse normal CDF
        let z_scores: Vec<f64> = self.quantiles.iter().map(|&q| quantile_normal(q)).collect();

        Ok(NormalResult {
            mean,
            std_dev,
            z_scores,
            quantiles: self.quantiles.clone(),
        })
    }

    /// Generate quantile forecasts for new point forecasts.
    ///
    /// # Arguments
    ///
    /// * `result` - The fitted normal result
    /// * `point_forecasts` - New point forecasts
    ///
    /// # Returns
    ///
    /// Quantile forecasts at the specified quantile levels.
    pub fn predict(
        &self,
        result: &NormalResult,
        point_forecasts: &PointForecasts,
    ) -> Result<QuantileForecasts> {
        let values = point_forecasts.values();
        let adjustments = result.quantile_adjustments();

        // For each time point, add quantile adjustments
        let forecast_values: Vec<Vec<f64>> = values
            .iter()
            .map(|&v| adjustments.iter().map(|&adj| v + adj).collect())
            .collect();

        QuantileForecasts::new(
            point_forecasts.timestamps().to_vec(),
            self.quantiles.clone(),
            forecast_values,
        )
    }

    /// Generate quantile forecasts from raw values.
    pub fn predict_values(
        &self,
        result: &NormalResult,
        values: &[f64],
    ) -> Result<QuantileForecasts> {
        let adjustments = result.quantile_adjustments();

        let forecast_values: Vec<Vec<f64>> = values
            .iter()
            .map(|&v| adjustments.iter().map(|&adj| v + adj).collect())
            .collect();

        QuantileForecasts::from_values(self.quantiles.clone(), forecast_values)
    }
}

/// Compute the quantile (inverse CDF) of the standard normal distribution.
///
/// Uses the Abramowitz and Stegun approximation.
fn quantile_normal(p: f64) -> f64 {
    // Handle edge cases
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Rational approximation for the standard normal quantile function
    // Based on Abramowitz and Stegun approximation 26.2.23
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let q: f64;
    let mut r: f64;

    if p < p_low {
        // Lower tail
        q = (-2.0 * p.ln()).sqrt();
        r = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if p <= p_high {
        // Central region
        q = p - 0.5;
        r = q * q;
        r = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        // Upper tail
        q = (-2.0 * (1.0 - p).ln()).sqrt();
        r = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }

    // =========================================================================
    // quantile_normal tests
    // =========================================================================

    mod quantile_normal_tests {
        use super::*;

        #[test]
        fn median_is_zero() {
            let z = quantile_normal(0.5);
            assert_relative_eq!(z, 0.0, epsilon = 1e-10);
        }

        #[test]
        fn symmetric_around_median() {
            let z_low = quantile_normal(0.25);
            let z_high = quantile_normal(0.75);
            assert_relative_eq!(z_low + z_high, 0.0, epsilon = 1e-6);
        }

        #[test]
        fn symmetric_around_median_for_tails() {
            let z_low = quantile_normal(0.1);
            let z_high = quantile_normal(0.9);
            assert_relative_eq!(z_low + z_high, 0.0, epsilon = 1e-6);
        }

        #[test]
        fn symmetric_around_median_for_extreme_tails() {
            let z_low = quantile_normal(0.01);
            let z_high = quantile_normal(0.99);
            assert_relative_eq!(z_low + z_high, 0.0, epsilon = 1e-4);
        }

        #[test]
        fn known_values() {
            // Standard normal quantiles from tables
            assert_relative_eq!(quantile_normal(0.1), -1.2816, epsilon = 0.01);
            assert_relative_eq!(quantile_normal(0.9), 1.2816, epsilon = 0.01);
            assert_relative_eq!(quantile_normal(0.95), 1.6449, epsilon = 0.01);
            assert_relative_eq!(quantile_normal(0.025), -1.96, epsilon = 0.01);
            assert_relative_eq!(quantile_normal(0.975), 1.96, epsilon = 0.01);
        }

        #[test]
        fn monotonically_increasing() {
            let quantiles = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
            let z_scores: Vec<f64> = quantiles.iter().map(|&q| quantile_normal(q)).collect();

            for i in 1..z_scores.len() {
                assert!(
                    z_scores[i] > z_scores[i - 1],
                    "z-scores should be monotonically increasing"
                );
            }
        }

        #[test]
        fn extreme_quantiles() {
            let z_01 = quantile_normal(0.01);
            let z_99 = quantile_normal(0.99);

            assert!(z_01 < -2.0, "z(0.01) should be < -2");
            assert!(z_99 > 2.0, "z(0.99) should be > 2");
        }

        #[test]
        fn boundary_returns_infinity() {
            assert!(quantile_normal(0.0).is_infinite());
            assert!(quantile_normal(0.0).is_sign_negative());
            assert!(quantile_normal(1.0).is_infinite());
            assert!(quantile_normal(1.0).is_sign_positive());
        }

        #[test]
        fn fine_grained_monotonicity() {
            let quantiles: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();
            let z_scores: Vec<f64> = quantiles.iter().map(|&q| quantile_normal(q)).collect();

            for i in 1..z_scores.len() {
                assert!(
                    z_scores[i] > z_scores[i - 1],
                    "z({}) = {} should be > z({}) = {}",
                    quantiles[i],
                    z_scores[i],
                    quantiles[i - 1],
                    z_scores[i - 1]
                );
            }
        }
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    mod construction {
        use super::*;

        #[test]
        fn new_creates_predictor() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            assert_eq!(pred.quantiles(), &[0.1, 0.5, 0.9]);
        }

        #[test]
        fn new_single_quantile() {
            let pred = NormalPredictor::new(vec![0.5]);
            assert_eq!(pred.quantiles(), &[0.5]);
        }

        #[test]
        fn new_many_quantiles() {
            let quantiles: Vec<f64> = (1..20).map(|i| i as f64 / 20.0).collect();
            let pred = NormalPredictor::new(quantiles.clone());
            assert_eq!(pred.quantiles(), &quantiles);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_zero_quantile() {
            NormalPredictor::new(vec![0.0, 0.5, 0.9]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_one_quantile() {
            NormalPredictor::new(vec![0.1, 0.5, 1.0]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_negative_quantile() {
            NormalPredictor::new(vec![-0.1, 0.5, 0.9]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_quantile_greater_than_one() {
            NormalPredictor::new(vec![0.1, 0.5, 1.5]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be sorted")]
        fn new_panics_on_unsorted_quantiles() {
            NormalPredictor::new(vec![0.9, 0.5, 0.1]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be sorted")]
        fn new_panics_on_duplicate_quantiles() {
            NormalPredictor::new(vec![0.5, 0.5, 0.9]);
        }

        #[test]
        fn predictor_is_clonable() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let cloned = pred.clone();
            assert_eq!(pred.quantiles(), cloned.quantiles());
        }

        #[test]
        fn predictor_is_debuggable() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let debug_str = format!("{:?}", pred);
            assert!(debug_str.contains("NormalPredictor"));
        }
    }

    // =========================================================================
    // Fit tests
    // =========================================================================

    mod fit {
        use super::*;

        #[test]
        fn fit_returns_result() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
            assert_eq!(result.z_scores().len(), 3);
        }

        #[test]
        fn fit_fails_on_length_mismatch() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 11.5];

            let err = pred.fit(&forecasts, &actuals).unwrap_err();
            assert!(
                matches!(err, ForecastError::DimensionMismatch { .. }),
                "Expected DimensionMismatch, got {:?}",
                err
            );
        }

        #[test]
        fn fit_fails_on_empty_data() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts: Vec<f64> = vec![];
            let actuals: Vec<f64> = vec![];

            let err = pred.fit(&forecasts, &actuals).unwrap_err();
            assert!(
                matches!(err, ForecastError::EmptyData),
                "Expected EmptyData, got {:?}",
                err
            );
        }

        #[test]
        fn fit_fails_on_single_point() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0];
            let actuals = vec![10.5];

            let err = pred.fit(&forecasts, &actuals).unwrap_err();
            assert!(
                matches!(
                    err,
                    ForecastError::InsufficientData {
                        needed: 2,
                        got: 1,
                        ..
                    }
                ),
                "Expected InsufficientData, got {:?}",
                err
            );
        }

        #[test]
        fn mean_is_average_error() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Errors: 1, 1, 1, 1, 1 -> mean = 1
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 1.0, epsilon = 1e-10);
        }

        #[test]
        fn mean_computed_correctly_for_mixed_errors() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Errors: -2, -1, 0, 1, 2 -> mean = 0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 10.0, 11.0, 12.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 0.0, epsilon = 1e-10);
        }

        #[test]
        fn std_dev_is_zero_for_constant_errors() {
            let pred = NormalPredictor::new(vec![0.5]);

            // All errors are 1.0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.std_dev(), 0.0, epsilon = 1e-10);
        }

        #[test]
        fn std_dev_is_positive_for_varying_errors() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Varying errors
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![9.0, 10.0, 11.0, 12.0, 13.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert!(result.std_dev() > 0.0);
        }

        #[test]
        fn std_dev_uses_sample_formula() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Errors: -1, 0, 1 -> mean = 0, sample variance = (1+0+1)/2 = 1
            let forecasts = vec![10.0, 10.0, 10.0];
            let actuals = vec![9.0, 10.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.std_dev(), 1.0, epsilon = 1e-10);
        }

        #[test]
        fn unbiased_forecasts_have_zero_mean() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Symmetric errors around 0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 11.0, 12.0]; // Errors: -2, -1, 1, 2

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 0.0, epsilon = 1e-10);
        }

        #[test]
        fn z_scores_match_quantile_normal() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            assert_relative_eq!(result.z_scores()[0], quantile_normal(0.1), epsilon = 1e-10);
            assert_relative_eq!(result.z_scores()[1], quantile_normal(0.5), epsilon = 1e-10);
            assert_relative_eq!(result.z_scores()[2], quantile_normal(0.9), epsilon = 1e-10);
        }

        #[test]
        fn fit_with_negative_values() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![-5.0, -3.0, -1.0, 1.0, 3.0];
            let actuals = vec![-4.5, -2.5, -0.5, 1.5, 3.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 0.5, epsilon = 1e-10);
        }
    }

    // =========================================================================
    // Predict tests
    // =========================================================================

    mod predict {
        use super::*;

        #[test]
        fn predict_returns_quantile_forecasts() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0, 21.0]);
            let quantiles = pred.predict(&result, &new_forecasts).unwrap();

            assert_eq!(quantiles.n_times(), 2);
            assert_eq!(quantiles.n_quantiles(), 3);
        }

        #[test]
        fn predict_with_timestamps() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let timestamps = make_timestamps(2);
            let new_forecasts = PointForecasts::new(timestamps.clone(), vec![20.0, 21.0]).unwrap();
            let quantiles = pred.predict(&result, &new_forecasts).unwrap();

            assert!(quantiles.has_timestamps());
            assert_eq!(quantiles.timestamps(), &timestamps);
        }

        #[test]
        fn predict_values_works() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let quantiles = pred.predict_values(&result, &[20.0, 21.0]).unwrap();

            assert_eq!(quantiles.n_times(), 2);
            assert!(!quantiles.has_timestamps());
        }

        #[test]
        fn predict_single_value() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let quantiles = pred.predict_values(&result, &[20.0]).unwrap();

            assert_eq!(quantiles.n_times(), 1);
            assert_eq!(quantiles.n_quantiles(), 1);
            assert_relative_eq!(quantiles.at_time(0).unwrap()[0], 21.0, epsilon = 1e-10);
        }

        #[test]
        fn quantile_values_are_monotonic() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.0, 12.5, 13.5, 14.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = pred.predict(&result, &new_forecasts).unwrap();

            let row = quantiles.at_time(0).unwrap();
            assert!(row[0] <= row[1] && row[1] <= row[2]);
        }

        #[test]
        fn median_forecast_shifted_by_mean_error() {
            let pred = NormalPredictor::new(vec![0.5]);

            // All errors are +1
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = pred.predict(&result, &new_forecasts).unwrap();

            let median = quantiles.at_time(0).unwrap()[0];
            // Point forecast 20.0 + mean error 1.0 = 21.0
            assert_relative_eq!(median, 21.0, epsilon = 1e-10);
        }

        #[test]
        fn zero_std_dev_gives_point_forecasts() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);

            // Constant error = 1.0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = pred.predict(&result, &new_forecasts).unwrap();

            let row = quantiles.at_time(0).unwrap();
            // With zero std dev, all quantiles should equal point + mean
            assert_relative_eq!(row[0], 21.0, epsilon = 1e-10);
            assert_relative_eq!(row[1], 21.0, epsilon = 1e-10);
            assert_relative_eq!(row[2], 21.0, epsilon = 1e-10);
        }

        #[test]
        fn larger_std_dev_gives_wider_intervals() {
            let pred = NormalPredictor::new(vec![0.1, 0.9]);

            // Small variance
            let forecasts_small = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals_small = vec![9.9, 10.0, 10.1, 10.0, 10.0];

            // Large variance
            let forecasts_large = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals_large = vec![8.0, 10.0, 12.0, 10.0, 10.0];

            let result_small = pred.fit(&forecasts_small, &actuals_small).unwrap();
            let result_large = pred.fit(&forecasts_large, &actuals_large).unwrap();

            let q_small = pred.predict_values(&result_small, &[20.0]).unwrap();
            let q_large = pred.predict_values(&result_large, &[20.0]).unwrap();

            let width_small = q_small.at_time(0).unwrap()[1] - q_small.at_time(0).unwrap()[0];
            let width_large = q_large.at_time(0).unwrap()[1] - q_large.at_time(0).unwrap()[0];

            assert!(width_large > width_small);
        }

        #[test]
        fn quantile_adjustments_are_symmetric_for_symmetric_quantiles() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);

            // Symmetric errors: mean=0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 11.0, 12.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let adjustments = result.quantile_adjustments();

            // q0.1 adjustment should be -q0.9 adjustment (symmetric around mean=0)
            assert_relative_eq!(adjustments[0], -adjustments[2], epsilon = 1e-6);
            // Median adjustment should be ~0 (mean=0, z=0)
            assert_relative_eq!(adjustments[1], 0.0, epsilon = 1e-10);
        }

        #[test]
        fn predict_returns_finite_values() {
            let pred = NormalPredictor::new(vec![0.01, 0.5, 0.99]);
            let forecasts: Vec<f64> = (0..20).map(|i| i as f64 * 10.0).collect();
            let actuals: Vec<f64> = forecasts
                .iter()
                .enumerate()
                .map(|(i, &f)| f + (i as f64).sin() * 5.0)
                .collect();

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let quantiles = pred.predict_values(&result, &[50.0, 100.0]).unwrap();

            for t in 0..2 {
                let row = quantiles.at_time(t).unwrap();
                for &val in row {
                    assert!(val.is_finite(), "All predicted quantiles must be finite");
                }
            }
        }

        #[test]
        fn predict_and_predict_values_agree() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let values = vec![20.0, 21.0];
            let q_predict = pred
                .predict(&result, &PointForecasts::from_values(values.clone()))
                .unwrap();
            let q_values = pred.predict_values(&result, &values).unwrap();

            for t in 0..2 {
                let row_predict = q_predict.at_time(t).unwrap();
                let row_values = q_values.at_time(t).unwrap();
                for i in 0..3 {
                    assert_relative_eq!(row_predict[i], row_values[i], epsilon = 1e-10);
                }
            }
        }
    }

    // =========================================================================
    // Result tests
    // =========================================================================

    mod normal_result {
        use super::*;

        #[test]
        fn accessors_return_correct_values() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            assert_relative_eq!(result.mean(), 0.5, epsilon = 1e-10);
            assert!(result.std_dev() >= 0.0);
            assert_eq!(result.z_scores().len(), 3);
            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
        }

        #[test]
        fn quantile_adjustments_computed_correctly() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let adjustments = result.quantile_adjustments();

            // Mean = 1.0, std_dev = 0, z(0.5) = 0
            // Adjustment = 1.0 + 0 * 0 = 1.0
            assert_eq!(adjustments.len(), 1);
            assert_relative_eq!(adjustments[0], 1.0, epsilon = 1e-10);
        }

        #[test]
        fn quantile_adjustments_formula_verification() {
            let pred = NormalPredictor::new(vec![0.1, 0.9]);

            // Errors: -1, 0, 1 -> mean=0, std_dev=1 (sample)
            let forecasts = vec![10.0, 10.0, 10.0];
            let actuals = vec![9.0, 10.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let adjustments = result.quantile_adjustments();

            // adj[i] = mean + z_score[i] * std_dev = 0 + z * 1 = z
            assert_relative_eq!(adjustments[0], quantile_normal(0.1), epsilon = 1e-6);
            assert_relative_eq!(adjustments[1], quantile_normal(0.9), epsilon = 1e-6);
        }

        #[test]
        fn result_is_clonable() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let cloned = result.clone();

            assert_relative_eq!(result.mean(), cloned.mean(), epsilon = 1e-10);
            assert_relative_eq!(result.std_dev(), cloned.std_dev(), epsilon = 1e-10);
            assert_eq!(result.z_scores(), cloned.z_scores());
            assert_eq!(result.quantiles(), cloned.quantiles());
        }

        #[test]
        fn result_is_debuggable() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0, 12.0, 14.0];
            let actuals = vec![10.5, 12.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let debug_str = format!("{:?}", result);
            assert!(debug_str.contains("NormalResult"));
        }
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    mod edge_cases {
        use super::*;

        #[test]
        fn two_data_points() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0, 12.0];
            let actuals = vec![11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_ok());
        }

        #[test]
        fn two_data_points_correct_statistics() {
            let pred = NormalPredictor::new(vec![0.5]);
            // Errors: 1.0, -1.0 -> mean=0, sample_variance=2/(2-1)=2, std=sqrt(2)
            let forecasts = vec![10.0, 12.0];
            let actuals = vec![11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(result.std_dev(), std::f64::consts::SQRT_2, epsilon = 1e-10);
        }

        #[test]
        fn negative_errors() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 7.0]; // All negative errors

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert!(result.mean() < 0.0);
        }

        #[test]
        fn many_quantiles() {
            let quantiles: Vec<f64> = (1..10).map(|i| i as f64 / 10.0).collect();
            let pred = NormalPredictor::new(quantiles.clone());
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![9.0, 10.0, 11.0, 12.0, 8.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let q_forecasts = pred.predict_values(&result, &[20.0]).unwrap();

            assert_eq!(q_forecasts.n_quantiles(), 9);

            // All quantile values should be monotonically non-decreasing
            let row = q_forecasts.at_time(0).unwrap();
            for i in 1..row.len() {
                assert!(
                    row[i] >= row[i - 1],
                    "Quantiles should be monotonic: q[{}]={} < q[{}]={}",
                    i - 1,
                    row[i - 1],
                    i,
                    row[i]
                );
            }
        }

        #[test]
        fn identical_forecasts_and_actuals() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![5.0, 5.0, 5.0, 5.0, 5.0];
            let actuals = vec![5.0, 5.0, 5.0, 5.0, 5.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(result.std_dev(), 0.0, epsilon = 1e-10);

            // All quantile predictions should equal the point forecast
            let q = pred.predict_values(&result, &[42.0]).unwrap();
            let row = q.at_time(0).unwrap();
            for &val in row {
                assert_relative_eq!(val, 42.0, epsilon = 1e-10);
            }
        }

        #[test]
        fn large_values() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![1e10, 2e10, 3e10, 4e10, 5e10];
            let actuals = vec![1.1e10, 2.1e10, 3.1e10, 4.1e10, 5.1e10];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 1e9, epsilon = 1e2);
        }

        #[test]
        fn very_small_values() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![1e-10, 2e-10, 3e-10, 4e-10, 5e-10];
            let actuals = vec![1.1e-10, 2.1e-10, 3.1e-10, 4.1e-10, 5.1e-10];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert_relative_eq!(result.mean(), 1e-11, epsilon = 1e-18);
        }
    }
}
