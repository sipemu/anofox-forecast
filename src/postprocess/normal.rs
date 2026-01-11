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
            return Err(ForecastError::InsufficientData { needed: 2, got: n });
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
        let z_scores: Vec<f64> = self
            .quantiles
            .iter()
            .map(|&q| quantile_normal(q))
            .collect();

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
        1.383577518672690e+02,
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
            assert!(z.abs() < 1e-10, "z(0.5) should be 0, got {}", z);
        }

        #[test]
        fn symmetric_around_median() {
            let z_low = quantile_normal(0.25);
            let z_high = quantile_normal(0.75);
            assert!(
                (z_low + z_high).abs() < 1e-6,
                "z(0.25) + z(0.75) should be ~0"
            );
        }

        #[test]
        fn known_values() {
            // Standard normal quantiles
            let z_10 = quantile_normal(0.1);
            let z_90 = quantile_normal(0.9);
            let z_95 = quantile_normal(0.95);

            // Expected values from standard tables
            assert!((z_10 - (-1.2816)).abs() < 0.01, "z(0.1) ≈ -1.28");
            assert!((z_90 - 1.2816).abs() < 0.01, "z(0.9) ≈ 1.28");
            assert!((z_95 - 1.6449).abs() < 0.01, "z(0.95) ≈ 1.64");
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
        #[should_panic(expected = "quantiles must be sorted")]
        fn new_panics_on_unsorted_quantiles() {
            NormalPredictor::new(vec![0.9, 0.5, 0.1]);
        }

        #[test]
        fn predictor_is_clonable() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let cloned = pred.clone();
            assert_eq!(pred.quantiles(), cloned.quantiles());
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

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_empty_data() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts: Vec<f64> = vec![];
            let actuals: Vec<f64> = vec![];

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_single_point() {
            let pred = NormalPredictor::new(vec![0.5]);
            let forecasts = vec![10.0];
            let actuals = vec![10.5];

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn mean_is_average_error() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Errors: 1, 1, 1, 1, 1 -> mean = 1
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert!((result.mean() - 1.0).abs() < 1e-10);
        }

        #[test]
        fn std_dev_is_zero_for_constant_errors() {
            let pred = NormalPredictor::new(vec![0.5]);

            // All errors are 1.0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert!(result.std_dev().abs() < 1e-10);
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
        fn unbiased_forecasts_have_zero_mean() {
            let pred = NormalPredictor::new(vec![0.5]);

            // Symmetric errors around 0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 11.0, 12.0]; // Errors: -2, -1, 1, 2

            let result = pred.fit(&forecasts, &actuals).unwrap();
            assert!(result.mean().abs() < 1e-10);
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
            let new_forecasts =
                PointForecasts::new(timestamps.clone(), vec![20.0, 21.0]).unwrap();
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
            assert!((median - 21.0).abs() < 1e-10);
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
            assert!((row[0] - 21.0).abs() < 1e-10);
            assert!((row[1] - 21.0).abs() < 1e-10);
            assert!((row[2] - 21.0).abs() < 1e-10);
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

            assert!((result.mean() - 0.5).abs() < 1e-10);
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
            assert!((adjustments[0] - 1.0).abs() < 1e-10);
        }

        #[test]
        fn result_is_clonable() {
            let pred = NormalPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let cloned = result.clone();

            assert!((result.mean() - cloned.mean()).abs() < 1e-10);
            assert!((result.std_dev() - cloned.std_dev()).abs() < 1e-10);
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
        }
    }
}
