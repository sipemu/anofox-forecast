//! Historical simulation for non-parametric probabilistic forecasting.
//!
//! Historical simulation uses the empirical distribution of forecast errors
//! to generate probabilistic forecasts. This is a non-parametric method
//! that makes no distributional assumptions.
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::HistoricalSimulator;
//!
//! let mut simulator = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
//!
//! // Fit on historical forecast errors
//! let result = simulator.fit(&historical_forecasts, &historical_actuals).unwrap();
//!
//! // Generate quantile forecasts
//! let quantiles = simulator.predict(&result, &new_forecasts);
//! ```

use crate::error::{ForecastError, Result};
use crate::postprocess::{PointForecasts, QuantileForecasts};

/// Result of fitting a historical simulator.
#[derive(Debug, Clone)]
pub struct HistoricalSimResult {
    /// Sorted errors (residuals: actual - forecast).
    errors: Vec<f64>,
    /// Quantile values computed from the error distribution.
    quantile_values: Vec<f64>,
    /// The quantile levels.
    quantiles: Vec<f64>,
}

impl HistoricalSimResult {
    /// Get the sorted errors.
    pub fn errors(&self) -> &[f64] {
        &self.errors
    }

    /// Get the quantile values.
    pub fn quantile_values(&self) -> &[f64] {
        &self.quantile_values
    }

    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }
}

/// Historical simulator for non-parametric quantile forecasting.
///
/// Uses the empirical distribution of forecast errors to compute
/// quantile adjustments for point forecasts.
#[derive(Debug, Clone)]
pub struct HistoricalSimulator {
    /// Target quantile levels.
    quantiles: Vec<f64>,
    /// Optional rolling window size for non-stationarity.
    window: Option<usize>,
}

impl HistoricalSimulator {
    /// Create a new historical simulator.
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

        Self {
            quantiles,
            window: None,
        }
    }

    /// Create a historical simulator with a rolling window.
    ///
    /// # Arguments
    ///
    /// * `quantiles` - Target quantile levels
    /// * `window` - Number of recent observations to use
    pub fn with_window(quantiles: Vec<f64>, window: usize) -> Self {
        let mut sim = Self::new(quantiles);
        sim.window = Some(window);
        sim
    }

    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Get the window size if set.
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// Fit the historical simulator on historical forecasts and actuals.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - Historical point forecasts
    /// * `actuals` - Corresponding actual values
    ///
    /// # Returns
    ///
    /// A `HistoricalSimResult` containing the error distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forecasts and actuals have different lengths
    /// - Not enough data points
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<HistoricalSimResult> {
        if forecasts.len() != actuals.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: forecasts.len(),
                got: actuals.len(),
            });
        }

        if forecasts.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        // Compute errors (actual - forecast)
        let errors: Vec<f64> = if let Some(w) = self.window {
            // Use only the last w observations
            let start = forecasts.len().saturating_sub(w);
            forecasts[start..]
                .iter()
                .zip(actuals[start..].iter())
                .map(|(f, a)| a - f)
                .collect()
        } else {
            forecasts
                .iter()
                .zip(actuals.iter())
                .map(|(f, a)| a - f)
                .collect()
        };

        if errors.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        // Sort errors for quantile computation
        let mut sorted_errors = errors.clone();
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute quantile values
        let n = sorted_errors.len();
        let quantile_values: Vec<f64> = self
            .quantiles
            .iter()
            .map(|&q| {
                let idx = ((n as f64) * q).floor() as usize;
                let idx = idx.min(n - 1);
                sorted_errors[idx]
            })
            .collect();

        Ok(HistoricalSimResult {
            errors: sorted_errors,
            quantile_values,
            quantiles: self.quantiles.clone(),
        })
    }

    /// Generate quantile forecasts for new point forecasts.
    ///
    /// # Arguments
    ///
    /// * `result` - The fitted historical simulation result
    /// * `point_forecasts` - New point forecasts
    ///
    /// # Returns
    ///
    /// Quantile forecasts at the specified quantile levels.
    pub fn predict(
        &self,
        result: &HistoricalSimResult,
        point_forecasts: &PointForecasts,
    ) -> Result<QuantileForecasts> {
        let values = point_forecasts.values();
        let q_vals = &result.quantile_values;

        // For each time point, add quantile adjustments
        let forecast_values: Vec<Vec<f64>> = values
            .iter()
            .map(|&v| q_vals.iter().map(|&q| v + q).collect())
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
        result: &HistoricalSimResult,
        values: &[f64],
    ) -> Result<QuantileForecasts> {
        let q_vals = &result.quantile_values;

        let forecast_values: Vec<Vec<f64>> = values
            .iter()
            .map(|&v| q_vals.iter().map(|&q| v + q).collect())
            .collect();

        QuantileForecasts::from_values(self.quantiles.clone(), forecast_values)
    }
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
    // Construction tests
    // =========================================================================

    mod construction {
        use super::*;

        #[test]
        fn new_creates_simulator() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            assert_eq!(sim.quantiles(), &[0.1, 0.5, 0.9]);
            assert!(sim.window().is_none());
        }

        #[test]
        fn with_window_sets_window() {
            let sim = HistoricalSimulator::with_window(vec![0.1, 0.5, 0.9], 50);
            assert_eq!(sim.window(), Some(50));
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_zero_quantile() {
            HistoricalSimulator::new(vec![0.0, 0.5, 0.9]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_one_quantile() {
            HistoricalSimulator::new(vec![0.1, 0.5, 1.0]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be sorted")]
        fn new_panics_on_unsorted_quantiles() {
            HistoricalSimulator::new(vec![0.9, 0.5, 0.1]);
        }

        #[test]
        fn simulator_is_clonable() {
            let sim = HistoricalSimulator::with_window(vec![0.1, 0.5, 0.9], 50);
            let cloned = sim.clone();
            assert_eq!(sim.quantiles(), cloned.quantiles());
            assert_eq!(sim.window(), cloned.window());
        }
    }

    // =========================================================================
    // Fit tests
    // =========================================================================

    mod fit {
        use super::*;

        #[test]
        fn fit_returns_result() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
            assert_eq!(result.quantile_values().len(), 3);
        }

        #[test]
        fn fit_fails_on_length_mismatch() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 10.5];

            let result = sim.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_empty_data() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts: Vec<f64> = vec![];
            let actuals: Vec<f64> = vec![];

            let result = sim.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn errors_are_sorted() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.0, 12.5, 13.5, 14.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            let errors = result.errors();

            for i in 1..errors.len() {
                assert!(errors[i] >= errors[i - 1], "Errors should be sorted");
            }
        }

        #[test]
        fn quantile_values_are_sorted() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.0, 12.5, 13.5, 14.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            let q_vals = result.quantile_values();

            for i in 1..q_vals.len() {
                assert!(
                    q_vals[i] >= q_vals[i - 1],
                    "Quantile values should be sorted"
                );
            }
        }

        #[test]
        fn with_window_uses_recent_data() {
            let sim = HistoricalSimulator::with_window(vec![0.5], 3);

            // Errors for full data: [0.5, -0.5, 0.5, -0.5, 0.5]
            // Errors for last 3: [0.5, -0.5, 0.5]
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            // Should only use last 3 data points
            assert_eq!(result.errors().len(), 3);
        }

        #[test]
        fn window_larger_than_data_uses_all() {
            let sim = HistoricalSimulator::with_window(vec![0.5], 100);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 10.5, 12.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            assert_eq!(result.errors().len(), 3);
        }

        #[test]
        fn median_quantile_is_median_error() {
            let sim = HistoricalSimulator::new(vec![0.5]);

            // Errors: -1, 0, 1, 2, 3 -> median should be around 1
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![9.0, 10.0, 11.0, 12.0, 13.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            let median = result.quantile_values()[0];

            // With 5 points, median index = floor(5 * 0.5) = 2
            // Sorted errors: [-1, 0, 1, 2, 3], so median = 1
            assert!((median - 1.0).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Predict tests
    // =========================================================================

    mod predict {
        use super::*;

        #[test]
        fn predict_returns_quantile_forecasts() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0, 21.0]);
            let quantiles = sim.predict(&result, &new_forecasts).unwrap();

            assert_eq!(quantiles.n_times(), 2);
            assert_eq!(quantiles.n_quantiles(), 3);
        }

        #[test]
        fn predict_with_timestamps() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            let timestamps = make_timestamps(2);
            let new_forecasts = PointForecasts::new(timestamps.clone(), vec![20.0, 21.0]).unwrap();
            let quantiles = sim.predict(&result, &new_forecasts).unwrap();

            assert!(quantiles.has_timestamps());
            assert_eq!(quantiles.timestamps(), &timestamps);
        }

        #[test]
        fn predict_values_works() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            let quantiles = sim.predict_values(&result, &[20.0, 21.0]).unwrap();

            assert_eq!(quantiles.n_times(), 2);
            assert!(!quantiles.has_timestamps());
        }

        #[test]
        fn quantile_values_are_monotonic() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.0, 12.5, 13.5, 14.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = sim.predict(&result, &new_forecasts).unwrap();

            let row = quantiles.at_time(0).unwrap();
            assert!(row[0] <= row[1] && row[1] <= row[2]);
        }

        #[test]
        fn median_forecast_shifts_by_median_error() {
            let sim = HistoricalSimulator::new(vec![0.5]);

            // Errors: all +1 -> median error = 1
            let forecasts = vec![10.0, 10.0, 10.0, 10.0, 10.0];
            let actuals = vec![11.0, 11.0, 11.0, 11.0, 11.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = sim.predict(&result, &new_forecasts).unwrap();

            let median = quantiles.at_time(0).unwrap()[0];
            // Point forecast 20.0 + median error 1.0 = 21.0
            assert!((median - 21.0).abs() < 1e-10);
        }

        #[test]
        fn unbiased_forecasts_have_symmetric_quantiles() {
            let sim = HistoricalSimulator::new(vec![0.25, 0.5, 0.75]);

            // Symmetric errors around 0
            let forecasts = vec![10.0, 10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 11.0, 12.0]; // Errors: -2, -1, 1, 2

            let result = sim.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = sim.predict(&result, &new_forecasts).unwrap();

            let row = quantiles.at_time(0).unwrap();
            let point = 20.0;

            // Lower quantile distance from point
            let lower_dist = point - row[0];
            // Upper quantile distance from point
            let upper_dist = row[2] - point;

            // For symmetric error distribution, distances should be similar
            // (not exactly equal due to discrete nature of empirical quantiles)
            assert!(
                (lower_dist - upper_dist).abs() < 3.0,
                "Distances should be roughly symmetric"
            );
        }
    }

    // =========================================================================
    // Result tests
    // =========================================================================

    mod historical_sim_result {
        use super::*;

        #[test]
        fn accessors_return_correct_values() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            assert!(!result.errors().is_empty());
            assert_eq!(result.quantile_values().len(), 3);
            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
        }

        #[test]
        fn result_is_clonable() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.5, 12.5, 12.5, 14.5];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            let cloned = result.clone();

            assert_eq!(result.errors(), cloned.errors());
            assert_eq!(result.quantile_values(), cloned.quantile_values());
        }
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    mod edge_cases {
        use super::*;

        #[test]
        fn single_data_point() {
            let sim = HistoricalSimulator::new(vec![0.5]);
            let forecasts = vec![10.0];
            let actuals = vec![11.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();
            assert_eq!(result.errors().len(), 1);
        }

        #[test]
        fn all_zero_errors() {
            let sim = HistoricalSimulator::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.0, 11.0, 12.0];

            let result = sim.fit(&forecasts, &actuals).unwrap();

            // All errors are 0
            for &e in result.errors() {
                assert!((e - 0.0).abs() < 1e-10);
            }

            // All quantile values should be 0
            for &q in result.quantile_values() {
                assert!((q - 0.0).abs() < 1e-10);
            }
        }

        #[test]
        fn negative_errors() {
            let sim = HistoricalSimulator::new(vec![0.5]);
            let forecasts = vec![10.0, 10.0, 10.0];
            let actuals = vec![8.0, 9.0, 7.0]; // All negative errors

            let result = sim.fit(&forecasts, &actuals).unwrap();

            // All errors should be negative
            assert!(result.errors().iter().all(|&e| e < 0.0));
        }
    }
}
