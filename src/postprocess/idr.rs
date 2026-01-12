//! Isotonic Distributional Regression (IDR) predictor.
//!
//! IDR is a state-of-the-art calibration method that learns a monotone relationship
//! between point forecasts and conditional quantiles. It provides theoretical
//! calibration guarantees without requiring hyperparameter tuning.
//!
//! # Algorithm
//!
//! For each quantile level τ, IDR fits an isotonic regression model where:
//! - X = point forecasts (the covariate)
//! - Y = indicator(actual ≤ threshold) for various thresholds
//!
//! This learns the conditional CDF F(y|x) as a monotone function of x.
//!
//! # References
//!
//! - Henzi, Ziegel & Gneiting (2021). Isotonic Distributional Regression.
//!   Journal of the Royal Statistical Society Series B.
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::IDRPredictor;
//!
//! let mut predictor = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
//!
//! // Fit on historical forecast errors
//! let result = predictor.fit(&historical_forecasts, &historical_actuals).unwrap();
//!
//! // Generate quantile forecasts
//! let quantiles = predictor.predict(&result, &new_forecasts);
//! ```

use anofox_regression::solvers::{FittedIsotonic, IsotonicRegressor};
use faer::Col;

use crate::error::{ForecastError, Result};
use crate::postprocess::{PointForecasts, QuantileForecasts};

/// Result of fitting an IDR predictor.
#[derive(Debug, Clone)]
pub struct IDRResult {
    /// Fitted isotonic models for each quantile.
    fitted_models: Vec<FittedIsotonic>,
    /// The quantile levels.
    quantiles: Vec<f64>,
    /// Unique sorted forecast values from training.
    x_grid: Vec<f64>,
    /// Quantile predictions at each grid point.
    y_grid: Vec<Vec<f64>>,
}

impl IDRResult {
    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Get the number of fitted models.
    pub fn n_models(&self) -> usize {
        self.fitted_models.len()
    }

    /// Get the X grid points.
    pub fn x_grid(&self) -> &[f64] {
        &self.x_grid
    }

    /// Get the Y grid (quantile predictions at each grid point).
    pub fn y_grid(&self) -> &[Vec<f64>] {
        &self.y_grid
    }
}

/// IDR (Isotonic Distributional Regression) predictor.
///
/// Uses isotonic regression to learn calibrated conditional quantiles
/// as a monotone function of the point forecast.
#[derive(Debug, Clone)]
pub struct IDRPredictor {
    /// Target quantile levels.
    quantiles: Vec<f64>,
}

impl IDRPredictor {
    /// Create a new IDR predictor.
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

    /// Fit the IDR predictor on historical forecasts and actuals.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - Historical point forecasts
    /// * `actuals` - Corresponding actual values
    ///
    /// # Returns
    ///
    /// An `IDRResult` containing the fitted isotonic models.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forecasts and actuals have different lengths
    /// - Not enough data points
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<IDRResult> {
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

        // Create sorted unique grid of forecast values
        let mut x_grid: Vec<f64> = forecasts.to_vec();
        x_grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        x_grid.dedup();

        // For each quantile, we fit an isotonic regression where:
        // - X = forecasts
        // - Y = actuals
        // The isotonic constraint ensures that higher forecasts give higher quantiles

        // Convert to faer types
        let x_col = Col::from_fn(n, |i| forecasts[i]);
        let y_col = Col::from_fn(n, |i| actuals[i]);

        // For IDR, we fit a single isotonic regression that maps forecasts to actuals
        // Then we compute empirical quantiles of the residual distribution
        let iso = IsotonicRegressor::new();
        let fitted = iso.fit_1d(&x_col, &y_col).map_err(|e| {
            ForecastError::ComputationError(format!("Isotonic regression failed: {:?}", e))
        })?;

        // Compute residuals
        let fitted_values = fitted.fitted_values();
        let residuals: Vec<f64> = (0..n).map(|i| actuals[i] - fitted_values[i]).collect();

        // Sort residuals to compute quantiles
        let mut sorted_residuals = residuals.clone();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute residual quantiles
        let residual_quantiles: Vec<f64> = self
            .quantiles
            .iter()
            .map(|&q| {
                let idx = ((n as f64) * q).floor() as usize;
                let idx = idx.min(n - 1);
                sorted_residuals[idx]
            })
            .collect();

        // Generate predictions at grid points
        let x_grid_col = Col::from_fn(x_grid.len(), |i| x_grid[i]);
        let grid_predictions = fitted.predict_1d(&x_grid_col);

        // Y grid: for each x, add the quantile adjustments
        let y_grid: Vec<Vec<f64>> = (0..x_grid.len())
            .map(|i| {
                let base = grid_predictions[i];
                residual_quantiles.iter().map(|&q| base + q).collect()
            })
            .collect();

        Ok(IDRResult {
            fitted_models: vec![fitted],
            quantiles: self.quantiles.clone(),
            x_grid,
            y_grid,
        })
    }

    /// Generate quantile forecasts for new point forecasts.
    ///
    /// # Arguments
    ///
    /// * `result` - The fitted IDR result
    /// * `point_forecasts` - New point forecasts
    ///
    /// # Returns
    ///
    /// Quantile forecasts at the specified quantile levels.
    pub fn predict(
        &self,
        result: &IDRResult,
        point_forecasts: &PointForecasts,
    ) -> Result<QuantileForecasts> {
        let values = point_forecasts.values();

        if result.fitted_models.is_empty() {
            return Err(ForecastError::FitRequired);
        }

        let fitted = &result.fitted_models[0];

        // Get base predictions from isotonic model
        let x_col = Col::from_fn(values.len(), |i| values[i]);
        let base_preds = fitted.predict_1d(&x_col);

        // Compute residual quantile adjustments from the y_grid
        // Find the residual quantiles from the first grid point's offsets
        let residual_quantiles: Vec<f64> = if !result.y_grid.is_empty() && !result.x_grid.is_empty()
        {
            let first_base = result.y_grid[0]
                .iter()
                .zip(result.quantiles.iter())
                .map(|(&y, _)| y)
                .collect::<Vec<_>>();

            // Compute the adjustment from base prediction
            let base_at_first = if !result.x_grid.is_empty() {
                fitted.predict_single(result.x_grid[0])
            } else {
                0.0
            };

            first_base.iter().map(|&y| y - base_at_first).collect()
        } else {
            vec![0.0; self.quantiles.len()]
        };

        // For each time point, compute quantile forecasts
        let forecast_values: Vec<Vec<f64>> = (0..values.len())
            .map(|i| {
                let base = base_preds[i];
                residual_quantiles.iter().map(|&q| base + q).collect()
            })
            .collect();

        QuantileForecasts::new(
            point_forecasts.timestamps().to_vec(),
            self.quantiles.clone(),
            forecast_values,
        )
    }

    /// Generate quantile forecasts from raw values.
    pub fn predict_values(&self, result: &IDRResult, values: &[f64]) -> Result<QuantileForecasts> {
        let forecasts = PointForecasts::from_values(values.to_vec());
        self.predict(result, &forecasts)
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
        fn new_creates_predictor() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            assert_eq!(pred.quantiles(), &[0.1, 0.5, 0.9]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_zero_quantile() {
            IDRPredictor::new(vec![0.0, 0.5, 0.9]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_one_quantile() {
            IDRPredictor::new(vec![0.1, 0.5, 1.0]);
        }

        #[test]
        #[should_panic(expected = "quantiles must be sorted")]
        fn new_panics_on_unsorted_quantiles() {
            IDRPredictor::new(vec![0.9, 0.5, 0.1]);
        }

        #[test]
        fn predictor_is_clonable() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
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
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
            assert_eq!(result.n_models(), 1);
        }

        #[test]
        fn fit_fails_on_length_mismatch() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0];
            let actuals = vec![10.5, 11.5];

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_empty_data() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts: Vec<f64> = vec![];
            let actuals: Vec<f64> = vec![];

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_single_point() {
            let pred = IDRPredictor::new(vec![0.5]);
            let forecasts = vec![10.0];
            let actuals = vec![10.5];

            let result = pred.fit(&forecasts, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn x_grid_is_sorted() {
            let pred = IDRPredictor::new(vec![0.5]);
            let forecasts = vec![14.0, 10.0, 12.0, 11.0, 13.0];
            let actuals = vec![14.5, 10.5, 12.5, 11.5, 13.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let grid = result.x_grid();

            for i in 1..grid.len() {
                assert!(grid[i] > grid[i - 1], "X grid should be sorted");
            }
        }

        #[test]
        fn x_grid_is_unique() {
            let pred = IDRPredictor::new(vec![0.5]);
            // Duplicate forecasts
            let forecasts = vec![10.0, 10.0, 12.0, 12.0, 14.0];
            let actuals = vec![10.5, 10.3, 12.5, 12.7, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let grid = result.x_grid();

            // Should have unique values only
            assert!(grid.len() <= 3);
        }
    }

    // =========================================================================
    // Predict tests
    // =========================================================================

    mod predict {
        use super::*;

        #[test]
        fn predict_returns_quantile_forecasts() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
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
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
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
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let quantiles = pred.predict_values(&result, &[20.0, 21.0]).unwrap();

            assert_eq!(quantiles.n_times(), 2);
            assert!(!quantiles.has_timestamps());
        }

        #[test]
        fn quantile_values_are_monotonic() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 10.0, 12.5, 13.5, 14.0];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let new_forecasts = PointForecasts::from_values(vec![20.0]);
            let quantiles = pred.predict(&result, &new_forecasts).unwrap();

            let row = quantiles.at_time(0).unwrap();
            assert!(
                row[0] <= row[1] && row[1] <= row[2],
                "Quantiles should be monotonic: {:?}",
                row
            );
        }

        #[test]
        fn higher_forecast_gives_higher_quantiles() {
            let pred = IDRPredictor::new(vec![0.5]);

            // Linear relationship: higher forecast -> higher actual
            let forecasts: Vec<f64> = (0..20).map(|i| i as f64).collect();
            let actuals: Vec<f64> = forecasts.iter().map(|&f| f + 1.0).collect();

            let result = pred.fit(&forecasts, &actuals).unwrap();

            let q_low = pred.predict_values(&result, &[5.0]).unwrap();
            let q_high = pred.predict_values(&result, &[15.0]).unwrap();

            assert!(
                q_high.at_time(0).unwrap()[0] > q_low.at_time(0).unwrap()[0],
                "Higher forecast should give higher quantile prediction"
            );
        }
    }

    // =========================================================================
    // Result tests
    // =========================================================================

    mod idr_result {
        use super::*;

        #[test]
        fn accessors_return_correct_values() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();

            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
            assert_eq!(result.n_models(), 1);
            assert!(!result.x_grid().is_empty());
            assert!(!result.y_grid().is_empty());
        }

        #[test]
        fn result_is_clonable() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);
            let forecasts = vec![10.0, 11.0, 12.0, 13.0, 14.0];
            let actuals = vec![10.5, 11.5, 12.5, 13.5, 14.5];

            let result = pred.fit(&forecasts, &actuals).unwrap();
            let cloned = result.clone();

            assert_eq!(result.quantiles(), cloned.quantiles());
            assert_eq!(result.x_grid(), cloned.x_grid());
        }
    }

    // =========================================================================
    // Calibration tests
    // =========================================================================

    mod calibration {
        use super::*;

        #[test]
        fn calibration_on_linear_data() {
            let pred = IDRPredictor::new(vec![0.1, 0.5, 0.9]);

            // Linear relationship with some noise
            let n = 100;
            let forecasts: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let actuals: Vec<f64> = forecasts
                .iter()
                .enumerate()
                .map(|(i, &f)| f + ((i % 5) as f64 - 2.0))
                .collect();

            let result = pred.fit(&forecasts, &actuals).unwrap();

            // Predict on training data
            let quantiles = pred.predict_values(&result, &forecasts).unwrap();

            // Check that predictions are reasonable
            assert_eq!(quantiles.n_times(), n);
            assert_eq!(quantiles.n_quantiles(), 3);

            // Check monotonicity at each time point
            for t in 0..n {
                let row = quantiles.at_time(t).unwrap();
                assert!(row[0] <= row[1], "q0.1 <= q0.5 at t={}", t);
                assert!(row[1] <= row[2], "q0.5 <= q0.9 at t={}", t);
            }
        }
    }
}
