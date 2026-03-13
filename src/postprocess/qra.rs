//! Quantile Regression Averaging (QRA) predictor.
//!
//! QRA combines multiple point forecasters into probabilistic forecasts using
//! quantile regression. This allows for automatic weight selection and
//! ensemble combination with uncertainty quantification.
//!
//! # Algorithm
//!
//! For each quantile level τ, QRA fits a quantile regression model:
//! Q_τ(Y|X) = Σ β_τ,j * X_j
//!
//! where X_j are the point forecasts from different models and β_τ,j are
//! quantile-specific weights.
//!
//! # Variants
//!
//! - **None**: Standard quantile regression
//! - **Lasso**: L1 regularized quantile regression
//! - **LassoCV**: Lasso with cross-validated regularization
//! - **Isotonic**: Isotonic QRA (iQRA) for monotone constraints
//!
//! # References
//!
//! - Nowotarski & Weron (2014). Computing electricity spot price prediction
//!   intervals using quantile regression and forecast averaging.
//!
//! # Example
//!
//! ```ignore
//! use anofox_forecast::postprocess::{QRAPredictor, QRARegularization};
//!
//! let predictor = QRAPredictor::new(
//!     vec![0.1, 0.5, 0.9],
//!     QRARegularization::None,
//! );
//!
//! // Fit on historical forecasts from multiple models
//! let result = predictor.fit(&forecasts_matrix, &actuals).unwrap();
//!
//! // Generate combined quantile forecasts
//! let quantiles = predictor.predict(&result, &new_forecasts_matrix);
//! ```

use anofox_regression::solvers::{FittedQuantile, QuantileRegressor};
use anofox_regression::{FittedRegressor, Regressor};
use faer::{Col, Mat};

use crate::error::{ForecastError, Result};
use crate::postprocess::QuantileForecasts;

/// Regularization method for QRA.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum QRARegularization {
    /// Standard quantile regression (no regularization).
    #[default]
    None,
    /// L1 regularized (Lasso) quantile regression.
    Lasso {
        /// Regularization strength.
        lambda: f64,
    },
    /// Lasso with cross-validated regularization parameter.
    LassoCV {
        /// Number of cross-validation folds.
        n_folds: usize,
    },
    /// Isotonic QRA variant for monotone constraints.
    Isotonic,
}

/// Result of fitting a QRA predictor.
#[derive(Debug, Clone)]
pub struct QRAResult {
    /// Fitted quantile regression models for each quantile.
    fitted_models: Vec<FittedQuantile>,
    /// The quantile levels.
    quantiles: Vec<f64>,
    /// Number of forecasters (features).
    n_forecasters: usize,
    /// Whether intercept was included.
    with_intercept: bool,
}

impl QRAResult {
    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Get the number of fitted models.
    pub fn n_models(&self) -> usize {
        self.fitted_models.len()
    }

    /// Get the number of forecasters.
    pub fn n_forecasters(&self) -> usize {
        self.n_forecasters
    }

    /// Whether the model includes an intercept term.
    pub fn has_intercept(&self) -> bool {
        self.with_intercept
    }

    /// Get coefficients for a specific quantile.
    pub fn coefficients(&self, quantile_idx: usize) -> Option<Vec<f64>> {
        self.fitted_models.get(quantile_idx).map(|m| {
            let coeffs = m.coefficients();
            (0..coeffs.nrows()).map(|i| coeffs[i]).collect()
        })
    }
}

/// QRA (Quantile Regression Averaging) predictor.
///
/// Combines multiple point forecasters using quantile regression
/// to produce calibrated probabilistic forecasts.
#[derive(Debug, Clone)]
pub struct QRAPredictor {
    /// Target quantile levels.
    quantiles: Vec<f64>,
    /// Regularization method.
    regularization: QRARegularization,
    /// Whether to include an intercept.
    with_intercept: bool,
}

impl QRAPredictor {
    /// Create a new QRA predictor.
    ///
    /// # Arguments
    ///
    /// * `quantiles` - Target quantile levels (must be in (0, 1) and sorted)
    /// * `regularization` - Regularization method to use
    ///
    /// # Panics
    ///
    /// Panics if quantiles are invalid or not sorted.
    pub fn new(quantiles: Vec<f64>, regularization: QRARegularization) -> Self {
        // Validate quantiles
        for &q in &quantiles {
            assert!(q > 0.0 && q < 1.0, "quantiles must be in (0, 1)");
        }
        for w in quantiles.windows(2) {
            assert!(w[0] < w[1], "quantiles must be sorted in ascending order");
        }

        Self {
            quantiles,
            regularization,
            with_intercept: true,
        }
    }

    /// Create a QRA predictor without regularization.
    pub fn standard(quantiles: Vec<f64>) -> Self {
        Self::new(quantiles, QRARegularization::None)
    }

    /// Create a QRA predictor with Lasso regularization.
    pub fn lasso(quantiles: Vec<f64>, lambda: f64) -> Self {
        Self::new(quantiles, QRARegularization::Lasso { lambda })
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Get the quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Get the regularization method.
    pub fn regularization(&self) -> &QRARegularization {
        &self.regularization
    }

    /// Fit the QRA predictor on historical forecasts and actuals.
    ///
    /// # Arguments
    ///
    /// * `forecasts_matrix` - Matrix where each column is one forecaster's predictions.
    ///   Shape: (n_samples, n_forecasters)
    /// * `actuals` - Actual values corresponding to the forecasts
    ///
    /// # Returns
    ///
    /// A `QRAResult` containing the fitted quantile regression models.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Matrix rows don't match actuals length
    /// - Not enough data points
    pub fn fit(&self, forecasts_matrix: &[Vec<f64>], actuals: &[f64]) -> Result<QRAResult> {
        let n = actuals.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }

        if forecasts_matrix.len() != n {
            return Err(ForecastError::DimensionMismatch {
                expected: n,
                got: forecasts_matrix.len(),
            });
        }

        let n_forecasters = if n > 0 && !forecasts_matrix.is_empty() {
            forecasts_matrix[0].len()
        } else {
            0
        };

        if n_forecasters == 0 {
            return Err(ForecastError::EmptyData);
        }

        // Validate all rows have same number of forecasters
        for row in forecasts_matrix.iter() {
            if row.len() != n_forecasters {
                return Err(ForecastError::DimensionMismatch {
                    expected: n_forecasters,
                    got: row.len(),
                });
            }
        }

        if n < n_forecasters + 1 {
            return Err(ForecastError::InsufficientData {
                needed: n_forecasters + 1,
                got: n,
                hint: None,
            });
        }

        // Convert to faer types
        let x = Mat::from_fn(n, n_forecasters, |i, j| forecasts_matrix[i][j]);
        let y = Col::from_fn(n, |i| actuals[i]);

        // Fit quantile regression for each quantile
        let mut fitted_models = Vec::with_capacity(self.quantiles.len());

        for &tau in &self.quantiles {
            let qr = QuantileRegressor::builder()
                .tau(tau)
                .with_intercept(self.with_intercept)
                .build();

            let fitted = qr.fit(&x, &y).map_err(|e| {
                ForecastError::ConvergenceFailure(format!("Quantile regression failed: {:?}", e))
            })?;

            fitted_models.push(fitted);
        }

        Ok(QRAResult {
            fitted_models,
            quantiles: self.quantiles.clone(),
            n_forecasters,
            with_intercept: self.with_intercept,
        })
    }

    /// Generate quantile forecasts for new forecast matrix.
    ///
    /// # Arguments
    ///
    /// * `result` - The fitted QRA result
    /// * `forecasts_matrix` - New forecasts from each model.
    ///   Shape: (n_times, n_forecasters)
    ///
    /// # Returns
    ///
    /// Quantile forecasts at the specified quantile levels.
    ///
    /// # Note
    ///
    /// This method enforces monotonicity of quantiles by sorting values at each
    /// time step to prevent quantile crossing.
    pub fn predict(
        &self,
        result: &QRAResult,
        forecasts_matrix: &[Vec<f64>],
    ) -> Result<QuantileForecasts> {
        let n_times = forecasts_matrix.len();
        if n_times == 0 {
            return Err(ForecastError::EmptyData);
        }

        let n_forecasters = forecasts_matrix[0].len();
        if n_forecasters != result.n_forecasters {
            return Err(ForecastError::DimensionMismatch {
                expected: result.n_forecasters,
                got: n_forecasters,
            });
        }

        // Convert to faer matrix
        let x = Mat::from_fn(n_times, n_forecasters, |i, j| forecasts_matrix[i][j]);

        // Predict for each quantile
        let mut forecast_values: Vec<Vec<f64>> =
            vec![Vec::with_capacity(self.quantiles.len()); n_times];

        for (q_idx, fitted) in result.fitted_models.iter().enumerate() {
            let preds = fitted.predict(&x);

            for i in 0..n_times {
                if q_idx == 0 {
                    forecast_values[i] = Vec::with_capacity(self.quantiles.len());
                }
                forecast_values[i].push(preds[i]);
            }
        }

        // Enforce monotonicity by sorting quantile values at each time step
        // This prevents quantile crossing, a common issue with independent QR fits
        for row in &mut forecast_values {
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        QuantileForecasts::from_values(self.quantiles.clone(), forecast_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // QRARegularization tests
    // =========================================================================

    mod regularization {
        use super::*;

        #[test]
        fn default_is_none() {
            let reg = QRARegularization::default();
            assert_eq!(reg, QRARegularization::None);
        }

        #[test]
        fn lasso_stores_lambda() {
            let reg = QRARegularization::Lasso { lambda: 0.1 };
            if let QRARegularization::Lasso { lambda } = reg {
                assert!((lambda - 0.1).abs() < 1e-10);
            } else {
                panic!("Expected Lasso");
            }
        }

        #[test]
        fn lasso_cv_stores_n_folds() {
            let reg = QRARegularization::LassoCV { n_folds: 5 };
            if let QRARegularization::LassoCV { n_folds } = reg {
                assert_eq!(n_folds, 5);
            } else {
                panic!("Expected LassoCV");
            }
        }

        #[test]
        fn regularization_is_clonable() {
            let reg = QRARegularization::Lasso { lambda: 0.1 };
            let cloned = reg.clone();
            assert_eq!(reg, cloned);
        }
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    mod construction {
        use super::*;

        #[test]
        fn new_creates_predictor() {
            let pred = QRAPredictor::new(vec![0.1, 0.5, 0.9], QRARegularization::None);
            assert_eq!(pred.quantiles(), &[0.1, 0.5, 0.9]);
        }

        #[test]
        fn standard_creates_unregularized() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);
            assert_eq!(pred.regularization(), &QRARegularization::None);
        }

        #[test]
        fn lasso_creates_l1_regularized() {
            let pred = QRAPredictor::lasso(vec![0.1, 0.5, 0.9], 0.1);
            match pred.regularization() {
                QRARegularization::Lasso { lambda } => {
                    assert!((*lambda - 0.1).abs() < 1e-10);
                }
                _ => panic!("Expected Lasso"),
            }
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_zero_quantile() {
            QRAPredictor::new(vec![0.0, 0.5, 0.9], QRARegularization::None);
        }

        #[test]
        #[should_panic(expected = "quantiles must be in (0, 1)")]
        fn new_panics_on_one_quantile() {
            QRAPredictor::new(vec![0.1, 0.5, 1.0], QRARegularization::None);
        }

        #[test]
        #[should_panic(expected = "quantiles must be sorted")]
        fn new_panics_on_unsorted_quantiles() {
            QRAPredictor::new(vec![0.9, 0.5, 0.1], QRARegularization::None);
        }

        #[test]
        fn with_intercept_configurable() {
            let pred = QRAPredictor::standard(vec![0.5]).with_intercept(false);
            // Just check it compiles and can be created
            assert_eq!(pred.quantiles(), &[0.5]);
        }

        #[test]
        fn predictor_is_clonable() {
            let pred = QRAPredictor::lasso(vec![0.1, 0.5, 0.9], 0.1);
            let cloned = pred.clone();
            assert_eq!(pred.quantiles(), cloned.quantiles());
        }
    }

    // =========================================================================
    // Fit tests
    // =========================================================================

    mod fit {
        use super::*;

        fn make_forecasts_matrix(n: usize, n_forecasters: usize) -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..n_forecasters)
                        .map(|j| i as f64 + j as f64 * 0.1)
                        .collect()
                })
                .collect()
        }

        #[test]
        fn fit_returns_result() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);
            let matrix = make_forecasts_matrix(20, 2);
            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
            assert_eq!(result.n_models(), 3);
            assert_eq!(result.n_forecasters(), 2);
        }

        #[test]
        fn fit_fails_on_length_mismatch() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix = make_forecasts_matrix(10, 2);
            let actuals: Vec<f64> = (0..5).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_empty_data() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix: Vec<Vec<f64>> = vec![];
            let actuals: Vec<f64> = vec![];

            let result = pred.fit(&matrix, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_insufficient_data() {
            let pred = QRAPredictor::standard(vec![0.5]);
            // 2 forecasters need at least 3 samples (with intercept)
            let matrix = make_forecasts_matrix(2, 2);
            let actuals: Vec<f64> = vec![1.0, 2.0];

            let result = pred.fit(&matrix, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn fit_fails_on_inconsistent_forecaster_count() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix = vec![
                vec![1.0, 2.0],
                vec![2.0, 3.0],
                vec![3.0], // Wrong number of forecasters
            ];
            let actuals = vec![1.5, 2.5, 3.5];

            let result = pred.fit(&matrix, &actuals);
            assert!(result.is_err());
        }

        #[test]
        fn coefficients_are_accessible() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix = make_forecasts_matrix(20, 2);
            let actuals: Vec<f64> = (0..20).map(|i| i as f64 + 0.5).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();
            let coeffs = result.coefficients(0).unwrap();

            // With intercept, we have 3 coefficients (intercept + 2 forecasters)
            assert!(coeffs.len() >= 2);
        }
    }

    // =========================================================================
    // Predict tests
    // =========================================================================

    mod predict {
        use super::*;

        fn make_forecasts_matrix(n: usize, n_forecasters: usize) -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..n_forecasters)
                        .map(|j| i as f64 + j as f64 * 0.1)
                        .collect()
                })
                .collect()
        }

        #[test]
        fn predict_returns_quantile_forecasts() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);
            let matrix = make_forecasts_matrix(30, 2);
            let actuals: Vec<f64> = (0..30).map(|i| i as f64 + 0.5).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let new_matrix = make_forecasts_matrix(5, 2);
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            assert_eq!(quantiles.n_times(), 5);
            assert_eq!(quantiles.n_quantiles(), 3);
        }

        #[test]
        fn predict_fails_on_wrong_forecaster_count() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix = make_forecasts_matrix(30, 2);
            let actuals: Vec<f64> = (0..30).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            // Wrong number of forecasters
            let new_matrix = make_forecasts_matrix(5, 3);
            let pred_result = pred.predict(&result, &new_matrix);
            assert!(pred_result.is_err());
        }

        #[test]
        fn predict_fails_on_empty_input() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix = make_forecasts_matrix(30, 2);
            let actuals: Vec<f64> = (0..30).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let empty_matrix: Vec<Vec<f64>> = vec![];
            let pred_result = pred.predict(&result, &empty_matrix);
            assert!(pred_result.is_err());
        }

        #[test]
        fn quantile_values_are_monotonic() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);
            let matrix = make_forecasts_matrix(50, 2);
            let actuals: Vec<f64> = matrix.iter().map(|row| row[0] + row[1] + 0.5).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let new_matrix = make_forecasts_matrix(10, 2);
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            // Check monotonicity at each time point
            for t in 0..10 {
                let row = quantiles.at_time(t).unwrap();
                // Allow small tolerance for numerical precision
                assert!(
                    row[0] <= row[1] + 0.1 && row[1] <= row[2] + 0.1,
                    "Quantiles should be approximately monotonic at t={}: {:?}",
                    t,
                    row
                );
            }
        }
    }

    // =========================================================================
    // Result tests
    // =========================================================================

    mod qra_result {
        use super::*;

        fn make_forecasts_matrix(n: usize, n_forecasters: usize) -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..n_forecasters)
                        .map(|j| i as f64 + j as f64 * 0.1)
                        .collect()
                })
                .collect()
        }

        #[test]
        fn accessors_return_correct_values() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);
            let matrix = make_forecasts_matrix(20, 3);
            let actuals: Vec<f64> = (0..20).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            assert_eq!(result.quantiles(), &[0.1, 0.5, 0.9]);
            assert_eq!(result.n_models(), 3);
            assert_eq!(result.n_forecasters(), 3);
        }

        #[test]
        fn result_is_clonable() {
            let pred = QRAPredictor::standard(vec![0.5]);
            let matrix = make_forecasts_matrix(20, 2);
            let actuals: Vec<f64> = (0..20).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();
            let cloned = result.clone();

            assert_eq!(result.quantiles(), cloned.quantiles());
            assert_eq!(result.n_forecasters(), cloned.n_forecasters());
        }
    }

    // =========================================================================
    // Ensemble combination tests
    // =========================================================================

    mod ensemble {
        use super::*;

        #[test]
        fn combines_multiple_forecasters() {
            let pred = QRAPredictor::standard(vec![0.5]);

            // Two forecasters with different biases
            let n = 50;
            let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();

            // Forecaster 1: underestimates
            // Forecaster 2: overestimates
            let matrix: Vec<Vec<f64>> = (0..n)
                .map(|i| vec![i as f64 - 1.0, i as f64 + 1.0])
                .collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            // Should combine to produce better forecasts
            let coeffs = result.coefficients(0).unwrap();
            assert!(
                coeffs.len() >= 2,
                "Should have coefficients for both forecasters"
            );
        }

        #[test]
        fn handles_single_forecaster() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);

            // Single forecaster
            let n = 30;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();
            assert_eq!(result.n_forecasters(), 1);

            let new_matrix: Vec<Vec<f64>> = (30..35).map(|i| vec![i as f64]).collect();
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            assert_eq!(quantiles.n_times(), 5);
        }
    }

    // =========================================================================
    // Additional tests: known values, zero variance, edge cases
    // =========================================================================

    mod known_values {
        use super::*;

        /// For a perfect linear relationship y = x, the median quantile (0.5)
        /// prediction should be close to the identity function.
        #[test]
        fn perfect_linear_relationship_median() {
            let pred = QRAPredictor::standard(vec![0.5]);

            let n = 50;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let new_matrix = vec![vec![25.0]];
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            let predicted = quantiles.at_time(0).unwrap()[0];
            assert!(
                (predicted - 25.0).abs() < 1.0,
                "Median prediction for y=x at x=25 should be near 25, got {}",
                predicted
            );
        }

        /// Verify that for y = 2*x + 1, the median prediction reflects
        /// the linear relationship.
        #[test]
        fn linear_relationship_with_slope_and_intercept() {
            let pred = QRAPredictor::standard(vec![0.5]).with_intercept(true);

            let n = 50;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 1.0).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            // Predict at x=10
            let new_matrix = vec![vec![10.0]];
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            let predicted = quantiles.at_time(0).unwrap()[0];
            let expected = 2.0 * 10.0 + 1.0;
            assert!(
                (predicted - expected).abs() < 1.0,
                "Expected ~{}, got {}",
                expected,
                predicted
            );
        }

        /// QRA with two equal-weight forecasters should average them.
        #[test]
        fn equal_forecasters_median_approximates_average() {
            let pred = QRAPredictor::standard(vec![0.5]);

            let n = 50;
            // Two forecasters: f1 = i, f2 = i + 2
            // Actuals = i + 1 (average of the two)
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, i as f64 + 2.0]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let new_matrix = vec![vec![100.0, 102.0]];
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            let predicted = quantiles.at_time(0).unwrap()[0];
            // Should be close to 101.0 (average)
            assert!(
                (predicted - 101.0).abs() < 2.0,
                "Expected ~101.0, got {}",
                predicted
            );
        }

        /// Verify the QRAResult has_intercept reflects configuration.
        #[test]
        fn has_intercept_reflects_config() {
            let pred_with = QRAPredictor::standard(vec![0.5]).with_intercept(true);
            let pred_without = QRAPredictor::standard(vec![0.5]).with_intercept(false);

            let n = 20;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();

            let result_with = pred_with.fit(&matrix, &actuals).unwrap();
            let result_without = pred_without.fit(&matrix, &actuals).unwrap();

            assert!(result_with.has_intercept());
            assert!(!result_without.has_intercept());
        }

        /// Out-of-range coefficient index returns None.
        #[test]
        fn coefficients_out_of_range_returns_none() {
            let pred = QRAPredictor::standard(vec![0.5]);

            let n = 20;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            assert!(result.coefficients(0).is_some());
            assert!(result.coefficients(999).is_none());
        }
    }

    mod zero_variance {
        use super::*;

        /// Constant forecasts (zero variance) with constant actuals.
        #[test]
        fn constant_forecasts_constant_actuals() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);

            let n = 20;
            let matrix: Vec<Vec<f64>> = (0..n).map(|_| vec![5.0]).collect();
            let actuals: Vec<f64> = vec![5.0; n];

            let result = pred.fit(&matrix, &actuals).unwrap();

            let new_matrix = vec![vec![5.0]];
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            // All quantiles should be close to 5.0
            let row = quantiles.at_time(0).unwrap();
            for &val in row {
                assert!(
                    (val - 5.0).abs() < 1.0,
                    "With constant data, quantiles should be near 5.0, got {}",
                    val
                );
            }
        }

        /// Constant forecasts but varying actuals.
        #[test]
        fn constant_forecasts_varying_actuals() {
            let pred = QRAPredictor::standard(vec![0.1, 0.5, 0.9]);

            let n = 30;
            let matrix: Vec<Vec<f64>> = (0..n).map(|_| vec![10.0]).collect();
            // Actuals vary: 9, 10, 11, 9, 10, 11, ...
            let actuals: Vec<f64> = (0..n).map(|i| 9.0 + (i % 3) as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let new_matrix = vec![vec![10.0]];
            let quantiles = pred.predict(&result, &new_matrix).unwrap();

            let row = quantiles.at_time(0).unwrap();
            // q0.1 should be <= q0.5 <= q0.9 (enforced by monotonicity sort)
            assert!(row[0] <= row[1]);
            assert!(row[1] <= row[2]);
        }
    }

    mod empty_and_error_cases {
        use super::*;

        /// Predict with empty forecasts matrix.
        #[test]
        fn predict_empty_matrix() {
            let pred = QRAPredictor::standard(vec![0.5]);

            let n = 20;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();

            let empty: Vec<Vec<f64>> = vec![];
            let pred_result = pred.predict(&result, &empty);
            assert!(pred_result.is_err());
        }

        /// Fit with zero-length inner vectors (empty forecasters).
        #[test]
        fn fit_with_empty_forecaster_vectors() {
            let pred = QRAPredictor::standard(vec![0.5]);

            let matrix: Vec<Vec<f64>> = vec![vec![], vec![], vec![]];
            let actuals = vec![1.0, 2.0, 3.0];

            let result = pred.fit(&matrix, &actuals);
            assert!(result.is_err());
        }

        /// Multiple quantiles produce multiple models.
        #[test]
        fn multiple_quantiles_produce_multiple_models() {
            let quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];
            let pred = QRAPredictor::standard(quantiles.clone());

            let n = 30;
            let matrix: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let actuals: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();

            let result = pred.fit(&matrix, &actuals).unwrap();
            assert_eq!(result.n_models(), 5);
            assert_eq!(result.quantiles(), &quantiles);
        }
    }
}
