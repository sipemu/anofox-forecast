//! Ordinary Least Squares (OLS) regression utilities for exogenous variable support.
//!
//! This module provides OLS fitting and prediction capabilities used by forecasting
//! models that support exogenous regressors (ARIMA, MFLES, baseline models).

use crate::error::{ForecastError, Result};
use std::collections::HashMap;

/// OLS regression coefficients and intercept.
#[derive(Debug, Clone)]
pub struct OLSResult {
    /// Regression coefficients (one per regressor).
    pub coefficients: Vec<f64>,
    /// Intercept term.
    pub intercept: f64,
    /// Names of regressors in order.
    pub regressor_names: Vec<String>,
}

impl OLSResult {
    /// Predict values using the fitted OLS model.
    ///
    /// # Arguments
    /// * `regressors` - HashMap of regressor name -> values
    ///
    /// # Returns
    /// Predicted values for each observation.
    pub fn predict(&self, regressors: &HashMap<String, Vec<f64>>) -> Result<Vec<f64>> {
        if regressors.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "No regressors provided for prediction".into(),
            ));
        }

        // Get the length from first regressor
        let first_name = self
            .regressor_names
            .first()
            .ok_or_else(|| ForecastError::InvalidParameter("No regressor names stored".into()))?;

        let first_values = regressors.get(first_name).ok_or_else(|| {
            ForecastError::InvalidParameter(format!(
                "Missing regressor '{}' in prediction data",
                first_name
            ))
        })?;

        let n = first_values.len();

        // Validate all regressors have same length
        for name in &self.regressor_names {
            let values = regressors.get(name).ok_or_else(|| {
                ForecastError::InvalidParameter(format!(
                    "Missing regressor '{}' in prediction data",
                    name
                ))
            })?;
            if values.len() != n {
                return Err(ForecastError::DimensionMismatch {
                    expected: n,
                    got: values.len(),
                });
            }
        }

        // Compute predictions: intercept + sum(coef_i * x_i)
        let mut predictions = vec![self.intercept; n];
        for (i, name) in self.regressor_names.iter().enumerate() {
            let values = &regressors[name];
            for (j, pred) in predictions.iter_mut().enumerate() {
                *pred += self.coefficients[i] * values[j];
            }
        }

        Ok(predictions)
    }

    /// Get the number of regressors.
    pub fn num_regressors(&self) -> usize {
        self.coefficients.len()
    }
}

/// Fit OLS regression: y = intercept + X @ coefficients
///
/// Uses Cholesky decomposition to solve the normal equations.
///
/// # Arguments
/// * `y` - Target values (length n)
/// * `regressors` - HashMap of regressor name -> values (each length n)
///
/// # Returns
/// OLS result with coefficients and intercept.
pub fn ols_fit(y: &[f64], regressors: &HashMap<String, Vec<f64>>) -> Result<OLSResult> {
    let n = y.len();

    if n == 0 {
        return Err(ForecastError::InsufficientData { needed: 1, got: 0 });
    }

    if regressors.is_empty() {
        // No regressors - just return the mean as intercept
        let intercept = y.iter().sum::<f64>() / n as f64;
        return Ok(OLSResult {
            coefficients: vec![],
            intercept,
            regressor_names: vec![],
        });
    }

    // Collect regressor names in deterministic order
    let mut regressor_names: Vec<String> = regressors.keys().cloned().collect();
    regressor_names.sort();

    let k = regressor_names.len();

    // Validate dimensions
    for name in &regressor_names {
        let values = &regressors[name];
        if values.len() != n {
            return Err(ForecastError::DimensionMismatch {
                expected: n,
                got: values.len(),
            });
        }
    }

    // Build design matrix X as column vectors (including intercept)
    // We'll fit: y = β0 + β1*x1 + β2*x2 + ...
    // Design matrix has k+1 columns: [1, x1, x2, ...]
    let num_params = k + 1;

    // Build X'X matrix and X'y vector
    // X'X[i,j] = sum over observations of x_i * x_j
    // X'y[i] = sum over observations of x_i * y

    let mut xtx = vec![vec![0.0; num_params]; num_params];
    let mut xty = vec![0.0; num_params];

    // Collect regressor values as slices for efficient access
    let x_cols: Vec<&[f64]> = regressor_names
        .iter()
        .map(|name| regressors[name].as_slice())
        .collect();

    for obs in 0..n {
        // x_full = [1, x1[obs], x2[obs], ...]
        let y_obs = y[obs];

        // X'X contributions
        // (0,0): 1 * 1
        xtx[0][0] += 1.0;
        // (0,j) and (j,0) for j>0: 1 * x_j
        for j in 0..k {
            let xj = x_cols[j][obs];
            xtx[0][j + 1] += xj;
            xtx[j + 1][0] += xj;
        }
        // (i,j) for i,j > 0: x_i * x_j
        for i in 0..k {
            let xi = x_cols[i][obs];
            for j in 0..k {
                let xj = x_cols[j][obs];
                xtx[i + 1][j + 1] += xi * xj;
            }
        }

        // X'y contributions
        xty[0] += y_obs;
        for i in 0..k {
            xty[i + 1] += x_cols[i][obs] * y_obs;
        }
    }

    // Add small regularization to diagonal for numerical stability
    for i in 0..num_params {
        xtx[i][i] += 1e-8;
    }

    // Solve X'X @ beta = X'y using Cholesky decomposition
    let beta = solve_symmetric(&xtx, &xty).ok_or_else(|| {
        ForecastError::InvalidParameter(
            "OLS regression failed: matrix not positive definite".into(),
        )
    })?;

    Ok(OLSResult {
        intercept: beta[0],
        coefficients: beta[1..].to_vec(),
        regressor_names,
    })
}

/// Solve symmetric positive definite system using Cholesky decomposition.
///
/// Solves A @ x = b where A is symmetric positive definite.
fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return None;
    }

    // Cholesky decomposition A = L @ L'
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }

            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward substitution: L @ y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }

    // Backward substitution: L' @ x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }

    Some(x)
}

/// Compute residuals after removing OLS fit.
///
/// # Arguments
/// * `y` - Original values
/// * `ols_result` - Fitted OLS model
/// * `regressors` - Regressor values used for fitting
///
/// # Returns
/// Residuals (y - y_hat)
pub fn ols_residuals(
    y: &[f64],
    ols_result: &OLSResult,
    regressors: &HashMap<String, Vec<f64>>,
) -> Result<Vec<f64>> {
    let predictions = ols_result.predict(regressors)?;

    if predictions.len() != y.len() {
        return Err(ForecastError::DimensionMismatch {
            expected: y.len(),
            got: predictions.len(),
        });
    }

    Ok(y.iter()
        .zip(predictions.iter())
        .map(|(yi, pi)| yi - pi)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ols_fit_simple_linear() {
        // y = 2 + 3*x
        let y = vec![5.0, 8.0, 11.0, 14.0, 17.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut regressors = HashMap::new();
        regressors.insert("x".to_string(), x);

        let result = ols_fit(&y, &regressors).unwrap();

        assert_relative_eq!(result.intercept, 2.0, epsilon = 1e-6);
        assert_eq!(result.coefficients.len(), 1);
        assert_relative_eq!(result.coefficients[0], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn ols_fit_multiple_regressors() {
        // y = 1 + 2*x1 + 3*x2
        // Use non-collinear regressors
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = vec![0.5, 2.5, 1.0, 3.0, 1.5, 3.5, 2.0, 4.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| 1.0 + 2.0 * a + 3.0 * b)
            .collect();

        let mut regressors = HashMap::new();
        regressors.insert("x1".to_string(), x1);
        regressors.insert("x2".to_string(), x2);

        let result = ols_fit(&y, &regressors).unwrap();

        assert_relative_eq!(result.intercept, 1.0, epsilon = 1e-4);
        assert_eq!(result.coefficients.len(), 2);

        // Find coefficient for x1 and x2 (sorted order)
        let x1_idx = result
            .regressor_names
            .iter()
            .position(|n| n == "x1")
            .unwrap();
        let x2_idx = result
            .regressor_names
            .iter()
            .position(|n| n == "x2")
            .unwrap();

        assert_relative_eq!(result.coefficients[x1_idx], 2.0, epsilon = 1e-4);
        assert_relative_eq!(result.coefficients[x2_idx], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn ols_fit_no_regressors() {
        // Should return mean as intercept
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let regressors = HashMap::new();

        let result = ols_fit(&y, &regressors).unwrap();

        assert_relative_eq!(result.intercept, 6.0, epsilon = 1e-10);
        assert!(result.coefficients.is_empty());
    }

    #[test]
    fn ols_predict() {
        // Fit y = 2 + 3*x
        let y = vec![5.0, 8.0, 11.0, 14.0, 17.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut regressors = HashMap::new();
        regressors.insert("x".to_string(), x);

        let result = ols_fit(&y, &regressors).unwrap();

        // Predict for new values
        let mut future_regressors = HashMap::new();
        future_regressors.insert("x".to_string(), vec![6.0, 7.0, 8.0]);

        let predictions = result.predict(&future_regressors).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_relative_eq!(predictions[0], 20.0, epsilon = 1e-6); // 2 + 3*6
        assert_relative_eq!(predictions[1], 23.0, epsilon = 1e-6); // 2 + 3*7
        assert_relative_eq!(predictions[2], 26.0, epsilon = 1e-6); // 2 + 3*8
    }

    #[test]
    fn ols_residuals_calculation() {
        // y = 2 + 3*x + noise
        let y = vec![5.1, 7.9, 11.2, 13.8, 17.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut regressors = HashMap::new();
        regressors.insert("x".to_string(), x.clone());

        let result = ols_fit(&y, &regressors).unwrap();
        let residuals = ols_residuals(&y, &result, &regressors).unwrap();

        assert_eq!(residuals.len(), 5);

        // Residuals should sum to approximately 0 (with floating point tolerance)
        let sum: f64 = residuals.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn ols_fit_dimension_mismatch() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 2.0]; // Wrong length

        let mut regressors = HashMap::new();
        regressors.insert("x".to_string(), x);

        assert!(ols_fit(&y, &regressors).is_err());
    }

    #[test]
    fn ols_predict_missing_regressor() {
        let y = vec![5.0, 8.0, 11.0];
        let mut regressors = HashMap::new();
        regressors.insert("x".to_string(), vec![1.0, 2.0, 3.0]);

        let result = ols_fit(&y, &regressors).unwrap();

        // Predict with missing regressor
        let mut future_regressors = HashMap::new();
        future_regressors.insert("wrong_name".to_string(), vec![4.0, 5.0]);

        assert!(result.predict(&future_regressors).is_err());
    }

    #[test]
    fn ols_with_noise() {
        // Test with noisy data
        let n = 100;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.5 + 1.7 * xi + (i as f64 * 0.13).sin() * 0.1)
            .collect();

        let mut regressors = HashMap::new();
        regressors.insert("x".to_string(), x);

        let result = ols_fit(&y, &regressors).unwrap();

        // Should be close to true values despite noise
        assert_relative_eq!(result.intercept, 2.5, epsilon = 0.1);
        assert_relative_eq!(result.coefficients[0], 1.7, epsilon = 0.1);
    }
}
