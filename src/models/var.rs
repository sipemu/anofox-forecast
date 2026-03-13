//! Vector Autoregression (VAR) model for multivariate time series forecasting.
//!
//! A VAR(p) model jointly models k time series where each variable is a linear
//! function of its own lags and the lags of all other variables:
//!
//! y(t) = c + A_1 * y(t-1) + A_2 * y(t-2) + ... + A_p * y(t-p) + e(t)
//!
//! where y(t) is a k-vector, c is a k-vector of intercepts, A_j are k x k
//! coefficient matrices, and e(t) is a k-vector of white noise innovations.
//!
//! Estimation is performed equation-by-equation via OLS.

use crate::error::{ForecastError, Result};
use std::collections::HashMap;

use crate::utils::ols::ols_fit;

/// Vector Autoregression model of order p.
///
/// Models k time series jointly where each variable at time t is a linear
/// function of the p most recent lags of all k variables, plus an intercept.
///
/// # Example
/// ```
/// use anofox_forecast::models::var::VAR;
///
/// let mut model = VAR::new(2); // VAR(2) model
///
/// // Two correlated time series, each with 50 observations
/// let y1: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
/// let y2: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).cos()).collect();
///
/// model.fit(&[y1, y2]).unwrap();
/// let forecasts = model.predict(5).unwrap();
/// assert_eq!(forecasts.len(), 2);      // one per variable
/// assert_eq!(forecasts[0].len(), 5);   // 5-step horizon
/// ```
#[derive(Debug, Clone)]
pub struct VAR {
    /// Number of lags (p).
    order: usize,
    /// Number of variables (k). Set after fitting.
    n_vars: usize,
    /// Coefficient matrices: coefficients[i][j][lag] is the coefficient in
    /// equation i for variable j at lag (lag+1).
    /// Dimensions: k x k x p.
    coefficients: Option<Vec<Vec<Vec<f64>>>>,
    /// Intercept for each equation (length k).
    intercepts: Option<Vec<f64>>,
    /// Residuals for each variable (k vectors, each of length n - p).
    residuals: Option<Vec<Vec<f64>>>,
    /// Fitted values for each variable (k vectors, each of length n - p).
    fitted_values: Option<Vec<Vec<f64>>>,
    /// Original training data stored for forecasting and Granger causality tests.
    /// Each inner vector is one variable's full time series.
    training_data: Option<Vec<Vec<f64>>>,
    /// Number of effective observations (n - p) used in estimation.
    n_effective: usize,
}

impl VAR {
    /// Create a new VAR model with the given lag order.
    ///
    /// # Arguments
    /// * `order` - Number of lags (p). Must be at least 1.
    pub fn new(order: usize) -> Self {
        Self {
            order,
            n_vars: 0,
            coefficients: None,
            intercepts: None,
            residuals: None,
            fitted_values: None,
            training_data: None,
            n_effective: 0,
        }
    }

    /// Fit the VAR model to multivariate time series data.
    ///
    /// Each element of `data` is one variable's full time series. All variables
    /// must have the same length.
    ///
    /// Estimation proceeds equation-by-equation via OLS. For equation i the
    /// dependent variable is y_i(t) and the regressors are
    /// \[y_1(t-1), ..., y_k(t-1), y_1(t-2), ..., y_k(t-p)\].
    ///
    /// # Arguments
    /// * `data` - Slice of k vectors, each containing n observations.
    ///
    /// # Errors
    /// * `EmptyData` if `data` is empty or any series is empty.
    /// * `InvalidParameter` if `order` is 0.
    /// * `InsufficientData` if n <= p (need at least p+1 observations).
    /// * `DimensionMismatch` if series have different lengths.
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<()> {
        let k = data.len();
        if k == 0 {
            return Err(ForecastError::EmptyData);
        }
        if self.order == 0 {
            return Err(ForecastError::InvalidParameter(
                "VAR order must be at least 1".into(),
            ));
        }

        let n = data[0].len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }

        for (i, series) in data.iter().enumerate() {
            if series.len() != n {
                return Err(ForecastError::DimensionMismatch {
                    expected: n,
                    got: series.len(),
                });
            }
            if series.iter().any(|v| v.is_nan() || v.is_infinite()) {
                return Err(ForecastError::InvalidParameter(format!(
                    "Variable {} contains NaN or Inf values",
                    i
                )));
            }
        }

        let p = self.order;
        if n <= p {
            return Err(ForecastError::InsufficientData {
                needed: p + 1,
                got: n,
                hint: Some(format!(
                    "VAR({}) requires at least {} observations",
                    p,
                    p + 1
                )),
            });
        }

        let n_eff = n - p;
        self.n_vars = k;
        self.n_effective = n_eff;

        // Build the regressor map shared across all equations.
        let regressor_map = build_regressor_map(data, k, p, n);

        // Fit each equation via OLS.
        let mut coefficients = vec![vec![vec![0.0; p]; k]; k];
        let mut intercepts = vec![0.0; k];
        let mut residuals = vec![vec![0.0; n_eff]; k];
        let mut fitted_vals = vec![vec![0.0; n_eff]; k];

        for eq in 0..k {
            let y: Vec<f64> = data[eq][p..n].to_vec();
            let ols_result = ols_fit(&y, &regressor_map)?;

            intercepts[eq] = ols_result.intercept;

            // Extract coefficients into the 3D structure.
            for lag in 1..=p {
                for var in 0..k {
                    let name = regressor_name(var, lag);
                    if let Some(idx) = ols_result.regressor_names.iter().position(|n| n == &name) {
                        coefficients[eq][var][lag - 1] = ols_result.coefficients[idx];
                    }
                }
            }

            // Compute fitted values and residuals.
            let predictions = ols_result.predict(&regressor_map)?;
            for t in 0..n_eff {
                fitted_vals[eq][t] = predictions[t];
                residuals[eq][t] = y[t] - predictions[t];
            }
        }

        self.coefficients = Some(coefficients);
        self.intercepts = Some(intercepts);
        self.residuals = Some(residuals);
        self.fitted_values = Some(fitted_vals);
        self.training_data = Some(data.to_vec());

        Ok(())
    }

    /// Forecast h steps ahead for all variables.
    ///
    /// Uses the fitted model to iteratively produce multi-step forecasts.
    /// Each forecast step uses the previously forecasted values as inputs
    /// for subsequent steps.
    ///
    /// # Arguments
    /// * `horizon` - Number of steps to forecast.
    ///
    /// # Returns
    /// A vector of k vectors, each containing h forecast values.
    ///
    /// # Errors
    /// * `FitRequired` if the model has not been fitted.
    /// * `InvalidParameter` if horizon is 0.
    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        let intercepts = self.intercepts.as_ref().ok_or(ForecastError::FitRequired)?;
        let training_data = self
            .training_data
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Err(ForecastError::InvalidParameter(
                "Forecast horizon must be at least 1".into(),
            ));
        }

        let k = self.n_vars;
        let p = self.order;
        let n = training_data[0].len();

        // Build a rolling history buffer with the last p observations per variable.
        let mut history: Vec<Vec<f64>> = (0..k)
            .map(|var| training_data[var][(n - p)..n].to_vec())
            .collect();

        let mut forecasts = vec![vec![0.0; horizon]; k];

        for h in 0..horizon {
            let mut y_new = vec![0.0; k];
            for eq in 0..k {
                let mut val = intercepts[eq];
                for lag in 1..=p {
                    for var in 0..k {
                        let hist_len = history[var].len();
                        val += coefficients[eq][var][lag - 1] * history[var][hist_len - lag];
                    }
                }
                y_new[eq] = val;
            }

            for var in 0..k {
                history[var].push(y_new[var]);
                forecasts[var][h] = y_new[var];
            }
        }

        Ok(forecasts)
    }

    /// Return the fitted values for each variable.
    ///
    /// Each inner vector has length n - p where n is the original series length
    /// and p is the lag order.
    pub fn fitted_values(&self) -> Option<&Vec<Vec<f64>>> {
        self.fitted_values.as_ref()
    }

    /// Return the residuals for each variable.
    ///
    /// Each inner vector has length n - p.
    pub fn residuals(&self) -> Option<&Vec<Vec<f64>>> {
        self.residuals.as_ref()
    }

    /// Return the coefficient matrices.
    ///
    /// `coefficients()[i][j][l]` is the coefficient in equation i for
    /// variable j at lag l+1.
    pub fn coefficients(&self) -> Option<&Vec<Vec<Vec<f64>>>> {
        self.coefficients.as_ref()
    }

    /// Return the intercept vector.
    pub fn intercepts(&self) -> Option<&Vec<f64>> {
        self.intercepts.as_ref()
    }

    /// Return the lag order p.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Return the number of variables k.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Perform a Granger causality test.
    ///
    /// Tests whether variable `cause` Granger-causes variable `effect` by
    /// comparing the unrestricted model (full VAR) with a restricted model
    /// that excludes lags of the `cause` variable from the `effect` equation.
    ///
    /// The test statistic is an F-statistic:
    ///
    ///   F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - k*p - 1))
    ///
    /// where RSS_r and RSS_u are the residual sums of squares from the
    /// restricted and unrestricted models, T is the effective sample size,
    /// and p is the number of restrictions (lag order).
    ///
    /// # Arguments
    /// * `cause` - Index of the potentially causal variable (0-based).
    /// * `effect` - Index of the effect variable (0-based).
    ///
    /// # Returns
    /// The F-statistic. Higher values indicate stronger evidence that
    /// `cause` Granger-causes `effect`.
    ///
    /// # Errors
    /// * `FitRequired` if the model has not been fitted.
    /// * `IndexOutOfBounds` if `cause` or `effect` >= k.
    /// * `InvalidParameter` if `cause == effect`.
    pub fn granger_causality_test(&self, cause: usize, effect: usize) -> Result<f64> {
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let training_data = self
            .training_data
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        let k = self.n_vars;
        let p = self.order;

        if cause >= k {
            return Err(ForecastError::IndexOutOfBounds {
                index: cause,
                size: k,
            });
        }
        if effect >= k {
            return Err(ForecastError::IndexOutOfBounds {
                index: effect,
                size: k,
            });
        }
        if cause == effect {
            return Err(ForecastError::InvalidParameter(
                "Cause and effect variables must be different".into(),
            ));
        }

        let n_eff = self.n_effective;
        let n = training_data[0].len();

        // Unrestricted RSS (from the full model already fitted).
        let rss_u: f64 = residuals[effect].iter().map(|r| r * r).sum();

        // Build the dependent variable for the effect equation.
        let y_effect: Vec<f64> = training_data[effect][p..n].to_vec();

        // Build restricted regressor map (exclude cause variable's lags).
        let mut restricted_regressors: HashMap<String, Vec<f64>> = HashMap::new();
        for lag in 1..=p {
            for var in 0..k {
                if var == cause {
                    continue;
                }
                let name = regressor_name(var, lag);
                let values: Vec<f64> = (p..n).map(|t| training_data[var][t - lag]).collect();
                restricted_regressors.insert(name, values);
            }
        }

        // Fit restricted model and compute RSS.
        let restricted_ols = ols_fit(&y_effect, &restricted_regressors)?;
        let restricted_predictions = if restricted_regressors.is_empty() {
            // Single-variable case after excluding the cause: only intercept remains.
            vec![restricted_ols.intercept; n_eff]
        } else {
            restricted_ols.predict(&restricted_regressors)?
        };

        let rss_r: f64 = (0..n_eff)
            .map(|t| {
                let r = y_effect[t] - restricted_predictions[t];
                r * r
            })
            .sum();

        // F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - k*p - 1))
        let df_num = p as f64;
        let df_den = n_eff as f64 - (k * p) as f64 - 1.0;

        if df_den <= 0.0 {
            return Err(ForecastError::InsufficientData {
                needed: k * p + 2,
                got: n_eff,
                hint: Some("Not enough observations for Granger causality test".into()),
            });
        }

        let f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den);

        Ok(f_stat)
    }
}

/// Build the full regressor map for the VAR model.
///
/// For each time t in [p..n), creates regressors y{var}_lag{lag} containing
/// the value of variable `var` at time `t - lag`.
fn build_regressor_map(
    data: &[Vec<f64>],
    k: usize,
    p: usize,
    n: usize,
) -> HashMap<String, Vec<f64>> {
    let mut regressor_map: HashMap<String, Vec<f64>> = HashMap::new();
    for lag in 1..=p {
        for var in 0..k {
            let name = regressor_name(var, lag);
            let values: Vec<f64> = (p..n).map(|t| data[var][t - lag]).collect();
            regressor_map.insert(name, values);
        }
    }
    regressor_map
}

/// Generate a regressor name for variable `var` at lag `lag`.
fn regressor_name(var: usize, lag: usize) -> String {
    format!("y{}_lag{}", var, lag)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a simple 2-variable VAR(1) system with small noise.
    ///
    /// y1(t) = c1 + a11 * y1(t-1) + a12 * y2(t-1) + e1(t)
    /// y2(t) = c2 + a21 * y1(t-1) + a22 * y2(t-1) + e2(t)
    fn generate_var1_data(n: usize, c: [f64; 2], a: [[f64; 2]; 2], seed: u64) -> Vec<Vec<f64>> {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];

        y1[0] = rng.gen_range(-1.0..1.0);
        y2[0] = rng.gen_range(-1.0..1.0);

        for t in 1..n {
            let e1: f64 = rng.gen_range(-0.01..0.01);
            let e2: f64 = rng.gen_range(-0.01..0.01);
            y1[t] = c[0] + a[0][0] * y1[t - 1] + a[0][1] * y2[t - 1] + e1;
            y2[t] = c[1] + a[1][0] * y1[t - 1] + a[1][1] * y2[t - 1] + e2;
        }

        vec![y1, y2]
    }

    #[test]
    fn fit_and_predict_basic() {
        let data = generate_var1_data(100, [0.5, 0.3], [[0.6, 0.1], [0.05, 0.7]], 42);

        let mut model = VAR::new(1);
        model.fit(&data).unwrap();

        let forecasts = model.predict(5).unwrap();
        assert_eq!(forecasts.len(), 2);
        assert_eq!(forecasts[0].len(), 5);
        assert_eq!(forecasts[1].len(), 5);
    }

    #[test]
    fn coefficient_recovery() {
        let c = [0.5, 0.3];
        let a = [[0.6, 0.1], [0.05, 0.7]];
        let data = generate_var1_data(500, c, a, 123);

        let mut model = VAR::new(1);
        model.fit(&data).unwrap();

        let coefs = model.coefficients().unwrap();
        let intercepts = model.intercepts().unwrap();

        assert!(
            (intercepts[0] - c[0]).abs() < 0.1,
            "intercept[0]: expected ~{}, got {}",
            c[0],
            intercepts[0]
        );
        assert!(
            (intercepts[1] - c[1]).abs() < 0.1,
            "intercept[1]: expected ~{}, got {}",
            c[1],
            intercepts[1]
        );

        // coefs[eq][var][lag]: equation 0, variable 0, lag 0 => a[0][0]
        assert!(
            (coefs[0][0][0] - a[0][0]).abs() < 0.05,
            "a[0][0]: expected ~{}, got {}",
            a[0][0],
            coefs[0][0][0]
        );
        assert!(
            (coefs[0][1][0] - a[0][1]).abs() < 0.05,
            "a[0][1]: expected ~{}, got {}",
            a[0][1],
            coefs[0][1][0]
        );
        assert!(
            (coefs[1][0][0] - a[1][0]).abs() < 0.05,
            "a[1][0]: expected ~{}, got {}",
            a[1][0],
            coefs[1][0][0]
        );
        assert!(
            (coefs[1][1][0] - a[1][1]).abs() < 0.05,
            "a[1][1]: expected ~{}, got {}",
            a[1][1],
            coefs[1][1][0]
        );
    }

    #[test]
    fn forecast_dimensions() {
        let data = generate_var1_data(50, [0.1, 0.2], [[0.5, 0.0], [0.0, 0.5]], 99);

        let mut model = VAR::new(2);
        model.fit(&data).unwrap();

        for h in [1, 5, 10, 20] {
            let forecasts = model.predict(h).unwrap();
            assert_eq!(forecasts.len(), 2, "should have 2 variables");
            assert_eq!(forecasts[0].len(), h, "horizon mismatch for h={}", h);
            assert_eq!(forecasts[1].len(), h, "horizon mismatch for h={}", h);
        }
    }

    #[test]
    fn single_variable_degenerates_to_ar() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(77);
        let n = 200;
        let phi = 0.8;
        let c = 1.0;
        let mut y = vec![0.0; n];
        y[0] = rng.gen_range(-1.0..1.0);
        for t in 1..n {
            y[t] = c + phi * y[t - 1] + rng.gen_range(-0.01..0.01);
        }

        let mut model = VAR::new(1);
        model.fit(&[y]).unwrap();

        let coefs = model.coefficients().unwrap();
        let intercepts = model.intercepts().unwrap();

        assert_eq!(coefs.len(), 1);
        assert_eq!(coefs[0].len(), 1);
        assert_eq!(coefs[0][0].len(), 1);

        assert!(
            (coefs[0][0][0] - phi).abs() < 0.05,
            "phi: expected ~{}, got {}",
            phi,
            coefs[0][0][0]
        );
        assert!(
            (intercepts[0] - c).abs() < 0.15,
            "c: expected ~{}, got {}",
            c,
            intercepts[0]
        );
    }

    #[test]
    fn insufficient_data_error() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut model = VAR::new(2);
        let result = model.fit(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::InsufficientData { .. }
        ));
    }

    #[test]
    fn empty_data_error() {
        let mut model = VAR::new(1);
        assert!(matches!(
            model.fit(&[]).unwrap_err(),
            ForecastError::EmptyData
        ));
    }

    #[test]
    fn zero_order_error() {
        let mut model = VAR::new(0);
        let data = vec![vec![1.0, 2.0, 3.0]];
        assert!(matches!(
            model.fit(&data).unwrap_err(),
            ForecastError::InvalidParameter(_)
        ));
    }

    #[test]
    fn predict_before_fit_error() {
        let model = VAR::new(1);
        assert!(matches!(
            model.predict(5).unwrap_err(),
            ForecastError::FitRequired
        ));
    }

    #[test]
    fn zero_horizon_error() {
        let data = generate_var1_data(50, [0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]], 1);
        let mut model = VAR::new(1);
        model.fit(&data).unwrap();
        assert!(matches!(
            model.predict(0).unwrap_err(),
            ForecastError::InvalidParameter(_)
        ));
    }

    #[test]
    fn dimension_mismatch_error() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        let mut model = VAR::new(1);
        assert!(matches!(
            model.fit(&data).unwrap_err(),
            ForecastError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn residuals_and_fitted_values() {
        let data = generate_var1_data(50, [0.1, 0.2], [[0.5, 0.1], [0.1, 0.5]], 55);
        let mut model = VAR::new(1);
        model.fit(&data).unwrap();

        let residuals = model.residuals().unwrap();
        let fitted = model.fitted_values().unwrap();

        assert_eq!(residuals.len(), 2);
        assert_eq!(fitted.len(), 2);
        assert_eq!(residuals[0].len(), 49);
        assert_eq!(fitted[0].len(), 49);

        // fitted + residuals should reconstruct the original data (from index p onwards).
        for var in 0..2 {
            for t in 0..49 {
                let reconstructed = fitted[var][t] + residuals[var][t];
                let original = data[var][t + 1]; // offset by p=1
                assert!(
                    (reconstructed - original).abs() < 1e-10,
                    "Reconstruction mismatch at var={}, t={}",
                    var,
                    t
                );
            }
        }
    }

    #[test]
    fn granger_causality_on_causal_system() {
        // y1(t) = 0.5 * y1(t-1) + e1       (y2 does NOT cause y1)
        // y2(t) = 0.3 * y1(t-1) + 0.5 * y2(t-1) + e2  (y1 causes y2)
        let data = generate_var1_data(300, [0.0, 0.0], [[0.5, 0.0], [0.3, 0.5]], 42);

        let mut model = VAR::new(1);
        model.fit(&data).unwrap();

        let f_12 = model.granger_causality_test(0, 1).unwrap();
        let f_21 = model.granger_causality_test(1, 0).unwrap();

        assert!(
            f_12 > f_21,
            "y1->y2 F-stat ({}) should be larger than y2->y1 F-stat ({})",
            f_12,
            f_21
        );
        assert!(
            f_12 > 4.0,
            "y1->y2 F-stat ({}) should be significant (> 4.0)",
            f_12
        );
    }

    #[test]
    fn granger_causality_same_variable_error() {
        let data = generate_var1_data(50, [0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]], 1);
        let mut model = VAR::new(1);
        model.fit(&data).unwrap();

        assert!(matches!(
            model.granger_causality_test(0, 0).unwrap_err(),
            ForecastError::InvalidParameter(_)
        ));
    }

    #[test]
    fn granger_causality_out_of_bounds_error() {
        let data = generate_var1_data(50, [0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]], 1);
        let mut model = VAR::new(1);
        model.fit(&data).unwrap();

        assert!(matches!(
            model.granger_causality_test(2, 0).unwrap_err(),
            ForecastError::IndexOutOfBounds { .. }
        ));
        assert!(matches!(
            model.granger_causality_test(0, 2).unwrap_err(),
            ForecastError::IndexOutOfBounds { .. }
        ));
    }

    #[test]
    fn var2_model() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(88);
        let n = 500;
        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];

        y1[0] = 0.1;
        y1[1] = 0.2;
        y2[0] = -0.1;
        y2[1] = 0.0;

        for t in 2..n {
            y1[t] =
                0.4 * y1[t - 1] + 0.1 * y1[t - 2] + 0.05 * y2[t - 1] + rng.gen_range(-0.01..0.01);
            y2[t] =
                0.3 * y2[t - 1] + 0.15 * y2[t - 2] + 0.1 * y1[t - 1] + rng.gen_range(-0.01..0.01);
        }

        let data = vec![y1, y2];
        let mut model = VAR::new(2);
        model.fit(&data).unwrap();

        let coefs = model.coefficients().unwrap();

        // Equation 0: y1(t) = 0.4*y1(t-1) + 0.1*y1(t-2) + 0.05*y2(t-1) + 0*y2(t-2)
        assert!(
            (coefs[0][0][0] - 0.4).abs() < 0.05,
            "y1 lag1 in eq0: {}",
            coefs[0][0][0]
        );
        assert!(
            (coefs[0][0][1] - 0.1).abs() < 0.05,
            "y1 lag2 in eq0: {}",
            coefs[0][0][1]
        );
        assert!(
            (coefs[0][1][0] - 0.05).abs() < 0.05,
            "y2 lag1 in eq0: {}",
            coefs[0][1][0]
        );

        let forecasts = model.predict(10).unwrap();
        assert_eq!(forecasts.len(), 2);
        assert_eq!(forecasts[0].len(), 10);
    }

    #[test]
    fn three_variable_system() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(55);
        let n = 150;
        let mut y = vec![vec![0.0; n]; 3];

        for var in 0..3 {
            y[var][0] = rng.gen_range(-0.5..0.5);
        }

        for t in 1..n {
            y[0][t] = 0.5 * y[0][t - 1] + rng.gen_range(-0.01..0.01);
            y[1][t] = 0.3 * y[0][t - 1] + 0.4 * y[1][t - 1] + rng.gen_range(-0.01..0.01);
            y[2][t] = 0.2 * y[1][t - 1] + 0.3 * y[2][t - 1] + rng.gen_range(-0.01..0.01);
        }

        let mut model = VAR::new(1);
        model.fit(&y).unwrap();

        assert_eq!(model.n_vars(), 3);

        let forecasts = model.predict(5).unwrap();
        assert_eq!(forecasts.len(), 3);
        for var in 0..3 {
            assert_eq!(forecasts[var].len(), 5);
        }

        let coefs = model.coefficients().unwrap();
        assert_eq!(coefs.len(), 3);
        assert_eq!(coefs[0].len(), 3);
        assert_eq!(coefs[0][0].len(), 1);
    }

    #[test]
    fn nan_in_data_error() {
        let data = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];
        let mut model = VAR::new(1);
        assert!(model.fit(&data).is_err());
    }

    #[test]
    fn granger_before_fit_error() {
        let model = VAR::new(1);
        assert!(matches!(
            model.granger_causality_test(0, 1).unwrap_err(),
            ForecastError::FitRequired
        ));
    }
}
