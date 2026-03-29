//! ARIMA and SARIMA (Seasonal ARIMA) models.
//!
//! ARIMA(p, d, q) combines:
//! - AR(p): Autoregressive component
//! - I(d): Differencing for stationarity
//! - MA(q): Moving average component
//!
//! SARIMA(p, d, q)(P, D, Q)\[s\] extends ARIMA with seasonal components:
//! - SAR(P): Seasonal autoregressive component
//! - SI(D): Seasonal differencing
//! - SMA(Q): Seasonal moving average component
//! - s: Seasonal period
//!
//! Both ARIMA and SARIMA support exogenous regressors (ARIMAX/SARIMAX).
//! When a TimeSeries with regressors is provided:
//! 1. OLS regression removes the exogenous effects
//! 2. ARIMA/SARIMA is fit on the residuals
//! 3. Forecasts add back the exogenous contribution

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::arima::diff::{difference, integrate};
use crate::models::{validate_series_complete, FittedParams, Forecaster};
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use crate::utils::optimization::{lbfgs_optimize, nelder_mead, LbfgsConfig, NelderMeadConfig};
use crate::utils::stats::quantile_normal;
use std::collections::HashMap;

/// ARIMA model specification (non-seasonal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ARIMASpec {
    /// AR order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// MA order (q)
    pub q: usize,
}

impl ARIMASpec {
    /// Create a new ARIMA specification.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.p + self.q + 1 // AR + MA + intercept
    }
}

impl Default for ARIMASpec {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

/// SARIMA model specification (seasonal ARIMA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SARIMASpec {
    /// Non-seasonal AR order (p)
    pub p: usize,
    /// Non-seasonal differencing order (d)
    pub d: usize,
    /// Non-seasonal MA order (q)
    pub q: usize,
    /// Seasonal AR order (P)
    pub cap_p: usize,
    /// Seasonal differencing order (D)
    pub cap_d: usize,
    /// Seasonal MA order (Q)
    pub cap_q: usize,
    /// Seasonal period (s)
    pub s: usize,
}

impl SARIMASpec {
    /// Create a new SARIMA specification.
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        cap_p: usize,
        cap_d: usize,
        cap_q: usize,
        s: usize,
    ) -> Self {
        Self {
            p,
            d,
            q,
            cap_p,
            cap_d,
            cap_q,
            s,
        }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.p + self.q + self.cap_p + self.cap_q + 1 // AR + MA + SAR + SMA + intercept
    }

    /// Check if the model has seasonal components.
    pub fn is_seasonal(&self) -> bool {
        self.s > 1 && (self.cap_p > 0 || self.cap_d > 0 || self.cap_q > 0)
    }
}

impl Default for SARIMASpec {
    fn default() -> Self {
        Self::new(1, 1, 1, 0, 0, 0, 1)
    }
}

/// ARIMA forecasting model.
///
/// ARIMA(p, d, q) combines:
/// - AR(p): Autoregressive component
/// - I(d): Differencing for stationarity
/// - MA(q): Moving average component
///
/// Supports exogenous regressors (ARIMAX) via TimeSeries.regressors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ARIMA {
    /// Model specification.
    spec: ARIMASpec,
    /// AR coefficients.
    ar_coefficients: Vec<f64>,
    /// MA coefficients.
    ma_coefficients: Vec<f64>,
    /// Intercept (mean of differenced series).
    intercept: f64,
    /// Original series (for integration).
    original: Option<Vec<f64>>,
    /// Differenced series.
    differenced: Option<Vec<f64>>,
    /// Fitted values on differenced scale.
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
    fitted_diff: Option<Vec<f64>>,
    /// Residuals.
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// AIC.
    aic: Option<f64>,
    /// BIC.
    bic: Option<f64>,
    /// Series length.
    n: usize,
    /// OLS result for exogenous regressors (if any).
    #[cfg_attr(feature = "serde", serde(skip))]
    exog_ols: Option<OLSResult>,
    /// Whether to skip optimization when fit() is called (warm-start mode).
    skip_optimization: bool,
}

impl ARIMA {
    /// Create a new ARIMA model.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            spec: ARIMASpec::new(p, d, q),
            ar_coefficients: vec![],
            ma_coefficients: vec![],
            intercept: 0.0,
            original: None,
            differenced: None,
            fitted_diff: None,
            residuals: None,
            residual_variance: None,
            aic: None,
            bic: None,
            n: 0,
            exog_ols: None,
            skip_optimization: false,
        }
    }

    /// Create a warm-started ARIMA model with pre-fitted coefficients.
    ///
    /// The resulting model can be used for forecasting after calling `fit()` with
    /// data (needed for differencing context). When `fit()` is called, the
    /// provided coefficients are used directly without re-optimization.
    ///
    /// # Arguments
    /// * `p` - AR order
    /// * `d` - Differencing order
    /// * `q` - MA order
    /// * `ar_coeffs` - Pre-fitted AR coefficients (length must equal p)
    /// * `ma_coeffs` - Pre-fitted MA coefficients (length must equal q)
    /// * `intercept` - Pre-fitted intercept value
    pub fn with_coefficients(
        p: usize,
        d: usize,
        q: usize,
        ar_coeffs: Vec<f64>,
        ma_coeffs: Vec<f64>,
        intercept: f64,
    ) -> Self {
        Self {
            spec: ARIMASpec::new(p, d, q),
            ar_coefficients: ar_coeffs,
            ma_coefficients: ma_coeffs,
            intercept,
            original: None,
            differenced: None,
            fitted_diff: None,
            residuals: None,
            residual_variance: None,
            aic: None,
            bic: None,
            n: 0,
            exog_ols: None,
            skip_optimization: true,
        }
    }

    /// Create an ARIMA(1,1,1) model.
    pub fn arima_111() -> Self {
        Self::new(1, 1, 1)
    }

    /// Create an AR(p) model (ARIMA with d=0, q=0).
    pub fn ar(p: usize) -> Self {
        Self::new(p, 0, 0)
    }

    /// Create an MA(q) model (ARIMA with p=0, d=0).
    pub fn ma(q: usize) -> Self {
        Self::new(0, 0, q)
    }

    /// Get the model specification.
    pub fn spec(&self) -> ARIMASpec {
        self.spec
    }

    /// Get AR coefficients.
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coefficients
    }

    /// Get MA coefficients.
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coefficients
    }

    /// Get the intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get AIC.
    pub fn aic(&self) -> Option<f64> {
        self.aic
    }

    /// Get BIC.
    pub fn bic(&self) -> Option<f64> {
        self.bic
    }

    /// Score-only evaluation: compute AIC or BIC from a pre-computed differenced series
    /// without storing any model state. Used by AutoARIMA to avoid full model construction
    /// during candidate search.
    ///
    /// Returns `Some(score)` on success, `None` if fitting fails.
    pub(crate) fn score_order(
        p: usize,
        q: usize,
        diff_series: &[f64],
        use_aic: bool,
    ) -> Option<f64> {
        let start = p.max(q);
        if diff_series.len() <= start + 2 {
            return None;
        }

        if p == 0 && q == 0 {
            // Just intercept model: compute variance directly
            let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;
            let n_eff = (diff_series.len() - start) as f64;
            let variance = diff_series[start..]
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / n_eff;
            if variance <= 0.0 || !variance.is_finite() {
                return None;
            }
            let k = 1.0; // just intercept
            let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());
            let score = if use_aic {
                -2.0 * ll + 2.0 * k
            } else {
                -2.0 * ll + k * n_eff.ln()
            };
            return if score.is_finite() { Some(score) } else { None };
        }

        // score_order operates on already-differenced data — no intercept needed
        // (matches statsmodels convention: ARIMA with d>0 has no constant)
        let hr_init = if p + q >= 2 {
            let hr = Self::hannan_rissanen_init(diff_series, p, q);
            hr[1..].to_vec() // skip intercept
        } else {
            let mut init = Vec::with_capacity(p + q);
            for i in 0..p {
                init.push(0.3 / (i + 1) as f64);
            }
            for i in 0..q {
                init.push(0.3 / (i + 1) as f64);
            }
            init
        };
        let n_params = p + q;

        let mut bounds = Vec::with_capacity(p + q);
        for _ in 0..(p + q) {
            bounds.push((-0.99, 0.99));
        }

        let config = NelderMeadConfig {
            max_iter: 100,
            tolerance: 1e-6,
            ..Default::default()
        };

        let residuals_buf = std::cell::RefCell::new(vec![0.0; diff_series.len()]);

        // Optimize using CSS (robust landscape)
        let result = nelder_mead(
            |params| {
                let mut buf = residuals_buf.borrow_mut();
                Self::calculate_css(diff_series, p, q, &params[..p], &params[p..], 0.0, &mut buf)
            },
            &hr_init,
            Some(&bounds),
            config,
        );

        // Use CSS for AIC/BIC scoring (consistent with CSS optimization)
        let css = result.optimal_value;
        let n_params_aic = n_params + 1; // +1 for sigma^2
        if !css.is_finite() || css <= 0.0 {
            return None;
        }

        let start = p.max(q);
        let n_eff = (diff_series.len() - start) as f64;
        let variance = css / n_eff;
        let k = n_params_aic as f64;
        let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());

        let score = if use_aic {
            -2.0 * ll + 2.0 * k
        } else {
            -2.0 * ll + k * n_eff.ln()
        };

        if score.is_finite() {
            Some(score)
        } else {
            None
        }
    }

    /// Compute exact negative log-likelihood via the innovations algorithm.
    ///
    /// This implements the Python statsmodels innovations algorithm:
    /// 1. `lfilter(ma_poly, ar_poly, impulse)` to get MA(∞) coefficients
    /// 2. `arma_acovf` via Brockwell-Davis linear system (eq 3.3.8)
    /// 3. Innovations recursion to get prediction variances v and coefficients theta
    /// 4. Filter to compute one-step-ahead errors u
    /// 5. Concentrated negative log-likelihood (conditional on first m observations)
    ///
    /// Sigma^2 is concentrated out (profiled likelihood).
    /// Public accessor for testing MLE computation.
    #[doc(hidden)]
    pub fn calculate_mle_pub(
        diff_series: &[f64],
        p: usize,
        q: usize,
        ar: &[f64],
        ma: &[f64],
        intercept: f64,
        residuals: &mut [f64],
    ) -> f64 {
        Self::calculate_mle(diff_series, p, q, ar, ma, intercept, residuals)
    }

    /// IIR filter (scipy.signal.lfilter equivalent).
    ///
    /// Implements: `a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - ...`
    fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut y = vec![0.0; n];
        let a0 = a[0];
        for i in 0..n {
            let mut val = 0.0;
            for (j, &bj) in b.iter().enumerate() {
                if i >= j {
                    val += bj * x[i - j];
                }
            }
            for (j, &aj) in a.iter().enumerate().skip(1) {
                if i >= j {
                    val -= aj * y[i - j];
                }
            }
            y[i] = val / a0;
        }
        y
    }

    /// Compute ARMA autocovariances via exact Brockwell-Davis linear system (eq 3.3.8).
    ///
    /// `ar_poly` = [1, -phi1, -phi2, ...], `ma_poly` = [1, theta1, theta2, ...]
    /// Returns gamma(0..nobs-1) with sigma2=1.
    fn arma_acovf(ar_poly: &[f64], ma_poly: &[f64], nobs: usize) -> Vec<f64> {
        let p = ar_poly.len() - 1; // AR order
        let q = ma_poly.len() - 1; // MA order
        let m = p.max(q) + 1;

        // Step 1: Compute MA(infinity) coefficients via lfilter
        let leads = m;
        let mut impulse = vec![0.0; leads];
        if !impulse.is_empty() {
            impulse[0] = 1.0;
        }
        let ma_coeffs = Self::lfilter(ma_poly, ar_poly, &impulse);

        // Step 2: Build linear system A * gamma = b (Brockwell-Davis eq 3.3.8)
        // Pad ar_poly to length m
        let mut tmp_ar = vec![0.0; m];
        for (i, &v) in ar_poly.iter().enumerate().take(m) {
            tmp_ar[i] = v;
        }

        let mut a_mat = vec![vec![0.0; m]; m];
        let mut b_vec = vec![0.0; m];

        for k in 0..m {
            // A[k, :k+1] = tmp_ar[:k+1][::-1]
            for j in 0..=k {
                a_mat[k][j] += tmp_ar[k - j];
            }
            // A[k, 1:m-k] += tmp_ar[k+1:m]
            if k + 1 < m {
                for j in 1..m - k {
                    if k + j < m {
                        a_mat[k][j] += tmp_ar[k + j];
                    }
                }
            }
            // b[k] = sigma2 * dot(ma_poly[k:q+1], ma_coeffs[:max(q+1-k, 0)])
            let ma_start = k;
            let dot_len = if q + 1 > k { q + 1 - k } else { 0 };
            let mut dot = 0.0;
            for i in 0..dot_len {
                if ma_start + i < ma_poly.len() && i < ma_coeffs.len() {
                    dot += ma_poly[ma_start + i] * ma_coeffs[i];
                }
            }
            b_vec[k] = dot; // sigma2 = 1
        }

        // Step 3: Solve A * gamma = b via Gaussian elimination with partial pivoting
        let mut acovf = vec![0.0; nobs];

        if let Some(gamma) = Self::solve_linear_system_inline(&a_mat, &b_vec) {
            for (i, &g) in gamma.iter().enumerate().take(m.min(nobs)) {
                acovf[i] = g;
            }
        } else {
            // Fallback: white noise
            acovf[0] = 1.0;
            return acovf;
        }

        // Step 4: Extend via AR recursion for lags h >= m
        // gamma(h) = -sum(ar_poly[1:] * gamma[h-1:h-p:-1])
        for h in m..nobs {
            let mut val = 0.0;
            for i in 1..ar_poly.len() {
                if h >= i {
                    val -= ar_poly[i] * acovf[h - i];
                }
            }
            acovf[h] = val;
        }

        acovf
    }

    /// Solve A*x = b via Gaussian elimination with partial pivoting.
    fn solve_linear_system_inline(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        if n == 0 || a.len() != n {
            return None;
        }

        // Build augmented matrix [A | b]
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = Vec::with_capacity(n + 1);
                r.extend_from_slice(row);
                r.push(b[i]);
                r
            })
            .collect();

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                let v = aug[row][col].abs();
                if v > max_val {
                    max_row = row;
                    max_val = v;
                }
            }
            if max_val < 1e-14 {
                return None; // singular
            }
            if max_row != col {
                aug.swap(col, max_row);
            }
            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    let val = aug[col][j];
                    aug[row][j] -= factor * val;
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }
        Some(x)
    }

    /// Check stationarity: verify all roots of the AR polynomial lie outside the unit circle.
    ///
    /// For AR(1): |phi1| < 1
    /// For AR(2): phi2 + phi1 < 1, phi2 - phi1 < 1, |phi2| < 1
    /// For higher orders: use the companion matrix eigenvalue check.
    fn check_stationarity(ar: &[f64]) -> bool {
        let p = ar.len();
        if p == 0 {
            return true;
        }
        if p == 1 {
            return ar[0].abs() < 1.0;
        }
        if p == 2 {
            // Triangle conditions for AR(2) stationarity
            return ar[1] + ar[0] < 1.0 && ar[1] - ar[0] < 1.0 && ar[1].abs() < 1.0;
        }
        // For higher orders: check necessary conditions
        // Sum of absolute AR coefficients < 1 is too strict but safe
        let sum_abs: f64 = ar.iter().map(|a| a.abs()).sum();
        sum_abs < 1.5 && ar.iter().all(|a| a.abs() < 1.0)
    }

    /// Toeplitz helper: fill a symmetric Toeplitz block in a flat row-major matrix.
    ///
    /// Ported from statsmodels `_arma_innovations.pyx::dtoeplitz`.
    fn toeplitz_fill(
        n: usize,
        offset0: usize,
        offset1: usize,
        column: &[f64],
        out: &mut [f64],
        out_cols: usize,
    ) {
        for i in 0..n {
            for j in 0..=i {
                out[(offset0 + i) * out_cols + (offset1 + j)] = column[i - j];
                if i != j {
                    out[(offset0 + j) * out_cols + (offset1 + i)] = column[i - j];
                }
            }
        }
    }

    /// Compute the transformed autocovariance matrix and residual acovf vector.
    ///
    /// Ported from statsmodels `_arma_innovations.pyx::darma_transformed_acovf_fast`.
    /// Returns (acovf_matrix [n x n flat row-major], acovf2 [1-D], n) where n = min(2*m, nobs).
    fn arma_transformed_acovf_fast(
        ar_poly: &[f64],
        ma_poly: &[f64],
        arma_acovf: &[f64],
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let nobs = arma_acovf.len();
        let p = ar_poly.len() - 1;
        let q = ma_poly.len() - 1;
        let m = p.max(q);
        let m2 = 2 * m;
        let n = m2.min(nobs);

        // acovf: m2 x m2 matrix (flat row-major)
        let mut acovf = vec![0.0; m2 * m2];
        // acovf2: 1-D vector of length max(nobs - m, 0)
        let acovf2_len = if nobs > m { nobs - m } else { 0 };
        let mut acovf2 = vec![0.0; acovf2_len];

        // Fill upper-left m x m Toeplitz block
        if m > 0 {
            Self::toeplitz_fill(m, 0, 0, arma_acovf, &mut acovf, m2);
        }

        // Fill lower-left m x m block (rows m..m2, cols 0..m) and transpose to upper-right
        if nobs > m {
            for j in 0..m {
                for i in m..m2 {
                    let mut val = arma_acovf[i - j];
                    for r in 1..=p {
                        let tmp_ix = if r > (i - j) {
                            r - (i - j)
                        } else {
                            (i - j) - r
                        };
                        val -= -ar_poly[r] * arma_acovf[tmp_ix];
                    }
                    acovf[i * m2 + j] = val;
                }
            }
            // acovf[:m, m:m2] = acovf[m:m2, :m].T
            for i in 0..m {
                for j in m..m2 {
                    acovf[i * m2 + j] = acovf[j * m2 + i];
                }
            }
        }

        // Fill acovf2: the stationary part
        if nobs > m {
            for i in 0..acovf2_len {
                for r in 0..=(q.saturating_sub(i)) {
                    if r < ma_poly.len() && r + i < ma_poly.len() {
                        acovf2[i] += ma_poly[r] * ma_poly[r + i];
                    }
                }
            }
        }

        // Return the n x n submatrix (copy into a properly sized flat array)
        let mut acovf_out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                acovf_out[i * n + j] = acovf[i * m2 + j];
            }
        }

        (acovf_out, acovf2, n)
    }

    /// Innovations algorithm: O(n * m^2) version.
    ///
    /// Ported from statsmodels `_arma_innovations.pyx::darma_innovations_algo_fast`.
    /// Returns (theta [nobs x (m+1) flat row-major], v [nobs]).
    fn arma_innovations_algo_fast(
        nobs: usize,
        ar_params: &[f64],
        ma_params: &[f64],
        acovf: &[f64],
        acovf_cols: usize,
        acovf2: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let p = ar_params.len();
        let q = ma_params.len();
        let m = p.max(q);
        let m2 = 2 * m;
        let theta_cols = m + 1;

        let mut v = vec![0.0; nobs];
        let mut theta = vec![0.0; nobs * theta_cols];

        if m > 0 {
            v[0] = acovf[0]; // acovf[0, 0]
        } else {
            v[0] = if !acovf2.is_empty() { acovf2[0] } else { 1.0 };
        }

        for n_idx in 0..(nobs - 1) {
            let _n = n_idx + 1;

            let start = if n_idx < m { 0 } else { n_idx + 1 - q };
            for k in start..=n_idx {
                if n_idx >= m && n_idx - k >= q {
                    continue;
                }

                let col = n_idx - k;
                if col >= theta_cols {
                    continue;
                }

                // Initialize theta[_n, n_idx - k]
                if _n < m2 && k < m {
                    // Use acovf matrix: acovf[n_idx + 1, k]
                    if n_idx + 1 < acovf_cols && k < acovf_cols {
                        theta[_n * theta_cols + col] = acovf[(n_idx + 1) * acovf_cols + k];
                    }
                } else {
                    // Use acovf2 vector
                    let idx = _n - k;
                    if idx < acovf2.len() {
                        theta[_n * theta_cols + col] = acovf2[idx];
                    }
                }

                let start2 = if n_idx < m { 0 } else { n_idx - m };
                for j in start2..k {
                    let nj = n_idx - j;
                    if nj < theta_cols {
                        let kj = k - j - 1;
                        // theta[k-1+1, k-j-1] = theta[k, kj]
                        if kj < theta_cols {
                            theta[_n * theta_cols + col] -=
                                theta[k * theta_cols + kj] * theta[_n * theta_cols + nj] * v[j];
                        }
                    }
                }
                if v[k].abs() > 0.0 {
                    theta[_n * theta_cols + col] /= v[k];
                }
            }

            // Compute v[n_idx + 1]
            if _n < m {
                // v[n+1] = acovf[n+1, n+1]
                if _n < acovf_cols {
                    v[_n] = acovf[_n * acovf_cols + _n];
                }
            } else {
                v[_n] = if !acovf2.is_empty() { acovf2[0] } else { 1.0 };
            }

            let start_v = if n_idx + 2 > (m + 1) {
                n_idx + 2 - (m + 1)
            } else {
                0
            };
            for i in start_v..=n_idx {
                let ni = n_idx - i;
                if ni < theta_cols {
                    v[_n] -= theta[_n * theta_cols + ni].powi(2) * v[i];
                }
            }
        }

        (theta, v)
    }

    /// Innovations filter: compute one-step-ahead prediction errors.
    ///
    /// Ported from statsmodels `_arma_innovations.pyx::darma_innovations_filter`.
    fn arma_innovations_filter(
        endog: &[f64],
        ar_params: &[f64],
        ma_params: &[f64],
        theta: &[f64],
        theta_cols: usize,
    ) -> Vec<f64> {
        let p = ar_params.len();
        let q = ma_params.len();
        let m = p.max(q);
        let nobs = endog.len();

        let mut u = vec![0.0; nobs];
        u[0] = endog[0];

        for i in 1..nobs {
            let mut hat = 0.0;
            if i < m {
                for j in 0..i {
                    if j < theta_cols {
                        hat += theta[i * theta_cols + j] * u[i - j - 1];
                    }
                }
            } else {
                for j in 0..p {
                    hat += ar_params[j] * endog[i - j - 1];
                }
                for j in 0..q {
                    if j < theta_cols {
                        hat += theta[i * theta_cols + j] * u[i - j - 1];
                    }
                }
            }
            u[i] = endog[i] - hat;
        }

        u
    }

    fn calculate_mle(
        diff_series: &[f64],
        p: usize,
        q: usize,
        ar: &[f64],
        ma: &[f64],
        _intercept: f64,
        _residuals: &mut [f64],
    ) -> f64 {
        let n = diff_series.len();
        if n < 3 {
            return f64::MAX;
        }

        // Check stationarity
        if !Self::check_stationarity(ar) {
            return f64::MAX;
        }

        // Build polynomial representations:
        // ar_poly = [1, -phi1, -phi2, ...]  (Python convention)
        // ma_poly = [1, theta1, theta2, ...]
        let mut ar_poly = vec![0.0; p + 1];
        ar_poly[0] = 1.0;
        for i in 0..p {
            ar_poly[i + 1] = -ar[i];
        }

        let mut ma_poly = vec![0.0; q + 1];
        ma_poly[0] = 1.0;
        for i in 0..q {
            ma_poly[i + 1] = ma[i];
        }

        // Step 1: Compute ARMA autocovariances
        let arma_acov = Self::arma_acovf(&ar_poly, &ma_poly, n);
        if arma_acov[0] <= 0.0 || !arma_acov[0].is_finite() {
            return f64::MAX;
        }

        // Step 2: Transformed autocovariance (statsmodels fast path)
        let (acovf, acovf2, acovf_n) =
            Self::arma_transformed_acovf_fast(&ar_poly, &ma_poly, &arma_acov);

        // Step 3: Innovations algorithm (O(n * m^2))
        // ar_params = [phi1, phi2, ...], ma_params = [theta1, theta2, ...]
        // (raw coefficients, not polynomial form)
        let (theta, v) = Self::arma_innovations_algo_fast(
            n, ar, // [phi1, phi2, ...]
            ma, // [theta1, theta2, ...]
            &acovf, acovf_n, &acovf2,
        );

        // Step 4: Innovations filter
        let theta_cols = p.max(q) + 1;
        let u = Self::arma_innovations_filter(
            diff_series,
            ar, // [phi1, phi2, ...]
            ma, // [theta1, theta2, ...]
            &theta,
            theta_cols,
        );

        // Step 5: Concentrated negative log-likelihood (full, all n observations)
        // NLL = 0.5 * (n * ln(2*pi*sigma2) + sum(ln(v_t)) + n)
        // where sigma2 = (1/n) * sum(u_t^2 / v_t)
        let mut sum_log_v = 0.0;
        let mut sum_e2_v = 0.0;
        for t in 0..n {
            if v[t] <= 0.0 || !v[t].is_finite() {
                return f64::MAX;
            }
            sum_log_v += v[t].ln();
            sum_e2_v += u[t] * u[t] / v[t];
        }

        let sigma2 = sum_e2_v / n as f64;
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return f64::MAX;
        }

        let nll =
            0.5 * (n as f64 * (sigma2.ln() + (2.0 * std::f64::consts::PI).ln() + 1.0) + sum_log_v);

        if !nll.is_finite() {
            return f64::MAX;
        }

        nll
    }

    /// CSS fallback (kept for non-differenced models where intercept matters).
    fn calculate_css(
        diff_series: &[f64],
        p: usize,
        q: usize,
        ar: &[f64],
        ma: &[f64],
        intercept: f64,
        residuals: &mut [f64],
    ) -> f64 {
        let n = diff_series.len();
        let start = p.max(q);

        if n <= start {
            return f64::MAX;
        }

        residuals[..n].fill(0.0);
        let mut css = 0.0;

        for t in start..n {
            let mut pred = intercept;

            for i in 0..p {
                pred += ar[i] * (diff_series[t - 1 - i] - intercept);
            }

            for i in 0..q {
                pred += ma[i] * residuals[t - 1 - i];
            }

            let error = diff_series[t] - pred;
            residuals[t] = error;
            css += error * error;
        }

        css
    }

    /// Hannan-Rissanen initialization for ARMA parameters.
    ///
    /// Provides near-optimal starting values by:
    /// 1. Fitting a long AR model via Yule-Walker to estimate residuals
    /// 2. Regressing y on lagged y and lagged residuals via OLS
    ///
    /// This gives much better initial estimates than arbitrary constants,
    /// enabling faster convergence and better accuracy.
    fn hannan_rissanen_init(diff_series: &[f64], p: usize, q: usize) -> Vec<f64> {
        let n = diff_series.len();
        let mean = diff_series.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = diff_series.iter().map(|&v| v - mean).collect();

        // Step 1: Fit a long AR model to get residual estimates
        let ar_order = (p + q + 5).min(n / 3).max(p.max(q) + 1);

        // Yule-Walker: solve R * phi = r where R is autocorrelation matrix
        let mut acf = vec![0.0; ar_order + 1];
        for lag in 0..=ar_order {
            let mut sum = 0.0;
            for t in lag..n {
                sum += centered[t] * centered[t - lag];
            }
            acf[lag] = sum / n as f64;
        }

        if acf[0] <= 0.0 {
            // Constant series — return zeros
            let mut init = vec![0.0; p + q + 1];
            init[0] = mean;
            return init;
        }

        // Solve Toeplitz system using Levinson-Durbin
        let ar_long = Self::levinson_durbin(&acf, ar_order);

        // Compute residuals from the long AR model
        let mut residuals = vec![0.0; n];
        for t in ar_order..n {
            let mut pred = 0.0;
            for (k, &phi) in ar_long.iter().enumerate() {
                pred += phi * centered[t - 1 - k];
            }
            residuals[t] = centered[t] - pred;
        }

        // Step 2: OLS regression of centered[t] on lagged centered and lagged residuals
        let start = ar_order.max(p.max(q));
        let n_obs = n - start;
        if n_obs < p + q + 2 {
            let mut init = vec![0.0; p + q + 1];
            init[0] = mean;
            return init;
        }

        let n_regressors = p + q;
        if n_regressors == 0 {
            return vec![mean];
        }

        // Build X matrix and y vector for OLS
        let mut xtx = vec![0.0; n_regressors * n_regressors];
        let mut xty = vec![0.0; n_regressors];

        for t in start..n {
            // Regressors: [y_{t-1}, ..., y_{t-p}, e_{t-1}, ..., e_{t-q}]
            let mut x_row = Vec::with_capacity(n_regressors);
            for i in 0..p {
                x_row.push(centered[t - 1 - i]);
            }
            for i in 0..q {
                x_row.push(residuals[t - 1 - i]);
            }

            // Accumulate X'X and X'y
            for i in 0..n_regressors {
                xty[i] += x_row[i] * centered[t];
                for j in 0..n_regressors {
                    xtx[i * n_regressors + j] += x_row[i] * x_row[j];
                }
            }
        }

        // Solve X'X * beta = X'y using Cholesky
        let beta = Self::solve_symmetric_positive(&xtx, &xty, n_regressors);

        // Build initial parameter vector: [intercept, ar1..arp, ma1..maq]
        let mut init = vec![0.0; p + q + 1];
        init[0] = mean;
        for i in 0..p {
            init[1 + i] = beta.get(i).copied().unwrap_or(0.0).clamp(-0.95, 0.95);
        }
        for i in 0..q {
            init[1 + p + i] = beta.get(p + i).copied().unwrap_or(0.0).clamp(-0.95, 0.95);
        }

        init
    }

    /// Levinson-Durbin algorithm for solving Yule-Walker equations.
    fn levinson_durbin(acf: &[f64], order: usize) -> Vec<f64> {
        if order == 0 || acf[0] <= 0.0 {
            return vec![];
        }

        let mut phi = vec![0.0; order];
        let mut phi_prev = vec![0.0; order];
        let mut err = acf[0];

        for k in 0..order {
            // Compute reflection coefficient
            let mut num = acf[k + 1];
            for j in 0..k {
                num -= phi_prev[j] * acf[k - j];
            }
            let kappa = num / err;

            if !kappa.is_finite() || kappa.abs() >= 1.0 {
                break;
            }

            // Update coefficients
            phi[k] = kappa;
            for j in 0..k {
                phi[j] = phi_prev[j] - kappa * phi_prev[k - 1 - j];
            }

            err *= 1.0 - kappa * kappa;
            if err <= 0.0 {
                break;
            }

            phi_prev[..=k].copy_from_slice(&phi[..=k]);
        }

        phi
    }

    /// Solve symmetric positive definite system A*x = b via Cholesky.
    fn solve_symmetric_positive(a_flat: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        // Cholesky: A = L * L'
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let diag = a_flat[i * n + i] - sum;
                    if diag <= 0.0 {
                        // Not positive definite — fall back to zeros
                        return vec![0.0; n];
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    l[i * n + j] = (a_flat[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        // Forward substitution: L * y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[i * n + j] * y[j];
            }
            y[i] = (b[i] - sum) / l[i * n + i];
        }

        // Back substitution: L' * x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[j * n + i] * x[j];
            }
            x[i] = (y[i] - sum) / l[i * n + i];
        }

        x
    }

    /// Estimate parameters using conditional least squares.
    fn estimate_parameters(&mut self, diff_series: &[f64]) {
        let p = self.spec.p;
        let q = self.spec.q;

        let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;

        if p == 0 && q == 0 {
            // When d > 0, no intercept (matches statsmodels convention)
            self.intercept = if self.spec.d > 0 { 0.0 } else { mean };
            self.ar_coefficients = vec![];
            self.ma_coefficients = vec![];
            return;
        }

        // When d > 0: don't include intercept (statsmodels convention)
        // When d = 0: include intercept
        let include_intercept = self.spec.d == 0;
        let n_opt_params = if include_intercept { p + q + 1 } else { p + q };

        // Hannan-Rissanen initialization for p+q >= 2
        let initial = if p + q >= 2 {
            let hr = Self::hannan_rissanen_init(diff_series, p, q);
            if include_intercept {
                hr
            } else {
                hr[1..].to_vec() // skip intercept
            }
        } else {
            let mut init = Vec::with_capacity(n_opt_params);
            if include_intercept {
                init.push(mean);
            }
            for i in 0..p {
                init.push(0.3 / (i + 1) as f64);
            }
            for i in 0..q {
                init.push(0.3 / (i + 1) as f64);
            }
            init
        };

        let mut bounds = Vec::with_capacity(n_opt_params);
        if include_intercept {
            bounds.push((f64::NEG_INFINITY, f64::INFINITY));
        }
        for _ in 0..(p + q) {
            bounds.push((-0.99, 0.99));
        }

        let residuals_buf = std::cell::RefCell::new(vec![0.0; diff_series.len()]);

        // Stage 1: Use CSS for optimization (robust, well-behaved landscape).
        // For d>0, follow with MLE refinement to match statsmodels.
        let obj_fn = if include_intercept {
            Box::new(|params: &[f64]| {
                let mut buf = residuals_buf.borrow_mut();
                Self::calculate_css(
                    diff_series,
                    p,
                    q,
                    &params[1..1 + p],
                    &params[1 + p..],
                    params[0],
                    &mut buf,
                )
            }) as Box<dyn Fn(&[f64]) -> f64>
        } else {
            // d > 0: CSS without intercept (matches statsmodels convention)
            Box::new(|params: &[f64]| {
                let mut buf = residuals_buf.borrow_mut();
                Self::calculate_css(diff_series, p, q, &params[..p], &params[p..], 0.0, &mut buf)
            }) as Box<dyn Fn(&[f64]) -> f64>
        };

        // Two-phase: L-BFGS warm-start (CSS) → NM refinement (CSS) → MLE refinement (d>0)
        let use_lbfgs = p + q >= 2;

        // Phase 1: CSS warm-start via L-BFGS to get near the optimum
        let css_buf = std::cell::RefCell::new(vec![0.0; diff_series.len()]);
        let css_fn: Box<dyn Fn(&[f64]) -> f64> = if include_intercept {
            Box::new(|params: &[f64]| {
                let mut buf = css_buf.borrow_mut();
                Self::calculate_css(
                    diff_series,
                    p,
                    q,
                    &params[1..1 + p],
                    &params[1 + p..],
                    params[0],
                    &mut buf,
                )
            })
        } else {
            Box::new(|params: &[f64]| {
                let mut buf = css_buf.borrow_mut();
                Self::calculate_css(diff_series, p, q, &params[..p], &params[p..], 0.0, &mut buf)
            })
        };

        let lbfgs_result = if use_lbfgs {
            Some(lbfgs_optimize(
                &css_fn,
                &initial,
                Some(&bounds),
                LbfgsConfig {
                    max_iter: 100,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        // NM refinement: start from L-BFGS solution if available, else from init
        let nm_start = lbfgs_result
            .as_ref()
            .map(|r| r.optimal_point.as_slice())
            .unwrap_or(&initial);

        // Phase 2: NM refinement on CSS objective
        let nm_iters = if use_lbfgs { 500 } else { 1000 };
        let config = NelderMeadConfig {
            max_iter: nm_iters,
            tolerance: 1e-10,
            ..Default::default()
        };
        let nm_result = nelder_mead(&obj_fn, nm_start, Some(&bounds), config);

        let css_result = match &lbfgs_result {
            Some(lr) if lr.optimal_value < nm_result.optimal_value => lr.clone(),
            _ => nm_result,
        };

        // Extract CSS-optimized parameters
        if include_intercept {
            self.intercept = css_result.optimal_point[0];
            self.ar_coefficients = css_result.optimal_point[1..1 + p].to_vec();
            self.ma_coefficients = css_result.optimal_point[1 + p..].to_vec();
        } else {
            self.intercept = 0.0;
            self.ar_coefficients = css_result.optimal_point[..p].to_vec();
            self.ma_coefficients = css_result.optimal_point[p..].to_vec();

            // Phase 3 (d>0 only): MLE NM refinement starting from CSS optimum
            // Small steps near the CSS solution to match statsmodels exact MLE
            let mle_buf = std::cell::RefCell::new(vec![0.0; diff_series.len()]);
            let mle_fn = |params: &[f64]| {
                let mut buf = mle_buf.borrow_mut();
                Self::calculate_mle(diff_series, p, q, &params[..p], &params[p..], 0.0, &mut buf)
            };

            let mle_config = NelderMeadConfig {
                max_iter: 200,
                tolerance: 1e-10,
                initial_step: 0.02,
                ..Default::default()
            };
            let mle_result = nelder_mead(
                &mle_fn,
                &css_result.optimal_point,
                Some(&bounds),
                mle_config,
            );

            // Evaluate MLE at the CSS optimum for comparison
            let css_mle_val = {
                let mut buf = vec![0.0; diff_series.len()];
                Self::calculate_mle(
                    diff_series,
                    p,
                    q,
                    &css_result.optimal_point[..p],
                    &css_result.optimal_point[p..],
                    0.0,
                    &mut buf,
                )
            };

            // Use MLE params only if they strictly improve, are stationary,
            // and AR coefficients are not too close to the unit root boundary
            // (near-unit-root params can cause explosive integrated forecasts).
            let mle_ar = &mle_result.optimal_point[..p];
            let ar_sum: f64 = mle_ar.iter().sum();
            let ar_abs_sum: f64 = mle_ar.iter().map(|a| a.abs()).sum();
            let mle_safe = ar_sum < 0.97 && ar_abs_sum < 1.95;
            if mle_result.optimal_value < css_mle_val
                && mle_result.optimal_value < f64::MAX
                && Self::check_stationarity(mle_ar)
                && mle_safe
            {
                self.ar_coefficients = mle_ar.to_vec();
                self.ma_coefficients = mle_result.optimal_point[p..].to_vec();
            }
        }
    }

    /// Calculate fitted values and residuals.
    fn calculate_fitted(&mut self, diff_series: &[f64]) {
        let n = diff_series.len();
        let p = self.spec.p;
        let q = self.spec.q;
        let start = p.max(q);

        let mut fitted = vec![f64::NAN; n];
        let mut residuals = vec![0.0; n];

        for t in start..n {
            let mut pred = self.intercept;

            // AR component
            for i in 0..p {
                pred += self.ar_coefficients[i] * (diff_series[t - 1 - i] - self.intercept);
            }

            // MA component
            for i in 0..q {
                pred += self.ma_coefficients[i] * residuals[t - 1 - i];
            }

            fitted[t] = pred;
            residuals[t] = diff_series[t] - pred;
        }

        // Calculate residual variance
        let valid_residuals: Vec<f64> = residuals[start..].to_vec();
        if !valid_residuals.is_empty() {
            let variance =
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);

            // Calculate information criteria
            let n_eff = valid_residuals.len() as f64;
            let k = self.spec.num_params() as f64;
            let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());

            self.aic = Some(-2.0 * ll + 2.0 * k);
            self.bic = Some(-2.0 * ll + k * n_eff.ln());
        }

        self.fitted_diff = Some(fitted);
        self.residuals = Some(residuals);
    }

    /// Internal prediction method that handles both with and without exogenous cases.
    fn predict_internal(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<Forecast> {
        let original = self
            .original
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let diff_series = self
            .differenced
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let residuals = self
            .residuals
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Calculate exogenous contribution if applicable
        let exog_contribution = if let Some(ols) = &self.exog_ols {
            let future = future_regressors.ok_or_else(|| {
                ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors. Future regressor values required."
                        .into(),
                )
            })?;

            // Validate future regressors have correct length
            for name in &ols.regressor_names {
                let values = future.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }

            // Predict exogenous contribution
            Some(ols.predict(future)?)
        } else {
            if future_regressors.is_some_and(|r| !r.is_empty()) {
                return Err(ForecastError::InvalidParameter(
                    "Model was not fit with exogenous regressors".into(),
                ));
            }
            None
        };

        let p = self.spec.p;
        let q = self.spec.q;

        // Forecast on differenced scale — clone then reserve to avoid reallocation
        let diff_len = diff_series.len();
        let mut extended_diff = diff_series.to_vec();
        extended_diff.reserve(horizon);
        let mut extended_residuals = residuals.to_vec();
        extended_residuals.reserve(horizon);

        for _ in 0..horizon {
            let t = extended_diff.len();
            let mut pred = self.intercept;

            // AR component
            for i in 0..p {
                if t > i {
                    pred += self.ar_coefficients[i] * (extended_diff[t - 1 - i] - self.intercept);
                }
            }

            // MA component (residuals become 0 for forecasts)
            for i in 0..q {
                if t > i {
                    pred += self.ma_coefficients[i] * extended_residuals[t - 1 - i];
                }
            }

            extended_diff.push(pred);
            extended_residuals.push(0.0); // Future residuals are 0
        }

        // Extract forecast on differenced scale
        let forecast_diff: Vec<f64> = extended_diff[diff_len..].to_vec();

        // Integrate back to original scale
        let mut predictions = if self.spec.d > 0 {
            integrate(&forecast_diff, original, self.spec.d)
        } else {
            forecast_diff
        };

        // Add exogenous contribution
        if let Some(exog) = exog_contribution {
            for (i, pred) in predictions.iter_mut().enumerate() {
                *pred += exog[i];
            }
        }

        Ok(Forecast::from_values(predictions))
    }
}

impl Default for ARIMA {
    fn default() -> Self {
        Self::arima_111()
    }
}

impl Forecaster for ARIMA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();
        let min_len = self.spec.d + self.spec.p.max(self.spec.q) + 2;

        if values.len() < min_len {
            return Err(ForecastError::InsufficientData {
                needed: min_len,
                got: values.len(),
                hint: Some(format!(
                    "ARIMA({},{},{}) requires d + max(p,q) + 2 = {} observations",
                    self.spec.p, self.spec.d, self.spec.q, min_len
                )),
            });
        }

        self.n = values.len();

        // Check for exogenous regressors
        let adjusted_values = if series.has_regressors() {
            // Extract regressors from TimeSeries
            let regressors = series.all_regressors();

            // Fit OLS: y ~ X
            let ols_result = ols_fit(values, &regressors)?;

            // Calculate residuals (y - OLS prediction)
            let adjusted = ols_residuals(values, &ols_result, &regressors)?;

            // Store OLS result for prediction
            self.exog_ols = Some(ols_result);

            adjusted
        } else {
            self.exog_ols = None;
            values.to_vec()
        };

        self.original = Some(adjusted_values.clone());

        // Apply differencing
        let diff_series = difference(&adjusted_values, self.spec.d);
        self.differenced = Some(diff_series.clone());

        // Estimate parameters (skip when warm-started with pre-fitted coefficients)
        if !self.skip_optimization {
            self.estimate_parameters(&diff_series);
        }

        // Calculate fitted values and residuals
        self.calculate_fitted(&diff_series);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        // If model was fit with exogenous regressors, require predict_with_exog
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog() and provide future regressor values.".into()
            ));
        }

        self.predict_internal(horizon, None)
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        self.exog_ols.is_some()
    }

    fn exog_names(&self) -> Option<&[String]> {
        self.exog_ols
            .as_ref()
            .map(|ols| ols.regressor_names.as_slice())
    }

    fn exog_coefficients(&self) -> Option<&OLSResult> {
        self.exog_ols.as_ref()
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        self.predict_internal(horizon, Some(future_regressors))
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        let forecast = self.predict_with_exog(horizon, future_regressors)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let cumulative_var = variance * h as f64;
            let se = cumulative_var.sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        // Calculate cumulative variance for each horizon
        for h in 1..=horizon {
            // Simplified variance calculation
            // For ARIMA, the variance grows with horizon
            let cumulative_var = variance * h as f64;
            let se = cumulative_var.sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted_diff.as_deref()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted_diff.as_ref()?;
        let variance = self.residual_variance?;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let sigma = variance.sqrt();

        let lower: Vec<f64> = fitted.iter().map(|&f| f - z * sigma).collect();
        let upper: Vec<f64> = fitted.iter().map(|&f| f + z * sigma).collect();

        Some(Forecast::from_values_with_intervals(
            fitted.clone(),
            lower,
            upper,
        ))
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "ARIMA"
    }

    fn fitted_params(&self) -> Option<FittedParams> {
        if self.ar_coefficients.is_empty()
            && self.ma_coefficients.is_empty()
            && self.intercept == 0.0
            && self.original.is_none()
        {
            return None;
        }
        let mut params = HashMap::new();
        params.insert("p".to_string(), self.spec.p as f64);
        params.insert("d".to_string(), self.spec.d as f64);
        params.insert("q".to_string(), self.spec.q as f64);
        params.insert("intercept".to_string(), self.intercept);
        for (i, &c) in self.ar_coefficients.iter().enumerate() {
            params.insert(format!("ar_{}", i + 1), c);
        }
        for (i, &c) in self.ma_coefficients.iter().enumerate() {
            params.insert(format!("ma_{}", i + 1), c);
        }
        Some(FittedParams {
            params,
            seasonal: None,
        })
    }
}

// ============================================================================
// SARIMA (Seasonal ARIMA)
// ============================================================================

/// SARIMA (Seasonal ARIMA) forecasting model.
///
/// SARIMA(p, d, q)(P, D, Q)\[s\] extends ARIMA with seasonal components:
/// - p, d, q: Non-seasonal orders
/// - P, D, Q: Seasonal orders
/// - s: Seasonal period
///
/// Supports exogenous regressors (SARIMAX) via TimeSeries.regressors.
///
/// # Example
/// ```
/// use anofox_forecast::models::arima::SARIMA;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..100).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
/// let values: Vec<f64> = (0..100).map(|i| {
///     50.0 + 0.5 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
/// }).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(12).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SARIMA {
    /// Model specification.
    spec: SARIMASpec,
    /// Non-seasonal AR coefficients.
    ar_coefficients: Vec<f64>,
    /// Non-seasonal MA coefficients.
    ma_coefficients: Vec<f64>,
    /// Seasonal AR coefficients.
    seasonal_ar_coefficients: Vec<f64>,
    /// Seasonal MA coefficients.
    seasonal_ma_coefficients: Vec<f64>,
    /// Intercept.
    intercept: f64,
    /// Original series.
    original: Option<Vec<f64>>,
    /// Differenced series (both regular and seasonal).
    differenced: Option<Vec<f64>>,
    /// Last values for non-seasonal integration.
    last_values: Vec<f64>,
    /// Last values for seasonal integration (from non-seasonally differenced series).
    seasonal_last_values: Vec<f64>,
    /// Residuals.
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
    residuals: Option<Vec<f64>>,
    /// Last residuals for MA forecasting.
    last_residuals: Vec<f64>,
    /// Last residuals for seasonal MA forecasting.
    seasonal_last_residuals: Vec<f64>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// AIC.
    aic: Option<f64>,
    /// BIC.
    bic: Option<f64>,
    /// Series length.
    n: usize,
    /// OLS result for exogenous regressors (if any).
    #[cfg_attr(feature = "serde", serde(skip))]
    exog_ols: Option<OLSResult>,
}

impl SARIMA {
    /// Create a new SARIMA model.
    ///
    /// # Arguments
    /// * `p` - Non-seasonal AR order
    /// * `d` - Non-seasonal differencing order
    /// * `q` - Non-seasonal MA order
    /// * `cap_p` - Seasonal AR order (P)
    /// * `cap_d` - Seasonal differencing order (D)
    /// * `cap_q` - Seasonal MA order (Q)
    /// * `s` - Seasonal period
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        cap_p: usize,
        cap_d: usize,
        cap_q: usize,
        s: usize,
    ) -> Self {
        Self {
            spec: SARIMASpec::new(p, d, q, cap_p, cap_d, cap_q, s),
            ar_coefficients: vec![],
            ma_coefficients: vec![],
            seasonal_ar_coefficients: vec![],
            seasonal_ma_coefficients: vec![],
            intercept: 0.0,
            original: None,
            differenced: None,
            last_values: vec![],
            seasonal_last_values: vec![],
            residuals: None,
            last_residuals: vec![],
            seasonal_last_residuals: vec![],
            residual_variance: None,
            aic: None,
            bic: None,
            n: 0,
            exog_ols: None,
        }
    }

    /// Create a SARIMA model from specification.
    pub fn from_spec(spec: SARIMASpec) -> Self {
        Self::new(
            spec.p, spec.d, spec.q, spec.cap_p, spec.cap_d, spec.cap_q, spec.s,
        )
    }

    /// Get the model specification.
    pub fn spec(&self) -> SARIMASpec {
        self.spec
    }

    /// Get non-seasonal AR coefficients.
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coefficients
    }

    /// Get non-seasonal MA coefficients.
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coefficients
    }

    /// Get seasonal AR coefficients.
    pub fn seasonal_ar_coefficients(&self) -> &[f64] {
        &self.seasonal_ar_coefficients
    }

    /// Get seasonal MA coefficients.
    pub fn seasonal_ma_coefficients(&self) -> &[f64] {
        &self.seasonal_ma_coefficients
    }

    /// Get the intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get AIC.
    pub fn aic(&self) -> Option<f64> {
        self.aic
    }

    /// Get BIC.
    pub fn bic(&self) -> Option<f64> {
        self.bic
    }

    /// Score-only evaluation: compute AIC or BIC from a pre-computed differenced series
    /// without storing any model state. Used by AutoARIMA to avoid full model construction
    /// during candidate search.
    pub(crate) fn score_order(
        p: usize,
        q: usize,
        cap_p: usize,
        cap_q: usize,
        s: usize,
        diff_series: &[f64],
        use_aic: bool,
    ) -> Option<f64> {
        let max_ar_lag = if cap_p > 0 && s > 1 {
            p + cap_p * s
        } else {
            p.max(cap_p * s)
        };
        let max_ma_lag = if cap_q > 0 && s > 1 {
            q + cap_q * s
        } else {
            q.max(cap_q * s)
        };
        let start = max_ar_lag.max(max_ma_lag);

        if diff_series.len() <= start + 2 {
            return None;
        }

        let n_params = 1 + p + q + cap_p + cap_q;

        if p == 0 && q == 0 && cap_p == 0 && cap_q == 0 {
            // Just intercept model
            let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;
            let n_eff = (diff_series.len() - start) as f64;
            let variance = diff_series[start..]
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / n_eff;
            if variance <= 0.0 || !variance.is_finite() {
                return None;
            }
            let k = 1.0;
            let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());
            let score = if use_aic {
                -2.0 * ll + 2.0 * k
            } else {
                -2.0 * ll + k * n_eff.ln()
            };
            return if score.is_finite() { Some(score) } else { None };
        }

        // Set up optimization
        let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;
        let mut initial = vec![0.0; n_params];
        initial[0] = mean;

        let mut idx = 1;
        for i in 0..p {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }
        idx += p;
        for i in 0..q {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }
        idx += q;
        for i in 0..cap_p {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }
        idx += cap_p;
        for i in 0..cap_q {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }

        let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        for _ in 0..(p + q + cap_p + cap_q) {
            bounds.push((-0.99, 0.99));
        }

        // Use L-BFGS for fast convergence
        let lbfgs_config = LbfgsConfig {
            max_iter: 50,
            tolerance: 1e-6,
            ..Default::default()
        };

        let residuals_buf = std::cell::RefCell::new(vec![0.0; diff_series.len()]);

        let result = lbfgs_optimize(
            |params| {
                let ar_end = 1 + p;
                let ma_end = ar_end + q;
                let sar_end = ma_end + cap_p;
                let sma_end = sar_end + cap_q;

                let mut buf = residuals_buf.borrow_mut();
                Self::calculate_css(
                    diff_series,
                    p,
                    q,
                    cap_p,
                    cap_q,
                    s,
                    &params[1..ar_end],
                    &params[ar_end..ma_end],
                    &params[ma_end..sar_end],
                    &params[sar_end..sma_end],
                    params[0],
                    &mut buf,
                )
            },
            &initial,
            Some(&bounds),
            lbfgs_config,
        );

        // Compute AIC/BIC directly from CSS
        let css = result.optimal_value;
        if !css.is_finite() || css <= 0.0 {
            return None;
        }

        let n_eff = (diff_series.len() - start) as f64;
        let variance = css / n_eff;
        let k = n_params as f64;
        let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());

        let score = if use_aic {
            -2.0 * ll + 2.0 * k
        } else {
            -2.0 * ll + k * n_eff.ln()
        };

        if score.is_finite() {
            Some(score)
        } else {
            None
        }
    }

    /// Apply seasonal differencing.
    pub(crate) fn seasonal_difference(data: &[f64], cap_d: usize, s: usize) -> Vec<f64> {
        if cap_d == 0 || s <= 1 {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        for _ in 0..cap_d {
            if result.len() <= s {
                break;
            }
            let mut temp = Vec::with_capacity(result.len() - s);
            for i in s..result.len() {
                temp.push(result[i] - result[i - s]);
            }
            result = temp;
        }
        result
    }

    /// Apply seasonal integration (reverse of seasonal differencing).
    fn seasonal_integrate(
        forecast: &[f64],
        last_values: &[f64],
        cap_d: usize,
        s: usize,
    ) -> Vec<f64> {
        if cap_d == 0 || s <= 1 {
            return forecast.to_vec();
        }

        let mut result = forecast.to_vec();
        for _ in 0..cap_d {
            let mut integrated = Vec::with_capacity(result.len());
            for (h, &val) in result.iter().enumerate() {
                if h < s {
                    // Use historical values
                    let history_idx = last_values.len().saturating_sub(s) + h;
                    if history_idx < last_values.len() {
                        integrated.push(val + last_values[history_idx]);
                    } else {
                        integrated.push(val);
                    }
                } else {
                    // Use previously integrated values
                    integrated.push(val + integrated[h - s]);
                }
            }
            result = integrated;
        }
        result
    }

    /// Calculate conditional sum of squares for SARIMA.
    ///
    /// Uses multiplicative seasonal formulation where AR and MA polynomials
    /// are multiplied, creating interaction terms at lag (i+1) + (j+1)*s.
    /// Uses a pre-allocated residuals buffer to avoid allocation per call.
    /// The buffer must be at least `diff_series.len()` elements.
    fn calculate_css(
        diff_series: &[f64],
        p: usize,
        q: usize,
        cap_p: usize,
        cap_q: usize,
        s: usize,
        ar: &[f64],
        ma: &[f64],
        sar: &[f64],
        sma: &[f64],
        intercept: f64,
        residuals: &mut [f64],
    ) -> f64 {
        let n = diff_series.len();
        let max_ar_lag = if cap_p > 0 && s > 1 {
            p + cap_p * s
        } else {
            p.max(cap_p * s)
        };
        let max_ma_lag = if cap_q > 0 && s > 1 {
            q + cap_q * s
        } else {
            q.max(cap_q * s)
        };
        let start = max_ar_lag.max(max_ma_lag);

        if n <= start {
            return f64::MAX;
        }

        // Zero out the residuals buffer
        residuals[..n].fill(0.0);
        let mut css = 0.0;

        for t in start..n {
            let mut pred = intercept;

            // Non-seasonal AR terms
            for i in 0..p {
                let lag = i + 1;
                pred += ar[i] * diff_series[t - lag];
            }

            // Seasonal AR terms
            for j in 0..cap_p {
                let lag = (j + 1) * s;
                pred += sar[j] * diff_series[t - lag];
            }

            // AR interaction terms (multiplicative seasonal)
            for i in 0..p {
                for j in 0..cap_p {
                    let lag = (i + 1) + (j + 1) * s;
                    pred -= ar[i] * sar[j] * diff_series[t - lag];
                }
            }

            // Non-seasonal MA terms
            for i in 0..q {
                let lag = i + 1;
                pred += ma[i] * residuals[t - lag];
            }

            // Seasonal MA terms
            for j in 0..cap_q {
                let lag = (j + 1) * s;
                pred += sma[j] * residuals[t - lag];
            }

            // MA interaction terms (multiplicative seasonal)
            for i in 0..q {
                for j in 0..cap_q {
                    let lag = (i + 1) + (j + 1) * s;
                    pred += ma[i] * sma[j] * residuals[t - lag];
                }
            }

            let error = diff_series[t] - pred;
            residuals[t] = error;
            css += error * error;
        }

        css
    }

    /// Estimate SARIMA parameters.
    fn estimate_parameters(&mut self, diff_series: &[f64]) {
        let p = self.spec.p;
        let q = self.spec.q;
        let cap_p = self.spec.cap_p;
        let cap_q = self.spec.cap_q;
        let s = self.spec.s;

        let mean = diff_series.iter().sum::<f64>() / diff_series.len() as f64;

        if p == 0 && q == 0 && cap_p == 0 && cap_q == 0 {
            self.intercept = mean;
            return;
        }

        // Set up optimization
        let n_params = 1 + p + q + cap_p + cap_q;
        let mut initial = vec![0.0; n_params];
        initial[0] = mean;

        // Initialize coefficients
        let mut idx = 1;
        for i in 0..p {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }
        idx += p;
        for i in 0..q {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }
        idx += q;
        for i in 0..cap_p {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }
        idx += cap_p;
        for i in 0..cap_q {
            initial[idx + i] = 0.1 / (i + 1) as f64;
        }

        // Set up bounds
        let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)]; // intercept
        for _ in 0..(p + q + cap_p + cap_q) {
            bounds.push((-0.99, 0.99));
        }

        let config = NelderMeadConfig {
            max_iter: 2000,
            tolerance: 1e-8,
            ..Default::default()
        };

        // Pre-allocate residuals buffer, shared via RefCell since nelder_mead takes Fn (not FnMut)
        let residuals_buf = std::cell::RefCell::new(vec![0.0; diff_series.len()]);

        let result = nelder_mead(
            |params| {
                let ar_end = 1 + p;
                let ma_end = ar_end + q;
                let sar_end = ma_end + cap_p;
                let sma_end = sar_end + cap_q;

                let mut buf = residuals_buf.borrow_mut();
                Self::calculate_css(
                    diff_series,
                    p,
                    q,
                    cap_p,
                    cap_q,
                    s,
                    &params[1..ar_end],
                    &params[ar_end..ma_end],
                    &params[ma_end..sar_end],
                    &params[sar_end..sma_end],
                    params[0],
                    &mut buf,
                )
            },
            &initial,
            Some(&bounds),
            config,
        );

        // Extract optimized parameters
        self.intercept = result.optimal_point[0];
        let mut idx = 1;
        self.ar_coefficients = result.optimal_point[idx..idx + p].to_vec();
        idx += p;
        self.ma_coefficients = result.optimal_point[idx..idx + q].to_vec();
        idx += q;
        self.seasonal_ar_coefficients = result.optimal_point[idx..idx + cap_p].to_vec();
        idx += cap_p;
        self.seasonal_ma_coefficients = result.optimal_point[idx..idx + cap_q].to_vec();
    }

    /// Calculate fitted values and residuals.
    fn calculate_fitted(&mut self, diff_series: &[f64]) {
        let n = diff_series.len();
        let p = self.spec.p;
        let q = self.spec.q;
        let cap_p = self.spec.cap_p;
        let cap_q = self.spec.cap_q;
        let s = self.spec.s;

        // Account for interaction terms
        let max_ar_lag = if cap_p > 0 && s > 1 {
            p + cap_p * s
        } else {
            p.max(cap_p * s)
        };
        let max_ma_lag = if cap_q > 0 && s > 1 {
            q + cap_q * s
        } else {
            q.max(cap_q * s)
        };
        let start = max_ar_lag.max(max_ma_lag);

        let mut fitted = vec![f64::NAN; n];
        let mut residuals = vec![0.0; n];

        for t in start..n {
            let mut pred = self.intercept;

            // Non-seasonal AR terms
            for i in 0..p {
                let lag = i + 1;
                if t >= lag {
                    pred += self.ar_coefficients[i] * diff_series[t - lag];
                }
            }

            // Seasonal AR terms
            for j in 0..cap_p {
                let lag = (j + 1) * s;
                if t >= lag {
                    pred += self.seasonal_ar_coefficients[j] * diff_series[t - lag];
                }
            }

            // AR interaction terms
            for i in 0..p {
                for j in 0..cap_p {
                    let lag = (i + 1) + (j + 1) * s;
                    if t >= lag {
                        pred -= self.ar_coefficients[i]
                            * self.seasonal_ar_coefficients[j]
                            * diff_series[t - lag];
                    }
                }
            }

            // Non-seasonal MA terms
            for i in 0..q {
                let lag = i + 1;
                if t >= lag {
                    pred += self.ma_coefficients[i] * residuals[t - lag];
                }
            }

            // Seasonal MA terms
            for j in 0..cap_q {
                let lag = (j + 1) * s;
                if t >= lag {
                    pred += self.seasonal_ma_coefficients[j] * residuals[t - lag];
                }
            }

            // MA interaction terms
            for i in 0..q {
                for j in 0..cap_q {
                    let lag = (i + 1) + (j + 1) * s;
                    if t >= lag {
                        pred += self.ma_coefficients[i]
                            * self.seasonal_ma_coefficients[j]
                            * residuals[t - lag];
                    }
                }
            }

            fitted[t] = pred;
            residuals[t] = diff_series[t] - pred;
        }

        // Store last residuals for forecasting (need enough for interaction terms)
        // For MA interaction, max lag is q + Q*s
        let max_ma_history = if cap_q > 0 && s > 1 {
            q + cap_q * s
        } else {
            q.max(cap_q * s)
        };
        if max_ma_history > 0 {
            let retain = max_ma_history.min(residuals.len());
            self.last_residuals = residuals[residuals.len() - retain..].to_vec();
            self.seasonal_last_residuals = self.last_residuals.clone();
        }

        // Calculate residual variance
        let valid_residuals: Vec<f64> = residuals[start..].to_vec();
        if !valid_residuals.is_empty() {
            let variance =
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);

            // Calculate information criteria
            let n_eff = valid_residuals.len() as f64;
            let k = self.spec.num_params() as f64;
            let ll = -0.5 * n_eff * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());

            self.aic = Some(-2.0 * ll + 2.0 * k);
            self.bic = Some(-2.0 * ll + k * n_eff.ln());
        }

        self.residuals = Some(residuals);
    }

    /// Internal prediction method that handles both with and without exogenous cases.
    fn predict_internal(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<Forecast> {
        let original = self
            .original
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let diff_series = self
            .differenced
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Calculate exogenous contribution if applicable
        let exog_contribution = if let Some(ols) = &self.exog_ols {
            let future = future_regressors.ok_or_else(|| {
                ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors. Future regressor values required."
                        .into(),
                )
            })?;

            // Validate future regressors have correct length
            for name in &ols.regressor_names {
                let values = future.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }

            // Predict exogenous contribution
            Some(ols.predict(future)?)
        } else {
            if future_regressors.is_some_and(|r| !r.is_empty()) {
                return Err(ForecastError::InvalidParameter(
                    "Model was not fit with exogenous regressors".into(),
                ));
            }
            None
        };

        let p = self.spec.p;
        let q = self.spec.q;
        let cap_p = self.spec.cap_p;
        let cap_q = self.spec.cap_q;
        let s = self.spec.s;
        let d = self.spec.d;
        let cap_d = self.spec.cap_d;

        // Forecast on differenced scale — clone then reserve to avoid reallocation
        let diff_len = diff_series.len();
        let mut extended_diff = diff_series.to_vec();
        extended_diff.reserve(horizon);
        let mut extended_residuals = if cap_q > 0 && s > 1 {
            self.seasonal_last_residuals.clone()
        } else {
            self.last_residuals.clone()
        };
        extended_residuals.reserve(horizon);

        // Ensure we have enough history
        if let Some(residuals) = &self.residuals {
            if extended_residuals.len() < cap_q * s {
                let need = (cap_q * s).saturating_sub(extended_residuals.len());
                let start = residuals
                    .len()
                    .saturating_sub(need + extended_residuals.len());
                let mut prefix =
                    residuals[start..residuals.len() - extended_residuals.len()].to_vec();
                prefix.append(&mut extended_residuals);
                extended_residuals = prefix;
            }
        }

        for _ in 0..horizon {
            let t = extended_diff.len();
            let mut pred = self.intercept;

            // Non-seasonal AR terms
            for i in 0..p {
                let lag = i + 1;
                if t >= lag {
                    pred += self.ar_coefficients[i] * extended_diff[t - lag];
                }
            }

            // Seasonal AR terms
            for j in 0..cap_p {
                let lag = (j + 1) * s;
                if t >= lag {
                    pred += self.seasonal_ar_coefficients[j] * extended_diff[t - lag];
                }
            }

            // AR interaction terms
            for i in 0..p {
                for j in 0..cap_p {
                    let lag = (i + 1) + (j + 1) * s;
                    if t >= lag {
                        pred -= self.ar_coefficients[i]
                            * self.seasonal_ar_coefficients[j]
                            * extended_diff[t - lag];
                    }
                }
            }

            // Non-seasonal MA terms (future residuals are 0)
            let res_len = extended_residuals.len();
            for i in 0..q {
                let lag = i + 1;
                if res_len >= lag {
                    pred += self.ma_coefficients[i] * extended_residuals[res_len - lag];
                }
            }

            // Seasonal MA terms
            for j in 0..cap_q {
                let lag = (j + 1) * s;
                if res_len >= lag {
                    pred += self.seasonal_ma_coefficients[j] * extended_residuals[res_len - lag];
                }
            }

            // MA interaction terms
            for i in 0..q {
                for j in 0..cap_q {
                    let lag = (i + 1) + (j + 1) * s;
                    if res_len >= lag {
                        pred += self.ma_coefficients[i]
                            * self.seasonal_ma_coefficients[j]
                            * extended_residuals[res_len - lag];
                    }
                }
            }

            extended_diff.push(pred);
            extended_residuals.push(0.0); // Future residuals are 0
        }

        // Extract forecast on differenced scale
        let forecast_diff: Vec<f64> = extended_diff[diff_len..].to_vec();

        // Apply integration (seasonal first, then non-seasonal - reverse of differencing)
        let mut result = forecast_diff;
        if cap_d > 0 && s > 1 {
            result = Self::seasonal_integrate(&result, &self.seasonal_last_values, cap_d, s);
        }
        if d > 0 {
            result = integrate(&result, original, d);
        }

        // Add exogenous contribution
        if let Some(exog) = exog_contribution {
            for (i, pred) in result.iter_mut().enumerate() {
                *pred += exog[i];
            }
        }

        Ok(Forecast::from_values(result))
    }
}

impl Default for SARIMA {
    fn default() -> Self {
        Self::new(1, 1, 1, 0, 0, 0, 1)
    }
}

impl Forecaster for SARIMA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();
        let d = self.spec.d;
        let cap_d = self.spec.cap_d;
        let s = self.spec.s;
        let p = self.spec.p;
        let q = self.spec.q;
        let cap_p = self.spec.cap_p;
        let cap_q = self.spec.cap_q;

        // Calculate minimum required data
        let seasonal_lag = if s > 1 { cap_p.max(cap_q) * s } else { 0 };
        let min_len = d + cap_d * s + p.max(q).max(seasonal_lag) + 2;

        if values.len() < min_len {
            return Err(ForecastError::InsufficientData {
                needed: min_len,
                got: values.len(),
                hint: Some(format!(
                    "SARIMA({},{},{})({},{},{})_{} requires at least {} observations",
                    p, d, q, cap_p, cap_d, cap_q, s, min_len
                )),
            });
        }

        self.n = values.len();

        // Check for exogenous regressors
        let adjusted_values = if series.has_regressors() {
            // Extract regressors from TimeSeries
            let regressors = series.all_regressors();

            // Fit OLS: y ~ X
            let ols_result = ols_fit(values, &regressors)?;

            // Calculate residuals (y - OLS prediction)
            let adjusted = ols_residuals(values, &ols_result, &regressors)?;

            // Store OLS result for prediction
            self.exog_ols = Some(ols_result);

            adjusted
        } else {
            self.exog_ols = None;
            values.to_vec()
        };

        self.original = Some(adjusted_values.clone());

        // Store last values for non-seasonal integration
        if d > 0 {
            self.last_values = vec![0.0]; // placeholder
            let mut current = adjusted_values.clone();
            self.last_values.push(*current.last().unwrap_or(&0.0));

            for _diff_level in 1..d {
                current = difference(&current, 1);
                if !current.is_empty() {
                    self.last_values.push(*current.last().unwrap_or(&0.0));
                }
            }
        }

        // Apply non-seasonal differencing first
        let nonseasonal_diff = difference(&adjusted_values, d);

        // Store values for seasonal integration (from non-seasonally differenced series)
        if cap_d > 0 && s > 1 {
            let retain = cap_d * s + s;
            let start = nonseasonal_diff.len().saturating_sub(retain);
            self.seasonal_last_values = nonseasonal_diff[start..].to_vec();
        }

        // Apply seasonal differencing
        let diff_series = Self::seasonal_difference(&nonseasonal_diff, cap_d, s);

        if diff_series.is_empty() {
            return Err(ForecastError::InsufficientData {
                needed: min_len,
                got: values.len(),
                hint: Some("series too short after seasonal differencing".into()),
            });
        }

        self.differenced = Some(diff_series.clone());

        // Estimate parameters
        self.estimate_parameters(&diff_series);

        // Calculate fitted values and residuals
        self.calculate_fitted(&diff_series);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        // If model was fit with exogenous regressors, require predict_with_exog
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog() and provide future regressor values.".into()
            ));
        }

        self.predict_internal(horizon, None)
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        self.exog_ols.is_some()
    }

    fn exog_names(&self) -> Option<&[String]> {
        self.exog_ols
            .as_ref()
            .map(|ols| ols.regressor_names.as_slice())
    }

    fn exog_coefficients(&self) -> Option<&OLSResult> {
        self.exog_ols.as_ref()
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        self.predict_internal(horizon, Some(future_regressors))
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        let forecast = self.predict_with_exog(horizon, future_regressors)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let cumulative_var = variance * (1.0 + 0.1 * h as f64);
            let se = cumulative_var.sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let cumulative_var = variance * (1.0 + 0.1 * h as f64);
            let se = cumulative_var.sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        None // Fitted values are on differenced scale
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        if self.spec.is_seasonal() {
            "SARIMA"
        } else {
            "ARIMA"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    #[test]
    fn arima_basic_fit() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        assert!(model.ar_coefficients().len() == 1);
        assert!(model.ma_coefficients().len() == 1);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_ar1() {
        let timestamps = make_timestamps(100);
        // Generate AR(1) process: y_t = 0.7 * y_{t-1} + e_t
        let mut values = vec![10.0];
        for i in 1..100 {
            values.push(0.7 * values[i - 1] + (i as f64 * 0.1).sin());
        }
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::ar(1);
        model.fit(&ts).unwrap();

        // AR coefficient should be close to 0.7
        assert!(model.ar_coefficients()[0] > 0.3);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_ma1() {
        let timestamps = make_timestamps(100);
        // Simple series for MA testing
        let values: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.2).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::ma(1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_011_random_walk() {
        let timestamps = make_timestamps(50);
        // Random walk-like series
        let mut values = vec![10.0];
        for i in 1..50 {
            values.push(values[i - 1] + 0.5 + (i as f64 * 0.1).sin() * 0.1);
        }
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(0, 1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn arima_with_differencing() {
        let timestamps = make_timestamps(50);
        // Strong trend
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = ARIMA::new(1, 1, 0);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // Forecast should continue the trend
        assert!(preds[0] > values.last().unwrap() - 5.0);
    }

    #[test]
    fn arima_confidence_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        // Intervals should exist and be reasonable
        for i in 0..5 {
            assert!(lower[i].is_finite());
            assert!(upper[i].is_finite());
            assert!(upper[i] >= lower[i]);
        }
    }

    #[test]
    fn arima_information_criteria() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 0, 1);
        model.fit(&ts).unwrap();

        assert!(model.aic().is_some());
        assert!(model.bic().is_some());
    }

    #[test]
    fn arima_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(2, 1, 1);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn arima_requires_fit() {
        let model = ARIMA::new(1, 1, 1);
        assert!(matches!(
            model.predict(5),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn arima_zero_horizon() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn arima_spec() {
        let spec = ARIMASpec::new(2, 1, 3);
        assert_eq!(spec.p, 2);
        assert_eq!(spec.d, 1);
        assert_eq!(spec.q, 3);
        assert_eq!(spec.num_params(), 6); // 2 AR + 3 MA + 1 intercept
    }

    #[test]
    fn arima_default() {
        let model = ARIMA::default();
        assert_eq!(model.spec().p, 1);
        assert_eq!(model.spec().d, 1);
        assert_eq!(model.spec().q, 1);
    }

    #[test]
    fn arima_name() {
        let model = ARIMA::new(1, 1, 1);
        assert_eq!(model.name(), "ARIMA");
    }

    #[test]
    fn arima_getters() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        assert!(!model.ar_coefficients().is_empty());
        assert!(!model.ma_coefficients().is_empty());
        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
    }

    // SARIMA tests
    #[test]
    fn sarima_basic() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                50.0 + 0.5 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn sarima_non_seasonal() {
        // SARIMA with no seasonal components should work like ARIMA
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SARIMA::new(1, 1, 1, 0, 0, 0, 12);
        model.fit(&ts).unwrap();

        assert_eq!(model.name(), "ARIMA"); // Not seasonal

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn sarima_seasonal_only() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SARIMA::new(0, 0, 0, 1, 1, 1, 12);
        model.fit(&ts).unwrap();

        assert_eq!(model.name(), "SARIMA");

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn sarima_confidence_intervals() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                50.0 + 0.5 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SARIMA::new(1, 1, 1, 1, 0, 1, 12);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(12, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn sarima_spec() {
        let spec = SARIMASpec::new(1, 1, 1, 2, 1, 2, 12);
        assert_eq!(spec.p, 1);
        assert_eq!(spec.d, 1);
        assert_eq!(spec.q, 1);
        assert_eq!(spec.cap_p, 2);
        assert_eq!(spec.cap_d, 1);
        assert_eq!(spec.cap_q, 2);
        assert_eq!(spec.s, 12);
        assert!(spec.is_seasonal());
        assert_eq!(spec.num_params(), 7); // 1 AR + 1 MA + 2 SAR + 2 SMA + 1 intercept
    }

    #[test]
    fn sarima_insufficient_data() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn sarima_requires_fit() {
        let model = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
        assert!(matches!(
            model.predict(5),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    // =========================================================================
    // Warm-start tests
    // =========================================================================

    #[test]
    fn arima_extract_params_then_warm_start() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64) * 0.3).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Fit original model
        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();
        let forecast1 = model.predict(5).unwrap();

        // Extract params and warm-start
        let fp = model.fitted_params().unwrap();
        let ar_1 = fp.params["ar_1"];
        let ma_1 = fp.params["ma_1"];
        let intercept = fp.params["intercept"];

        let mut warm = ARIMA::with_coefficients(1, 1, 1, vec![ar_1], vec![ma_1], intercept);
        warm.fit(&ts).unwrap(); // ARIMA needs fit for differencing context
        let forecast2 = warm.predict(5).unwrap();

        // Forecasts should be identical since same coefficients are used
        for (a, b) in forecast1.primary().iter().zip(forecast2.primary().iter()) {
            assert!((a - b).abs() < 1e-6, "predictions differ: {} vs {}", a, b);
        }
    }

    #[test]
    fn arima_warm_start_fit_uses_provided_coefficients() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64) * 0.3).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Create warm-started ARIMA with specific coefficients
        let mut model = ARIMA::with_coefficients(1, 1, 0, vec![0.5], vec![], 0.3);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        for &v in forecast.primary() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn arima_fitted_params_returns_none_before_fit() {
        let model = ARIMA::new(1, 1, 1);
        assert!(model.fitted_params().is_none());
    }

    #[test]
    fn arima_fitted_params_contains_expected_keys() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64) * 0.3).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 1);
        model.fit(&ts).unwrap();

        let fp = model.fitted_params().unwrap();
        assert!(fp.params.contains_key("p"));
        assert!(fp.params.contains_key("d"));
        assert!(fp.params.contains_key("q"));
        assert!(fp.params.contains_key("intercept"));
        assert!(fp.params.contains_key("ar_1"));
        assert!(fp.params.contains_key("ma_1"));
        assert!(fp.seasonal.is_none());
    }
}
