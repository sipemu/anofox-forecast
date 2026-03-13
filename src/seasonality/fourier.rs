//! Prophet-style Fourier seasonality modeling.
//!
//! This module provides flexible seasonality modeling using Fourier terms,
//! as popularized by Facebook Prophet. Seasonal patterns are represented as
//! a sum of sine and cosine terms at different frequencies, which can model
//! arbitrary periodic patterns.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::FourierSeasonality;
//!
//! // Create a weekly seasonality model with 3 Fourier terms.
//! // Period = 7.0 when timestamps are in fractional days.
//! let mut model = FourierSeasonality::new(7.0, 3).unwrap();
//!
//! // Timestamps as fractional days from some epoch
//! let timestamps: Vec<f64> = (0..28).map(|i| i as f64).collect();
//! let values: Vec<f64> = timestamps
//!     .iter()
//!     .map(|&t| (2.0 * std::f64::consts::PI * t / 7.0).sin() * 5.0)
//!     .collect();
//!
//! model.fit(&timestamps, &values).unwrap();
//! let predicted = model.predict(&timestamps).unwrap();
//! ```

use crate::error::{ForecastError, Result};
use std::f64::consts::PI;

/// Seconds in one day (86400).
const SECONDS_PER_DAY: f64 = 86_400.0;

/// Generate Fourier basis vectors (sin/cos pairs) for given timestamps and period.
///
/// For each Fourier order `k` in `1..=order`, two basis vectors are produced:
/// - `sin(2 * pi * k * t / period)`
/// - `cos(2 * pi * k * t / period)`
///
/// Returns `2 * order` vectors, each of length `timestamps.len()`.
/// The vectors are ordered as: [sin_1, cos_1, sin_2, cos_2, ...].
pub fn fourier_terms(timestamps: &[f64], period: f64, order: usize) -> Result<Vec<Vec<f64>>> {
    if timestamps.is_empty() {
        return Err(ForecastError::EmptyData);
    }
    if period <= 0.0 || !period.is_finite() {
        return Err(ForecastError::InvalidParameter(
            "period must be a positive finite number".to_string(),
        ));
    }
    if order == 0 {
        return Err(ForecastError::InvalidParameter(
            "order must be at least 1".to_string(),
        ));
    }

    let n = timestamps.len();
    let mut basis = Vec::with_capacity(2 * order);

    for k in 1..=order {
        let mut sin_vec = Vec::with_capacity(n);
        let mut cos_vec = Vec::with_capacity(n);
        let freq = 2.0 * PI * k as f64 / period;

        for &t in timestamps {
            let angle = freq * t;
            sin_vec.push(angle.sin());
            cos_vec.push(angle.cos());
        }

        basis.push(sin_vec);
        basis.push(cos_vec);
    }

    Ok(basis)
}

/// Prophet-style Fourier seasonality model.
///
/// Models a seasonal component as a weighted sum of sine and cosine terms
/// at harmonics of a fundamental frequency. The number of terms (`order`)
/// controls the flexibility: higher orders can capture sharper seasonal patterns
/// but may overfit with limited data.
#[derive(Debug, Clone)]
pub struct FourierSeasonality {
    /// The seasonal period (in the same time units as the input timestamps).
    period: f64,
    /// The number of Fourier pairs (sin + cos). Total basis functions = 2 * order.
    order: usize,
    /// Fitted coefficients, one per basis function (length = 2 * order).
    coefficients: Option<Vec<f64>>,
}

impl FourierSeasonality {
    /// Create a new Fourier seasonality model with the specified period and order.
    ///
    /// - `period`: The length of one seasonal cycle (same units as timestamps).
    /// - `order`: The number of Fourier pairs. Higher values capture more detail.
    pub fn new(period: f64, order: usize) -> Result<Self> {
        if period <= 0.0 || !period.is_finite() {
            return Err(ForecastError::InvalidParameter(
                "period must be a positive finite number".to_string(),
            ));
        }
        if order == 0 {
            return Err(ForecastError::InvalidParameter(
                "order must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            period,
            order,
            coefficients: None,
        })
    }

    /// Create a daily seasonality model (period = 1 day = 86400 seconds).
    ///
    /// Use this when timestamps are in seconds (e.g., Unix timestamps).
    /// For timestamps already in fractional days, use `new(1.0, order)` instead.
    pub fn daily(order: usize) -> Result<Self> {
        Self::new(SECONDS_PER_DAY, order)
    }

    /// Create a weekly seasonality model (period = 7 days = 604800 seconds).
    ///
    /// Use this when timestamps are in seconds (e.g., Unix timestamps).
    /// For timestamps already in fractional days, use `new(7.0, order)` instead.
    pub fn weekly(order: usize) -> Result<Self> {
        Self::new(7.0 * SECONDS_PER_DAY, order)
    }

    /// Create a yearly seasonality model (period = 365.25 days in seconds).
    ///
    /// Use this when timestamps are in seconds (e.g., Unix timestamps).
    /// For timestamps already in fractional days, use `new(365.25, order)` instead.
    pub fn yearly(order: usize) -> Result<Self> {
        Self::new(365.25 * SECONDS_PER_DAY, order)
    }

    /// Get the period of this seasonality model.
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Get the Fourier order of this seasonality model.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the fitted coefficients, if the model has been fitted.
    pub fn coefficients(&self) -> Option<&[f64]> {
        self.coefficients.as_deref()
    }

    /// Fit the Fourier seasonality model to the given data.
    ///
    /// Uses the normal equations (X^T X) beta = X^T y to solve for coefficients
    /// via Cholesky-like decomposition. No external dependencies are added.
    ///
    /// - `timestamps`: Time values (e.g., fractional days or Unix timestamps).
    /// - `values`: The observed values corresponding to each timestamp.
    pub fn fit(&mut self, timestamps: &[f64], values: &[f64]) -> Result<()> {
        if timestamps.is_empty() || values.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        if timestamps.len() != values.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: timestamps.len(),
                got: values.len(),
            });
        }

        let n = timestamps.len();
        let p = 2 * self.order;

        if n < p {
            return Err(ForecastError::InsufficientData {
                needed: p,
                got: n,
                hint: Some(format!(
                    "need at least {} data points for {} Fourier terms",
                    p, p
                )),
            });
        }

        // Build the design matrix columns (Fourier basis)
        let basis = fourier_terms(timestamps, self.period, self.order)?;

        // Solve normal equations: (X^T X) beta = X^T y
        // Build X^T X (p x p symmetric matrix, stored as flat Vec)
        let mut xtx = vec![0.0; p * p];
        for i in 0..p {
            for j in i..p {
                let dot: f64 = basis[i]
                    .iter()
                    .zip(basis[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                xtx[i * p + j] = dot;
                xtx[j * p + i] = dot;
            }
        }

        // Build X^T y (p-vector)
        let mut xty = vec![0.0; p];
        for i in 0..p {
            xty[i] = basis[i].iter().zip(values.iter()).map(|(a, b)| a * b).sum();
        }

        // Solve via Cholesky decomposition: X^T X = L L^T
        let l = cholesky_decompose(&xtx, p)?;
        let coefficients = cholesky_solve(&l, &xty, p);

        self.coefficients = Some(coefficients);
        Ok(())
    }

    /// Predict the seasonal component for the given timestamps.
    ///
    /// The model must be fitted first via [`fit`](Self::fit).
    pub fn predict(&self, timestamps: &[f64]) -> Result<Vec<f64>> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;

        if timestamps.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let basis = fourier_terms(timestamps, self.period, self.order)?;
        let n = timestamps.len();
        let mut result = vec![0.0; n];

        for (j, coeff) in coefficients.iter().enumerate() {
            for i in 0..n {
                result[i] += coeff * basis[j][i];
            }
        }

        Ok(result)
    }
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that A = L L^T.
/// The input `a` is a row-major flat array of size `n * n`.
fn cholesky_decompose(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }

            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return Err(ForecastError::SingularMatrix(
                        "Fourier basis matrix is singular or nearly singular; \
                         try reducing the order or ensuring varied timestamps"
                            .to_string(),
                    ));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }

    Ok(l)
}

/// Solve L L^T x = b given the Cholesky factor L.
///
/// First solves L z = b (forward substitution), then L^T x = z (back substitution).
fn cholesky_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Forward substitution: L z = b
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * n + j] * z[j];
        }
        z[i] = (b[i] - sum) / l[i * n + i];
    }

    // Back substitution: L^T x = z
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j];
        }
        x[i] = (z[i] - sum) / l[i * n + i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    fn assert_approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} ≈ {}, diff = {}",
            a,
            b,
            (a - b).abs()
        );
    }

    // ── fourier_terms tests ──────────────────────────────────────────────

    #[test]
    fn fourier_terms_basic_shape() {
        let ts: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let basis = fourier_terms(&ts, 7.0, 3).unwrap();
        assert_eq!(basis.len(), 6); // 2 * order
        for col in &basis {
            assert_eq!(col.len(), 10);
        }
    }

    #[test]
    fn fourier_terms_values_at_zero() {
        let ts = vec![0.0];
        let basis = fourier_terms(&ts, 10.0, 2).unwrap();
        // sin(0) = 0, cos(0) = 1 for all harmonics
        assert_approx_eq(basis[0][0], 0.0, TOLERANCE); // sin_1
        assert_approx_eq(basis[1][0], 1.0, TOLERANCE); // cos_1
        assert_approx_eq(basis[2][0], 0.0, TOLERANCE); // sin_2
        assert_approx_eq(basis[3][0], 1.0, TOLERANCE); // cos_2
    }

    #[test]
    fn fourier_terms_periodicity() {
        // Values at t and t + period should be the same
        let period = 7.0;
        let ts = vec![1.5, 1.5 + period, 1.5 + 2.0 * period];
        let basis = fourier_terms(&ts, period, 4).unwrap();
        for col in &basis {
            assert_approx_eq(col[0], col[1], TOLERANCE);
            assert_approx_eq(col[0], col[2], TOLERANCE);
        }
    }

    #[test]
    fn fourier_terms_orthogonality() {
        // Over a full period, different harmonics should be approximately orthogonal
        let period = 100.0;
        let n = 1000;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * period / n as f64).collect();
        let basis = fourier_terms(&ts, period, 3).unwrap();

        // sin_1 and cos_1 should be nearly orthogonal
        let dot: f64 = basis[0]
            .iter()
            .zip(basis[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_approx_eq(dot / n as f64, 0.0, 0.01);

        // sin_1 and sin_2 should be nearly orthogonal
        let dot: f64 = basis[0]
            .iter()
            .zip(basis[2].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_approx_eq(dot / n as f64, 0.0, 0.01);
    }

    #[test]
    fn fourier_terms_empty_timestamps() {
        let result = fourier_terms(&[], 7.0, 3);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn fourier_terms_invalid_period() {
        let ts = vec![1.0, 2.0];
        assert!(fourier_terms(&ts, 0.0, 1).is_err());
        assert!(fourier_terms(&ts, -5.0, 1).is_err());
        assert!(fourier_terms(&ts, f64::NAN, 1).is_err());
        assert!(fourier_terms(&ts, f64::INFINITY, 1).is_err());
    }

    #[test]
    fn fourier_terms_invalid_order() {
        let ts = vec![1.0, 2.0];
        assert!(fourier_terms(&ts, 7.0, 0).is_err());
    }

    // ── FourierSeasonality construction tests ────────────────────────────

    #[test]
    fn new_valid_parameters() {
        let fs = FourierSeasonality::new(7.0, 3).unwrap();
        assert_approx_eq(fs.period(), 7.0, TOLERANCE);
        assert_eq!(fs.order(), 3);
        assert!(fs.coefficients().is_none());
    }

    #[test]
    fn new_invalid_period() {
        assert!(FourierSeasonality::new(0.0, 3).is_err());
        assert!(FourierSeasonality::new(-1.0, 3).is_err());
        assert!(FourierSeasonality::new(f64::NAN, 3).is_err());
    }

    #[test]
    fn new_invalid_order() {
        assert!(FourierSeasonality::new(7.0, 0).is_err());
    }

    #[test]
    fn preset_daily() {
        let fs = FourierSeasonality::daily(4).unwrap();
        assert_approx_eq(fs.period(), SECONDS_PER_DAY, TOLERANCE);
        assert_eq!(fs.order(), 4);
    }

    #[test]
    fn preset_weekly() {
        let fs = FourierSeasonality::weekly(3).unwrap();
        assert_approx_eq(fs.period(), 7.0 * SECONDS_PER_DAY, TOLERANCE);
        assert_eq!(fs.order(), 3);
    }

    #[test]
    fn preset_yearly() {
        let fs = FourierSeasonality::yearly(10).unwrap();
        assert_approx_eq(fs.period(), 365.25 * SECONDS_PER_DAY, TOLERANCE);
        assert_eq!(fs.order(), 10);
    }

    // ── Fit and predict tests ────────────────────────────────────────────

    #[test]
    fn fit_recovers_single_sinusoid() {
        let period = 7.0;
        let n = 100;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * period / n as f64).collect();
        // y = 3 * sin(2*pi*t/7)
        let values: Vec<f64> = ts
            .iter()
            .map(|&t| 3.0 * (2.0 * PI * t / period).sin())
            .collect();

        let mut model = FourierSeasonality::new(period, 3).unwrap();
        model.fit(&ts, &values).unwrap();

        let coeffs = model.coefficients().unwrap();
        assert_eq!(coeffs.len(), 6);
        // First coefficient (sin_1) should be ~3.0
        assert_approx_eq(coeffs[0], 3.0, 0.01);
        // cos_1 should be ~0.0
        assert_approx_eq(coeffs[1], 0.0, 0.01);
        // Higher harmonics should be ~0.0
        for &c in &coeffs[2..] {
            assert_approx_eq(c, 0.0, 0.01);
        }
    }

    #[test]
    fn fit_recovers_mixed_harmonics() {
        let period = 10.0;
        let n = 200;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * period / n as f64).collect();
        // y = 2*sin(2*pi*t/10) + 1.5*cos(4*pi*t/10)
        let values: Vec<f64> = ts
            .iter()
            .map(|&t| 2.0 * (2.0 * PI * t / period).sin() + 1.5 * (4.0 * PI * t / period).cos())
            .collect();

        let mut model = FourierSeasonality::new(period, 3).unwrap();
        model.fit(&ts, &values).unwrap();

        let coeffs = model.coefficients().unwrap();
        assert_approx_eq(coeffs[0], 2.0, 0.01); // sin_1
        assert_approx_eq(coeffs[1], 0.0, 0.01); // cos_1
        assert_approx_eq(coeffs[2], 0.0, 0.01); // sin_2
        assert_approx_eq(coeffs[3], 1.5, 0.01); // cos_2
        assert_approx_eq(coeffs[4], 0.0, 0.01); // sin_3
        assert_approx_eq(coeffs[5], 0.0, 0.01); // cos_3
    }

    #[test]
    fn predict_matches_original() {
        let period = 7.0;
        let n = 50;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let values: Vec<f64> = ts
            .iter()
            .map(|&t| 2.0 * (2.0 * PI * t / period).sin() - 1.0 * (2.0 * PI * t / period).cos())
            .collect();

        let mut model = FourierSeasonality::new(period, 3).unwrap();
        model.fit(&ts, &values).unwrap();

        let predicted = model.predict(&ts).unwrap();
        assert_eq!(predicted.len(), n);
        for i in 0..n {
            assert_approx_eq(predicted[i], values[i], 0.05);
        }
    }

    #[test]
    fn predict_on_new_timestamps() {
        let period = 7.0;
        let n = 100;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = ts
            .iter()
            .map(|&t| 4.0 * (2.0 * PI * t / period).sin())
            .collect();

        let mut model = FourierSeasonality::new(period, 2).unwrap();
        model.fit(&ts, &values).unwrap();

        // Predict on shifted timestamps (out-of-sample)
        let new_ts: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 0.1).collect();
        let predicted = model.predict(&new_ts).unwrap();
        assert_eq!(predicted.len(), 20);

        for (i, &t) in new_ts.iter().enumerate() {
            let expected = 4.0 * (2.0 * PI * t / period).sin();
            assert_approx_eq(predicted[i], expected, 0.1);
        }
    }

    #[test]
    fn predict_before_fit_errors() {
        let model = FourierSeasonality::new(7.0, 3).unwrap();
        let ts = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            model.predict(&ts),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn fit_empty_data() {
        let mut model = FourierSeasonality::new(7.0, 3).unwrap();
        assert!(matches!(model.fit(&[], &[]), Err(ForecastError::EmptyData)));
    }

    #[test]
    fn fit_mismatched_lengths() {
        let mut model = FourierSeasonality::new(7.0, 3).unwrap();
        let ts = vec![1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0];
        assert!(matches!(
            model.fit(&ts, &values),
            Err(ForecastError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_insufficient_data() {
        let mut model = FourierSeasonality::new(7.0, 3).unwrap();
        // Need at least 6 data points for order=3
        let ts = vec![1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            model.fit(&ts, &values),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn predict_empty_timestamps() {
        let period = 7.0;
        let ts: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let values: Vec<f64> = ts.iter().map(|&t| (2.0 * PI * t / period).sin()).collect();

        let mut model = FourierSeasonality::new(period, 2).unwrap();
        model.fit(&ts, &values).unwrap();

        assert!(matches!(model.predict(&[]), Err(ForecastError::EmptyData)));
    }

    // ── Cholesky tests ───────────────────────────────────────────────────

    #[test]
    fn cholesky_2x2() {
        // A = [[4, 2], [2, 3]]  => L = [[2, 0], [1, sqrt(2)]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky_decompose(&a, 2).unwrap();
        assert_approx_eq(l[0], 2.0, TOLERANCE);
        assert_approx_eq(l[1], 0.0, TOLERANCE);
        assert_approx_eq(l[2], 1.0, TOLERANCE);
        assert_approx_eq(l[3], 2.0_f64.sqrt(), TOLERANCE);
    }

    #[test]
    fn cholesky_solve_2x2() {
        // Solve [[4,2],[2,3]] x = [8, 7]  => x = [1, 1+ 2/3] roughly
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky_decompose(&a, 2).unwrap();
        let b = vec![8.0, 7.0];
        let x = cholesky_solve(&l, &b, 2);
        // Verify A*x = b
        let r0 = 4.0 * x[0] + 2.0 * x[1];
        let r1 = 2.0 * x[0] + 3.0 * x[1];
        assert_approx_eq(r0, 8.0, TOLERANCE);
        assert_approx_eq(r1, 7.0, TOLERANCE);
    }

    #[test]
    fn cholesky_singular_matrix() {
        // Not positive definite
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_decompose(&a, 2).is_err());
    }

    // ── Round-trip and clone tests ───────────────────────────────────────

    #[test]
    fn model_is_cloneable() {
        let period = 7.0;
        let ts: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let values: Vec<f64> = ts.iter().map(|&t| (2.0 * PI * t / period).sin()).collect();

        let mut model = FourierSeasonality::new(period, 2).unwrap();
        model.fit(&ts, &values).unwrap();

        let clone = model.clone();
        let pred1 = model.predict(&ts).unwrap();
        let pred2 = clone.predict(&ts).unwrap();
        for (a, b) in pred1.iter().zip(pred2.iter()) {
            assert_approx_eq(*a, *b, TOLERANCE);
        }
    }

    #[test]
    fn fit_can_be_called_multiple_times() {
        let period = 7.0;
        let ts: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let values1: Vec<f64> = ts
            .iter()
            .map(|&t| 2.0 * (2.0 * PI * t / period).sin())
            .collect();
        let values2: Vec<f64> = ts
            .iter()
            .map(|&t| 5.0 * (2.0 * PI * t / period).cos())
            .collect();

        let mut model = FourierSeasonality::new(period, 2).unwrap();

        model.fit(&ts, &values1).unwrap();
        let pred1 = model.predict(&ts).unwrap();

        model.fit(&ts, &values2).unwrap();
        let pred2 = model.predict(&ts).unwrap();

        // Second fit should produce different predictions
        let diff: f64 = pred1
            .iter()
            .zip(pred2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0, "re-fitting should change predictions");
    }

    #[test]
    fn multiple_periods_combined() {
        // Simulate data with both a weekly and a yearly component,
        // fit two separate models, combine predictions.
        let n = 400;
        let weekly_period = 7.0;
        let yearly_period = 365.25;
        let ts: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let weekly_component: Vec<f64> = ts
            .iter()
            .map(|&t| 3.0 * (2.0 * PI * t / weekly_period).sin())
            .collect();
        let yearly_component: Vec<f64> = ts
            .iter()
            .map(|&t| 2.0 * (2.0 * PI * t / yearly_period).cos())
            .collect();
        let values: Vec<f64> = weekly_component
            .iter()
            .zip(yearly_component.iter())
            .map(|(w, y)| w + y)
            .collect();

        let mut weekly_model = FourierSeasonality::new(weekly_period, 3).unwrap();
        weekly_model.fit(&ts, &values).unwrap();
        let weekly_pred = weekly_model.predict(&ts).unwrap();

        // The weekly model should capture the weekly part well
        // (yearly part gets averaged out over many cycles)
        let weekly_rmse: f64 = weekly_pred
            .iter()
            .zip(weekly_component.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n as f64;
        let weekly_rmse = weekly_rmse.sqrt();
        // The RMSE should be moderate (yearly is aliased/partially captured)
        assert!(weekly_rmse < 3.0, "weekly RMSE {} is too high", weekly_rmse);
    }

    #[test]
    fn high_order_captures_sharp_pattern() {
        // A square-wave-like pattern needs higher harmonics
        let period = 10.0;
        let n = 200;
        let ts: Vec<f64> = (0..n).map(|i| i as f64 * period / n as f64).collect();
        let values: Vec<f64> = ts
            .iter()
            .map(|&t| {
                if (t % period) < period / 2.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        let mut low_order = FourierSeasonality::new(period, 1).unwrap();
        low_order.fit(&ts, &values).unwrap();
        let pred_low = low_order.predict(&ts).unwrap();

        let mut high_order = FourierSeasonality::new(period, 10).unwrap();
        high_order.fit(&ts, &values).unwrap();
        let pred_high = high_order.predict(&ts).unwrap();

        let rmse_low: f64 = pred_low
            .iter()
            .zip(values.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n as f64;
        let rmse_high: f64 = pred_high
            .iter()
            .zip(values.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n as f64;

        assert!(
            rmse_high < rmse_low,
            "higher order should fit better: rmse_high={} vs rmse_low={}",
            rmse_high,
            rmse_low
        );
    }
}
