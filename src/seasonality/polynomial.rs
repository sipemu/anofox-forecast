//! Polynomial trend modeling via least squares regression.
//!
//! Fits a polynomial of degree 1 (linear), 2 (quadratic), or 3 (cubic)
//! to the recency window of the series using normal equations solved by
//! Cholesky decomposition. No external linear algebra dependencies are
//! required.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::polynomial::PolynomialTrend;
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! // Fit a quadratic trend to y = t^2
//! let values: Vec<f64> = (0..30).map(|t| (t as f64).powi(2)).collect();
//! let mut trend = PolynomialTrend::new(2);
//! trend.fit_trend(&values).unwrap();
//!
//! let fitted = trend.fitted_trend();
//! assert_eq!(fitted.len(), 30);
//!
//! let forecast = trend.predict_trend(5);
//! assert_eq!(forecast.len(), 5);
//! // forecast[0] should approximate 30^2 = 900
//! assert!((forecast[0] - 900.0).abs() < 1.0);
//! ```

use super::traits::{Recency, TrendComponent};
use crate::error::{ForecastError, Result};

/// Polynomial trend component.
///
/// Fits a polynomial of the specified degree (1-3) to the recency window of the
/// input series using the normal equations solved via Cholesky decomposition.
/// Fitted values cover the entire series, with values outside the recency window
/// obtained by evaluating the polynomial at those indices (backwards
/// extrapolation).
#[derive(Debug, Clone)]
pub struct PolynomialTrend {
    /// Polynomial degree: 1 (linear), 2 (quadratic), or 3 (cubic).
    degree: usize,
    /// Controls which portion of data is used for fitting.
    recency: Recency,
    /// Fitted polynomial coefficients `[c0, c1, ..., c_degree]` such that
    /// `y(t) = c0 + c1*t + c2*t^2 + ...`.
    coefficients: Option<Vec<f64>>,
    /// Fitted trend values (same length as training data).
    fitted: Vec<f64>,
    /// Length of training data.
    n_train: usize,
    /// R-squared of the fit (computed at fit time over the recency window).
    r_squared: f64,
}

impl PolynomialTrend {
    /// Create a new polynomial trend with the given degree.
    ///
    /// The degree is clamped to the range 1..=3. The default recency is
    /// `Recency::Fraction(0.3)`.
    pub fn new(degree: usize) -> Self {
        Self {
            degree: degree.clamp(1, 3),
            recency: Recency::Fraction(0.3),
            coefficients: None,
            fitted: Vec::new(),
            n_train: 0,
            r_squared: 0.0,
        }
    }

    /// Set the recency window for fitting (builder-style).
    pub fn with_recency(mut self, recency: Recency) -> Self {
        self.recency = recency;
        self
    }

    /// Return the fitted polynomial coefficients, if available.
    ///
    /// The coefficients are ordered `[c0, c1, ..., c_degree]` so that
    /// `y(t) = c0 + c1*t + c2*t^2 + ...`.
    pub fn coefficients(&self) -> Option<&[f64]> {
        self.coefficients.as_deref()
    }
}

/// Evaluate a polynomial at index `t` given coefficients `[c0, c1, ..., c_d]`.
#[inline]
fn poly_eval(coeffs: &[f64], t: f64) -> f64 {
    // Horner's method (reverse order since coeffs[0] is the constant term).
    let mut val = 0.0;
    for c in coeffs.iter().rev() {
        val = val * t + c;
    }
    val
}

/// Build the Vandermonde matrix row for index `t` with the given degree.
///
/// Returns `[1, t, t^2, ..., t^degree]`.
#[inline]
fn vandermonde_row(t: f64, degree: usize) -> Vec<f64> {
    let mut row = Vec::with_capacity(degree + 1);
    let mut power = 1.0;
    for _ in 0..=degree {
        row.push(power);
        power *= t;
    }
    row
}

/// Solve the system `A * x = b` where `A` is symmetric positive definite,
/// using Cholesky decomposition (LL^T).
///
/// `a` is stored as a flat row-major `p x p` matrix.
/// Returns `Err` if the matrix is not positive definite (singular or
/// near-singular).
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>> {
    // Cholesky decomposition: A = L * L^T
    let mut l = vec![0.0; p * p];

    for i in 0..p {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * p + k] * l[j * p + k];
            }
            if i == j {
                let diag = a[i * p + i] - sum;
                if diag <= 1e-14 {
                    return Err(ForecastError::SingularMatrix(
                        "Cholesky decomposition failed: matrix not positive definite".into(),
                    ));
                }
                l[i * p + j] = diag.sqrt();
            } else {
                l[i * p + j] = (a[i * p + j] - sum) / l[j * p + j];
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = vec![0.0; p];
    for i in 0..p {
        let mut sum = 0.0;
        for k in 0..i {
            sum += l[i * p + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * p + i];
    }

    // Back substitution: L^T * x = y
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = 0.0;
        for k in (i + 1)..p {
            sum += l[k * p + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * p + i];
    }

    Ok(x)
}

impl TrendComponent for PolynomialTrend {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let n = values.len();

        if n == 1 {
            self.coefficients = Some(vec![values[0]]);
            self.fitted = vec![values[0]];
            self.n_train = 1;
            self.r_squared = 1.0;
            return Ok(());
        }

        let (rec_start, rec_end) = self.recency.resolve_with_data(values);
        let window_len = rec_end - rec_start;

        // Clamp effective degree to at most window_len - 1 so the system is
        // not underdetermined.
        let eff_degree = self.degree.min(window_len.saturating_sub(1));
        let p = eff_degree + 1; // number of parameters

        // Build X'X and X'y from the recency window.
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];

        for idx in rec_start..rec_end {
            let t = idx as f64;
            let row = vandermonde_row(t, eff_degree);
            let y = values[idx];

            for i in 0..p {
                xty[i] += row[i] * y;
                for j in 0..p {
                    xtx[i * p + j] += row[i] * row[j];
                }
            }
        }

        let coeffs = cholesky_solve(&xtx, &xty, p)?;

        // Compute fitted values for ALL indices 0..n.
        let fitted: Vec<f64> = (0..n).map(|i| poly_eval(&coeffs, i as f64)).collect();

        // Compute R-squared over the recency window.
        let window_vals = &values[rec_start..rec_end];
        let window_fitted: Vec<f64> = (rec_start..rec_end)
            .map(|i| poly_eval(&coeffs, i as f64))
            .collect();

        let mean = window_vals.iter().sum::<f64>() / window_len as f64;
        let ss_tot: f64 = window_vals.iter().map(|&v| (v - mean).powi(2)).sum();
        let ss_res: f64 = window_vals
            .iter()
            .zip(window_fitted.iter())
            .map(|(&v, &f)| (v - f).powi(2))
            .sum();

        self.r_squared = if ss_tot < 1e-12 {
            if ss_res < 1e-12 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ss_res / ss_tot
        };

        // Pad coefficients vector to full degree+1 length if effective degree
        // was reduced (fill higher-order terms with 0).
        let mut full_coeffs = coeffs;
        full_coeffs.resize(self.degree + 1, 0.0);

        self.coefficients = Some(full_coeffs);
        self.fitted = fitted;
        self.n_train = n;

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        let coeffs = match &self.coefficients {
            Some(c) => c,
            None => return vec![f64::NAN; n_ahead],
        };

        (0..n_ahead)
            .map(|i| poly_eval(coeffs, (self.n_train + i) as f64))
            .collect()
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        let coeffs = match &self.coefficients {
            Some(c) => c,
            None => return Vec::new(),
        };

        let slope = if coeffs.len() > 1 { coeffs[1] } else { 0.0 };
        let leading = coeffs[self.degree.min(coeffs.len() - 1)];

        vec![
            ("polynomial_r_squared", self.r_squared),
            ("polynomial_degree", self.degree as f64),
            ("polynomial_leading_coeff", leading),
            ("polynomial_slope", slope),
        ]
    }

    fn trend_name(&self) -> &str {
        "polynomial"
    }

    fn n_params(&self) -> usize {
        self.degree + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Construction and defaults ─────────────────────────────────────

    #[test]
    fn default_recency_is_fraction_03() {
        let trend = PolynomialTrend::new(2);
        assert_eq!(trend.recency, Recency::Fraction(0.3));
    }

    #[test]
    fn degree_clamped_to_1_3() {
        let t0 = PolynomialTrend::new(0);
        assert_eq!(t0.degree, 1);

        let t5 = PolynomialTrend::new(5);
        assert_eq!(t5.degree, 3);

        let t2 = PolynomialTrend::new(2);
        assert_eq!(t2.degree, 2);
    }

    #[test]
    fn n_params_returns_degree_plus_one() {
        assert_eq!(PolynomialTrend::new(1).n_params(), 2);
        assert_eq!(PolynomialTrend::new(2).n_params(), 3);
        assert_eq!(PolynomialTrend::new(3).n_params(), 4);
    }

    #[test]
    fn trend_name_is_polynomial() {
        assert_eq!(PolynomialTrend::new(1).trend_name(), "polynomial");
    }

    // ── Linear data (degree 1) ───────────────────────────────────────

    #[test]
    fn linear_fit_coefficients() {
        // y = 3*t + 5
        let values: Vec<f64> = (0..50).map(|t| 3.0 * t as f64 + 5.0).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        assert_abs_diff_eq!(coeffs[0], 5.0, epsilon = 1e-8);
        assert_abs_diff_eq!(coeffs[1], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn linear_fit_fitted_values() {
        let values: Vec<f64> = (0..30).map(|t| 2.0 * t as f64 + 1.0).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 30);
        for (i, (&f, &v)) in fitted.iter().zip(values.iter()).enumerate() {
            assert_abs_diff_eq!(f, v, epsilon = 1e-8);
            let _ = i;
        }
    }

    #[test]
    fn linear_fit_r_squared_near_one() {
        let values: Vec<f64> = (0..40).map(|t| 7.0 * t as f64 - 3.0).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();
        let r2 = features
            .iter()
            .find(|(n, _)| *n == "polynomial_r_squared")
            .unwrap()
            .1;
        assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-8);
    }

    // ── Quadratic data (degree 2) ────────────────────────────────────

    #[test]
    fn quadratic_fit_coefficients() {
        // y = t^2 + 2*t + 1
        let values: Vec<f64> = (0..50)
            .map(|t| {
                let t = t as f64;
                t * t + 2.0 * t + 1.0
            })
            .collect();
        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(coeffs[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn quadratic_fit_r_squared() {
        let values: Vec<f64> = (0..50)
            .map(|t| {
                let t = t as f64;
                t * t + 2.0 * t + 1.0
            })
            .collect();
        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();
        let r2 = features
            .iter()
            .find(|(n, _)| *n == "polynomial_r_squared")
            .unwrap()
            .1;
        assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-8);
    }

    // ── Predict extrapolates correctly ───────────────────────────────

    #[test]
    fn predict_linear_extrapolation() {
        // y = 2*t + 1, n=20 -> predict at t=20,21,...,24
        let values: Vec<f64> = (0..20).map(|t| 2.0 * t as f64 + 1.0).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        for (j, &f) in forecast.iter().enumerate() {
            let expected = 2.0 * (20 + j) as f64 + 1.0;
            assert_abs_diff_eq!(f, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn predict_quadratic_extrapolation() {
        // y = t^2 + 2*t + 1
        let values: Vec<f64> = (0..30)
            .map(|t| {
                let t = t as f64;
                t * t + 2.0 * t + 1.0
            })
            .collect();
        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(5);
        for (j, &f) in forecast.iter().enumerate() {
            let t = (30 + j) as f64;
            let expected = t * t + 2.0 * t + 1.0;
            assert_abs_diff_eq!(f, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn predict_unfitted_returns_nan() {
        let trend = PolynomialTrend::new(2);
        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        assert!(forecast[0].is_nan());
    }

    #[test]
    fn predict_zero_ahead() {
        let values: Vec<f64> = (0..20).map(|t| t as f64).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(0);
        assert!(forecast.is_empty());
    }

    // ── Recency window ───────────────────────────────────────────────

    #[test]
    fn recency_window_fits_on_subset() {
        // First half constant at 0, second half linear y = t.
        // With recency window on last 50%, the polynomial should capture the
        // linear portion and extrapolate backwards.
        let n = 40;
        let values: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Fraction(0.5));
        trend.fit_trend(&values).unwrap();

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), n);

        // Fitted values should still cover the full series.
        // The recency window starts around index 20, so the polynomial was fit
        // on indices 20..40. For pure linear data, this should match everywhere.
        for (i, &f) in fitted.iter().enumerate() {
            assert_abs_diff_eq!(f, i as f64, epsilon = 1e-6);
        }
    }

    #[test]
    fn recency_window_returns_full_fitted() {
        let values: Vec<f64> = (0..100).map(|t| (t as f64).powi(2)).collect();
        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Window(30));
        trend.fit_trend(&values).unwrap();

        // Fitted values must cover the entire series.
        assert_eq!(trend.fitted_trend().len(), 100);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn fit_empty_data_error() {
        let mut trend = PolynomialTrend::new(1);
        let result = trend.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn fit_single_point() {
        let mut trend = PolynomialTrend::new(2);
        trend.fit_trend(&[42.0]).unwrap();

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 1);
        assert_abs_diff_eq!(fitted[0], 42.0, epsilon = 1e-10);

        // Coefficients: just the constant term stored.
        let coeffs = trend.coefficients().unwrap();
        assert_abs_diff_eq!(coeffs[0], 42.0, epsilon = 1e-10);

        // Predict should extrapolate the constant.
        let forecast = trend.predict_trend(3);
        for &f in &forecast {
            assert_abs_diff_eq!(f, 42.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn fit_two_points_linear() {
        // Two points: y(0)=1, y(1)=3 -> slope=2, intercept=1
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&[1.0, 3.0]).unwrap();

        let coeffs = trend.coefficients().unwrap();
        assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(coeffs[1], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn constant_data() {
        let values = vec![5.0; 30];
        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        assert_abs_diff_eq!(coeffs[0], 5.0, epsilon = 1e-8);
        assert_abs_diff_eq!(coeffs[1], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(coeffs[2], 0.0, epsilon = 1e-8);

        let forecast = trend.predict_trend(5);
        for &f in &forecast {
            assert_abs_diff_eq!(f, 5.0, epsilon = 1e-6);
        }
    }

    // ── Features extraction ──────────────────────────────────────────

    #[test]
    fn features_extraction() {
        // y = t^2 + 2*t + 1
        let values: Vec<f64> = (0..50)
            .map(|t| {
                let t = t as f64;
                t * t + 2.0 * t + 1.0
            })
            .collect();
        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();

        let get = |name: &str| -> f64 {
            features
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("feature '{}' not found", name))
        };

        assert_abs_diff_eq!(get("polynomial_r_squared"), 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(get("polynomial_degree"), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(get("polynomial_leading_coeff"), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(get("polynomial_slope"), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn features_before_fit_empty() {
        let trend = PolynomialTrend::new(1);
        assert!(trend.trend_features().is_empty());
    }

    #[test]
    fn features_linear_leading_coeff_is_slope() {
        // For degree 1, leading coeff = coefficients[1] = slope
        let values: Vec<f64> = (0..30).map(|t| 4.0 * t as f64 + 7.0).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();
        let leading = features
            .iter()
            .find(|(n, _)| *n == "polynomial_leading_coeff")
            .unwrap()
            .1;
        let slope = features
            .iter()
            .find(|(n, _)| *n == "polynomial_slope")
            .unwrap()
            .1;
        assert_abs_diff_eq!(leading, 4.0, epsilon = 1e-8);
        assert_abs_diff_eq!(slope, 4.0, epsilon = 1e-8);
    }

    // ── Cubic (degree 3) ─────────────────────────────────────────────

    #[test]
    fn cubic_fit() {
        // y = t^3 - 2*t^2 + t + 3
        let values: Vec<f64> = (0..40)
            .map(|t| {
                let t = t as f64;
                t.powi(3) - 2.0 * t.powi(2) + t + 3.0
            })
            .collect();
        let mut trend = PolynomialTrend::new(3).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        assert_abs_diff_eq!(coeffs[0], 3.0, epsilon = 1e-4);
        assert_abs_diff_eq!(coeffs[1], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(coeffs[2], -2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(coeffs[3], 1.0, epsilon = 1e-4);

        assert_eq!(trend.n_params(), 4);
    }

    // ── with_recency builder ─────────────────────────────────────────

    #[test]
    fn with_recency_builder() {
        let trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        assert_eq!(trend.recency, Recency::Full);

        let trend = PolynomialTrend::new(1).with_recency(Recency::Window(50));
        assert_eq!(trend.recency, Recency::Window(50));
    }

    // ── coefficients accessor ────────────────────────────────────────

    #[test]
    fn coefficients_none_before_fit() {
        let trend = PolynomialTrend::new(2);
        assert!(trend.coefficients().is_none());
    }

    #[test]
    fn coefficients_some_after_fit() {
        let values: Vec<f64> = (0..20).map(|t| t as f64).collect();
        let mut trend = PolynomialTrend::new(1).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();
        assert!(trend.coefficients().is_some());
        assert_eq!(trend.coefficients().unwrap().len(), 2);
    }
}
