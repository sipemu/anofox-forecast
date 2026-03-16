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
    /// Fitted polynomial coefficients in raw index space `[c0, c1, ..., c_degree]`
    /// such that `y(t) = c0 + c1*t + c2*t^2 + ...`.
    coefficients: Option<Vec<f64>>,
    /// Fitted coefficients in the centered/scaled basis (for prediction).
    norm_coefficients: Option<Vec<f64>>,
    /// Center of the normalization transform: `u = (t - t_center) / t_scale`.
    t_center: f64,
    /// Scale of the normalization transform.
    t_scale: f64,
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
            norm_coefficients: None,
            t_center: 0.0,
            t_scale: 1.0,
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

/// Back-transform polynomial coefficients from normalized space `u = (t - center) / scale`
/// to raw index space `t`.
///
/// Given `y(u) = a0 + a1*u + a2*u^2 + a3*u^3`, express as `y(t) = b0 + b1*t + b2*t^2 + b3*t^3`
/// using the binomial expansion of `((t - center) / scale)^k`.
fn denormalize_coefficients(norm_coeffs: &[f64], center: f64, scale: f64) -> Vec<f64> {
    let d = norm_coeffs.len(); // degree + 1
    let mut raw = vec![0.0; d];

    // For each normalized coefficient a_k (of u^k),
    // expand u^k = ((t - center)/scale)^k using binomial theorem:
    //   u^k = Σ_{j=0}^{k} C(k,j) * (t/scale)^j * (-center/scale)^(k-j)
    // Then a_k * u^k contributes a_k * C(k,j) * (-center)^(k-j) / scale^k to raw[j].
    for k in 0..d {
        let a_k = norm_coeffs[k];
        let inv_scale_k = 1.0 / scale.powi(k as i32);
        let mut binom = 1.0; // C(k, j), starting at C(k, 0) = 1
        for j in 0..=k {
            let neg_center_pow = (-center).powi((k - j) as i32);
            raw[j] += a_k * binom * neg_center_pow * inv_scale_k;
            // Update binomial coefficient: C(k, j+1) = C(k, j) * (k - j) / (j + 1)
            if j < k {
                binom *= (k - j) as f64 / (j + 1) as f64;
            }
        }
    }

    raw
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
            self.norm_coefficients = Some(vec![values[0]]);
            self.t_center = 0.0;
            self.t_scale = 1.0;
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

        // Center and scale the time index to [-1, 1] over the recency window.
        // This dramatically improves the conditioning of the Vandermonde matrix
        // for higher degrees and large indices.
        let t_center = (rec_start + rec_end - 1) as f64 / 2.0;
        let t_scale = if window_len > 1 {
            (window_len - 1) as f64 / 2.0
        } else {
            1.0
        };

        // Build X'X and X'y from the recency window using normalized indices.
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];

        for idx in rec_start..rec_end {
            let u = (idx as f64 - t_center) / t_scale;
            let row = vandermonde_row(u, eff_degree);
            let y = values[idx];

            for i in 0..p {
                xty[i] += row[i] * y;
                for j in 0..p {
                    xtx[i * p + j] += row[i] * row[j];
                }
            }
        }

        let norm_coeffs = cholesky_solve(&xtx, &xty, p)?;

        // Compute fitted values for ALL indices 0..n using normalized indices.
        let fitted: Vec<f64> = (0..n)
            .map(|i| {
                let u = (i as f64 - t_center) / t_scale;
                poly_eval(&norm_coeffs, u)
            })
            .collect();

        // Compute R-squared over the recency window.
        let window_vals = &values[rec_start..rec_end];
        let window_fitted = &fitted[rec_start..rec_end];

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

        // Pad normalized coefficients to full degree+1 length if effective
        // degree was reduced (fill higher-order terms with 0).
        let mut full_norm = norm_coeffs;
        full_norm.resize(self.degree + 1, 0.0);

        // Back-transform to raw index space for the public accessor.
        let raw_coeffs = denormalize_coefficients(&full_norm, t_center, t_scale);

        self.norm_coefficients = Some(full_norm);
        self.coefficients = Some(raw_coeffs);
        self.t_center = t_center;
        self.t_scale = t_scale;
        self.fitted = fitted;
        self.n_train = n;

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        let norm_coeffs = match &self.norm_coefficients {
            Some(c) => c,
            None => return vec![f64::NAN; n_ahead],
        };

        (0..n_ahead)
            .map(|i| {
                let u = ((self.n_train + i) as f64 - self.t_center) / self.t_scale;
                poly_eval(norm_coeffs, u)
            })
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

    // ── Numerical conditioning ────────────────────────────────────────

    #[test]
    fn cubic_large_series_coefficient_accuracy() {
        // y = 0.001*t^3 - 0.5*t^2 + 3*t + 100
        // With n=500 and raw indices, the Vandermonde Gram matrix
        // has entries up to Σt^6 ≈ 10^16, stressing double-precision.
        let n = 500;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                let t = t as f64;
                0.001 * t.powi(3) - 0.5 * t.powi(2) + 3.0 * t + 100.0
            })
            .collect();

        let mut trend = PolynomialTrend::new(3).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        // Check that coefficients are recovered with reasonable accuracy
        let c0_err = (coeffs[0] - 100.0).abs();
        let c1_err = (coeffs[1] - 3.0).abs();
        let c2_err = (coeffs[2] - (-0.5)).abs();
        let c3_err = (coeffs[3] - 0.001).abs();

        eprintln!("n={n}, cubic with raw indices:");
        eprintln!("  c0={:.10} (expected 100.0,  err={c0_err:.2e})", coeffs[0]);
        eprintln!("  c1={:.10} (expected 3.0,    err={c1_err:.2e})", coeffs[1]);
        eprintln!("  c2={:.10} (expected -0.5,   err={c2_err:.2e})", coeffs[2]);
        eprintln!("  c3={:.10} (expected 0.001,  err={c3_err:.2e})", coeffs[3]);

        assert!(c0_err < 1.0, "c0 error {c0_err:.2e} too large");
        assert!(c1_err < 0.01, "c1 error {c1_err:.2e} too large");
        assert!(c2_err < 1e-4, "c2 error {c2_err:.2e} too large");
        assert!(c3_err < 1e-6, "c3 error {c3_err:.2e} too large");
    }

    #[test]
    fn cubic_recency_window_high_offset() {
        // Recency window at indices 700..1000: raw t^6 reaches ~10^17.
        // This is the worst case for numerical conditioning.
        let n = 1000;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                let t = t as f64;
                0.001 * t.powi(3) - 0.5 * t.powi(2) + 3.0 * t + 100.0
            })
            .collect();

        let mut trend = PolynomialTrend::new(3).with_recency(Recency::Fraction(0.3));
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        let c0_err = (coeffs[0] - 100.0).abs();
        let c1_err = (coeffs[1] - 3.0).abs();
        let c2_err = (coeffs[2] - (-0.5)).abs();
        let c3_err = (coeffs[3] - 0.001).abs();

        eprintln!("n={n}, recency=0.3 (indices ~700..1000):");
        eprintln!("  c0={:.10} (expected 100.0,  err={c0_err:.2e})", coeffs[0]);
        eprintln!("  c1={:.10} (expected 3.0,    err={c1_err:.2e})", coeffs[1]);
        eprintln!("  c2={:.10} (expected -0.5,   err={c2_err:.2e})", coeffs[2]);
        eprintln!("  c3={:.10} (expected 0.001,  err={c3_err:.2e})", coeffs[3]);

        assert!(c0_err < 1.0, "c0 error {c0_err:.2e} too large");
        assert!(c1_err < 0.01, "c1 error {c1_err:.2e} too large");
        assert!(c2_err < 1e-4, "c2 error {c2_err:.2e} too large");
        assert!(c3_err < 1e-6, "c3 error {c3_err:.2e} too large");
    }

    #[test]
    fn cubic_very_large_series_recency() {
        // n=5000 with recency=0.3 → fitting on indices ~3500..5000.
        // t^6 reaches ~10^22, severely stressing conditioning.
        let n = 5000;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                let t = t as f64;
                0.001 * t.powi(3) - 0.5 * t.powi(2) + 3.0 * t + 100.0
            })
            .collect();

        let mut trend = PolynomialTrend::new(3).with_recency(Recency::Fraction(0.3));
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        let c0_err = (coeffs[0] - 100.0).abs();
        let c3_err = (coeffs[3] - 0.001).abs();

        eprintln!("n={n}, recency=0.3 (indices ~3500..5000):");
        eprintln!("  c0={:.6} (expected 100.0,  err={c0_err:.2e})", coeffs[0]);
        eprintln!("  c1={:.6} (expected 3.0,    err={:.2e})", coeffs[1], (coeffs[1] - 3.0).abs());
        eprintln!("  c2={:.6} (expected -0.5,   err={:.2e})", coeffs[2], (coeffs[2] - (-0.5)).abs());
        eprintln!("  c3={:.10} (expected 0.001,  err={c3_err:.2e})", coeffs[3]);

        // Test prediction accuracy (what actually matters for forecasting)
        let pred = trend.predict_trend(10);
        let max_pred_err: f64 = (0..10)
            .map(|i| {
                let t = (n + i) as f64;
                let expected = 0.001 * t.powi(3) - 0.5 * t.powi(2) + 3.0 * t + 100.0;
                (pred[i] - expected).abs() / expected.abs()
            })
            .fold(0.0_f64, f64::max);

        eprintln!("  max relative prediction error: {max_pred_err:.2e}");
    }

    #[test]
    fn quadratic_noisy_data_prediction_accuracy() {
        // Noisy quadratic: y = t^2 + 2t + 1 + noise
        // Validates that the fit produces usable predictions despite noise.
        let n = 200;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                let t = t as f64;
                let signal = t * t + 2.0 * t + 1.0;
                let noise = 50.0 * ((t * 0.7).cos()); // deterministic "noise"
                signal + noise
            })
            .collect();

        let mut trend = PolynomialTrend::new(2).with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let coeffs = trend.coefficients().unwrap();
        eprintln!("n={n}, noisy quadratic (Recency::Full):");
        eprintln!("  c0={:.4} (true 1.0)", coeffs[0]);
        eprintln!("  c1={:.4} (true 2.0)", coeffs[1]);
        eprintln!("  c2={:.6} (true 1.0)", coeffs[2]);
        eprintln!("  R²={:.6}", trend.trend_features().iter().find(|(n, _)| *n == "polynomial_r_squared").unwrap().1);

        // c2 (quadratic term) should be close to 1.0 since noise is bounded
        assert!((coeffs[2] - 1.0).abs() < 0.05, "c2 should be near 1.0, got {}", coeffs[2]);
    }
}
