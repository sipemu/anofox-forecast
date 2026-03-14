//! Hodrick-Prescott filter for trend-cycle decomposition.
//!
//! The HP filter extracts a smooth trend component by minimizing:
//!
//! ```text
//! Σ(y_t - τ_t)² + λ Σ(Δ²τ_t)²
//! ```
//!
//! where `τ` is the trend, `λ` controls smoothness (higher = smoother),
//! and `Δ²` is the second difference operator.
//!
//! This reduces to solving the pentadiagonal linear system `(I + λ K'K) τ = y`
//! where `K` is the second-difference matrix. The matrix is symmetric positive
//! definite with bandwidth 2, so it can be solved in O(n) time.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::hp_filter::HodrickPrescottFilter;
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! let mut hp = HodrickPrescottFilter::quarterly();
//! let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5 + (i as f64 * 0.3).sin()).collect();
//! hp.fit_trend(&data).unwrap();
//! let trend = hp.fitted_trend();
//! let cycle = hp.cycle();
//! let forecast = hp.predict_trend(12);
//! ```

use super::traits::TrendComponent;
use crate::error::{ForecastError, Result};

/// Hodrick-Prescott filter for trend-cycle decomposition.
///
/// Decomposes a time series into a smooth trend component and a cyclical
/// component by solving a penalized least squares problem. The smoothing
/// parameter `λ` controls the trade-off between fit and smoothness.
///
/// Common lambda values:
/// - Annual data: 6.25
/// - Quarterly data: 1600
/// - Monthly data: 129600
#[derive(Debug, Clone)]
pub struct HodrickPrescottFilter {
    /// Smoothing parameter. Common values: 1600 (quarterly), 129600 (monthly), 6.25 (annual).
    lambda: f64,
    /// Fitted trend values.
    fitted: Vec<f64>,
    /// Cycle component (data - trend).
    cycle: Vec<f64>,
}

impl HodrickPrescottFilter {
    /// Create a new Hodrick-Prescott filter with the given smoothing parameter.
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` if `lambda <= 0` or is not finite.
    pub fn new(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 || !lambda.is_finite() {
            return Err(ForecastError::InvalidParameter(
                "lambda must be a positive finite number".to_string(),
            ));
        }
        Ok(Self {
            lambda,
            fitted: Vec::new(),
            cycle: Vec::new(),
        })
    }

    /// Create an HP filter for quarterly data (λ = 1600).
    pub fn quarterly() -> Self {
        Self {
            lambda: 1600.0,
            fitted: Vec::new(),
            cycle: Vec::new(),
        }
    }

    /// Create an HP filter for monthly data (λ = 129600).
    pub fn monthly() -> Self {
        Self {
            lambda: 129600.0,
            fitted: Vec::new(),
            cycle: Vec::new(),
        }
    }

    /// Create an HP filter for annual data (λ = 6.25).
    pub fn annual() -> Self {
        Self {
            lambda: 6.25,
            fitted: Vec::new(),
            cycle: Vec::new(),
        }
    }

    /// Return the cycle component (data minus trend).
    ///
    /// Returns an empty slice if the filter has not been fitted.
    pub fn cycle(&self) -> &[f64] {
        &self.cycle
    }

    /// Solve the pentadiagonal system `(I + λ K'K) τ = y`.
    ///
    /// The matrix `M = I + λ K'K` is symmetric positive definite and pentadiagonal
    /// (bandwidth 2). We solve it using banded Cholesky (LDL^T) factorization,
    /// which runs in O(n) time.
    fn solve_pentadiagonal(y: &[f64], lambda: f64) -> Vec<f64> {
        let n = y.len();

        // For n <= 2, the second-difference matrix K has zero rows,
        // so K'K = 0 and M = I, meaning trend = data.
        if n <= 2 {
            return y.to_vec();
        }

        // Build the diagonals of M = I + λ K'K.
        // M is pentadiagonal with diagonals at offsets -2, -1, 0, +1, +2.
        // We store: d[i] = main diagonal, e[i] = sub-diagonal (offset 1), f[i] = sub-sub-diagonal (offset 2).

        // Main diagonal d[i] = M[i][i]
        let mut d = vec![0.0; n];
        // Sub-diagonal e[i] = M[i+1][i] (length n-1)
        let mut e = vec![0.0; n.saturating_sub(1)];
        // Sub-sub-diagonal f[i] = M[i+2][i] (length n-2)
        let mut f = vec![0.0; n.saturating_sub(2)];

        // Build the matrix entries from K'K.
        // K is (n-2) x n, with rows: K[j] = [0...0, 1, -2, 1, 0...0] starting at column j.
        // (K'K)[i][j] = sum_k K[k][i] * K[k][j]
        //
        // Diagonal (K'K)[i][i]:
        //   i=0:    1           (only K[0] contributes)
        //   i=1:    1+4 = 5     (K[0] and K[1])
        //   2<=i<=n-3: 1+4+1=6  (K[i-2], K[i-1], K[i])
        //   i=n-2:  4+1 = 5
        //   i=n-1:  1
        //
        // Off-diagonal (K'K)[i][i+1]:
        //   i=0:    -2          (only K[0])
        //   1<=i<=n-3: -2-2=-4  (K[i-1] and K[i])
        //   i=n-2:  -2
        // But we only have n-3 valid i values for the general case.
        //
        // Off-diagonal (K'K)[i][i+2]:
        //   0<=i<=n-3: 1        (K[i])

        // Main diagonal
        if n >= 3 {
            d[0] = 1.0 + lambda;
            d[1] = 1.0 + 5.0 * lambda;
            for i in 2..n - 2 {
                d[i] = 1.0 + 6.0 * lambda;
            }
            d[n - 2] = 1.0 + 5.0 * lambda;
            d[n - 1] = 1.0 + lambda;
        }

        // Sub-diagonal (offset 1): e[i] = M[i+1][i]
        if n >= 3 {
            e[0] = -2.0 * lambda;
            for i in 1..n - 2 {
                e[i] = -4.0 * lambda;
            }
            e[n - 2] = -2.0 * lambda;
        }

        // Sub-sub-diagonal (offset 2): f[i] = M[i+2][i]
        for i in 0..n - 2 {
            f[i] = lambda;
        }

        // Solve using banded Cholesky factorization.
        // For a symmetric positive definite banded matrix with bandwidth 2,
        // we compute L such that M = L L^T, where L is lower triangular with
        // bandwidth 2 (i.e., L[i][i], L[i+1][i], L[i+2][i] are the only
        // nonzero entries in column i).
        //
        // We store:
        //   ld[i] = L[i][i]       (main diagonal of L)
        //   le[i] = L[i+1][i]     (sub-diagonal of L)
        //   lf[i] = L[i+2][i]     (sub-sub-diagonal of L)

        let mut ld = vec![0.0; n];
        let mut le = vec![0.0; n.saturating_sub(1)];
        let mut lf = vec![0.0; n.saturating_sub(2)];

        // Column 0
        ld[0] = d[0].sqrt();
        if n > 1 {
            le[0] = e[0] / ld[0];
        }
        if n > 2 {
            lf[0] = f[0] / ld[0];
        }

        // Column 1
        if n > 1 {
            ld[1] = (d[1] - le[0] * le[0]).sqrt();
            if n > 2 {
                le[1] = (e[1] - lf[0] * le[0]) / ld[1];
            }
            if n > 3 {
                lf[1] = f[1] / ld[1];
            }
        }

        // Columns 2..n-1
        for i in 2..n {
            ld[i] = (d[i]
                - le[i - 1] * le[i - 1]
                - lf.get(i.wrapping_sub(2)).copied().unwrap_or(0.0).powi(2))
            .sqrt();
            if i < n - 1 {
                le[i] = (e[i] - lf[i - 1] * le[i - 1]) / ld[i];
            }
            if i < n - 2 {
                lf[i] = f[i] / ld[i];
            }
        }

        // Forward substitution: L z = y
        let mut z = vec![0.0; n];
        z[0] = y[0] / ld[0];
        if n > 1 {
            z[1] = (y[1] - le[0] * z[0]) / ld[1];
        }
        for i in 2..n {
            z[i] = (y[i] - le[i - 1] * z[i - 1] - lf[i - 2] * z[i - 2]) / ld[i];
        }

        // Back substitution: L^T x = z
        let mut x = vec![0.0; n];
        x[n - 1] = z[n - 1] / ld[n - 1];
        if n > 1 {
            x[n - 2] = (z[n - 2] - le[n - 2] * x[n - 1]) / ld[n - 2];
        }
        for i in (0..n.saturating_sub(2)).rev() {
            x[i] = (z[i] - le[i] * x[i + 1] - lf[i] * x[i + 2]) / ld[i];
        }

        x
    }
}

impl TrendComponent for HodrickPrescottFilter {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        self.fitted = Self::solve_pentadiagonal(values, self.lambda);
        self.cycle = values
            .iter()
            .zip(self.fitted.iter())
            .map(|(y, t)| y - t)
            .collect();

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        if self.fitted.len() < 2 || n_ahead == 0 {
            return vec![0.0; n_ahead];
        }

        let n = self.fitted.len();
        let last = self.fitted[n - 1];
        let slope = last - self.fitted[n - 2];

        (1..=n_ahead).map(|i| last + slope * i as f64).collect()
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        if self.fitted.is_empty() || self.cycle.is_empty() {
            return vec![
                ("hp_trend_strength", 0.0),
                ("hp_cycle_variance_ratio", 0.0),
                ("hp_trend_slope", 0.0),
            ];
        }

        let n = self.fitted.len();

        // Variance of data (fitted + cycle = original data)
        let data: Vec<f64> = self
            .fitted
            .iter()
            .zip(self.cycle.iter())
            .map(|(t, c)| t + c)
            .collect();
        let var_data = variance(&data);
        let var_cycle = variance(&self.cycle);

        // hp_trend_strength: 1 - var(cycle) / var(data), clamped to [0, 1]
        let trend_strength = if var_data > 0.0 {
            (1.0 - var_cycle / var_data).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // hp_cycle_variance_ratio: var(cycle) / var(data)
        let cycle_var_ratio = if var_data > 0.0 {
            var_cycle / var_data
        } else {
            0.0
        };

        // hp_trend_slope: linear regression slope of trend on indices
        let trend_slope = if n >= 2 {
            linear_regression_slope(&self.fitted)
        } else {
            0.0
        };

        vec![
            ("hp_trend_strength", trend_strength),
            ("hp_cycle_variance_ratio", cycle_var_ratio),
            ("hp_trend_slope", trend_slope),
        ]
    }

    fn trend_name(&self) -> &str {
        "HodrickPrescottFilter"
    }
}

/// Compute the population variance of a slice.
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Compute the slope of a simple linear regression of `values` on their indices (0, 1, 2, ...).
fn linear_regression_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        num += (x - x_mean) * (y - y_mean);
        den += (x - x_mean) * (x - x_mean);
    }

    if den.abs() < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

/// Compute the HP trend strength: `1 - var(cycle) / var(data)`, clamped to `[0, 1]`.
///
/// This is a standalone convenience function that fits an HP filter internally.
/// Returns 0.0 if the data is empty or has zero variance.
pub fn hp_trend_strength(values: &[f64], lambda: f64) -> f64 {
    if values.is_empty() || lambda <= 0.0 || !lambda.is_finite() {
        return 0.0;
    }

    let trend = HodrickPrescottFilter::solve_pentadiagonal(values, lambda);
    let cycle: Vec<f64> = values
        .iter()
        .zip(trend.iter())
        .map(|(y, t)| y - t)
        .collect();

    let var_data = variance(values);
    let var_cycle = variance(&cycle);

    if var_data > 0.0 {
        (1.0 - var_cycle / var_data).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute the HP cycle variance ratio: `var(cycle) / var(data)`.
///
/// This is a standalone convenience function that fits an HP filter internally.
/// Returns 0.0 if the data is empty or has zero variance.
pub fn hp_cycle_variance_ratio(values: &[f64], lambda: f64) -> f64 {
    if values.is_empty() || lambda <= 0.0 || !lambda.is_finite() {
        return 0.0;
    }

    let trend = HodrickPrescottFilter::solve_pentadiagonal(values, lambda);
    let cycle: Vec<f64> = values
        .iter()
        .zip(trend.iter())
        .map(|(y, t)| y - t)
        .collect();

    let var_data = variance(values);
    let var_cycle = variance(&cycle);

    if var_data > 0.0 {
        var_cycle / var_data
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    // ── Construction tests ──────────────────────────────────────────────

    #[test]
    fn new_valid_lambda() {
        let hp = HodrickPrescottFilter::new(1600.0).unwrap();
        assert_abs_diff_eq!(hp.lambda, 1600.0);
        assert!(hp.fitted.is_empty());
        assert!(hp.cycle.is_empty());
    }

    #[test]
    fn new_invalid_lambda_zero() {
        let result = HodrickPrescottFilter::new(0.0);
        assert!(matches!(result, Err(ForecastError::InvalidParameter(_))));
    }

    #[test]
    fn new_invalid_lambda_negative() {
        let result = HodrickPrescottFilter::new(-1.0);
        assert!(matches!(result, Err(ForecastError::InvalidParameter(_))));
    }

    #[test]
    fn new_invalid_lambda_nan() {
        let result = HodrickPrescottFilter::new(f64::NAN);
        assert!(matches!(result, Err(ForecastError::InvalidParameter(_))));
    }

    #[test]
    fn new_invalid_lambda_infinity() {
        let result = HodrickPrescottFilter::new(f64::INFINITY);
        assert!(matches!(result, Err(ForecastError::InvalidParameter(_))));
    }

    #[test]
    fn preset_quarterly() {
        let hp = HodrickPrescottFilter::quarterly();
        assert_abs_diff_eq!(hp.lambda, 1600.0);
    }

    #[test]
    fn preset_monthly() {
        let hp = HodrickPrescottFilter::monthly();
        assert_abs_diff_eq!(hp.lambda, 129600.0);
    }

    #[test]
    fn preset_annual() {
        let hp = HodrickPrescottFilter::annual();
        assert_abs_diff_eq!(hp.lambda, 6.25);
    }

    // ── Fit error cases ─────────────────────────────────────────────────

    #[test]
    fn fit_empty_data() {
        let mut hp = HodrickPrescottFilter::quarterly();
        let result = hp.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    // ── Constant data ───────────────────────────────────────────────────

    #[test]
    fn constant_data_trend_equals_data() {
        let data = vec![5.0; 50];
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let trend = hp.fitted_trend();
        assert_eq!(trend.len(), 50);
        for &t in trend {
            assert_abs_diff_eq!(t, 5.0, epsilon = 1e-10);
        }

        let cycle = hp.cycle();
        assert_eq!(cycle.len(), 50);
        for &c in cycle {
            assert_abs_diff_eq!(c, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn constant_data_different_lambda() {
        for &lambda in &[1.0, 100.0, 1600.0, 129600.0] {
            let data = vec![42.0; 30];
            let mut hp = HodrickPrescottFilter::new(lambda).unwrap();
            hp.fit_trend(&data).unwrap();
            for &t in hp.fitted_trend() {
                assert_abs_diff_eq!(t, 42.0, epsilon = 1e-8);
            }
        }
    }

    // ── Linear data ─────────────────────────────────────────────────────

    #[test]
    fn linear_data_trend_approx_equals_data() {
        let data: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 3.0).collect();
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let trend = hp.fitted_trend();
        // Interior points should match closely; boundary points may differ slightly.
        for i in 5..95 {
            assert_abs_diff_eq!(trend[i], data[i], epsilon = 1e-4);
        }

        // Cycle should be near zero for interior points.
        let cycle = hp.cycle();
        for i in 5..95 {
            assert_abs_diff_eq!(cycle[i], 0.0, epsilon = 1e-4);
        }
    }

    // ── Sinusoidal data ─────────────────────────────────────────────────

    #[test]
    fn sinusoidal_data_trend_is_smooth() {
        let n = 200;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                0.1 * t + 5.0 * (2.0 * PI * t / 20.0).sin()
            })
            .collect();

        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let trend = hp.fitted_trend();
        let cycle = hp.cycle();

        // Trend should be smoother than original: check that second differences
        // of trend are much smaller than those of the original data.
        let trend_roughness: f64 = (2..n)
            .map(|i| (trend[i] - 2.0 * trend[i - 1] + trend[i - 2]).powi(2))
            .sum();
        let data_roughness: f64 = (2..n)
            .map(|i| (data[i] - 2.0 * data[i - 1] + data[i - 2]).powi(2))
            .sum();
        assert!(
            trend_roughness < data_roughness,
            "trend should be smoother: trend_roughness={}, data_roughness={}",
            trend_roughness,
            data_roughness
        );

        // Cycle should capture the oscillation.
        let var_cycle = variance(cycle);
        assert!(var_cycle > 0.1, "cycle should have nonzero variance");
    }

    // ── Lambda affects smoothness ───────────────────────────────────────

    #[test]
    fn higher_lambda_smoother_trend() {
        let n = 100;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                0.05 * t + 3.0 * (2.0 * PI * t / 10.0).sin()
            })
            .collect();

        let mut hp_low = HodrickPrescottFilter::new(10.0).unwrap();
        let mut hp_high = HodrickPrescottFilter::new(100000.0).unwrap();

        hp_low.fit_trend(&data).unwrap();
        hp_high.fit_trend(&data).unwrap();

        let trend_low = hp_low.fitted_trend();
        let trend_high = hp_high.fitted_trend();

        // Measure roughness (sum of squared second differences).
        let roughness = |trend: &[f64]| -> f64 {
            (2..trend.len())
                .map(|i| (trend[i] - 2.0 * trend[i - 1] + trend[i - 2]).powi(2))
                .sum()
        };

        assert!(
            roughness(trend_high) < roughness(trend_low),
            "higher lambda should produce smoother trend"
        );
    }

    // ── Small inputs ────────────────────────────────────────────────────

    #[test]
    fn single_data_point() {
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&[7.0]).unwrap();
        assert_abs_diff_eq!(hp.fitted_trend()[0], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hp.cycle()[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn two_data_points() {
        let data = vec![3.0, 5.0];
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();
        assert_abs_diff_eq!(hp.fitted_trend()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hp.fitted_trend()[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn three_data_points() {
        let data = vec![1.0, 2.0, 3.0];
        let mut hp = HodrickPrescottFilter::new(1.0).unwrap();
        hp.fit_trend(&data).unwrap();

        // Trend + cycle should reconstruct original data exactly.
        for i in 0..3 {
            assert_abs_diff_eq!(
                hp.fitted_trend()[i] + hp.cycle()[i],
                data[i],
                epsilon = 1e-10
            );
        }

        // Trend should be monotonically increasing for linear input.
        assert!(hp.fitted_trend()[0] <= hp.fitted_trend()[1]);
        assert!(hp.fitted_trend()[1] <= hp.fitted_trend()[2]);
    }

    // ── Predict ─────────────────────────────────────────────────────────

    #[test]
    fn predict_linear_extrapolation() {
        let data: Vec<f64> = (0..50).map(|i| 2.0 * i as f64).collect();
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let forecast = hp.predict_trend(5);
        assert_eq!(forecast.len(), 5);

        let trend = hp.fitted_trend();
        let n = trend.len();
        let slope = trend[n - 1] - trend[n - 2];

        for i in 0..5 {
            let expected = trend[n - 1] + slope * (i + 1) as f64;
            assert_abs_diff_eq!(forecast[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn predict_zero_ahead() {
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&[1.0, 2.0, 3.0]).unwrap();
        let forecast = hp.predict_trend(0);
        assert!(forecast.is_empty());
    }

    #[test]
    fn predict_before_fit() {
        let hp = HodrickPrescottFilter::quarterly();
        let forecast = hp.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        for &v in &forecast {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    // ── Features ────────────────────────────────────────────────────────

    #[test]
    fn features_constant_data() {
        let data = vec![10.0; 50];
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let features = hp.trend_features();
        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();

        // Constant data: no cycle, no trend slope
        // var(data) = 0, so trend_strength = 0 and cycle_variance_ratio = 0
        assert_abs_diff_eq!(map["hp_trend_strength"], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(map["hp_cycle_variance_ratio"], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(map["hp_trend_slope"], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn features_linear_data() {
        let data: Vec<f64> = (0..100).map(|i| 3.0 * i as f64 + 1.0).collect();
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let features = hp.trend_features();
        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();

        // Linear data: trend captures almost everything
        assert!(map["hp_trend_strength"] > 0.99);
        assert!(map["hp_cycle_variance_ratio"] < 0.01);
        // Slope should be close to 3.0
        assert_abs_diff_eq!(map["hp_trend_slope"], 3.0, epsilon = 0.1);
    }

    #[test]
    fn features_noisy_data() {
        let n = 200;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                0.1 * t + 5.0 * (2.0 * PI * t / 20.0).sin()
            })
            .collect();

        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let features = hp.trend_features();
        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();

        // Should have meaningful cycle component
        assert!(map["hp_cycle_variance_ratio"] > 0.0);
        assert!(map["hp_trend_strength"] >= 0.0);
        assert!(map["hp_trend_strength"] <= 1.0);
    }

    #[test]
    fn features_before_fit() {
        let hp = HodrickPrescottFilter::quarterly();
        let features = hp.trend_features();
        assert_eq!(features.len(), 3);
        for (_, v) in &features {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-10);
        }
    }

    // ── Standalone functions ────────────────────────────────────────────

    #[test]
    fn standalone_hp_trend_strength_constant() {
        let data = vec![5.0; 50];
        let strength = hp_trend_strength(&data, 1600.0);
        assert_abs_diff_eq!(strength, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standalone_hp_trend_strength_linear() {
        let data: Vec<f64> = (0..100).map(|i| 2.0 * i as f64).collect();
        let strength = hp_trend_strength(&data, 1600.0);
        assert!(strength > 0.99);
    }

    #[test]
    fn standalone_hp_trend_strength_empty() {
        assert_abs_diff_eq!(hp_trend_strength(&[], 1600.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standalone_hp_trend_strength_invalid_lambda() {
        let data = vec![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(hp_trend_strength(&data, -1.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hp_trend_strength(&data, 0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standalone_hp_cycle_variance_ratio_constant() {
        let data = vec![5.0; 50];
        let ratio = hp_cycle_variance_ratio(&data, 1600.0);
        assert_abs_diff_eq!(ratio, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standalone_hp_cycle_variance_ratio_linear() {
        let data: Vec<f64> = (0..100).map(|i| 2.0 * i as f64).collect();
        let ratio = hp_cycle_variance_ratio(&data, 1600.0);
        assert!(ratio < 0.01);
    }

    #[test]
    fn standalone_hp_cycle_variance_ratio_empty() {
        assert_abs_diff_eq!(hp_cycle_variance_ratio(&[], 1600.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standalone_hp_cycle_variance_ratio_invalid_lambda() {
        let data = vec![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(hp_cycle_variance_ratio(&data, -1.0), 0.0, epsilon = 1e-10);
    }

    // ── Consistency between standalone and struct ────────────────────────

    #[test]
    fn standalone_matches_struct_features() {
        let data: Vec<f64> = (0..80)
            .map(|i| {
                let t = i as f64;
                0.2 * t + 3.0 * (2.0 * PI * t / 12.0).sin()
            })
            .collect();

        let lambda = 1600.0;

        let mut hp = HodrickPrescottFilter::new(lambda).unwrap();
        hp.fit_trend(&data).unwrap();
        let features = hp.trend_features();
        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();

        let standalone_strength = hp_trend_strength(&data, lambda);
        let standalone_ratio = hp_cycle_variance_ratio(&data, lambda);

        assert_abs_diff_eq!(
            map["hp_trend_strength"],
            standalone_strength,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            map["hp_cycle_variance_ratio"],
            standalone_ratio,
            epsilon = 1e-10
        );
    }

    // ── Trend name ──────────────────────────────────────────────────────

    #[test]
    fn trend_name() {
        let hp = HodrickPrescottFilter::quarterly();
        assert_eq!(hp.trend_name(), "HodrickPrescottFilter");
    }

    // ── Clone ───────────────────────────────────────────────────────────

    #[test]
    fn clone_preserves_state() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let cloned = hp.clone();
        assert_eq!(hp.fitted_trend(), cloned.fitted_trend());
        assert_eq!(hp.cycle(), cloned.cycle());
    }

    // ── Cycle + trend = original ────────────────────────────────────────

    #[test]
    fn trend_plus_cycle_equals_original() {
        let data: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64;
                t * 0.3 + (t * 0.5).sin() * 2.0
            })
            .collect();

        let mut hp = HodrickPrescottFilter::quarterly();
        hp.fit_trend(&data).unwrap();

        let trend = hp.fitted_trend();
        let cycle = hp.cycle();

        for i in 0..data.len() {
            assert_abs_diff_eq!(trend[i] + cycle[i], data[i], epsilon = 1e-10);
        }
    }

    // ── Symmetry ────────────────────────────────────────────────────────

    #[test]
    fn symmetric_data_symmetric_trend() {
        // A symmetric pulse: [0, 0, ..., 0, 1, 0, ..., 0, 0]
        let n = 21;
        let mut data = vec![0.0; n];
        data[n / 2] = 1.0;

        let mut hp = HodrickPrescottFilter::new(100.0).unwrap();
        hp.fit_trend(&data).unwrap();

        let trend = hp.fitted_trend();
        let mid = n / 2;
        for i in 0..mid {
            assert_abs_diff_eq!(trend[i], trend[n - 1 - i], epsilon = 1e-10);
        }
    }
}
