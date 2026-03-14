//! Theil-Sen robust trend estimator.
//!
//! The Theil-Sen estimator computes the trend slope as the median of all
//! pairwise slopes, making it robust to up to ~29% outliers. The intercept
//! is the median of `y[i] - slope * i`. For large windows (n > 100),
//! a deterministic subsample of ~5000 pairs is used for efficiency.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::theilsen::TheilSenTrend;
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! // Fit a robust linear trend
//! let values: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 1.0).collect();
//! let mut trend = TheilSenTrend::new();
//! trend.fit_trend(&values).unwrap();
//!
//! assert_eq!(trend.fitted_trend().len(), 50);
//!
//! let forecast = trend.predict_trend(10);
//! assert_eq!(forecast.len(), 10);
//!
//! let features = trend.trend_features();
//! assert!(!features.is_empty());
//! ```

use super::traits::{Recency, TrendComponent};
use crate::error::{ForecastError, Result};

/// Theil-Sen robust linear trend estimator.
///
/// Computes the slope as the median of all pairwise slopes and the intercept
/// as the median of `y[i] - slope * i`. This makes the estimator robust
/// to outliers (up to ~29% breakdown point).
///
/// For large recency windows (n > 100), a deterministic subsample of ~5000
/// pairs is drawn using a linear congruential generator to keep computation
/// time manageable while preserving robustness.
#[derive(Debug, Clone)]
pub struct TheilSenTrend {
    /// Controls which portion of the data is used for parameter estimation.
    recency: Recency,
    /// Fitted slope.
    slope: f64,
    /// Fitted intercept (at index 0).
    intercept: f64,
    /// Fitted trend values (same length as training data).
    fitted: Vec<f64>,
    /// Length of training data.
    n_train: usize,
    /// R-squared of the fit on the recency window.
    r_squared: f64,
}

impl TheilSenTrend {
    /// Create a new Theil-Sen trend estimator with default settings.
    ///
    /// Default recency: `Recency::Fraction(0.3)`.
    pub fn new() -> Self {
        Self {
            recency: Recency::Fraction(0.3),
            slope: 0.0,
            intercept: 0.0,
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

    /// Return the fitted slope.
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Return the fitted intercept (at index 0).
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Default for TheilSenTrend {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the median of a mutable slice of `f64` values.
///
/// Sorts the slice in place and returns the middle element (or average of the
/// two middle elements for even-length slices).
fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Compute R-squared of fitted values against original values on a slice.
fn compute_r_squared(values: &[f64], fitted: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 1.0;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
    let ss_res: f64 = values
        .iter()
        .zip(fitted.iter())
        .map(|(&v, &f)| (v - f).powi(2))
        .sum();

    if ss_tot < 1e-12 {
        if ss_res < 1e-12 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

impl TrendComponent for TheilSenTrend {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let n = values.len();

        if n == 1 {
            self.slope = 0.0;
            self.intercept = values[0];
            self.fitted = vec![values[0]];
            self.n_train = 1;
            self.r_squared = 1.0;
            return Ok(());
        }

        let (rec_start, rec_end) = self.recency.resolve_with_data(values);
        let n_window = rec_end - rec_start;

        if n_window < 2 {
            // Degenerate case: single point in window.
            self.slope = 0.0;
            self.intercept = values[rec_start];
            self.fitted = (0..n).map(|_| values[rec_start]).collect();
            self.n_train = n;
            self.r_squared = 1.0;
            return Ok(());
        }

        // Compute pairwise slopes.
        let n_pairs_total = n_window * (n_window - 1) / 2;

        let mut slopes = if n_window > 100 {
            // Subsample ~5000 pairs using a deterministic LCG.
            let n_samples = 5000.min(n_pairs_total);
            let mut sampled_slopes = Vec::with_capacity(n_samples);
            let mut state: u64 = 12345;

            for _ in 0..n_samples {
                // Generate two random indices within the window.
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx_a = (state >> 33) as usize % n_window;

                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx_b = (state >> 33) as usize % n_window;

                if idx_a == idx_b {
                    continue;
                }

                let (i, j) = if idx_a < idx_b {
                    (idx_a, idx_b)
                } else {
                    (idx_b, idx_a)
                };

                // Absolute indices.
                let abs_i = rec_start + i;
                let abs_j = rec_start + j;

                let slope_ij = (values[abs_j] - values[abs_i]) / (abs_j - abs_i) as f64;
                sampled_slopes.push(slope_ij);
            }

            sampled_slopes
        } else {
            // Enumerate all pairs.
            let mut all_slopes = Vec::with_capacity(n_pairs_total);
            for i in 0..n_window {
                for j in (i + 1)..n_window {
                    let abs_i = rec_start + i;
                    let abs_j = rec_start + j;
                    let slope_ij = (values[abs_j] - values[abs_i]) / (abs_j - abs_i) as f64;
                    all_slopes.push(slope_ij);
                }
            }
            all_slopes
        };

        let slope = median(&mut slopes);

        // Compute intercept as median of y[i] - slope * i for all i in recency window.
        let mut intercepts: Vec<f64> = (rec_start..rec_end)
            .map(|i| values[i] - slope * i as f64)
            .collect();
        let intercept = median(&mut intercepts);

        // Compute fitted values for ALL indices 0..n.
        let fitted: Vec<f64> = (0..n).map(|t| intercept + slope * t as f64).collect();

        // Compute R-squared on the recency window.
        let window_values = &values[rec_start..rec_end];
        let window_fitted = &fitted[rec_start..rec_end];
        let r_squared = compute_r_squared(window_values, window_fitted);

        self.slope = slope;
        self.intercept = intercept;
        self.fitted = fitted;
        self.n_train = n;
        self.r_squared = r_squared;

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        (0..n_ahead)
            .map(|i| self.intercept + self.slope * (self.n_train + i) as f64)
            .collect()
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        vec![
            ("theilsen_slope", self.slope),
            ("theilsen_intercept", self.intercept),
            ("theilsen_r_squared", self.r_squared),
        ]
    }

    fn trend_name(&self) -> &str {
        "TheilSen"
    }

    fn n_params(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Pure linear data ───────────────────────────────────────────────

    #[test]
    fn pure_linear_slope_and_intercept() {
        // y = 2*x + 1 over 50 points
        let values: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.slope(), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.intercept(), 1.0, epsilon = 1e-10);

        // R-squared should be 1.0 for perfect linear data.
        assert_abs_diff_eq!(trend.r_squared, 1.0, epsilon = 1e-10);

        // Fitted values should match exactly.
        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 50);
        for (i, &f) in fitted.iter().enumerate() {
            assert_abs_diff_eq!(f, 2.0 * i as f64 + 1.0, epsilon = 1e-10);
        }
    }

    // ── Robustness to outliers ─────────────────────────────────────────

    #[test]
    fn robust_to_outliers() {
        // y = 3*x + 5 with a few extreme outliers
        let n = 40;
        let mut values: Vec<f64> = (0..n).map(|i| 3.0 * i as f64 + 5.0).collect();

        // Inject outliers (~10% of data)
        values[5] = 1000.0;
        values[15] = -500.0;
        values[25] = 2000.0;
        values[35] = -1000.0;

        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        // Theil-Sen should still recover approximately the correct slope.
        assert_abs_diff_eq!(trend.slope(), 3.0, epsilon = 0.5);
        assert_abs_diff_eq!(trend.intercept(), 5.0, epsilon = 5.0);
    }

    // ── Predict extrapolates correctly ─────────────────────────────────

    #[test]
    fn predict_extrapolates_correctly() {
        // y = 2*x + 1
        let values: Vec<f64> = (0..30).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);

        for (i, &f) in forecast.iter().enumerate() {
            let expected = 2.0 * (30 + i) as f64 + 1.0;
            assert_abs_diff_eq!(f, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn predict_zero_ahead() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(0);
        assert!(forecast.is_empty());
    }

    // ── Recency window ─────────────────────────────────────────────────

    #[test]
    fn recency_window_works() {
        // First half: slope 1, second half: slope 5
        let n = 100;
        let mut values = Vec::with_capacity(n);
        for i in 0..50 {
            values.push(1.0 * i as f64);
        }
        for i in 50..100 {
            values.push(49.0 + 5.0 * (i - 50) as f64);
        }

        // Use only last 30% — should capture the slope ~5 region.
        let mut trend = TheilSenTrend::new().with_recency(Recency::Fraction(0.3));
        trend.fit_trend(&values).unwrap();

        // Slope should be close to 5, not 1.
        assert_abs_diff_eq!(trend.slope(), 5.0, epsilon = 0.5);
    }

    #[test]
    fn recency_window_absolute() {
        // y = 4*x + 2
        let values: Vec<f64> = (0..50).map(|i| 4.0 * i as f64 + 2.0).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Window(20));
        trend.fit_trend(&values).unwrap();

        // Slope should still be 4 since the data is perfectly linear.
        assert_abs_diff_eq!(trend.slope(), 4.0, epsilon = 1e-10);
    }

    // ── Subsampling for large data ─────────────────────────────────────

    #[test]
    fn subsampling_works_for_large_data() {
        // y = 1.5*x + 10 with 200 points (n_window > 100 triggers subsampling)
        let values: Vec<f64> = (0..200).map(|i| 1.5 * i as f64 + 10.0).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        // With perfect linear data and deterministic subsampling,
        // slope should be recovered exactly.
        assert_abs_diff_eq!(trend.slope(), 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.intercept(), 10.0, epsilon = 1e-8);
    }

    #[test]
    fn subsampling_robust_to_outliers() {
        // y = 2*x + 3 with ~10% outliers, 200 points
        let n = 200;
        let mut values: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 3.0).collect();

        // Inject 20 outliers (~10%)
        for i in (0..n).step_by(10) {
            values[i] = if i % 20 == 0 { 5000.0 } else { -5000.0 };
        }

        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        // Should approximately recover the slope despite outliers.
        assert_abs_diff_eq!(trend.slope(), 2.0, epsilon = 1.0);
    }

    // ── n_params ───────────────────────────────────────────────────────

    #[test]
    fn n_params_is_2() {
        let trend = TheilSenTrend::new();
        assert_eq!(trend.n_params(), 2);
    }

    // ── Feature extraction ─────────────────────────────────────────────

    #[test]
    fn features_extraction() {
        let values: Vec<f64> = (0..50).map(|i| 3.0 * i as f64 + 7.0).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();
        assert_eq!(features.len(), 3);

        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();
        assert_abs_diff_eq!(map["theilsen_slope"], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(map["theilsen_intercept"], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(map["theilsen_r_squared"], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn features_before_fit() {
        let trend = TheilSenTrend::new();
        let features = trend.trend_features();
        assert_eq!(features.len(), 3);

        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();
        assert_abs_diff_eq!(map["theilsen_slope"], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(map["theilsen_intercept"], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(map["theilsen_r_squared"], 0.0, epsilon = 1e-10);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn empty_data_error() {
        let mut trend = TheilSenTrend::new();
        let result = trend.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn single_point() {
        let mut trend = TheilSenTrend::new();
        trend.fit_trend(&[42.0]).unwrap();

        assert_abs_diff_eq!(trend.slope(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.intercept(), 42.0, epsilon = 1e-10);
        assert_eq!(trend.fitted_trend().len(), 1);
        assert_abs_diff_eq!(trend.fitted_trend()[0], 42.0, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.r_squared, 1.0, epsilon = 1e-10);

        let forecast = trend.predict_trend(3);
        assert_eq!(forecast.len(), 3);
        for &f in &forecast {
            assert_abs_diff_eq!(f, 42.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn two_points() {
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&[10.0, 20.0]).unwrap();

        assert_abs_diff_eq!(trend.slope(), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.intercept(), 10.0, epsilon = 1e-10);

        let forecast = trend.predict_trend(1);
        assert_abs_diff_eq!(forecast[0], 30.0, epsilon = 1e-10);
    }

    // ── Trend name ─────────────────────────────────────────────────────

    #[test]
    fn trend_name_is_correct() {
        let trend = TheilSenTrend::new();
        assert_eq!(trend.trend_name(), "TheilSen");
    }

    // ── Default trait ──────────────────────────────────────────────────

    #[test]
    fn default_matches_new() {
        let a = TheilSenTrend::new();
        let b = TheilSenTrend::default();
        assert_abs_diff_eq!(a.slope, b.slope, epsilon = 1e-10);
        assert_abs_diff_eq!(a.intercept, b.intercept, epsilon = 1e-10);
        assert_eq!(a.recency, b.recency);
    }

    // ── Constant data ──────────────────────────────────────────────────

    #[test]
    fn constant_data() {
        let values = vec![7.0; 30];
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.slope(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.intercept(), 7.0, epsilon = 1e-10);

        for &f in trend.fitted_trend() {
            assert_abs_diff_eq!(f, 7.0, epsilon = 1e-10);
        }

        let forecast = trend.predict_trend(5);
        for &f in &forecast {
            assert_abs_diff_eq!(f, 7.0, epsilon = 1e-10);
        }
    }

    // ── Predict unfitted returns zeros ─────────────────────────────────

    #[test]
    fn predict_unfitted_returns_zeros() {
        let trend = TheilSenTrend::new();
        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        for &f in &forecast {
            assert_abs_diff_eq!(f, 0.0, epsilon = 1e-10);
        }
    }

    // ── Negative slope ─────────────────────────────────────────────────

    #[test]
    fn negative_slope() {
        let values: Vec<f64> = (0..40).map(|i| 100.0 - 2.5 * i as f64).collect();
        let mut trend = TheilSenTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.slope(), -2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.intercept(), 100.0, epsilon = 1e-10);
    }
}
