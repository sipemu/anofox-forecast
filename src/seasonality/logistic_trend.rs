//! Logistic (S-curve) trend component.
//!
//! Models trend as a logistic growth function `K / (1 + exp(-steepness * (t - midpoint)))`,
//! where `K` is the carrying capacity, `steepness` controls the growth rate, and `midpoint`
//! is the inflection point. This is useful for series that exhibit saturating growth
//! (e.g., market adoption, population growth, resource-constrained processes).
//!
//! The capacity `K` can be specified explicitly via [`CapacityMode::Fixed`] or estimated
//! automatically from the data via [`CapacityMode::Auto`] (default: `max(window) * 1.5`).
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::logistic_trend::LogisticTrend;
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! // Generate logistic growth data: y = 100 / (1 + exp(-0.1*(t-50)))
//! let values: Vec<f64> = (0..100).map(|t| {
//!     100.0 / (1.0 + (-0.1 * (t as f64 - 50.0)).exp())
//! }).collect();
//!
//! let mut trend = LogisticTrend::new();
//! trend.fit_trend(&values).unwrap();
//!
//! let fitted = trend.fitted_trend();
//! assert_eq!(fitted.len(), 100);
//!
//! let forecast = trend.predict_trend(10);
//! assert_eq!(forecast.len(), 10);
//! // Predictions saturate towards the capacity
//! assert!(forecast.last().unwrap() < &trend.capacity());
//! ```

use super::traits::{Recency, TrendComponent};
use crate::error::{ForecastError, Result};

/// Controls how the carrying capacity K is determined.
#[derive(Debug, Clone, PartialEq)]
pub enum CapacityMode {
    /// Use a fixed, user-specified capacity.
    Fixed(f64),
    /// Estimate capacity automatically as `max(window) * 1.5`.
    Auto,
}

/// Logistic trend component.
///
/// Fits a logistic (S-curve) growth function to the data using linearized OLS
/// regression on the recency window, then evaluates the curve over the full series.
#[derive(Debug, Clone)]
pub struct LogisticTrend {
    recency: Recency,
    capacity: f64,
    midpoint: f64,
    steepness: f64,
    fitted: Vec<f64>,
    n_train: usize,
    r_squared: f64,
    capacity_mode: CapacityMode,
}

impl LogisticTrend {
    /// Create a new logistic trend with default parameters.
    ///
    /// Defaults: `Recency::Fraction(0.3)`, `CapacityMode::Auto`.
    pub fn new() -> Self {
        Self {
            recency: Recency::Fraction(0.3),
            capacity: 0.0,
            midpoint: 0.0,
            steepness: 0.0,
            fitted: Vec::new(),
            n_train: 0,
            r_squared: 0.0,
            capacity_mode: CapacityMode::Auto,
        }
    }

    /// Set the recency window for fitting (builder-style).
    pub fn with_recency(mut self, recency: Recency) -> Self {
        self.recency = recency;
        self
    }

    /// Set a fixed carrying capacity (builder-style).
    ///
    /// This sets `CapacityMode::Fixed(capacity)`.
    pub fn with_capacity(mut self, capacity: f64) -> Self {
        self.capacity_mode = CapacityMode::Fixed(capacity);
        self
    }

    /// Return the carrying capacity K.
    pub fn capacity(&self) -> f64 {
        self.capacity
    }

    /// Return the midpoint (inflection point).
    pub fn midpoint(&self) -> f64 {
        self.midpoint
    }

    /// Return the steepness (growth rate).
    pub fn steepness(&self) -> f64 {
        self.steepness
    }
}

impl Default for LogisticTrend {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate the logistic function at index `t`.
#[inline]
fn logistic(capacity: f64, steepness: f64, midpoint: f64, t: f64) -> f64 {
    capacity / (1.0 + (-steepness * (t - midpoint)).exp())
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

impl TrendComponent for LogisticTrend {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let n = values.len();
        let (rec_start, rec_end) = self.recency.resolve_with_data(values);
        let window = &values[rec_start..rec_end];
        let window_len = window.len();

        // Determine capacity K.
        let k = match self.capacity_mode {
            CapacityMode::Fixed(k) => k,
            CapacityMode::Auto => {
                let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                max_val * 1.5
            }
        };

        if !k.is_finite() || k <= 0.0 {
            return Err(ForecastError::InvalidParameter(
                "capacity K must be positive and finite".to_string(),
            ));
        }

        // Linearize: log(y / (K - y)) = steepness * t - steepness * midpoint
        //                              = b * t + a
        // where b = steepness, a = -steepness * midpoint
        let epsilon = 1e-10;

        let mut sum_t = 0.0_f64;
        let mut sum_t2 = 0.0_f64;
        let mut sum_z = 0.0_f64;
        let mut sum_tz = 0.0_f64;
        let n_w = window_len as f64;

        for (j, &y) in window.iter().enumerate() {
            let t = (rec_start + j) as f64; // absolute index
            let y_clamped = y.clamp(epsilon, k - epsilon);

            if y_clamped <= 0.0 || y_clamped >= k {
                return Err(ForecastError::ComputationError(
                    "logistic linearization failed: value out of range after clamping".to_string(),
                ));
            }

            let z = (y_clamped / (k - y_clamped)).ln();

            sum_t += t;
            sum_t2 += t * t;
            sum_z += z;
            sum_tz += t * z;
        }

        // OLS: z = a + b*t
        let ss_tt = sum_t2 - sum_t * sum_t / n_w;
        let ss_tz = sum_tz - sum_t * sum_z / n_w;

        let b = if ss_tt.abs() < 1e-12 {
            0.0
        } else {
            ss_tz / ss_tt
        };
        let a = (sum_z - b * sum_t) / n_w;

        // steepness = b, midpoint = -a/b
        self.steepness = b;
        self.midpoint = if b.abs() < 1e-12 { 0.0 } else { -a / b };
        self.capacity = k;

        // Compute fitted values for ALL indices 0..n.
        self.fitted = (0..n)
            .map(|i| logistic(k, self.steepness, self.midpoint, i as f64))
            .collect();
        self.n_train = n;

        // R² computed on the recency window.
        let fitted_window: Vec<f64> = (rec_start..rec_end).map(|i| self.fitted[i]).collect();
        self.r_squared = compute_r_squared(window, &fitted_window);

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        (0..n_ahead)
            .map(|i| {
                logistic(
                    self.capacity,
                    self.steepness,
                    self.midpoint,
                    (self.n_train + i) as f64,
                )
            })
            .collect()
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        if self.fitted.is_empty() {
            return Vec::new();
        }

        let last_fitted = self.fitted.last().copied().unwrap_or(0.0);
        let saturation_pct = if self.capacity > 0.0 {
            last_fitted / self.capacity * 100.0
        } else {
            0.0
        };

        vec![
            ("logistic_capacity", self.capacity),
            ("logistic_midpoint", self.midpoint),
            ("logistic_steepness", self.steepness),
            ("logistic_saturation_pct", saturation_pct),
        ]
    }

    fn trend_name(&self) -> &str {
        "logistic"
    }

    fn n_params(&self) -> usize {
        match self.capacity_mode {
            CapacityMode::Auto => 3,
            CapacityMode::Fixed(_) => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Generate logistic growth data: y = K / (1 + exp(-steepness*(t - midpoint)))
    fn logistic_data(k: f64, steepness: f64, midpoint: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| k / (1.0 + (-steepness * (t as f64 - midpoint)).exp()))
            .collect()
    }

    // ── Logistic growth data: recover params approximately ──────────

    #[test]
    fn fit_recovers_logistic_params() {
        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new().with_capacity(100.0);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.capacity(), 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(trend.midpoint(), 50.0, epsilon = 1.0);
        assert_abs_diff_eq!(trend.steepness(), 0.1, epsilon = 0.02);
    }

    // ── Fixed capacity works ────────────────────────────────────────

    #[test]
    fn fixed_capacity() {
        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new().with_capacity(100.0);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.capacity(), 100.0, epsilon = 1e-10);
        assert_eq!(trend.n_params(), 2);
    }

    // ── Auto capacity works ─────────────────────────────────────────

    #[test]
    fn auto_capacity() {
        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new();
        trend.fit_trend(&values).unwrap();

        // Auto capacity should be max(window) * 1.5
        // The recency window is the last 30% by default, so max of that window
        let (rec_start, _) = Recency::Fraction(0.3).resolve(100);
        let window_max = values[rec_start..]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_abs_diff_eq!(trend.capacity(), window_max * 1.5, epsilon = 1e-6);
        assert_eq!(trend.n_params(), 3);
    }

    // ── Fitted close to original for logistic data ──────────────────

    #[test]
    fn fitted_close_to_original() {
        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new().with_capacity(100.0);
        trend.fit_trend(&values).unwrap();

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 100);

        for (i, (&f, &v)) in fitted.iter().zip(values.iter()).enumerate() {
            assert_abs_diff_eq!(f, v, epsilon = 0.5);
            let _ = i;
        }

        assert!(
            trend.r_squared > 0.99,
            "R² should be > 0.99, got {}",
            trend.r_squared
        );
    }

    // ── Predict saturates at capacity ───────────────────────────────

    #[test]
    fn predict_saturates_at_capacity() {
        use super::super::traits::Recency;

        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new()
            .with_capacity(100.0)
            .with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(1000);
        let last = *forecast.last().unwrap();

        // Should be at or very close to capacity (at float precision,
        // exp(-large) rounds to 0.0 so last may equal K exactly)
        assert!(
            last <= 100.0 + 1e-10,
            "prediction should not exceed capacity, got {}",
            last
        );
        assert_abs_diff_eq!(last, 100.0, epsilon = 0.01);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn empty_data_error() {
        let mut trend = LogisticTrend::new();
        let result = trend.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn single_point() {
        let mut trend = LogisticTrend::new().with_capacity(100.0);
        let result = trend.fit_trend(&[50.0]);
        assert!(result.is_ok());
        assert_eq!(trend.fitted_trend().len(), 1);
    }

    #[test]
    fn predict_zero_ahead() {
        let values = logistic_data(100.0, 0.1, 50.0, 50);
        let mut trend = LogisticTrend::new().with_capacity(100.0);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(0);
        assert!(forecast.is_empty());
    }

    #[test]
    fn predict_unfitted_returns_zeros() {
        let trend = LogisticTrend::new();
        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        // With capacity=0, logistic function returns 0/... which is 0 or NaN
        // but since steepness=0, midpoint=0, capacity=0 -> all 0/(1+exp(0))=0
        for &v in &forecast {
            assert!(v.is_nan() || v == 0.0);
        }
    }

    // ── n_params correct for Auto vs Fixed ──────────────────────────

    #[test]
    fn n_params_auto() {
        let trend = LogisticTrend::new();
        assert_eq!(trend.n_params(), 3);
    }

    #[test]
    fn n_params_fixed() {
        let trend = LogisticTrend::new().with_capacity(100.0);
        assert_eq!(trend.n_params(), 2);
    }

    // ── Features extraction ─────────────────────────────────────────

    #[test]
    fn features_extraction() {
        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new().with_capacity(100.0);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();
        assert_eq!(features.len(), 4);

        let get = |name: &str| -> f64 {
            features
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("feature '{}' not found", name))
        };

        assert_abs_diff_eq!(get("logistic_capacity"), 100.0, epsilon = 1e-6);
        assert!(get("logistic_midpoint").is_finite());
        assert!(get("logistic_steepness").is_finite());

        let sat = get("logistic_saturation_pct");
        assert!(
            sat > 0.0 && sat <= 100.0,
            "saturation_pct should be in (0, 100], got {}",
            sat
        );
    }

    #[test]
    fn features_before_fit_empty() {
        let trend = LogisticTrend::new();
        let features = trend.trend_features();
        assert!(features.is_empty());
    }

    // ── Trait basics ────────────────────────────────────────────────

    #[test]
    fn trend_name() {
        let trend = LogisticTrend::new();
        assert_eq!(trend.trend_name(), "logistic");
    }

    #[test]
    fn default_same_as_new() {
        let a = LogisticTrend::new();
        let b = LogisticTrend::default();
        assert_eq!(a.capacity_mode, b.capacity_mode);
        assert_abs_diff_eq!(a.capacity, b.capacity, epsilon = 1e-12);
        assert_abs_diff_eq!(a.midpoint, b.midpoint, epsilon = 1e-12);
        assert_abs_diff_eq!(a.steepness, b.steepness, epsilon = 1e-12);
    }

    // ── Builder pattern ─────────────────────────────────────────────

    #[test]
    fn builder_recency() {
        let trend = LogisticTrend::new().with_recency(Recency::Full);
        assert_eq!(trend.recency, Recency::Full);
    }

    #[test]
    fn builder_capacity_sets_fixed_mode() {
        let trend = LogisticTrend::new().with_capacity(200.0);
        assert_eq!(trend.capacity_mode, CapacityMode::Fixed(200.0));
    }

    // ── Full recency with fixed capacity ────────────────────────────

    #[test]
    fn full_recency_fixed_capacity_recovers_params() {
        let values = logistic_data(100.0, 0.1, 50.0, 100);
        let mut trend = LogisticTrend::new()
            .with_recency(Recency::Full)
            .with_capacity(100.0);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.steepness(), 0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(trend.midpoint(), 50.0, epsilon = 1e-6);
    }
}
