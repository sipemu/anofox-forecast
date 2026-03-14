//! Exponential trend modeling.
//!
//! Models trend as `y = exp(a + b*t)` by fitting OLS on the log-transformed
//! values within the recency window. The growth rate `b` is the key parameter:
//! positive means exponential growth, negative means exponential decay.
//!
//! # Example
//!
//! ``` ignore
//! use anofox_forecast::seasonality::exponential_trend::ExponentialTrend;
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! let values: Vec<f64> = (0..50).map(|t| 2.0 * (0.05 * t as f64).exp()).collect();
//! let mut trend = ExponentialTrend::new();
//! trend.fit_trend(&values).unwrap();
//!
//! let fitted = trend.fitted_trend();
//! assert_eq!(fitted.len(), 50);
//!
//! let forecast = trend.predict_trend(10);
//! assert_eq!(forecast.len(), 10);
//!
//! let rate = trend.growth_rate();
//! assert!((rate - 0.05).abs() < 1e-6);
//! ```

use super::traits::{Recency, TrendComponent};
use crate::error::{ForecastError, Result};

/// Exponential trend component.
///
/// Fits `log(y) = a + b*t` via OLS on the recency window, then produces
/// fitted values `exp(a + b*t)` for the full series (backwards extrapolation
/// for indices before the recency window).
#[derive(Debug, Clone)]
pub struct ExponentialTrend {
    /// Recency window specification.
    recency: Recency,
    /// Intercept of the log-linear model.
    a: f64,
    /// Slope (growth rate) of the log-linear model.
    b: f64,
    /// Fitted values for the full training series.
    fitted: Vec<f64>,
    /// Length of training data.
    n_train: usize,
    /// R-squared of the fit on the recency window.
    r_squared: f64,
}

impl ExponentialTrend {
    /// Create a new exponential trend with default parameters.
    ///
    /// Default recency: `Recency::Fraction(0.3)`.
    pub fn new() -> Self {
        Self {
            recency: Recency::Fraction(0.3),
            a: 0.0,
            b: 0.0,
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

    /// Return the exponential growth rate `b` from `log(y) = a + b*t`.
    ///
    /// Positive values indicate exponential growth, negative values indicate
    /// exponential decay.
    pub fn growth_rate(&self) -> f64 {
        self.b
    }
}

impl Default for ExponentialTrend {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendComponent for ExponentialTrend {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let n = values.len();

        if n == 1 {
            if values[0] <= 0.0 {
                return Err(ForecastError::InvalidParameter(
                    "ExponentialTrend requires all positive values in the recency window"
                        .to_string(),
                ));
            }
            self.a = values[0].ln();
            self.b = 0.0;
            self.fitted = vec![values[0]];
            self.n_train = 1;
            self.r_squared = 1.0;
            return Ok(());
        }

        let (rec_start, rec_end) = self.recency.resolve_with_data(values);
        let window = &values[rec_start..rec_end];

        // Log-transform the window; error on non-positive values.
        let log_y: Vec<f64> = window
            .iter()
            .map(|&y| {
                if y <= 0.0 {
                    Err(ForecastError::InvalidParameter(
                        "ExponentialTrend requires all positive values in the recency window"
                            .to_string(),
                    ))
                } else {
                    Ok(y.ln())
                }
            })
            .collect::<Result<Vec<f64>>>()?;

        let w = log_y.len();
        let w_f = w as f64;

        // Absolute indices for the window: rec_start, rec_start+1, ..., rec_end-1.
        let t_sum: f64 = (rec_start..rec_end).map(|i| i as f64).sum();
        let t_mean = t_sum / w_f;
        let logy_mean = log_y.iter().sum::<f64>() / w_f;

        let mut ss_tt = 0.0;
        let mut ss_ty = 0.0;
        for (j, &ly) in log_y.iter().enumerate() {
            let t = (rec_start + j) as f64;
            let dt = t - t_mean;
            let dy = ly - logy_mean;
            ss_tt += dt * dt;
            ss_ty += dt * dy;
        }

        let slope = if ss_tt.abs() < 1e-15 {
            0.0
        } else {
            ss_ty / ss_tt
        };
        let intercept = logy_mean - slope * t_mean;

        self.a = intercept;
        self.b = slope;

        // Compute fitted values for ALL indices 0..n (backwards extrapolation).
        self.fitted = (0..n)
            .map(|i| (intercept + slope * i as f64).exp())
            .collect();
        self.n_train = n;

        // Compute R-squared on the recency window using original values vs fitted.
        let window_fitted = &self.fitted[rec_start..rec_end];
        let mean_y = window.iter().sum::<f64>() / w_f;
        let ss_tot: f64 = window.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = window
            .iter()
            .zip(window_fitted.iter())
            .map(|(&y, &f)| (y - f).powi(2))
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

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        &self.fitted
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        (0..n_ahead)
            .map(|i| (self.a + self.b * (self.n_train + i) as f64).exp())
            .collect()
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        vec![
            ("exponential_growth_rate", self.b),
            ("exponential_r_squared", self.r_squared),
        ]
    }

    fn trend_name(&self) -> &str {
        "exponential"
    }

    fn n_params(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Pure exponential data ──────────────────────────────────────────

    #[test]
    fn pure_exponential_coefficients() {
        // y = 2.0 * exp(0.05 * t)
        let values: Vec<f64> = (0..100).map(|t| 2.0 * (0.05 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        // a = ln(2.0), b = 0.05
        assert_abs_diff_eq!(trend.a, 2.0_f64.ln(), epsilon = 1e-10);
        assert_abs_diff_eq!(trend.b, 0.05, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.growth_rate(), 0.05, epsilon = 1e-10);
    }

    // ── Fitted values close to original ────────────────────────────────

    #[test]
    fn fitted_values_close_to_original() {
        let values: Vec<f64> = (0..50).map(|t| 2.0 * (0.05 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let fitted = trend.fitted_trend();
        assert_eq!(fitted.len(), 50);
        for (i, (&f, &v)) in fitted.iter().zip(values.iter()).enumerate() {
            assert_abs_diff_eq!(f, v, epsilon = 1e-8);
            let _ = i;
        }
    }

    // ── Predict extrapolates correctly ─────────────────────────────────

    #[test]
    fn predict_extrapolates_correctly() {
        let n = 50;
        let values: Vec<f64> = (0..n).map(|t| 2.0 * (0.05 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        for (i, &f) in forecast.iter().enumerate() {
            let expected = 2.0 * (0.05 * (n + i) as f64).exp();
            assert_abs_diff_eq!(f, expected, epsilon = 1e-6);
        }
    }

    // ── Non-positive values return error ───────────────────────────────

    #[test]
    fn non_positive_values_error() {
        let values = vec![1.0, 2.0, 0.0, 4.0, 5.0];
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        let result = trend.fit_trend(&values);
        assert!(
            matches!(result, Err(ForecastError::InvalidParameter(ref msg)) if msg.contains("positive"))
        );
    }

    #[test]
    fn negative_values_error() {
        let values = vec![1.0, -2.0, 3.0];
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        let result = trend.fit_trend(&values);
        assert!(matches!(result, Err(ForecastError::InvalidParameter(_))));
    }

    // ── Recency window works ───────────────────────────────────────────

    #[test]
    fn recency_window_works() {
        // First half: y = exp(0.1*t), second half: y = C * exp(0.02*t)
        // With a recency window, only the recent portion should determine the rate.
        let n = 100;
        let values: Vec<f64> = (0..n).map(|t| 3.0 * (0.02 * t as f64).exp()).collect();

        let mut trend_full = ExponentialTrend::new().with_recency(Recency::Full);
        trend_full.fit_trend(&values).unwrap();

        let mut trend_recent = ExponentialTrend::new().with_recency(Recency::Fraction(0.3));
        trend_recent.fit_trend(&values).unwrap();

        // For pure exponential data, both should find the same rate.
        assert_abs_diff_eq!(trend_full.growth_rate(), 0.02, epsilon = 1e-10);
        assert_abs_diff_eq!(trend_recent.growth_rate(), 0.02, epsilon = 1e-6);

        // Fitted values should have full length regardless of recency.
        assert_eq!(trend_recent.fitted_trend().len(), n);
    }

    #[test]
    fn recency_window_different_rate() {
        // Construct data where first and second halves have different growth rates.
        // The recency-windowed fit should pick up the recent rate.
        let n = 100;
        let mut values = Vec::with_capacity(n);
        // First 70: y = exp(0.1*t)
        for t in 0..70 {
            values.push((0.1 * t as f64).exp());
        }
        // Last 30: y = C * exp(0.02*t) where C makes it continuous at t=70.
        let c = (0.1 * 70.0_f64).exp() / (0.02 * 70.0_f64).exp();
        for t in 70..100 {
            values.push(c * (0.02 * t as f64).exp());
        }

        let mut trend = ExponentialTrend::new().with_recency(Recency::Fraction(0.3));
        trend.fit_trend(&values).unwrap();

        // The growth rate should be close to 0.02 (the recent rate), not 0.1.
        assert_abs_diff_eq!(trend.growth_rate(), 0.02, epsilon = 1e-4);
    }

    // ── n_params returns 2 ─────────────────────────────────────────────

    #[test]
    fn n_params_returns_2() {
        let trend = ExponentialTrend::new();
        assert_eq!(trend.n_params(), 2);
    }

    // ── Features extraction ────────────────────────────────────────────

    #[test]
    fn features_extraction() {
        let values: Vec<f64> = (0..50).map(|t| 2.0 * (0.05 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let features = trend.trend_features();
        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();

        assert!(map.contains_key("exponential_growth_rate"));
        assert!(map.contains_key("exponential_r_squared"));
        assert_abs_diff_eq!(map["exponential_growth_rate"], 0.05, epsilon = 1e-10);
        assert!(
            map["exponential_r_squared"] > 0.999,
            "R^2 should be ~1 for pure exponential data, got {}",
            map["exponential_r_squared"]
        );
    }

    #[test]
    fn features_before_fit() {
        let trend = ExponentialTrend::new();
        let features = trend.trend_features();
        assert_eq!(features.len(), 2);

        let map: std::collections::HashMap<&str, f64> = features.into_iter().collect();
        assert_abs_diff_eq!(map["exponential_growth_rate"], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(map["exponential_r_squared"], 0.0, epsilon = 1e-15);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn empty_data_error() {
        let mut trend = ExponentialTrend::new();
        let result = trend.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn single_point() {
        let mut trend = ExponentialTrend::new();
        trend.fit_trend(&[5.0]).unwrap();

        assert_eq!(trend.fitted_trend().len(), 1);
        assert_abs_diff_eq!(trend.fitted_trend()[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.growth_rate(), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(trend.r_squared, 1.0, epsilon = 1e-15);

        // Predict from single point should return constant.
        let forecast = trend.predict_trend(3);
        assert_eq!(forecast.len(), 3);
        for &f in &forecast {
            assert_abs_diff_eq!(f, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn single_non_positive_point_error() {
        let mut trend = ExponentialTrend::new();
        let result = trend.fit_trend(&[0.0]);
        assert!(matches!(result, Err(ForecastError::InvalidParameter(_))));

        let mut trend2 = ExponentialTrend::new();
        let result2 = trend2.fit_trend(&[-3.0]);
        assert!(matches!(result2, Err(ForecastError::InvalidParameter(_))));
    }

    // ── Trend name ─────────────────────────────────────────────────────

    #[test]
    fn trend_name_is_correct() {
        let trend = ExponentialTrend::new();
        assert_eq!(trend.trend_name(), "exponential");
    }

    // ── R-squared ──────────────────────────────────────────────────────

    #[test]
    fn r_squared_pure_exponential() {
        let values: Vec<f64> = (0..50).map(|t| 10.0 * (0.03 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.r_squared, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn r_squared_noisy_data() {
        // Exponential with added noise; R^2 should be less than 1.
        let values: Vec<f64> = (0..100)
            .map(|t| {
                let base = 2.0 * (0.03 * t as f64).exp();
                // Simple deterministic "noise" using sin.
                base + 0.5 * (t as f64 * 1.7).sin()
            })
            .collect();

        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert!(
            trend.r_squared < 1.0,
            "R^2 should be less than 1 for noisy data"
        );
        assert!(
            trend.r_squared > 0.5,
            "R^2 should still be reasonable for mostly exponential data"
        );
    }

    // ── Predict zero ahead ─────────────────────────────────────────────

    #[test]
    fn predict_zero_ahead() {
        let values: Vec<f64> = (0..20).map(|t| (0.1 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        let forecast = trend.predict_trend(0);
        assert!(forecast.is_empty());
    }

    // ── Predict before fit ─────────────────────────────────────────────

    #[test]
    fn predict_before_fit() {
        let trend = ExponentialTrend::new();
        let forecast = trend.predict_trend(5);
        assert_eq!(forecast.len(), 5);
        // a=0, b=0, n_train=0 => exp(0 + 0*i) = 1.0 for all i
        for &f in &forecast {
            assert_abs_diff_eq!(f, 1.0, epsilon = 1e-15);
        }
    }

    // ── Default trait ──────────────────────────────────────────────────

    #[test]
    fn default_same_as_new() {
        let a = ExponentialTrend::new();
        let b = ExponentialTrend::default();
        assert_eq!(a.recency, b.recency);
        assert_abs_diff_eq!(a.a, b.a, epsilon = 1e-15);
        assert_abs_diff_eq!(a.b, b.b, epsilon = 1e-15);
    }

    // ── Decay (negative growth rate) ───────────────────────────────────

    #[test]
    fn exponential_decay() {
        let values: Vec<f64> = (0..80).map(|t| 10.0 * (-0.03 * t as f64).exp()).collect();
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.growth_rate(), -0.03, epsilon = 1e-10);
        assert_abs_diff_eq!(trend.r_squared, 1.0, epsilon = 1e-6);
    }

    // ── Constant data (b = 0) ──────────────────────────────────────────

    #[test]
    fn constant_positive_data() {
        let values = vec![5.0; 30];
        let mut trend = ExponentialTrend::new().with_recency(Recency::Full);
        trend.fit_trend(&values).unwrap();

        assert_abs_diff_eq!(trend.growth_rate(), 0.0, epsilon = 1e-10);
        for &f in trend.fitted_trend() {
            assert_abs_diff_eq!(f, 5.0, epsilon = 1e-10);
        }
    }
}
