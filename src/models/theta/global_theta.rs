//! Global Theta: shared α across many series.
//!
//! Fits a single SES smoothing parameter α by minimizing the total SSE
//! across all series. Each series retains its own level and slope (b).
//! Uses the Standard Theta Method (θ=2) formulation.
//!
//! # Example
//!
//! ```rust
//! use anofox_forecast::models::theta::GlobalTheta;
//!
//! let series = vec![
//!     (0..50).map(|i| 10.0 + 0.3 * i as f64).collect::<Vec<_>>(),
//!     (0..50).map(|i| 20.0 - 0.1 * i as f64).collect::<Vec<_>>(),
//! ];
//! let mut model = GlobalTheta::new();
//! model.fit(&series).unwrap();
//! let forecasts = model.predict(10);
//! assert_eq!(forecasts.len(), 2);
//! ```

use crate::error::{ForecastError, Result};
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// Per-series state after fitting.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ThetaState {
    level: f64,
    b: f64,
    n: usize,
}

/// Global Theta model: shared α, per-series level and slope.
#[derive(Debug, Clone)]
pub struct GlobalTheta {
    theta: f64,
    alpha: f64,
    states: Vec<ThetaState>,
    fitted: bool,
}

impl GlobalTheta {
    /// Create with default θ=2 (Standard Theta Method).
    pub fn new() -> Self {
        Self {
            theta: 2.0,
            alpha: 0.5,
            states: Vec::new(),
            fitted: false,
        }
    }

    /// Create with a custom θ value.
    pub fn with_theta(theta: f64) -> Self {
        Self {
            theta,
            ..Self::new()
        }
    }

    /// Get the fitted α.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Fit shared α across all series.
    pub fn fit(&mut self, all_series: &[Vec<f64>]) -> Result<()> {
        if all_series.is_empty() {
            return Err(ForecastError::InsufficientData {
                needed: 1,
                got: 0,
                hint: Some("GlobalTheta requires at least one series".into()),
            });
        }

        // Optimize α: minimize total SSE across all series
        let result = nelder_mead(
            |params| {
                let alpha = params[0];
                Self::total_sse(all_series, alpha)
            },
            &[0.5],
            Some(&[(0.0001, 0.9999)]),
            NelderMeadConfig {
                max_iter: 500,
                tolerance: 1e-8,
                stagnation_window: 100,
                ..Default::default()
            },
        );

        self.alpha = result.optimal_point[0].clamp(0.0001, 0.9999);

        // Compute per-series states (level, slope)
        self.states = all_series
            .iter()
            .map(|values| {
                let n = values.len();
                // SES to get final level
                let mut level = values[0];
                for &y in values.iter().skip(1) {
                    level = self.alpha * y + (1.0 - self.alpha) * level;
                }
                // OLS slope
                let b = Self::ols_slope(values);
                ThetaState { level, b, n }
            })
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Predict h steps ahead for all series.
    pub fn predict(&self, horizon: usize) -> Vec<Vec<f64>> {
        if !self.fitted {
            return vec![];
        }
        self.states
            .iter()
            .map(|state| {
                let mut forecasts = Vec::with_capacity(horizon);
                for h in 1..=horizon {
                    let fc = state.level
                        + (1.0 - 1.0 / self.theta) * state.b * (1.0 / self.alpha + h as f64 - 1.0);
                    forecasts.push(fc);
                }
                forecasts
            })
            .collect()
    }

    /// Total SSE across all series.
    fn total_sse(all_series: &[Vec<f64>], alpha: f64) -> f64 {
        if alpha <= 0.0001 || alpha >= 0.9999 {
            return f64::MAX;
        }
        let mut total = 0.0;
        for values in all_series {
            if values.len() < 2 {
                continue;
            }
            let mut level = values[0];
            for &y in values.iter().skip(1) {
                let error = y - level;
                total += error * error;
                level = alpha * y + (1.0 - alpha) * level;
            }
        }
        total
    }

    /// OLS slope: b = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
    fn ols_slope(values: &[f64]) -> f64 {
        let n = values.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let sum_y: f64 = values.iter().sum();
        let mean_y = sum_y / n_f;
        let mean_x = (n_f - 1.0) / 2.0;

        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for (i, &y) in values.iter().enumerate() {
            let dx = i as f64 - mean_x;
            ss_xy += dx * (y - mean_y);
            ss_xx += dx * dx;
        }

        if ss_xx.abs() < 1e-10 {
            0.0
        } else {
            ss_xy / ss_xx
        }
    }
}

impl Default for GlobalTheta {
    fn default() -> Self {
        Self::new()
    }
}
