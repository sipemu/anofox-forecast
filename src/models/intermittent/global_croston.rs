//! Global Croston: shared α across many intermittent demand series.
//!
//! Fits a single smoothing parameter α by minimizing the total MSE
//! across all series. Each series retains its own demand/interval levels.
//!
//! # Example
//!
//! ```rust
//! use anofox_forecast::models::intermittent::GlobalCroston;
//!
//! let series = vec![
//!     vec![0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0],
//!     vec![0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0],
//! ];
//! let mut model = GlobalCroston::new();
//! model.fit(&series).unwrap();
//! let forecasts = model.predict(4);
//! assert_eq!(forecasts.len(), 2);
//! ```

use crate::error::{ForecastError, Result};
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// Croston variant for bias correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrostonVariant {
    /// Classic Croston (no correction)
    Classic,
    /// Syntetos-Boylan Approximation (multiply by 1 - α/2)
    SBA,
}

/// Global Croston model: shared α, per-series demand/interval levels.
#[derive(Debug, Clone)]
pub struct GlobalCroston {
    variant: CrostonVariant,
    alpha: f64,
    /// Per-series final states: (demand_level, interval_level)
    states: Vec<(f64, f64)>,
    fitted: bool,
}

impl GlobalCroston {
    /// Create a new GlobalCroston with Classic variant.
    pub fn new() -> Self {
        Self {
            variant: CrostonVariant::Classic,
            alpha: 0.1,
            states: Vec::new(),
            fitted: false,
        }
    }

    /// Create with SBA bias correction.
    pub fn sba() -> Self {
        Self {
            variant: CrostonVariant::SBA,
            ..Self::new()
        }
    }

    /// Create with a specific variant.
    pub fn with_variant(variant: CrostonVariant) -> Self {
        Self {
            variant,
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
                hint: Some("GlobalCroston requires at least one series".into()),
            });
        }

        // V-02: per-series NaN/Inf guard — raw &[Vec<f64>] API bypasses validate_series_complete()
        for (i, s) in all_series.iter().enumerate() {
            if s.iter().any(|v| !v.is_finite()) {
                return Err(ForecastError::InvalidParameter(format!(
                    "series {} contains NaN or Inf values",
                    i
                )));
            }
        }

        // Pre-extract demands and intervals for all series
        let extracted: Vec<Option<(Vec<f64>, Vec<f64>)>> = all_series
            .iter()
            .map(|values| {
                let (demands, intervals) = Self::extract_demands(values);
                if demands.len() >= 2 {
                    Some((demands, intervals))
                } else {
                    None // Not enough demand occurrences
                }
            })
            .collect();

        let n_valid = extracted.iter().filter(|e| e.is_some()).count();
        if n_valid == 0 {
            return Err(ForecastError::ConvergenceFailure(
                "No series with at least 2 demand occurrences".into(),
            ));
        }

        // Optimize α: minimize total MSE across all valid series
        let result = nelder_mead(
            |params| {
                let alpha = params[0];
                Self::total_mse(&extracted, alpha)
            },
            &[0.1],
            Some(&[(0.01, 0.99)]),
            NelderMeadConfig {
                max_iter: 200,
                tolerance: 1e-6,
                stagnation_window: 50,
                ..Default::default()
            },
        );

        self.alpha = result.optimal_point[0].clamp(0.01, 0.99);

        // Compute final states per series
        self.states = extracted
            .iter()
            .map(|ext| {
                if let Some((demands, intervals)) = ext {
                    let dl = Self::fit_ses(demands, self.alpha);
                    let il = Self::fit_ses(intervals, self.alpha);
                    (dl, il)
                } else {
                    (0.0, 1.0) // fallback for invalid series
                }
            })
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Predict h steps ahead for all series (flat forecasts).
    pub fn predict(&self, horizon: usize) -> Vec<Vec<f64>> {
        if !self.fitted {
            return vec![];
        }
        self.states
            .iter()
            .map(|&(dl, il)| {
                let fc = self.apply_bias_correction(dl, il);
                vec![fc; horizon]
            })
            .collect()
    }

    /// Total MSE across all valid series.
    fn total_mse(extracted: &[Option<(Vec<f64>, Vec<f64>)>], alpha: f64) -> f64 {
        if alpha <= 0.01 || alpha >= 0.99 {
            return f64::MAX;
        }
        let mut total = 0.0;
        let mut count = 0;
        for (demands, intervals) in extracted.iter().flatten() {
            total += Self::series_mse(demands, intervals, alpha);
            count += 1;
        }
        if count == 0 {
            f64::MAX
        } else {
            total / count as f64
        }
    }

    /// MSE for a single series' demand + interval sub-series.
    fn series_mse(demands: &[f64], intervals: &[f64], alpha: f64) -> f64 {
        let mut demand_sse = 0.0;
        let mut demand_level = demands[0];
        for &d in demands.iter().skip(1) {
            let err = d - demand_level;
            demand_sse += err * err;
            demand_level = alpha * d + (1.0 - alpha) * demand_level;
        }

        let mut interval_sse = 0.0;
        let mut interval_level = intervals[0];
        for &iv in intervals.iter().skip(1) {
            let err = iv - interval_level;
            interval_sse += err * err;
            interval_level = alpha * iv + (1.0 - alpha) * interval_level;
        }

        let n = demands.len() + intervals.len();
        if n <= 2 {
            0.0
        } else {
            (demand_sse + interval_sse) / (n - 2) as f64
        }
    }

    /// Extract demand sizes and inter-demand intervals.
    fn extract_demands(values: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut demands = Vec::new();
        let mut intervals = Vec::new();
        let mut zeros_since_last = 0usize;

        for &v in values {
            if v > 0.0 {
                demands.push(v);
                intervals.push((zeros_since_last + 1) as f64);
                zeros_since_last = 0;
            } else {
                zeros_since_last += 1;
            }
        }

        (demands, intervals)
    }

    /// Simple exponential smoothing — returns final level.
    fn fit_ses(values: &[f64], alpha: f64) -> f64 {
        let mut level = values[0];
        for &v in values.iter().skip(1) {
            level = alpha * v + (1.0 - alpha) * level;
        }
        level
    }

    /// Apply variant-specific bias correction.
    fn apply_bias_correction(&self, demand_level: f64, interval_level: f64) -> f64 {
        let base = demand_level / interval_level.max(0.001);
        match self.variant {
            CrostonVariant::Classic => base,
            CrostonVariant::SBA => base * (1.0 - self.alpha / 2.0),
        }
    }
}

impl Default for GlobalCroston {
    fn default() -> Self {
        Self::new()
    }
}
