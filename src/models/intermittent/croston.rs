//! Croston's method for intermittent demand forecasting.
//!
//! Croston's method separately forecasts demand sizes and inter-arrival times
//! using exponential smoothing, then combines them for the final forecast.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// Croston's method variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrostonVariant {
    /// Classic Croston method.
    Classic,
    /// Syntetos-Boylan Approximation (SBA) - applies bias correction.
    SBA,
    /// Syntetos-Boylan Approximation with optimized alpha.
    SBAOptimized,
}

/// Croston's method for intermittent demand forecasting.
#[derive(Debug, Clone)]
pub struct Croston {
    /// Smoothing parameter (0.0 to 1.0).
    alpha: f64,
    /// Whether to optimize alpha.
    optimize_alpha: bool,
    /// Variant of Croston's method.
    variant: CrostonVariant,
    /// Estimated demand level.
    demand_level: Option<f64>,
    /// Estimated interval level.
    interval_level: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
}

impl Croston {
    /// Create a new Croston model with default alpha (0.1).
    pub fn new() -> Self {
        Self {
            alpha: 0.1,
            optimize_alpha: false,
            variant: CrostonVariant::Classic,
            demand_level: None,
            interval_level: None,
            fitted: None,
            residuals: None,
            n: 0,
        }
    }

    /// Create Croston model with specified alpha.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.01, 0.99);
        self.optimize_alpha = false;
        self
    }

    /// Enable alpha optimization.
    pub fn optimized(mut self) -> Self {
        self.optimize_alpha = true;
        self
    }

    /// Use SBA (Syntetos-Boylan Approximation) variant.
    pub fn sba(mut self) -> Self {
        self.variant = CrostonVariant::SBA;
        self
    }

    /// Use SBA variant with optimized alpha.
    pub fn sba_optimized(mut self) -> Self {
        self.variant = CrostonVariant::SBAOptimized;
        self.optimize_alpha = true;
        self
    }

    /// Get the estimated demand level.
    pub fn demand_level(&self) -> Option<f64> {
        self.demand_level
    }

    /// Get the estimated interval level.
    pub fn interval_level(&self) -> Option<f64> {
        self.interval_level
    }

    /// Get the current alpha value.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the variant being used.
    pub fn variant(&self) -> CrostonVariant {
        self.variant
    }

    /// Extract demand sizes and intervals from the series.
    fn extract_demands(values: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let mut demands = Vec::new();
        let mut intervals = Vec::new();
        let mut last_demand_idx: Option<usize> = None;

        for (i, &v) in values.iter().enumerate() {
            if v > 0.0 {
                demands.push(v);
                if let Some(last_idx) = last_demand_idx {
                    intervals.push(i - last_idx);
                }
                last_demand_idx = Some(i);
            }
        }

        (demands, intervals)
    }

    /// Fit SES to a sequence and return the final level.
    fn fit_ses(values: &[f64], alpha: f64) -> f64 {
        if values.is_empty() {
            return 1.0;
        }
        let mut level = values[0];
        for &v in values.iter().skip(1) {
            level = alpha * v + (1.0 - alpha) * level;
        }
        level
    }

    /// Compute objective (MSE) for alpha optimization.
    fn compute_mse(values: &[f64], alpha: f64) -> f64 {
        let (demands, intervals) = Self::extract_demands(values);
        if demands.len() < 2 || intervals.is_empty() {
            return f64::INFINITY;
        }

        let intervals_f64: Vec<f64> = intervals.iter().map(|&i| i as f64).collect();

        // Compute one-step-ahead forecast errors for demand
        let mut demand_level = demands[0];
        let mut demand_sse = 0.0;
        for &d in demands.iter().skip(1) {
            let error = d - demand_level;
            demand_sse += error * error;
            demand_level = alpha * d + (1.0 - alpha) * demand_level;
        }

        // Compute one-step-ahead forecast errors for intervals
        let mut interval_level = intervals_f64[0];
        let mut interval_sse = 0.0;
        for &i in intervals_f64.iter().skip(1) {
            let error = i - interval_level;
            interval_sse += error * error;
            interval_level = alpha * i + (1.0 - alpha) * interval_level;
        }

        (demand_sse + interval_sse) / (demands.len() + intervals.len() - 2) as f64
    }

    /// Optimize alpha using Nelder-Mead.
    fn optimize_alpha(values: &[f64]) -> f64 {
        let objective = |params: &[f64]| {
            let alpha = params[0];
            if alpha <= 0.01 || alpha >= 0.99 {
                return f64::INFINITY;
            }
            Self::compute_mse(values, alpha)
        };

        let config = NelderMeadConfig {
            tolerance: 1e-4,
            ..Default::default()
        };

        let result = nelder_mead(objective, &[0.1], Some(&[(0.01, 0.99)]), config);
        result.optimal_point[0].clamp(0.01, 0.99)
    }

    /// Compute fitted values for the original series.
    fn compute_fitted(&self, values: &[f64], alpha: f64) -> Vec<f64> {
        let n = values.len();
        let mut fitted = vec![0.0; n];

        let (demands, intervals) = Self::extract_demands(values);
        if demands.is_empty() {
            return fitted;
        }

        let intervals_f64: Vec<f64> = intervals.iter().map(|&i| i as f64).collect();

        // Initialize levels
        let mut demand_level = demands[0];
        let mut interval_level = if intervals_f64.is_empty() {
            1.0
        } else {
            intervals_f64[0]
        };

        // Compute forecast at each point
        let mut demand_idx = 0;
        let mut _interval_idx = 0;
        let mut periods_since_demand = 0;

        for i in 0..n {
            // Compute forecast
            let forecast = self.apply_bias_correction(demand_level, interval_level, alpha);
            fitted[i] = forecast;

            // Update if this is a demand point
            if values[i] > 0.0 {
                if demand_idx > 0 {
                    demand_level = alpha * values[i] + (1.0 - alpha) * demand_level;
                    interval_level =
                        alpha * (periods_since_demand as f64) + (1.0 - alpha) * interval_level;
                    _interval_idx += 1;
                }
                demand_idx += 1;
                periods_since_demand = 0;
            }
            periods_since_demand += 1;
        }

        fitted
    }

    /// Apply bias correction based on variant.
    fn apply_bias_correction(&self, demand_level: f64, interval_level: f64, alpha: f64) -> f64 {
        let base_forecast = demand_level / interval_level.max(0.001);
        match self.variant {
            CrostonVariant::Classic => base_forecast,
            CrostonVariant::SBA | CrostonVariant::SBAOptimized => {
                // SBA bias correction factor
                base_forecast * (1.0 - alpha / 2.0)
            }
        }
    }
}

impl Default for Croston {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for Croston {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        if values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
                got: values.len(),
            });
        }

        // Extract demands and intervals
        let (demands, intervals) = Self::extract_demands(values);

        if demands.len() < 2 {
            return Err(ForecastError::ComputationError(
                "Insufficient demand occurrences (need at least 2)".to_string(),
            ));
        }

        if intervals.is_empty() {
            return Err(ForecastError::ComputationError(
                "No intervals between demands".to_string(),
            ));
        }

        // Optimize alpha if requested
        if self.optimize_alpha {
            self.alpha = Self::optimize_alpha(values);
        }

        let intervals_f64: Vec<f64> = intervals.iter().map(|&i| i as f64).collect();

        // Fit SES to demands and intervals
        self.demand_level = Some(Self::fit_ses(&demands, self.alpha));
        self.interval_level = Some(Self::fit_ses(&intervals_f64, self.alpha));

        // Compute fitted values
        let fitted = self.compute_fitted(values, self.alpha);

        // Compute residuals
        let residuals: Vec<f64> = values
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| y - f)
            .collect();

        self.fitted = Some(fitted);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let demand_level = self.demand_level.ok_or(ForecastError::FitRequired)?;
        let interval_level = self.interval_level.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        // Croston produces flat forecasts
        let forecast_value = self.apply_bias_correction(demand_level, interval_level, self.alpha);
        let values = vec![forecast_value; horizon];

        Ok(Forecast::from_values(values))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let point_forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        // Compute variance from residuals
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let variance = if residuals.len() > 1 {
            let mean_resid: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
            residuals
                .iter()
                .map(|r| (r - mean_resid).powi(2))
                .sum::<f64>()
                / (residuals.len() - 1) as f64
        } else {
            1.0
        };
        let std_dev = variance.sqrt();

        let z = crate::utils::quantile_normal(0.5 + level / 2.0);

        let lower: Vec<f64> = point_forecast
            .primary()
            .iter()
            .map(|&f| f - z * std_dev)
            .collect();
        let upper: Vec<f64> = point_forecast
            .primary()
            .iter()
            .map(|&f| f + z * std_dev)
            .collect();

        Ok(Forecast::from_values_with_intervals(
            point_forecast.primary().to_vec(),
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        match self.variant {
            CrostonVariant::Classic => {
                if self.optimize_alpha {
                    "Croston (Optimized)"
                } else {
                    "Croston"
                }
            }
            CrostonVariant::SBA => "Croston SBA",
            CrostonVariant::SBAOptimized => "Croston SBA (Optimized)",
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

    fn make_intermittent_series() -> TimeSeries {
        let timestamps = make_timestamps(20);
        // Intermittent demand pattern: demand occurs at irregular intervals
        let values = vec![
            5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
        ];
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn croston_basic() {
        let ts = make_intermittent_series();
        let mut model = Croston::new();
        model.fit(&ts).unwrap();

        assert!(model.demand_level().is_some());
        assert!(model.interval_level().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);

        // Forecast should be positive for intermittent demand
        for &v in forecast.primary() {
            assert!(v > 0.0, "Forecast should be positive");
        }
    }

    #[test]
    fn croston_flat_forecast() {
        let ts = make_intermittent_series();
        let mut model = Croston::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(10).unwrap();

        // Croston produces flat forecasts
        let first = forecast.primary()[0];
        for &v in forecast.primary() {
            assert!(
                (v - first).abs() < 1e-10,
                "Croston should produce flat forecasts"
            );
        }
    }

    #[test]
    fn croston_with_alpha() {
        let ts = make_intermittent_series();
        let mut model = Croston::new().with_alpha(0.2);
        model.fit(&ts).unwrap();

        assert!((model.alpha() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn croston_optimized() {
        let ts = make_intermittent_series();
        let mut model = Croston::new().optimized();
        model.fit(&ts).unwrap();

        // Alpha should be within valid range
        assert!(model.alpha() > 0.0 && model.alpha() < 1.0);
    }

    #[test]
    fn croston_sba() {
        let ts = make_intermittent_series();
        let mut model = Croston::new().sba();
        model.fit(&ts).unwrap();

        assert_eq!(model.variant(), CrostonVariant::SBA);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn croston_sba_lower_than_classic() {
        let ts = make_intermittent_series();

        let mut classic = Croston::new().with_alpha(0.1);
        classic.fit(&ts).unwrap();
        let classic_forecast = classic.predict(1).unwrap();

        let mut sba = Croston::new().with_alpha(0.1).sba();
        sba.fit(&ts).unwrap();
        let sba_forecast = sba.predict(1).unwrap();

        // SBA applies bias correction (1 - alpha/2), so forecast should be lower
        assert!(
            sba_forecast.primary()[0] < classic_forecast.primary()[0],
            "SBA forecast should be lower due to bias correction"
        );
    }

    #[test]
    fn croston_sba_optimized() {
        let ts = make_intermittent_series();
        let mut model = Croston::new().sba_optimized();
        model.fit(&ts).unwrap();

        assert_eq!(model.variant(), CrostonVariant::SBAOptimized);
    }

    #[test]
    fn croston_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 0.0, 2.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Croston::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn croston_no_demands() {
        let timestamps = make_timestamps(10);
        let values = vec![0.0; 10]; // All zeros
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Croston::new();
        assert!(model.fit(&ts).is_err());
    }

    #[test]
    fn croston_requires_fit() {
        let model = Croston::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn croston_zero_horizon() {
        let ts = make_intermittent_series();
        let mut model = Croston::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn croston_confidence_intervals() {
        let ts = make_intermittent_series();
        let mut model = Croston::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        // Lower < point < upper
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        for i in 0..5 {
            assert!(lower[i] <= forecast.primary()[i]);
            assert!(forecast.primary()[i] <= upper[i]);
        }
    }

    #[test]
    fn croston_fitted_and_residuals() {
        let ts = make_intermittent_series();
        let mut model = Croston::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 20);
        assert_eq!(model.residuals().unwrap().len(), 20);
    }

    #[test]
    fn croston_name() {
        let classic = Croston::new();
        assert_eq!(classic.name(), "Croston");

        let optimized = Croston::new().optimized();
        assert_eq!(optimized.name(), "Croston (Optimized)");

        let sba = Croston::new().sba();
        assert_eq!(sba.name(), "Croston SBA");

        let sba_opt = Croston::new().sba_optimized();
        assert_eq!(sba_opt.name(), "Croston SBA (Optimized)");
    }

    #[test]
    fn croston_default() {
        let model = Croston::default();
        assert!((model.alpha() - 0.1).abs() < 1e-10);
        assert_eq!(model.variant(), CrostonVariant::Classic);
    }

    #[test]
    fn croston_alpha_clamped() {
        let model = Croston::new().with_alpha(2.0);
        assert!(model.alpha() <= 0.99);

        let model = Croston::new().with_alpha(-0.5);
        assert!(model.alpha() >= 0.01);
    }
}
