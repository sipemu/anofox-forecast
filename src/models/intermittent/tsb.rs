//! TSB (Teunter-Syntetos-Babai) method for intermittent demand forecasting.
//!
//! TSB models demand probability and demand size separately, then combines them.
//! Unlike Croston, TSB allows the probability estimate to decrease over time
//! during periods of no demand.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// TSB method for intermittent demand forecasting.
#[derive(Debug, Clone)]
pub struct TSB {
    /// Smoothing parameter for demand size.
    alpha_d: f64,
    /// Smoothing parameter for probability.
    alpha_p: f64,
    /// Whether to optimize parameters.
    optimize: bool,
    /// Estimated demand level.
    demand_level: Option<f64>,
    /// Estimated demand probability.
    probability: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
}

impl TSB {
    /// Create a new TSB model with default parameters.
    pub fn new() -> Self {
        Self {
            alpha_d: 0.1,
            alpha_p: 0.1,
            optimize: false,
            demand_level: None,
            probability: None,
            fitted: None,
            residuals: None,
            n: 0,
        }
    }

    /// Create TSB model with specified parameters.
    pub fn with_params(mut self, alpha_d: f64, alpha_p: f64) -> Self {
        self.alpha_d = alpha_d.clamp(0.01, 0.99);
        self.alpha_p = alpha_p.clamp(0.01, 0.99);
        self.optimize = false;
        self
    }

    /// Enable parameter optimization.
    pub fn optimized(mut self) -> Self {
        self.optimize = true;
        self
    }

    /// Get the demand smoothing parameter.
    pub fn alpha_d(&self) -> f64 {
        self.alpha_d
    }

    /// Get the probability smoothing parameter.
    pub fn alpha_p(&self) -> f64 {
        self.alpha_p
    }

    /// Get the estimated demand level.
    pub fn demand_level(&self) -> Option<f64> {
        self.demand_level
    }

    /// Get the estimated demand probability.
    pub fn probability(&self) -> Option<f64> {
        self.probability
    }

    /// Compute objective (MSE) for parameter optimization.
    fn compute_mse(values: &[f64], alpha_d: f64, alpha_p: f64) -> f64 {
        let n = values.len();
        if n < 4 {
            return f64::INFINITY;
        }

        // Find first demand to initialize
        let first_idx = match values.iter().position(|&v| v > 0.0) {
            Some(idx) => idx,
            None => return f64::INFINITY,
        };

        // Initialize
        let mut demand_level = values[first_idx];
        let mut probability = 0.5; // Start with 50% probability

        let mut sse = 0.0;
        let mut count = 0;

        for &y in values.iter().skip(first_idx + 1) {
            // Compute forecast
            let forecast = demand_level * probability;
            let error = y - forecast;
            sse += error * error;
            count += 1;

            // Update
            if y > 0.0 {
                demand_level = alpha_d * y + (1.0 - alpha_d) * demand_level;
                probability = alpha_p * 1.0 + (1.0 - alpha_p) * probability;
            } else {
                probability = alpha_p * 0.0 + (1.0 - alpha_p) * probability;
            }
        }

        if count == 0 {
            f64::INFINITY
        } else {
            sse / count as f64
        }
    }

    /// Optimize parameters using Nelder-Mead.
    fn optimize_params(values: &[f64]) -> (f64, f64) {
        let objective = |params: &[f64]| {
            let alpha_d = params[0];
            let alpha_p = params[1];
            if alpha_d <= 0.01 || alpha_d >= 0.99 || alpha_p <= 0.01 || alpha_p >= 0.99 {
                return f64::INFINITY;
            }
            Self::compute_mse(values, alpha_d, alpha_p)
        };

        let config = NelderMeadConfig {
            tolerance: 1e-4,
            ..Default::default()
        };

        let result = nelder_mead(
            objective,
            &[0.1, 0.1],
            Some(&[(0.01, 0.99), (0.01, 0.99)]),
            config,
        );
        (
            result.optimal_point[0].clamp(0.01, 0.99),
            result.optimal_point[1].clamp(0.01, 0.99),
        )
    }
}

impl Default for TSB {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for TSB {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        if values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
                got: values.len(),
            });
        }

        // Find first demand
        let first_idx =
            values
                .iter()
                .position(|&v| v > 0.0)
                .ok_or(ForecastError::ComputationError(
                    "No demand occurrences in series".to_string(),
                ))?;

        // Count demands
        let demand_count = values.iter().filter(|&&v| v > 0.0).count();
        if demand_count < 2 {
            return Err(ForecastError::ComputationError(
                "Insufficient demand occurrences (need at least 2)".to_string(),
            ));
        }

        // Optimize if requested
        if self.optimize {
            let (alpha_d, alpha_p) = Self::optimize_params(values);
            self.alpha_d = alpha_d;
            self.alpha_p = alpha_p;
        }

        // Fit the model
        let mut demand_level = values[first_idx];
        let mut probability = 0.5;
        let mut fitted = vec![0.0; values.len()];
        let mut residuals = vec![0.0; values.len()];

        for (i, &y) in values.iter().enumerate() {
            // Compute forecast
            let forecast = demand_level * probability;
            fitted[i] = forecast;
            residuals[i] = y - forecast;

            // Update (skip first demand point for initialization)
            if i >= first_idx {
                if y > 0.0 {
                    demand_level = self.alpha_d * y + (1.0 - self.alpha_d) * demand_level;
                    probability = self.alpha_p * 1.0 + (1.0 - self.alpha_p) * probability;
                } else {
                    probability = self.alpha_p * 0.0 + (1.0 - self.alpha_p) * probability;
                }
            }
        }

        self.demand_level = Some(demand_level);
        self.probability = Some(probability);
        self.fitted = Some(fitted);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let demand_level = self.demand_level.ok_or(ForecastError::FitRequired)?;
        let probability = self.probability.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        // TSB forecast: decreasing probability over time
        let mut values = Vec::with_capacity(horizon);
        let mut prob = probability;

        for _ in 0..horizon {
            values.push(demand_level * prob);
            // Probability decays assuming no demand
            prob *= 1.0 - self.alpha_p;
        }

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
        if self.optimize {
            "TSB (Optimized)"
        } else {
            "TSB"
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
        let values = vec![
            5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
        ];
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn tsb_basic() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        assert!(model.demand_level().is_some());
        assert!(model.probability().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn tsb_decreasing_forecast() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(10).unwrap();

        // TSB forecast should decrease over time (probability decay)
        for i in 1..forecast.horizon() {
            assert!(
                forecast.primary()[i] <= forecast.primary()[i - 1],
                "TSB forecast should decrease: {} > {} at index {}",
                forecast.primary()[i],
                forecast.primary()[i - 1],
                i
            );
        }
    }

    #[test]
    fn tsb_with_params() {
        let ts = make_intermittent_series();
        let mut model = TSB::new().with_params(0.2, 0.3);
        model.fit(&ts).unwrap();

        assert!((model.alpha_d() - 0.2).abs() < 1e-10);
        assert!((model.alpha_p() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn tsb_optimized() {
        let ts = make_intermittent_series();
        let mut model = TSB::new().optimized();
        model.fit(&ts).unwrap();

        // Parameters should be within valid range
        assert!(model.alpha_d() > 0.0 && model.alpha_d() < 1.0);
        assert!(model.alpha_p() > 0.0 && model.alpha_p() < 1.0);
    }

    #[test]
    fn tsb_probability_range() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let prob = model.probability().unwrap();
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability should be in [0, 1]: {}",
            prob
        );
    }

    #[test]
    fn tsb_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 0.0, 2.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TSB::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn tsb_no_demands() {
        let timestamps = make_timestamps(10);
        let values = vec![0.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TSB::new();
        assert!(model.fit(&ts).is_err());
    }

    #[test]
    fn tsb_requires_fit() {
        let model = TSB::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn tsb_zero_horizon() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn tsb_confidence_intervals() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn tsb_fitted_and_residuals() {
        let ts = make_intermittent_series();
        let mut model = TSB::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 20);
    }

    #[test]
    fn tsb_name() {
        let model = TSB::new();
        assert_eq!(model.name(), "TSB");

        let optimized = TSB::new().optimized();
        assert_eq!(optimized.name(), "TSB (Optimized)");
    }

    #[test]
    fn tsb_default() {
        let model = TSB::default();
        assert!((model.alpha_d() - 0.1).abs() < 1e-10);
        assert!((model.alpha_p() - 0.1).abs() < 1e-10);
    }
}
