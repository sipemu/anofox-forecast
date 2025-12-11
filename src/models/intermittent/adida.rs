//! ADIDA (Aggregate-Disaggregate Intermittent Demand Approach).
//!
//! ADIDA aggregates the time series to a lower frequency where demand
//! is less intermittent, applies SES, then disaggregates the forecast.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// ADIDA method for intermittent demand forecasting.
#[derive(Debug, Clone)]
pub struct ADIDA {
    /// Smoothing parameter for SES.
    alpha: f64,
    /// Aggregation level (automatically determined or fixed).
    aggregation_level: Option<usize>,
    /// Whether to automatically determine aggregation level.
    auto_aggregate: bool,
    /// Final forecast level.
    forecast_level: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
}

impl ADIDA {
    /// Create a new ADIDA model with default settings.
    pub fn new() -> Self {
        Self {
            alpha: 0.1,
            aggregation_level: None,
            auto_aggregate: true,
            forecast_level: None,
            fitted: None,
            residuals: None,
            n: 0,
        }
    }

    /// Set the smoothing parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.01, 0.99);
        self
    }

    /// Set a fixed aggregation level.
    pub fn with_aggregation_level(mut self, level: usize) -> Self {
        self.aggregation_level = Some(level.max(1));
        self.auto_aggregate = false;
        self
    }

    /// Get the smoothing parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the aggregation level used.
    pub fn aggregation_level(&self) -> Option<usize> {
        self.aggregation_level
    }

    /// Get the forecast level.
    pub fn forecast_level(&self) -> Option<f64> {
        self.forecast_level
    }

    /// Calculate mean inter-demand interval.
    fn mean_inter_demand_interval(values: &[f64]) -> f64 {
        let mut intervals = Vec::new();
        let mut last_demand_idx: Option<usize> = None;

        for (i, &v) in values.iter().enumerate() {
            if v > 0.0 {
                if let Some(last_idx) = last_demand_idx {
                    intervals.push((i - last_idx) as f64);
                }
                last_demand_idx = Some(i);
            }
        }

        if intervals.is_empty() {
            1.0
        } else {
            intervals.iter().sum::<f64>() / intervals.len() as f64
        }
    }

    /// Aggregate the series to a lower frequency.
    fn aggregate(values: &[f64], level: usize) -> Vec<f64> {
        if level <= 1 {
            return values.to_vec();
        }

        values
            .chunks(level)
            .map(|chunk| chunk.iter().sum())
            .collect()
    }

    /// Disaggregate the forecast to original frequency.
    fn disaggregate(aggregated_forecast: f64, level: usize) -> f64 {
        aggregated_forecast / level as f64
    }

    /// Fit SES to aggregated series.
    fn fit_ses(values: &[f64], alpha: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut level = values[0];
        for &v in values.iter().skip(1) {
            level = alpha * v + (1.0 - alpha) * level;
        }
        level
    }
}

impl Default for ADIDA {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for ADIDA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        if values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
                got: values.len(),
            });
        }

        // Count demands
        let demand_count = values.iter().filter(|&&v| v > 0.0).count();
        if demand_count < 2 {
            return Err(ForecastError::ComputationError(
                "Insufficient demand occurrences (need at least 2)".to_string(),
            ));
        }

        // Determine aggregation level
        let level = if self.auto_aggregate {
            let mean_interval = Self::mean_inter_demand_interval(values);
            (mean_interval.round() as usize)
                .max(1)
                .min(values.len() / 2)
        } else {
            self.aggregation_level.unwrap_or(1)
        };
        self.aggregation_level = Some(level);

        // Aggregate the series
        let aggregated = Self::aggregate(values, level);

        if aggregated.len() < 2 {
            return Err(ForecastError::ComputationError(
                "Aggregated series too short".to_string(),
            ));
        }

        // Fit SES to aggregated series
        let aggregated_forecast = Self::fit_ses(&aggregated, self.alpha);

        // Disaggregate the forecast
        let forecast = Self::disaggregate(aggregated_forecast, level);
        self.forecast_level = Some(forecast);

        // Compute fitted values (constant forecast at original frequency)
        let fitted = vec![forecast; values.len()];
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
        let forecast_level = self.forecast_level.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        // ADIDA produces flat forecasts
        let values = vec![forecast_level; horizon];
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
        "ADIDA"
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
    fn adida_basic() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        assert!(model.forecast_level().is_some());
        assert!(model.aggregation_level().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn adida_flat_forecast() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(10).unwrap();

        // ADIDA produces flat forecasts
        let first = forecast.primary()[0];
        for &v in forecast.primary() {
            assert!(
                (v - first).abs() < 1e-10,
                "ADIDA should produce flat forecasts"
            );
        }
    }

    #[test]
    fn adida_with_alpha() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new().with_alpha(0.3);
        model.fit(&ts).unwrap();

        assert!((model.alpha() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn adida_with_fixed_aggregation() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new().with_aggregation_level(4);
        model.fit(&ts).unwrap();

        assert_eq!(model.aggregation_level(), Some(4));
    }

    #[test]
    fn adida_auto_aggregation() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        // Auto-aggregation should determine a reasonable level
        let level = model.aggregation_level().unwrap();
        assert!(level >= 1);
    }

    #[test]
    fn adida_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 0.0, 2.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ADIDA::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn adida_no_demands() {
        let timestamps = make_timestamps(10);
        let values = vec![0.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ADIDA::new();
        assert!(model.fit(&ts).is_err());
    }

    #[test]
    fn adida_requires_fit() {
        let model = ADIDA::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn adida_zero_horizon() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn adida_confidence_intervals() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn adida_fitted_and_residuals() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 20);
    }

    #[test]
    fn adida_name() {
        let model = ADIDA::new();
        assert_eq!(model.name(), "ADIDA");
    }

    #[test]
    fn adida_default() {
        let model = ADIDA::default();
        assert!((model.alpha() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn adida_positive_forecast() {
        let ts = make_intermittent_series();
        let mut model = ADIDA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        for &v in forecast.primary() {
            assert!(v > 0.0, "Forecast should be positive for demand data");
        }
    }
}
