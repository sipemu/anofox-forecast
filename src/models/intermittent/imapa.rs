//! IMAPA (Intermittent Multiple Aggregation Prediction Algorithm).
//!
//! IMAPA extends ADIDA by testing multiple aggregation levels from 1 to the
//! mean inter-demand interval, applying optimized SES at each level, and
//! averaging the forecasts.
//!
//! Reference: Petropoulos & Kourentzes (2015) "Forecast combinations for
//! intermittent demand"

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// IMAPA method for intermittent demand forecasting.
///
/// IMAPA provides more robust forecasts than ADIDA by:
/// 1. Testing multiple aggregation levels (1 to mean_interval)
/// 2. Optimizing SES alpha at each level
/// 3. Averaging forecasts across all levels
///
/// # Example
/// ```
/// use anofox_forecast::models::intermittent::IMAPA;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..20).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
/// // Intermittent demand pattern
/// let values = vec![5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 3.0, 0.0, 0.0];
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = IMAPA::new();
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct IMAPA {
    /// Maximum aggregation level to consider.
    max_aggregation: Option<usize>,
    /// Whether to auto-determine max aggregation.
    auto_max: bool,
    /// Forecasts at each aggregation level.
    level_forecasts: Option<Vec<(usize, f64)>>,
    /// Combined forecast level.
    forecast_level: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
}

impl IMAPA {
    /// Create a new IMAPA model with automatic max aggregation.
    pub fn new() -> Self {
        Self {
            max_aggregation: None,
            auto_max: true,
            level_forecasts: None,
            forecast_level: None,
            fitted: None,
            residuals: None,
            n: 0,
        }
    }

    /// Set a fixed maximum aggregation level.
    pub fn with_max_aggregation(mut self, max_level: usize) -> Self {
        self.max_aggregation = Some(max_level.max(1));
        self.auto_max = false;
        self
    }

    /// Get the combined forecast level.
    pub fn forecast_level(&self) -> Option<f64> {
        self.forecast_level
    }

    /// Get the forecasts at each aggregation level.
    pub fn level_forecasts(&self) -> Option<&[(usize, f64)]> {
        self.level_forecasts.as_deref()
    }

    /// Get the max aggregation level used.
    pub fn max_aggregation(&self) -> Option<usize> {
        self.max_aggregation
    }

    /// Calculate inter-demand intervals (statsforecast compatible).
    /// Returns intervals between non-zero values, with first interval
    /// being distance from index 0 to first non-zero.
    fn intervals(values: &[f64]) -> Vec<f64> {
        let nonzero_idxs: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| i)
            .collect();

        if nonzero_idxs.is_empty() {
            return vec![];
        }

        // First interval is index + 1 (1-based)
        let mut intervals = vec![(nonzero_idxs[0] + 1) as f64];

        // Subsequent intervals are differences
        for i in 1..nonzero_idxs.len() {
            intervals.push((nonzero_idxs[i] - nonzero_idxs[i - 1]) as f64);
        }

        intervals
    }

    /// Aggregate the series to a lower frequency.
    /// Drops remainder at the BEGINNING to match statsforecast.
    fn aggregate(values: &[f64], level: usize) -> Vec<f64> {
        if level <= 1 {
            return values.to_vec();
        }

        let lost_remainder = values.len() % level;
        let y_cut = &values[lost_remainder..];

        y_cut
            .chunks(level)
            .map(|chunk| chunk.iter().sum())
            .collect()
    }

    /// Compute SES SSE for optimization.
    fn ses_sse(values: &[f64], alpha: f64) -> f64 {
        if values.len() < 2 {
            return f64::MAX;
        }

        let mut level = values[0];
        let mut sse = 0.0;

        for &v in values.iter().skip(1) {
            let error = v - level;
            sse += error * error;
            level = alpha * v + (1.0 - alpha) * level;
        }

        sse
    }

    /// Optimize alpha by grid search in (0.1, 0.3) range.
    fn optimize_alpha(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.1;
        }

        let mut best_alpha = 0.1;
        let mut best_sse = f64::MAX;

        // Grid search with fine granularity
        let steps = 100;
        for i in 0..=steps {
            let alpha = 0.1 + (0.3 - 0.1) * (i as f64 / steps as f64);
            let sse = Self::ses_sse(values, alpha);
            if sse < best_sse {
                best_sse = sse;
                best_alpha = alpha;
            }
        }

        best_alpha
    }

    /// Fit SES with given alpha, returns forecast.
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

    /// Disaggregate the forecast to original frequency.
    fn disaggregate(aggregated_forecast: f64, level: usize) -> f64 {
        aggregated_forecast / level as f64
    }
}

impl Default for IMAPA {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for IMAPA {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        if values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
                got: values.len(),
            });
        }

        // Check for all zeros
        if values.iter().all(|&v| v == 0.0) {
            self.forecast_level = Some(0.0);
            self.fitted = Some(vec![0.0; values.len()]);
            self.residuals = Some(vec![0.0; values.len()]);
            self.max_aggregation = Some(1);
            self.level_forecasts = Some(vec![(1, 0.0)]);
            return Ok(());
        }

        // Calculate intervals and mean interval (statsforecast compatible)
        let intervals = Self::intervals(values);
        if intervals.is_empty() {
            return Err(ForecastError::ComputationError(
                "No non-zero demand values found".to_string(),
            ));
        }

        let mean_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;

        // Determine max aggregation level
        let max_level = if self.auto_max {
            (mean_interval.round() as usize).max(1)
        } else {
            self.max_aggregation.unwrap_or(1)
        };
        self.max_aggregation = Some(max_level);

        // Test each aggregation level from 1 to max_level
        let mut level_forecasts = Vec::new();

        for agg_level in 1..=max_level {
            let aggregated = Self::aggregate(values, agg_level);

            if aggregated.is_empty() {
                continue;
            }

            // Optimize SES for this aggregation level
            let alpha = Self::optimize_alpha(&aggregated);
            let agg_forecast = Self::fit_ses(&aggregated, alpha);

            // Disaggregate to original frequency
            let forecast = Self::disaggregate(agg_forecast, agg_level);

            level_forecasts.push((agg_level, forecast));
        }

        if level_forecasts.is_empty() {
            return Err(ForecastError::ComputationError(
                "No valid aggregation levels found".to_string(),
            ));
        }

        // Average forecasts across all levels
        let combined_forecast: f64 =
            level_forecasts.iter().map(|(_, f)| f).sum::<f64>() / level_forecasts.len() as f64;

        self.level_forecasts = Some(level_forecasts);
        self.forecast_level = Some(combined_forecast);

        // Compute fitted values (constant forecast at original frequency)
        let fitted = vec![combined_forecast; values.len()];
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

        // IMAPA produces flat forecasts
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
        "IMAPA"
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
    fn imapa_basic() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        assert!(model.forecast_level().is_some());
        assert!(model.level_forecasts().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn imapa_flat_forecast() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(10).unwrap();

        // IMAPA produces flat forecasts
        let first = forecast.primary()[0];
        for &v in forecast.primary() {
            assert!(
                (v - first).abs() < 1e-10,
                "IMAPA should produce flat forecasts"
            );
        }
    }

    #[test]
    fn imapa_multiple_aggregation_levels() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        let level_forecasts = model.level_forecasts().unwrap();
        assert!(!level_forecasts.is_empty());

        // Should have multiple levels
        assert!(!level_forecasts.is_empty());
    }

    #[test]
    fn imapa_with_max_aggregation() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new().with_max_aggregation(3);
        model.fit(&ts).unwrap();

        let level_forecasts = model.level_forecasts().unwrap();
        // Should have at most 3 levels
        assert!(level_forecasts.len() <= 3);
    }

    #[test]
    fn imapa_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 0.0, 2.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = IMAPA::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn imapa_no_demands() {
        let timestamps = make_timestamps(10);
        let values = vec![0.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // statsforecast returns 0 for all-zero series
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();
        for &v in forecast.primary() {
            assert!(
                (v - 0.0).abs() < 1e-10,
                "All-zero series should forecast zero"
            );
        }
    }

    #[test]
    fn imapa_requires_fit() {
        let model = IMAPA::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn imapa_zero_horizon() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn imapa_confidence_intervals() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn imapa_fitted_and_residuals() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 20);
    }

    #[test]
    fn imapa_name() {
        let model = IMAPA::new();
        assert_eq!(model.name(), "IMAPA");
    }

    #[test]
    fn imapa_default() {
        let model = IMAPA::default();
        assert!(model.max_aggregation().is_none());
    }

    #[test]
    fn imapa_positive_forecast() {
        let ts = make_intermittent_series();
        let mut model = IMAPA::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        for &v in forecast.primary() {
            assert!(v > 0.0, "Forecast should be positive for demand data");
        }
    }
}
