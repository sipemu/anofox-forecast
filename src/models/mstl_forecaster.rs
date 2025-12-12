//! MSTL Forecaster - Decomposition-based forecasting with multiple seasonality.
//!
//! MSTLForecaster uses Multiple Seasonal-Trend decomposition using LOESS (MSTL)
//! to decompose a time series, then forecasts each component separately and
//! combines them.
//!
//! This matches the statsforecast MSTL implementation which uses MSTL decomposition
//! followed by trend forecasting (default: AutoETS).

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::exponential::{AutoETS, AutoETSConfig, SimpleExponentialSmoothing};
use crate::models::Forecaster;
use crate::seasonality::{MSTLResult, MSTL};

/// Method for forecasting the deseasonalized (trend + remainder) component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendForecastMethod {
    /// Use AutoETS for trend forecasting (default, most accurate).
    #[default]
    AutoETS,
    /// Use Simple Exponential Smoothing (faster).
    SES,
    /// Use linear extrapolation.
    Linear,
    /// Use last value (naive).
    Naive,
}

/// Method for forecasting seasonal components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeasonalForecastMethod {
    /// Repeat the last seasonal cycle (default, matches statsforecast).
    #[default]
    Naive,
    /// Average of all seasonal cycles.
    Average,
}

/// MSTL-based forecaster for time series with multiple seasonalities.
///
/// The MSTL forecaster:
/// 1. Decomposes the series using MSTL into trend, multiple seasonal components, and remainder
/// 2. Forecasts the deseasonalized series (trend + remainder) using the specified method
/// 3. Projects seasonal components forward using naive repetition
/// 4. Combines forecasts: forecast = trend_forecast + sum(seasonal_forecasts)
///
/// # Example
/// ```
/// use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// // Create hourly data with daily seasonality
/// let timestamps: Vec<_> = (0..100).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::hours(i)).collect();
/// let values: Vec<f64> = (0..100).map(|i| {
///     50.0 + 0.1 * i as f64  // trend
///         + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin()  // daily
/// }).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = MSTLForecaster::new(vec![24]);
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(24).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MSTLForecaster {
    /// Seasonal periods for MSTL decomposition.
    seasonal_periods: Vec<usize>,
    /// Number of MSTL iterations.
    mstl_iterations: usize,
    /// Use robust fitting in MSTL.
    robust: bool,
    /// Method for forecasting trend component.
    trend_method: TrendForecastMethod,
    /// Method for forecasting seasonal components.
    seasonal_method: SeasonalForecastMethod,
    /// Decomposition result.
    decomposition: Option<MSTLResult>,
    /// Fitted trend forecaster.
    trend_forecaster: Option<Box<dyn TrendForecasterTrait>>,
    /// Original series length.
    n: usize,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance for confidence intervals.
    residual_variance: Option<f64>,
}

/// Internal trait for trend forecasters (to allow different types).
trait TrendForecasterTrait: std::fmt::Debug + Send + Sync {
    fn predict(&self, horizon: usize) -> Result<Vec<f64>>;
    fn clone_box(&self) -> Box<dyn TrendForecasterTrait>;
}

impl Clone for Box<dyn TrendForecasterTrait> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Wrapper for AutoETS trend forecaster.
#[derive(Debug, Clone)]
struct AutoETSTrendForecaster {
    model: AutoETS,
}

impl TrendForecasterTrait for AutoETSTrendForecaster {
    fn predict(&self, horizon: usize) -> Result<Vec<f64>> {
        let forecast = self.model.predict(horizon)?;
        Ok(forecast.primary().to_vec())
    }

    fn clone_box(&self) -> Box<dyn TrendForecasterTrait> {
        Box::new(self.clone())
    }
}

/// Wrapper for SES trend forecaster.
#[derive(Debug, Clone)]
struct SESTrendForecaster {
    model: SimpleExponentialSmoothing,
}

impl TrendForecasterTrait for SESTrendForecaster {
    fn predict(&self, horizon: usize) -> Result<Vec<f64>> {
        let forecast = self.model.predict(horizon)?;
        Ok(forecast.primary().to_vec())
    }

    fn clone_box(&self) -> Box<dyn TrendForecasterTrait> {
        Box::new(self.clone())
    }
}

/// Linear trend forecaster.
#[derive(Debug, Clone)]
struct LinearTrendForecaster {
    intercept: f64,
    slope: f64,
    n: usize,
}

impl TrendForecasterTrait for LinearTrendForecaster {
    fn predict(&self, horizon: usize) -> Result<Vec<f64>> {
        let mut forecasts = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let t = (self.n + h) as f64;
            forecasts.push(self.intercept + self.slope * t);
        }
        Ok(forecasts)
    }

    fn clone_box(&self) -> Box<dyn TrendForecasterTrait> {
        Box::new(self.clone())
    }
}

/// Naive trend forecaster (last value).
#[derive(Debug, Clone)]
struct NaiveTrendForecaster {
    last_value: f64,
}

impl TrendForecasterTrait for NaiveTrendForecaster {
    fn predict(&self, horizon: usize) -> Result<Vec<f64>> {
        Ok(vec![self.last_value; horizon])
    }

    fn clone_box(&self) -> Box<dyn TrendForecasterTrait> {
        Box::new(self.clone())
    }
}

impl MSTLForecaster {
    /// Create a new MSTL forecaster with the given seasonal periods.
    ///
    /// # Arguments
    /// * `seasonal_periods` - Vector of seasonal periods (e.g., [24, 168] for daily/weekly in hourly data)
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        Self {
            seasonal_periods,
            mstl_iterations: 2,
            robust: false,
            trend_method: TrendForecastMethod::AutoETS,
            seasonal_method: SeasonalForecastMethod::Naive,
            decomposition: None,
            trend_forecaster: None,
            n: 0,
            fitted: None,
            residuals: None,
            residual_variance: None,
        }
    }

    /// Set the number of MSTL iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.mstl_iterations = iterations;
        self
    }

    /// Enable robust fitting in MSTL decomposition.
    pub fn robust(mut self) -> Self {
        self.robust = true;
        self
    }

    /// Set the trend forecasting method.
    pub fn with_trend_method(mut self, method: TrendForecastMethod) -> Self {
        self.trend_method = method;
        self
    }

    /// Set the seasonal forecasting method.
    pub fn with_seasonal_method(mut self, method: SeasonalForecastMethod) -> Self {
        self.seasonal_method = method;
        self
    }

    /// Get the decomposition result.
    pub fn decomposition(&self) -> Option<&MSTLResult> {
        self.decomposition.as_ref()
    }

    /// Get seasonal periods.
    pub fn seasonal_periods(&self) -> &[usize] {
        &self.seasonal_periods
    }

    /// Project a seasonal component forward.
    fn project_seasonal(&self, seasonal: &[f64], period: usize, horizon: usize) -> Vec<f64> {
        match self.seasonal_method {
            SeasonalForecastMethod::Naive => {
                // Repeat the last cycle
                let last_cycle_start = seasonal.len().saturating_sub(period);
                let last_cycle = &seasonal[last_cycle_start..];
                (0..horizon)
                    .map(|h| last_cycle[h % last_cycle.len()])
                    .collect()
            }
            SeasonalForecastMethod::Average => {
                // Average all cycles
                let mut avg_cycle = vec![0.0; period];
                let mut counts = vec![0usize; period];
                for (i, &s) in seasonal.iter().enumerate() {
                    avg_cycle[i % period] += s;
                    counts[i % period] += 1;
                }
                for i in 0..period {
                    if counts[i] > 0 {
                        avg_cycle[i] /= counts[i] as f64;
                    }
                }
                (0..horizon).map(|h| avg_cycle[h % period]).collect()
            }
        }
    }

    /// Fit a linear trend model.
    fn fit_linear(values: &[f64]) -> (f64, f64) {
        let n = values.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = values.iter().sum::<f64>() / n as f64;

        let mut ss_xx = 0.0;
        let mut ss_xy = 0.0;
        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            ss_xx += (x - x_mean).powi(2);
            ss_xy += (x - x_mean) * (y - y_mean);
        }

        let slope = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        let intercept = y_mean - slope * x_mean;

        (intercept, slope)
    }
}

impl Default for MSTLForecaster {
    fn default() -> Self {
        Self::new(vec![12])
    }
}

impl Forecaster for MSTLForecaster {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        if self.seasonal_periods.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "At least one seasonal period is required".to_string(),
            ));
        }

        let max_period = *self.seasonal_periods.iter().max().unwrap_or(&1);
        if values.len() < 2 * max_period {
            return Err(ForecastError::InsufficientData {
                needed: 2 * max_period,
                got: values.len(),
            });
        }

        // Create MSTL decomposer
        let mut mstl =
            MSTL::new(self.seasonal_periods.clone()).with_iterations(self.mstl_iterations);
        if self.robust {
            mstl = mstl.robust();
        }

        // Decompose
        let decomposition = mstl.decompose(values).ok_or_else(|| {
            ForecastError::ComputationError("MSTL decomposition failed".to_string())
        })?;

        // Create deseasonalized series (trend + remainder)
        let deseasonalized: Vec<f64> = decomposition
            .trend
            .iter()
            .zip(decomposition.remainder.iter())
            .map(|(t, r)| t + r)
            .collect();

        // Create a temporary time series for the deseasonalized component
        let deseas_ts =
            TimeSeries::univariate(series.timestamps().to_vec(), deseasonalized.clone())?;

        // Fit trend forecaster based on method
        let trend_forecaster: Box<dyn TrendForecasterTrait> = match self.trend_method {
            TrendForecastMethod::AutoETS => {
                let config = AutoETSConfig::non_seasonal();
                let mut model = AutoETS::with_config(config);
                model.fit(&deseas_ts)?;
                Box::new(AutoETSTrendForecaster { model })
            }
            TrendForecastMethod::SES => {
                let mut model = SimpleExponentialSmoothing::new(0.3);
                model.fit(&deseas_ts)?;
                Box::new(SESTrendForecaster { model })
            }
            TrendForecastMethod::Linear => {
                let (intercept, slope) = Self::fit_linear(&deseasonalized);
                Box::new(LinearTrendForecaster {
                    intercept,
                    slope,
                    n: self.n,
                })
            }
            TrendForecastMethod::Naive => {
                let last_value = *deseasonalized.last().unwrap_or(&0.0);
                Box::new(NaiveTrendForecaster { last_value })
            }
        };

        // Compute fitted values (reconstruct from components)
        let fitted: Vec<f64> = (0..self.n)
            .map(|i| {
                let mut val = decomposition.trend[i] + decomposition.remainder[i];
                for seasonal in &decomposition.seasonal_components {
                    val += seasonal[i];
                }
                val
            })
            .collect();

        // Compute residuals
        let residuals: Vec<f64> = values
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| y - f)
            .collect();

        // Compute residual variance
        if residuals.len() > 1 {
            let variance = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.decomposition = Some(decomposition);
        self.trend_forecaster = Some(trend_forecaster);
        self.fitted = Some(fitted);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let decomposition = self
            .decomposition
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;
        let trend_forecaster = self
            .trend_forecaster
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Forecast deseasonalized component
        let trend_forecast = trend_forecaster.predict(horizon)?;

        // Project each seasonal component
        let mut seasonal_forecasts: Vec<Vec<f64>> = Vec::new();
        for (idx, seasonal) in decomposition.seasonal_components.iter().enumerate() {
            let period = decomposition.seasonal_periods[idx];
            let seasonal_forecast = self.project_seasonal(seasonal, period, horizon);
            seasonal_forecasts.push(seasonal_forecast);
        }

        // Combine forecasts
        let mut forecasts = trend_forecast;
        for seasonal_forecast in &seasonal_forecasts {
            for (i, &s) in seasonal_forecast.iter().enumerate() {
                forecasts[i] += s;
            }
        }

        Ok(Forecast::from_values(forecasts))
    }

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 || variance <= 0.0 {
            return Ok(forecast);
        }

        let z = crate::utils::stats::quantile_normal((1.0 + confidence) / 2.0);
        let se = variance.sqrt();

        let preds = forecast.primary();
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        // Simple constant variance intervals (could be improved with fan-out)
        for h in 0..horizon {
            // Variance increases with horizon (simple approximation)
            let h_factor = (1.0 + 0.1 * h as f64).sqrt();
            lower.push(preds[h] - z * se * h_factor);
            upper.push(preds[h] + z * se * h_factor);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
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
        "MSTLForecaster"
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

    fn make_multi_seasonal_series(n: usize, periods: &[usize]) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 50.0 + 0.1 * i as f64;
                let mut seasonal = 0.0;
                for (idx, &period) in periods.iter().enumerate() {
                    let amplitude = 5.0 / (idx + 1) as f64;
                    seasonal +=
                        amplitude * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                }
                trend + seasonal
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn mstl_forecaster_basic() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mstl_forecaster_multiple_seasonalities() {
        let ts = make_multi_seasonal_series(200, &[12, 24]);
        let mut model = MSTLForecaster::new(vec![12, 24]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);

        // Decomposition should have 2 seasonal components
        let decomp = model.decomposition().unwrap();
        assert_eq!(decomp.seasonal_components.len(), 2);
    }

    #[test]
    fn mstl_forecaster_with_ses() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]).with_trend_method(TrendForecastMethod::SES);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mstl_forecaster_with_linear() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model =
            MSTLForecaster::new(vec![12]).with_trend_method(TrendForecastMethod::Linear);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mstl_forecaster_with_naive() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]).with_trend_method(TrendForecastMethod::Naive);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);

        // Naive produces flat trend component
        // (though seasonal pattern still varies)
    }

    #[test]
    fn mstl_forecaster_robust() {
        // Generate series with outliers
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let timestamps: Vec<_> = (0..100).map(|i| base + Duration::hours(i)).collect();
        let mut values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 50.0 + 0.5 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin();
                trend + seasonal
            })
            .collect();
        // Add outliers
        values[30] = 200.0;
        values[60] = -50.0;
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MSTLForecaster::new(vec![12]).robust();
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mstl_forecaster_confidence_intervals() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(12, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..12 {
            assert!(
                lower[i] <= preds[i],
                "Lower bound {} should be <= prediction {} at index {}",
                lower[i],
                preds[i],
                i
            );
            assert!(
                upper[i] >= preds[i],
                "Upper bound {} should be >= prediction {} at index {}",
                upper[i],
                preds[i],
                i
            );
        }
    }

    #[test]
    fn mstl_forecaster_fitted_residuals() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 100);
        assert_eq!(model.residuals().unwrap().len(), 100);
    }

    #[test]
    fn mstl_forecaster_insufficient_data() {
        let ts = make_multi_seasonal_series(20, &[12]);
        let mut model = MSTLForecaster::new(vec![12]);
        assert!(model.fit(&ts).is_err());
    }

    #[test]
    fn mstl_forecaster_empty_periods() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![]);
        assert!(model.fit(&ts).is_err());
    }

    #[test]
    fn mstl_forecaster_requires_fit() {
        let model = MSTLForecaster::new(vec![12]);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn mstl_forecaster_zero_horizon() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn mstl_forecaster_name() {
        let model = MSTLForecaster::new(vec![12]);
        assert_eq!(model.name(), "MSTLForecaster");
    }

    #[test]
    fn mstl_forecaster_seasonal_average_method() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model =
            MSTLForecaster::new(vec![12]).with_seasonal_method(SeasonalForecastMethod::Average);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mstl_forecaster_with_iterations() {
        let ts = make_multi_seasonal_series(100, &[12]);
        let mut model = MSTLForecaster::new(vec![12]).with_iterations(3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }
}
