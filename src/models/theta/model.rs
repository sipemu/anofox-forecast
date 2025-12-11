//! Theta forecasting model.
//!
//! The Theta method was the winning method in the M3 forecasting competition.
//! It decomposes the series using "theta lines" and combines forecasts.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Theta forecasting model.
///
/// The standard Theta method (STM) uses theta=2, which doubles the local curvature
/// of the series. The forecast is a weighted combination of SES forecasts on the
/// theta-modified series and a linear trend.
#[derive(Debug, Clone)]
pub struct Theta {
    /// Theta parameter (default: 2.0 for STM).
    theta: f64,
    /// SES smoothing parameter.
    alpha: Option<f64>,
    /// Whether to optimize alpha.
    optimize: bool,
    /// Seasonal period (0 for non-seasonal).
    seasonal_period: usize,
    /// Fitted drift (linear trend slope).
    drift: Option<f64>,
    /// Fitted level (from SES).
    level: Option<f64>,
    /// Seasonal indices.
    seasonals: Option<Vec<f64>>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Series length.
    n: usize,
}

impl Theta {
    /// Create a new Theta model with default parameters.
    pub fn new() -> Self {
        Self {
            theta: 2.0,
            alpha: None,
            optimize: true,
            seasonal_period: 0,
            drift: None,
            level: None,
            seasonals: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a Theta model with custom theta parameter.
    pub fn with_theta(theta: f64) -> Self {
        Self {
            theta,
            ..Self::new()
        }
    }

    /// Create a seasonal Theta model.
    pub fn seasonal(period: usize) -> Self {
        Self {
            seasonal_period: period,
            ..Self::new()
        }
    }

    /// Create a Theta model with fixed alpha.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            optimize: false,
            ..Self::new()
        }
    }

    /// Get the theta parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    /// Get the drift (linear trend slope).
    pub fn drift(&self) -> Option<f64> {
        self.drift
    }

    /// Deseasonalize a series.
    fn deseasonalize(&self, series: &[f64], seasonals: &[f64]) -> Vec<f64> {
        if seasonals.is_empty() || self.seasonal_period == 0 {
            return series.to_vec();
        }
        series
            .iter()
            .enumerate()
            .map(|(i, &y)| y - seasonals[i % self.seasonal_period])
            .collect()
    }

    /// Reseasonalize forecasts.
    fn reseasonalize(&self, forecasts: &[f64], start_idx: usize, seasonals: &[f64]) -> Vec<f64> {
        if seasonals.is_empty() || self.seasonal_period == 0 {
            return forecasts.to_vec();
        }
        forecasts
            .iter()
            .enumerate()
            .map(|(i, &y)| y + seasonals[(start_idx + i) % self.seasonal_period])
            .collect()
    }

    /// Calculate seasonal indices using classical decomposition.
    fn calculate_seasonals(&self, series: &[f64]) -> Vec<f64> {
        let period = self.seasonal_period;
        if period == 0 || series.len() < 2 * period {
            return vec![];
        }

        // Calculate trend using centered moving average
        let half = period / 2;
        let mut trend = vec![f64::NAN; series.len()];

        for i in half..(series.len() - half) {
            let sum: f64 = if period.is_multiple_of(2) {
                // Even period: weighted endpoints
                let mut s = 0.5 * series[i - half] + 0.5 * series[i + half];
                for &val in series.iter().take(i + half).skip(i - half + 1) {
                    s += val;
                }
                s / period as f64
            } else {
                // Odd period: simple average
                let start = i - period / 2;
                let end = i + period / 2 + 1;
                series[start..end].iter().sum::<f64>() / period as f64
            };
            trend[i] = sum;
        }

        // Calculate detrended series
        let detrended: Vec<f64> = series
            .iter()
            .zip(trend.iter())
            .map(|(&y, &t)| if t.is_nan() { f64::NAN } else { y - t })
            .collect();

        // Average by season
        let mut seasonals = vec![0.0; period];
        let mut counts = vec![0usize; period];

        for (i, &d) in detrended.iter().enumerate() {
            if !d.is_nan() {
                seasonals[i % period] += d;
                counts[i % period] += 1;
            }
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonals[i] /= counts[i] as f64;
            }
        }

        // Normalize to sum to zero
        let mean = seasonals.iter().sum::<f64>() / period as f64;
        for s in &mut seasonals {
            *s -= mean;
        }

        seasonals
    }

    /// Apply theta transformation.
    fn apply_theta(&self, series: &[f64]) -> Vec<f64> {
        let n = series.len();
        if n < 2 {
            return series.to_vec();
        }

        // Calculate linear trend
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = series.iter().sum::<f64>() / n as f64;

        let mut ss_xx = 0.0;
        let mut ss_xy = 0.0;
        for (i, &y) in series.iter().enumerate() {
            let x = i as f64;
            ss_xx += (x - x_mean).powi(2);
            ss_xy += (x - x_mean) * (y - y_mean);
        }

        let slope = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        let intercept = y_mean - slope * x_mean;

        // Theta line: z_t(theta) = theta * y_t + (1 - theta) * L_t
        // where L_t is the linear trend
        series
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let trend = intercept + slope * i as f64;
                self.theta * y + (1.0 - self.theta) * trend
            })
            .collect()
    }

    /// Calculate SSE for SES on theta-modified series.
    fn calculate_sse(series: &[f64], alpha: f64) -> f64 {
        if series.is_empty() {
            return f64::MAX;
        }

        let mut level = series[0];
        let mut sse = 0.0;

        for &y in &series[1..] {
            let error = y - level;
            sse += error * error;
            level = alpha * y + (1.0 - alpha) * level;
        }

        sse
    }

    /// Optimize alpha for SES.
    fn optimize_alpha(series: &[f64]) -> f64 {
        let config = NelderMeadConfig {
            max_iter: 500,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = nelder_mead(
            |params| Self::calculate_sse(series, params[0]),
            &[0.5],
            Some(&[(0.0001, 0.9999)]),
            config,
        );

        result.optimal_point[0].clamp(0.0001, 0.9999)
    }
}

impl Default for Theta {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for Theta {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
                got: values.len(),
            });
        }

        self.n = values.len();

        // Calculate seasonal indices if needed
        let seasonals = if self.seasonal_period > 0 {
            self.calculate_seasonals(values)
        } else {
            vec![]
        };

        // Deseasonalize
        let deseasonalized = self.deseasonalize(values, &seasonals);

        // Calculate linear trend (drift) for forecasting
        let n = deseasonalized.len();
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = deseasonalized.iter().sum::<f64>() / n as f64;

        let mut ss_xx = 0.0;
        let mut ss_xy = 0.0;
        for (i, &y) in deseasonalized.iter().enumerate() {
            let x = i as f64;
            ss_xx += (x - x_mean).powi(2);
            ss_xy += (x - x_mean) * (y - y_mean);
        }

        let slope = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        self.drift = Some(slope);

        // Apply theta transformation
        let theta_series = self.apply_theta(&deseasonalized);

        // Optimize alpha if needed
        if self.optimize {
            self.alpha = Some(Self::optimize_alpha(&theta_series));
        }

        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;

        // Fit SES on theta series
        let mut level = theta_series[0];
        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        fitted.push(if seasonals.is_empty() {
            theta_series[0]
        } else {
            theta_series[0] + seasonals[0]
        });
        residuals.push(0.0);

        for i in 1..self.n {
            let forecast = level;
            let seasonalized_forecast = if seasonals.is_empty() {
                forecast
            } else {
                forecast + seasonals[i % self.seasonal_period]
            };

            fitted.push(seasonalized_forecast);
            residuals.push(values[i] - seasonalized_forecast);

            level = alpha * theta_series[i] + (1.0 - alpha) * level;
        }

        self.level = Some(level);
        self.seasonals = if seasonals.is_empty() {
            None
        } else {
            Some(seasonals)
        };
        self.fitted = Some(fitted);

        // Calculate residual variance
        let valid_residuals: Vec<f64> = residuals[1..].to_vec();
        if !valid_residuals.is_empty() {
            let variance =
                valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let level = self.level.ok_or(ForecastError::FitRequired)?;
        let drift = self.drift.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Standard Theta method forecast:
        // forecast = SES_level + drift * (h + (n-1)/2)
        // This combines the level from SES on the theta line with the trend
        let mut predictions = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            // The drift contribution accounts for extrapolating the trend
            let trend_contrib = drift * ((self.n as f64 - 1.0) / 2.0 + h as f64);
            let forecast = level + trend_contrib;
            predictions.push(forecast);
        }

        // Reseasonalize if needed
        let predictions = if let Some(ref seasonals) = self.seasonals {
            self.reseasonalize(&predictions, self.n, seasonals)
        } else {
            predictions
        };

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + confidence) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        let alpha = self.alpha.unwrap_or(0.3);

        for h in 1..=horizon {
            // Variance increases with horizon
            let factor = if h == 1 {
                1.0
            } else {
                let beta = 1.0 - alpha;
                1.0 + beta.powi(2) * (1.0 - beta.powi(2 * (h as i32 - 1))) / (1.0 - beta.powi(2))
            };
            let se = (variance * factor).sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
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
        "Theta"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    #[test]
    fn theta_basic() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn theta_with_trend() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // Forecast should continue the trend
        assert!(preds[0] > values.last().unwrap() - 10.0);
    }

    #[test]
    fn theta_seasonal() {
        let timestamps = make_timestamps(48);
        let values: Vec<f64> = (0..48)
            .map(|i| 10.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn theta_with_alpha() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::with_alpha(0.5);
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.alpha().unwrap(), 0.5, epsilon = 1e-10);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn theta_custom_theta() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::with_theta(1.5);
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.theta(), 1.5, epsilon = 1e-10);
    }

    #[test]
    fn theta_confidence_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.2).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..5 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }
    }

    #[test]
    fn theta_fitted_and_residuals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());

        let fitted = model.fitted_values().unwrap();
        assert_eq!(fitted.len(), 30);
    }

    #[test]
    fn theta_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn theta_requires_fit() {
        let model = Theta::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn theta_zero_horizon() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn theta_name() {
        let model = Theta::new();
        assert_eq!(model.name(), "Theta");
    }

    #[test]
    fn theta_default() {
        let model = Theta::default();
        assert_relative_eq!(model.theta(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn theta_drift_positive_for_trend() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        // Drift should be positive for upward trend
        assert!(model.drift().unwrap() > 0.0);
    }
}
