//! Holt-Winters forecasting model.
//!
//! Also known as triple exponential smoothing, this model handles
//! data with both trend and seasonality.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Type of seasonal component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeasonalType {
    /// Additive seasonality: y_t = l_t + b_t + s_t + e_t
    #[default]
    Additive,
    /// Multiplicative seasonality: y_t = (l_t + b_t) * s_t + e_t
    Multiplicative,
}

/// Holt-Winters forecaster.
///
/// The model equations for additive seasonality:
/// - Level: `l_t = α(y_t - s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})`
/// - Trend: `b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}`
/// - Seasonal: `s_t = γ(y_t - l_t) + (1-γ)s_{t-m}`
/// - Forecast: `ŷ_{t+h} = l_t + h*b_t + s_{t+h-m}`
///
/// For multiplicative seasonality:
/// - Level: `l_t = α(y_t / s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})`
/// - Trend: `b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}`
/// - Seasonal: `s_t = γ(y_t / l_t) + (1-γ)s_{t-m}`
/// - Forecast: `ŷ_{t+h} = (l_t + h*b_t) * s_{t+h-m}`
#[derive(Debug, Clone)]
pub struct HoltWinters {
    /// Level smoothing parameter (0 < alpha < 1).
    alpha: Option<f64>,
    /// Trend smoothing parameter (0 < beta < 1).
    beta: Option<f64>,
    /// Seasonal smoothing parameter (0 < gamma < 1).
    gamma: Option<f64>,
    /// Seasonal period.
    seasonal_period: usize,
    /// Type of seasonality.
    seasonal_type: SeasonalType,
    /// Whether to optimize parameters.
    optimize: bool,
    /// Current level state.
    level: Option<f64>,
    /// Current trend state.
    trend: Option<f64>,
    /// Seasonal indices.
    seasonals: Option<Vec<f64>>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Original series length.
    n: usize,
}

impl HoltWinters {
    /// Create a new Holt-Winters model with fixed parameters.
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        seasonal_period: usize,
        seasonal_type: SeasonalType,
    ) -> Self {
        Self {
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            beta: Some(beta.clamp(0.0001, 0.9999)),
            gamma: Some(gamma.clamp(0.0001, 0.9999)),
            seasonal_period,
            seasonal_type,
            optimize: false,
            level: None,
            trend: None,
            seasonals: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a new Holt-Winters model with additive seasonality.
    pub fn additive(alpha: f64, beta: f64, gamma: f64, seasonal_period: usize) -> Self {
        Self::new(alpha, beta, gamma, seasonal_period, SeasonalType::Additive)
    }

    /// Create a new Holt-Winters model with multiplicative seasonality.
    pub fn multiplicative(alpha: f64, beta: f64, gamma: f64, seasonal_period: usize) -> Self {
        Self::new(
            alpha,
            beta,
            gamma,
            seasonal_period,
            SeasonalType::Multiplicative,
        )
    }

    /// Create a new Holt-Winters model with automatic parameter optimization.
    pub fn auto(seasonal_period: usize, seasonal_type: SeasonalType) -> Self {
        Self {
            alpha: None,
            beta: None,
            gamma: None,
            seasonal_period,
            seasonal_type,
            optimize: true,
            level: None,
            trend: None,
            seasonals: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Get the smoothing parameters.
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    pub fn beta(&self) -> Option<f64> {
        self.beta
    }

    pub fn gamma(&self) -> Option<f64> {
        self.gamma
    }

    /// Get the seasonal period.
    pub fn seasonal_period(&self) -> usize {
        self.seasonal_period
    }

    /// Get the seasonal type.
    pub fn seasonal_type(&self) -> SeasonalType {
        self.seasonal_type
    }

    /// Get the current level.
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Get the current trend.
    pub fn trend(&self) -> Option<f64> {
        self.trend
    }

    /// Get the seasonal indices.
    pub fn seasonals(&self) -> Option<&[f64]> {
        self.seasonals.as_deref()
    }

    /// Initialize state from first complete season(s).
    fn initialize_state(
        values: &[f64],
        period: usize,
        seasonal_type: SeasonalType,
    ) -> (f64, f64, Vec<f64>) {
        // Initial level: average of first season
        let first_season: Vec<f64> = values.iter().take(period).copied().collect();
        let level = first_season.iter().sum::<f64>() / period as f64;

        // Initial trend: average of seasonal differences if we have 2+ seasons
        let trend = if values.len() >= 2 * period {
            let sum: f64 = (0..period)
                .map(|i| (values[period + i] - values[i]) / period as f64)
                .sum();
            sum / period as f64
        } else {
            0.0
        };

        // Initial seasonal indices
        let mut seasonals: Vec<f64> = match seasonal_type {
            SeasonalType::Additive => first_season.iter().map(|y| y - level).collect(),
            SeasonalType::Multiplicative => first_season
                .iter()
                .map(|y| if level.abs() > 1e-10 { y / level } else { 1.0 })
                .collect(),
        };

        // Normalize seasonal components
        Self::normalize_seasonals(&mut seasonals, seasonal_type);

        (level, trend, seasonals)
    }

    /// Normalize seasonal components to maintain constraints.
    /// Additive: seasonals sum to 0
    /// Multiplicative: seasonals average to 1
    fn normalize_seasonals(seasonals: &mut [f64], seasonal_type: SeasonalType) {
        let period = seasonals.len();
        if period == 0 {
            return;
        }

        match seasonal_type {
            SeasonalType::Additive => {
                // Ensure seasonals sum to 0
                let sum: f64 = seasonals.iter().sum();
                let adjustment = sum / period as f64;
                for s in seasonals.iter_mut() {
                    *s -= adjustment;
                }
            }
            SeasonalType::Multiplicative => {
                // Ensure seasonals average to 1
                let mean: f64 = seasonals.iter().sum::<f64>() / period as f64;
                if mean.abs() > 1e-10 {
                    for s in seasonals.iter_mut() {
                        *s /= mean;
                    }
                }
            }
        }
    }

    /// Calculate SSE for given parameters.
    fn calculate_sse(
        values: &[f64],
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        seasonal_type: SeasonalType,
    ) -> f64 {
        if values.len() < period {
            return f64::MAX;
        }

        let (mut level, mut trend, mut seasonals) =
            Self::initialize_state(values, period, seasonal_type);

        let mut sse = 0.0;

        for (t, &y) in values.iter().enumerate().skip(period) {
            let season_idx = t % period;
            let s = seasonals[season_idx];

            let forecast = match seasonal_type {
                SeasonalType::Additive => level + trend + s,
                SeasonalType::Multiplicative => (level + trend) * s,
            };

            let error = y - forecast;
            sse += error * error;

            let level_prev = level;

            // Update equations
            match seasonal_type {
                SeasonalType::Additive => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                SeasonalType::Multiplicative => {
                    let y_deseasonalized = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_deseasonalized + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    let s_new = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                    seasonals[season_idx] = s_new;
                }
            }
        }

        sse
    }

    /// Optimize parameters using Nelder-Mead.
    fn optimize_params(
        values: &[f64],
        period: usize,
        seasonal_type: SeasonalType,
    ) -> (f64, f64, f64) {
        let config = NelderMeadConfig {
            max_iter: 1000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = nelder_mead(
            |params| {
                Self::calculate_sse(
                    values,
                    params[0],
                    params[1],
                    params[2],
                    period,
                    seasonal_type,
                )
            },
            &[0.3, 0.1, 0.1],
            Some(&[(0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999)]),
            config,
        );

        (
            result.optimal_point[0].clamp(0.0001, 0.9999),
            result.optimal_point[1].clamp(0.0001, 0.9999),
            result.optimal_point[2].clamp(0.0001, 0.9999),
        )
    }
}

impl Default for HoltWinters {
    fn default() -> Self {
        Self::auto(12, SeasonalType::Additive)
    }
}

impl Forecaster for HoltWinters {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < 2 * self.seasonal_period {
            return Err(ForecastError::InsufficientData {
                needed: 2 * self.seasonal_period,
                got: values.len(),
            });
        }

        self.n = values.len();

        // Optimize if needed
        if self.optimize {
            let (alpha, beta, gamma) =
                Self::optimize_params(values, self.seasonal_period, self.seasonal_type);
            self.alpha = Some(alpha);
            self.beta = Some(beta);
            self.gamma = Some(gamma);
        }

        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;
        let beta = self.beta.ok_or(ForecastError::FitRequired)?;
        let gamma = self.gamma.ok_or(ForecastError::FitRequired)?;
        let period = self.seasonal_period;

        // Initialize state
        let (mut level, mut trend, mut seasonals) =
            Self::initialize_state(values, period, self.seasonal_type);

        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // First season has no fitted values (used for initialization)
        for &val in values.iter().take(period) {
            fitted.push(val); // Use actual as "fitted"
            residuals.push(0.0);
        }

        // Process remaining data
        for (t, &y) in values.iter().enumerate().skip(period) {
            let season_idx = t % period;
            let s = seasonals[season_idx];

            let forecast = match self.seasonal_type {
                SeasonalType::Additive => level + trend + s,
                SeasonalType::Multiplicative => (level + trend) * s,
            };

            fitted.push(forecast);
            residuals.push(y - forecast);

            let level_prev = level;

            // Update equations
            match self.seasonal_type {
                SeasonalType::Additive => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                SeasonalType::Multiplicative => {
                    let y_deseasonalized = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_deseasonalized + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    let s_new = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                    seasonals[season_idx] = s_new;
                }
            }
        }

        self.level = Some(level);
        self.trend = Some(trend);
        self.seasonals = Some(seasonals);
        self.fitted = Some(fitted);

        // Calculate residual variance
        let valid_residuals: Vec<f64> = residuals[period..].to_vec();
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
        let trend = self.trend.ok_or(ForecastError::FitRequired)?;
        let seasonals = self.seasonals.as_ref().ok_or(ForecastError::FitRequired)?;
        let period = self.seasonal_period;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let predictions: Vec<f64> = (1..=horizon)
            .map(|h| {
                // Get the appropriate seasonal index
                let season_idx = (self.n + h - 1) % period;
                let s = seasonals[season_idx];

                match self.seasonal_type {
                    SeasonalType::Additive => level + (h as f64) * trend + s,
                    SeasonalType::Multiplicative => (level + (h as f64) * trend) * s,
                }
            })
            .collect();

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let l = self.level.ok_or(ForecastError::FitRequired)?;
        let b = self.trend.ok_or(ForecastError::FitRequired)?;
        let seasonals = self.seasonals.as_ref().ok_or(ForecastError::FitRequired)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let period = self.seasonal_period;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let season_idx = (self.n + h - 1) % period;
            let s = seasonals[season_idx];

            let pred = match self.seasonal_type {
                SeasonalType::Additive => l + (h as f64) * b + s,
                SeasonalType::Multiplicative => (l + (h as f64) * b) * s,
            };
            predictions.push(pred);

            // Simplified standard error approximation
            let k = ((h - 1) / period) + 1;
            let se = (variance * k as f64).sqrt();

            lower.push(pred - z * se);
            upper.push(pred + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            predictions,
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
        match self.seasonal_type {
            SeasonalType::Additive => "HoltWinters(additive)",
            SeasonalType::Multiplicative => "HoltWinters(multiplicative)",
        }
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

    fn make_seasonal_data(n: usize, period: usize, trend: f64, amplitude: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                let seasonal = amplitude * (2.0 * std::f64::consts::PI * t / period as f64).sin();
                10.0 + trend * t + seasonal
            })
            .collect()
    }

    #[test]
    fn hw_additive_basic() {
        let timestamps = make_timestamps(32);
        let values = make_seasonal_data(32, 8, 0.1, 5.0);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 8);
        model.fit(&ts).unwrap();

        let forecast = model.predict(8).unwrap();
        assert_eq!(forecast.horizon(), 8);

        // Forecasts should show seasonal pattern
        let preds = forecast.primary();
        assert!(preds.len() == 8);
    }

    #[test]
    fn hw_multiplicative_basic() {
        let timestamps = make_timestamps(32);
        // Multiplicative data: base level with multiplied seasonal effect
        let values: Vec<f64> = (0..32)
            .map(|i| {
                let base = 100.0 + 0.5 * i as f64;
                let seasonal = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin();
                base * seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::multiplicative(0.3, 0.1, 0.1, 8);
        model.fit(&ts).unwrap();

        let forecast = model.predict(8).unwrap();
        assert_eq!(forecast.horizon(), 8);
    }

    #[test]
    fn hw_auto_optimization() {
        let timestamps = make_timestamps(48);
        let values = make_seasonal_data(48, 12, 0.1, 3.0);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::auto(12, SeasonalType::Additive);
        model.fit(&ts).unwrap();

        assert!(model.alpha().unwrap() > 0.0);
        assert!(model.beta().unwrap() > 0.0);
        assert!(model.gamma().unwrap() > 0.0);

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn hw_captures_seasonality() {
        let timestamps = make_timestamps(32);
        // Strong seasonal pattern: [high, low, high, low, ...]
        let values: Vec<f64> = (0..32)
            .map(|i| if i % 4 < 2 { 20.0 } else { 10.0 })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.5, 0.1, 0.5, 4);
        model.fit(&ts).unwrap();

        let forecast = model.predict(4).unwrap();
        let preds = forecast.primary();

        // Should capture the high-low pattern
        // First two should be higher, next two should be lower
        assert!(preds[0] > preds[2] || preds[1] > preds[3]);
    }

    #[test]
    fn hw_confidence_intervals() {
        let timestamps = make_timestamps(32);
        let values = make_seasonal_data(32, 8, 0.1, 3.0);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 8);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(8, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..8 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }
    }

    #[test]
    fn hw_insufficient_data() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 8);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData {
                needed: 16,
                got: 10
            })
        ));
    }

    #[test]
    fn hw_requires_fit_before_predict() {
        let model = HoltWinters::additive(0.3, 0.1, 0.1, 4);
        assert!(matches!(model.predict(4), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn hw_zero_horizon() {
        let timestamps = make_timestamps(16);
        let values: Vec<f64> = (0..16).map(|i| 10.0 + (i % 4) as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 4);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn hw_name_reflects_type() {
        let additive = HoltWinters::additive(0.3, 0.1, 0.1, 4);
        assert_eq!(additive.name(), "HoltWinters(additive)");

        let multiplicative = HoltWinters::multiplicative(0.3, 0.1, 0.1, 4);
        assert_eq!(multiplicative.name(), "HoltWinters(multiplicative)");
    }

    #[test]
    fn hw_fitted_and_residuals() {
        let timestamps = make_timestamps(24);
        let values = make_seasonal_data(24, 6, 0.1, 2.0);
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 6);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        let residuals = model.residuals().unwrap();

        assert_eq!(fitted.len(), 24);
        assert_eq!(residuals.len(), 24);

        // Check residuals = actual - fitted for data after initialization
        for i in 6..24 {
            assert_relative_eq!(residuals[i], values[i] - fitted[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn hw_seasonals_have_correct_length() {
        let timestamps = make_timestamps(24);
        let values = make_seasonal_data(24, 6, 0.1, 2.0);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.3, 0.1, 0.1, 6);
        model.fit(&ts).unwrap();

        let seasonals = model.seasonals().unwrap();
        assert_eq!(seasonals.len(), 6);
    }

    #[test]
    fn hw_getters_work() {
        let model = HoltWinters::new(0.3, 0.2, 0.1, 12, SeasonalType::Multiplicative);

        assert_relative_eq!(model.alpha().unwrap(), 0.3, epsilon = 1e-10);
        assert_relative_eq!(model.beta().unwrap(), 0.2, epsilon = 1e-10);
        assert_relative_eq!(model.gamma().unwrap(), 0.1, epsilon = 1e-10);
        assert_eq!(model.seasonal_period(), 12);
        assert_eq!(model.seasonal_type(), SeasonalType::Multiplicative);
    }

    #[test]
    fn hw_default() {
        let model = HoltWinters::default();
        assert_eq!(model.seasonal_period(), 12);
        assert_eq!(model.seasonal_type(), SeasonalType::Additive);
        assert!(model.optimize);
    }

    #[test]
    fn hw_multi_season_forecast() {
        let timestamps = make_timestamps(24);
        let values = make_seasonal_data(24, 4, 0.0, 3.0); // No trend
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltWinters::additive(0.5, 0.1, 0.5, 4);
        model.fit(&ts).unwrap();

        // Forecast 3 full seasons
        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Each season should have similar pattern
        // (allowing for some model adaptation)
        for i in 0..4 {
            let s1 = preds[i];
            let s2 = preds[i + 4];
            let s3 = preds[i + 8];
            // All three seasonal periods should be similar (within 20% for no-trend data)
            assert!((s1 - s2).abs() / s1.abs().max(1.0) < 0.2);
            assert!((s2 - s3).abs() / s2.abs().max(1.0) < 0.2);
        }
    }
}
