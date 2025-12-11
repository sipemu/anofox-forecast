//! Holt's Linear Trend forecasting model.
//!
//! Also known as double exponential smoothing, this model is suitable for
//! data with a linear trend but no seasonality.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Holt's Linear Trend forecaster.
///
/// The model equations are:
/// - Level: `l_t = α × y_t + (1-α) × (l_{t-1} + b_{t-1})`
/// - Trend: `b_t = β × (l_t - l_{t-1}) + (1-β) × b_{t-1}`
/// - Forecast: `ŷ_{t+h} = l_t + h × b_t`
///
/// Optional damping with phi parameter:
/// - Trend: `b_t = β × (l_t - l_{t-1}) + (1-β) × φ × b_{t-1}`
/// - Forecast: `ŷ_{t+h} = l_t + (φ + φ² + ... + φ^h) × b_t`
#[derive(Debug, Clone)]
pub struct HoltLinearTrend {
    /// Level smoothing parameter (0 < alpha < 1).
    alpha: Option<f64>,
    /// Trend smoothing parameter (0 < beta < 1).
    beta: Option<f64>,
    /// Damping parameter (0 < phi <= 1). None means no damping.
    phi: Option<f64>,
    /// Whether to optimize parameters automatically.
    optimize: bool,
    /// Current level state.
    level: Option<f64>,
    /// Current trend state.
    trend: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Original series length.
    n: usize,
}

impl HoltLinearTrend {
    /// Create a new Holt model with fixed parameters.
    ///
    /// # Arguments
    /// * `alpha` - Level smoothing parameter (0 < alpha < 1)
    /// * `beta` - Trend smoothing parameter (0 < beta < 1)
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            beta: Some(beta.clamp(0.0001, 0.9999)),
            phi: None,
            optimize: false,
            level: None,
            trend: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a new Holt model with damping.
    ///
    /// # Arguments
    /// * `alpha` - Level smoothing parameter (0 < alpha < 1)
    /// * `beta` - Trend smoothing parameter (0 < beta < 1)
    /// * `phi` - Damping parameter (0 < phi <= 1)
    pub fn damped(alpha: f64, beta: f64, phi: f64) -> Self {
        Self {
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            beta: Some(beta.clamp(0.0001, 0.9999)),
            phi: Some(phi.clamp(0.8, 1.0)),
            optimize: false,
            level: None,
            trend: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a new Holt model with automatic parameter optimization.
    pub fn auto() -> Self {
        Self {
            alpha: None,
            beta: None,
            phi: None,
            optimize: true,
            level: None,
            trend: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a new Holt model with automatic optimization including damping.
    pub fn auto_damped() -> Self {
        Self {
            alpha: None,
            beta: None,
            phi: Some(0.98), // Will be optimized
            optimize: true,
            level: None,
            trend: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Get the level smoothing parameter.
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    /// Get the trend smoothing parameter.
    pub fn beta(&self) -> Option<f64> {
        self.beta
    }

    /// Get the damping parameter.
    pub fn phi(&self) -> Option<f64> {
        self.phi
    }

    /// Get the current level.
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Get the current trend.
    pub fn trend(&self) -> Option<f64> {
        self.trend
    }

    /// Calculate SSE for given parameters.
    fn calculate_sse(values: &[f64], alpha: f64, beta: f64, phi: Option<f64>) -> f64 {
        if values.len() < 2 {
            return f64::MAX;
        }

        let phi = phi.unwrap_or(1.0);

        // Initialize with simple linear regression on first few points
        let (level, trend) = Self::initialize_state(values);

        let mut l = level;
        let mut b = trend;
        let mut sse = 0.0;

        for &y in values.iter().skip(1) {
            let forecast = l + phi * b;
            let error = y - forecast;
            sse += error * error;

            let l_prev = l;
            l = alpha * y + (1.0 - alpha) * (l_prev + phi * b);
            b = beta * (l - l_prev) + (1.0 - beta) * phi * b;
        }

        sse
    }

    /// Initialize level and trend using first observations.
    fn initialize_state(values: &[f64]) -> (f64, f64) {
        if values.len() < 2 {
            return (values.get(0).copied().unwrap_or(0.0), 0.0);
        }

        // Use first value as initial level
        let level = values[0];
        // Use first difference as initial trend
        let trend = values[1] - values[0];

        (level, trend)
    }

    /// Optimize parameters using Nelder-Mead.
    fn optimize_params(values: &[f64], with_damping: bool) -> (f64, f64, Option<f64>) {
        let config = NelderMeadConfig {
            max_iter: 1000,
            tolerance: 1e-8,
            ..Default::default()
        };

        if with_damping {
            let result = nelder_mead(
                |params| Self::calculate_sse(values, params[0], params[1], Some(params[2])),
                &[0.3, 0.1, 0.98],
                Some(&[(0.0001, 0.9999), (0.0001, 0.9999), (0.8, 1.0)]),
                config,
            );
            (
                result.optimal_point[0].clamp(0.0001, 0.9999),
                result.optimal_point[1].clamp(0.0001, 0.9999),
                Some(result.optimal_point[2].clamp(0.8, 1.0)),
            )
        } else {
            let result = nelder_mead(
                |params| Self::calculate_sse(values, params[0], params[1], None),
                &[0.3, 0.1],
                Some(&[(0.0001, 0.9999), (0.0001, 0.9999)]),
                config,
            );
            (
                result.optimal_point[0].clamp(0.0001, 0.9999),
                result.optimal_point[1].clamp(0.0001, 0.9999),
                None,
            )
        }
    }

    /// Calculate damped sum: phi + phi^2 + ... + phi^h.
    fn damped_sum(phi: f64, h: usize) -> f64 {
        if (phi - 1.0).abs() < 1e-10 {
            h as f64
        } else {
            phi * (1.0 - phi.powi(h as i32)) / (1.0 - phi)
        }
    }
}

impl Default for HoltLinearTrend {
    fn default() -> Self {
        Self::auto()
    }
}

impl Forecaster for HoltLinearTrend {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: values.len(),
            });
        }

        self.n = values.len();

        // Optimize if needed
        if self.optimize {
            let with_damping = self.phi.is_some();
            let (alpha, beta, phi) = Self::optimize_params(values, with_damping);
            self.alpha = Some(alpha);
            self.beta = Some(beta);
            if with_damping {
                self.phi = phi;
            }
        }

        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;
        let beta = self.beta.ok_or(ForecastError::FitRequired)?;
        let phi = self.phi.unwrap_or(1.0);

        // Initialize state
        let (mut l, mut b) = Self::initialize_state(values);

        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // First fitted value
        fitted.push(l);
        residuals.push(0.0);

        // Update state and compute fitted values
        for &y in values.iter().skip(1) {
            let forecast = l + phi * b;
            fitted.push(forecast);
            residuals.push(y - forecast);

            let l_prev = l;
            l = alpha * y + (1.0 - alpha) * (l_prev + phi * b);
            b = beta * (l - l_prev) + (1.0 - beta) * phi * b;
        }

        self.level = Some(l);
        self.trend = Some(b);
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
        let l = self.level.ok_or(ForecastError::FitRequired)?;
        let b = self.trend.ok_or(ForecastError::FitRequired)?;
        let phi = self.phi.unwrap_or(1.0);

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let predictions: Vec<f64> = (1..=horizon)
            .map(|h| l + Self::damped_sum(phi, h) * b)
            .collect();

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let l = self.level.ok_or(ForecastError::FitRequired)?;
        let b = self.trend.ok_or(ForecastError::FitRequired)?;
        let phi = self.phi.unwrap_or(1.0);
        let variance = self.residual_variance.unwrap_or(0.0);
        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;
        let beta = self.beta.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let pred = l + Self::damped_sum(phi, h) * b;
            predictions.push(pred);

            // Approximate standard error for Holt's method
            // This is a simplified approximation
            let c = if h == 1 {
                1.0
            } else {
                let mut sum = 1.0;
                for j in 1..h {
                    let phi_sum = Self::damped_sum(phi, j);
                    sum += (alpha + alpha * beta * phi_sum).powi(2);
                }
                sum
            };
            let se = (variance * c).sqrt();

            lower.push(pred - z * se);
            upper.push(pred + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            predictions, lower, upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        if self.phi.is_some() {
            "HoltLinearTrend(damped)"
        } else {
            "HoltLinearTrend"
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

    #[test]
    fn holt_with_fixed_params() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::new(0.3, 0.1);
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.alpha().unwrap(), 0.3, epsilon = 1e-10);
        assert_relative_eq!(model.beta().unwrap(), 0.1, epsilon = 1e-10);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);

        // With linear trend, forecasts should increase
        let preds = forecast.primary();
        assert!(preds[1] > preds[0]);
        assert!(preds[2] > preds[1]);
    }

    #[test]
    fn holt_auto_optimization() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + 1.5 * i as f64 + (i as f64 * 0.5).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::auto();
        model.fit(&ts).unwrap();

        assert!(model.alpha().unwrap() > 0.0);
        assert!(model.beta().unwrap() > 0.0);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn holt_damped_trend() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model_undamped = HoltLinearTrend::new(0.3, 0.1);
        let mut model_damped = HoltLinearTrend::damped(0.3, 0.1, 0.9);

        model_undamped.fit(&ts).unwrap();
        model_damped.fit(&ts).unwrap();

        let forecast_undamped = model_undamped.predict(10).unwrap();
        let forecast_damped = model_damped.predict(10).unwrap();

        // Damped trend should be more conservative at longer horizons
        let undamped_10 = forecast_undamped.primary()[9];
        let damped_10 = forecast_damped.primary()[9];

        // For a positive trend, undamped should be higher at long horizons
        assert!(undamped_10 > damped_10);
    }

    #[test]
    fn holt_linear_trend_exact() {
        // For a perfect linear trend, Holt should eventually converge
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 5.0 + 3.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::new(0.9, 0.9);
        model.fit(&ts).unwrap();

        // Trend should be close to 3
        assert!((model.trend().unwrap() - 3.0).abs() < 1.0);
    }

    #[test]
    fn holt_constant_series() {
        let timestamps = make_timestamps(10);
        let values = vec![10.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::new(0.3, 0.1);
        model.fit(&ts).unwrap();

        // Trend should be close to 0
        assert!(model.trend().unwrap().abs() < 1.0);

        let forecast = model.predict(3).unwrap();
        // Forecasts should be close to 10
        for pred in forecast.primary() {
            assert!((pred - 10.0).abs() < 2.0);
        }
    }

    #[test]
    fn holt_confidence_intervals() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::new(0.3, 0.1);
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
    fn holt_insufficient_data() {
        let timestamps = make_timestamps(1);
        let values = vec![10.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::new(0.3, 0.1);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { needed: 2, got: 1 })
        ));
    }

    #[test]
    fn holt_requires_fit_before_predict() {
        let model = HoltLinearTrend::new(0.3, 0.1);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn holt_zero_horizon() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::new(0.3, 0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn holt_name_reflects_damping() {
        let model_undamped = HoltLinearTrend::new(0.3, 0.1);
        assert_eq!(model_undamped.name(), "HoltLinearTrend");

        let model_damped = HoltLinearTrend::damped(0.3, 0.1, 0.9);
        assert_eq!(model_damped.name(), "HoltLinearTrend(damped)");
    }

    #[test]
    fn holt_fitted_and_residuals() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| 5.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = HoltLinearTrend::new(0.3, 0.1);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        let residuals = model.residuals().unwrap();

        assert_eq!(fitted.len(), 10);
        assert_eq!(residuals.len(), 10);

        // Check residuals = actual - fitted
        for i in 1..10 {
            assert_relative_eq!(residuals[i], values[i] - fitted[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn holt_default_is_auto() {
        let model = HoltLinearTrend::default();
        assert!(model.optimize);
        assert!(model.alpha.is_none());
        assert!(model.beta.is_none());
    }

    #[test]
    fn holt_auto_damped() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HoltLinearTrend::auto_damped();
        model.fit(&ts).unwrap();

        assert!(model.phi().is_some());
        let phi = model.phi().unwrap();
        assert!(phi >= 0.8 && phi <= 1.0);
    }
}
