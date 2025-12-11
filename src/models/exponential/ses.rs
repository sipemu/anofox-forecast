//! Simple Exponential Smoothing (SES) forecasting model.
//!
//! SES is suitable for forecasting data with no clear trend or seasonality.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Simple Exponential Smoothing forecaster.
///
/// The model equation is:
/// `level_t = α × y_t + (1-α) × level_{t-1}`
///
/// where α (alpha) is the smoothing parameter (0 < α < 1).
///
/// # Example
/// ```
/// use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc, Duration};
///
/// let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
/// let timestamps: Vec<_> = (0..10).map(|i| base + Duration::hours(i)).collect();
/// let values = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = SimpleExponentialSmoothing::new(0.3);
/// model.fit(&ts).unwrap();
///
/// let forecast = model.predict(3).unwrap();
/// assert_eq!(forecast.horizon(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleExponentialSmoothing {
    /// Smoothing parameter (0 < alpha < 1).
    alpha: Option<f64>,
    /// Whether to optimize alpha automatically.
    optimize: bool,
    /// Current level state.
    level: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance for prediction intervals.
    residual_variance: Option<f64>,
    /// Original series length.
    n: usize,
}

impl SimpleExponentialSmoothing {
    /// Create a new SES model with a fixed smoothing parameter.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter (0 < alpha < 1)
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            optimize: false,
            level: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a new SES model with automatic alpha optimization.
    ///
    /// Alpha will be chosen to minimize the sum of squared errors.
    pub fn auto() -> Self {
        Self {
            alpha: None,
            optimize: true,
            level: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Get the smoothing parameter.
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    /// Get the current level.
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Calculate SSE for a given alpha value.
    fn calculate_sse(values: &[f64], alpha: f64) -> f64 {
        if values.is_empty() {
            return f64::MAX;
        }

        let mut level = values[0];
        let mut sse = 0.0;

        for &y in &values[1..] {
            let error = y - level;
            sse += error * error;
            level = alpha * y + (1.0 - alpha) * level;
        }

        sse
    }

    /// Optimize alpha using Nelder-Mead.
    fn optimize_alpha(values: &[f64]) -> f64 {
        let config = NelderMeadConfig {
            max_iter: 500,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = nelder_mead(
            |params| Self::calculate_sse(values, params[0]),
            &[0.5],
            Some(&[(0.0001, 0.9999)]),
            config,
        );

        result.optimal_point[0].clamp(0.0001, 0.9999)
    }
}

impl Default for SimpleExponentialSmoothing {
    fn default() -> Self {
        Self::auto()
    }
}

impl Forecaster for SimpleExponentialSmoothing {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        self.n = values.len();

        // Optimize alpha if needed
        if self.optimize {
            self.alpha = Some(Self::optimize_alpha(values));
        }

        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;

        // Initialize level with first observation
        let mut level = values[0];
        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // First fitted value is the initial level
        fitted.push(level);
        residuals.push(0.0); // No residual for first observation

        // Compute fitted values and residuals
        for &y in &values[1..] {
            let forecast = level;
            fitted.push(forecast);
            residuals.push(y - forecast);
            level = alpha * y + (1.0 - alpha) * level;
        }

        self.level = Some(level);
        self.fitted = Some(fitted);

        // Calculate residual variance (excluding first observation)
        let valid_residuals: Vec<f64> = residuals[1..].to_vec();
        if !valid_residuals.is_empty() {
            let variance = valid_residuals.iter().map(|r| r * r).sum::<f64>()
                / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let level = self.level.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // SES produces flat forecasts at the final level
        let predictions = vec![level; horizon];
        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let current_level = self.level.ok_or(ForecastError::FitRequired)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let predictions = vec![current_level; horizon];
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            // Variance increases with forecast horizon
            // Var(e_{n+h}) = sigma^2 * (1 + sum_{j=1}^{h-1} (1-alpha)^{2j})
            // = sigma^2 * (1 + (1-alpha)^2 * (1 - (1-alpha)^{2(h-1)}) / (1 - (1-alpha)^2))
            let factor = if h == 1 {
                1.0
            } else {
                let beta = 1.0 - alpha;
                let beta2 = beta * beta;
                if (1.0 - beta2).abs() < 1e-10 {
                    h as f64
                } else {
                    1.0 + beta2 * (1.0 - beta2.powi((h - 1) as i32)) / (1.0 - beta2)
                }
            };
            let se = (variance * factor).sqrt();
            lower.push(current_level - z * se);
            upper.push(current_level + z * se);
        }

        Ok(Forecast::from_values_with_intervals(predictions, lower, upper))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "SimpleExponentialSmoothing"
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
    fn ses_with_fixed_alpha() {
        let timestamps = make_timestamps(10);
        let values = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.alpha().unwrap(), 0.3, epsilon = 1e-10);
        assert!(model.level().is_some());

        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.horizon(), 3);

        // All forecasts should be equal (flat)
        let preds = forecast.primary();
        assert_relative_eq!(preds[0], preds[1], epsilon = 1e-10);
        assert_relative_eq!(preds[1], preds[2], epsilon = 1e-10);
    }

    #[test]
    fn ses_auto_optimization() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + 0.5 * (i as f64).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();

        let alpha = model.alpha().unwrap();
        assert!(alpha > 0.0 && alpha < 1.0);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn ses_constant_series() {
        let timestamps = make_timestamps(10);
        let values = vec![5.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.5);
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        let preds = forecast.primary();

        // For constant series, forecast should equal the constant
        for pred in preds {
            assert_relative_eq!(*pred, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn ses_known_calculation() {
        // Manually verify SES calculation
        let timestamps = make_timestamps(4);
        let values = vec![10.0, 12.0, 14.0, 13.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let alpha = 0.5;
        let mut model = SimpleExponentialSmoothing::new(alpha);
        model.fit(&ts).unwrap();

        // Level calculation:
        // l_0 = 10
        // l_1 = 0.5*12 + 0.5*10 = 11
        // l_2 = 0.5*14 + 0.5*11 = 12.5
        // l_3 = 0.5*13 + 0.5*12.5 = 12.75
        assert_relative_eq!(model.level().unwrap(), 12.75, epsilon = 1e-10);

        // Fitted values should be the previous levels
        let fitted = model.fitted_values().unwrap();
        assert_relative_eq!(fitted[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(fitted[1], 10.0, epsilon = 1e-10);
        assert_relative_eq!(fitted[2], 11.0, epsilon = 1e-10);
        assert_relative_eq!(fitted[3], 12.5, epsilon = 1e-10);
    }

    #[test]
    fn ses_residuals_are_correct() {
        let timestamps = make_timestamps(5);
        let values = vec![10.0, 12.0, 11.0, 13.0, 14.0];
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        let residuals = model.residuals().unwrap();

        // Check residuals = actual - fitted
        for i in 1..5 {
            assert_relative_eq!(residuals[i], values[i] - fitted[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn ses_confidence_intervals() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        // Intervals should contain the point forecast
        for i in 0..5 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }

        // Intervals should widen with horizon
        let width_1 = upper[0] - lower[0];
        let width_5 = upper[4] - lower[4];
        assert!(width_5 >= width_1);
    }

    #[test]
    fn ses_alpha_clamped_to_valid_range() {
        // Alpha should be clamped to (0, 1)
        let model_low = SimpleExponentialSmoothing::new(-0.5);
        assert!(model_low.alpha().unwrap() > 0.0);

        let model_high = SimpleExponentialSmoothing::new(1.5);
        assert!(model_high.alpha().unwrap() < 1.0);
    }

    #[test]
    fn ses_empty_data_returns_error() {
        let timestamps: Vec<chrono::DateTime<Utc>> = vec![];
        let values: Vec<f64> = vec![];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        assert!(matches!(model.fit(&ts), Err(ForecastError::EmptyData)));
    }

    #[test]
    fn ses_requires_fit_before_predict() {
        let model = SimpleExponentialSmoothing::new(0.3);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn ses_zero_horizon_returns_empty() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn ses_name_is_correct() {
        let model = SimpleExponentialSmoothing::new(0.3);
        assert_eq!(model.name(), "SimpleExponentialSmoothing");
    }

    #[test]
    fn ses_default_is_auto() {
        let model = SimpleExponentialSmoothing::default();
        assert!(model.optimize);
        assert!(model.alpha.is_none());
    }

    #[test]
    fn ses_high_alpha_responds_quickly() {
        let timestamps = make_timestamps(10);
        // Step change from 10 to 20
        let values = vec![10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model_low = SimpleExponentialSmoothing::new(0.1);
        let mut model_high = SimpleExponentialSmoothing::new(0.9);

        model_low.fit(&ts).unwrap();
        model_high.fit(&ts).unwrap();

        // High alpha should be closer to 20
        assert!(model_high.level().unwrap() > model_low.level().unwrap());
    }
}
