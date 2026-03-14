//! Simple Moving Average and related forecasting models.
//!
//! This module provides:
//! - `SimpleMovingAverage`: Forecasts using the mean of the last `window` observations
//! - `HistoricAverage`: Forecasts using the mean of ALL historical values
//! - `WindowAverage`: Forecasts using the mean of the last N observations

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::{validate_series_complete, Forecaster};

/// Simple Moving Average forecaster.
///
/// Predicts future values as the mean of the last `window` observations.
/// If window is 0, uses the mean of all historical data.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimpleMovingAverage {
    window: usize, // 0 means use all data
    /// If set, constrains the effective window to at most `n - changepoint`
    /// so that only post-changepoint observations are averaged.
    changepoint: Option<usize>,
    last_mean: Option<f64>,
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
    fitted: Option<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
    residuals: Option<Vec<f64>>,
    residual_variance: Option<f64>,
}

/// Builder for SimpleMovingAverage.
#[derive(Debug, Clone, Default)]
pub struct SmaBuilder {
    window: usize,
    changepoint: Option<usize>,
}

impl SmaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the window size. Use 0 for full history mean.
    pub fn window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Set the changepoint index. The effective window will be constrained
    /// to `min(window, n - changepoint)`.
    pub fn changepoint(mut self, cp: usize) -> Self {
        self.changepoint = Some(cp);
        self
    }

    pub fn build(self) -> Result<SimpleMovingAverage> {
        Ok(SimpleMovingAverage {
            window: self.window,
            changepoint: self.changepoint,
            last_mean: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
        })
    }
}

impl SimpleMovingAverage {
    /// Create a new SMA with the given window size.
    /// Window of 0 means use the entire history.
    pub fn new(window: usize) -> Self {
        Self {
            window,
            changepoint: None,
            last_mean: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
        }
    }

    /// Set the last changepoint index (builder-style).
    ///
    /// When set, the effective averaging window is constrained to
    /// `min(window, n - changepoint)` so that only post-changepoint
    /// observations contribute to the forecast. This avoids averaging
    /// across regime changes without requiring data truncation.
    ///
    /// # Example
    ///
    /// ```
    /// use anofox_forecast::models::baseline::SimpleMovingAverage;
    ///
    /// // SMA(50) but data has a changepoint at index 180 in 200 observations.
    /// // Effective window becomes min(50, 200-180) = 20.
    /// let model = SimpleMovingAverage::new(50).with_changepoint(180);
    /// ```
    pub fn with_changepoint(mut self, changepoint: usize) -> Self {
        self.changepoint = Some(changepoint);
        self
    }

    /// Get the changepoint index, if set.
    pub fn changepoint(&self) -> Option<usize> {
        self.changepoint
    }

    /// Create a builder for more complex configuration.
    pub fn builder() -> SmaBuilder {
        SmaBuilder::new()
    }

    /// Get the window size.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Calculate the moving average at a position.
    fn calculate_ma(&self, values: &[f64], end: usize) -> f64 {
        let actual_window = if self.window == 0 || self.window > end {
            end
        } else {
            self.window
        };

        if actual_window == 0 {
            return f64::NAN;
        }

        let start = end - actual_window;
        values[start..end].iter().sum::<f64>() / actual_window as f64
    }
}

impl Default for SimpleMovingAverage {
    fn default() -> Self {
        Self::new(0) // Full history mean (no changepoint constraint)
    }
}

impl Forecaster for SimpleMovingAverage {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let values = series.primary_values();

        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        // Check if we have enough data for the window
        if self.window > 0 && values.len() < self.window {
            return Err(ForecastError::InsufficientData {
                needed: self.window,
                got: values.len(),
                hint: None,
            });
        }

        // Check for multivariate
        if series.is_multivariate() {
            return Err(ForecastError::InvalidParameter(
                "SMA only supports univariate series".to_string(),
            ));
        }

        let n = values.len();

        // Calculate the forecast value (mean of last window)
        let mut actual_window = if self.window == 0 { n } else { self.window };

        // Constrain window to post-changepoint data if changepoint is set
        if let Some(cp) = self.changepoint {
            if cp < n {
                let post_cp = n - cp;
                actual_window = actual_window.min(post_cp);
            }
        }

        self.last_mean =
            Some(values[n - actual_window..].iter().sum::<f64>() / actual_window as f64);

        // Calculate fitted values (rolling mean)
        let mut fitted = Vec::with_capacity(n);
        for i in 0..n {
            if i < 1 {
                fitted.push(f64::NAN);
            } else {
                fitted.push(self.calculate_ma(values, i));
            }
        }
        self.fitted = Some(fitted.clone());

        // Calculate residuals
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                if fitted[i].is_nan() {
                    f64::NAN
                } else {
                    values[i] - fitted[i]
                }
            })
            .collect();

        // Residual variance
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();
        if !valid_residuals.is_empty() {
            let variance =
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let mean = self
            .last_mean
            .ok_or(ForecastError::FitRequired { model: None })?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // SMA predicts the same value for all horizons
        let predictions = vec![mean; horizon];
        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let mean = self
            .last_mean
            .ok_or(ForecastError::FitRequired { model: None })?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let sigma = variance.sqrt();

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);

        // Standard error accounts for estimation uncertainty: se = sigma * sqrt(1 + 1/w)
        let w = if self.window == 0 {
            self.fitted.as_ref().map_or(1, |f| f.len()) as f64
        } else {
            self.window as f64
        };
        let se = sigma * (1.0 + 1.0 / w).sqrt();

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for _ in 0..horizon {
            predictions.push(mean);
            lower.push(mean - z * se);
            upper.push(mean + z * se);
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

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted.as_ref()?;
        let variance = self.residual_variance?;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let sigma = variance.sqrt();

        // Standard error accounts for estimation uncertainty: se = sigma * sqrt(1 + 1/w)
        let w = if self.window == 0 {
            fitted.len() as f64
        } else {
            self.window as f64
        };
        let se = sigma * (1.0 + 1.0 / w).sqrt();

        let lower: Vec<f64> = fitted.iter().map(|&f| f - z * se).collect();
        let upper: Vec<f64> = fitted.iter().map(|&f| f + z * se).collect();

        Some(Forecast::from_values_with_intervals(
            fitted.clone(),
            lower,
            upper,
        ))
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "SimpleMovingAverage"
    }
}

fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

// ============================================================================
// HistoricAverage - Wrapper for full-history mean
// ============================================================================

/// HistoricAverage forecaster.
///
/// Predicts future values as the mean of ALL historical observations.
/// This is equivalent to `SimpleMovingAverage` with `window = 0`.
///
/// Matches statsforecast's `HistoricAverage` model.
///
/// # Example
/// ```
/// use anofox_forecast::models::baseline::HistoricAverage;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..10).map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i, 0, 0).unwrap()).collect();
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = HistoricAverage::new();
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(3).unwrap();
/// // All predictions will be 5.5 (mean of 1..10)
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoricAverage {
    inner: SimpleMovingAverage,
}

impl HistoricAverage {
    /// Create a new HistoricAverage forecaster.
    pub fn new() -> Self {
        Self {
            inner: SimpleMovingAverage::new(0),
        }
    }
}

impl Default for HistoricAverage {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for HistoricAverage {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        self.inner.fit(series)
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        self.inner.predict(horizon)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        self.inner.predict_with_intervals(horizon, level)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.inner.fitted_values()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        self.inner.fitted_values_with_intervals(level)
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.inner.residuals()
    }

    fn name(&self) -> &str {
        "HistoricAverage"
    }
}

// ============================================================================
// WindowAverage - Wrapper for fixed-window mean
// ============================================================================

/// WindowAverage forecaster.
///
/// Predicts future values as the mean of the last `window_size` observations.
/// This is equivalent to `SimpleMovingAverage` with a specified window.
///
/// Matches statsforecast's `WindowAverage` model.
///
/// # Example
/// ```
/// use anofox_forecast::models::baseline::WindowAverage;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..10).map(|i| Utc.with_ymd_and_hms(2024, 1, 1, i, 0, 0).unwrap()).collect();
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = WindowAverage::new(3);  // Use last 3 values
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(3).unwrap();
/// // All predictions will be 9.0 (mean of 8, 9, 10)
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowAverage {
    inner: SimpleMovingAverage,
}

impl WindowAverage {
    /// Create a new WindowAverage forecaster with the specified window size.
    ///
    /// The window size must be at least 1. If 0 is passed, it will be set to 1.
    pub fn new(window_size: usize) -> Self {
        Self {
            inner: SimpleMovingAverage::new(window_size.max(1)),
        }
    }

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.inner.window()
    }
}

impl Forecaster for WindowAverage {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        self.inner.fit(series)
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        self.inner.predict(horizon)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        self.inner.predict_with_intervals(horizon, level)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.inner.fitted_values()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        self.inner.fitted_values_with_intervals(level)
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.inner.residuals()
    }

    fn name(&self) -> &str {
        "WindowAverage"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 1, 1, i as u32 % 24, 0, 0)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn sma_builder_validates_window() {
        // Window 0 is valid (full history)
        let model = SimpleMovingAverage::builder().window(0).build().unwrap();
        assert_eq!(model.window(), 0);
        assert_eq!(model.name(), "SimpleMovingAverage");

        // Positive window is valid
        let model = SimpleMovingAverage::builder().window(5).build().unwrap();
        assert_eq!(model.window(), 5);
    }

    #[test]
    fn sma_requires_sufficient_history() {
        let timestamps = make_timestamps(2);
        let values = vec![1.0, 2.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(3);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData {
                needed: 3,
                got: 2,
                hint: None
            })
        ));

        // Unfitted model can't predict
        assert!(matches!(
            model.predict(5),
            Err(ForecastError::FitRequired { .. })
        ));
    }

    #[test]
    fn sma_rejects_multivariate_input() {
        let timestamps = make_timestamps(5);
        let values = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]];

        let ts = crate::core::TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .multivariate_values(values, crate::core::ValueLayout::Column)
            .build()
            .unwrap();

        let mut model = SimpleMovingAverage::new(3);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InvalidParameter(_))
        ));
    }

    #[test]
    fn sma_forecasts_repeating_averages() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        let preds = forecast.primary();

        // Mean of last 3 values: (3 + 4 + 5) / 3 = 4
        let expected = 4.0;
        assert_relative_eq!(preds[0], expected, epsilon = 1e-10);
        assert_relative_eq!(preds[1], expected, epsilon = 1e-10);
        assert_relative_eq!(preds[2], expected, epsilon = 1e-10);
    }

    #[test]
    fn sma_handles_zero_horizon() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert!(forecast.is_empty());
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn sma_window_0_uses_full_history() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(0);
        model.fit(&ts).unwrap();

        let forecast = model.predict(1).unwrap();
        // Mean of all: (1+2+3+4+5)/5 = 3
        assert_relative_eq!(forecast.primary()[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn sma_window_0_backward_compatibility() {
        // Window 0 should work same as default
        let timestamps = make_timestamps(5);
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(0);
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        // Mean of all: (10+20+30+40+50)/5 = 30
        for pred in forecast.primary() {
            assert_relative_eq!(*pred, 30.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn sma_window_0_vs_window_size() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model0 = SimpleMovingAverage::new(0);
        let mut model5 = SimpleMovingAverage::new(5);

        model0.fit(&ts).unwrap();
        model5.fit(&ts).unwrap();

        let f0 = model0.predict(1).unwrap();
        let f5 = model5.predict(1).unwrap();

        // Both should give same result
        assert_relative_eq!(f0.primary()[0], f5.primary()[0], epsilon = 1e-10);
    }

    #[test]
    fn sma_window_0_on_empty_data() {
        let ts = TimeSeries::univariate(vec![], vec![]).unwrap();
        let mut model = SimpleMovingAverage::new(0);

        assert!(matches!(model.fit(&ts), Err(ForecastError::EmptyData)));
    }

    #[test]
    fn sma_confidence_intervals() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10)
            .map(|i| (i as f64) + 0.5 * (i as f64).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(5);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(3, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..3 {
            assert!(lower[i] < preds[i]);
            assert!(preds[i] < upper[i]);
        }
    }

    #[test]
    fn sma_name_is_correct() {
        let model = SimpleMovingAverage::new(5);
        assert_eq!(model.name(), "SimpleMovingAverage");
    }

    #[test]
    fn sma_fitted_values_and_residuals() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(2);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        let residuals = model.residuals().unwrap();

        assert!(fitted[0].is_nan());
        // fitted[1] = mean(values[0..1]) = 1.0
        assert_relative_eq!(fitted[1], 1.0, epsilon = 1e-10);
        // fitted[2] = mean(values[0..2]) = (1+3)/2 = 2.0
        assert_relative_eq!(fitted[2], 2.0, epsilon = 1e-10);

        // residual[1] = 3.0 - 1.0 = 2.0
        assert_relative_eq!(residuals[1], 2.0, epsilon = 1e-10);
    }

    // =======================================================================
    // HistoricAverage Tests
    // =======================================================================

    #[test]
    fn historic_average_basic() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HistoricAverage::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        // Mean of all: (1+2+3+4+5)/5 = 3
        for pred in forecast.primary() {
            assert_relative_eq!(*pred, 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn historic_average_name() {
        let model = HistoricAverage::new();
        assert_eq!(model.name(), "HistoricAverage");
    }

    #[test]
    fn historic_average_default() {
        let model = HistoricAverage::default();
        assert_eq!(model.name(), "HistoricAverage");
    }

    #[test]
    fn historic_average_with_intervals() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = HistoricAverage::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(3, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    // =======================================================================
    // WindowAverage Tests
    // =======================================================================

    #[test]
    fn window_average_basic() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = WindowAverage::new(3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        // Mean of last 3: (3+4+5)/3 = 4
        for pred in forecast.primary() {
            assert_relative_eq!(*pred, 4.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn window_average_name() {
        let model = WindowAverage::new(5);
        assert_eq!(model.name(), "WindowAverage");
    }

    #[test]
    fn window_average_size() {
        let model = WindowAverage::new(12);
        assert_eq!(model.window_size(), 12);
    }

    #[test]
    fn window_average_minimum_size() {
        // Window of 0 should become 1
        let model = WindowAverage::new(0);
        assert_eq!(model.window_size(), 1);
    }

    #[test]
    fn window_average_with_intervals() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = WindowAverage::new(5);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(3, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    // =======================================================================
    // Changepoint-aware SMA tests
    // =======================================================================

    #[test]
    fn sma_changepoint_constrains_window() {
        // Regime: 10 values of 100.0, then 10 values of 0.0
        // Changepoint at index 10
        let timestamps = make_timestamps(20);
        let mut values = vec![100.0; 10];
        values.extend(vec![0.0; 10]);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Without changepoint: SMA(0) averages all 20 → 50.0
        let mut model = SimpleMovingAverage::new(0);
        model.fit(&ts).unwrap();
        let forecast = model.predict(1).unwrap();
        assert_relative_eq!(forecast.primary()[0], 50.0, epsilon = 1e-10);

        // With changepoint at 10: effective window = 10, averages post-CP → 0.0
        let mut model = SimpleMovingAverage::new(0).with_changepoint(10);
        model.fit(&ts).unwrap();
        let forecast = model.predict(1).unwrap();
        assert_relative_eq!(forecast.primary()[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn sma_changepoint_window_already_smaller() {
        // If window is already smaller than post-CP data, changepoint has no effect
        let timestamps = make_timestamps(20);
        let mut values = vec![100.0; 10];
        values.extend(vec![0.0; 10]);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // SMA(3) with CP at 10: window=3 < post_cp=10, no constraint
        let mut model = SimpleMovingAverage::new(3).with_changepoint(10);
        model.fit(&ts).unwrap();
        let forecast = model.predict(1).unwrap();
        // Mean of last 3 zeros
        assert_relative_eq!(forecast.primary()[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn sma_changepoint_constrains_large_window() {
        // SMA(15) with CP at 15: should only use last 5 observations
        let timestamps = make_timestamps(20);
        let mut values = vec![100.0; 15];
        values.extend(vec![10.0; 5]);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleMovingAverage::new(15).with_changepoint(15);
        model.fit(&ts).unwrap();
        let forecast = model.predict(1).unwrap();
        assert_relative_eq!(forecast.primary()[0], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn sma_builder_with_changepoint() {
        let model = SimpleMovingAverage::builder()
            .window(50)
            .changepoint(180)
            .build()
            .unwrap();
        assert_eq!(model.window(), 50);
        assert_eq!(model.changepoint(), Some(180));
    }

    #[test]
    fn sma_changepoint_regime_change_slope() {
        // Slope +3 until index 15, then slope -2
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20)
            .map(|i| {
                if i < 15 {
                    3.0 * i as f64
                } else {
                    3.0 * 15.0 - 2.0 * (i - 15) as f64
                }
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        // Without CP: averages across both regimes
        let mut without = SimpleMovingAverage::new(0);
        without.fit(&ts).unwrap();
        let fc_without = without.predict(1).unwrap();

        // With CP at 15: averages only post-CP (5 values)
        let mut with_cp = SimpleMovingAverage::new(0).with_changepoint(15);
        with_cp.fit(&ts).unwrap();
        let fc_with = with_cp.predict(1).unwrap();

        let post_cp_mean: f64 = values[15..].iter().sum::<f64>() / 5.0;
        assert_relative_eq!(fc_with.primary()[0], post_cp_mean, epsilon = 1e-10);
        // The CP-aware forecast should be closer to recent values
        assert!(
            (fc_with.primary()[0] - values[19]).abs()
                < (fc_without.primary()[0] - values[19]).abs()
        );
    }
}
