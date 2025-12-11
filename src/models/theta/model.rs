//! Theta forecasting model.
//!
//! The Theta method was the winning method in the M3 forecasting competition.
//! This implementation follows the state-space formulation from Fiorucci et al. (2016)
//! and matches the NIXTLA statsforecast implementation.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Type of seasonal decomposition for Theta model.
///
/// Controls how seasonal components are extracted and reapplied.
/// Default is `Multiplicative` to match NIXTLA statsforecast behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompositionType {
    /// Additive decomposition: y = trend + seasonal + remainder
    ///
    /// Use when seasonal variation is constant regardless of the level.
    Additive,
    /// Multiplicative decomposition: y = trend * seasonal * remainder
    ///
    /// Use when seasonal variation scales with the level (default, matching statsforecast).
    #[default]
    Multiplicative,
}

/// Theta forecasting model.
///
/// The standard Theta method (STM) uses theta=2. This implementation follows
/// the state-space formulation from Fiorucci et al. (2016) which applies SES
/// to the original series and uses the regression slope for the drift component.
///
/// Forecast formula: `smoothed + (1 - 1/theta) * b * (1/alpha + h - 1)`
///
/// For seasonal data, supports both additive and multiplicative decomposition.
/// Default is multiplicative to match NIXTLA statsforecast behavior.
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
    /// Type of seasonal decomposition (default: Multiplicative).
    decomposition_type: DecompositionType,
    /// Whether decomposition type was automatically changed due to fallback rules.
    decomposition_fallback: bool,
    /// Linear regression slope (B coefficient).
    b: Option<f64>,
    /// Fitted level (from SES on original series).
    level: Option<f64>,
    /// Seasonal indices (averaged, used for deseasonalizing).
    seasonals: Option<Vec<f64>>,
    /// Seasonal forecast pattern (last cycle, used for reseasonalizing forecasts).
    /// This matches statsforecast's seasonal_naive behavior.
    seasonal_forecast: Option<Vec<f64>>,
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
    ///
    /// Uses the Standard Theta Model (STM) from Fiorucci et al. (2016):
    /// - theta = 2.0
    /// - alpha = 0.1 (fixed, not optimized, matching statsforecast)
    pub fn new() -> Self {
        Self {
            theta: 2.0,
            alpha: Some(0.1), // Fixed alpha=0.1 to match statsforecast STM
            optimize: false,  // STM uses fixed parameters
            seasonal_period: 0,
            decomposition_type: DecompositionType::Multiplicative,
            decomposition_fallback: false,
            b: None,
            level: None,
            seasonals: None,
            seasonal_forecast: None,
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

    /// Create a seasonal Theta model with multiplicative decomposition (default, matching statsforecast).
    pub fn seasonal(period: usize) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: DecompositionType::Multiplicative,
            ..Self::new()
        }
    }

    /// Create a seasonal Theta model with explicit decomposition type.
    ///
    /// # Arguments
    /// * `period` - The seasonal period (e.g., 12 for monthly data with yearly seasonality)
    /// * `decomposition` - The type of seasonal decomposition to use
    ///
    /// # Example
    /// ```
    /// use anofox_forecast::models::theta::{Theta, DecompositionType};
    ///
    /// // Use additive decomposition for data with constant seasonal amplitude
    /// let model = Theta::seasonal_with_decomposition(12, DecompositionType::Additive);
    /// ```
    pub fn seasonal_with_decomposition(period: usize, decomposition: DecompositionType) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: decomposition,
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

    /// Create a Theta model with optimized alpha.
    ///
    /// Unlike the default STM which uses fixed alpha=0.1, this variant
    /// optimizes alpha by minimizing the sum of squared errors.
    ///
    /// # Example
    /// ```
    /// use anofox_forecast::models::theta::Theta;
    /// use anofox_forecast::models::Forecaster;
    /// use anofox_forecast::core::TimeSeries;
    /// use chrono::{TimeZone, Utc};
    ///
    /// let timestamps: Vec<_> = (0..50).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
    /// let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
    /// let ts = TimeSeries::univariate(timestamps, values).unwrap();
    ///
    /// let mut model = Theta::optimized();
    /// model.fit(&ts).unwrap();
    /// println!("Optimized alpha: {}", model.alpha().unwrap());
    /// ```
    pub fn optimized() -> Self {
        Self {
            alpha: None,
            optimize: true,
            ..Self::new()
        }
    }

    /// Create a seasonal Theta model with optimized alpha.
    ///
    /// Combines seasonal decomposition with alpha optimization.
    pub fn seasonal_optimized(period: usize) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: DecompositionType::Multiplicative,
            alpha: None,
            optimize: true,
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

    /// Get the regression slope (B coefficient).
    pub fn slope(&self) -> Option<f64> {
        self.b
    }

    /// Get the drift (linear trend slope).
    /// Deprecated: use slope() instead.
    #[deprecated(note = "Use slope() instead")]
    pub fn drift(&self) -> Option<f64> {
        self.b
    }

    /// Get the decomposition type used.
    ///
    /// Note: This may differ from the initially requested type if fallback rules
    /// were applied (e.g., multiplicative fell back to additive for non-positive data).
    pub fn decomposition_type(&self) -> DecompositionType {
        self.decomposition_type
    }

    /// Check if the decomposition type was automatically changed due to fallback rules.
    ///
    /// Returns `true` if multiplicative decomposition was requested but the model
    /// fell back to additive due to:
    /// - Non-positive values in the data
    /// - Seasonal factors less than 0.01
    pub fn used_fallback(&self) -> bool {
        self.decomposition_fallback
    }

    /// Deseasonalize a series based on decomposition type.
    fn deseasonalize(&self, series: &[f64], seasonals: &[f64]) -> Vec<f64> {
        if seasonals.is_empty() || self.seasonal_period == 0 {
            return series.to_vec();
        }

        match self.decomposition_type {
            DecompositionType::Additive => {
                // Additive: y_deseasonalized = y - seasonal
                series
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| y - seasonals[i % self.seasonal_period])
                    .collect()
            }
            DecompositionType::Multiplicative => {
                // Multiplicative: y_deseasonalized = y / seasonal
                series
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| {
                        let s = seasonals[i % self.seasonal_period];
                        if s.abs() < 1e-10 {
                            y
                        } else {
                            y / s
                        }
                    })
                    .collect()
            }
        }
    }

    /// Reseasonalize forecasts based on decomposition type.
    fn reseasonalize(&self, forecasts: &[f64], start_idx: usize, seasonals: &[f64]) -> Vec<f64> {
        if seasonals.is_empty() || self.seasonal_period == 0 {
            return forecasts.to_vec();
        }

        match self.decomposition_type {
            DecompositionType::Additive => {
                // Additive: forecast_reseasonalized = forecast + seasonal
                forecasts
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| y + seasonals[(start_idx + i) % self.seasonal_period])
                    .collect()
            }
            DecompositionType::Multiplicative => {
                // Multiplicative: forecast_reseasonalized = forecast * seasonal
                forecasts
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| y * seasonals[(start_idx + i) % self.seasonal_period])
                    .collect()
            }
        }
    }

    /// Calculate full seasonal component using classical decomposition.
    ///
    /// This matches statsforecast's use of statsmodels seasonal_decompose.
    ///
    /// # Arguments
    /// * `series` - The time series data
    /// * `decomposition` - The type of decomposition to use
    ///
    /// # Returns
    /// Tuple of (full_seasonal_component, last_cycle_for_forecasting)
    fn calculate_seasonal_component(
        &self,
        series: &[f64],
        decomposition: DecompositionType,
    ) -> (Vec<f64>, Vec<f64>) {
        let period = self.seasonal_period;
        if period == 0 || series.len() < 2 * period {
            return (vec![], vec![]);
        }

        // Calculate trend using centered moving average
        let half = period / 2;
        let mut trend = vec![f64::NAN; series.len()];

        for i in half..(series.len() - half) {
            let sum: f64 = if period.is_multiple_of(2) {
                // Even period: weighted endpoints (2x12-MA for period=12)
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

        // Calculate detrended series based on decomposition type
        let detrended: Vec<f64> = match decomposition {
            DecompositionType::Additive => series
                .iter()
                .zip(trend.iter())
                .map(|(&y, &t)| if t.is_nan() { f64::NAN } else { y - t })
                .collect(),
            DecompositionType::Multiplicative => series
                .iter()
                .zip(trend.iter())
                .map(|(&y, &t)| {
                    if t.is_nan() || t.abs() < 1e-10 {
                        f64::NAN
                    } else {
                        y / t
                    }
                })
                .collect(),
        };

        // Average by season to get seasonal indices
        let mut seasonal_indices = vec![0.0; period];
        let mut counts = vec![0usize; period];

        for (i, &d) in detrended.iter().enumerate() {
            if !d.is_nan() {
                seasonal_indices[i % period] += d;
                counts[i % period] += 1;
            }
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonal_indices[i] /= counts[i] as f64;
            }
        }

        // Normalize based on decomposition type
        match decomposition {
            DecompositionType::Additive => {
                let mean = seasonal_indices.iter().sum::<f64>() / period as f64;
                for s in &mut seasonal_indices {
                    *s -= mean;
                }
            }
            DecompositionType::Multiplicative => {
                let mean = seasonal_indices.iter().sum::<f64>() / period as f64;
                if mean.abs() > 1e-10 {
                    for s in &mut seasonal_indices {
                        *s /= mean;
                    }
                }
            }
        }

        // Build full seasonal component by repeating indices
        let full_seasonal: Vec<f64> = (0..series.len())
            .map(|i| seasonal_indices[i % period])
            .collect();

        // Extract last cycle for forecasting (matches statsforecast's seasonal_naive)
        // The last cycle starts at (n - period) and contains period values
        let last_cycle: Vec<f64> = full_seasonal[(series.len() - period)..].to_vec();

        (full_seasonal, last_cycle)
    }

    /// Calculate seasonal indices using classical decomposition.
    ///
    /// # Arguments
    /// * `series` - The time series data
    /// * `decomposition` - The type of decomposition to use
    ///
    /// # Returns
    /// Vector of seasonal indices (length = seasonal_period)
    fn calculate_seasonals(&self, series: &[f64], decomposition: DecompositionType) -> Vec<f64> {
        let (_, last_cycle) = self.calculate_seasonal_component(series, decomposition);
        // Return averaged indices for backward compatibility
        if last_cycle.is_empty() {
            return vec![];
        }
        let period = self.seasonal_period;
        let mut seasonals = vec![0.0; period];
        for (i, &s) in last_cycle.iter().enumerate() {
            seasonals[i] = s;
        }
        seasonals
    }

    /// Determine effective decomposition type with statsforecast-style fallback.
    ///
    /// Fallback rules (matching statsforecast):
    /// 1. Non-positive data -> switch to additive
    /// 2. Seasonal factors < 0.01 -> switch to additive
    fn determine_decomposition(&mut self, series: &[f64]) -> DecompositionType {
        // If additive was explicitly requested, use it
        if self.decomposition_type == DecompositionType::Additive {
            return DecompositionType::Additive;
        }

        // Rule 1: Check for non-positive values
        // Multiplicative decomposition requires all positive values
        if series.iter().any(|&y| y <= 0.0) {
            self.decomposition_fallback = true;
            return DecompositionType::Additive;
        }

        // Rule 2: Try multiplicative and check if seasonal factors are valid
        let seasonals = self.calculate_seasonals(series, DecompositionType::Multiplicative);
        if !seasonals.is_empty() {
            // Check if any seasonal factor is too small (< 0.01)
            // This matches statsforecast's behavior for numerical stability
            if seasonals.iter().any(|&s| s < 0.01) {
                self.decomposition_fallback = true;
                return DecompositionType::Additive;
            }
        }

        DecompositionType::Multiplicative
    }

    /// Calculate SSE for SES.
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
        self.decomposition_fallback = false;

        // Determine effective decomposition type with fallback rules (if seasonal)
        let effective_decomposition = if self.seasonal_period > 0 {
            self.determine_decomposition(values)
        } else {
            self.decomposition_type
        };
        self.decomposition_type = effective_decomposition;

        // Calculate seasonal component if needed using the determined decomposition type
        let (full_seasonal, seasonal_forecast) = if self.seasonal_period > 0 {
            self.calculate_seasonal_component(values, effective_decomposition)
        } else {
            (vec![], vec![])
        };

        // Deseasonalize using full seasonal component
        let deseasonalized = self.deseasonalize(values, &full_seasonal);

        // Calculate linear regression slope (B coefficient)
        // This matches statsforecast's approach
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

        let b = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        self.b = Some(b);

        // Optimize alpha on the ORIGINAL (deseasonalized) series
        // Note: statsforecast applies SES to original series, not theta-transformed
        if self.optimize {
            self.alpha = Some(Self::optimize_alpha(&deseasonalized));
        }

        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;

        // Apply SES to the ORIGINAL (deseasonalized) series
        // This matches statsforecast's approach
        let mut level = deseasonalized[0];
        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // First fitted value (reseasonalized)
        let first_fitted = if full_seasonal.is_empty() {
            deseasonalized[0]
        } else {
            match self.decomposition_type {
                DecompositionType::Additive => deseasonalized[0] + full_seasonal[0],
                DecompositionType::Multiplicative => deseasonalized[0] * full_seasonal[0],
            }
        };
        fitted.push(first_fitted);
        residuals.push(0.0);

        for i in 1..self.n {
            let forecast = level;
            let seasonalized_forecast = if full_seasonal.is_empty() {
                forecast
            } else {
                match self.decomposition_type {
                    DecompositionType::Additive => forecast + full_seasonal[i],
                    DecompositionType::Multiplicative => forecast * full_seasonal[i],
                }
            };

            fitted.push(seasonalized_forecast);
            residuals.push(values[i] - seasonalized_forecast);

            level = alpha * deseasonalized[i] + (1.0 - alpha) * level;
        }

        self.level = Some(level);
        // Store averaged seasonal indices (for backward compatibility)
        self.seasonals = if seasonal_forecast.is_empty() {
            None
        } else {
            // Convert last_cycle back to position-based indices for backward compatibility
            let period = self.seasonal_period;
            let mut averaged_indices = vec![0.0; period];
            let start_pos = (self.n - period) % period;
            for i in 0..period {
                averaged_indices[(start_pos + i) % period] = seasonal_forecast[i];
            }
            Some(averaged_indices)
        };
        // Store seasonal forecast pattern for forecasting (last cycle, used with start_idx=0)
        self.seasonal_forecast = if seasonal_forecast.is_empty() {
            None
        } else {
            Some(seasonal_forecast)
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
        let smoothed = self.level.ok_or(ForecastError::FitRequired)?;
        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;
        let b = self.b.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Fiorucci et al. (2016) state-space formulation matching statsforecast:
        // forecast(h) = smoothed + (1 - 1/theta) * b * (1/alpha + h - 1)
        let mut predictions = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let forecast =
                smoothed + (1.0 - 1.0 / self.theta) * b * (1.0 / alpha + (h as f64 - 1.0));
            predictions.push(forecast);
        }

        // Reseasonalize if needed using seasonal_forecast (last cycle)
        // with start_idx=0 to match statsforecast's _repeat_val_seas behavior
        let predictions = if let Some(ref seasonal_forecast) = self.seasonal_forecast {
            self.reseasonalize(&predictions, 0, seasonal_forecast)
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
    fn theta_slope_positive_for_trend() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::new();
        model.fit(&ts).unwrap();

        // Slope should be positive for upward trend
        assert!(model.slope().unwrap() > 0.0);
    }

    #[test]
    fn theta_multiplicative_default() {
        // Default seasonal model should use multiplicative decomposition
        let model = Theta::seasonal(12);
        assert_eq!(
            model.decomposition_type(),
            DecompositionType::Multiplicative
        );
        assert!(!model.used_fallback());
    }

    #[test]
    fn theta_additive_explicit() {
        // Explicit additive decomposition
        let model = Theta::seasonal_with_decomposition(12, DecompositionType::Additive);
        assert_eq!(model.decomposition_type(), DecompositionType::Additive);
    }

    #[test]
    fn theta_multiplicative_seasonal_positive_data() {
        // Multiplicative decomposition on positive seasonal data
        let timestamps = make_timestamps(48);
        // Create multiplicative seasonal pattern: base * seasonal_factor
        let values: Vec<f64> = (0..48)
            .map(|i| {
                let base = 100.0;
                // Seasonal factor oscillates around 1.0 (e.g., 0.8 to 1.2)
                let seasonal_factor =
                    1.0 + 0.2 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                base * seasonal_factor
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12);
        model.fit(&ts).unwrap();

        // Should stay multiplicative (positive data, reasonable seasonal factors)
        assert_eq!(
            model.decomposition_type(),
            DecompositionType::Multiplicative
        );
        assert!(!model.used_fallback());

        // Seasonal indices should average close to 1.0 for multiplicative
        if let Some(seasonals) = &model.seasonals {
            let mean = seasonals.iter().sum::<f64>() / seasonals.len() as f64;
            assert_relative_eq!(mean, 1.0, epsilon = 0.05);
        }

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn theta_fallback_for_negative_values() {
        // Series with negative values should trigger fallback to additive
        let timestamps = make_timestamps(48);
        // Values that go negative
        let values: Vec<f64> = (0..48)
            .map(|i| 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12);
        model.fit(&ts).unwrap();

        // Should have fallen back to additive due to negative values
        assert!(model.used_fallback());
        assert_eq!(model.decomposition_type(), DecompositionType::Additive);
    }

    #[test]
    fn theta_additive_seasonals_sum_to_zero() {
        // Additive seasonal indices should sum to zero
        let timestamps = make_timestamps(48);
        let values: Vec<f64> = (0..48)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal_with_decomposition(12, DecompositionType::Additive);
        model.fit(&ts).unwrap();

        assert_eq!(model.decomposition_type(), DecompositionType::Additive);

        if let Some(seasonals) = &model.seasonals {
            let sum: f64 = seasonals.iter().sum();
            assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn theta_multiplicative_seasonals_average_to_one() {
        // Multiplicative seasonal indices should average to 1.0
        let timestamps = make_timestamps(48);
        // All positive values
        let values: Vec<f64> = (0..48)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12);
        model.fit(&ts).unwrap();

        // Should be multiplicative for all-positive data
        assert_eq!(
            model.decomposition_type(),
            DecompositionType::Multiplicative
        );

        if let Some(seasonals) = &model.seasonals {
            let mean = seasonals.iter().sum::<f64>() / seasonals.len() as f64;
            assert_relative_eq!(mean, 1.0, epsilon = 0.01);
        }
    }

    #[test]
    fn theta_decomposition_type_enum_default() {
        // DecompositionType should default to Multiplicative
        let default_type: DecompositionType = Default::default();
        assert_eq!(default_type, DecompositionType::Multiplicative);
    }

    #[test]
    fn theta_stm_uses_fixed_alpha() {
        // Standard Theta Model (STM) should use fixed alpha=0.1 (matching statsforecast)
        let model = Theta::new();
        assert_relative_eq!(model.alpha().unwrap(), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn theta_optimized_finds_alpha() {
        // Optimized variant should find alpha via optimization
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::optimized();
        // Before fit, alpha should be None
        assert!(model.alpha().is_none());

        model.fit(&ts).unwrap();

        // After fit, alpha should be optimized (likely different from 0.1)
        let alpha = model.alpha().unwrap();
        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn theta_seasonal_optimized() {
        // Seasonal optimized variant should combine seasonality with alpha optimization
        let timestamps = make_timestamps(48);
        let values: Vec<f64> = (0..48)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal_optimized(12);
        model.fit(&ts).unwrap();

        // Should have seasonal forecast stored
        assert!(model.seasonal_forecast.is_some());

        // Alpha should be optimized
        let alpha = model.alpha().unwrap();
        assert!(alpha > 0.0 && alpha < 1.0);

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn theta_seasonal_forecast_stored() {
        // Seasonal forecast pattern (last cycle) should be stored for forecasting
        let timestamps = make_timestamps(48);
        let values: Vec<f64> = (0..48)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12);
        model.fit(&ts).unwrap();

        // seasonal_forecast should be stored and have length = period
        assert!(model.seasonal_forecast.is_some());
        assert_eq!(model.seasonal_forecast.as_ref().unwrap().len(), 12);
    }

    #[test]
    fn theta_seasonal_forecast_matches_last_cycle() {
        // The seasonal_forecast should contain the last cycle of the seasonal component
        let timestamps = make_timestamps(48);
        // Create a clear seasonal pattern
        let values: Vec<f64> = (0..48)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal_with_decomposition(12, DecompositionType::Additive);
        model.fit(&ts).unwrap();

        let seasonal_forecast = model.seasonal_forecast.as_ref().unwrap();

        // The seasonal_forecast[0] should correspond to the seasonal value at position 36 (48-12)
        // which is seasonal position 36 % 12 = 0
        // For pure sine wave centered at 50, this should be close to 0 (the seasonal component)
        // Since seasonal_forecast is the last cycle, it should repeat the seasonal pattern
        assert_eq!(seasonal_forecast.len(), 12);

        // Verify it sums to approximately 0 (additive normalization)
        let sum: f64 = seasonal_forecast.iter().sum();
        assert_relative_eq!(sum, 0.0, epsilon = 0.1);
    }

    #[test]
    fn theta_additive_seasonal_forecast_accuracy() {
        // Test that additive seasonal forecasting produces consistent results
        let timestamps = make_timestamps(100);
        // Create additive seasonal pattern
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal_with_decomposition(12, DecompositionType::Additive);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Forecasts should follow the seasonal pattern
        // The pattern should repeat every 12 steps
        assert_eq!(preds.len(), 12);

        // Forecasts should be reasonable (within the data range)
        for &p in preds {
            assert!(p > 30.0 && p < 70.0, "Forecast {} out of expected range", p);
        }
    }

    #[test]
    fn theta_multiplicative_seasonal_forecast_accuracy() {
        // Test that multiplicative seasonal forecasting produces consistent results
        let timestamps = make_timestamps(100);
        // Create multiplicative seasonal pattern: level * seasonal_factor
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let level = 100.0 + 0.5 * i as f64; // Slight trend
                let seasonal_factor =
                    1.0 + 0.3 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                level * seasonal_factor
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12);
        model.fit(&ts).unwrap();

        // Should stay multiplicative
        assert_eq!(
            model.decomposition_type(),
            DecompositionType::Multiplicative
        );

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Forecasts should be reasonable
        assert_eq!(preds.len(), 12);
        for &p in preds {
            assert!(
                p > 50.0 && p < 250.0,
                "Forecast {} out of expected range",
                p
            );
        }
    }

    #[test]
    fn theta_fallback_preserves_forecast_quality() {
        // When falling back to additive, forecasts should still be reasonable
        let timestamps = make_timestamps(100);
        // Series with negative values (triggers fallback)
        let values: Vec<f64> = (0..100)
            .map(|i| 5.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Theta::seasonal(12); // Requests multiplicative
        model.fit(&ts).unwrap();

        // Should have fallen back
        assert!(model.used_fallback());
        assert_eq!(model.decomposition_type(), DecompositionType::Additive);

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Forecasts should still be reasonable despite fallback
        for &p in preds {
            assert!(
                p > -10.0 && p < 20.0,
                "Forecast {} out of expected range after fallback",
                p
            );
        }
    }
}
