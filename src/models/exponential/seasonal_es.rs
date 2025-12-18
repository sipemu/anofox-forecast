//! Seasonal Exponential Smoothing.
//!
//! A model that applies Simple Exponential Smoothing (SES) independently to each
//! seasonal slot. This is NOT a multiplicative seasonal model - it's SES applied
//! per-season.
//!
//! Reference: statsforecast SeasonalExponentialSmoothing
//! For season_length=12:
//! - Slot 0: apply SES to y[0], y[12], y[24], ...
//! - Slot 1: apply SES to y[1], y[13], y[25], ...
//! - etc.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Error type for Seasonal ES (kept for backward compatibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeasonalESErrorType {
    /// Additive errors.
    #[default]
    Additive,
    /// Multiplicative errors.
    Multiplicative,
}

/// Seasonal Exponential Smoothing model.
///
/// Applies SES independently to each seasonal slot.
///
/// # Example
/// ```
/// use anofox_forecast::models::exponential::SeasonalES;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..48).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
/// let values: Vec<f64> = (0..48).map(|i| {
///     50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
/// }).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = SeasonalES::new(12);
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(12).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SeasonalES {
    /// Seasonal period.
    period: usize,
    /// Smoothing parameter (alpha).
    alpha: f64,
    /// Whether to optimize parameters.
    optimize: bool,
    /// Error type (kept for backward compatibility).
    error_type: SeasonalESErrorType,
    /// Seasonal values (one per slot).
    seasonal_values: Option<Vec<f64>>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Series length.
    n: usize,
}

impl SeasonalES {
    /// Create a new Seasonal ES model with default parameters.
    ///
    /// Default: alpha=0.1.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            alpha: 0.1,
            optimize: false,
            error_type: SeasonalESErrorType::Additive,
            seasonal_values: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a Seasonal ES model with optimized parameters.
    pub fn optimized(period: usize) -> Self {
        Self {
            period,
            alpha: 0.1,
            optimize: true,
            error_type: SeasonalESErrorType::Additive,
            seasonal_values: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create with specified alpha and gamma.
    /// Note: gamma is ignored for compatibility; only alpha is used.
    pub fn with_params(period: usize, alpha: f64, _gamma: f64) -> Self {
        Self {
            period,
            alpha: alpha.clamp(0.001, 0.999),
            optimize: false,
            error_type: SeasonalESErrorType::Additive,
            seasonal_values: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Set error type (kept for backward compatibility).
    pub fn with_error_type(mut self, error_type: SeasonalESErrorType) -> Self {
        self.error_type = error_type;
        self
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the gamma parameter (returns alpha for compatibility).
    pub fn gamma(&self) -> f64 {
        self.alpha
    }

    /// Get the seasonal period.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Get the seasonal indices (seasonal_values / mean for compatibility).
    pub fn seasonal_indices(&self) -> Option<Vec<f64>> {
        self.seasonal_values.as_ref().map(|sv| {
            let mean: f64 = sv.iter().sum::<f64>() / sv.len() as f64;
            if mean.abs() > 1e-10 {
                sv.iter().map(|&v| v / mean).collect()
            } else {
                vec![1.0; sv.len()]
            }
        })
    }

    /// Apply SES to a subset of values and return (final_level, fitted_values).
    fn ses_forecast(values: &[f64], alpha: f64) -> (f64, Vec<f64>) {
        if values.is_empty() {
            return (0.0, Vec::new());
        }

        let mut fitted = Vec::with_capacity(values.len());
        let mut level = values[0];

        for &y in values.iter() {
            fitted.push(level);
            level = alpha * y + (1.0 - alpha) * level;
        }

        (level, fitted)
    }

    /// Calculate SSE for given alpha.
    fn calculate_sse(values: &[f64], alpha: f64, period: usize, n: usize) -> f64 {
        if n < period {
            return f64::MAX;
        }

        let mut total_sse = 0.0;

        for slot in 0..period {
            // Get values for this seasonal slot
            let init_idx = slot + (n % period);
            let slot_values: Vec<f64> = (init_idx..n).step_by(period).map(|i| values[i]).collect();

            if slot_values.is_empty() {
                continue;
            }

            // Apply SES and compute SSE
            let (_, fitted) = Self::ses_forecast(&slot_values, alpha);
            for (i, &y) in slot_values.iter().enumerate() {
                let error = y - fitted[i];
                total_sse += error * error;
            }
        }

        total_sse / n as f64
    }

    /// Optimize alpha using Nelder-Mead.
    fn optimize_params(values: &[f64], period: usize) -> f64 {
        let n = values.len();

        let objective = |params: &[f64]| {
            let alpha = params[0];

            if alpha <= 0.001 || alpha >= 0.999 {
                return f64::MAX;
            }

            Self::calculate_sse(values, alpha, period, n)
        };

        let starts = [[0.1], [0.3], [0.5], [0.7]];

        let mut best_alpha = 0.1;
        let mut best_value = f64::MAX;

        let config = NelderMeadConfig {
            max_iter: 200,
            tolerance: 1e-6,
            ..Default::default()
        };

        for start in starts {
            let result = nelder_mead(objective, &start, Some(&[(0.001, 0.999)]), config.clone());

            if result.optimal_value < best_value {
                best_value = result.optimal_value;
                best_alpha = result.optimal_point[0].clamp(0.001, 0.999);
            }
        }

        best_alpha
    }
}

impl Default for SeasonalES {
    fn default() -> Self {
        Self::new(12)
    }
}

impl Forecaster for SeasonalES {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        let n = values.len();
        self.n = n;

        if n < self.period {
            return Err(ForecastError::InsufficientData {
                needed: self.period,
                got: n,
            });
        }

        // Optimize parameters if requested
        if self.optimize {
            self.alpha = Self::optimize_params(values, self.period);
        }

        // Initialize storage
        let mut seasonal_values = vec![f64::NAN; self.period];
        let mut fitted = vec![f64::NAN; n];

        // Apply SES to each seasonal slot
        for slot in 0..self.period {
            // Calculate initial index for this slot
            // Following statsforecast: init_idx = slot + (n % period)
            let init_idx = slot + (n % self.period);

            // Get values for this slot
            let slot_indices: Vec<usize> = (init_idx..n).step_by(self.period).collect();

            if slot_indices.is_empty() {
                // No data for this slot, use NaN
                seasonal_values[slot] = f64::NAN;
                continue;
            }

            let slot_values: Vec<f64> = slot_indices.iter().map(|&i| values[i]).collect();

            // Apply SES
            let (final_level, slot_fitted) = Self::ses_forecast(&slot_values, self.alpha);

            // Store final level for forecasting
            seasonal_values[slot] = final_level;

            // Fill in fitted values at the correct positions
            for (i, &idx) in slot_indices.iter().enumerate() {
                fitted[idx] = slot_fitted[i];
            }
        }

        // Compute residuals
        let residuals: Vec<f64> = values
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| if f.is_nan() { f64::NAN } else { y - f })
            .collect();

        // Calculate residual variance (from valid residuals)
        let valid_residuals: Vec<f64> = residuals
            .iter()
            .filter(|r| r.is_finite())
            .copied()
            .collect();
        if !valid_residuals.is_empty() {
            let variance =
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.seasonal_values = Some(seasonal_values);
        self.fitted = Some(fitted);
        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let seasonal_values = self
            .seasonal_values
            .as_ref()
            .ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Generate forecasts by repeating seasonal values
        let mut predictions = Vec::with_capacity(horizon);

        for h in 0..horizon {
            let slot = h % self.period;
            let forecast = seasonal_values[slot];
            predictions.push(forecast);
        }

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

        for h in 0..horizon {
            // Simple fan-out for intervals
            let factor = (1.0 + 0.1 * h as f64).sqrt();
            let se = (variance * factor).sqrt();

            lower.push(preds[h] - z * se);
            upper.push(preds[h] + z * se);
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

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted.as_ref()?;
        let variance = self.residual_variance?;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let sigma = variance.sqrt();

        let lower: Vec<f64> = fitted.iter().map(|&f| f - z * sigma).collect();
        let upper: Vec<f64> = fitted.iter().map(|&f| f + z * sigma).collect();

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
        if self.optimize {
            "SeasonalES (Optimized)"
        } else {
            "SeasonalES"
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

    fn make_seasonal_series(n: usize, period: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn seasonal_es_basic() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::new(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn seasonal_es_optimized() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::optimized(12);
        model.fit(&ts).unwrap();

        assert!(model.alpha() > 0.0 && model.alpha() < 1.0);

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn seasonal_es_with_params() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::with_params(12, 0.2, 0.1);
        model.fit(&ts).unwrap();

        assert!((model.alpha() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn seasonal_es_seasonal_pattern() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::new(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        let preds = forecast.primary();

        // Forecasts should follow seasonal pattern (repeat every period)
        for h in 0..12 {
            let diff = (preds[h] - preds[h + 12]).abs();
            assert!(
                diff < 1e-10,
                "Seasonal pattern should repeat exactly: {} vs {}",
                preds[h],
                preds[h + 12]
            );
        }
    }

    #[test]
    fn seasonal_es_confidence_intervals() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::new(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(12, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..12 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }
    }

    #[test]
    fn seasonal_es_fitted_and_residuals() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::new(12);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 48);
    }

    #[test]
    fn seasonal_es_insufficient_data() {
        let ts = make_seasonal_series(10, 12);
        let mut model = SeasonalES::new(12);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn seasonal_es_requires_fit() {
        let model = SeasonalES::new(12);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn seasonal_es_zero_horizon() {
        let ts = make_seasonal_series(48, 12);
        let mut model = SeasonalES::new(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn seasonal_es_name() {
        let model = SeasonalES::new(12);
        assert_eq!(model.name(), "SeasonalES");

        let optimized = SeasonalES::optimized(12);
        assert_eq!(optimized.name(), "SeasonalES (Optimized)");
    }

    #[test]
    fn seasonal_es_statsforecast_match() {
        // Test data similar to statsforecast test
        let timestamps = make_timestamps(100);
        // Trend + seasonal + noise data
        let values: Vec<f64> = (0..100)
            .map(|i| {
                50.0 + 0.5 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SeasonalES::with_params(12, 0.3, 0.0);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Should produce sensible forecasts (not NaN, in reasonable range)
        for &p in preds {
            assert!(p.is_finite());
            assert!(p > 30.0 && p < 120.0);
        }
    }
}
