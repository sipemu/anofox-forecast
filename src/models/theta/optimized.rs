//! Optimized Theta Model (OTM).
//!
//! The Optimized Theta Model extends the Standard Theta Model by optimizing
//! both the smoothing parameter (alpha) and theta parameter to minimize
//! prediction error.
//!
//! Based on Fiorucci et al. (2016) "Models for optimising the theta method
//! and their relationship to state space models."

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::theta::DecompositionType;
use crate::models::Forecaster;
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

/// Optimized Theta Model (OTM).
///
/// Unlike the Standard Theta Model (STM) which uses fixed parameters,
/// OTM optimizes alpha and theta to minimize in-sample prediction error.
///
/// Supports seasonal decomposition matching statsforecast's OptimizedTheta.
///
/// # Example
/// ```
/// use anofox_forecast::models::theta::OptimizedTheta;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..50).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
/// let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin()).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = OptimizedTheta::new();
/// model.fit(&ts).unwrap();
/// println!("Optimized alpha: {:.4}, theta: {:.4}", model.alpha().unwrap(), model.theta().unwrap());
/// ```
#[derive(Debug, Clone)]
pub struct OptimizedTheta {
    /// Optimized smoothing parameter.
    alpha: Option<f64>,
    /// Optimized theta parameter.
    theta: Option<f64>,
    /// Linear regression slope (B coefficient).
    b: Option<f64>,
    /// Initial level.
    initial_level: Option<f64>,
    /// Fitted level (from SES).
    level: Option<f64>,
    /// Seasonal period (0 for non-seasonal).
    seasonal_period: usize,
    /// Type of seasonal decomposition (default: Multiplicative).
    decomposition_type: DecompositionType,
    /// Whether decomposition type was automatically changed due to fallback rules.
    decomposition_fallback: bool,
    /// Seasonal forecast pattern (last cycle, used for reseasonalizing forecasts).
    seasonal_forecast: Option<Vec<f64>>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Series length.
    n: usize,
    /// OLS result for exogenous regressors (if any).
    exog_ols: Option<OLSResult>,
}

impl OptimizedTheta {
    /// Create a new Optimized Theta model (non-seasonal).
    pub fn new() -> Self {
        Self {
            alpha: None,
            theta: None,
            b: None,
            initial_level: None,
            level: None,
            seasonal_period: 0,
            decomposition_type: DecompositionType::Multiplicative,
            decomposition_fallback: false,
            seasonal_forecast: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
            exog_ols: None,
        }
    }

    /// Create a seasonal Optimized Theta model with multiplicative decomposition (default).
    ///
    /// Matches statsforecast's OptimizedTheta(season_length=period).
    pub fn seasonal(period: usize) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: DecompositionType::Multiplicative,
            ..Self::new()
        }
    }

    /// Create a seasonal Optimized Theta model with explicit decomposition type.
    pub fn seasonal_with_decomposition(period: usize, decomposition: DecompositionType) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: decomposition,
            ..Self::new()
        }
    }

    /// Get the optimized alpha parameter.
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    /// Get the optimized theta parameter.
    pub fn theta(&self) -> Option<f64> {
        self.theta
    }

    /// Get the regression slope.
    pub fn slope(&self) -> Option<f64> {
        self.b
    }

    /// Get the decomposition type used.
    pub fn decomposition_type(&self) -> DecompositionType {
        self.decomposition_type
    }

    /// Check if the decomposition type was automatically changed due to fallback rules.
    pub fn used_fallback(&self) -> bool {
        self.decomposition_fallback
    }

    /// Calculate linear regression slope.
    fn calculate_slope(values: &[f64]) -> f64 {
        let n = values.len();
        if n < 2 {
            return 0.0;
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

        if ss_xx > 0.0 {
            ss_xy / ss_xx
        } else {
            0.0
        }
    }

    /// Calculate MSE for given parameters using multi-step ahead forecasting.
    fn calculate_mse(values: &[f64], alpha: f64, theta: f64, b: f64, steps_ahead: usize) -> f64 {
        if values.len() < steps_ahead + 2 {
            return f64::MAX;
        }

        let mut level = values[0];
        let mut sse = 0.0;
        let mut count = 0;

        // Calculate multi-step ahead forecast errors
        for i in 1..(values.len() - steps_ahead) {
            // Make h-step forecast from position i
            for h in 1..=steps_ahead {
                if i + h < values.len() {
                    let forecast =
                        level + (1.0 - 1.0 / theta) * b * (1.0 / alpha + (h as f64 - 1.0));
                    let error = values[i + h - 1] - forecast;
                    sse += error * error;
                    count += 1;
                }
            }

            // Update level
            level = alpha * values[i] + (1.0 - alpha) * level;
        }

        if count > 0 {
            sse / count as f64
        } else {
            f64::MAX
        }
    }

    /// Calculate autocorrelation function (ACF) for given lags.
    fn acf(series: &[f64], nlags: usize) -> Vec<f64> {
        let n = series.len();
        if n < 2 || nlags == 0 {
            return vec![1.0];
        }

        let mean = crate::simd::mean(series);
        let var = crate::simd::variance(series);

        if var < 1e-10 {
            return vec![1.0; nlags + 1];
        }

        let mut acf_values = Vec::with_capacity(nlags + 1);
        acf_values.push(1.0);

        for lag in 1..=nlags {
            if lag >= n {
                acf_values.push(0.0);
                continue;
            }
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += (series[i] - mean) * (series[i + lag] - mean);
            }
            acf_values.push(sum / (n as f64 * var));
        }

        acf_values
    }

    /// Perform seasonal test using ACF.
    fn seasonal_test(series: &[f64], period: usize) -> bool {
        if period < 4 || series.len() < 2 * period {
            return false;
        }

        let acf_vals = Self::acf(series, period);
        let r: Vec<f64> = acf_vals[1..].to_vec();
        let r_sq_sum = crate::simd::sum_of_squares(&r[..r.len() - 1]);
        let stat = ((1.0 + 2.0 * r_sq_sum) / series.len() as f64).sqrt();
        let r_m = r[r.len() - 1];

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_95 = normal.inverse_cdf(0.95);

        (r_m.abs() / stat) > z_95
    }

    /// Calculate seasonal component using classical decomposition.
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
                let mut s = 0.5 * series[i - half] + 0.5 * series[i + half];
                for &val in series.iter().take(i + half).skip(i - half + 1) {
                    s += val;
                }
                s / period as f64
            } else {
                let start = i - period / 2;
                let end = i + period / 2 + 1;
                series[start..end].iter().sum::<f64>() / period as f64
            };
            trend[i] = sum;
        }

        // Calculate detrended series
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

        // Average by season
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

        // Normalize
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

        let full_seasonal: Vec<f64> = (0..series.len())
            .map(|i| seasonal_indices[i % period])
            .collect();

        let last_cycle: Vec<f64> = full_seasonal[(series.len() - period)..].to_vec();

        (full_seasonal, last_cycle)
    }

    /// Determine effective decomposition type with fallback rules.
    fn determine_decomposition(&mut self, series: &[f64]) -> DecompositionType {
        if self.decomposition_type == DecompositionType::Additive {
            return DecompositionType::Additive;
        }

        if series.iter().any(|&y| y <= 0.0) {
            self.decomposition_fallback = true;
            return DecompositionType::Additive;
        }

        let (_, last_cycle) =
            self.calculate_seasonal_component(series, DecompositionType::Multiplicative);
        if !last_cycle.is_empty() && last_cycle.iter().any(|&s| s < 0.01) {
            self.decomposition_fallback = true;
            return DecompositionType::Additive;
        }

        DecompositionType::Multiplicative
    }

    /// Deseasonalize a series.
    fn deseasonalize(&self, series: &[f64], seasonals: &[f64]) -> Vec<f64> {
        if seasonals.is_empty() || self.seasonal_period == 0 {
            return series.to_vec();
        }

        match self.decomposition_type {
            DecompositionType::Additive => series
                .iter()
                .enumerate()
                .map(|(i, &y)| y - seasonals[i % self.seasonal_period])
                .collect(),
            DecompositionType::Multiplicative => series
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
                .collect(),
        }
    }

    /// Reseasonalize forecasts.
    fn reseasonalize(&self, forecasts: &[f64], start_idx: usize, seasonals: &[f64]) -> Vec<f64> {
        if seasonals.is_empty() || self.seasonal_period == 0 {
            return forecasts.to_vec();
        }

        match self.decomposition_type {
            DecompositionType::Additive => forecasts
                .iter()
                .enumerate()
                .map(|(i, &y)| y + seasonals[(start_idx + i) % self.seasonal_period])
                .collect(),
            DecompositionType::Multiplicative => forecasts
                .iter()
                .enumerate()
                .map(|(i, &y)| y * seasonals[(start_idx + i) % self.seasonal_period])
                .collect(),
        }
    }

    /// Optimize parameters using Nelder-Mead.
    fn optimize_parameters(values: &[f64], b: f64) -> (f64, f64) {
        let steps_ahead = 3; // Optimize for 3-step ahead forecasting (matches C++)

        let objective = |params: &[f64]| {
            let alpha = params[0];
            let theta = params[1];

            // Bounds check
            if alpha <= 0.01 || alpha >= 0.99 || !(1.0..=10.0).contains(&theta) {
                return f64::MAX;
            }

            Self::calculate_mse(values, alpha, theta, b, steps_ahead)
        };

        // Try multiple starting points to avoid local minima
        let starts = [[0.1, 2.0], [0.3, 2.0], [0.5, 2.0], [0.1, 3.0], [0.3, 1.5]];

        let mut best_params = [0.3, 2.0];
        let mut best_value = f64::MAX;

        let config = NelderMeadConfig {
            max_iter: 200,
            tolerance: 1e-6,
            ..Default::default()
        };

        for start in starts {
            let result = nelder_mead(
                objective,
                &start,
                Some(&[(0.01, 0.99), (1.0, 10.0)]),
                config.clone(),
            );

            if result.optimal_value < best_value {
                best_value = result.optimal_value;
                best_params = [
                    result.optimal_point[0].clamp(0.01, 0.99),
                    result.optimal_point[1].clamp(1.0, 10.0),
                ];
            }
        }

        (best_params[0], best_params[1])
    }

    /// Internal prediction method that handles both with and without exogenous cases.
    fn predict_internal(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<Forecast> {
        let level = self.level.ok_or(ForecastError::FitRequired)?;
        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;
        let theta = self.theta.ok_or(ForecastError::FitRequired)?;
        let b = self.b.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Calculate exogenous contribution if applicable
        let exog_contribution = if let Some(ols) = &self.exog_ols {
            let future = future_regressors.ok_or_else(|| {
                ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors. Future regressor values required."
                        .into(),
                )
            })?;

            // Validate future regressors have correct length
            for name in &ols.regressor_names {
                let values = future.get(name).ok_or_else(|| {
                    ForecastError::InvalidParameter(format!(
                        "Missing future values for regressor '{}'",
                        name
                    ))
                })?;
                if values.len() != horizon {
                    return Err(ForecastError::DimensionMismatch {
                        expected: horizon,
                        got: values.len(),
                    });
                }
            }

            Some(ols.predict(future)?)
        } else {
            if future_regressors.is_some_and(|r| !r.is_empty()) {
                return Err(ForecastError::InvalidParameter(
                    "Model was not fit with exogenous regressors".into(),
                ));
            }
            None
        };

        let mut predictions = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let mut forecast = level + (1.0 - 1.0 / theta) * b * (1.0 / alpha + (h as f64 - 1.0));
            if let Some(ref exog) = exog_contribution {
                forecast += exog[h - 1];
            }
            predictions.push(forecast);
        }

        // Reseasonalize if needed
        let predictions = if let Some(ref seasonal_forecast) = self.seasonal_forecast {
            self.reseasonalize(&predictions, 0, seasonal_forecast)
        } else {
            predictions
        };

        Ok(Forecast::from_values(predictions))
    }
}

impl Default for OptimizedTheta {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for OptimizedTheta {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let raw_values = series.primary_values();
        if raw_values.len() < 6 {
            return Err(ForecastError::InsufficientData {
                needed: 6,
                got: raw_values.len(),
            });
        }

        // Handle exogenous regressors
        let values: Vec<f64> = if series.has_regressors() {
            let regressors = series.all_regressors();
            let ols_result = ols_fit(raw_values, &regressors)?;
            let adjusted = ols_residuals(raw_values, &ols_result, &regressors)?;
            self.exog_ols = Some(ols_result);
            adjusted
        } else {
            self.exog_ols = None;
            raw_values.to_vec()
        };

        self.n = values.len();
        self.decomposition_fallback = false;

        // Perform seasonal test and decomposition if seasonal period is set
        let should_decompose = self.seasonal_period >= 4
            && values.len() >= 2 * self.seasonal_period
            && Self::seasonal_test(&values, self.seasonal_period);

        // Determine effective decomposition type with fallback rules
        let effective_decomposition = if should_decompose {
            self.determine_decomposition(&values)
        } else {
            self.decomposition_type
        };
        self.decomposition_type = effective_decomposition;

        // Calculate seasonal component if needed
        let (full_seasonal, seasonal_forecast) = if should_decompose {
            self.calculate_seasonal_component(&values, effective_decomposition)
        } else {
            (vec![], vec![])
        };

        // Deseasonalize
        let deseasonalized = self.deseasonalize(&values, &full_seasonal);

        // Calculate slope on deseasonalized data
        let b = Self::calculate_slope(&deseasonalized);
        self.b = Some(b);

        // Optimize alpha and theta on deseasonalized data
        let (alpha, theta) = Self::optimize_parameters(&deseasonalized, b);
        self.alpha = Some(alpha);
        self.theta = Some(theta);

        // Apply SES with optimized alpha on deseasonalized data
        let mut level = deseasonalized[0];
        self.initial_level = Some(level);

        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // First observation (reseasonalized)
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
            // Forecast using Theta formula on deseasonalized data
            let forecast = level + (1.0 - 1.0 / theta) * b * (1.0 / alpha);

            // Reseasonalize fitted value
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

            // Update level on deseasonalized data
            level = alpha * deseasonalized[i] + (1.0 - alpha) * level;
        }

        self.level = Some(level);
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
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        // If model was fit with exogenous regressors, require predict_with_exog
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog() and provide future regressor values.".into()
            ));
        }

        self.predict_internal(horizon, None)
    }

    fn supports_exog(&self) -> bool {
        true
    }

    fn has_exog(&self) -> bool {
        self.exog_ols.is_some()
    }

    fn exog_names(&self) -> Option<&[String]> {
        self.exog_ols
            .as_ref()
            .map(|ols| ols.regressor_names.as_slice())
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        self.predict_internal(horizon, Some(future_regressors))
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        let forecast = self.predict_internal(horizon, Some(future_regressors))?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let preds = forecast.primary();
        let alpha = self.alpha.unwrap_or(0.3);

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
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

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;
        let variance = self.residual_variance.unwrap_or(0.0);

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + confidence) / 2.0);
        let preds = forecast.primary();
        let alpha = self.alpha.unwrap_or(0.3);

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
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
        "OptimizedTheta"
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

    #[test]
    fn optimized_theta_basic() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        assert!(model.alpha().is_some());
        assert!(model.theta().is_some());

        let alpha = model.alpha().unwrap();
        let theta = model.theta().unwrap();

        assert!(alpha > 0.0 && alpha < 1.0);
        assert!((1.0..=10.0).contains(&theta));

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn optimized_theta_trending_data() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // Forecasts should continue the trend
        assert!(preds[0] > *values.last().unwrap() - 5.0);
        assert!(preds[4] > preds[0]); // Increasing trend
    }

    #[test]
    fn optimized_theta_confidence_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.2).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
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
    fn optimized_theta_fitted_and_residuals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 30);
    }

    #[test]
    fn optimized_theta_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn optimized_theta_requires_fit() {
        let model = OptimizedTheta::new();
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn optimized_theta_name() {
        let model = OptimizedTheta::new();
        assert_eq!(model.name(), "OptimizedTheta");
    }

    #[test]
    fn optimized_theta_slope_positive_for_trend() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        assert!(model.slope().unwrap() > 0.0);
    }

    #[test]
    fn optimized_theta_default() {
        let model = OptimizedTheta::default();
        assert!(model.alpha().is_none());
        assert!(model.theta().is_none());
        assert_eq!(
            model.decomposition_type(),
            DecompositionType::Multiplicative
        );
        assert!(!model.used_fallback());
    }

    #[test]
    fn optimized_theta_seasonal() {
        let timestamps = make_timestamps(100);
        // Create seasonal data with period 12
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let seasonal =
                    10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin() + 15.0;
                trend + seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::seasonal(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn optimized_theta_seasonal_with_additive_decomposition() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 50.0 + 0.5 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin();
                trend + seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model =
            OptimizedTheta::seasonal_with_decomposition(12, DecompositionType::Additive);
        model.fit(&ts).unwrap();

        assert_eq!(model.decomposition_type(), DecompositionType::Additive);
        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn optimized_theta_multiplicative_fallback_for_negative() {
        let timestamps = make_timestamps(100);
        // Series with negative values should trigger fallback
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = i as f64 * 0.5;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin();
                trend + seasonal - 30.0 // Some values will be negative
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::seasonal(12); // Default multiplicative
        model.fit(&ts).unwrap();

        // Should fallback to additive
        assert_eq!(model.decomposition_type(), DecompositionType::Additive);
        assert!(model.used_fallback());
    }

    #[test]
    fn optimized_theta_zero_horizon() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);

        let forecast_with_intervals = model.predict_with_intervals(0, 0.95).unwrap();
        assert_eq!(forecast_with_intervals.horizon(), 0);
    }

    #[test]
    fn optimized_theta_fitted_values_with_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        let fitted_forecast = model.fitted_values_with_intervals(0.95).unwrap();
        assert_eq!(fitted_forecast.horizon(), 50);
        assert!(fitted_forecast.has_lower());
        assert!(fitted_forecast.has_upper());

        let fitted = fitted_forecast.primary();
        let lower = fitted_forecast.lower_series(0).unwrap();
        let upper = fitted_forecast.upper_series(0).unwrap();

        for i in 0..50 {
            assert!(lower[i] <= fitted[i], "Lower should be <= fitted at {}", i);
            assert!(upper[i] >= fitted[i], "Upper should be >= fitted at {}", i);
        }
    }

    #[test]
    fn optimized_theta_acf_edge_cases() {
        // Empty series
        let acf_empty = OptimizedTheta::acf(&[], 5);
        assert_eq!(acf_empty, vec![1.0]);

        // Single element
        let acf_single = OptimizedTheta::acf(&[5.0], 5);
        assert_eq!(acf_single, vec![1.0]);

        // nlags = 0
        let acf_zero_lags = OptimizedTheta::acf(&[1.0, 2.0, 3.0], 0);
        assert_eq!(acf_zero_lags, vec![1.0]);

        // Constant series (variance = 0)
        let acf_constant = OptimizedTheta::acf(&[5.0, 5.0, 5.0, 5.0, 5.0], 3);
        assert_eq!(acf_constant, vec![1.0; 4]);

        // Normal series
        let series: Vec<f64> = (0..20).map(|i| i as f64 + (i as f64 * 0.5).sin()).collect();
        let acf_normal = OptimizedTheta::acf(&series, 5);
        assert_eq!(acf_normal.len(), 6);
        assert_eq!(acf_normal[0], 1.0); // Lag 0 is always 1

        // Lag larger than series length
        let acf_large_lag = OptimizedTheta::acf(&[1.0, 2.0, 3.0], 10);
        assert_eq!(acf_large_lag.len(), 11);
        // Lags >= n should be 0
        assert_eq!(acf_large_lag[3], 0.0);
        assert_eq!(acf_large_lag[4], 0.0);
    }

    #[test]
    fn optimized_theta_seasonal_test() {
        // Non-seasonal random-like data
        let non_seasonal: Vec<f64> = (0..50)
            .map(|i| 10.0 + i as f64 + ((i * 17) % 11) as f64)
            .collect();
        assert!(!OptimizedTheta::seasonal_test(&non_seasonal, 12));

        // Very short series
        assert!(!OptimizedTheta::seasonal_test(&[1.0, 2.0, 3.0], 12));

        // Period too small
        assert!(!OptimizedTheta::seasonal_test(&[1.0; 50], 3));

        // Series too short for period
        assert!(!OptimizedTheta::seasonal_test(&[1.0; 15], 12));
    }

    #[test]
    fn optimized_theta_calculate_slope() {
        // Positive slope
        let increasing: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let slope = OptimizedTheta::calculate_slope(&increasing);
        assert!(
            (slope - 1.0).abs() < 0.01,
            "Expected slope ~1.0, got {}",
            slope
        );

        // Negative slope
        let decreasing: Vec<f64> = (0..10).map(|i| 10.0 - i as f64).collect();
        let slope = OptimizedTheta::calculate_slope(&decreasing);
        assert!(
            (slope - (-1.0)).abs() < 0.01,
            "Expected slope ~-1.0, got {}",
            slope
        );

        // Constant (zero slope)
        let constant = vec![5.0; 10];
        let slope = OptimizedTheta::calculate_slope(&constant);
        assert!(slope.abs() < 0.01, "Expected slope ~0, got {}", slope);

        // Single element
        let single = OptimizedTheta::calculate_slope(&[5.0]);
        assert_eq!(single, 0.0);

        // Empty
        let empty = OptimizedTheta::calculate_slope(&[]);
        assert_eq!(empty, 0.0);
    }

    #[test]
    fn optimized_theta_calculate_mse() {
        let values: Vec<f64> = (0..20).map(|i| 10.0 + i as f64).collect();

        // Test with valid parameters
        let mse = OptimizedTheta::calculate_mse(&values, 0.3, 2.0, 1.0, 3);
        assert!(mse < f64::MAX);
        assert!(mse >= 0.0);

        // Test with insufficient data
        let mse_short = OptimizedTheta::calculate_mse(&[1.0, 2.0], 0.3, 2.0, 1.0, 3);
        assert_eq!(mse_short, f64::MAX);
    }

    #[test]
    fn optimized_theta_long_horizon() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict(100).unwrap();
        assert_eq!(forecast.horizon(), 100);

        // All predictions should be finite
        for &val in forecast.primary() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn optimized_theta_seasonal_short_series() {
        // Series just barely long enough (2 * period = 24)
        let timestamps = make_timestamps(25);
        let values: Vec<f64> = (0..25)
            .map(|i| {
                let trend = 50.0 + 0.5 * i as f64;
                let seasonal =
                    10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin() + 15.0;
                trend + seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::seasonal(12);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn optimized_theta_predict_with_intervals_multi_step() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(10, 0.95).unwrap();
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        // Intervals should widen with horizon
        let width_1 = upper[0] - lower[0];
        let width_10 = upper[9] - lower[9];
        assert!(
            width_10 > width_1,
            "Interval width should increase with horizon"
        );
    }

    #[test]
    fn optimized_theta_noisy_data() {
        let timestamps = make_timestamps(100);
        // Create noisy data
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 50.0 + 0.5 * i as f64;
                let noise = ((i * 17) % 10) as f64 - 5.0;
                trend + noise
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::new();
        model.fit(&ts).unwrap();

        // Model should still produce reasonable forecasts
        let forecast = model.predict(10).unwrap();
        assert_eq!(forecast.horizon(), 10);

        // All values should be finite
        for &val in forecast.primary() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn optimized_theta_small_seasonal_period() {
        // Test with small seasonal period (4 is minimum for seasonal)
        let timestamps = make_timestamps(40);
        let values: Vec<f64> = (0..40)
            .map(|i| {
                let trend = 50.0 + 0.5 * i as f64;
                let seasonal =
                    5.0 * (2.0 * std::f64::consts::PI * (i % 4) as f64 / 4.0).sin() + 10.0;
                trend + seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::seasonal(4);
        model.fit(&ts).unwrap();

        let forecast = model.predict(8).unwrap();
        assert_eq!(forecast.horizon(), 8);
    }

    #[test]
    fn optimized_theta_odd_seasonal_period() {
        // Test with odd seasonal period
        let timestamps = make_timestamps(70);
        let values: Vec<f64> = (0..70)
            .map(|i| {
                let trend = 50.0 + 0.3 * i as f64;
                let seasonal =
                    5.0 * (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin() + 10.0;
                trend + seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = OptimizedTheta::seasonal(7);
        model.fit(&ts).unwrap();

        let forecast = model.predict(7).unwrap();
        assert_eq!(forecast.horizon(), 7);
    }
}
