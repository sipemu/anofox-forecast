//! Dynamic Theta Models (DSTM and DOTM).
//!
//! Dynamic Theta models extend the standard Theta method by updating
//! the linear coefficients (An, Bn) at each time step rather than
//! keeping them fixed.
//!
//! Based on Fiorucci et al. (2016) "Models for optimising the theta method
//! and their relationship to state space models."

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

/// Type of seasonal decomposition for Dynamic Theta model.
///
/// Controls how seasonal components are extracted and reapplied.
/// Default is `Multiplicative` to match NIXTLA statsforecast behavior.
pub use super::model::DecompositionType;

/// Dynamic Standard Theta Model (DSTM).
///
/// DSTM updates the linear regression coefficients (An, Bn) at each time step,
/// allowing the model to adapt to changing trends.
///
/// For seasonal data, supports both additive and multiplicative decomposition
/// matching statsforecast's `DynamicTheta(season_length, decomposition_type)`.
///
/// State vector: [level, meany, An, Bn, mu]
/// - level: Exponentially smoothed level
/// - meany: Running mean of observations
/// - An: Intercept of dynamic linear fit
/// - Bn: Slope of dynamic linear fit
/// - mu: Current forecast
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DynamicTheta {
    /// Smoothing parameter.
    alpha: f64,
    /// Theta parameter (default: 2.0).
    theta: f64,
    /// Whether to optimize parameters.
    optimize: bool,
    /// Seasonal period (0 for non-seasonal).
    seasonal_period: usize,
    /// Type of seasonal decomposition (default: Multiplicative).
    decomposition_type: DecompositionType,
    /// Whether decomposition type was automatically changed due to fallback rules.
    decomposition_fallback: bool,
    /// Seasonal indices (for deseasonalizing).
    seasonals: Option<Vec<f64>>,
    /// Seasonal forecast pattern (last cycle, for reseasonalizing forecasts).
    seasonal_forecast: Option<Vec<f64>>,
    /// Fitted level.
    level: Option<f64>,
    /// Current An coefficient.
    an: Option<f64>,
    /// Current Bn coefficient.
    bn: Option<f64>,
    /// Running mean.
    meany: Option<f64>,
    /// Series length (for coefficient updates).
    series_len: Option<usize>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Number of observations.
    n: usize,
    /// OLS result for exogenous regressors (if any).
    exog_ols: Option<OLSResult>,
}

impl DynamicTheta {
    /// Create a new Dynamic Standard Theta Model with fixed alpha.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 0.99),
            theta: 2.0,
            optimize: false,
            seasonal_period: 0,
            decomposition_type: DecompositionType::Multiplicative,
            decomposition_fallback: false,
            seasonals: None,
            seasonal_forecast: None,
            level: None,
            an: None,
            bn: None,
            meany: None,
            series_len: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
            exog_ols: None,
        }
    }

    /// Create a Dynamic Theta Model with optimized parameters.
    pub fn optimized() -> Self {
        Self {
            alpha: 0.1,
            theta: 2.0,
            optimize: true,
            seasonal_period: 0,
            decomposition_type: DecompositionType::Multiplicative,
            decomposition_fallback: false,
            seasonals: None,
            seasonal_forecast: None,
            level: None,
            an: None,
            bn: None,
            meany: None,
            series_len: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
            exog_ols: None,
        }
    }

    /// Create a seasonal Dynamic Theta Model with multiplicative decomposition (default).
    ///
    /// Matches statsforecast's `DynamicTheta(season_length=period)`.
    pub fn seasonal(period: usize) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: DecompositionType::Multiplicative,
            ..Self::new(0.1)
        }
    }

    /// Create a seasonal Dynamic Theta Model with explicit decomposition type.
    ///
    /// Matches statsforecast's `DynamicTheta(season_length=period, decomposition_type=...)`.
    pub fn seasonal_with_decomposition(period: usize, decomposition: DecompositionType) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: decomposition,
            ..Self::new(0.1)
        }
    }

    /// Create a seasonal optimized Dynamic Theta Model.
    ///
    /// Matches statsforecast's `DynamicOptimizedTheta(season_length=period)`.
    pub fn seasonal_optimized(period: usize) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: DecompositionType::Multiplicative,
            ..Self::optimized()
        }
    }

    /// Create a seasonal optimized Dynamic Theta Model with explicit decomposition type.
    pub fn seasonal_optimized_with_decomposition(
        period: usize,
        decomposition: DecompositionType,
    ) -> Self {
        Self {
            seasonal_period: period,
            decomposition_type: decomposition,
            ..Self::optimized()
        }
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the theta parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Get the current slope (Bn coefficient).
    pub fn slope(&self) -> Option<f64> {
        self.bn
    }

    /// Get the decomposition type used.
    pub fn decomposition_type(&self) -> DecompositionType {
        self.decomposition_type
    }

    /// Check if the decomposition type was automatically changed due to fallback rules.
    pub fn used_fallback(&self) -> bool {
        self.decomposition_fallback
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

    /// Perform seasonal test using ACF (matching statsforecast).
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

        // Build full seasonal and last cycle
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

        // Non-positive values -> additive
        if series.iter().any(|&y| y <= 0.0) {
            self.decomposition_fallback = true;
            return DecompositionType::Additive;
        }

        // Check if seasonal factors are valid
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

    /// Initialize state from data.
    fn init_state(values: &[f64]) -> (f64, f64, f64, f64) {
        let n = values.len();
        if n == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let level = values[0];
        let meany = values[0];

        // Initial An and Bn from simple regression on first few points
        let init_n = n.min(10);
        let x_mean = (init_n - 1) as f64 / 2.0;
        let y_mean: f64 = values.iter().take(init_n).sum::<f64>() / init_n as f64;

        let mut ss_xx = 0.0;
        let mut ss_xy = 0.0;
        for (i, &y) in values.iter().take(init_n).enumerate() {
            let x = i as f64;
            ss_xx += (x - x_mean).powi(2);
            ss_xy += (x - x_mean) * (y - y_mean);
        }

        let bn = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
        let an = y_mean - bn * x_mean;

        (level, meany, an, bn)
    }

    /// Update state at time i with observation y.
    fn update_state(
        &self,
        y: f64,
        level: f64,
        meany: f64,
        _an: f64,
        bn: f64,
        i: usize,
    ) -> (f64, f64, f64, f64, f64) {
        let i_f = i as f64;

        // Update level using SES
        let new_level = self.alpha * y + (1.0 - self.alpha) * level;

        // Update running mean
        let new_meany = (i_f * meany + y) / (i_f + 1.0);

        // Update dynamic linear coefficients (matching C++ implementation)
        // Bn = ((i-1) * Bn + 6 * (y - meany) / (i+1)) / (i+2)
        // An = meany - Bn * (i+2) / 2
        let new_bn = ((i_f - 1.0) * bn + 6.0 * (y - meany) / (i_f + 1.0)) / (i_f + 2.0);
        let new_an = new_meany - new_bn * (i_f + 2.0) / 2.0;

        // Calculate current forecast (mu)
        // mu = level + (1 - 1/theta) * (An * (1-alpha)^i + Bn * (1 - (1-alpha)^(i+1)) / alpha)
        let beta = 1.0 - self.alpha;
        let mu = new_level
            + (1.0 - 1.0 / self.theta)
                * (new_an * beta.powi(i as i32 + 1)
                    + new_bn * (1.0 - beta.powi(i as i32 + 2)) / self.alpha);

        (new_level, new_meany, new_an, new_bn, mu)
    }

    /// Calculate MSE for given alpha and theta.
    fn calculate_mse(values: &[f64], alpha: f64, theta: f64) -> f64 {
        if values.len() < 3 {
            return f64::MAX;
        }

        let model = Self {
            alpha,
            theta,
            optimize: false,
            seasonal_period: 0,
            decomposition_type: DecompositionType::Multiplicative,
            decomposition_fallback: false,
            seasonals: None,
            seasonal_forecast: None,
            level: None,
            an: None,
            bn: None,
            meany: None,
            series_len: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
            exog_ols: None,
        };

        let (mut level, mut meany, mut an, mut bn) = Self::init_state(values);
        let mut sse = 0.0;
        let mut count = 0;

        for (i, &y) in values.iter().enumerate().skip(1) {
            // Forecast
            let beta = 1.0 - alpha;
            let forecast = level
                + (1.0 - 1.0 / theta)
                    * (an * beta.powi(i as i32) + bn * (1.0 - beta.powi(i as i32 + 1)) / alpha);

            let error = y - forecast;
            sse += error * error;
            count += 1;

            // Update state
            (level, meany, an, bn, _) = model.update_state(y, level, meany, an, bn, i);
        }

        if count > 0 {
            sse / count as f64
        } else {
            f64::MAX
        }
    }

    /// Optimize parameters.
    fn optimize_parameters(values: &[f64]) -> (f64, f64) {
        let objective = |params: &[f64]| {
            let alpha = params[0];
            let theta = params[1];

            if alpha <= 0.01 || alpha >= 0.99 || !(1.0..=10.0).contains(&theta) {
                return f64::MAX;
            }

            Self::calculate_mse(values, alpha, theta)
        };

        let starts = [[0.1, 2.0], [0.3, 2.0], [0.5, 2.0], [0.1, 3.0]];

        let mut best_params = [0.1, 2.0];
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
        let an = self.an.ok_or(ForecastError::FitRequired)?;
        let bn = self.bn.ok_or(ForecastError::FitRequired)?;

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

        // Use dynamic coefficients for forecasting
        let n = self.series_len.ok_or(ForecastError::FitRequired)?;
        let beta = 1.0 - self.alpha;

        let mut predictions = Vec::with_capacity(horizon);
        // Forecast from the end of the series (on deseasonalized scale)
        for h in 1..=horizon {
            let i = n + h - 1;
            let mut forecast = level
                + (1.0 - 1.0 / self.theta)
                    * (an * beta.powi(i as i32)
                        + bn * (1.0 - beta.powi(i as i32 + 1)) / self.alpha);
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

impl Default for DynamicTheta {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl Forecaster for DynamicTheta {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let raw_values = series.primary_values();
        if raw_values.len() < 4 {
            return Err(ForecastError::InsufficientData {
                needed: 4,
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

        // Perform seasonal test (matching statsforecast)
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

        // Store seasonal forecast for prediction
        self.seasonal_forecast = if seasonal_forecast.is_empty() {
            None
        } else {
            Some(seasonal_forecast)
        };

        // Deseasonalize using full seasonal component
        let deseasonalized = self.deseasonalize(&values, &full_seasonal);

        // Optimize if requested (on deseasonalized data)
        if self.optimize {
            let (alpha, theta) = Self::optimize_parameters(&deseasonalized);
            self.alpha = alpha;
            self.theta = theta;
        }

        // Initialize state on deseasonalized data
        let (mut level, mut meany, mut an, mut bn) = Self::init_state(&deseasonalized);

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

        // Process each observation
        for (i, &y) in deseasonalized.iter().enumerate().skip(1) {
            // Calculate forecast for current position (on deseasonalized scale)
            let beta = 1.0 - self.alpha;
            let forecast_deseas = level
                + (1.0 - 1.0 / self.theta)
                    * (an * beta.powi(i as i32)
                        + bn * (1.0 - beta.powi(i as i32 + 1)) / self.alpha);

            // Reseasonalize fitted value
            let forecast = if full_seasonal.is_empty() {
                forecast_deseas
            } else {
                match self.decomposition_type {
                    DecompositionType::Additive => forecast_deseas + full_seasonal[i],
                    DecompositionType::Multiplicative => forecast_deseas * full_seasonal[i],
                }
            };

            fitted.push(forecast);
            residuals.push(values[i] - forecast);

            // Update state on deseasonalized data
            let new_state = self.update_state(y, level, meany, an, bn, i);
            level = new_state.0;
            meany = new_state.1;
            an = new_state.2;
            bn = new_state.3;
        }

        self.level = Some(level);
        self.meany = Some(meany);
        self.an = Some(an);
        self.bn = Some(bn);
        self.series_len = Some(self.n);
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

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let factor = if h == 1 {
                1.0
            } else {
                let beta = 1.0 - self.alpha;
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

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let factor = if h == 1 {
                1.0
            } else {
                let beta = 1.0 - self.alpha;
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
        if self.optimize {
            "DynamicOptimizedTheta"
        } else {
            "DynamicTheta"
        }
    }
}

/// Dynamic Optimized Theta Model (DOTM).
///
/// DOTM combines dynamic coefficient updates with parameter optimization.
/// This was a component of the winning method in the M4 competition.
///
/// Alias for `DynamicTheta::optimized()`.
pub type DynamicOptimizedTheta = DynamicTheta;

impl DynamicOptimizedTheta {
    /// Create a new Dynamic Optimized Theta Model.
    pub fn dotm() -> Self {
        Self::optimized()
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
    fn dynamic_theta_basic() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn dynamic_theta_optimized() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::optimized();
        model.fit(&ts).unwrap();

        assert!(model.alpha() > 0.0 && model.alpha() < 1.0);
        assert!(model.theta() >= 1.0 && model.theta() <= 10.0);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn dynamic_theta_trending_data() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // Should continue trend
        assert!(preds[0] > values.last().unwrap() - 10.0);
    }

    #[test]
    fn dynamic_theta_confidence_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
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
    fn dynamic_theta_fitted_and_residuals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 30);
    }

    #[test]
    fn dynamic_theta_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn dynamic_theta_requires_fit() {
        let model = DynamicTheta::new(0.1);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn dynamic_theta_name() {
        let model = DynamicTheta::new(0.1);
        assert_eq!(model.name(), "DynamicTheta");

        let optimized = DynamicTheta::optimized();
        assert_eq!(optimized.name(), "DynamicOptimizedTheta");
    }

    #[test]
    fn dynamic_theta_dotm() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicOptimizedTheta::dotm();
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn dynamic_theta_slope() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        assert!(model.slope().is_some());
        assert!(model.slope().unwrap() > 0.0);
    }

    #[test]
    fn dynamic_theta_seasonal() {
        let n = 96;
        let period = 12;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                10.0 + 0.2 * i as f64
                    + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::seasonal(period);
        model.fit(&ts).unwrap();

        let forecast = model.predict(period).unwrap();
        assert_eq!(forecast.horizon(), period);
    }

    #[test]
    fn dynamic_theta_seasonal_optimized() {
        let n = 96;
        let period = 12;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                20.0 + 0.3 * i as f64
                    + 8.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::seasonal_optimized(period);
        model.fit(&ts).unwrap();

        let forecast = model.predict(period).unwrap();
        assert_eq!(forecast.horizon(), period);
    }

    #[test]
    fn dynamic_theta_additive_decomposition() {
        let n = 96;
        let period = 12;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                10.0 + 0.2 * i as f64
                    + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model =
            DynamicTheta::seasonal_with_decomposition(period, DecompositionType::Additive);
        model.fit(&ts).unwrap();

        let forecast = model.predict(6).unwrap();
        assert_eq!(forecast.horizon(), 6);
    }

    #[test]
    fn dynamic_theta_multiplicative_decomposition() {
        let n = 96;
        let period = 12;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                50.0 + 0.5 * i as f64
                    + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model =
            DynamicTheta::seasonal_with_decomposition(period, DecompositionType::Multiplicative);
        model.fit(&ts).unwrap();

        let forecast = model.predict(6).unwrap();
        assert_eq!(forecast.horizon(), 6);
    }

    #[test]
    fn dynamic_theta_multiplicative_fallback_negative_values() {
        let n = 50;
        let timestamps = make_timestamps(n);
        // Include negative values to trigger fallback to additive
        let values: Vec<f64> = (0..n).map(|i| -5.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model =
            DynamicTheta::seasonal_with_decomposition(12, DecompositionType::Multiplicative);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn dynamic_theta_seasonal_optimized_with_decomposition() {
        let n = 96;
        let period = 12;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                30.0 + 0.4 * i as f64
                    + 6.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::seasonal_optimized_with_decomposition(
            period,
            DecompositionType::Additive,
        );
        model.fit(&ts).unwrap();

        let forecast = model.predict(period).unwrap();
        assert_eq!(forecast.horizon(), period);
    }

    #[test]
    fn dynamic_theta_zero_horizon() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn dynamic_theta_constant_series() {
        let timestamps = make_timestamps(30);
        let values = vec![5.0; 30];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();
        for &pred in preds {
            assert!((pred - 5.0).abs() < 1.0);
        }
    }

    #[test]
    fn dynamic_theta_fitted_values_with_intervals() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::new(0.1);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values_with_intervals(0.95).unwrap();
        assert!(fitted.has_lower());
        assert!(fitted.has_upper());
    }

    #[test]
    fn dynamic_theta_odd_period_seasonal() {
        let n = 77;
        let period = 7;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                20.0 + 0.3 * i as f64
                    + 4.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = DynamicTheta::seasonal(period);
        model.fit(&ts).unwrap();

        let forecast = model.predict(period).unwrap();
        assert_eq!(forecast.horizon(), period);
    }

    #[test]
    fn dynamic_theta_no_seasonality_detected() {
        // Short period or non-seasonal data
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Period too short for seasonal test
        let mut model = DynamicTheta::seasonal(3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }
}
