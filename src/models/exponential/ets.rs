//! ETS (Error-Trend-Seasonal) state-space forecasting model.
//!
//! ETS provides a unified framework for exponential smoothing methods,
//! with 30 possible model combinations based on error, trend, and seasonal components.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Error component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorType {
    /// Additive errors
    #[default]
    Additive,
    /// Multiplicative errors
    Multiplicative,
}

/// Trend component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendType {
    /// No trend
    #[default]
    None,
    /// Additive trend
    Additive,
    /// Additive damped trend
    AdditiveDamped,
}

/// Seasonal component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeasonalType {
    /// No seasonality
    #[default]
    None,
    /// Additive seasonality
    Additive,
    /// Multiplicative seasonality
    Multiplicative,
}

/// ETS model specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ETSSpec {
    pub error: ErrorType,
    pub trend: TrendType,
    pub seasonal: SeasonalType,
}

impl ETSSpec {
    /// Create a new ETS specification.
    pub fn new(error: ErrorType, trend: TrendType, seasonal: SeasonalType) -> Self {
        Self {
            error,
            trend,
            seasonal,
        }
    }

    /// ETS(A,N,N) - Simple exponential smoothing with additive errors.
    pub fn ann() -> Self {
        Self::new(ErrorType::Additive, TrendType::None, SeasonalType::None)
    }

    /// ETS(A,A,N) - Holt's linear method with additive errors.
    pub fn aan() -> Self {
        Self::new(ErrorType::Additive, TrendType::Additive, SeasonalType::None)
    }

    /// ETS(A,Ad,N) - Damped trend with additive errors.
    pub fn aadn() -> Self {
        Self::new(
            ErrorType::Additive,
            TrendType::AdditiveDamped,
            SeasonalType::None,
        )
    }

    /// ETS(A,A,A) - Holt-Winters additive.
    pub fn aaa() -> Self {
        Self::new(
            ErrorType::Additive,
            TrendType::Additive,
            SeasonalType::Additive,
        )
    }

    /// ETS(A,A,M) - Holt-Winters multiplicative seasonality.
    pub fn aam() -> Self {
        Self::new(
            ErrorType::Additive,
            TrendType::Additive,
            SeasonalType::Multiplicative,
        )
    }

    /// ETS(M,N,N) - Simple exponential smoothing with multiplicative errors.
    pub fn mnn() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::None,
            SeasonalType::None,
        )
    }

    /// ETS(M,A,M) - Multiplicative Holt-Winters.
    pub fn mam() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::Additive,
            SeasonalType::Multiplicative,
        )
    }

    /// Get a short name for this specification.
    pub fn short_name(&self) -> String {
        let e = match self.error {
            ErrorType::Additive => "A",
            ErrorType::Multiplicative => "M",
        };
        let t = match self.trend {
            TrendType::None => "N",
            TrendType::Additive => "A",
            TrendType::AdditiveDamped => "Ad",
        };
        let s = match self.seasonal {
            SeasonalType::None => "N",
            SeasonalType::Additive => "A",
            SeasonalType::Multiplicative => "M",
        };
        format!("ETS({},{},{})", e, t, s)
    }

    /// Check if this model has a trend component.
    pub fn has_trend(&self) -> bool {
        !matches!(self.trend, TrendType::None)
    }

    /// Check if this model has a seasonal component.
    pub fn has_seasonal(&self) -> bool {
        !matches!(self.seasonal, SeasonalType::None)
    }

    /// Check if this model has damping.
    pub fn is_damped(&self) -> bool {
        matches!(self.trend, TrendType::AdditiveDamped)
    }
}

/// ETS state-space model.
#[derive(Debug, Clone)]
pub struct ETS {
    /// Model specification.
    spec: ETSSpec,
    /// Seasonal period.
    seasonal_period: usize,
    /// Level smoothing parameter.
    alpha: Option<f64>,
    /// Trend smoothing parameter.
    beta: Option<f64>,
    /// Seasonal smoothing parameter.
    gamma: Option<f64>,
    /// Damping parameter.
    phi: Option<f64>,
    /// Whether to optimize parameters.
    optimize: bool,
    /// Current level state.
    level: Option<f64>,
    /// Current trend state.
    trend: Option<f64>,
    /// Seasonal states.
    seasonals: Option<Vec<f64>>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance.
    residual_variance: Option<f64>,
    /// Log-likelihood.
    log_likelihood: Option<f64>,
    /// AIC.
    aic: Option<f64>,
    /// AICc.
    aicc: Option<f64>,
    /// BIC.
    bic: Option<f64>,
    /// Series length.
    n: usize,
}

impl ETS {
    /// Create a new ETS model with the given specification.
    pub fn new(spec: ETSSpec, seasonal_period: usize) -> Self {
        Self {
            spec,
            seasonal_period,
            alpha: None,
            beta: None,
            gamma: None,
            phi: None,
            optimize: true,
            level: None,
            trend: None,
            seasonals: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            log_likelihood: None,
            aic: None,
            aicc: None,
            bic: None,
            n: 0,
        }
    }

    /// Create an ETS model with fixed parameters.
    pub fn with_params(
        spec: ETSSpec,
        seasonal_period: usize,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> Self {
        Self {
            spec,
            seasonal_period,
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            beta: beta.map(|b| b.clamp(0.0001, 0.9999)),
            gamma: gamma.map(|g| g.clamp(0.0001, 0.9999)),
            phi: phi.map(|p| p.clamp(0.8, 0.98)),
            optimize: false,
            level: None,
            trend: None,
            seasonals: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            log_likelihood: None,
            aic: None,
            aicc: None,
            bic: None,
            n: 0,
        }
    }

    /// Get the model specification.
    pub fn spec(&self) -> ETSSpec {
        self.spec
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
    pub fn phi(&self) -> Option<f64> {
        self.phi
    }

    /// Get information criteria.
    pub fn aic(&self) -> Option<f64> {
        self.aic
    }
    pub fn aicc(&self) -> Option<f64> {
        self.aicc
    }
    pub fn bic(&self) -> Option<f64> {
        self.bic
    }
    pub fn log_likelihood(&self) -> Option<f64> {
        self.log_likelihood
    }

    /// Initialize state components.
    fn initialize_state(&self, values: &[f64]) -> (f64, f64, Vec<f64>) {
        let period = self.seasonal_period;

        // Initial level
        let level = if self.spec.has_seasonal() && values.len() >= period {
            values.iter().take(period).sum::<f64>() / period as f64
        } else {
            values[0]
        };

        // Initial trend
        let trend = if self.spec.has_trend() && values.len() >= 2 {
            if self.spec.has_seasonal() && values.len() >= 2 * period {
                let sum: f64 = (0..period)
                    .map(|i| (values[period + i] - values[i]) / period as f64)
                    .sum();
                sum / period as f64
            } else {
                values[1] - values[0]
            }
        } else {
            0.0
        };

        // Initial seasonal indices
        let seasonals = if self.spec.has_seasonal() && values.len() >= period {
            match self.spec.seasonal {
                SeasonalType::Additive => values.iter().take(period).map(|y| y - level).collect(),
                SeasonalType::Multiplicative => values
                    .iter()
                    .take(period)
                    .map(|y| if level.abs() > 1e-10 { y / level } else { 1.0 })
                    .collect(),
                SeasonalType::None => vec![],
            }
        } else {
            vec![]
        };

        (level, trend, seasonals)
    }

    /// Calculate negative log-likelihood for given parameters.
    fn calculate_likelihood(
        &self,
        values: &[f64],
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> f64 {
        let n = values.len();
        let period = self.seasonal_period;
        let start_idx = if self.spec.has_seasonal() { period } else { 1 };

        if n <= start_idx {
            return f64::MAX;
        }

        let (mut level, mut trend, mut seasonals) = self.initialize_state(values);
        let phi = phi.unwrap_or(1.0);
        let beta = beta.unwrap_or(0.0);
        let gamma = gamma.unwrap_or(0.0);

        let mut sum_sq_errors = 0.0;
        let mut sum_log_y = 0.0;
        let mut count = 0;

        for (t, &y) in values.iter().enumerate().skip(start_idx) {
            let season_idx = if self.spec.has_seasonal() {
                t % period
            } else {
                0
            };
            let s = if self.spec.has_seasonal() {
                seasonals[season_idx]
            } else {
                1.0
            };

            // One-step forecast
            let forecast = match (self.spec.trend, self.spec.seasonal) {
                (TrendType::None, SeasonalType::None) => level,
                (TrendType::None, SeasonalType::Additive) => level + s,
                (TrendType::None, SeasonalType::Multiplicative) => level * s,
                (TrendType::Additive, SeasonalType::None) => level + trend,
                (TrendType::Additive, SeasonalType::Additive) => level + trend + s,
                (TrendType::Additive, SeasonalType::Multiplicative) => (level + trend) * s,
                (TrendType::AdditiveDamped, SeasonalType::None) => level + phi * trend,
                (TrendType::AdditiveDamped, SeasonalType::Additive) => level + phi * trend + s,
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    (level + phi * trend) * s
                }
            };

            let error = y - forecast;

            // For multiplicative errors, we'd use relative error
            let scaled_error =
                if self.spec.error == ErrorType::Multiplicative && forecast.abs() > 1e-10 {
                    error / forecast
                } else {
                    error
                };

            sum_sq_errors += scaled_error * scaled_error;

            if self.spec.error == ErrorType::Multiplicative {
                sum_log_y += y.abs().ln();
            }
            count += 1;

            // Update state
            let level_prev = level;

            match (self.spec.trend, self.spec.seasonal) {
                (TrendType::None, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * level;
                }
                (TrendType::None, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * level;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::None, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * level;
                    seasonals[season_idx] = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                }
                (TrendType::Additive, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                }
                (TrendType::Additive, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonals[season_idx] = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                }
                (TrendType::AdditiveDamped, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                }
                (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                    seasonals[season_idx] = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                }
            }
        }

        if count == 0 {
            return f64::MAX;
        }

        // Negative log-likelihood (simplified)
        let sigma2 = sum_sq_errors / count as f64;
        let ll = if self.spec.error == ErrorType::Multiplicative {
            -0.5 * count as f64 * (1.0 + sigma2.ln() + (2.0 * std::f64::consts::PI).ln())
                - sum_log_y
        } else {
            -0.5 * count as f64 * (1.0 + sigma2.ln() + (2.0 * std::f64::consts::PI).ln())
        };

        -ll // Return negative log-likelihood for minimization
    }

    /// Optimize parameters.
    fn optimize_params(&self, values: &[f64]) -> (f64, Option<f64>, Option<f64>, Option<f64>) {
        let config = NelderMeadConfig {
            max_iter: 1000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let has_trend = self.spec.has_trend();
        let has_seasonal = self.spec.has_seasonal();
        let is_damped = self.spec.is_damped();

        // Determine number of parameters
        let n_params = 1
            + (if has_trend { 1 } else { 0 })
            + (if has_seasonal { 1 } else { 0 })
            + (if is_damped { 1 } else { 0 });

        match n_params {
            1 => {
                // Just alpha
                let result = nelder_mead(
                    |p| self.calculate_likelihood(values, p[0], None, None, None),
                    &[0.3],
                    Some(&[(0.0001, 0.9999)]),
                    config,
                );
                (
                    result.optimal_point[0].clamp(0.0001, 0.9999),
                    None,
                    None,
                    None,
                )
            }
            2 if has_trend && !is_damped => {
                // alpha, beta
                let result = nelder_mead(
                    |p| self.calculate_likelihood(values, p[0], Some(p[1]), None, None),
                    &[0.3, 0.1],
                    Some(&[(0.0001, 0.9999), (0.0001, 0.9999)]),
                    config,
                );
                (
                    result.optimal_point[0].clamp(0.0001, 0.9999),
                    Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                    None,
                    None,
                )
            }
            2 if has_seasonal => {
                // alpha, gamma
                let result = nelder_mead(
                    |p| self.calculate_likelihood(values, p[0], None, Some(p[1]), None),
                    &[0.3, 0.1],
                    Some(&[(0.0001, 0.9999), (0.0001, 0.9999)]),
                    config,
                );
                (
                    result.optimal_point[0].clamp(0.0001, 0.9999),
                    None,
                    Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                    None,
                )
            }
            3 if has_trend && !is_damped && has_seasonal => {
                // alpha, beta, gamma
                let result = nelder_mead(
                    |p| self.calculate_likelihood(values, p[0], Some(p[1]), Some(p[2]), None),
                    &[0.3, 0.1, 0.1],
                    Some(&[(0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.9999)]),
                    config,
                );
                (
                    result.optimal_point[0].clamp(0.0001, 0.9999),
                    Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                    Some(result.optimal_point[2].clamp(0.0001, 0.9999)),
                    None,
                )
            }
            3 if is_damped && !has_seasonal => {
                // alpha, beta, phi
                let result = nelder_mead(
                    |p| self.calculate_likelihood(values, p[0], Some(p[1]), None, Some(p[2])),
                    &[0.3, 0.1, 0.98],
                    Some(&[(0.0001, 0.9999), (0.0001, 0.9999), (0.8, 0.98)]),
                    config,
                );
                (
                    result.optimal_point[0].clamp(0.0001, 0.9999),
                    Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                    None,
                    Some(result.optimal_point[2].clamp(0.8, 0.98)),
                )
            }
            4 => {
                // alpha, beta, gamma, phi
                let result = nelder_mead(
                    |p| self.calculate_likelihood(values, p[0], Some(p[1]), Some(p[2]), Some(p[3])),
                    &[0.3, 0.1, 0.1, 0.98],
                    Some(&[
                        (0.0001, 0.9999),
                        (0.0001, 0.9999),
                        (0.0001, 0.9999),
                        (0.8, 0.98),
                    ]),
                    config,
                );
                (
                    result.optimal_point[0].clamp(0.0001, 0.9999),
                    Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                    Some(result.optimal_point[2].clamp(0.0001, 0.9999)),
                    Some(result.optimal_point[3].clamp(0.8, 0.98)),
                )
            }
            _ => (0.3, None, None, None),
        }
    }

    /// Calculate damped sum for forecasting.
    fn damped_sum(phi: f64, h: usize) -> f64 {
        if (phi - 1.0).abs() < 1e-10 {
            h as f64
        } else {
            phi * (1.0 - phi.powi(h as i32)) / (1.0 - phi)
        }
    }

    /// Count number of parameters.
    fn num_params(&self) -> usize {
        let mut count = 1; // alpha
        if self.spec.has_trend() {
            count += 1;
        } // beta
        if self.spec.has_seasonal() {
            count += 1;
        } // gamma
        if self.spec.is_damped() {
            count += 1;
        } // phi
          // Add initial states
        count += 1; // initial level
        if self.spec.has_trend() {
            count += 1;
        } // initial trend
        if self.spec.has_seasonal() {
            count += self.seasonal_period;
        } // initial seasonals
        count
    }
}

impl Default for ETS {
    fn default() -> Self {
        Self::new(ETSSpec::ann(), 1)
    }
}

impl Forecaster for ETS {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        let min_len = if self.spec.has_seasonal() {
            2 * self.seasonal_period
        } else {
            2
        };

        if values.len() < min_len {
            return Err(ForecastError::InsufficientData {
                needed: min_len,
                got: values.len(),
            });
        }

        self.n = values.len();

        // Optimize parameters if needed
        if self.optimize {
            let (alpha, beta, gamma, phi) = self.optimize_params(values);
            self.alpha = Some(alpha);
            self.beta = beta;
            self.gamma = gamma;
            self.phi = phi;
        }

        let alpha = self.alpha.unwrap_or(0.3);
        let beta = self.beta.unwrap_or(0.1);
        let gamma = self.gamma.unwrap_or(0.1);
        let phi = self.phi.unwrap_or(1.0);
        let period = self.seasonal_period;

        // Initialize state
        let (mut level, mut trend, mut seasonals) = self.initialize_state(values);
        let start_idx = if self.spec.has_seasonal() { period } else { 1 };

        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // Fill initial values
        for &val in values.iter().take(start_idx) {
            fitted.push(val);
            residuals.push(0.0);
        }

        // Process remaining data
        for (t, &y) in values.iter().enumerate().skip(start_idx) {
            let season_idx = if self.spec.has_seasonal() {
                t % period
            } else {
                0
            };
            let s = if self.spec.has_seasonal() {
                seasonals[season_idx]
            } else {
                1.0
            };

            // One-step forecast
            let forecast = match (self.spec.trend, self.spec.seasonal) {
                (TrendType::None, SeasonalType::None) => level,
                (TrendType::None, SeasonalType::Additive) => level + s,
                (TrendType::None, SeasonalType::Multiplicative) => level * s,
                (TrendType::Additive, SeasonalType::None) => level + trend,
                (TrendType::Additive, SeasonalType::Additive) => level + trend + s,
                (TrendType::Additive, SeasonalType::Multiplicative) => (level + trend) * s,
                (TrendType::AdditiveDamped, SeasonalType::None) => level + phi * trend,
                (TrendType::AdditiveDamped, SeasonalType::Additive) => level + phi * trend + s,
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    (level + phi * trend) * s
                }
            };

            fitted.push(forecast);
            residuals.push(y - forecast);

            // Update state
            let level_prev = level;

            match (self.spec.trend, self.spec.seasonal) {
                (TrendType::None, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * level;
                }
                (TrendType::None, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * level;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::None, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * level;
                    seasonals[season_idx] = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                }
                (TrendType::Additive, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                }
                (TrendType::Additive, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonals[season_idx] = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                }
                (TrendType::AdditiveDamped, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                }
                (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                    seasonals[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                    seasonals[season_idx] = if level.abs() > 1e-10 {
                        gamma * (y / level) + (1.0 - gamma) * s
                    } else {
                        s
                    };
                }
            }
        }

        self.level = Some(level);
        self.trend = Some(trend);
        if self.spec.has_seasonal() {
            self.seasonals = Some(seasonals);
        }
        self.fitted = Some(fitted);

        // Calculate residual variance and information criteria
        let valid_residuals: Vec<f64> = residuals[start_idx..].to_vec();
        if !valid_residuals.is_empty() {
            let variance =
                valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);

            // Calculate information criteria
            let n = valid_residuals.len() as f64;
            let k = self.num_params() as f64;
            let ll = -0.5 * n * (1.0 + variance.ln() + (2.0 * std::f64::consts::PI).ln());

            self.log_likelihood = Some(ll);
            self.aic = Some(-2.0 * ll + 2.0 * k);
            self.aicc = Some(-2.0 * ll + 2.0 * k * n / (n - k - 1.0).max(1.0));
            self.bic = Some(-2.0 * ll + k * n.ln());
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let level = self.level.ok_or(ForecastError::FitRequired)?;
        let trend = self.trend.unwrap_or(0.0);
        let phi = self.phi.unwrap_or(1.0);
        let period = self.seasonal_period;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let seasonals_ref = if self.spec.has_seasonal() {
            Some(self.seasonals.as_ref().ok_or(ForecastError::FitRequired)?)
        } else {
            None
        };

        let predictions: Vec<f64> = (1..=horizon)
            .map(|h| {
                let s = if let Some(seasonals) = seasonals_ref {
                    seasonals[(self.n + h - 1) % period]
                } else {
                    1.0
                };

                let trend_component = if self.spec.has_trend() {
                    if self.spec.is_damped() {
                        Self::damped_sum(phi, h) * trend
                    } else {
                        h as f64 * trend
                    }
                } else {
                    0.0
                };

                match self.spec.seasonal {
                    SeasonalType::None => level + trend_component,
                    SeasonalType::Additive => level + trend_component + s,
                    SeasonalType::Multiplicative => (level + trend_component) * s,
                }
            })
            .collect();

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        let level = self.level.ok_or(ForecastError::FitRequired)?;
        let trend = self.trend.unwrap_or(0.0);
        let phi = self.phi.unwrap_or(1.0);
        let variance = self.residual_variance.unwrap_or(0.0);
        let period = self.seasonal_period;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + confidence) / 2.0);

        let seasonals_ref = if self.spec.has_seasonal() {
            Some(self.seasonals.as_ref().ok_or(ForecastError::FitRequired)?)
        } else {
            None
        };

        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let s = if let Some(seasonals) = seasonals_ref {
                seasonals[(self.n + h - 1) % period]
            } else {
                1.0
            };

            let trend_component = if self.spec.has_trend() {
                if self.spec.is_damped() {
                    Self::damped_sum(phi, h) * trend
                } else {
                    h as f64 * trend
                }
            } else {
                0.0
            };

            let pred = match self.spec.seasonal {
                SeasonalType::None => level + trend_component,
                SeasonalType::Additive => level + trend_component + s,
                SeasonalType::Multiplicative => (level + trend_component) * s,
            };
            predictions.push(pred);

            // Simplified variance calculation
            let k = if self.spec.has_seasonal() {
                ((h - 1) / period) + 1
            } else {
                h
            };
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
        "ETS"
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
    fn ets_ann_simple() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);

        // ANN produces flat forecasts
        let preds = forecast.primary();
        assert_relative_eq!(preds[0], preds[4], epsilon = 1e-10);
    }

    #[test]
    fn ets_aan_with_trend() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::aan(), 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // AAN should show increasing forecasts
        assert!(preds[4] > preds[0]);
    }

    #[test]
    fn ets_aaa_seasonal() {
        let timestamps = make_timestamps(32);
        let values: Vec<f64> = (0..32)
            .map(|i| 10.0 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::aaa(), 8);
        model.fit(&ts).unwrap();

        let forecast = model.predict(8).unwrap();
        assert_eq!(forecast.horizon(), 8);
    }

    #[test]
    fn ets_damped_trend() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + 2.0 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model_undamped = ETS::new(ETSSpec::aan(), 1);
        let mut model_damped = ETS::new(ETSSpec::aadn(), 1);

        model_undamped.fit(&ts).unwrap();
        model_damped.fit(&ts).unwrap();

        let f_undamped = model_undamped.predict(10).unwrap();
        let f_damped = model_damped.predict(10).unwrap();

        // Damped should be more conservative
        assert!(f_undamped.primary()[9] > f_damped.primary()[9]);
    }

    #[test]
    fn ets_with_fixed_params() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::with_params(ETSSpec::aan(), 1, 0.5, Some(0.1), None, None);
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.alpha().unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(model.beta().unwrap(), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn ets_confidence_intervals() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::ann(), 1);
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
    fn ets_information_criteria() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64 * 0.5).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();

        assert!(model.aic().is_some());
        assert!(model.aicc().is_some());
        assert!(model.bic().is_some());
        assert!(model.log_likelihood().is_some());
    }

    #[test]
    fn ets_spec_short_names() {
        assert_eq!(ETSSpec::ann().short_name(), "ETS(A,N,N)");
        assert_eq!(ETSSpec::aan().short_name(), "ETS(A,A,N)");
        assert_eq!(ETSSpec::aadn().short_name(), "ETS(A,Ad,N)");
        assert_eq!(ETSSpec::aaa().short_name(), "ETS(A,A,A)");
        assert_eq!(ETSSpec::aam().short_name(), "ETS(A,A,M)");
        assert_eq!(ETSSpec::mnn().short_name(), "ETS(M,N,N)");
    }

    #[test]
    fn ets_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::aaa(), 8);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn ets_requires_fit() {
        let model = ETS::new(ETSSpec::ann(), 1);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn ets_zero_horizon() {
        let timestamps = make_timestamps(10);
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn ets_multiplicative_seasonal() {
        let timestamps = make_timestamps(24);
        let values: Vec<f64> = (0..24)
            .map(|i| {
                let base = 100.0;
                let seasonal = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * i as f64 / 6.0).sin();
                base * seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::aam(), 6);
        model.fit(&ts).unwrap();

        let forecast = model.predict(6).unwrap();
        assert_eq!(forecast.horizon(), 6);
    }
}
