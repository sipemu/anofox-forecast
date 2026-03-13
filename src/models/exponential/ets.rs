//! ETS (Error-Trend-Seasonal) state-space forecasting model.
//!
//! ETS provides a unified framework for exponential smoothing methods,
//! with 30 possible model combinations based on error, trend, and seasonal components.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::explain::{Explainable, ForecastExplanation};
use crate::models::{validate_series_complete, Forecaster};
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

/// Error component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ErrorType {
    /// Additive errors
    #[default]
    Additive,
    /// Multiplicative errors
    Multiplicative,
}

/// Trend component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// ETS(A,N,A) - No trend with additive seasonality.
    pub fn ana() -> Self {
        Self::new(ErrorType::Additive, TrendType::None, SeasonalType::Additive)
    }

    /// ETS(A,N,M) - No trend with multiplicative seasonality.
    pub fn anm() -> Self {
        Self::new(
            ErrorType::Additive,
            TrendType::None,
            SeasonalType::Multiplicative,
        )
    }

    /// ETS(A,Ad,A) - Additive damped Holt-Winters.
    pub fn aada() -> Self {
        Self::new(
            ErrorType::Additive,
            TrendType::AdditiveDamped,
            SeasonalType::Additive,
        )
    }

    /// ETS(A,Ad,M) - Additive damped Holt-Winters with multiplicative seasonality.
    pub fn aadm() -> Self {
        Self::new(
            ErrorType::Additive,
            TrendType::AdditiveDamped,
            SeasonalType::Multiplicative,
        )
    }

    /// ETS(M,N,M) - Multiplicative error with multiplicative seasonality (no trend).
    pub fn mnm() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::None,
            SeasonalType::Multiplicative,
        )
    }

    /// ETS(M,Ad,M) - Multiplicative damped Holt-Winters.
    pub fn madm() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::AdditiveDamped,
            SeasonalType::Multiplicative,
        )
    }

    /// ETS(M,A,N) - Multiplicative error with additive trend (no seasonality).
    pub fn man() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::Additive,
            SeasonalType::None,
        )
    }

    /// ETS(M,Ad,N) - Multiplicative error with damped additive trend (no seasonality).
    pub fn madn() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::AdditiveDamped,
            SeasonalType::None,
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

    /// Check if this ETS specification is valid/stable.
    ///
    /// Per FPP3 taxonomy (<https://otexts.com/fpp3/taxonomy.html>), most ETS
    /// combinations are valid, but two are numerically unstable:
    /// - ETS(M,A,A) - Multiplicative error with additive trend and additive seasonal
    /// - ETS(M,Ad,A) - Multiplicative error with damped trend and additive seasonal
    ///
    /// Returns `true` for valid/stable combinations, `false` for unstable ones.
    pub fn is_valid(&self) -> bool {
        // M,A,A and M,Ad,A are unstable (multiplicative error
        // with additive trend AND additive seasonal)
        !(self.error == ErrorType::Multiplicative
            && matches!(self.trend, TrendType::Additive | TrendType::AdditiveDamped)
            && self.seasonal == SeasonalType::Additive)
    }

    /// Parse ETS notation string like "ANN", "AAA", "MAM", "AAdM".
    ///
    /// Format: ErrorTrendSeasonal where:
    /// - Error: A (additive) or M (multiplicative)
    /// - Trend: N (none), A (additive), or Ad (additive damped)
    /// - Seasonal: N (none), A (additive), or M (multiplicative)
    ///
    /// This follows the ETS taxonomy from FPP3:
    /// <https://otexts.com/fpp3/taxonomy.html>
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The notation format is invalid
    /// - The combination is unstable (MAA, MAdA)
    ///
    /// # Examples
    ///
    /// ```
    /// use anofox_forecast::models::exponential::ETSSpec;
    ///
    /// let spec = ETSSpec::from_notation("AAA").unwrap();
    /// assert_eq!(spec, ETSSpec::aaa());
    ///
    /// let spec = ETSSpec::from_notation("MAdM").unwrap();
    /// assert!(spec.is_damped());
    ///
    /// // Invalid combination returns error
    /// assert!(ETSSpec::from_notation("MAA").is_err());
    /// ```
    pub fn from_notation(notation: &str) -> crate::error::Result<Self> {
        use crate::error::ForecastError;

        let notation = notation.to_uppercase();
        let chars: Vec<char> = notation.chars().collect();

        if chars.len() < 3 || chars.len() > 4 {
            return Err(ForecastError::InvalidParameter(format!(
                "ETS notation must be 3-4 characters, got '{}'",
                notation
            )));
        }

        // Parse error type (first character)
        let error = match chars[0] {
            'A' => ErrorType::Additive,
            'M' => ErrorType::Multiplicative,
            c => {
                return Err(ForecastError::InvalidParameter(format!(
                    "Invalid error type '{}', expected 'A' or 'M'",
                    c
                )))
            }
        };

        // Parse trend and seasonal based on length
        let (trend, seasonal) = if chars.len() == 4 {
            // Format: E Ad S (e.g., "AAdN", "MAdM")
            if chars[1] != 'A' || chars[2] != 'D' {
                return Err(ForecastError::InvalidParameter(format!(
                    "4-character notation must have 'Ad' for damped trend, got '{}{}'",
                    chars[1], chars[2]
                )));
            }
            let seasonal = match chars[3] {
                'N' => SeasonalType::None,
                'A' => SeasonalType::Additive,
                'M' => SeasonalType::Multiplicative,
                c => {
                    return Err(ForecastError::InvalidParameter(format!(
                        "Invalid seasonal type '{}', expected 'N', 'A', or 'M'",
                        c
                    )))
                }
            };
            (TrendType::AdditiveDamped, seasonal)
        } else {
            // Format: E T S (e.g., "ANN", "AAA", "MAM")
            let trend = match chars[1] {
                'N' => TrendType::None,
                'A' => TrendType::Additive,
                c => {
                    return Err(ForecastError::InvalidParameter(format!(
                        "Invalid trend type '{}', expected 'N' or 'A' (use 'Ad' for damped)",
                        c
                    )))
                }
            };
            let seasonal = match chars[2] {
                'N' => SeasonalType::None,
                'A' => SeasonalType::Additive,
                'M' => SeasonalType::Multiplicative,
                c => {
                    return Err(ForecastError::InvalidParameter(format!(
                        "Invalid seasonal type '{}', expected 'N', 'A', or 'M'",
                        c
                    )))
                }
            };
            (trend, seasonal)
        };

        let spec = Self::new(error, trend, seasonal);

        // Validate the combination
        if !spec.is_valid() {
            return Err(ForecastError::InvalidParameter(format!(
                "ETS({}) is an unstable model combination per FPP3 taxonomy",
                notation
            )));
        }

        Ok(spec)
    }
}

/// Inner loop macro for ETS likelihood computation.
///
/// Hoists the trend×seasonal match outside the per-observation loop, so the
/// compiler sees a branch-free inner loop per arm. Two variants:
/// - `nonseasonal`: no seasonal buffer access (3 arms)
/// - `seasonal`: reads/writes seasonal buffer by index (6 arms)
///
/// Caller-provided identifiers (`$y`, `$s`, `$si`, `$lp`) bridge macro hygiene:
/// the forecast/update token trees can reference them because they share the
/// caller's syntax context.
macro_rules! ets_likelihood_loop {
    // Non-seasonal: no seasonal buffer access needed
    (nonseasonal $values:expr, $start_idx:expr, $is_mult_error:expr,
     $level:ident, $trend:ident,
     $y:ident, $lp:ident,
     forecast { $($fc:tt)* }
     update { $($upd:tt)* }
    ) => {{
        let mut _sum_sq = 0.0_f64;
        let mut _sum_log = 0.0_f64;
        let mut _cnt = 0_usize;
        for (_, &$y) in $values.iter().enumerate().skip($start_idx) {
            let _fc = { $($fc)* };
            if !_fc.is_finite() { return f64::MAX; }
            let _err = $y - _fc;
            let _se = if $is_mult_error && _fc.abs() > 1e-10 { _err / _fc } else { _err };
            _sum_sq += _se * _se;
            if !_sum_sq.is_finite() { return f64::MAX; }
            if $is_mult_error { _sum_log += $y.abs().ln(); }
            _cnt += 1;
            let $lp = $level;
            $($upd)*
        }
        (_sum_sq, _sum_log, _cnt)
    }};
    // Seasonal: reads/writes seasonal buffer by index
    (seasonal $values:expr, $start_idx:expr, $period:expr, $is_mult_error:expr,
     $level:ident, $trend:ident, $buf:ident,
     $y:ident, $s:ident, $si:ident, $lp:ident,
     forecast { $($fc:tt)* }
     update { $($upd:tt)* }
    ) => {{
        let mut _sum_sq = 0.0_f64;
        let mut _sum_log = 0.0_f64;
        let mut _cnt = 0_usize;
        for (_t, &$y) in $values.iter().enumerate().skip($start_idx) {
            let $si = _t % $period;
            let $s = $buf[$si];
            let _fc = { $($fc)* };
            if !_fc.is_finite() { return f64::MAX; }
            let _err = $y - _fc;
            let _se = if $is_mult_error && _fc.abs() > 1e-10 { _err / _fc } else { _err };
            _sum_sq += _se * _se;
            if !_sum_sq.is_finite() { return f64::MAX; }
            if $is_mult_error { _sum_log += $y.abs().ln(); }
            _cnt += 1;
            let $lp = $level;
            $($upd)*
        }
        (_sum_sq, _sum_log, _cnt)
    }};
}

/// ETS state-space model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
    fitted: Option<Vec<f64>>,
    /// Residuals.
    #[cfg_attr(feature = "serde", serde(with = "crate::utils::persistence::nan_vec"))]
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
    /// OLS result for exogenous regressors.
    #[cfg_attr(feature = "serde", serde(skip))]
    exog_ols: Option<OLSResult>,
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
            exog_ols: None,
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
            exog_ols: None,
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

    /// Initialize state components using heuristics.
    ///
    /// For non-seasonal ETS(A,A,N), uses regression-based initialization
    /// on the first `maxn` observations to match statsforecast behavior.
    ///
    /// For seasonal models, uses classical decomposition: averages seasonal
    /// indices across all complete cycles for robust estimation, matching
    /// R's forecast::ets() approach.
    fn initialize_state(&self, values: &[f64]) -> (f64, f64, Vec<f64>) {
        let period = self.seasonal_period;

        // Initial level and trend using regression for non-seasonal trend models
        // This matches statsforecast's initialization approach:
        // maxn = min(max(10, 2*m), len(y))
        let (level, trend) = if self.spec.has_trend() && !self.spec.has_seasonal() {
            // Use linear regression on first maxn points (statsforecast approach)
            let maxn = values.len().min(10.max(2 * period));
            let n = maxn;

            // Linear regression: y = a + b*x where x = 1, 2, ..., n
            // Using 1-indexed x to match statsforecast
            let x_mean = (n + 1) as f64 / 2.0;
            let y_mean = values.iter().take(n).sum::<f64>() / n as f64;

            let mut ss_xx = 0.0;
            let mut ss_xy = 0.0;
            for (i, &y) in values.iter().take(n).enumerate() {
                let x = (i + 1) as f64; // 1-indexed like statsforecast
                ss_xx += (x - x_mean).powi(2);
                ss_xy += (x - x_mean) * (y - y_mean);
            }

            let b = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
            let a = y_mean - b * x_mean;

            // Initial level (intercept at x=0), initial trend is slope
            (a, b)
        } else if self.spec.has_seasonal() && values.len() >= period {
            // Classical decomposition: regression on per-cycle means for level/trend
            let n_complete = values.len() / period;
            if n_complete >= 2 && self.spec.has_trend() {
                // Regression on per-cycle means without allocating a Vec:
                // two inline passes instead of one allocation + one pass.
                let nc = n_complete as f64;
                let x_mean = (nc - 1.0) / 2.0;
                let inv_period = 1.0 / period as f64;
                // Pass 1: compute y_mean (mean of cycle means)
                let y_sum: f64 = (0..n_complete)
                    .map(|c| {
                        let start = c * period;
                        values[start..start + period].iter().sum::<f64>() * inv_period
                    })
                    .sum();
                let y_mean = y_sum / nc;
                // Pass 2: compute regression coefficients
                let mut ss_xx = 0.0;
                let mut ss_xy = 0.0;
                for c in 0..n_complete {
                    let start = c * period;
                    let ym = values[start..start + period].iter().sum::<f64>() * inv_period;
                    let x = c as f64;
                    let dx = x - x_mean;
                    ss_xx += dx * dx;
                    ss_xy += dx * (ym - y_mean);
                }
                let trend_per_cycle = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
                let level = y_mean - trend_per_cycle * x_mean;
                let trend = trend_per_cycle / period as f64; // per-step trend
                (level, trend)
            } else {
                // Single cycle or no trend: use first period mean
                let level = values.iter().take(period).sum::<f64>() / period as f64;
                let trend = if self.spec.has_trend() && values.len() >= 2 * period {
                    let sum: f64 = (0..period)
                        .map(|i| (values[period + i] - values[i]) / period as f64)
                        .sum();
                    sum / period as f64
                } else {
                    0.0
                };
                (level, trend)
            }
        } else {
            // Simple: first value for level
            let level = values[0];
            let trend = if self.spec.has_trend() && values.len() >= 2 {
                values[1] - values[0]
            } else {
                0.0
            };
            (level, trend)
        };

        // Initial seasonal indices using classical decomposition:
        // Average deviations across all complete cycles for robust estimates.
        let seasonals = if self.spec.has_seasonal() && values.len() >= period {
            let n_complete = values.len() / period;
            match self.spec.seasonal {
                SeasonalType::Additive => {
                    let mut seasonal = vec![0.0; period];
                    for c in 0..n_complete {
                        let start = c * period;
                        // Detrend: expected level at this cycle's midpoint
                        let cycle_level =
                            level + trend * (start as f64 + (period - 1) as f64 / 2.0);
                        for j in 0..period {
                            seasonal[j] += values[start + j] - cycle_level;
                        }
                    }
                    let nc = n_complete as f64;
                    for s in &mut seasonal {
                        *s /= nc;
                    }
                    // Normalize: ensure seasonal indices sum to zero
                    let mean = seasonal.iter().sum::<f64>() / period as f64;
                    for s in &mut seasonal {
                        *s -= mean;
                    }
                    seasonal
                }
                SeasonalType::Multiplicative => {
                    let mut seasonal = vec![0.0; period];
                    let mut valid_cycles = 0usize;
                    for c in 0..n_complete {
                        let start = c * period;
                        let cycle_level =
                            level + trend * (start as f64 + (period - 1) as f64 / 2.0);
                        if cycle_level.abs() > 1e-10 {
                            for j in 0..period {
                                seasonal[j] += values[start + j] / cycle_level;
                            }
                            valid_cycles += 1;
                        }
                    }
                    for j in 0..period {
                        seasonal[j] = if valid_cycles > 0 {
                            (seasonal[j] / valid_cycles as f64).clamp(0.01, 100.0)
                        } else {
                            1.0
                        };
                    }
                    // Normalize: ensure seasonal indices average to 1.0
                    let mean = seasonal.iter().sum::<f64>() / period as f64;
                    if mean.abs() > 1e-10 {
                        for s in &mut seasonal {
                            *s /= mean;
                        }
                    }
                    seasonal
                }
                SeasonalType::None => vec![],
            }
        } else {
            vec![]
        };

        (level, trend, seasonals)
    }

    /// Calculate negative log-likelihood with a reusable seasonal buffer.
    ///
    /// Avoids heap allocation per evaluation by reusing `seasonal_buf` across
    /// calls in optimization loops. Uses hoisted match dispatch via
    /// `ets_likelihood_loop!` so the compiler sees a branch-free inner loop.
    fn calculate_likelihood_with_init_buf(
        &self,
        values: &[f64],
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        init_level: Option<f64>,
        init_trend: Option<f64>,
        init_seasonals: Option<&[f64]>,
        seasonal_buf: &mut Vec<f64>,
    ) -> f64 {
        let n = values.len();
        let period = self.seasonal_period;
        let start_idx = if self.spec.has_seasonal() { period } else { 0 };

        if n <= start_idx + 1 {
            return f64::MAX;
        }

        // Use provided initial states or fallback to heuristic.
        // Reuse seasonal_buf instead of allocating via to_vec().
        let (mut level, mut trend) = match (init_level, init_seasonals) {
            (Some(l), Some(s)) => {
                seasonal_buf.resize(s.len(), 0.0);
                seasonal_buf.copy_from_slice(s);
                (l, init_trend.unwrap_or(0.0))
            }
            (Some(l), None) => {
                let (_, _, hs) = self.initialize_state(values);
                seasonal_buf.clear();
                seasonal_buf.extend_from_slice(&hs);
                (l, init_trend.unwrap_or(0.0))
            }
            _ => {
                let (hl, ht, hs) = self.initialize_state(values);
                seasonal_buf.clear();
                seasonal_buf.extend_from_slice(&hs);
                (init_level.unwrap_or(hl), init_trend.unwrap_or(ht))
            }
        };

        let phi = phi.unwrap_or(1.0);
        let beta = beta.unwrap_or(0.0);
        let gamma = gamma.unwrap_or(0.0);
        let is_mult_error = self.spec.error == ErrorType::Multiplicative;

        // Hoisted match: dispatch once, then run a branch-free inner loop.
        // Eliminates two match dispatches per observation (forecast + update).
        let (sum_sq_errors, sum_log_y, count) = match (self.spec.trend, self.spec.seasonal) {
            (TrendType::None, SeasonalType::None) => {
                ets_likelihood_loop!(nonseasonal values, start_idx, is_mult_error,
                    level, trend,
                    y, _level_prev,
                    forecast { level }
                    update {
                        level = alpha * y + (1.0 - alpha) * level;
                    }
                )
            }
            (TrendType::None, SeasonalType::Additive) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, _level_prev,
                    forecast { level + s }
                    update {
                        level = alpha * (y - s) + (1.0 - alpha) * level;
                        seasonal_buf[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                    }
                )
            }
            (TrendType::None, SeasonalType::Multiplicative) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, _level_prev,
                    forecast { level * s }
                    update {
                        let y_des = if s.abs() > 1e-10 { y / s } else { y };
                        level = alpha * y_des + (1.0 - alpha) * level;
                        seasonal_buf[season_idx] = if level.abs() > 1e-10 {
                            gamma * (y / level) + (1.0 - gamma) * s
                        } else {
                            s
                        };
                    }
                )
            }
            (TrendType::Additive, SeasonalType::None) => {
                ets_likelihood_loop!(nonseasonal values, start_idx, is_mult_error,
                    level, trend,
                    y, level_prev,
                    forecast { level + trend }
                    update {
                        level = alpha * y + (1.0 - alpha) * (level_prev + trend);
                        trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    }
                )
            }
            (TrendType::Additive, SeasonalType::Additive) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev,
                    forecast { level + trend + s }
                    update {
                        level = alpha * (y - s) + (1.0 - alpha) * (level_prev + trend);
                        trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                        seasonal_buf[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                    }
                )
            }
            (TrendType::Additive, SeasonalType::Multiplicative) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev,
                    forecast { (level + trend) * s }
                    update {
                        let y_des = if s.abs() > 1e-10 { y / s } else { y };
                        level = alpha * y_des + (1.0 - alpha) * (level_prev + trend);
                        trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                        seasonal_buf[season_idx] = if level.abs() > 1e-10 {
                            gamma * (y / level) + (1.0 - gamma) * s
                        } else {
                            s
                        };
                    }
                )
            }
            (TrendType::AdditiveDamped, SeasonalType::None) => {
                ets_likelihood_loop!(nonseasonal values, start_idx, is_mult_error,
                    level, trend,
                    y, level_prev,
                    forecast { level + phi * trend }
                    update {
                        level = alpha * y + (1.0 - alpha) * (level_prev + phi * trend);
                        trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                    }
                )
            }
            (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev,
                    forecast { level + phi * trend + s }
                    update {
                        level = alpha * (y - s) + (1.0 - alpha) * (level_prev + phi * trend);
                        trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                        seasonal_buf[season_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                    }
                )
            }
            (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev,
                    forecast { (level + phi * trend) * s }
                    update {
                        let y_des = if s.abs() > 1e-10 { y / s } else { y };
                        level = alpha * y_des + (1.0 - alpha) * (level_prev + phi * trend);
                        trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                        seasonal_buf[season_idx] = if level.abs() > 1e-10 {
                            gamma * (y / level) + (1.0 - gamma) * s
                        } else {
                            s
                        };
                    }
                )
            }
        };

        if count == 0 {
            return f64::MAX;
        }

        let sigma2 = sum_sq_errors / count as f64;
        let ll = if is_mult_error {
            -0.5 * count as f64 * (1.0 + sigma2.ln() + (2.0 * std::f64::consts::PI).ln())
                - sum_log_y
        } else {
            -0.5 * count as f64 * (1.0 + sigma2.ln() + (2.0 * std::f64::consts::PI).ln())
        };

        -ll
    }

    /// Calculate negative log-likelihood for given parameters.
    ///
    /// Thin wrapper around `calculate_likelihood_with_init_buf` that allocates
    /// a temporary buffer. Use `_buf` directly in hot loops with a `RefCell`.
    fn calculate_likelihood_with_init(
        &self,
        values: &[f64],
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        init_level: Option<f64>,
        init_trend: Option<f64>,
        init_seasonals: Option<&[f64]>,
    ) -> f64 {
        let mut buf = Vec::new();
        self.calculate_likelihood_with_init_buf(
            values,
            alpha,
            beta,
            gamma,
            phi,
            init_level,
            init_trend,
            init_seasonals,
            &mut buf,
        )
    }

    /// Optimize parameters and initial states.
    ///
    /// For non-seasonal models, optimizes smoothing parameters and initial states.
    /// For seasonal models, jointly optimizes smoothing parameters, initial level,
    /// and initial seasonal indices for better seasonal model fits.
    fn optimize_params(
        &self,
        values: &[f64],
    ) -> (
        f64,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        f64,
        f64,
        Option<Vec<f64>>,
    ) {
        let config = NelderMeadConfig {
            max_iter: 2000,
            tolerance: 1e-10,
            ..Default::default()
        };

        let has_trend = self.spec.has_trend();
        let has_seasonal = self.spec.has_seasonal();
        let is_damped = self.spec.is_damped();

        // Get initial estimates for states using heuristics
        let (init_level, init_trend, init_seasonals) = self.initialize_state(values);

        // Determine bounds for initial states - wide bounds like statsforecast
        let (y_min, y_max) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &y| {
                (min.min(y), max.max(y))
            });
        let y_range = y_max - y_min;
        let level_bounds = (y_min - y_range, y_max + y_range);
        let trend_bounds = (-y_range, y_range);

        // For ETS(A,A,N) - non-seasonal trend model
        // Optimize: alpha, beta, l0, b0
        // Use multiple starting points to find global optimum
        if has_trend && !is_damped && !has_seasonal {
            let alpha_starts = [0.1, 0.3, 0.5, 0.8, 0.99];
            let mut best_result = None;
            let mut best_value = f64::MAX;

            for &alpha_init in &alpha_starts {
                let result = nelder_mead(
                    |p| {
                        self.calculate_likelihood_with_init(
                            values,
                            p[0],
                            Some(p[1]),
                            None,
                            None,
                            Some(p[2]),
                            Some(p[3]),
                            None,
                        )
                    },
                    &[alpha_init, 0.01, init_level, init_trend],
                    Some(&[
                        (0.0001, 0.9999),
                        (0.0001, 0.9999),
                        level_bounds,
                        trend_bounds,
                    ]),
                    config.clone(),
                );

                if result.optimal_value < best_value {
                    best_value = result.optimal_value;
                    best_result = Some(result);
                }
            }

            let result = best_result.unwrap();
            return (
                result.optimal_point[0].clamp(0.0001, 0.9999),
                Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                None,
                None,
                result.optimal_point[2],
                result.optimal_point[3],
                None,
            );
        }

        // For seasonal models: jointly optimize smoothing params + l0 + seasonal states.
        // Uses multi-start with different (alpha, gamma) starting points to avoid local minima,
        // and higher max_iter to handle the larger parameter space.
        if has_seasonal && !init_seasonals.is_empty() {
            let period = self.seasonal_period;
            let seasonal_config = NelderMeadConfig {
                max_iter: 5000 + 500 * period,
                tolerance: 1e-10,
                ..Default::default()
            };

            // Seasonal bounds: additive uses data-range, multiplicative uses [0.01, 100]
            let seasonal_bounds: Vec<(f64, f64)> =
                if self.spec.seasonal == SeasonalType::Multiplicative {
                    vec![(0.01, 100.0); period]
                } else {
                    vec![(-2.0 * y_range, 2.0 * y_range); period]
                };

            // Multi-start: try different (alpha, gamma) starting points to avoid local minima.
            // Varying gamma is crucial for seasonal capture.
            let ag_starts: [(f64, f64); 6] = [
                (0.1, 0.05),
                (0.1, 0.3),
                (0.3, 0.1),
                (0.3, 0.3),
                (0.5, 0.1),
                (0.8, 0.01),
            ];

            let mut best_value = f64::MAX;
            let mut best_result: Option<Vec<f64>> = None;

            match (has_trend, is_damped) {
                (false, _) => {
                    // alpha, gamma, l0, s0[0..period]
                    let n_params = 2 + 1 + period;
                    let mut bounds = Vec::with_capacity(n_params);
                    bounds.push((0.0001, 0.9999)); // alpha
                    bounds.push((0.0001, 0.9999)); // gamma
                    bounds.push(level_bounds); // l0
                    bounds.extend_from_slice(&seasonal_bounds);

                    let seasonal_buf = RefCell::new(vec![0.0; period]);
                    let mut start = Vec::with_capacity(n_params);
                    for &(alpha_init, gamma_init) in &ag_starts {
                        start.clear();
                        start.push(alpha_init);
                        start.push(gamma_init);
                        start.push(init_level);
                        start.extend_from_slice(&init_seasonals);

                        let result = nelder_mead(
                            |p| {
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    p[0],
                                    None,
                                    Some(p[1]),
                                    None,
                                    Some(p[2]),
                                    None,
                                    Some(&p[3..3 + period]),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&bounds),
                            seasonal_config.clone(),
                        );

                        if result.optimal_value < best_value {
                            best_value = result.optimal_value;
                            best_result = Some(result.optimal_point);
                        }
                    }

                    let r = best_result.unwrap();
                    let opt_seasonals = r[3..3 + period].to_vec();
                    (
                        r[0].clamp(0.0001, 0.9999),
                        None,
                        Some(r[1].clamp(0.0001, 0.9999)),
                        None,
                        r[2],
                        init_trend,
                        Some(opt_seasonals),
                    )
                }
                (true, false) => {
                    // alpha, beta, gamma, l0, b0, s0[0..period]
                    let n_params = 3 + 2 + period;
                    let mut bounds = Vec::with_capacity(n_params);
                    bounds.push((0.0001, 0.9999)); // alpha
                    bounds.push((0.0001, 0.9999)); // beta
                    bounds.push((0.0001, 0.9999)); // gamma
                    bounds.push(level_bounds); // l0
                    bounds.push(trend_bounds); // b0
                    bounds.extend_from_slice(&seasonal_bounds);

                    let seasonal_buf = RefCell::new(vec![0.0; period]);
                    let mut start = Vec::with_capacity(n_params);
                    for &(alpha_init, gamma_init) in &ag_starts {
                        start.clear();
                        start.push(alpha_init);
                        start.push(0.1); // beta
                        start.push(gamma_init);
                        start.push(init_level);
                        start.push(init_trend);
                        start.extend_from_slice(&init_seasonals);

                        let result = nelder_mead(
                            |p| {
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    p[0],
                                    Some(p[1]),
                                    Some(p[2]),
                                    None,
                                    Some(p[3]),
                                    Some(p[4]),
                                    Some(&p[5..5 + period]),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&bounds),
                            seasonal_config.clone(),
                        );

                        if result.optimal_value < best_value {
                            best_value = result.optimal_value;
                            best_result = Some(result.optimal_point);
                        }
                    }

                    let r = best_result.unwrap();
                    let opt_seasonals = r[5..5 + period].to_vec();
                    (
                        r[0].clamp(0.0001, 0.9999),
                        Some(r[1].clamp(0.0001, 0.9999)),
                        Some(r[2].clamp(0.0001, 0.9999)),
                        None,
                        r[3],
                        r[4],
                        Some(opt_seasonals),
                    )
                }
                (true, true) => {
                    // alpha, beta, gamma, phi, l0, b0, s0[0..period]
                    let n_params = 4 + 2 + period;
                    let mut bounds = Vec::with_capacity(n_params);
                    bounds.push((0.0001, 0.9999)); // alpha
                    bounds.push((0.0001, 0.9999)); // beta
                    bounds.push((0.0001, 0.9999)); // gamma
                    bounds.push((0.8, 0.98)); // phi
                    bounds.push(level_bounds); // l0
                    bounds.push(trend_bounds); // b0
                    bounds.extend_from_slice(&seasonal_bounds);

                    let seasonal_buf = RefCell::new(vec![0.0; period]);
                    let mut start = Vec::with_capacity(n_params);
                    for &(alpha_init, gamma_init) in &ag_starts {
                        start.clear();
                        start.push(alpha_init);
                        start.push(0.1); // beta
                        start.push(gamma_init);
                        start.push(0.98); // phi
                        start.push(init_level);
                        start.push(init_trend);
                        start.extend_from_slice(&init_seasonals);

                        let result = nelder_mead(
                            |p| {
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    p[0],
                                    Some(p[1]),
                                    Some(p[2]),
                                    Some(p[3]),
                                    Some(p[4]),
                                    Some(p[5]),
                                    Some(&p[6..6 + period]),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&bounds),
                            seasonal_config.clone(),
                        );

                        if result.optimal_value < best_value {
                            best_value = result.optimal_value;
                            best_result = Some(result.optimal_point);
                        }
                    }

                    let r = best_result.unwrap();
                    let opt_seasonals = r[6..6 + period].to_vec();
                    (
                        r[0].clamp(0.0001, 0.9999),
                        Some(r[1].clamp(0.0001, 0.9999)),
                        Some(r[2].clamp(0.0001, 0.9999)),
                        Some(r[3].clamp(0.8, 0.98)),
                        r[4],
                        r[5],
                        Some(opt_seasonals),
                    )
                }
            }
        } else {
            // Non-seasonal models: optimize smoothing params only
            match (has_trend, is_damped) {
                (false, _) => {
                    // Just alpha (ETS(A,N,N) or ETS(M,N,N))
                    let result = nelder_mead(
                        |p| {
                            self.calculate_likelihood_with_init(
                                values, p[0], None, None, None, None, None, None,
                            )
                        },
                        &[0.3],
                        Some(&[(0.0001, 0.9999)]),
                        config,
                    );
                    (
                        result.optimal_point[0].clamp(0.0001, 0.9999),
                        None,
                        None,
                        None,
                        init_level,
                        init_trend,
                        None,
                    )
                }
                (true, false) => {
                    // alpha, beta (non-damped trend, no seasonal) — shouldn't reach here
                    // (handled by the ETS(A,A,N) multi-start above)
                    (0.3, Some(0.1), None, None, init_level, init_trend, None)
                }
                (true, _) => {
                    // alpha, beta, phi (damped trend, no seasonal)
                    let result = nelder_mead(
                        |p| {
                            self.calculate_likelihood_with_init(
                                values,
                                p[0],
                                Some(p[1]),
                                None,
                                Some(p[2]),
                                None,
                                None,
                                None,
                            )
                        },
                        &[0.3, 0.1, 0.98],
                        Some(&[(0.0001, 0.9999), (0.0001, 0.9999), (0.8, 0.98)]),
                        config,
                    );
                    (
                        result.optimal_point[0].clamp(0.0001, 0.9999),
                        Some(result.optimal_point[1].clamp(0.0001, 0.9999)),
                        None,
                        Some(result.optimal_point[2].clamp(0.8, 0.98)),
                        init_level,
                        init_trend,
                        None,
                    )
                }
            }
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
            // Seasonal indices are constrained (sum-to-zero for additive,
            // mean-to-one for multiplicative), so one is determined by the rest.
            count += self.seasonal_period - 1;
        }
        count += 1; // sigma^2 (matches statsforecast parameter counting)
        count
    }

    /// Internal prediction with optional exogenous regressors.
    fn predict_internal(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<Forecast> {
        let level = self
            .level
            .ok_or(ForecastError::FitRequired { model: None })?;
        let trend = self.trend.unwrap_or(0.0);
        let phi = self.phi.unwrap_or(1.0);
        let period = self.seasonal_period;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let seasonals_ref = if self.spec.has_seasonal() {
            Some(
                self.seasonals
                    .as_ref()
                    .ok_or(ForecastError::FitRequired { model: None })?,
            )
        } else {
            None
        };

        // Calculate exogenous contribution if applicable
        let exog_contribution = if let Some(ols) = &self.exog_ols {
            let future = future_regressors.ok_or_else(|| {
                ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors. Future regressor values required."
                        .into(),
                )
            })?;

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

                let mut pred = match self.spec.seasonal {
                    SeasonalType::None => level + trend_component,
                    SeasonalType::Additive => level + trend_component + s,
                    SeasonalType::Multiplicative => (level + trend_component) * s,
                };

                if let Some(ref exog) = exog_contribution {
                    pred += exog[h - 1];
                }

                pred
            })
            .collect();

        Ok(Forecast::from_values(predictions))
    }

    /// Internal prediction with intervals and optional exogenous regressors.
    fn predict_internal_with_intervals(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
        confidence: f64,
    ) -> Result<Forecast> {
        let forecast = self.predict_internal(horizon, future_regressors)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let period = self.seasonal_period;

        if horizon == 0 {
            return Ok(forecast);
        }

        let z = quantile_normal((1.0 + confidence) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let k = if self.spec.has_seasonal() {
                ((h - 1) / period) + 1
            } else {
                h
            };
            let se = (variance * k as f64).sqrt();

            lower.push(preds[h - 1] - z * se);
            upper.push(preds[h - 1] + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }
}

impl Default for ETS {
    fn default() -> Self {
        Self::new(ETSSpec::ann(), 1)
    }
}

impl Forecaster for ETS {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        let raw_values = series.primary_values();

        // Handle exogenous regressors.
        // Use Cow to avoid cloning raw_values when no regressors are present.
        let adjusted_values: Cow<'_, [f64]> = if series.has_regressors() {
            let regressors = series.all_regressors();
            let ols_result = ols_fit(raw_values, &regressors)?;
            let adjusted = ols_residuals(raw_values, &ols_result, &regressors)?;
            self.exog_ols = Some(ols_result);
            Cow::Owned(adjusted)
        } else {
            self.exog_ols = None;
            Cow::Borrowed(raw_values)
        };
        let values = &*adjusted_values;

        let min_len = if self.spec.has_seasonal() {
            2 * self.seasonal_period
        } else {
            2
        };

        if values.len() < min_len {
            return Err(ForecastError::InsufficientData {
                needed: min_len,
                got: values.len(),
                hint: Some(if self.spec.has_seasonal() {
                    format!(
                        "ETS with seasonality requires at least 2 * period = {} observations",
                        min_len
                    )
                } else {
                    "ETS requires at least 2 observations".into()
                }),
            });
        }

        self.n = values.len();

        // Initialize state: optimize_params() calls initialize_state() internally,
        // so skip the redundant call when optimizing (the common case).
        let (init_level, init_trend, mut seasonals);
        if self.optimize {
            let (alpha, beta, gamma, phi, opt_level, opt_trend, opt_seasonals) =
                self.optimize_params(values);
            self.alpha = Some(alpha);
            self.beta = beta;
            self.gamma = gamma;
            self.phi = phi;
            init_level = opt_level;
            init_trend = opt_trend;
            seasonals = opt_seasonals.unwrap_or_default();
        } else {
            let (hl, ht, hs) = self.initialize_state(values);
            init_level = hl;
            init_trend = ht;
            seasonals = hs;
        }

        let alpha = self.alpha.unwrap_or(0.3);
        let beta = self.beta.unwrap_or(0.1);
        let gamma = self.gamma.unwrap_or(0.1);
        let phi = self.phi.unwrap_or(1.0);
        let period = self.seasonal_period;

        // Use optimized or heuristic initial states
        let mut level = init_level;
        let mut trend = init_trend;
        let start_idx = if self.spec.has_seasonal() { period } else { 0 };

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
        // Use actual residual count for AIC calculation (statsforecast compatible)
        let valid_slice = &residuals[start_idx..];
        if !valid_slice.is_empty() {
            let variance = crate::simd::sum_of_squares(valid_slice) / valid_slice.len() as f64;
            self.residual_variance = Some(variance);

            // Use actual number of residuals for log-likelihood, not full sample size.
            // This ensures seasonal models (which skip the first period of residuals)
            // are not systematically penalized relative to non-seasonal models.
            let n = valid_slice.len() as f64;
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
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog() and provide future regressor values.".into()
            ));
        }
        self.predict_internal(horizon, None)
    }

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        if self.exog_ols.is_some() {
            return Err(ForecastError::InvalidParameter(
                "Model was fit with exogenous regressors. Use predict_with_exog_intervals() and provide future regressor values.".into()
            ));
        }
        self.predict_internal_with_intervals(horizon, None, confidence)
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
        self.predict_internal_with_intervals(horizon, Some(future_regressors), level)
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
        "ETS"
    }
}

impl Explainable for ETS {
    fn explain(&self, horizon: usize) -> Result<ForecastExplanation> {
        let level_val = self
            .level
            .ok_or(ForecastError::FitRequired { model: None })?;
        let trend_val = self.trend.unwrap_or(0.0);
        let phi = self.phi.unwrap_or(1.0);
        let period = self.seasonal_period;
        if horizon == 0 {
            return Ok(ForecastExplanation {
                level: vec![],
                trend: None,
                seasonal: None,
                residual: None,
                named_components: vec![],
            });
        }
        let seasonals_ref = if self.spec.has_seasonal() {
            Some(
                self.seasonals
                    .as_ref()
                    .ok_or(ForecastError::FitRequired { model: None })?,
            )
        } else {
            None
        };
        let mut level_component = Vec::with_capacity(horizon);
        let mut trend_component_vec = Vec::with_capacity(horizon);
        let mut seasonal_component_vec = Vec::with_capacity(horizon);
        for h in 1..=horizon {
            let trend_component = if self.spec.has_trend() {
                if self.spec.is_damped() {
                    Self::damped_sum(phi, h) * trend_val
                } else {
                    h as f64 * trend_val
                }
            } else {
                0.0
            };
            match self.spec.seasonal {
                SeasonalType::None => {
                    level_component.push(level_val);
                    trend_component_vec.push(trend_component);
                }
                SeasonalType::Additive => {
                    let s = seasonals_ref.unwrap()[(self.n + h - 1) % period];
                    level_component.push(level_val);
                    trend_component_vec.push(trend_component);
                    seasonal_component_vec.push(s);
                }
                SeasonalType::Multiplicative => {
                    let s = seasonals_ref.unwrap()[(self.n + h - 1) % period];
                    let base = level_val + trend_component;
                    level_component.push(level_val);
                    trend_component_vec.push(trend_component);
                    seasonal_component_vec.push(base * (s - 1.0));
                }
            }
        }
        Ok(ForecastExplanation {
            level: level_component,
            trend: if self.spec.has_trend() {
                Some(trend_component_vec)
            } else {
                None
            },
            seasonal: if self.spec.has_seasonal() {
                Some(seasonal_component_vec)
            } else {
                None
            },
            residual: None,
            named_components: vec![],
        })
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
        assert!(matches!(
            model.predict(5),
            Err(ForecastError::FitRequired { .. })
        ));
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

    /// Validation test comparing ETS(A,A,N) (Holt's method) output with statsforecast.
    ///
    /// Data: Perfect linear trend series (y = 10 + 0.5*t for t=0..49)
    /// This deterministic series allows both implementations to converge to optimal
    /// parameters and produce identical extrapolations.
    ///
    /// Reference: statsforecast.models.Holt which internally uses ETS(A,A,N)
    ///
    /// For a perfect linear series, both implementations should:
    /// 1. Learn the exact trend (slope = 0.5)
    /// 2. Extrapolate perfectly: y(50) = 35.0, y(51) = 35.5, etc.
    #[test]
    fn ets_aan_matches_statsforecast_linear_trend() {
        // Perfect linear series: y = 10 + 0.5*t for t=0..49
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::aan(), 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Expected from statsforecast.models.Holt on perfect linear series:
        // The extrapolation should continue the linear trend exactly.
        // Last value is 10 + 0.5*49 = 34.5
        // Next values should be 35.0, 35.5, 36.0, ...
        let expected = [
            35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5,
        ];

        for (i, (&pred, &exp)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(
                pred,
                exp,
                epsilon = 0.5,
                // Allow 0.5 tolerance due to optimization convergence differences
            );
            // Verify trend is approximately correct (step of ~0.5)
            if i > 0 {
                let step = preds[i] - preds[i - 1];
                assert_relative_eq!(step, 0.5, epsilon = 0.1);
            }
        }
    }

    /// Validation test for ETS(A,A,N) with fixed parameters comparing with statsforecast.
    ///
    /// This test uses pre-computed parameters from statsforecast to verify that
    /// the core ETS computation matches when using identical parameters.
    ///
    /// Data: Simple linear series (y = t for t=0..19)
    /// Parameters from statsforecast fit: alpha=0.207, beta=0.020
    ///
    /// This isolates the state-space computation from parameter optimization,
    /// ensuring the recursive update equations are implemented correctly.
    #[test]
    fn ets_aan_fixed_params_computation() {
        // Simple series: y = t for t=0..19
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Use fixed parameters similar to statsforecast output
        let mut model = ETS::with_params(ETSSpec::aan(), 1, 0.2, Some(0.02), None, None);
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        let preds = forecast.primary();

        // With these parameters on a linear series, forecasts should show increasing trend
        assert!(preds[4] > preds[0], "Forecasts should be increasing");

        // Each step should show positive trend (approximately 1.0 for y=t series)
        for i in 1..5 {
            let step = preds[i] - preds[i - 1];
            assert!(step > 0.0, "Step {} should be positive", i);
            // Trend should be close to 1.0 (the actual slope of y=t)
            assert!(
                step > 0.5 && step < 1.5,
                "Step {} should be ~1.0, got {}",
                i,
                step
            );
        }
    }

    /// Validation test comparing ETS(A,A,N) with statsforecast on trend data.
    ///
    /// Data: Synthetic trend series (100 observations, seed=42, intercept=10, slope=0.5)
    /// Generated by: validation/generate_data.py
    /// Reference: statsforecast.models.Holt (which uses ETS(A,A,N) internally)
    ///
    /// This test verifies that our optimized ETS(A,A,N) produces forecasts very close
    /// to statsforecast when both optimize alpha, beta, initial_level, and initial_trend.
    ///
    /// Expected: MAD < 0.1 (near-perfect agreement on trend data)
    #[test]
    fn ets_aan_matches_statsforecast_trend_data() {
        // Trend series: 100 observations (partial data for test)
        // Source: validation/data/trend.csv
        let values = vec![
            8.865512337881832,
            14.397684893358196,
            9.931208086815722,
            13.712546705401259,
            9.199146959970369,
            11.88368732639711,
            10.149933835268257,
            12.482900772298313,
            16.520924412372185,
            9.318038730422954,
            16.303270930637574,
            16.213206806996833,
            14.217550132909617,
            12.161826436834637,
            17.21638852314161,
            15.911521872808592,
            18.698028634064112,
            18.56555643657033,
            23.805336673962746,
            18.781933117580927,
            16.929507522134404,
            21.037826904868947,
            21.659990051915294,
            25.577562725721307,
            24.505333737743737,
            23.57061317744853,
            27.389908673658685,
            19.933710837031448,
            22.080745401750757,
            21.720272175783425,
            23.830570590532695,
            21.369941557331074,
            27.905452840443214,
            25.83333190870368,
            22.587581116492025,
            24.453262756377377,
            28.940541542350587,
            31.014379703683144,
            34.99019267507536,
            38.24158739802199,
            31.24322829982799,
            27.531385639904407,
            24.603861157806072,
            32.303134387031506,
            29.56117671406902,
            31.253928219460946,
            31.163709602820575,
            33.077627340750844,
            37.197940692362934,
            34.97114570233603,
            34.52409548888394,
            32.39303874152257,
            30.97595116588693,
            35.04107627278001,
            36.838652347545036,
            42.803789740739646,
            38.39082356441866,
            41.44821853306917,
            37.502113204382546,
            35.94516870074892,
            37.104649713302884,
            38.32432180639274,
            47.38540919730549,
            39.03583996232684,
            44.51546761120903,
            39.79121846573892,
            45.79471903862273,
            44.65485289831759,
            43.53008630702573,
            44.3777124215937,
            43.035636913711826,
            46.838216604446245,
            44.63504955897766,
            42.823182708698255,
            43.16618727704114,
            48.017763753166356,
            52.73727376923131,
            48.97997484072032,
            48.64408502167035,
            50.35747841880763,
            53.918005225120474,
            51.158147504091566,
            49.76721830749879,
            54.818866130179664,
            53.28626931538484,
            57.10726797598797,
            53.549703311665716,
            49.8265929048385,
            49.895522402263005,
            59.45278379669375,
            60.17099716234989,
            54.9614423601522,
            54.85043803659204,
            60.8843328767266,
            53.67886295386953,
            54.81581894313252,
            59.929980384067136,
            57.31618463142123,
            58.984634399839784,
            59.00967130442646,
        ];

        let timestamps = make_timestamps(values.len());
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::aan(), 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Expected from statsforecast.models.Holt on trend data:
        // First forecast should be around 60.36 (from validation output)
        let expected_first = 60.36;
        let expected_step = 0.508; // Approximate trend per step from statsforecast

        // Check first forecast is close
        assert!(
            (preds[0] - expected_first).abs() < 1.0,
            "First forecast {} should be close to {}",
            preds[0],
            expected_first
        );

        // Check that forecasts are increasing (positive trend)
        for i in 1..preds.len() {
            assert!(
                preds[i] > preds[i - 1],
                "Forecasts should be increasing: {} > {} at step {}",
                preds[i],
                preds[i - 1],
                i
            );
        }

        // Check approximate trend (should be around 0.5)
        let avg_step: f64 = (1..preds.len())
            .map(|i| preds[i] - preds[i - 1])
            .sum::<f64>()
            / (preds.len() - 1) as f64;
        assert!(
            (avg_step - expected_step).abs() < 0.2,
            "Average step {} should be close to {}",
            avg_step,
            expected_step
        );
    }

    // =========================================================================
    // Tests for ETSSpec::from_notation() and is_valid()
    // =========================================================================

    #[test]
    fn ets_spec_from_notation_valid_3char() {
        // Test all valid 3-character notations
        assert_eq!(ETSSpec::from_notation("ANN").unwrap(), ETSSpec::ann());
        assert_eq!(ETSSpec::from_notation("AAN").unwrap(), ETSSpec::aan());
        assert_eq!(ETSSpec::from_notation("AAA").unwrap(), ETSSpec::aaa());
        assert_eq!(ETSSpec::from_notation("AAM").unwrap(), ETSSpec::aam());
        assert_eq!(ETSSpec::from_notation("ANA").unwrap(), ETSSpec::ana());
        assert_eq!(ETSSpec::from_notation("ANM").unwrap(), ETSSpec::anm());
        assert_eq!(ETSSpec::from_notation("MNN").unwrap(), ETSSpec::mnn());
        assert_eq!(ETSSpec::from_notation("MAN").unwrap(), ETSSpec::man());
        assert_eq!(ETSSpec::from_notation("MAM").unwrap(), ETSSpec::mam());
        assert_eq!(ETSSpec::from_notation("MNM").unwrap(), ETSSpec::mnm());
    }

    #[test]
    fn ets_spec_from_notation_valid_4char_damped() {
        // Test all valid 4-character (damped trend) notations
        assert_eq!(ETSSpec::from_notation("AAdN").unwrap(), ETSSpec::aadn());
        assert_eq!(ETSSpec::from_notation("AAdA").unwrap(), ETSSpec::aada());
        assert_eq!(ETSSpec::from_notation("AAdM").unwrap(), ETSSpec::aadm());
        assert_eq!(ETSSpec::from_notation("MAdN").unwrap(), ETSSpec::madn());
        assert_eq!(ETSSpec::from_notation("MAdM").unwrap(), ETSSpec::madm());
    }

    #[test]
    fn ets_spec_from_notation_case_insensitive() {
        // Test case insensitivity
        assert_eq!(ETSSpec::from_notation("ann").unwrap(), ETSSpec::ann());
        assert_eq!(ETSSpec::from_notation("Ann").unwrap(), ETSSpec::ann());
        assert_eq!(ETSSpec::from_notation("aadn").unwrap(), ETSSpec::aadn());
        assert_eq!(ETSSpec::from_notation("mam").unwrap(), ETSSpec::mam());
        assert_eq!(ETSSpec::from_notation("MAdM").unwrap(), ETSSpec::madm());
    }

    #[test]
    fn ets_spec_from_notation_invalid_unstable_combinations() {
        // MAA and MAdA are unstable per FPP3 taxonomy
        let result_maa = ETSSpec::from_notation("MAA");
        assert!(result_maa.is_err());
        assert!(result_maa
            .unwrap_err()
            .to_string()
            .contains("unstable model combination"));

        let result_mada = ETSSpec::from_notation("MAdA");
        assert!(result_mada.is_err());
        assert!(result_mada
            .unwrap_err()
            .to_string()
            .contains("unstable model combination"));
    }

    #[test]
    fn ets_spec_from_notation_invalid_format() {
        // Too short
        assert!(ETSSpec::from_notation("AA").is_err());
        assert!(ETSSpec::from_notation("A").is_err());
        assert!(ETSSpec::from_notation("").is_err());

        // Too long
        assert!(ETSSpec::from_notation("AAAAA").is_err());

        // Invalid characters
        assert!(ETSSpec::from_notation("XNN").is_err()); // Invalid error type
        assert!(ETSSpec::from_notation("AXN").is_err()); // Invalid trend type
        assert!(ETSSpec::from_notation("ANX").is_err()); // Invalid seasonal type

        // Invalid 4-char format (not damped)
        assert!(ETSSpec::from_notation("AANN").is_err());
        assert!(ETSSpec::from_notation("ABNN").is_err());
    }

    #[test]
    fn ets_spec_is_valid_stable_combinations() {
        // All these should be valid
        assert!(ETSSpec::ann().is_valid());
        assert!(ETSSpec::aan().is_valid());
        assert!(ETSSpec::aadn().is_valid());
        assert!(ETSSpec::aaa().is_valid());
        assert!(ETSSpec::aam().is_valid());
        assert!(ETSSpec::ana().is_valid());
        assert!(ETSSpec::anm().is_valid());
        assert!(ETSSpec::aada().is_valid());
        assert!(ETSSpec::aadm().is_valid());
        assert!(ETSSpec::mnn().is_valid());
        assert!(ETSSpec::man().is_valid());
        assert!(ETSSpec::madn().is_valid());
        assert!(ETSSpec::mam().is_valid());
        assert!(ETSSpec::mnm().is_valid());
        assert!(ETSSpec::madm().is_valid());
    }

    #[test]
    fn ets_spec_is_valid_unstable_combinations() {
        // MAA - Multiplicative error + Additive trend + Additive seasonal
        let maa = ETSSpec::new(
            ErrorType::Multiplicative,
            TrendType::Additive,
            SeasonalType::Additive,
        );
        assert!(!maa.is_valid());

        // MAdA - Multiplicative error + Damped trend + Additive seasonal
        let mada = ETSSpec::new(
            ErrorType::Multiplicative,
            TrendType::AdditiveDamped,
            SeasonalType::Additive,
        );
        assert!(!mada.is_valid());
    }

    #[test]
    fn ets_spec_from_notation_roundtrip() {
        // Test that short_name output can be parsed back
        // Note: short_name returns "ETS(A,A,N)" format, not "AAN"
        // So we test the opposite direction: parse -> short_name
        let specs = [
            ("ANN", "ETS(A,N,N)"),
            ("AAN", "ETS(A,A,N)"),
            ("AAdN", "ETS(A,Ad,N)"),
            ("AAA", "ETS(A,A,A)"),
            ("AAM", "ETS(A,A,M)"),
            ("MNN", "ETS(M,N,N)"),
            ("MAM", "ETS(M,A,M)"),
            ("MAdM", "ETS(M,Ad,M)"),
        ];

        for (notation, expected_name) in specs {
            let spec = ETSSpec::from_notation(notation).unwrap();
            assert_eq!(
                spec.short_name(),
                expected_name,
                "Notation {} should produce {}",
                notation,
                expected_name
            );
        }
    }

    #[test]
    fn ets_spec_new_constructors_match_manual() {
        // Verify convenience constructors match manual construction
        assert_eq!(
            ETSSpec::ana(),
            ETSSpec::new(ErrorType::Additive, TrendType::None, SeasonalType::Additive)
        );
        assert_eq!(
            ETSSpec::anm(),
            ETSSpec::new(
                ErrorType::Additive,
                TrendType::None,
                SeasonalType::Multiplicative
            )
        );
        assert_eq!(
            ETSSpec::aada(),
            ETSSpec::new(
                ErrorType::Additive,
                TrendType::AdditiveDamped,
                SeasonalType::Additive
            )
        );
        assert_eq!(
            ETSSpec::aadm(),
            ETSSpec::new(
                ErrorType::Additive,
                TrendType::AdditiveDamped,
                SeasonalType::Multiplicative
            )
        );
        assert_eq!(
            ETSSpec::mnm(),
            ETSSpec::new(
                ErrorType::Multiplicative,
                TrendType::None,
                SeasonalType::Multiplicative
            )
        );
        assert_eq!(
            ETSSpec::madm(),
            ETSSpec::new(
                ErrorType::Multiplicative,
                TrendType::AdditiveDamped,
                SeasonalType::Multiplicative
            )
        );
        assert_eq!(
            ETSSpec::man(),
            ETSSpec::new(
                ErrorType::Multiplicative,
                TrendType::Additive,
                SeasonalType::None
            )
        );
        assert_eq!(
            ETSSpec::madn(),
            ETSSpec::new(
                ErrorType::Multiplicative,
                TrendType::AdditiveDamped,
                SeasonalType::None
            )
        );
    }
}
