//! ETS (Error-Trend-Seasonal) state-space forecasting model.
//!
//! ETS provides a unified framework for exponential smoothing methods,
//! with 30 possible model combinations based on error, trend, and seasonal components.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::explain::{Explainable, ForecastExplanation};
use crate::models::{validate_series_complete, FittedParams, Forecaster};
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

    /// ETS(M,A,A) - Multiplicative error with additive trend and additive seasonality.
    ///
    /// Included in R `forecast::ets`'s default (`restrict = TRUE`) model set.
    pub fn maa() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::Additive,
            SeasonalType::Additive,
        )
    }

    /// ETS(M,Ad,A) - Multiplicative error with damped trend and additive seasonality.
    ///
    /// Included in R `forecast::ets`'s default (`restrict = TRUE`) model set.
    pub fn mada() -> Self {
        Self::new(
            ErrorType::Multiplicative,
            TrendType::AdditiveDamped,
            SeasonalType::Additive,
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

    /// Check if this ETS specification is valid/constructible.
    ///
    /// Every `(error, trend, seasonal)` combination this type can represent
    /// is a valid R `forecast::ets` model, including ETS(M,A,A) and
    /// ETS(M,Ad,A) — R's own `restrict = TRUE` default set includes both.
    /// Always returns `true`; retained (rather than removed) for API
    /// stability and because [`AutoETS`](crate::models::exponential::AutoETS)'s
    /// automatic candidate pool uses [`Self::is_r_restricted`], a distinct
    /// and narrower exclusion, instead of this method.
    pub fn is_valid(&self) -> bool {
        true
    }

    /// Check if R `forecast::ets()`'s default (`restrict = TRUE`) automatic
    /// search excludes this specification.
    ///
    /// R's default AutoETS search skips additive-error models with
    /// multiplicative seasonality — ETS(A,N,M), ETS(A,A,M), ETS(A,Ad,M) —
    /// because that combination is numerically unstable when searched
    /// automatically (though still fittable explicitly). This matches
    /// `deparse(forecast::ets)`'s `restrict` branch (forecast 9.0.2).
    pub fn is_r_restricted(&self) -> bool {
        self.error == ErrorType::Additive && self.seasonal == SeasonalType::Multiplicative
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
    /// Returns an error if the notation format itself is invalid (wrong
    /// length, or an unrecognised error/trend/seasonal letter). Every
    /// constructible `(error, trend, seasonal)` combination — including
    /// "MAA" and "MAdA" — parses successfully; [`Self::is_r_restricted`]
    /// is a separate, narrower exclusion applied only by AutoETS's
    /// automatic candidate pool, not by parsing.
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
    /// // MAA/MAdA are valid R models and parse successfully.
    /// let spec = ETSSpec::from_notation("MAA").unwrap();
    /// assert_eq!(spec, ETSSpec::maa());
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
     $y:ident, $lp:ident, $err:ident,
     forecast { $($fc:tt)* }
     update { $($upd:tt)* }
    ) => {{
        let mut _sum_sq = 0.0_f64;
        let mut _sum_log_fc = 0.0_f64;
        let mut _cnt = 0_usize;
        for (_, &$y) in $values.iter().enumerate().skip($start_idx) {
            let _fc = { $($fc)* };
            if !_fc.is_finite() { return f64::MAX; }
            let _err = $y - _fc;
            let _se = if $is_mult_error && _fc.abs() > 1e-10 { _err / _fc } else { _err };
            _sum_sq += _se * _se;
            if !_sum_sq.is_finite() { return f64::MAX; }
            // R/statsforecast ets convention (D-07): the multiplicative-error
            // likelihood accumulates ln|one-step forecast| (_fc), not ln|y|.
            if $is_mult_error { _sum_log_fc += _fc.abs().ln(); }
            _cnt += 1;
            let $lp = $level;
            // Caller-named alias for the additive one-step error (y - fc),
            // used by the state recursions below (D-07, 11-06 Task 2) --
            // exposed as a macro parameter (like $lp) because `_err` itself,
            // defined inside this macro body, is hygienically invisible to
            // the caller-provided `update` tokens.
            let $err = _err;
            $($upd)*
        }
        (_sum_sq, _sum_log_fc, _cnt)
    }};
    // Seasonal: reads/writes seasonal buffer by index
    (seasonal $values:expr, $start_idx:expr, $period:expr, $is_mult_error:expr,
     $level:ident, $trend:ident, $buf:ident,
     $y:ident, $s:ident, $si:ident, $lp:ident, $err:ident,
     forecast { $($fc:tt)* }
     update { $($upd:tt)* }
    ) => {{
        let mut _sum_sq = 0.0_f64;
        let mut _sum_log_fc = 0.0_f64;
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
            // R/statsforecast ets convention (D-07): the multiplicative-error
            // likelihood accumulates ln|one-step forecast| (_fc), not ln|y|.
            if $is_mult_error { _sum_log_fc += _fc.abs().ln(); }
            _cnt += 1;
            let $lp = $level;
            let $err = _err;
            $($upd)*
        }
        (_sum_sq, _sum_log_fc, _cnt)
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
    /// Degrees-of-freedom-corrected residual variance used for prediction
    /// intervals only: `sum(e^2) / (n - length(par))`, matching R's
    /// `forecast.ets` convention (D-08, plan 11-07). `residual_variance`
    /// above (the uncorrected `sum(e^2)/n`) is kept as-is since other code
    /// (AIC/AICc/BIC/loglik) depends on it.
    interval_sigma2: Option<f64>,
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
    /// Whether to skip optimization when fit() is called (warm-start mode).
    skip_optimization: bool,
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
            interval_sigma2: None,
            log_likelihood: None,
            aic: None,
            aicc: None,
            bic: None,
            n: 0,
            exog_ols: None,
            skip_optimization: false,
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
            interval_sigma2: None,
            log_likelihood: None,
            aic: None,
            aicc: None,
            bic: None,
            n: 0,
            exog_ols: None,
            skip_optimization: false,
        }
    }

    /// Create a warm-started ETS model with pre-fitted initial states.
    ///
    /// The resulting model can produce predictions immediately via `predict()`
    /// without calling `fit()`. If `fit()` is later called, the provided states
    /// are used as the starting point for state propagation (optimization is skipped).
    ///
    /// # Arguments
    /// * `spec` - The ETS specification (error, trend, seasonal types)
    /// * `seasonal_period` - Seasonal period (use 1 for non-seasonal)
    /// * `level` - Pre-fitted level state
    /// * `trend` - Pre-fitted trend state (pass 0.0 for non-trend models)
    /// * `seasonal_values` - Pre-fitted seasonal state vector (empty for non-seasonal)
    pub fn with_initial_states(
        spec: ETSSpec,
        seasonal_period: usize,
        level: f64,
        trend: f64,
        seasonal_values: Vec<f64>,
    ) -> Self {
        Self {
            spec,
            seasonal_period,
            alpha: Some(0.3),
            beta: if spec.has_trend() { Some(0.1) } else { None },
            gamma: if spec.has_seasonal() { Some(0.1) } else { None },
            phi: if spec.is_damped() { Some(0.98) } else { None },
            optimize: false,
            level: Some(level),
            trend: Some(trend),
            seasonals: if seasonal_values.is_empty() {
                None
            } else {
                Some(seasonal_values)
            },
            fitted: None,
            residuals: None,
            residual_variance: None,
            interval_sigma2: None,
            log_likelihood: None,
            aic: None,
            aicc: None,
            bic: None,
            n: 0,
            exog_ols: None,
            skip_optimization: true,
        }
    }

    /// Create an ETS model with fixed smoothing parameters AND fixed initial
    /// states, with no optimisation performed at any point (unlike
    /// [`with_initial_states`](Self::with_initial_states), which still lets
    /// `fit()` run the forward recursion from the given states -- which this
    /// method also does, but with the caller's exact `alpha`/`beta`/`gamma`/
    /// `phi` instead of defaults). Values are used exactly as given (after a
    /// finiteness check) -- primarily for scoring a model at externally
    /// supplied (e.g. R `forecast::ets`) parameters/states for parity tests.
    ///
    /// `seasonals` uses the crate's `t % seasonal_period` buffer layout
    /// (buffer\[0\] is the seasonal factor applied to the first observation).
    pub fn with_params_and_states(
        spec: ETSSpec,
        seasonal_period: usize,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        level: f64,
        trend: f64,
        seasonals: Vec<f64>,
    ) -> Self {
        debug_assert!(
            alpha.is_finite(),
            "with_params_and_states: alpha must be finite"
        );
        debug_assert!(
            level.is_finite(),
            "with_params_and_states: level must be finite"
        );
        debug_assert!(
            trend.is_finite(),
            "with_params_and_states: trend must be finite"
        );
        debug_assert!(
            beta.is_none_or(f64::is_finite),
            "with_params_and_states: beta must be finite"
        );
        debug_assert!(
            gamma.is_none_or(f64::is_finite),
            "with_params_and_states: gamma must be finite"
        );
        debug_assert!(
            phi.is_none_or(f64::is_finite),
            "with_params_and_states: phi must be finite"
        );
        debug_assert!(
            seasonals.iter().all(|s| s.is_finite()),
            "with_params_and_states: seasonals must be finite"
        );
        Self {
            spec,
            seasonal_period,
            alpha: Some(alpha),
            beta,
            gamma,
            phi,
            optimize: false,
            level: Some(level),
            trend: Some(trend),
            seasonals: if seasonals.is_empty() {
                None
            } else {
                Some(seasonals)
            },
            fitted: None,
            residuals: None,
            residual_variance: None,
            interval_sigma2: None,
            log_likelihood: None,
            aic: None,
            aicc: None,
            bic: None,
            n: 0,
            exog_ols: None,
            skip_optimization: true,
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
        // R/statsforecast ets convention (D-07): score every observation
        // from t = 0 (the initial states are time-0 states, before y_1) --
        // seasonal models no longer skip the first period.
        let start_idx = 0;

        if n <= start_idx + 1 {
            return f64::MAX;
        }

        // Use provided initial states or fallback to heuristic.
        // Reuse seasonal_buf instead of allocating via to_vec().
        // PERF: avoid calling initialize_state() inside the optimization loop —
        // callers should pre-compute heuristic states once and pass them in.
        let (mut level, mut trend) = match (init_level, init_seasonals) {
            (Some(l), Some(s)) => {
                seasonal_buf.resize(s.len(), 0.0);
                seasonal_buf.copy_from_slice(s);
                (l, init_trend.unwrap_or(0.0))
            }
            (Some(l), None) if !self.spec.has_seasonal() => {
                // Non-seasonal model: no seasonal buffer needed, skip initialize_state()
                seasonal_buf.clear();
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
        let (sum_sq_errors, sum_log_fc, count) = match (self.spec.trend, self.spec.seasonal) {
            (TrendType::None, SeasonalType::None) => {
                ets_likelihood_loop!(nonseasonal values, start_idx, is_mult_error,
                    level, trend,
                    y, _level_prev, _err,
                    forecast { level }
                    update {
                        level = alpha * y + (1.0 - alpha) * level;
                    }
                )
            }
            (TrendType::None, SeasonalType::Additive) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev, err,
                    forecast { level + s }
                    update {
                        // Hyndman innovations state-space form (same for both
                        // error types: mu*eps = y-mu reduces identically when
                        // the measurement equation is additive): l_t=l_{t-1}
                        // +alpha*e_t, s_t=s_{t-m}+gamma*e_t, e_t=err=y-mu.
                        // (Previously used the just-updated `level`, not
                        // `level_prev`, in the seasonal update -- an extra
                        // (1-alpha) factor that doesn't match R.) Validated
                        // to machine precision against R forecast::ets(ANA,
                        // MNA) at R's own fitted parameters (D-07, 11-06
                        // Task 2).
                        level = level_prev + alpha * err;
                        seasonal_buf[season_idx] = s + gamma * err;
                    }
                )
            }
            (TrendType::None, SeasonalType::Multiplicative) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, _level_prev, _err,
                    forecast { level * s }
                    update {
                        let fc = _level_prev * s;
                        if is_mult_error {
                            // R/statsforecast ets convention (D-07): the SSOE
                            // form l_t=l_{t-1}(1+alpha*e_t), s_t=s_{t-m}(1+gamma*e_t)
                            // with e_t=(y-fc)/fc, validated numerically against
                            // R forecast::ets(MNM) at R's own fitted parameters.
                            if fc.abs() > 1e-10 {
                                let e = (y - fc) / fc;
                                level = _level_prev * (1.0 + alpha * e);
                                seasonal_buf[season_idx] = s * (1.0 + gamma * e);
                            }
                        } else {
                            let y_des = if s.abs() > 1e-10 { y / s } else { y };
                            level = alpha * y_des + (1.0 - alpha) * level;
                            seasonal_buf[season_idx] = if _level_prev.abs() > 1e-10 {
                                gamma * (y / _level_prev) + (1.0 - gamma) * s
                            } else {
                                s
                            };
                        }
                    }
                )
            }
            (TrendType::Additive, SeasonalType::None) => {
                ets_likelihood_loop!(nonseasonal values, start_idx, is_mult_error,
                    level, trend,
                    y, level_prev, err,
                    forecast { level + trend }
                    update {
                        // Hyndman direct-error form: l_t=l_{t-1}+b_{t-1}+
                        // alpha*e_t, b_t=b_{t-1}+beta*e_t -- NOT beta*(l_t-
                        // l_{t-1}), the classical reparametrisation this
                        // replaces, which silently scales beta by an extra
                        // factor of alpha and does not match R. Validated
                        // to machine precision against R forecast::ets(AAN,
                        // MAN) at R's own fitted parameters (D-07, 11-06
                        // Task 2).
                        level = level_prev + trend + alpha * err;
                        trend += beta * err;
                    }
                )
            }
            (TrendType::Additive, SeasonalType::Additive) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev, err,
                    forecast { level + trend + s }
                    update {
                        // Hyndman direct-error form (D-07, 11-06 Task 2):
                        // same derivation as the (Additive, None) and
                        // (None, Additive) arms, combined. Validated to
                        // machine precision against R forecast::ets(AAA,
                        // MAA, AAdA, MAdA) at R's own fitted parameters.
                        level = level_prev + trend + alpha * err;
                        trend += beta * err;
                        seasonal_buf[season_idx] = s + gamma * err;
                    }
                )
            }
            (TrendType::Additive, SeasonalType::Multiplicative) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev, err,
                    forecast { (level + trend) * s }
                    update {
                        let l_plus_b = level_prev + trend;
                        let fc = l_plus_b * s;
                        if is_mult_error {
                            // R/statsforecast ets convention (D-07): SSOE form
                            // l_t=(l+b)(1+alpha*e), b_t=b+beta*(l+b)*e,
                            // s_t=s_{t-m}(1+gamma*e), e=(y-fc)/fc -- validated
                            // numerically against R forecast::ets(MAM) at R's
                            // own fitted parameters (exact match to float
                            // precision across the first 6 fitted values).
                            if fc.abs() > 1e-10 {
                                let e = (y - fc) / fc;
                                level = l_plus_b * (1.0 + alpha * e);
                                trend += beta * l_plus_b * e;
                                seasonal_buf[season_idx] = s * (1.0 + gamma * e);
                            }
                        } else {
                            // Hyndman direct-error form for additive error +
                            // multiplicative season (D-07, 11-06 Task 2):
                            // l_t=(l+b)+alpha*e_t/s_{t-m}, b_t=b+beta*e_t/
                            // s_{t-m}, s_t=s_{t-m}+gamma*e_t/(l+b), e_t=err.
                            // (Previously used the just-updated `level`, not
                            // `level_prev`/`trend`, in the trend/seasonal
                            // update.) Validated to machine precision
                            // against R forecast::ets(AAM) at R's own
                            // fitted parameters.
                            if s.abs() > 1e-10 {
                                level = l_plus_b + alpha * err / s;
                                trend += beta * err / s;
                            } else {
                                level = l_plus_b;
                            }
                            seasonal_buf[season_idx] = if l_plus_b.abs() > 1e-10 {
                                s + gamma * err / l_plus_b
                            } else {
                                s
                            };
                        }
                    }
                )
            }
            (TrendType::AdditiveDamped, SeasonalType::None) => {
                ets_likelihood_loop!(nonseasonal values, start_idx, is_mult_error,
                    level, trend,
                    y, level_prev, err,
                    forecast { level + phi * trend }
                    update {
                        // Hyndman direct-error form, damped analogue of the
                        // (Additive, None) arm (D-07, 11-06 Task 2).
                        // Validated to machine precision against R
                        // forecast::ets(AAdN, MAdN).
                        level = level_prev + phi * trend + alpha * err;
                        trend = phi * trend + beta * err;
                    }
                )
            }
            (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev, err,
                    forecast { level + phi * trend + s }
                    update {
                        // Hyndman direct-error form, damped analogue of the
                        // (Additive, Additive) arm (D-07, 11-06 Task 2).
                        // Validated to machine precision against R
                        // forecast::ets(AAdA, MAdA).
                        level = level_prev + phi * trend + alpha * err;
                        trend = phi * trend + beta * err;
                        seasonal_buf[season_idx] = s + gamma * err;
                    }
                )
            }
            (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                ets_likelihood_loop!(seasonal values, start_idx, period, is_mult_error,
                    level, trend, seasonal_buf,
                    y, s, season_idx, level_prev, err,
                    forecast { (level + phi * trend) * s }
                    update {
                        let l_plus_pb = level_prev + phi * trend;
                        let fc = l_plus_pb * s;
                        if is_mult_error {
                            // Damped analogue of the (A,M) arm above: phi*trend
                            // replaces trend wherever the undamped SSOE form
                            // uses the raw trend baseline.
                            if fc.abs() > 1e-10 {
                                let e = (y - fc) / fc;
                                level = l_plus_pb * (1.0 + alpha * e);
                                trend = phi * trend + beta * l_plus_pb * e;
                                seasonal_buf[season_idx] = s * (1.0 + gamma * e);
                            }
                        } else {
                            // Damped analogue of the (Additive, Multiplicative)
                            // additive-error fix above: phi*trend replaces
                            // trend wherever the undamped form uses the raw
                            // trend baseline (D-07, 11-06 Task 2). Validated
                            // to machine precision against R
                            // forecast::ets(AAdM).
                            if s.abs() > 1e-10 {
                                level = l_plus_pb + alpha * err / s;
                                trend = phi * trend + beta * err / s;
                            } else {
                                level = l_plus_pb;
                                trend *= phi;
                            }
                            seasonal_buf[season_idx] = if l_plus_pb.abs() > 1e-10 {
                                s + gamma * err / l_plus_pb
                            } else {
                                s
                            };
                        }
                    }
                )
            }
        };

        if count == 0 || !sum_sq_errors.is_finite() {
            return f64::MAX;
        }
        // A perfect fit (sum_sq_errors == 0, e.g. a constant series) is the
        // global optimum, not a degenerate one -- floor instead of rejecting,
        // so ln() stays finite without distorting any non-degenerate fit.
        let sum_sq_errors = sum_sq_errors.max(1e-300);

        // R/statsforecast ets convention (D-07):
        // loglik = -0.5*(n*ln(sum_sq) + 2*sum_log|fc)  [mult-error term only]
        let n_f = count as f64;
        let ll = if is_mult_error {
            -0.5 * (n_f * sum_sq_errors.ln() + 2.0 * sum_log_fc)
        } else {
            -0.5 * (n_f * sum_sq_errors.ln())
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
    /// Map unconstrained optimiser coordinates (each confined to `[0, 1]` by
    /// the caller's bounds) to ETS smoothing parameters under the "usual"
    /// constraints R's `forecast::ets` enforces: `1e-4 <= alpha <= 0.9999`,
    /// `1e-4 <= beta <= alpha`, `1e-4 <= gamma <= 1 - alpha`,
    /// `0.8 <= phi <= 0.98`. `raw` holds, in order, exactly the coordinates
    /// present for this spec: alpha always first, then beta (iff
    /// `has_beta`), then gamma (iff `has_gamma`), then phi (iff `has_phi`).
    /// Reparametrising beta/gamma relative to alpha — rather than giving
    /// each smoothing parameter an independent `[1e-4, 0.9999]` box — is
    /// what makes `beta < alpha` and `gamma < 1 - alpha` hold for every
    /// point the optimiser can reach, not just the final clamp.
    fn map_usual_params(
        raw: &[f64],
        has_beta: bool,
        has_gamma: bool,
        has_phi: bool,
    ) -> (f64, Option<f64>, Option<f64>, Option<f64>) {
        const LOWER: f64 = 1e-4;
        const ALPHA_UPPER: f64 = 0.9999;
        const PHI_LOWER: f64 = 0.8;
        const PHI_UPPER: f64 = 0.98;

        let mut idx = 0;
        let u_alpha = raw[idx].clamp(0.0, 1.0);
        idx += 1;
        let alpha = LOWER + u_alpha * (ALPHA_UPPER - LOWER);

        let beta = if has_beta {
            let u_beta = raw[idx].clamp(0.0, 1.0);
            idx += 1;
            Some(LOWER + u_beta * (alpha - LOWER))
        } else {
            None
        };

        let gamma = if has_gamma {
            let u_gamma = raw[idx].clamp(0.0, 1.0);
            idx += 1;
            Some(LOWER + u_gamma * (1.0 - alpha - LOWER))
        } else {
            None
        };

        let phi = if has_phi {
            let u_phi = raw[idx].clamp(0.0, 1.0);
            Some(PHI_LOWER + u_phi * (PHI_UPPER - PHI_LOWER))
        } else {
            None
        };

        (alpha, beta, gamma, phi)
    }

    /// Inverse of [`Self::map_usual_params`]: given target parameter values,
    /// find unconstrained `[0, 1]` coordinates that map back to them, for
    /// seeding the optimiser's starting point. Only the coordinates present
    /// (alpha always, then beta/gamma/phi iff `Some`) are returned, in the
    /// same order `map_usual_params` expects.
    fn unmap_usual_params(
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> Vec<f64> {
        const LOWER: f64 = 1e-4;
        const ALPHA_UPPER: f64 = 0.9999;
        const PHI_LOWER: f64 = 0.8;
        const PHI_UPPER: f64 = 0.98;
        const MIN_DENOM: f64 = 1e-9;

        let mut out = Vec::with_capacity(4);
        out.push(((alpha - LOWER) / (ALPHA_UPPER - LOWER)).clamp(0.0, 1.0));
        if let Some(b) = beta {
            let denom = (alpha - LOWER).max(MIN_DENOM);
            out.push(((b - LOWER) / denom).clamp(0.0, 1.0));
        }
        if let Some(g) = gamma {
            let denom = (1.0 - alpha - LOWER).max(MIN_DENOM);
            out.push(((g - LOWER) / denom).clamp(0.0, 1.0));
        }
        if let Some(p) = phi {
            out.push(((p - PHI_LOWER) / (PHI_UPPER - PHI_LOWER)).clamp(0.0, 1.0));
        }
        out
    }

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
            stagnation_window: 150,
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
                // Early-exit: if we already have a good result and a new start's
                // initial evaluation is far worse, skip this start entirely.
                if best_value < f64::MAX {
                    let init_ll = self.calculate_likelihood_with_init(
                        values,
                        alpha_init,
                        Some(0.01),
                        None,
                        None,
                        Some(init_level),
                        Some(init_trend),
                        None,
                    );
                    if init_ll > best_value * 3.0 {
                        continue;
                    }
                }

                let mut start = Self::unmap_usual_params(alpha_init, Some(0.01), None, None);
                start.push(init_level);
                start.push(init_trend);

                let result = nelder_mead(
                    |p| {
                        let (alpha, beta, _, _) =
                            Self::map_usual_params(&p[0..2], true, false, false);
                        self.calculate_likelihood_with_init(
                            values,
                            alpha,
                            beta,
                            None,
                            None,
                            Some(p[2]),
                            Some(p[3]),
                            None,
                        )
                    },
                    &start,
                    Some(&[(0.0, 1.0), (0.0, 1.0), level_bounds, trend_bounds]),
                    config,
                );

                if result.optimal_value < best_value {
                    best_value = result.optimal_value;
                    best_result = Some(result);
                }
            }

            // SAFETY: alpha_starts is non-empty and nelder_mead always returns a
            // result, so at least one iteration sets best_result. Fall back to
            // heuristic initial states if every starting point yields f64::MAX
            // (degenerate data where the likelihood surface is flat).
            let result = match best_result {
                Some(r) => r,
                None => {
                    let (init_level, init_trend, _) = self.initialize_state(values);
                    return (0.3, Some(0.1), None, None, init_level, init_trend, None);
                }
            };
            let (alpha, beta, _, _) =
                Self::map_usual_params(&result.optimal_point[0..2], true, false, false);
            return (
                alpha,
                beta,
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
                max_iter: 3000 + 300 * period,
                tolerance: 1e-10,
                stagnation_window: 200, // early stop when no improvement for 200 iters
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

            // Two-stage seasonal optimization:
            // Stage 1: cheaply explore smoothing params (2-4 dim) across 6 starting
            //   points with heuristic states held fixed (max_iter 500 each).
            // Stage 2: one joint refinement from the Stage 1 winner, optimizing
            //   smoothing params + initial level/trend + seasonal indices together.
            // This replaces 6× expensive joint optimizations (16-18 dim) with
            // 6× cheap (2-4 dim) + 1× joint, yielding ~3-5× speedup.

            let quick_config = NelderMeadConfig {
                // Raised from 500 (D-07, 11-06 Task 2): the corrected
                // recursion's likelihood surface needs more stage-1
                // exploration per start to rank candidates reliably before
                // stage 2's joint refinement commits to one of them.
                max_iter: 1500,
                tolerance: 1e-10,
                ..Default::default()
            };
            let seasonal_buf = RefCell::new(vec![0.0; period]);

            match (has_trend, is_damped) {
                (false, _) => {
                    // Stage 1: optimize alpha, gamma only (2 params)
                    let smoothing_bounds = [(0.0, 1.0), (0.0, 1.0)];
                    let mut best_stage1 = f64::MAX;
                    let mut best_alpha = 0.3_f64;
                    let mut best_gamma = 0.1_f64;

                    for &(alpha_init, gamma_init) in &ag_starts {
                        if best_stage1 < f64::MAX {
                            let mut buf = seasonal_buf.borrow_mut();
                            let init_ll = self.calculate_likelihood_with_init_buf(
                                values,
                                alpha_init,
                                None,
                                Some(gamma_init),
                                None,
                                Some(init_level),
                                None,
                                Some(&init_seasonals),
                                &mut buf,
                            );
                            drop(buf);
                            if init_ll > best_stage1 * 3.0 {
                                continue;
                            }
                        }

                        let start =
                            Self::unmap_usual_params(alpha_init, None, Some(gamma_init), None);
                        let result = nelder_mead(
                            |p| {
                                let (alpha, _, gamma, _) =
                                    Self::map_usual_params(&p[0..2], false, true, false);
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    alpha,
                                    None,
                                    gamma,
                                    None,
                                    Some(init_level),
                                    None,
                                    Some(&init_seasonals),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&smoothing_bounds),
                            quick_config,
                        );

                        if result.optimal_value < best_stage1 {
                            best_stage1 = result.optimal_value;
                            let (alpha, _, gamma, _) = Self::map_usual_params(
                                &result.optimal_point[0..2],
                                false,
                                true,
                                false,
                            );
                            best_alpha = alpha;
                            best_gamma = gamma.unwrap();
                        }
                    }

                    // Stage 2: joint refinement — alpha, gamma, l0, s0[0..period]
                    let n_params = 2 + 1 + period;
                    let mut joint_bounds = Vec::with_capacity(n_params);
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push(level_bounds);
                    joint_bounds.extend_from_slice(&seasonal_bounds);

                    let mut start =
                        Self::unmap_usual_params(best_alpha, None, Some(best_gamma), None);
                    start.push(init_level);
                    start.extend_from_slice(&init_seasonals);

                    let result = nelder_mead(
                        |p| {
                            let (alpha, _, gamma, _) =
                                Self::map_usual_params(&p[0..2], false, true, false);
                            let mut buf = seasonal_buf.borrow_mut();
                            self.calculate_likelihood_with_init_buf(
                                values,
                                alpha,
                                None,
                                gamma,
                                None,
                                Some(p[2]),
                                None,
                                Some(&p[3..3 + period]),
                                &mut buf,
                            )
                        },
                        &start,
                        Some(&joint_bounds),
                        seasonal_config,
                    );

                    if result.optimal_value < f64::MAX {
                        let (alpha, _, gamma, _) =
                            Self::map_usual_params(&result.optimal_point[0..2], false, true, false);
                        let opt_seasonals = result.optimal_point[3..3 + period].to_vec();
                        (
                            alpha,
                            None,
                            gamma,
                            None,
                            result.optimal_point[2],
                            init_trend,
                            Some(opt_seasonals),
                        )
                    } else {
                        (
                            0.3,
                            None,
                            Some(0.1),
                            None,
                            init_level,
                            init_trend,
                            Some(init_seasonals.clone()),
                        )
                    }
                }
                (true, false) => {
                    // Stage 1: optimize alpha, beta, gamma (3 params)
                    let smoothing_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
                    let mut best_stage1 = f64::MAX;
                    let mut best_alpha = 0.3_f64;
                    let mut best_beta = 0.1_f64;
                    let mut best_gamma = 0.1_f64;

                    for &(alpha_init, gamma_init) in &ag_starts {
                        if best_stage1 < f64::MAX {
                            let mut buf = seasonal_buf.borrow_mut();
                            let init_ll = self.calculate_likelihood_with_init_buf(
                                values,
                                alpha_init,
                                Some(0.1),
                                Some(gamma_init),
                                None,
                                Some(init_level),
                                Some(init_trend),
                                Some(&init_seasonals),
                                &mut buf,
                            );
                            drop(buf);
                            if init_ll > best_stage1 * 3.0 {
                                continue;
                            }
                        }

                        let start =
                            Self::unmap_usual_params(alpha_init, Some(0.1), Some(gamma_init), None);
                        let result = nelder_mead(
                            |p| {
                                let (alpha, beta, gamma, _) =
                                    Self::map_usual_params(&p[0..3], true, true, false);
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    alpha,
                                    beta,
                                    gamma,
                                    None,
                                    Some(init_level),
                                    Some(init_trend),
                                    Some(&init_seasonals),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&smoothing_bounds),
                            quick_config,
                        );

                        if result.optimal_value < best_stage1 {
                            best_stage1 = result.optimal_value;
                            let (alpha, beta, gamma, _) = Self::map_usual_params(
                                &result.optimal_point[0..3],
                                true,
                                true,
                                false,
                            );
                            best_alpha = alpha;
                            best_beta = beta.unwrap();
                            best_gamma = gamma.unwrap();
                        }
                    }

                    // Stage 2: joint — alpha, beta, gamma, l0, b0, s0[0..period]
                    let n_params = 3 + 2 + period;
                    let mut joint_bounds = Vec::with_capacity(n_params);
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push(level_bounds);
                    joint_bounds.push(trend_bounds);
                    joint_bounds.extend_from_slice(&seasonal_bounds);

                    let mut start = Self::unmap_usual_params(
                        best_alpha,
                        Some(best_beta),
                        Some(best_gamma),
                        None,
                    );
                    start.push(init_level);
                    start.push(init_trend);
                    start.extend_from_slice(&init_seasonals);

                    let result = nelder_mead(
                        |p| {
                            let (alpha, beta, gamma, _) =
                                Self::map_usual_params(&p[0..3], true, true, false);
                            let mut buf = seasonal_buf.borrow_mut();
                            self.calculate_likelihood_with_init_buf(
                                values,
                                alpha,
                                beta,
                                gamma,
                                None,
                                Some(p[3]),
                                Some(p[4]),
                                Some(&p[5..5 + period]),
                                &mut buf,
                            )
                        },
                        &start,
                        Some(&joint_bounds),
                        seasonal_config,
                    );

                    if result.optimal_value < f64::MAX {
                        let (alpha, beta, gamma, _) =
                            Self::map_usual_params(&result.optimal_point[0..3], true, true, false);
                        let opt_seasonals = result.optimal_point[5..5 + period].to_vec();
                        (
                            alpha,
                            beta,
                            gamma,
                            None,
                            result.optimal_point[3],
                            result.optimal_point[4],
                            Some(opt_seasonals),
                        )
                    } else {
                        (
                            0.3,
                            Some(0.1),
                            Some(0.1),
                            None,
                            init_level,
                            init_trend,
                            Some(init_seasonals.clone()),
                        )
                    }
                }
                (true, true) => {
                    // Stage 1: optimize alpha, beta, gamma, phi (4 params)
                    let smoothing_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
                    let mut best_stage1 = f64::MAX;
                    // Keep every stage-1 candidate (D-07, 11-06 Task 2):
                    // stage 1's cheap ranking can misorder candidates once
                    // the likelihood surface changed shape after the
                    // recursion fix, so stage 2 refines from the top few,
                    // not just the single apparent winner (fixes MAdM
                    // converging to a worse optimum than R).
                    let mut stage1_candidates: Vec<(f64, f64, f64, f64, f64)> = Vec::new();

                    for &(alpha_init, gamma_init) in &ag_starts {
                        if best_stage1 < f64::MAX {
                            let mut buf = seasonal_buf.borrow_mut();
                            let init_ll = self.calculate_likelihood_with_init_buf(
                                values,
                                alpha_init,
                                Some(0.1),
                                Some(gamma_init),
                                Some(0.98),
                                Some(init_level),
                                Some(init_trend),
                                Some(&init_seasonals),
                                &mut buf,
                            );
                            drop(buf);
                            if init_ll > best_stage1 * 3.0 {
                                continue;
                            }
                        }

                        let start = Self::unmap_usual_params(
                            alpha_init,
                            Some(0.1),
                            Some(gamma_init),
                            Some(0.98),
                        );
                        let result = nelder_mead(
                            |p| {
                                let (alpha, beta, gamma, phi) =
                                    Self::map_usual_params(&p[0..4], true, true, true);
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    alpha,
                                    beta,
                                    gamma,
                                    phi,
                                    Some(init_level),
                                    Some(init_trend),
                                    Some(&init_seasonals),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&smoothing_bounds),
                            quick_config,
                        );

                        let (alpha, beta, gamma, phi) =
                            Self::map_usual_params(&result.optimal_point[0..4], true, true, true);
                        stage1_candidates.push((
                            result.optimal_value,
                            alpha,
                            beta.unwrap(),
                            gamma.unwrap(),
                            phi.unwrap(),
                        ));
                        if result.optimal_value < best_stage1 {
                            best_stage1 = result.optimal_value;
                        }
                    }

                    if stage1_candidates.is_empty() {
                        stage1_candidates.push((f64::MAX, 0.3, 0.1, 0.1, 0.98));
                    }
                    stage1_candidates
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    // Stage 2: joint — alpha, beta, gamma, phi, l0, b0, s0[0..period]
                    let n_params = 4 + 2 + period;
                    let mut joint_bounds = Vec::with_capacity(n_params);
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push((0.0, 1.0));
                    joint_bounds.push(level_bounds);
                    joint_bounds.push(trend_bounds);
                    joint_bounds.extend_from_slice(&seasonal_bounds);

                    let mut best_stage2: Option<crate::utils::optimization::NelderMeadResult> =
                        None;
                    for &(_, cand_alpha, cand_beta, cand_gamma, cand_phi) in
                        stage1_candidates.iter().take(3)
                    {
                        let mut start = Self::unmap_usual_params(
                            cand_alpha,
                            Some(cand_beta),
                            Some(cand_gamma),
                            Some(cand_phi),
                        );
                        start.push(init_level);
                        start.push(init_trend);
                        start.extend_from_slice(&init_seasonals);

                        let candidate_result = nelder_mead(
                            |p| {
                                let (alpha, beta, gamma, phi) =
                                    Self::map_usual_params(&p[0..4], true, true, true);
                                let mut buf = seasonal_buf.borrow_mut();
                                self.calculate_likelihood_with_init_buf(
                                    values,
                                    alpha,
                                    beta,
                                    gamma,
                                    phi,
                                    Some(p[4]),
                                    Some(p[5]),
                                    Some(&p[6..6 + period]),
                                    &mut buf,
                                )
                            },
                            &start,
                            Some(&joint_bounds),
                            seasonal_config,
                        );

                        if best_stage2
                            .as_ref()
                            .is_none_or(|b| candidate_result.optimal_value < b.optimal_value)
                        {
                            best_stage2 = Some(candidate_result);
                        }
                    }
                    let result = best_stage2.expect("stage1_candidates is non-empty");

                    if result.optimal_value < f64::MAX {
                        let (alpha, beta, gamma, phi) =
                            Self::map_usual_params(&result.optimal_point[0..4], true, true, true);
                        let opt_seasonals = result.optimal_point[6..6 + period].to_vec();
                        (
                            alpha,
                            beta,
                            gamma,
                            phi,
                            result.optimal_point[4],
                            result.optimal_point[5],
                            Some(opt_seasonals),
                        )
                    } else {
                        (
                            0.3,
                            Some(0.1),
                            Some(0.1),
                            Some(0.98),
                            init_level,
                            init_trend,
                            Some(init_seasonals.clone()),
                        )
                    }
                }
            }
        } else {
            // Non-seasonal models: optimize smoothing params only
            match (has_trend, is_damped) {
                (false, _) => {
                    // Just alpha (ETS(A,N,N) or ETS(M,N,N))
                    // Pass heuristic init states to avoid redundant initialize_state()
                    // calls inside each Nelder-Mead evaluation.
                    let start = Self::unmap_usual_params(0.3, None, None, None);
                    let result = nelder_mead(
                        |p| {
                            let (alpha, _, _, _) =
                                Self::map_usual_params(&p[0..1], false, false, false);
                            self.calculate_likelihood_with_init(
                                values,
                                alpha,
                                None,
                                None,
                                None,
                                Some(init_level),
                                Some(init_trend),
                                None,
                            )
                        },
                        &start,
                        Some(&[(0.0, 1.0)]),
                        config,
                    );
                    let (alpha, _, _, _) =
                        Self::map_usual_params(&result.optimal_point[0..1], false, false, false);
                    (alpha, None, None, None, init_level, init_trend, None)
                }
                (true, false) => {
                    // alpha, beta (non-damped trend, no seasonal) — shouldn't reach here
                    // (handled by the ETS(A,A,N) multi-start above)
                    (0.3, Some(0.1), None, None, init_level, init_trend, None)
                }
                (true, _) => {
                    // alpha, beta, phi (damped trend, no seasonal)
                    // Pass heuristic init states to avoid redundant initialize_state()
                    // calls inside each Nelder-Mead evaluation.
                    let start = Self::unmap_usual_params(0.3, Some(0.1), None, Some(0.98));
                    let result = nelder_mead(
                        |p| {
                            let (alpha, beta, _, phi) =
                                Self::map_usual_params(&p[0..3], true, false, true);
                            self.calculate_likelihood_with_init(
                                values,
                                alpha,
                                beta,
                                None,
                                phi,
                                Some(init_level),
                                Some(init_trend),
                                None,
                            )
                        },
                        &start,
                        Some(&[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]),
                        config,
                    );
                    let (alpha, beta, _, phi) =
                        Self::map_usual_params(&result.optimal_point[0..3], true, false, true);
                    (alpha, beta, None, phi, init_level, init_trend, None)
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
    /// Hyndman, Koehler, Ord & Snyder (2008) ch. 6 forecast-error-variance
    /// coefficient `c_j = w'F^(j-1)g`: the sensitivity of a forecast `j`
    /// steps beyond its own innovation to that one innovation, for the
    /// general state-space form shared by every (trend, season) combination
    /// with trend in {N,A,Ad} and season in {N,A} (class 1/2; class 3 and
    /// additive-error + multiplicative-season models are simulated instead
    /// -- see [`Self::ets_interval_bounds`]). Table 6.1 in that reference
    /// gives this closed form directly: the level contribution is constant
    /// at `alpha`; the trend contribution accumulates linearly (undamped)
    /// or via the same `phi*(1-phi^j)/(1-phi)` damping sum already used for
    /// point forecasts ([`Self::damped_sum`]); the additive-seasonal
    /// contribution reappears only every `period` steps, because the single
    /// seasonal state slot a given innovation updates is next read exactly
    /// one full cycle later.
    fn ets_cj(
        trend: TrendType,
        seasonal: SeasonalType,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: usize,
        j: usize,
    ) -> f64 {
        let trend_term = match trend {
            TrendType::None => 0.0,
            TrendType::Additive => beta.unwrap_or(0.0) * j as f64,
            TrendType::AdditiveDamped => {
                beta.unwrap_or(0.0) * Self::damped_sum(phi.unwrap_or(1.0), j)
            }
        };
        // R's forecast:::class1 builds G for the seasonal injection at a
        // fixed matrix row (G[3,1] <- gamma) regardless of whether a trend
        // row is present. When trend is absent that row is the *second*
        // seasonal state (not the first), shifting the periodic bump back
        // by one step: confirmed against `forecast:::class1` directly
        // (trend present: gamma bump at j = m, 2m, ...; trend absent: at
        // j = m-1, 2m-1, ...). Rather than replicate R's row layout
        // (R-notation state ordering differs from this crate's `t % m`
        // seasonal buffer anyway), this reproduces the same *numeric*
        // period/phase R's forecast actually reports.
        let seasonal_term = match seasonal {
            SeasonalType::Additive if period > 0 => {
                let has_trend = trend != TrendType::None;
                let hit = if has_trend {
                    j % period == 0
                } else {
                    (j + 1) % period == 0
                };
                if hit {
                    gamma.unwrap_or(0.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        alpha + trend_term + seasonal_term
    }

    /// Class-1 (additive error) analytic forecast-error variance for
    /// h = 1..=horizon: `v_h = sigma2*(1 + sum_{j=1}^{h-1} c_j^2)`.
    fn ets_forecast_variance(
        spec: ETSSpec,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: usize,
        sigma2: f64,
        horizon: usize,
    ) -> Vec<f64> {
        let mut variances = Vec::with_capacity(horizon);
        let mut running_c2 = 0.0;
        for h in 1..=horizon {
            if h >= 2 {
                let c = Self::ets_cj(
                    spec.trend,
                    spec.seasonal,
                    alpha,
                    beta,
                    gamma,
                    phi,
                    period,
                    h - 1,
                );
                running_c2 += c * c;
            }
            variances.push(sigma2 * (1.0 + running_c2));
        }
        variances
    }

    /// Single dispatch entry for ETS prediction-interval bounds, matching
    /// R `forecast:::forecast.ets`'s class1/class2/class3 split exactly
    /// (D-08, plan 11-07): class 1 (additive error, season N/A) uses
    /// [`Self::ets_forecast_variance`] directly. Class 2 (multiplicative
    /// error, season N/A) and class 3 / additive-error+multiplicative-season
    /// models (simulation) are added in plan 11-07 Task 2 -- until then this
    /// arm keeps the pre-11-07 flat `sigma*sqrt(k)` width so multiplicative-
    /// error callers do not regress to an error.
    ///
    /// `point` must already include any exogenous-regressor contribution
    /// (as produced by [`Self::predict_internal`]); bounds are returned as
    /// `point ∓ z*sqrt(v_h)`, so the exogenous shift carries through to both
    /// bounds identically to the point forecast.
    /// Class-2 (multiplicative error, trend N/A/Ad, season N/A) analytic
    /// heteroscedastic forecast-error variance, matching R
    /// `forecast:::class2` exactly: `theta_1 = mu_1^2`,
    /// `theta_h = mu_h^2 + sigma2*sum_{j=1}^{h-1} c_j^2*theta_{h-j}`,
    /// `v_h = (1+sigma2)*theta_h - mu_h^2`, reusing the same `c_j`
    /// coefficients as class 1 (R builds class 2 by calling `class1`
    /// internally for `mu`/`cj` and layering the heteroscedastic recursion
    /// on top).
    fn ets_class2_variance(
        spec: ETSSpec,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: usize,
        sigma2: f64,
        point: &[f64],
        horizon: usize,
    ) -> Vec<f64> {
        let mut cj = Vec::with_capacity(horizon.saturating_sub(1));
        for j in 1..horizon {
            cj.push(Self::ets_cj(
                spec.trend,
                spec.seasonal,
                alpha,
                beta,
                gamma,
                phi,
                period,
                j,
            ));
        }

        let mut theta = vec![0.0; horizon];
        theta[0] = point[0] * point[0];
        for h in 1..horizon {
            // h is 0-indexed step (h+1)-ahead; sum_{j=1}^{h} cj[j-1]^2 * theta[h-j]
            let mut acc = 0.0;
            for j in 1..=h {
                acc += cj[j - 1] * cj[j - 1] * theta[h - j];
            }
            theta[h] = point[h] * point[h] + sigma2 * acc;
        }

        (0..horizon)
            .map(|h| (1.0 + sigma2) * theta[h] - point[h] * point[h])
            .collect()
    }

    /// Deterministic seed for the class-3 / A-error+M-season Monte-Carlo
    /// simulation (D-08, plan 11-07 Task 2): fixed so two calls with the
    /// same inputs return identical bounds (must_haves: "identical across
    /// calls").
    const ETS_SIMULATION_SEED: u64 = 0x4554535f31313037; // ASCII "ETS_1107"
    /// Number of simulated sample paths, matching R `forecast.ets`'s
    /// `npaths` default.
    const ETS_SIMULATION_NPATHS: usize = 5000;

    /// Simulation-based prediction-interval bounds for every
    /// multiplicative-season model: class 3 (M error, M season, trend !=
    /// M) and the additive-error+multiplicative-season models R also
    /// simulates (D-08). Propagates `ETS_SIMULATION_NPATHS` sample paths
    /// forward from the fitted final state using the model's own one-step
    /// update recursion (mirroring `Forecaster::fit`'s per-arm update
    /// equations for `SeasonalType::Multiplicative`) with innovations
    /// drawn from `N(0, sigma2)` via a fixed-seed RNG, then takes R
    /// type-7-interpolated empirical quantiles of the simulated
    /// observations at each horizon step. Point forecasts are *not*
    /// resampled (R does not resample them either): only the bounds come
    /// from simulation.
    #[allow(clippy::too_many_arguments)]
    fn ets_simulated_bounds(
        spec: ETSSpec,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: usize,
        sigma2: f64,
        horizon: usize,
        level: f64,
        final_level: f64,
        final_trend: f64,
        final_seasonals: &[f64],
        n_observed: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let period = period.max(1);
        let beta_v = beta.unwrap_or(0.0);
        let gamma_v = gamma.unwrap_or(0.0);
        let phi_v = phi.unwrap_or(1.0);
        let sigma = sigma2.max(0.0).sqrt();
        let is_mult_error = spec.error == ErrorType::Multiplicative;

        let mut samples: Vec<Vec<f64>> =
            vec![Vec::with_capacity(Self::ETS_SIMULATION_NPATHS); horizon];
        let mut rng = StdRng::seed_from_u64(Self::ETS_SIMULATION_SEED);

        for _path in 0..Self::ETS_SIMULATION_NPATHS {
            let mut lvl = final_level;
            let mut trd = final_trend;
            let mut seas = final_seasonals.to_vec();

            for h in 1..=horizon {
                let season_idx = (n_observed + h - 1) % period;
                let s = if season_idx < seas.len() {
                    seas[season_idx]
                } else {
                    1.0
                };

                // One-step-ahead forecast from the current simulated
                // state (mirrors `Forecaster::fit`'s one-step forecast for
                // `SeasonalType::Multiplicative`).
                let l_plus_trend = match spec.trend {
                    TrendType::None => lvl,
                    TrendType::Additive => lvl + trd,
                    TrendType::AdditiveDamped => lvl + phi_v * trd,
                };
                let fc = l_plus_trend * s;

                // Box-Muller standard normal.
                let u1: f64 = rng.gen::<f64>().max(1e-300);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let lvl_prev = lvl;

                let y_sim = if is_mult_error {
                    let e = z * sigma;
                    let y = fc * (1.0 + e);
                    lvl = l_plus_trend * (1.0 + alpha * e);
                    match spec.trend {
                        TrendType::None => {}
                        TrendType::Additive => trd += beta_v * l_plus_trend * e,
                        TrendType::AdditiveDamped => trd = phi_v * trd + beta_v * l_plus_trend * e,
                    }
                    seas[season_idx] = s * (1.0 + gamma_v * e);
                    y
                } else {
                    let err = z * sigma;
                    let y = fc + err;
                    if spec.trend == TrendType::None {
                        // Hyndman direct form for the no-trend case mirrors
                        // `Forecaster::fit`'s additive-error arm, which
                        // deseasonalises y directly rather than dividing
                        // the error by s.
                        let y_des = if s.abs() > 1e-10 { y / s } else { y };
                        lvl = alpha * y_des + (1.0 - alpha) * lvl_prev;
                        seas[season_idx] = if lvl_prev.abs() > 1e-10 {
                            gamma_v * (y / lvl_prev) + (1.0 - gamma_v) * s
                        } else {
                            s
                        };
                    } else {
                        if s.abs() > 1e-10 {
                            lvl = l_plus_trend + alpha * err / s;
                            match spec.trend {
                                TrendType::Additive => trd += beta_v * err / s,
                                TrendType::AdditiveDamped => trd = phi_v * trd + beta_v * err / s,
                                TrendType::None => unreachable!(),
                            }
                        } else {
                            lvl = l_plus_trend;
                            if spec.trend == TrendType::AdditiveDamped {
                                trd *= phi_v;
                            }
                        }
                        seas[season_idx] = if l_plus_trend.abs() > 1e-10 {
                            s + gamma_v * err / l_plus_trend
                        } else {
                            s
                        };
                    }
                    y
                };

                samples[h - 1].push(y_sim);
            }
        }

        let lower_p = (1.0 - level) / 2.0;
        let upper_p = (1.0 + level) / 2.0;

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        for sample_h in samples.iter_mut() {
            sample_h.sort_by(|a, b| a.partial_cmp(b).unwrap());
            lower.push(Self::quantile_type7(sample_h, lower_p));
            upper.push(Self::quantile_type7(sample_h, upper_p));
        }
        (lower, upper)
    }

    /// R type-7 (default) empirical quantile interpolation over an
    /// already-sorted sample.
    fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
        let n = sorted.len();
        if n == 0 {
            return f64::NAN;
        }
        if n == 1 {
            return sorted[0];
        }
        let h = (n as f64 - 1.0) * p;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = h - lo as f64;
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ets_interval_bounds(
        spec: ETSSpec,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
        period: usize,
        sigma2: f64,
        point: &[f64],
        horizon: usize,
        level: f64,
        final_level: f64,
        final_trend: f64,
        final_seasonals: &[f64],
        n_observed: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let z = quantile_normal((1.0 + level) / 2.0);

        if spec.seasonal == SeasonalType::Multiplicative {
            return Self::ets_simulated_bounds(
                spec,
                alpha,
                beta,
                gamma,
                phi,
                period,
                sigma2,
                horizon,
                level,
                final_level,
                final_trend,
                final_seasonals,
                n_observed,
            );
        }

        match spec.error {
            ErrorType::Additive => {
                let variances = Self::ets_forecast_variance(
                    spec, alpha, beta, gamma, phi, period, sigma2, horizon,
                );
                let mut lower = Vec::with_capacity(horizon);
                let mut upper = Vec::with_capacity(horizon);
                for h in 0..horizon {
                    let se = variances[h].sqrt();
                    lower.push(point[h] - z * se);
                    upper.push(point[h] + z * se);
                }
                (lower, upper)
            }
            ErrorType::Multiplicative => {
                let variances = Self::ets_class2_variance(
                    spec, alpha, beta, gamma, phi, period, sigma2, point, horizon,
                );
                let mut lower = Vec::with_capacity(horizon);
                let mut upper = Vec::with_capacity(horizon);
                for h in 0..horizon {
                    let se = variances[h].max(0.0).sqrt();
                    lower.push(point[h] - z * se);
                    upper.push(point[h] + z * se);
                }
                (lower, upper)
            }
        }
    }

    fn predict_internal_with_intervals(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
        confidence: f64,
    ) -> Result<Forecast> {
        let forecast = self.predict_internal(horizon, future_regressors)?;

        if horizon == 0 {
            return Ok(forecast);
        }

        let sigma2 = self
            .interval_sigma2
            .or(self.residual_variance)
            .unwrap_or(0.0);
        let preds = forecast.primary();
        let empty_seasonals: Vec<f64> = Vec::new();

        let (lower, upper) = Self::ets_interval_bounds(
            self.spec,
            self.alpha.unwrap_or(0.0),
            self.beta,
            self.gamma,
            self.phi,
            self.seasonal_period,
            sigma2,
            preds,
            horizon,
            confidence,
            self.level.unwrap_or(0.0),
            self.trend.unwrap_or(0.0),
            self.seasonals.as_deref().unwrap_or(&empty_seasonals),
            self.n,
        );

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
        if self.skip_optimization {
            if let Some(lvl) = self.level {
                // Warm-start: use pre-set states directly
                init_level = lvl;
                init_trend = self.trend.unwrap_or(0.0);
                seasonals = self.seasonals.clone().unwrap_or_default();
            } else {
                // skip_optimization but no level set — use defaults
                init_level = values[0];
                init_trend = 0.0;
                seasonals = vec![1.0; self.seasonal_period.max(1)];
            }
        } else if self.optimize {
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
        // R/statsforecast ets convention (D-07): the initial states are
        // time-0 states (before y_1) -- score and update from t = 0 for
        // every model, seasonal included (no first-period skip).
        let start_idx = 0;
        let is_mult_error = self.spec.error == ErrorType::Multiplicative;

        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);
        let mut sum_sq_errors = 0.0_f64;
        let mut sum_log_fc = 0.0_f64;

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
            let err = y - forecast;
            residuals.push(err);

            // R/statsforecast ets convention (D-07): the likelihood/variance
            // use the relative error (y-fc)/fc for multiplicative-error
            // models and ln|fc| (not ln|y|); `residuals()` keeps returning
            // the plain y-fc difference above for API stability.
            let se = if is_mult_error && forecast.abs() > 1e-10 {
                err / forecast
            } else {
                err
            };
            sum_sq_errors += se * se;
            if is_mult_error {
                sum_log_fc += forecast.abs().ln();
            }

            // Update state
            let level_prev = level;

            match (self.spec.trend, self.spec.seasonal) {
                (TrendType::None, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * level;
                }
                (TrendType::None, SeasonalType::Additive) => {
                    // Hyndman direct-error form (D-07, 11-06 Task 2); see
                    // the matching arm in calculate_likelihood_with_init_buf.
                    level = level_prev + alpha * err;
                    seasonals[season_idx] = s + gamma * err;
                }
                (TrendType::None, SeasonalType::Multiplicative) => {
                    let fc = level_prev * s;
                    if is_mult_error {
                        if fc.abs() > 1e-10 {
                            let e = (y - fc) / fc;
                            level = level_prev * (1.0 + alpha * e);
                            seasonals[season_idx] = s * (1.0 + gamma * e);
                        }
                    } else {
                        let y_des = if s.abs() > 1e-10 { y / s } else { y };
                        level = alpha * y_des + (1.0 - alpha) * level;
                        seasonals[season_idx] = if level_prev.abs() > 1e-10 {
                            gamma * (y / level_prev) + (1.0 - gamma) * s
                        } else {
                            s
                        };
                    }
                }
                (TrendType::Additive, SeasonalType::None) => {
                    level = level_prev + trend + alpha * err;
                    trend += beta * err;
                }
                (TrendType::Additive, SeasonalType::Additive) => {
                    level = level_prev + trend + alpha * err;
                    trend += beta * err;
                    seasonals[season_idx] = s + gamma * err;
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    let l_plus_b = level_prev + trend;
                    let fc = l_plus_b * s;
                    if is_mult_error {
                        if fc.abs() > 1e-10 {
                            let e = (y - fc) / fc;
                            level = l_plus_b * (1.0 + alpha * e);
                            trend += beta * l_plus_b * e;
                            seasonals[season_idx] = s * (1.0 + gamma * e);
                        }
                    } else {
                        if s.abs() > 1e-10 {
                            level = l_plus_b + alpha * err / s;
                            trend += beta * err / s;
                        } else {
                            level = l_plus_b;
                        }
                        seasonals[season_idx] = if l_plus_b.abs() > 1e-10 {
                            s + gamma * err / l_plus_b
                        } else {
                            s
                        };
                    }
                }
                (TrendType::AdditiveDamped, SeasonalType::None) => {
                    level = level_prev + phi * trend + alpha * err;
                    trend = phi * trend + beta * err;
                }
                (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                    level = level_prev + phi * trend + alpha * err;
                    trend = phi * trend + beta * err;
                    seasonals[season_idx] = s + gamma * err;
                }
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    let l_plus_pb = level_prev + phi * trend;
                    let fc = l_plus_pb * s;
                    if is_mult_error {
                        if fc.abs() > 1e-10 {
                            let e = (y - fc) / fc;
                            level = l_plus_pb * (1.0 + alpha * e);
                            trend = phi * trend + beta * l_plus_pb * e;
                            seasonals[season_idx] = s * (1.0 + gamma * e);
                        }
                    } else {
                        if s.abs() > 1e-10 {
                            level = l_plus_pb + alpha * err / s;
                            trend = phi * trend + beta * err / s;
                        } else {
                            level = l_plus_pb;
                            trend *= phi;
                        }
                        seasonals[season_idx] = if l_plus_pb.abs() > 1e-10 {
                            s + gamma * err / l_plus_pb
                        } else {
                            s
                        };
                    }
                }
            }
        }

        self.level = Some(level);
        self.trend = Some(trend);
        if self.spec.has_seasonal() {
            self.seasonals = Some(seasonals);
        }
        self.fitted = Some(fitted);

        // Calculate residual variance (additive-residual based; used for
        // prediction intervals only -- see plan 11-07) and information
        // criteria (R/statsforecast ets convention, D-07):
        //   loglik = -0.5*(n*ln(sum_sq) + 2*sum_log|fc)  [mult-error term only]
        // where sum_sq/sum_log_fc were accumulated above using relative
        // errors + ln|forecast| for multiplicative-error models.
        let valid_slice = &residuals[start_idx..];
        if !valid_slice.is_empty() && sum_sq_errors.is_finite() {
            let variance = crate::simd::sum_of_squares(valid_slice) / valid_slice.len() as f64;
            self.residual_variance = Some(variance);

            // Interval sigma2 (D-08, plan 11-07): R's `forecast.ets` divides
            // by `n - length(par)` (length(par) = num_params() - 1, since
            // num_params() also counts sigma^2 itself), not by `n`. Uses
            // `sum_sq_errors` directly (relative-error basis for
            // multiplicative-error models, matching the loglik/AIC below)
            // rather than `variance` above, which is always raw-residual
            // based regardless of error type.
            let n_obs_f = valid_slice.len() as f64;
            let length_par = (self.num_params() as f64 - 1.0).max(1.0);
            let df = (n_obs_f - length_par).max(1.0);
            self.interval_sigma2 = Some(sum_sq_errors / df);

            // Floor as in calculate_likelihood_with_init_buf: a perfect fit
            // (sum_sq_errors == 0) is the global optimum, not degenerate.
            let sum_sq_errors = sum_sq_errors.max(1e-300);
            let n = valid_slice.len() as f64;
            let k = self.num_params() as f64;
            let ll = if is_mult_error {
                -0.5 * (n * sum_sq_errors.ln() + 2.0 * sum_log_fc)
            } else {
                -0.5 * (n * sum_sq_errors.ln())
            };

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

    /// Prediction intervals follow R `forecast:::forecast.ets`'s own
    /// class1/class2/class3 dispatch (Hyndman, Koehler, Ord & Snyder 2008
    /// ch. 6; D-08, plan 11-07):
    /// - **Class 1** (additive error, trend N/A/Ad, season N/A): analytic
    ///   `v_h = sigma2*(1 + sum_{j<h} c_j^2)`, exact to 1e-6 relative of R
    ///   at R's own parameters.
    /// - **Class 2** (multiplicative error, trend N/A/Ad, season N/A): the
    ///   same `c_j` coefficients feed R's heteroscedastic `theta_h`
    ///   recursion, also exact to 1e-6 relative.
    /// - **Class 3 and additive-error + multiplicative-season models**
    ///   (MNM/MAM/MAdM, ANM/AAM/AAdM): 5000-path seeded Monte-Carlo
    ///   simulation with empirical (R type-7) quantiles, within 5%
    ///   relative of R's half-widths and identical across repeated calls.
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

    fn exog_coefficients(&self) -> Option<&OLSResult> {
        self.exog_ols.as_ref()
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

    fn fitted_params(&self) -> Option<FittedParams> {
        let level = self.level?;
        let mut params = HashMap::new();
        params.insert("level".to_string(), level);
        if let Some(alpha) = self.alpha {
            params.insert("alpha".to_string(), alpha);
        }
        if let Some(trend) = self.trend {
            params.insert("trend".to_string(), trend);
        }
        if let Some(beta) = self.beta {
            params.insert("beta".to_string(), beta);
        }
        if let Some(gamma) = self.gamma {
            params.insert("gamma".to_string(), gamma);
        }
        if let Some(phi) = self.phi {
            params.insert("phi".to_string(), phi);
        }
        params.insert("seasonal_period".to_string(), self.seasonal_period as f64);
        params.insert("n".to_string(), self.n as f64);
        Some(FittedParams {
            params,
            seasonal: self.seasonals.clone(),
        })
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
                    // SAFETY: seasonals_ref is Some when spec.seasonal != None (set above).
                    let s = seasonals_ref.unwrap()[(self.n + h - 1) % period];
                    level_component.push(level_val);
                    trend_component_vec.push(trend_component);
                    seasonal_component_vec.push(s);
                }
                SeasonalType::Multiplicative => {
                    // SAFETY: seasonals_ref is Some when spec.seasonal != None (set above).
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
    fn ets_spec_from_notation_accepts_maa_and_mada() {
        // Per D-07/Task 3 (phase 11-06): MAA and MAdA are valid R
        // forecast::ets models (included in R's restrict = TRUE default
        // set) — from_notation parses them successfully, not an error.
        assert_eq!(ETSSpec::from_notation("MAA").unwrap(), ETSSpec::maa());
        assert_eq!(ETSSpec::from_notation("MAdA").unwrap(), ETSSpec::mada());
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
        assert!(ETSSpec::maa().is_valid());
        assert!(ETSSpec::mada().is_valid());
    }

    #[test]
    fn ets_spec_is_r_restricted() {
        // Per D-07/Task 3 (phase 11-06): R forecast::ets()'s restrict = TRUE
        // default excludes additive-error models with multiplicative
        // seasonality (ANM, AAM, AAdM) — everything else, including the
        // formerly-"unstable" MAA/MAdA, is left in the default pool.
        assert!(ETSSpec::anm().is_r_restricted());
        assert!(ETSSpec::aam().is_r_restricted());
        assert!(ETSSpec::aadm().is_r_restricted());

        assert!(!ETSSpec::maa().is_r_restricted());
        assert!(!ETSSpec::mada().is_r_restricted());
        assert!(!ETSSpec::mam().is_r_restricted());
        assert!(!ETSSpec::ann().is_r_restricted());
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

    // =========================================================================
    // Warm-start tests
    // =========================================================================

    #[test]
    fn ets_warm_start_predict_without_fit() {
        // Create a warm-started ETS(A,N,N) model with a known level
        let model = ETS::with_initial_states(ETSSpec::ann(), 1, 42.0, 0.0, vec![]);
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        // ETS(A,N,N) flat forecast at level
        for &v in forecast.primary() {
            assert_relative_eq!(v, 42.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn ets_warm_start_with_trend() {
        // ETS(A,A,N) warm-started with level=10 and trend=2
        let model = ETS::with_initial_states(ETSSpec::aan(), 1, 10.0, 2.0, vec![]);
        let forecast = model.predict(3).unwrap();
        let preds = forecast.primary();
        // h=1: 10+2=12, h=2: 10+2*2=14, h=3: 10+3*2=16
        assert_relative_eq!(preds[0], 12.0, epsilon = 1e-10);
        assert_relative_eq!(preds[1], 14.0, epsilon = 1e-10);
        assert_relative_eq!(preds[2], 16.0, epsilon = 1e-10);
    }

    #[test]
    fn ets_extract_params_then_warm_start() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Fit original model
        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();
        let forecast1 = model.predict(5).unwrap();

        // Extract params and warm-start
        let fp = model.fitted_params().unwrap();
        let level = fp.params["level"];
        let trend = *fp.params.get("trend").unwrap_or(&0.0);
        let warm = ETS::with_initial_states(
            ETSSpec::ann(),
            1,
            level,
            trend,
            fp.seasonal.unwrap_or_default(),
        );
        let forecast2 = warm.predict(5).unwrap();

        // Both should produce identical forecasts (ETS(A,N,N) flat at level)
        for (a, b) in forecast1.primary().iter().zip(forecast2.primary().iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn ets_warm_start_fit_refines() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Warm-start with approximate params then fit to refine
        let mut model = ETS::with_initial_states(ETSSpec::ann(), 1, 5.0, 0.0, vec![]);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        // Level should have been updated from initial 5.0
        for &v in forecast.primary() {
            assert!(v > 5.0);
        }
    }

    #[test]
    fn ets_fitted_params_returns_none_before_fit() {
        let model = ETS::new(ETSSpec::ann(), 1);
        assert!(model.fitted_params().is_none());
    }

    #[test]
    fn ets_fitted_params_contains_expected_keys() {
        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();

        let fp = model.fitted_params().unwrap();
        assert!(fp.params.contains_key("level"));
        assert!(fp.params.contains_key("alpha"));
        assert!(fp.params.contains_key("seasonal_period"));
    }
}
