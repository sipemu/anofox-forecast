//! Global ETS: shared parameters across many series.
//!
//! Fits a single set of smoothing parameters (α, β, γ, φ) by evaluating
//! the likelihood across ALL series simultaneously in a vectorized loop.
//! Each series retains its own initial states (level, trend, seasonal),
//! but the smoothing parameters are pooled.
//!
//! This is the approach used by Nixtla's statsforecast for batch ETS:
//! one parameter search, N series evaluated per objective call.
//!
//! # Example
//!
//! ```rust
//! use anofox_forecast::models::exponential::{GlobalETS, ETSSpec};
//!
//! let series = vec![
//!     vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0],
//!     vec![20.0, 22.0, 21.0, 23.0, 22.0, 24.0, 23.0, 25.0],
//! ];
//! let mut model = GlobalETS::new(ETSSpec::ann(), 1);
//! model.fit(&series).unwrap();
//! let forecasts = model.predict(4);
//! assert_eq!(forecasts.len(), 2);
//! ```

use super::auto_ets::ModelPool;
use super::ets::{ETSSpec, ErrorType, SeasonalType, TrendType};
use crate::error::{ForecastError, Result};
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// Global ETS model: shared smoothing parameters, per-series states.
///
/// Optimizes a single set of smoothing parameters (α, β, γ, φ) by
/// evaluating the total negative log-likelihood across all series.
/// Each series gets its own initial states (level, trend, seasonal).
///
/// Much faster than N independent fits when N is large, because the
/// optimizer runs once instead of N times.
#[derive(Debug, Clone)]
pub struct GlobalETS {
    spec: ETSSpec,
    period: usize,
    /// Shared smoothing parameters
    alpha: f64,
    beta: Option<f64>,
    gamma: Option<f64>,
    phi: Option<f64>,
    /// Per-series initial states: (level, trend, seasonals)
    states: Vec<SeriesState>,
    /// Per-series final states for prediction
    final_states: Vec<SeriesState>,
    fitted: bool,
}

#[derive(Debug, Clone)]
struct SeriesState {
    level: f64,
    trend: f64,
    seasonals: Vec<f64>,
}

impl GlobalETS {
    /// Create a new GlobalETS with the given specification and seasonal period.
    pub fn new(spec: ETSSpec, period: usize) -> Self {
        Self {
            spec,
            period,
            alpha: 0.3,
            beta: if spec.has_trend() { Some(0.1) } else { None },
            gamma: if spec.has_seasonal() { Some(0.1) } else { None },
            phi: if spec.is_damped() { Some(0.98) } else { None },
            states: Vec::new(),
            final_states: Vec::new(),
            fitted: false,
        }
    }

    /// Fit shared parameters across all series.
    ///
    /// Each inner `&[f64]` is one time series. All must have the same length.
    pub fn fit(&mut self, all_series: &[Vec<f64>]) -> Result<()> {
        if all_series.is_empty() {
            return Err(ForecastError::InsufficientData {
                needed: 1,
                got: 0,
                hint: Some("GlobalETS requires at least one series".into()),
            });
        }

        let n = all_series[0].len();
        let _n_series = all_series.len();
        let period = self.period;
        let start_idx = if self.spec.has_seasonal() { period } else { 0 };

        if n <= start_idx + 2 {
            return Err(ForecastError::InsufficientData {
                needed: start_idx + 3,
                got: n,
                hint: Some("Series too short for this ETS spec".into()),
            });
        }

        // V-01: per-series NaN/Inf guard — raw &[Vec<f64>] API bypasses validate_series_complete()
        for (i, s) in all_series.iter().enumerate() {
            if s.iter().any(|v| !v.is_finite()) {
                return Err(ForecastError::InvalidParameter(format!(
                    "series {} contains NaN or Inf values",
                    i
                )));
            }
        }

        // Initialize per-series states using heuristics
        self.states = all_series
            .iter()
            .map(|values| Self::initialize_state(values, self.spec, period))
            .collect();

        // Build parameter vector and bounds
        let has_trend = self.spec.has_trend();
        let has_seasonal = self.spec.has_seasonal();
        let is_damped = self.spec.is_damped();

        let mut params = vec![0.3]; // alpha
        let mut bounds = vec![(0.0001, 0.9999)];

        if has_trend {
            params.push(0.1); // beta
            bounds.push((0.0001, 0.9999));
        }
        if has_seasonal {
            params.push(0.1); // gamma
            bounds.push((0.0001, 0.9999));
        }
        if is_damped {
            params.push(0.98); // phi
            bounds.push((0.8, 0.98));
        }

        // Objective: total NLL across all series
        let states_ref = &self.states;
        let spec = self.spec;

        let result = nelder_mead(
            |p| {
                let alpha = p[0];
                let mut idx = 1;
                let beta = if has_trend {
                    idx += 1;
                    Some(p[idx - 1])
                } else {
                    None
                };
                let gamma = if has_seasonal {
                    idx += 1;
                    Some(p[idx - 1])
                } else {
                    None
                };
                let phi = if is_damped { Some(p[idx]) } else { None };

                Self::total_nll(
                    all_series, states_ref, spec, period, alpha, beta, gamma, phi,
                )
            },
            &params,
            Some(&bounds),
            NelderMeadConfig {
                max_iter: 500,
                tolerance: 1e-8,
                stagnation_window: 100,
                ..Default::default()
            },
        );

        // Extract optimized parameters
        self.alpha = result.optimal_point[0].clamp(0.0001, 0.9999);
        let mut idx = 1;
        if has_trend {
            self.beta = Some(result.optimal_point[idx].clamp(0.0001, 0.9999));
            idx += 1;
        }
        if has_seasonal {
            self.gamma = Some(result.optimal_point[idx].clamp(0.0001, 0.9999));
            idx += 1;
        }
        if is_damped {
            self.phi = Some(result.optimal_point[idx].clamp(0.8, 0.98));
        }

        // Run final state recursion to get end-of-series states for prediction
        self.final_states = all_series
            .iter()
            .zip(self.states.iter())
            .map(|(values, init)| {
                Self::run_states(
                    values, init, spec, period, self.alpha, self.beta, self.gamma, self.phi,
                )
            })
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Get the fitted smoothing parameters.
    pub fn params(&self) -> (f64, Option<f64>, Option<f64>, Option<f64>) {
        (self.alpha, self.beta, self.gamma, self.phi)
    }

    /// Predict h steps ahead for all series.
    pub fn predict(&self, horizon: usize) -> Vec<Vec<f64>> {
        if !self.fitted {
            return vec![];
        }
        self.final_states
            .iter()
            .map(|state| {
                Self::forecast_from_state(state, self.spec, self.period, self.phi, horizon)
            })
            .collect()
    }

    /// Total negative log-likelihood across all series.
    fn total_nll(
        all_series: &[Vec<f64>],
        states: &[SeriesState],
        spec: ETSSpec,
        period: usize,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> f64 {
        let mut total = 0.0;
        for (values, init) in all_series.iter().zip(states.iter()) {
            let nll = Self::series_nll(values, init, spec, period, alpha, beta, gamma, phi);
            if !nll.is_finite() {
                return f64::MAX;
            }
            total += nll;
        }
        total
    }

    /// NLL for a single series with given parameters and initial states.
    fn series_nll(
        values: &[f64],
        init: &SeriesState,
        spec: ETSSpec,
        period: usize,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> f64 {
        let n = values.len();
        let start_idx = if spec.has_seasonal() { period } else { 0 };
        if n <= start_idx + 1 {
            return f64::MAX;
        }

        let mut level = init.level;
        let mut trend = init.trend;
        let mut seasonal_buf = init.seasonals.clone();
        let phi = phi.unwrap_or(1.0);
        let beta = beta.unwrap_or(0.0);
        let gamma = gamma.unwrap_or(0.0);
        let is_mult_error = spec.error == ErrorType::Multiplicative;

        let mut sum_sq = 0.0_f64;
        let mut sum_log = 0.0_f64;
        let mut count = 0_usize;

        for (t, &y) in values.iter().enumerate().skip(start_idx) {
            // Forecast
            let fc = match (spec.trend, spec.seasonal) {
                (TrendType::None, SeasonalType::None) => level,
                (TrendType::None, SeasonalType::Additive) => level + seasonal_buf[t % period],
                (TrendType::None, SeasonalType::Multiplicative) => level * seasonal_buf[t % period],
                (TrendType::Additive, SeasonalType::None) => level + trend,
                (TrendType::Additive, SeasonalType::Additive) => {
                    level + trend + seasonal_buf[t % period]
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    (level + trend) * seasonal_buf[t % period]
                }
                (TrendType::AdditiveDamped, SeasonalType::None) => level + phi * trend,
                (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                    level + phi * trend + seasonal_buf[t % period]
                }
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    (level + phi * trend) * seasonal_buf[t % period]
                }
            };

            if !fc.is_finite() {
                return f64::MAX;
            }

            let err = y - fc;
            let se = if is_mult_error && fc.abs() > 1e-10 {
                err / fc
            } else {
                err
            };
            sum_sq += se * se;
            if !sum_sq.is_finite() {
                return f64::MAX;
            }
            if is_mult_error {
                sum_log += y.abs().ln();
            }
            count += 1;

            // Update states
            let level_prev = level;
            let s_idx = t % period;
            let s = if spec.has_seasonal() {
                seasonal_buf[s_idx]
            } else {
                0.0
            };

            match (spec.trend, spec.seasonal) {
                (TrendType::None, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * level;
                }
                (TrendType::None, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * level;
                    seasonal_buf[s_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::None, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * level;
                    seasonal_buf[s_idx] = if level.abs() > 1e-10 {
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
                    seasonal_buf[s_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * trend;
                    seasonal_buf[s_idx] = if level.abs() > 1e-10 {
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
                    seasonal_buf[s_idx] = gamma * (y - level) + (1.0 - gamma) * s;
                }
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + phi * trend);
                    trend = beta * (level - level_prev) + (1.0 - beta) * phi * trend;
                    seasonal_buf[s_idx] = if level.abs() > 1e-10 {
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

        let sigma2 = sum_sq / count as f64;
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return f64::MAX;
        }

        let nll = if is_mult_error {
            -0.5 * count as f64 * (1.0 + sigma2.ln() + (2.0 * std::f64::consts::PI).ln()) - sum_log
        } else {
            -0.5 * count as f64 * (1.0 + sigma2.ln() + (2.0 * std::f64::consts::PI).ln())
        };

        -nll // return positive NLL for minimization
    }

    /// Run the state recursion to get final states.
    fn run_states(
        values: &[f64],
        init: &SeriesState,
        spec: ETSSpec,
        period: usize,
        alpha: f64,
        beta: Option<f64>,
        gamma: Option<f64>,
        phi: Option<f64>,
    ) -> SeriesState {
        let start_idx = if spec.has_seasonal() { period } else { 0 };
        let mut level = init.level;
        let mut trend = init.trend;
        let mut seasonal_buf = init.seasonals.clone();
        let phi_val = phi.unwrap_or(1.0);
        let beta_val = beta.unwrap_or(0.0);
        let gamma_val = gamma.unwrap_or(0.0);

        for (t, &y) in values.iter().enumerate().skip(start_idx) {
            let level_prev = level;
            let s_idx = t % period;
            let s = if spec.has_seasonal() {
                seasonal_buf[s_idx]
            } else {
                0.0
            };

            match (spec.trend, spec.seasonal) {
                (TrendType::None, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * level;
                }
                (TrendType::None, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * level;
                    seasonal_buf[s_idx] = gamma_val * (y - level) + (1.0 - gamma_val) * s;
                }
                (TrendType::None, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * level;
                    seasonal_buf[s_idx] = if level.abs() > 1e-10 {
                        gamma_val * (y / level) + (1.0 - gamma_val) * s
                    } else {
                        s
                    };
                }
                (TrendType::Additive, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * (level_prev + trend);
                    trend = beta_val * (level - level_prev) + (1.0 - beta_val) * trend;
                }
                (TrendType::Additive, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + trend);
                    trend = beta_val * (level - level_prev) + (1.0 - beta_val) * trend;
                    seasonal_buf[s_idx] = gamma_val * (y - level) + (1.0 - gamma_val) * s;
                }
                (TrendType::Additive, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + trend);
                    trend = beta_val * (level - level_prev) + (1.0 - beta_val) * trend;
                    seasonal_buf[s_idx] = if level.abs() > 1e-10 {
                        gamma_val * (y / level) + (1.0 - gamma_val) * s
                    } else {
                        s
                    };
                }
                (TrendType::AdditiveDamped, SeasonalType::None) => {
                    level = alpha * y + (1.0 - alpha) * (level_prev + phi_val * trend);
                    trend = beta_val * (level - level_prev) + (1.0 - beta_val) * phi_val * trend;
                }
                (TrendType::AdditiveDamped, SeasonalType::Additive) => {
                    level = alpha * (y - s) + (1.0 - alpha) * (level_prev + phi_val * trend);
                    trend = beta_val * (level - level_prev) + (1.0 - beta_val) * phi_val * trend;
                    seasonal_buf[s_idx] = gamma_val * (y - level) + (1.0 - gamma_val) * s;
                }
                (TrendType::AdditiveDamped, SeasonalType::Multiplicative) => {
                    let y_des = if s.abs() > 1e-10 { y / s } else { y };
                    level = alpha * y_des + (1.0 - alpha) * (level_prev + phi_val * trend);
                    trend = beta_val * (level - level_prev) + (1.0 - beta_val) * phi_val * trend;
                    seasonal_buf[s_idx] = if level.abs() > 1e-10 {
                        gamma_val * (y / level) + (1.0 - gamma_val) * s
                    } else {
                        s
                    };
                }
            }
        }

        SeriesState {
            level,
            trend,
            seasonals: seasonal_buf,
        }
    }

    /// Forecast from final state.
    fn forecast_from_state(
        state: &SeriesState,
        spec: ETSSpec,
        period: usize,
        phi: Option<f64>,
        horizon: usize,
    ) -> Vec<f64> {
        let phi_val = phi.unwrap_or(1.0);
        let mut forecasts = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let trend_component = match spec.trend {
                TrendType::None => 0.0,
                TrendType::Additive => state.trend * h as f64,
                TrendType::AdditiveDamped => {
                    // Damped trend: phi + phi^2 + ... + phi^h
                    let mut sum = 0.0;
                    let mut phi_pow = phi_val;
                    for _ in 0..h {
                        sum += phi_pow;
                        phi_pow *= phi_val;
                    }
                    state.trend * sum
                }
            };

            let seasonal = if spec.has_seasonal() && !state.seasonals.is_empty() {
                state.seasonals[(state.seasonals.len() - period + (h - 1) % period) % period]
            } else {
                match spec.seasonal {
                    SeasonalType::Multiplicative => 1.0,
                    _ => 0.0,
                }
            };

            let fc = match spec.seasonal {
                SeasonalType::Multiplicative => (state.level + trend_component) * seasonal,
                _ => state.level + trend_component + seasonal,
            };

            forecasts.push(fc);
        }

        forecasts
    }

    /// Initialize state for a single series using heuristics.
    fn initialize_state(values: &[f64], spec: ETSSpec, period: usize) -> SeriesState {
        let n = values.len();

        // Level: mean of first period (or first value)
        let level = if period > 0 && n >= period {
            values[..period].iter().sum::<f64>() / period as f64
        } else {
            values[0]
        };

        // Trend: average slope over first two periods
        let trend = if spec.has_trend() && n >= 2 * period && period > 0 {
            let first_mean = values[..period].iter().sum::<f64>() / period as f64;
            let second_mean = values[period..2 * period].iter().sum::<f64>() / period as f64;
            (second_mean - first_mean) / period as f64
        } else {
            0.0
        };

        // Seasonal: detrended ratios/differences from first cycle
        let seasonals = if spec.has_seasonal() && period > 0 && n >= period {
            match spec.seasonal {
                SeasonalType::Additive => (0..period).map(|i| values[i] - level).collect(),
                SeasonalType::Multiplicative => (0..period)
                    .map(|i| {
                        if level.abs() > 1e-10 {
                            values[i] / level
                        } else {
                            1.0
                        }
                    })
                    .collect(),
                SeasonalType::None => vec![],
            }
        } else {
            vec![]
        };

        SeriesState {
            level,
            trend,
            seasonals,
        }
    }
}

/// Global AutoETS: automatic model selection across many series using GlobalETS.
///
/// For each candidate ETS spec, fits a single GlobalETS across all series.
/// Then selects the best spec per series based on the per-series NLL.
/// Much faster than N independent AutoETS fits because each candidate
/// is evaluated with one NM optimization (not N).
///
/// # Example
///
/// ```rust
/// use anofox_forecast::models::exponential::{GlobalAutoETS, ModelPool};
///
/// let series = vec![
///     (0..28).map(|i| 10.0 + (i as f64 * 0.9).sin() * 5.0).collect::<Vec<_>>(),
///     (0..28).map(|i| 20.0 - i as f64 * 0.1).collect::<Vec<_>>(),
/// ];
/// let mut model = GlobalAutoETS::new(7, ModelPool::Reduced);
/// model.fit(&series).unwrap();
/// let forecasts = model.predict(7);
/// assert_eq!(forecasts.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct GlobalAutoETS {
    period: usize,
    pool: ModelPool,
    /// Per-series: (best_spec, forecaster)
    per_series: Vec<(
        ETSSpec,
        SeriesState,
        f64,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    )>,
    fitted: bool,
}

impl GlobalAutoETS {
    /// Create with a seasonal period and model pool.
    pub fn new(period: usize, pool: ModelPool) -> Self {
        Self {
            period,
            pool,
            per_series: Vec::new(),
            fitted: false,
        }
    }

    /// Fit: evaluate each candidate spec globally, pick best per series.
    pub fn fit(&mut self, all_series: &[Vec<f64>]) -> Result<()> {
        if all_series.is_empty() {
            return Err(ForecastError::InsufficientData {
                needed: 1,
                got: 0,
                hint: Some("GlobalAutoETS requires at least one series".into()),
            });
        }

        // V-01 (auto-wrapper): per-series NaN/Inf guard — must be checked before the candidate
        // loop, which swallows inner GlobalETS::fit errors with `.is_err() → continue`. Without
        // this guard, NaN input causes every candidate spec to fail silently, leaving the model
        // in a poisoned zero-state that unconditionally returns Ok(()) and emits zero forecasts.
        for (i, s) in all_series.iter().enumerate() {
            if s.iter().any(|v| !v.is_finite()) {
                return Err(ForecastError::InvalidParameter(format!(
                    "series {} contains NaN or Inf values",
                    i
                )));
            }
        }

        let n_series = all_series.len();
        let period = self.period;

        // Check for non-positive values (restricts multiplicative)
        let has_non_positive = all_series.iter().any(|s| s.iter().any(|&v| v <= 0.0));

        // Detect seasonality: simple check on first few series
        let has_seasonal = period > 1 && all_series[0].len() >= 2 * period;

        // Generate candidates based on pool
        let candidates = Self::generate_candidates(self.pool, has_seasonal, has_non_positive);

        // Per-series: track best NLL and corresponding spec
        let mut best_nll: Vec<f64> = vec![f64::MAX; n_series];
        let mut best_spec: Vec<ETSSpec> = vec![ETSSpec::ann(); n_series];
        let mut best_states: Vec<SeriesState> = (0..n_series)
            .map(|_| SeriesState {
                level: 0.0,
                trend: 0.0,
                seasonals: vec![],
            })
            .collect();
        let mut best_params: Vec<(f64, Option<f64>, Option<f64>, Option<f64>)> =
            vec![(0.3, None, None, None); n_series];

        // Evaluate each candidate spec globally
        for spec in &candidates {
            let mut global = GlobalETS::new(*spec, period);
            if global.fit(all_series).is_err() {
                continue;
            }

            let (alpha, beta, gamma, phi) = global.params();

            // Compute per-series NLL at the optimized parameters
            for (s, values) in all_series.iter().enumerate() {
                let init = &global.states[s];
                let nll =
                    GlobalETS::series_nll(values, init, *spec, period, alpha, beta, gamma, phi);
                if nll < best_nll[s] {
                    best_nll[s] = nll;
                    best_spec[s] = *spec;
                    best_states[s] = global.final_states[s].clone();
                    best_params[s] = (alpha, beta, gamma, phi);
                }
            }
        }

        // Store per-series results
        self.per_series = best_spec
            .into_iter()
            .zip(best_states)
            .zip(best_params)
            .map(|((spec, state), (a, b, g, p))| (spec, state, a, b, g, p))
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Predict h steps ahead for all series.
    pub fn predict(&self, horizon: usize) -> Vec<Vec<f64>> {
        if !self.fitted {
            return vec![];
        }
        self.per_series
            .iter()
            .map(|(spec, state, _, _, _, phi)| {
                GlobalETS::forecast_from_state(state, *spec, self.period, *phi, horizon)
            })
            .collect()
    }

    /// Get the selected spec per series.
    pub fn selected_specs(&self) -> Vec<ETSSpec> {
        self.per_series.iter().map(|(spec, ..)| *spec).collect()
    }

    /// Generate candidate specs based on pool.
    fn generate_candidates(
        pool: ModelPool,
        has_seasonal: bool,
        has_non_positive: bool,
    ) -> Vec<ETSSpec> {
        let error_types = if has_non_positive {
            vec![ErrorType::Additive]
        } else {
            vec![ErrorType::Additive, ErrorType::Multiplicative]
        };

        let trend_types = vec![
            TrendType::None,
            TrendType::Additive,
            TrendType::AdditiveDamped,
        ];

        let seasonal_types = if !has_seasonal {
            vec![SeasonalType::None]
        } else if has_non_positive {
            vec![SeasonalType::None, SeasonalType::Additive]
        } else {
            vec![
                SeasonalType::None,
                SeasonalType::Additive,
                SeasonalType::Multiplicative,
            ]
        };

        let mut candidates = Vec::new();
        for &error in &error_types {
            for &trend in &trend_types {
                for &seasonal in &seasonal_types {
                    // Skip M,A,A and M,Ad,A (unstable)
                    if error == ErrorType::Multiplicative
                        && (trend == TrendType::Additive || trend == TrendType::AdditiveDamped)
                        && seasonal == SeasonalType::Additive
                    {
                        continue;
                    }

                    // Apply pool filters
                    match pool {
                        ModelPool::Complete => {}
                        ModelPool::NoMultiplicativeTrend => {}
                        ModelPool::DampedTrendOnly => {
                            if trend == TrendType::Additive {
                                continue;
                            }
                        }
                        ModelPool::MatchErrorSeasonal => {
                            if error == ErrorType::Multiplicative
                                && seasonal == SeasonalType::Additive
                            {
                                continue;
                            }
                            if error == ErrorType::Additive
                                && seasonal == SeasonalType::Multiplicative
                            {
                                continue;
                            }
                        }
                        ModelPool::Reduced => {
                            if trend == TrendType::Additive {
                                continue;
                            }
                            if error == ErrorType::Multiplicative
                                && seasonal == SeasonalType::Additive
                            {
                                continue;
                            }
                            if error == ErrorType::Additive
                                && seasonal == SeasonalType::Multiplicative
                            {
                                continue;
                            }
                        }
                    }

                    candidates.push(ETSSpec::new(error, trend, seasonal));
                }
            }
        }
        candidates
    }
}
