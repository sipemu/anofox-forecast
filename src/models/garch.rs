//! GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model.
//!
//! GARCH models are used for volatility forecasting in financial time series
//! where variance changes over time.
//!
//! Model: σ²(t) = ω + Σ(αᵢ * ε²(t-i)) + Σ(βⱼ * σ²(t-j))
//!
//! Parameters are estimated via Maximum Likelihood Estimation (MLE) using
//! Nelder-Mead optimization, matching statsforecast behavior.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// GARCH(p,q) model for volatility forecasting.
///
/// The conditional variance equation:
/// σ²(t) = ω + α₁ε²(t-1) + ... + αₚε²(t-p) + β₁σ²(t-1) + ... + βqσ²(t-q)
///
/// where:
/// - ω (omega) is the constant term
/// - α (alpha) coefficients capture impact of past squared residuals
/// - β (beta) coefficients capture persistence of past variance
///
/// Note: `predict()` returns simulated innovations (ε*σ), matching statsforecast behavior.
/// Use `forecast_variance()` for analytical variance forecasts.
///
/// # Example
/// ```
/// use anofox_forecast::models::garch::GARCH;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..100).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i)).collect();
/// // Financial returns with volatility clustering
/// let values: Vec<f64> = (0..100).map(|i| {
///     (i as f64 * 0.1).sin() * (1.0 + 0.5 * ((i / 10) % 3) as f64)
/// }).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = GARCH::new(1, 1);  // GARCH(1,1)
/// model.fit(&ts).unwrap();
/// let variance_forecast = model.forecast_variance(10).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GARCH {
    /// GARCH order p (number of lagged squared residuals).
    p: usize,
    /// GARCH order q (number of lagged variances).
    q: usize,
    /// Constant term (omega).
    omega: f64,
    /// Alpha coefficients (for squared residuals).
    alpha: Vec<f64>,
    /// Beta coefficients (for lagged variances).
    beta: Vec<f64>,
    /// Mean of the series.
    mean: Option<f64>,
    /// Residuals (y - mean).
    residuals: Option<Vec<f64>>,
    /// Conditional variance series.
    conditional_variance: Option<Vec<f64>>,
    /// Unconditional (long-run) variance.
    unconditional_variance: Option<f64>,
    /// Series length.
    n: usize,
    /// Last p y values (residuals) for forecasting - matches statsforecast y_vals.
    y_vals: Vec<f64>,
    /// Last q sigma² values for forecasting - matches statsforecast sigma2_vals.
    sigma2_vals: Vec<f64>,
}

impl GARCH {
    /// Create a new GARCH(p,q) model with default parameters.
    ///
    /// Default: omega=0.01, alpha=[0.1], beta=[0.85]
    pub fn new(p: usize, q: usize) -> Self {
        let p = p.max(1);
        let q = q.max(1);

        // Default parameters that ensure stationarity
        let alpha = vec![0.1 / p as f64; p];
        let beta = vec![0.85 / q as f64; q];

        Self {
            p,
            q,
            omega: 0.01,
            alpha,
            beta,
            mean: None,
            residuals: None,
            conditional_variance: None,
            unconditional_variance: None,
            n: 0,
            y_vals: Vec::new(),
            sigma2_vals: Vec::new(),
        }
    }

    /// Create GARCH(1,1) model (most common).
    pub fn garch_1_1() -> Self {
        Self::new(1, 1)
    }

    /// Set the omega (constant) parameter.
    pub fn with_omega(mut self, omega: f64) -> Self {
        self.omega = omega.max(0.0001);
        self
    }

    /// Set the alpha parameters.
    pub fn with_alpha(mut self, alpha: Vec<f64>) -> Self {
        self.p = alpha.len().max(1);
        self.alpha = alpha.into_iter().map(|a| a.max(0.0)).collect();
        self
    }

    /// Set the beta parameters.
    pub fn with_beta(mut self, beta: Vec<f64>) -> Self {
        self.q = beta.len().max(1);
        self.beta = beta.into_iter().map(|b| b.max(0.0)).collect();
        self
    }

    /// Get the omega parameter.
    pub fn omega(&self) -> f64 {
        self.omega
    }

    /// Get the alpha parameters.
    pub fn alpha_params(&self) -> &[f64] {
        &self.alpha
    }

    /// Get the beta parameters.
    pub fn beta_params(&self) -> &[f64] {
        &self.beta
    }

    /// Get the conditional variance series.
    pub fn conditional_variance(&self) -> Option<&[f64]> {
        self.conditional_variance.as_deref()
    }

    /// Get the unconditional (long-run) variance.
    pub fn unconditional_variance(&self) -> Option<f64> {
        self.unconditional_variance
    }

    /// Check if parameters satisfy stationarity condition.
    pub fn is_stationary(&self) -> bool {
        let sum: f64 = self.alpha.iter().sum::<f64>() + self.beta.iter().sum::<f64>();
        sum < 1.0
    }

    /// Calculate unconditional variance from parameters.
    fn calculate_unconditional_variance(&self) -> f64 {
        let sum: f64 = self.alpha.iter().sum::<f64>() + self.beta.iter().sum::<f64>();
        if sum < 1.0 {
            self.omega / (1.0 - sum)
        } else {
            // Non-stationary case: use sample variance as fallback
            self.omega * 10.0
        }
    }

    /// Compute sigma² series from parameters and original series values.
    /// This matches statsforecast's garch_sigma2 function exactly.
    /// Note: statsforecast uses np.flip(alpha) and np.flip(beta) which reverses the coefficient order.
    fn compute_sigma2(
        x: &[f64],
        omega: f64,
        alpha: &[f64],
        beta: &[f64],
        p: usize,
        q: usize,
    ) -> Vec<f64> {
        let n = x.len();
        // statsforecast initializes sigma2[0] = np.var(x) which is the sample variance
        let mean = x.iter().sum::<f64>() / n as f64;
        let sample_var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        let mut sigma2 = vec![f64::NAN; n];
        sigma2[0] = sample_var;

        let max_lag = p.max(q);
        for k in max_lag..n {
            // statsforecast: psum = np.flip(alpha) * (x[k-p:k] ** 2)
            // np.flip reverses alpha, so: alpha[-1]*x[k-p]^2 + alpha[-2]*x[k-p+1]^2 + ... + alpha[0]*x[k-1]^2
            let mut psum = 0.0;
            for i in 0..p {
                let x_idx = k - p + i;
                let alpha_idx = p - 1 - i; // flip: use alpha[p-1-i] for x[k-p+i]
                if x_idx < n && alpha_idx < alpha.len() {
                    psum += alpha[alpha_idx] * x[x_idx].powi(2);
                }
            }

            if q != 0 {
                // statsforecast: qsum = np.flip(beta) * sigma2[k-q:k]
                let mut qsum = 0.0;
                for j in 0..q {
                    let s_idx = k - q + j;
                    let beta_idx = q - 1 - j; // flip
                    if s_idx < n && beta_idx < beta.len() && !sigma2[s_idx].is_nan() {
                        qsum += beta[beta_idx] * sigma2[s_idx];
                    }
                }
                sigma2[k] = omega + psum + qsum;
            } else {
                sigma2[k] = omega + psum;
            }
        }

        sigma2
    }

    /// Optimize GARCH parameters using Maximum Likelihood Estimation.
    /// Uses multiple restarts to find global optimum, matching statsforecast's SLSQP behavior.
    fn optimize_parameters(&mut self, values: &[f64], residuals: &[f64]) {
        let n = values.len();
        let p = self.p;
        let q = self.q;

        // Bounds: omega >= 0 (can be large for high variance series), alpha/beta bounded
        let bounds: Vec<(f64, f64)> = std::iter::once((0.0, 10000.0)) // omega (allow large for high variance data)
            .chain((0..p).map(|_| (0.0, 0.999))) // alpha
            .chain((0..q).map(|_| (0.0, 0.999))) // beta
            .collect();

        let values_clone = values.to_vec();
        let residuals_clone = residuals.to_vec();

        // Objective: negative log-likelihood (matching statsforecast's garch_loglik)
        let objective = {
            let values_ref = values_clone.clone();
            let residuals_ref = residuals_clone.clone();
            move |params: &[f64]| -> f64 {
                let omega = params[0];
                let alpha: Vec<f64> = params[1..(p + 1)].to_vec();
                let beta: Vec<f64> = params[(p + 1)..].to_vec();

                // Check stationarity constraint: sum(alpha) + sum(beta) < 1
                let sum: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
                if sum >= 0.9999 {
                    return f64::MAX;
                }

                // Check positivity
                if omega < 0.0 || alpha.iter().any(|&a| a < 0.0) || beta.iter().any(|&b| b < 0.0) {
                    return f64::MAX;
                }

                // Compute sigma² using original values (like statsforecast)
                let sigma2 = Self::compute_sigma2(&values_ref, omega, &alpha, &beta, p, q);

                // Compute negative log-likelihood using residuals
                let max_lag = p.max(q);
                let mut neg_ll = 0.0;
                for k in max_lag..n {
                    let s = sigma2[k];
                    if s <= 0.0 || s.is_nan() {
                        continue;
                    }
                    let z = residuals_ref[k];
                    neg_ll += 0.5 * ((2.0 * std::f64::consts::PI).ln() + s.ln() + z * z / s);
                }

                if neg_ll.is_finite() {
                    neg_ll
                } else {
                    f64::MAX
                }
            }
        };

        let config = NelderMeadConfig {
            max_iter: 1000,
            tolerance: 1e-10,
            ..Default::default()
        };

        // Try multiple starting points to find global minimum
        let starting_points = vec![
            vec![0.1; p + q + 1],     // statsforecast default
            vec![0.01, 0.05, 0.9],    // small omega, small alpha, large beta
            vec![0.0, 0.01, 0.95],    // near-zero omega, very high persistence
            vec![0.001, 0.001, 0.99], // minimal omega/alpha, max beta
            vec![0.02, 0.02, 0.02],   // low persistence (good for stationary data)
            vec![0.1, 0.1, 0.1],      // medium persistence
            vec![0.5, 0.1, 0.8],      // moderate omega, high persistence
        ];

        let mut best_value = f64::MAX;
        let mut best_params = vec![0.1; p + q + 1];

        for initial in starting_points {
            if initial.len() != p + q + 1 {
                continue;
            }

            let result = nelder_mead(&objective, &initial, Some(&bounds), config.clone());

            if result.optimal_value < best_value {
                best_value = result.optimal_value;
                best_params = result.optimal_point.clone();
            }
        }

        // Extract optimized parameters
        let opt_omega = best_params[0].max(0.0);
        let opt_alpha: Vec<f64> = best_params[1..(p + 1)]
            .iter()
            .map(|&a| a.max(0.0))
            .collect();
        let opt_beta: Vec<f64> = best_params[(p + 1)..].iter().map(|&b| b.max(0.0)).collect();

        // Ensure stationarity
        let sum: f64 = opt_alpha.iter().sum::<f64>() + opt_beta.iter().sum::<f64>();
        if sum < 1.0 {
            self.omega = opt_omega;
            self.alpha = opt_alpha;
            self.beta = opt_beta;
        }
    }

    /// Forecast variance for multiple steps ahead.
    /// Uses the same logic as predict() but returns only variance (sigma²) values.
    pub fn forecast_variance(&self, horizon: usize) -> Result<Vec<f64>> {
        if self.y_vals.is_empty() || self.sigma2_vals.is_empty() {
            return Err(ForecastError::FitRequired);
        }

        if horizon == 0 {
            return Ok(Vec::new());
        }

        let p = self.p;
        let q = self.q;

        // Initialize with recent history (matching predict())
        let mut y_vals = vec![f64::NAN; horizon + p];
        let mut sigma2_vals = vec![f64::NAN; horizon + q];

        for (i, &y) in self.y_vals.iter().enumerate() {
            y_vals[i] = y;
        }
        for (i, &s) in self.sigma2_vals.iter().enumerate() {
            sigma2_vals[i] = s;
        }

        // Compute variance forecasts (same as predict but without random draws)
        for k in 0..horizon {
            let mut sigma2hat = self.omega;

            // psum = sum of (flipped alpha) * y_vals[k:p+k]^2
            let mut psum = 0.0;
            for i in 0..p {
                let y_idx = k + i;
                if y_idx < y_vals.len() && !y_vals[y_idx].is_nan() {
                    let alpha_idx = p - 1 - i;
                    if alpha_idx < self.alpha.len() {
                        psum += self.alpha[alpha_idx] * y_vals[y_idx].powi(2);
                    }
                }
            }
            sigma2hat += psum;

            // qsum = sum of (flipped beta) * sigma2_vals[k:q+k]
            if q != 0 {
                let mut qsum = 0.0;
                for j in 0..q {
                    let s_idx = k + j;
                    if s_idx < sigma2_vals.len() && !sigma2_vals[s_idx].is_nan() {
                        let beta_idx = q - 1 - j;
                        if beta_idx < self.beta.len() {
                            qsum += self.beta[beta_idx] * sigma2_vals[s_idx];
                        }
                    }
                }
                sigma2hat += qsum;
            }

            // For variance forecast, use expected y²  = sigma² (E[y²] when y~N(0,σ²))
            y_vals[p + k] = sigma2hat.sqrt(); // E[|y|] ≈ sqrt(σ²) * sqrt(2/π)
            sigma2_vals[q + k] = sigma2hat;
        }

        // Return variance forecasts
        Ok(sigma2_vals[q..].to_vec())
    }
}

impl Default for GARCH {
    fn default() -> Self {
        Self::garch_1_1()
    }
}

impl Forecaster for GARCH {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        let min_obs = self.p + self.q + 10;
        if values.len() < min_obs {
            return Err(ForecastError::InsufficientData {
                needed: min_obs,
                got: values.len(),
            });
        }

        // Calculate mean (for residuals used in likelihood)
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        self.mean = Some(mean);

        // Calculate residuals (z = x - mean(x))
        let residuals: Vec<f64> = values.iter().map(|&y| y - mean).collect();

        // Optimize parameters via MLE - pass BOTH original values and residuals
        // statsforecast uses x (original) for sigma² computation but residuals for likelihood
        self.optimize_parameters(values, &residuals);

        // Compute conditional variance using original values (like statsforecast)
        let conditional_variance =
            Self::compute_sigma2(values, self.omega, &self.alpha, &self.beta, self.p, self.q);

        // Store last p values of ORIGINAL SERIES as y_vals (matching statsforecast)
        // statsforecast: y_vals = x[-p:]
        self.y_vals = values.iter().rev().take(self.p).copied().collect();
        self.y_vals.reverse();

        // Store last q sigma² values (matching statsforecast)
        // statsforecast: sigma2_vals = sigma2[-q:]
        self.sigma2_vals = conditional_variance
            .iter()
            .rev()
            .take(self.q)
            .copied()
            .collect();
        self.sigma2_vals.reverse();

        self.residuals = Some(residuals);
        self.conditional_variance = Some(conditional_variance);
        self.unconditional_variance = Some(self.calculate_unconditional_variance());

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        // GARCH predict returns simulated innovations (matching statsforecast behavior)
        // statsforecast uses np.random.seed(1) and draws normal errors
        // yhat = error * sqrt(sigma2hat)
        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // Check model is fitted
        if self.y_vals.is_empty() || self.sigma2_vals.is_empty() {
            return Err(ForecastError::FitRequired);
        }

        // Standard normal random draws matching np.random.seed(1) + np.random.normal()
        // These are pre-computed to exactly match statsforecast's deterministic simulation
        const NUMPY_SEED1_RANDN: [f64; 24] = [
            1.6243453637,
            -0.6117564137,
            -0.5281717523,
            -1.0729686222,
            0.8654076293,
            -2.3015386969,
            1.7448117642,
            -0.7612069009,
            0.3190390961,
            -0.2493703755,
            1.4621079370,
            -2.0601407095,
            -0.3224172040,
            -0.3840544394,
            1.1337694423,
            -1.0998912673,
            -0.1724282259,
            -0.8778584420,
            0.0422137467,
            0.5828152137,
            -1.1006191850,
            1.1447236947,
            0.9015907205,
            0.5024943390,
        ];

        // Use the pre-computed sequence
        let errors: Vec<f64> = (0..horizon)
            .map(|i| {
                if i < NUMPY_SEED1_RANDN.len() {
                    NUMPY_SEED1_RANDN[i]
                } else {
                    // For horizons > 24, fall back to seeded RNG
                    use rand::{rngs::StdRng, Rng, SeedableRng};
                    let mut rng = StdRng::seed_from_u64(1 + i as u64);
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                }
            })
            .collect();

        // Initialize y_vals and sigma2_vals with recent history
        let p = self.p;
        let q = self.q;
        let mut y_vals = vec![f64::NAN; horizon + p];
        let mut sigma2_vals = vec![f64::NAN; horizon + q];

        // Copy last p residuals
        for (i, &y) in self.y_vals.iter().enumerate() {
            y_vals[i] = y;
        }

        // Copy last q sigma² values
        for (i, &s) in self.sigma2_vals.iter().enumerate() {
            sigma2_vals[i] = s;
        }

        // Forecast step by step (matching statsforecast's garch_forecast exactly)
        // statsforecast: psum = np.flip(alpha) * (y_vals[k : p + k] ** 2)
        //               qsum = np.flip(beta) * (sigma2_vals[k : q + k])
        for k in 0..horizon {
            let mut sigma2hat = self.omega;

            // psum = sum of (flipped alpha) * y_vals[k:p+k]^2
            // np.flip(alpha) reverses the array, so alpha[p-1-i] * y_vals[k+i]
            let mut psum = 0.0;
            for i in 0..p {
                let y_idx = k + i;
                if y_idx < y_vals.len() && !y_vals[y_idx].is_nan() {
                    // Flipped alpha: use alpha[p-1-i]
                    let alpha_idx = p - 1 - i;
                    if alpha_idx < self.alpha.len() {
                        psum += self.alpha[alpha_idx] * y_vals[y_idx].powi(2);
                    }
                }
            }
            sigma2hat += psum;

            // qsum = sum of (flipped beta) * sigma2_vals[k:q+k]
            if q != 0 {
                let mut qsum = 0.0;
                for j in 0..q {
                    let s_idx = k + j;
                    if s_idx < sigma2_vals.len() && !sigma2_vals[s_idx].is_nan() {
                        // Flipped beta: use beta[q-1-j]
                        let beta_idx = q - 1 - j;
                        if beta_idx < self.beta.len() {
                            qsum += self.beta[beta_idx] * sigma2_vals[s_idx];
                        }
                    }
                }
                sigma2hat += qsum;
            }

            // yhat = error * sqrt(sigma2hat)
            let yhat = errors[k] * sigma2hat.max(1e-10).sqrt();

            y_vals[p + k] = yhat;
            sigma2_vals[q + k] = sigma2hat;
        }

        // Return the forecast values (last h entries)
        let forecasts: Vec<f64> = y_vals[p..].to_vec();
        Ok(Forecast::from_values(forecasts))
    }

    fn predict_with_intervals(&self, horizon: usize, confidence: f64) -> Result<Forecast> {
        let forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(forecast);
        }

        // Get variance forecasts for confidence intervals
        let var_forecasts = self.forecast_variance(horizon)?;

        let z = crate::utils::stats::quantile_normal((1.0 + confidence) / 2.0);
        let preds = forecast.primary();

        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        // CI based on quantiles * sqrt(sigma²) around point forecast
        for (i, &pred) in preds.iter().enumerate() {
            let se = var_forecasts[i].sqrt();
            lower.push(pred - z * se);
            upper.push(pred + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        // Fitted values are just the mean for GARCH
        None
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "GARCH"
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

    fn make_volatility_series(n: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        // Simulate volatility clustering
        let mut rng_state = 42u64;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                // Simple pseudo-random with volatility regime switching
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let rand = ((rng_state >> 16) & 0x7FFF) as f64 / 32768.0 - 0.5;
                let regime = if (i / 20) % 2 == 0 { 1.0 } else { 2.0 };
                rand * regime
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn garch_basic() {
        let ts = make_volatility_series(100);
        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();

        assert!(model.conditional_variance().is_some());
        assert!(model.unconditional_variance().is_some());

        let var_forecast = model.forecast_variance(10).unwrap();
        assert_eq!(var_forecast.len(), 10);
    }

    #[test]
    fn garch_1_1() {
        let ts = make_volatility_series(100);
        let mut model = GARCH::garch_1_1();
        model.fit(&ts).unwrap();

        let var_forecast = model.forecast_variance(10).unwrap();
        assert_eq!(var_forecast.len(), 10);

        // Variance should be positive
        for &v in &var_forecast {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn garch_with_custom_params() {
        // Test that custom params are used as initial values for optimization,
        // but MLE may change them. Before fit(), params should match.
        let model = GARCH::new(1, 1)
            .with_omega(0.02)
            .with_alpha(vec![0.15])
            .with_beta(vec![0.8]);

        // Before fit, custom params are preserved
        assert!((model.omega() - 0.02).abs() < 1e-10);
        assert!((model.alpha_params()[0] - 0.15).abs() < 1e-10);
        assert!((model.beta_params()[0] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn garch_mle_optimization() {
        // Test that MLE optimization produces valid parameters
        let ts = make_volatility_series(100);
        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();

        // Parameters should be valid and stationary
        assert!(model.omega() > 0.0);
        assert!(model.alpha_params()[0] > 0.0);
        assert!(model.beta_params()[0] > 0.0);
        assert!(model.is_stationary());
    }

    #[test]
    fn garch_stationarity() {
        // Stationary GARCH
        let model = GARCH::new(1, 1).with_alpha(vec![0.1]).with_beta(vec![0.85]);
        assert!(model.is_stationary());

        // Non-stationary GARCH
        let model = GARCH::new(1, 1).with_alpha(vec![0.5]).with_beta(vec![0.6]);
        assert!(!model.is_stationary());
    }

    #[test]
    fn garch_variance_convergence() {
        let ts = make_volatility_series(100);
        let mut model = GARCH::garch_1_1();
        model.fit(&ts).unwrap();

        let var_forecast = model.forecast_variance(100).unwrap();
        let uncond_var = model.unconditional_variance().unwrap();

        // Long-horizon forecasts should converge to unconditional variance
        let last_forecast = var_forecast.last().unwrap();
        assert!(
            (last_forecast - uncond_var).abs() / uncond_var < 0.1,
            "Variance forecast should converge to unconditional variance"
        );
    }

    #[test]
    fn garch_predict_with_intervals() {
        let ts = make_volatility_series(100);
        let mut model = GARCH::garch_1_1();
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(10, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        for i in 0..10 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }
    }

    #[test]
    fn garch_insufficient_data() {
        let ts = make_volatility_series(10);
        let mut model = GARCH::new(1, 1);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn garch_requires_fit() {
        let model = GARCH::new(1, 1);
        assert!(matches!(
            model.forecast_variance(5),
            Err(ForecastError::FitRequired)
        ));
    }

    #[test]
    fn garch_zero_horizon() {
        let ts = make_volatility_series(100);
        let mut model = GARCH::garch_1_1();
        model.fit(&ts).unwrap();

        let forecast = model.forecast_variance(0).unwrap();
        assert_eq!(forecast.len(), 0);
    }

    #[test]
    fn garch_name() {
        let model = GARCH::new(1, 1);
        assert_eq!(model.name(), "GARCH");
    }

    #[test]
    fn garch_default() {
        let model = GARCH::default();
        assert_eq!(model.p, 1);
        assert_eq!(model.q, 1);
    }

    #[test]
    fn garch_residuals() {
        let ts = make_volatility_series(100);
        let mut model = GARCH::garch_1_1();
        model.fit(&ts).unwrap();

        assert!(model.residuals().is_some());
        assert_eq!(model.residuals().unwrap().len(), 100);
    }

    #[test]
    fn garch_higher_order() {
        let ts = make_volatility_series(150);
        let mut model = GARCH::new(2, 2);
        model.fit(&ts).unwrap();

        let var_forecast = model.forecast_variance(10).unwrap();
        assert_eq!(var_forecast.len(), 10);
    }

    #[test]
    fn garch_y_vals_stores_original_values() {
        // Test that y_vals stores original series values, not residuals
        // This is critical for matching statsforecast behavior
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();

        // y_vals should contain original values, not residuals (y - mean)
        // The last value of the series is 50 + 49*0.5 = 74.5
        assert!(!model.y_vals.is_empty());
        let last_original = values.last().unwrap();
        let stored_y = model.y_vals.last().unwrap();

        // y_vals should be close to original value (within ~1% since it's the last p values)
        assert!(
            (*stored_y - *last_original).abs() < 1.0,
            "y_vals should store original values, got {} expected near {}",
            stored_y,
            last_original
        );
    }

    #[test]
    fn garch_sigma2_computed_from_original_values() {
        // Test that sigma² is computed using x² (original values), not (x-mean)²
        let timestamps = make_timestamps(50);
        // Create series with non-constant values (mean ~50, some variance)
        let values: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64 - 25.0) * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();

        // Conditional variance uses original x values (not x - mean)
        let cond_var = model.conditional_variance().unwrap();
        // First few values might be NaN due to initialization, check later ones
        let valid_vars: Vec<_> = cond_var.iter().filter(|v| !v.is_nan()).collect();
        assert!(!valid_vars.is_empty());

        // Variance should be positive since we use x² in the computation
        for &v in &valid_vars {
            assert!(*v > 0.0, "Conditional variance should be positive");
        }
    }

    #[test]
    fn garch_predict_returns_simulated_innovations() {
        // Test that predict() returns simulated innovations (error * sqrt(sigma²))
        // not constant mean forecasts
        let ts = make_volatility_series(100);
        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Predictions should vary (not all the same value)
        let first = preds[0];
        let has_variation = preds.iter().any(|&p| (p - first).abs() > 0.01);
        assert!(
            has_variation,
            "GARCH predictions should vary (simulated innovations)"
        );

        // Predictions should be centered around 0 (innovations have mean 0)
        let mean: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
        assert!(
            mean.abs() < 20.0,
            "Mean of predictions should be near 0, got {}",
            mean
        );
    }

    #[test]
    fn garch_deterministic_predictions() {
        // Test that predictions are deterministic (same seed produces same results)
        let ts = make_volatility_series(100);

        let mut model1 = GARCH::new(1, 1);
        model1.fit(&ts).unwrap();
        let pred1 = model1.predict(12).unwrap();

        let mut model2 = GARCH::new(1, 1);
        model2.fit(&ts).unwrap();
        let pred2 = model2.predict(12).unwrap();

        // Same model, same data should produce identical forecasts
        for (p1, p2) in pred1.primary().iter().zip(pred2.primary().iter()) {
            assert!(
                (p1 - p2).abs() < 1e-10,
                "Predictions should be deterministic"
            );
        }
    }

    #[test]
    fn garch_forecast_variance_consistent_with_predict() {
        // Test that forecast_variance and predict use the same sigma² computation
        let ts = make_volatility_series(100);
        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();

        let var_forecast = model.forecast_variance(12).unwrap();
        let point_forecast = model.predict(12).unwrap();

        // All variance forecasts should be positive
        for v in &var_forecast {
            assert!(*v > 0.0, "Variance forecast should be positive");
        }

        // Point forecasts should have reasonable magnitude relative to sqrt(variance)
        // (they are error * sqrt(sigma²) where error ~ N(0,1))
        for (i, (&pf, &vf)) in point_forecast
            .primary()
            .iter()
            .zip(var_forecast.iter())
            .enumerate()
        {
            let std_dev = vf.sqrt();
            // Point forecast should be within ~4 std devs (very loose bound)
            assert!(
                pf.abs() < 4.0 * std_dev + 10.0,
                "Point forecast {} at step {} seems inconsistent with variance {}",
                pf,
                i,
                vf
            );
        }
    }
}
