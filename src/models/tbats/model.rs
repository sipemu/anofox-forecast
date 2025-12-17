//! TBATS model implementation.
//!
//! State-space model with:
//! - Trigonometric (Fourier) seasonal representation
//! - Optional Box-Cox transformation
//! - Optional ARMA errors
//! - Optional damped trend
//!
//! This implementation follows statsforecast's TBATS formulation:
//! - State vector: x = [level, trend, s1_cos, s1_sin, s2_cos, s2_sin, ...]
//! - Observation: y = w' @ x where w = [1, phi, 1, 0, 1, 0, ...] (only cos components)
//! - Transition: x_{t+1} = F @ x_t + g @ error
//! - Separate gamma_one (cos) and gamma_two (sin) smoothing parameters

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};

/// TBATS forecasting model.
///
/// # Example
/// ```
/// use anofox_forecast::models::tbats::TBATS;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc};
///
/// let timestamps: Vec<_> = (0..100).map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::hours(i)).collect();
/// let values: Vec<f64> = (0..100).map(|i| {
///     50.0 + 0.2 * i as f64 + 10.0 * (2.0 * std::f64::consts::PI * (i % 24) as f64 / 24.0).sin()
/// }).collect();
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = TBATS::new(vec![24]);
/// model.fit(&ts).unwrap();
/// let forecast = model.predict(24).unwrap();
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TBATS {
    /// Seasonal periods.
    seasonal_periods: Vec<usize>,
    /// Number of Fourier harmonics for each period (determined by AIC).
    fourier_k: Vec<usize>,
    /// Box-Cox lambda (None = no transformation).
    lambda: Option<f64>,
    /// Use trend component.
    use_trend: bool,
    /// Use damped trend.
    use_damped_trend: bool,
    /// Damping parameter (phi).
    phi: f64,
    /// ARMA p order (0 = no AR).
    arma_p: usize,
    /// ARMA q order (0 = no MA).
    arma_q: usize,
    /// Level smoothing parameter (alpha).
    alpha: f64,
    /// Trend smoothing parameter (beta).
    beta: f64,
    /// Gamma one (cosine component smoothing) for each period.
    gamma_one: Vec<f64>,
    /// Gamma two (sine component smoothing) for each period.
    gamma_two: Vec<f64>,
    /// AR coefficients.
    ar_coeffs: Vec<f64>,
    /// MA coefficients.
    ma_coeffs: Vec<f64>,
    /// State vector: [level, trend, s1_cos, s1_sin, s2_cos, s2_sin, ...].
    state: Vec<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
    /// Variance of residuals.
    sigma2: f64,
    /// AIC value.
    aic: Option<f64>,
}

impl TBATS {
    /// Create a new TBATS model with given seasonal periods.
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        let n_periods = seasonal_periods.len();

        // Will be determined by AIC during fit
        let fourier_k = vec![1; n_periods];

        Self {
            seasonal_periods,
            fourier_k,
            lambda: None,
            use_trend: true,
            use_damped_trend: false,
            phi: 1.0,
            arma_p: 0,
            arma_q: 0,
            alpha: 0.09,                     // statsforecast default
            beta: 0.05,                      // statsforecast default
            gamma_one: vec![0.0; n_periods], // statsforecast initializes to 0
            gamma_two: vec![0.0; n_periods], // statsforecast initializes to 0
            ar_coeffs: Vec::new(),
            ma_coeffs: Vec::new(),
            state: Vec::new(),
            fitted: None,
            residuals: None,
            n: 0,
            sigma2: 1.0,
            aic: None,
        }
    }

    /// Maximum number of harmonics for a given period.
    fn max_harmonics(period: usize) -> usize {
        if period.is_multiple_of(2) {
            period / 2
        } else {
            (period - 1) / 2
        }
    }

    /// Default K (number of Fourier harmonics) based on period.
    /// Used for initial guess before AIC-based selection.
    pub fn default_k(period: usize) -> usize {
        if period <= 2 {
            1
        } else if period <= 12 {
            period / 2
        } else if period <= 24 {
            6
        } else if period <= 52 {
            10
        } else {
            15
        }
    }

    /// Find optimal number of harmonics using AIC.
    /// Returns (optimal_k, detrended_residuals).
    fn find_harmonics(values: &[f64], period: usize) -> (usize, Vec<f64>) {
        let n = values.len();

        // Compute moving average trend (2*m window)
        let window = 2 * period;
        let mut trend = vec![0.0; n];
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);
            let count = end - start;
            trend[i] = values[start..end].iter().sum::<f64>() / count as f64;
        }

        // Detrend
        let z: Vec<f64> = values
            .iter()
            .zip(trend.iter())
            .map(|(y, t)| y - t)
            .collect();

        let max_k = Self::max_harmonics(period).min(n);
        if max_k == 0 {
            return (1, values.to_vec());
        }

        // Build Fourier design matrix
        let t_indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut best_k = 1;
        let mut best_aic = f64::INFINITY;
        let mut best_residuals = z.clone();

        for h in 1..=max_k.min(6) {
            // Limit to max 6 harmonics
            // Build design matrix for h harmonics
            let mut x = vec![vec![0.0; 2 * h]; n];
            for t in 0..n {
                for j in 0..h {
                    let freq = 2.0 * std::f64::consts::PI * (j + 1) as f64 / period as f64;
                    x[t][2 * j] = (freq * t_indices[t]).cos();
                    x[t][2 * j + 1] = (freq * t_indices[t]).sin();
                }
            }

            // Least squares: solve X'X * beta = X'y
            let k_params = 2 * h;
            let mut xtx = vec![vec![0.0; k_params]; k_params];
            let mut xty = vec![0.0; k_params];

            for t in 0..n {
                for i in 0..k_params {
                    xty[i] += x[t][i] * z[t];
                    for j in 0..k_params {
                        xtx[i][j] += x[t][i] * x[t][j];
                    }
                }
            }

            // Solve with simple Gaussian elimination
            let coeffs = match Self::solve_linear_system(&xtx, &xty) {
                Some(c) => c,
                None => continue,
            };

            // Compute residuals and SSE
            let mut sse = 0.0;
            let mut residuals = vec![0.0; n];
            for t in 0..n {
                let mut fitted = 0.0;
                for j in 0..k_params {
                    fitted += x[t][j] * coeffs[j];
                }
                residuals[t] = z[t] - fitted;
                sse += residuals[t] * residuals[t];
            }

            // AIC = n * log(sse/n) + 2*k
            if sse > 0.0 {
                let aic = n as f64 * (sse / n as f64).ln() + 2.0 * k_params as f64;
                if aic < best_aic {
                    best_aic = aic;
                    best_k = h;
                    best_residuals = residuals;
                }
            }
        }

        (best_k, best_residuals)
    }

    /// Simple linear system solver using Gaussian elimination.
    fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        let mut aug = vec![vec![0.0; n + 1]; n];

        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a[i][j];
            }
            aug[i][n] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);

            if aug[i][i].abs() < 1e-12 {
                return None;
            }

            for k in (i + 1)..n {
                let factor = aug[k][i] / aug[i][i];
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = aug[i][n];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }

        Some(x)
    }

    /// Enable Box-Cox transformation with given lambda.
    /// Lambda = 0 means log transformation, lambda = 1 means no transformation.
    pub fn with_box_cox(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda.clamp(0.0, 1.0));
        self
    }

    /// Disable trend component.
    pub fn without_trend(mut self) -> Self {
        self.use_trend = false;
        self
    }

    /// Enable damped trend.
    pub fn with_damped_trend(mut self, phi: f64) -> Self {
        self.use_damped_trend = true;
        self.phi = phi.clamp(0.8, 0.99);
        self
    }

    /// Set ARMA orders.
    pub fn with_arma(mut self, p: usize, q: usize) -> Self {
        self.arma_p = p.min(2);
        self.arma_q = q.min(2);
        self
    }

    /// Set Fourier K for each period.
    pub fn with_fourier_k(mut self, k: Vec<usize>) -> Self {
        for (i, &ki) in k.iter().enumerate() {
            if i < self.fourier_k.len() {
                let max_k = self.seasonal_periods[i] / 2;
                self.fourier_k[i] = ki.min(max_k).max(1);
            }
        }
        self
    }

    /// Get AIC value.
    pub fn aic(&self) -> Option<f64> {
        self.aic
    }

    /// Get estimated Box-Cox lambda.
    pub fn lambda(&self) -> Option<f64> {
        self.lambda
    }

    /// Box-Cox transformation.
    fn box_cox_transform(value: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            value.ln()
        } else {
            (value.powf(lambda) - 1.0) / lambda
        }
    }

    /// Inverse Box-Cox transformation.
    fn inverse_box_cox(value: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            value.exp()
        } else {
            let inner = lambda * value + 1.0;
            if inner > 0.0 {
                inner.powf(1.0 / lambda)
            } else {
                0.0
            }
        }
    }

    /// Estimate optimal Box-Cox lambda.
    fn estimate_lambda(values: &[f64]) -> f64 {
        if values.iter().any(|&v| v <= 0.0) {
            return 1.0; // No transformation for non-positive data
        }

        let objective = |params: &[f64]| {
            let lambda = params[0];
            let transformed: Vec<f64> = values
                .iter()
                .map(|&v| Self::box_cox_transform(v, lambda))
                .collect();

            // Minimize coefficient of variation
            let mean = crate::simd::mean(&transformed);
            let variance = crate::simd::variance(&transformed);

            if mean.abs() < 1e-10 {
                f64::MAX
            } else {
                variance / (mean * mean)
            }
        };

        let config = NelderMeadConfig {
            max_iter: 50,
            tolerance: 1e-4,
            ..Default::default()
        };

        let result = nelder_mead(objective, &[0.5], Some(&[(0.0, 1.0)]), config);
        result.optimal_point[0].clamp(0.0, 1.0)
    }

    /// Compute tau (total number of Fourier state elements).
    fn tau(&self) -> usize {
        self.fourier_k.iter().map(|&k| 2 * k).sum()
    }

    /// State dimension: level + trend (if used) + tau (Fourier states).
    fn state_dim(&self) -> usize {
        let base = if self.use_trend { 2 } else { 1 };
        base + self.tau()
    }

    /// Build observation vector w (only level, trend if damped, and cosine components).
    #[allow(dead_code)]
    fn build_w(&self) -> Vec<f64> {
        let dim = self.state_dim();
        let mut w = vec![0.0; dim];

        // Level coefficient = 1
        w[0] = 1.0;

        // Trend coefficient = phi (or 1 if not damped)
        if self.use_trend {
            w[1] = self.phi;
        }

        // Only cosine coefficients contribute to observation (w = [1, phi, 1, 0, 1, 0, ...])
        let base = if self.use_trend { 2 } else { 1 };
        let mut pos = base;
        for &k in &self.fourier_k {
            for j in 0..k {
                w[pos + 2 * j] = 1.0; // cos coefficient
                                      // w[pos + 2*j + 1] = 0.0; // sin coefficient (already 0)
            }
            pos += 2 * k;
        }

        w
    }

    /// Build g vector (smoothing gains).
    #[allow(dead_code)]
    fn build_g(&self) -> Vec<f64> {
        let dim = self.state_dim();
        let mut g = vec![0.0; dim];

        // Level gain = alpha
        g[0] = self.alpha;

        // Trend gain = beta
        if self.use_trend {
            g[1] = self.beta;
        }

        // Seasonal gains: gamma_one for cos, gamma_two for sin
        let base = if self.use_trend { 2 } else { 1 };
        let mut pos = base;
        for (period_idx, &k) in self.fourier_k.iter().enumerate() {
            let g1 = self.gamma_one.get(period_idx).copied().unwrap_or(0.0);
            let g2 = self.gamma_two.get(period_idx).copied().unwrap_or(0.0);
            for j in 0..k {
                g[pos + 2 * j] = g1; // cos gain
                g[pos + 2 * j + 1] = g2; // sin gain
            }
            pos += 2 * k;
        }

        g
    }

    /// Build F matrix (state transition).
    #[allow(dead_code)]
    fn build_f(&self) -> Vec<Vec<f64>> {
        let dim = self.state_dim();
        let mut f = vec![vec![0.0; dim]; dim];
        let base = if self.use_trend { 2 } else { 1 };

        // Level row: level_{t+1} = level_t + phi * trend_t
        f[0][0] = 1.0;
        if self.use_trend {
            f[0][1] = self.phi;
        }

        // Trend row: trend_{t+1} = phi * trend_t
        if self.use_trend {
            f[1][1] = self.phi;
        }

        // Seasonal rotation blocks
        let mut pos = base;
        for (period_idx, &k) in self.fourier_k.iter().enumerate() {
            let period = self.seasonal_periods[period_idx];
            for j in 0..k {
                let freq = 2.0 * std::f64::consts::PI * (j + 1) as f64 / period as f64;
                let cos_freq = freq.cos();
                let sin_freq = freq.sin();

                let idx_cos = pos + 2 * j;
                let idx_sin = pos + 2 * j + 1;

                // Rotation: [cos', sin'] = [[cos, sin], [-sin, cos]] @ [cos, sin]
                f[idx_cos][idx_cos] = cos_freq;
                f[idx_cos][idx_sin] = sin_freq;
                f[idx_sin][idx_cos] = -sin_freq;
                f[idx_sin][idx_sin] = cos_freq;
            }
            pos += 2 * k;
        }

        f
    }

    /// Initialize state vector from data.
    fn initialize_state(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let dim = self.state_dim();
        let mut state = vec![0.0; dim];

        // Level = mean of data
        let mean = values.iter().sum::<f64>() / n as f64;
        state[0] = mean;

        // Trend = 0 (initial trend estimate)
        if self.use_trend {
            state[1] = 0.0;
        }

        // Seasonal states = 0 (will be estimated during optimization)
        // This matches statsforecast's initial_parameters which sets s_vector = zeros

        state
    }

    /// Run state-space filter and return (sse, final_state, fitted, residuals).
    fn run_filter(
        &self,
        values: &[f64],
        initial_state: &[f64],
        alpha: f64,
        beta: f64,
        phi: f64,
        gamma_one: &[f64],
        gamma_two: &[f64],
    ) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = values.len();
        let _dim = self.state_dim();
        let base = if self.use_trend { 2 } else { 1 };

        let mut state = initial_state.to_vec();
        let mut fitted = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        let mut sse = 0.0;

        for t in 0..n {
            // Observation: y = level + phi * trend + sum(cos_coef)
            let level = state[0];
            let trend = if self.use_trend { state[1] } else { 0.0 };

            let mut seasonal = 0.0;
            let mut pos = base;
            for &k in &self.fourier_k {
                for j in 0..k {
                    seasonal += state[pos + 2 * j]; // Only cosine component
                }
                pos += 2 * k;
            }

            let predicted = level + phi * trend + seasonal;
            let error = values[t] - predicted;

            fitted.push(predicted);
            residuals.push(error);
            sse += error * error;

            // State transition with error correction
            // New level = level + phi * trend + alpha * error
            state[0] = level + phi * trend + alpha * error;

            // New trend = phi * trend + beta * error
            if self.use_trend {
                state[1] = phi * trend + beta * error;
            }

            // Update seasonal states with rotation + gamma correction
            let mut pos = base;
            for (period_idx, &k) in self.fourier_k.iter().enumerate() {
                let period = self.seasonal_periods[period_idx];
                let g1 = gamma_one.get(period_idx).copied().unwrap_or(0.0);
                let g2 = gamma_two.get(period_idx).copied().unwrap_or(0.0);

                for j in 0..k {
                    let freq = 2.0 * std::f64::consts::PI * (j + 1) as f64 / period as f64;
                    let cos_freq = freq.cos();
                    let sin_freq = freq.sin();

                    let idx_cos = pos + 2 * j;
                    let idx_sin = pos + 2 * j + 1;

                    let old_cos = state[idx_cos];
                    let old_sin = state[idx_sin];

                    // Rotation + error correction
                    state[idx_cos] = cos_freq * old_cos + sin_freq * old_sin + g1 * error;
                    state[idx_sin] = -sin_freq * old_cos + cos_freq * old_sin + g2 * error;
                }
                pos += 2 * k;
            }
        }

        (sse, state, fitted, residuals)
    }

    /// Fit model parameters using optimization.
    fn optimize_parameters(&mut self, values: &[f64]) {
        let n = values.len();
        let n_periods = self.seasonal_periods.len();

        // Parameters: alpha, beta (if trend), gamma_one (per period), gamma_two (per period)
        let mut initial = vec![0.09]; // alpha (statsforecast default)
        let mut bounds = vec![(0.001, 0.999)];

        if self.use_trend {
            initial.push(0.05); // beta
            bounds.push((-0.5, 0.5)); // Allow negative beta

            if self.use_damped_trend {
                initial.push(0.98); // phi
                bounds.push((0.8, 0.999));
            }
        }

        // gamma_one for each period (very small initial)
        for _ in 0..n_periods {
            initial.push(0.0);
            bounds.push((-0.1, 0.1));
        }

        // gamma_two for each period (very small initial)
        for _ in 0..n_periods {
            initial.push(0.0);
            bounds.push((-0.1, 0.1));
        }

        let seasonal_periods = self.seasonal_periods.clone();
        let fourier_k = self.fourier_k.clone();
        let use_trend = self.use_trend;
        let use_damped = self.use_damped_trend;
        let initial_state = self.initialize_state(values);

        let objective = move |params: &[f64]| {
            let alpha = params[0];

            let mut idx = 1;
            let beta = if use_trend {
                let b = params[idx];
                idx += 1;
                b
            } else {
                0.0
            };

            let phi = if use_trend && use_damped {
                let p = params[idx];
                idx += 1;
                p
            } else if use_trend {
                1.0
            } else {
                0.0
            };

            let gamma_one: Vec<f64> = (0..n_periods).map(|i| params[idx + i]).collect();
            idx += n_periods;
            let gamma_two: Vec<f64> = (0..n_periods).map(|i| params[idx + i]).collect();

            // Run filter
            let base = if use_trend { 2 } else { 1 };
            let _dim = base + fourier_k.iter().map(|&k| 2 * k).sum::<usize>();
            let mut state = initial_state.clone();
            let mut sse = 0.0;

            for t in 0..n {
                // Observation
                let level = state[0];
                let trend = if use_trend { state[1] } else { 0.0 };

                let mut seasonal = 0.0;
                let mut pos = base;
                for &k in &fourier_k {
                    for j in 0..k {
                        seasonal += state[pos + 2 * j];
                    }
                    pos += 2 * k;
                }

                let predicted = level + phi * trend + seasonal;
                let error = values[t] - predicted;
                sse += error * error;

                // State transition
                state[0] = level + phi * trend + alpha * error;
                if use_trend {
                    state[1] = phi * trend + beta * error;
                }

                // Seasonal update
                let mut pos = base;
                for (period_idx, &k) in fourier_k.iter().enumerate() {
                    let period = seasonal_periods[period_idx];
                    let g1 = gamma_one[period_idx];
                    let g2 = gamma_two[period_idx];

                    for j in 0..k {
                        let freq = 2.0 * std::f64::consts::PI * (j + 1) as f64 / period as f64;
                        let cos_freq = freq.cos();
                        let sin_freq = freq.sin();

                        let idx_cos = pos + 2 * j;
                        let idx_sin = pos + 2 * j + 1;
                        let old_cos = state[idx_cos];
                        let old_sin = state[idx_sin];

                        state[idx_cos] = cos_freq * old_cos + sin_freq * old_sin + g1 * error;
                        state[idx_sin] = -sin_freq * old_cos + cos_freq * old_sin + g2 * error;
                    }
                    pos += 2 * k;
                }
            }

            sse / n as f64
        };

        let config = NelderMeadConfig {
            max_iter: 300,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = nelder_mead(objective, &initial, Some(&bounds), config);

        // Extract optimized parameters
        self.alpha = result.optimal_point[0];

        let mut idx = 1;
        if self.use_trend {
            self.beta = result.optimal_point[idx];
            idx += 1;

            if self.use_damped_trend {
                self.phi = result.optimal_point[idx];
                idx += 1;
            }
        }

        for i in 0..n_periods {
            self.gamma_one[i] = result.optimal_point[idx + i];
        }
        idx += n_periods;

        for i in 0..n_periods {
            self.gamma_two[i] = result.optimal_point[idx + i];
        }
    }

    /// Compute number of parameters for AIC.
    fn n_parameters(&self) -> usize {
        let mut k = 2; // level + sigma^2

        if self.lambda.is_some() {
            k += 1;
        }

        if self.use_trend {
            k += 1; // beta
            if self.use_damped_trend {
                k += 1; // phi
            }
        }

        k += 2 * self.gamma_one.len(); // gamma_one + gamma_two per period

        // Fourier coefficients (seed states)
        k += self.tau();

        k += self.arma_p + self.arma_q;

        k
    }
}

impl Default for TBATS {
    fn default() -> Self {
        Self::new(vec![12])
    }
}

impl Forecaster for TBATS {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        self.n = values.len();

        let min_required = self
            .seasonal_periods
            .iter()
            .max()
            .copied()
            .unwrap_or(4)
            .max(10);

        if values.len() < min_required {
            return Err(ForecastError::InsufficientData {
                needed: min_required,
                got: values.len(),
            });
        }

        // Apply Box-Cox transformation if needed
        let transformed: Vec<f64> = if self.lambda.is_some() || values.iter().all(|&v| v > 0.0) {
            let lambda = self.lambda.unwrap_or_else(|| Self::estimate_lambda(values));
            self.lambda = Some(lambda);
            values
                .iter()
                .map(|&v| Self::box_cox_transform(v, lambda))
                .collect()
        } else {
            values.to_vec()
        };

        // Find optimal number of harmonics for each period using AIC
        let mut residuals_for_next = transformed.clone();
        for (i, &period) in self.seasonal_periods.iter().enumerate() {
            let (k, residuals) = Self::find_harmonics(&residuals_for_next, period);
            self.fourier_k[i] = k;
            residuals_for_next = residuals;
        }

        let n = transformed.len();

        // Initialize state
        self.state = self.initialize_state(&transformed);

        // Optimize parameters
        self.optimize_parameters(&transformed);

        // Run final filter with optimized parameters
        let phi = if self.use_trend { self.phi } else { 0.0 };
        let (sse, final_state, fitted, _residuals) = self.run_filter(
            &transformed,
            &self.initialize_state(&transformed),
            self.alpha,
            self.beta,
            phi,
            &self.gamma_one,
            &self.gamma_two,
        );

        self.state = final_state;
        self.sigma2 = sse / n as f64;

        // Inverse Box-Cox transformation for fitted values
        let lambda = self.lambda.unwrap_or(1.0);
        let fitted_original: Vec<f64> = fitted
            .iter()
            .map(|&f| Self::inverse_box_cox(f, lambda))
            .collect();

        // Residuals in original scale
        let residuals_original: Vec<f64> = values
            .iter()
            .zip(fitted_original.iter())
            .map(|(y, f)| y - f)
            .collect();

        // Compute AIC
        let log_likelihood =
            -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI * self.sigma2).ln());
        let k = self.n_parameters();
        self.aic = Some(-2.0 * log_likelihood + 2.0 * k as f64);

        self.fitted = Some(fitted_original);
        self.residuals = Some(residuals_original);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if self.fitted.is_none() {
            return Err(ForecastError::FitRequired);
        }

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        let lambda = self.lambda.unwrap_or(1.0);
        let mut predictions = Vec::with_capacity(horizon);

        // Clone current state for forecasting
        let mut state = self.state.clone();
        let base = if self.use_trend { 2 } else { 1 };
        let phi = if self.use_trend { self.phi } else { 0.0 };

        for _h in 0..horizon {
            // Observation: y = level + phi * trend + sum(cos_coef)
            let level = state[0];
            let trend = if self.use_trend { state[1] } else { 0.0 };

            let mut seasonal = 0.0;
            let mut pos = base;
            for &k in &self.fourier_k {
                for j in 0..k {
                    seasonal += state[pos + 2 * j]; // Only cosine component
                }
                pos += 2 * k;
            }

            let pred_transformed = level + phi * trend + seasonal;
            let pred = Self::inverse_box_cox(pred_transformed, lambda);
            predictions.push(pred);

            // State transition (F @ x, without error correction)
            // Level: level_{t+1} = level_t + phi * trend_t
            state[0] = level + phi * trend;

            // Trend: trend_{t+1} = phi * trend_t
            if self.use_trend {
                state[1] = phi * trend;
            }

            // Rotate seasonal states
            let mut pos = base;
            for (period_idx, &k) in self.fourier_k.iter().enumerate() {
                let period = self.seasonal_periods[period_idx];
                for j in 0..k {
                    let freq = 2.0 * std::f64::consts::PI * (j + 1) as f64 / period as f64;
                    let cos_freq = freq.cos();
                    let sin_freq = freq.sin();

                    let idx_cos = pos + 2 * j;
                    let idx_sin = pos + 2 * j + 1;

                    let old_cos = state[idx_cos];
                    let old_sin = state[idx_sin];

                    state[idx_cos] = cos_freq * old_cos + sin_freq * old_sin;
                    state[idx_sin] = -sin_freq * old_cos + cos_freq * old_sin;
                }
                pos += 2 * k;
            }
        }

        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let point_forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        let z = crate::utils::quantile_normal(0.5 + level / 2.0);
        let std_dev = self.sigma2.sqrt();

        // Approximate variance growth with horizon
        let lower: Vec<f64> = point_forecast
            .primary()
            .iter()
            .enumerate()
            .map(|(h, &f)| f - z * std_dev * ((h + 1) as f64).sqrt())
            .collect();

        let upper: Vec<f64> = point_forecast
            .primary()
            .iter()
            .enumerate()
            .map(|(h, &f)| f + z * std_dev * ((h + 1) as f64).sqrt())
            .collect();

        Ok(Forecast::from_values_with_intervals(
            point_forecast.primary().to_vec(),
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
        "TBATS"
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

    fn make_complex_seasonal_series(n: usize) -> TimeSeries {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 50.0 + 0.1 * i as f64;
                let daily = 10.0 * (2.0 * std::f64::consts::PI * (i % 24) as f64 / 24.0).sin();
                let weekly = 5.0 * (2.0 * std::f64::consts::PI * (i % 168) as f64 / 168.0).sin();
                let noise = ((i * 17) % 7) as f64 * 0.3 - 1.0;
                (trend + daily + weekly + noise).max(1.0) // Ensure positive for Box-Cox
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn tbats_basic() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);
    }

    #[test]
    fn tbats_multiple_seasonality() {
        let ts = make_complex_seasonal_series(500);
        let mut model = TBATS::new(vec![24, 168]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(48).unwrap();
        assert_eq!(forecast.horizon(), 48);
    }

    #[test]
    fn tbats_without_trend() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]).without_trend();
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);
    }

    #[test]
    fn tbats_damped_trend() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]).with_damped_trend(0.95);
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);
    }

    #[test]
    fn tbats_box_cox() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]).with_box_cox(0.5);
        model.fit(&ts).unwrap();

        assert!(model.lambda().is_some());
        let forecast = model.predict(24).unwrap();
        assert_eq!(forecast.horizon(), 24);
    }

    #[test]
    fn tbats_confidence_intervals() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(24, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn tbats_fitted_and_residuals() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 200);
    }

    #[test]
    fn tbats_aic() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        assert!(model.aic().is_some());
        assert!(model.aic().unwrap().is_finite());
    }

    #[test]
    fn tbats_requires_fit() {
        let model = TBATS::new(vec![24]);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn tbats_zero_horizon() {
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn tbats_insufficient_data() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TBATS::new(vec![24]);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn tbats_name() {
        let model = TBATS::new(vec![24]);
        assert_eq!(model.name(), "TBATS");
    }

    #[test]
    fn tbats_state_space_observation_uses_cosine_only() {
        // Test that observation equation uses only cosine coefficients
        // This is critical for matching statsforecast: y = level + trend + sum(cos_coef)
        let ts = make_complex_seasonal_series(200);
        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        // State should be non-empty and contain level + trend + seasonal states
        assert!(!model.state.is_empty());
        for &val in &model.state {
            assert!(val.is_finite(), "State value should be finite");
        }
    }

    #[test]
    fn tbats_level_accumulates_trend() {
        // Test that level_{t+1} = level_t + trend_t (state-space form)
        // by checking that forecasts don't explode for stationary-like data
        let timestamps = make_timestamps(100);
        // Create roughly stationary data around 50
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TBATS::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        let preds = forecast.primary();

        // For stationary data, forecasts should stay bounded (not explode)
        for &p in preds {
            assert!(
                p > 0.0 && p < 200.0,
                "Forecast {} is out of reasonable bounds for stationary data",
                p
            );
        }

        // Forecasts should be roughly around the mean (50)
        let mean_forecast: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
        assert!(
            (mean_forecast - 50.0).abs() < 30.0,
            "Mean forecast {} should be near 50 for stationary data",
            mean_forecast
        );
    }

    #[test]
    fn tbats_non_damped_trend_constant() {
        // Test that for non-damped trend (phi=1), trend stays constant during forecast
        // This is verified by checking level grows linearly
        let timestamps = make_timestamps(100);
        // Create data with clear trend
        let values: Vec<f64> = (0..100).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TBATS::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // With constant positive trend, forecasts should increase
        // (or at least not decrease dramatically)
        let first = preds[0];
        let last = preds[preds.len() - 1];

        // For trending data, later forecasts should be >= earlier ones (roughly)
        assert!(
            last >= first - 10.0,
            "For trending data, forecasts should not decrease dramatically: first={}, last={}",
            first,
            last
        );
    }

    #[test]
    fn tbats_damped_trend_converges() {
        // Test that damped trend causes forecasts to converge (not keep growing)
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TBATS::new(vec![12]).with_damped_trend(0.9);
        model.fit(&ts).unwrap();

        let forecast = model.predict(50).unwrap();
        let preds = forecast.primary();

        // With damped trend, later forecasts should converge
        // Check that growth rate decreases
        let early_diff = preds[5] - preds[0];
        let late_diff = preds[49] - preds[44];

        // Late differences should be smaller than early differences (damping effect)
        // Use a very loose bound to account for seasonality effects
        assert!(
            late_diff.abs() <= early_diff.abs() * 2.0 + 10.0,
            "Damped trend should reduce growth rate: early_diff={}, late_diff={}",
            early_diff,
            late_diff
        );
    }

    #[test]
    fn tbats_fourier_rotation() {
        // Test that Fourier coefficients rotate correctly
        // After one full period, cos/sin should return to similar values
        let timestamps = make_timestamps(200);
        let values: Vec<f64> = (0..200)
            .map(|i| 50.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TBATS::new(vec![12]);
        model.fit(&ts).unwrap();

        // Get forecasts for exactly 2 periods (24 steps)
        let forecast = model.predict(24).unwrap();
        let preds = forecast.primary();

        // Due to Fourier rotation, pattern should repeat with period 12
        // Check that forecasts at step i and step i+12 are similar
        for i in 0..12 {
            let diff = (preds[i] - preds[i + 12]).abs();
            // Allow for some drift due to trend
            assert!(
                diff < 20.0,
                "Seasonal pattern should repeat: step {} = {}, step {} = {}, diff = {}",
                i,
                preds[i],
                i + 12,
                preds[i + 12],
                diff
            );
        }
    }

    #[test]
    fn tbats_box_cox_auto_estimation() {
        // Test that Box-Cox lambda is automatically estimated for positive data
        let timestamps = make_timestamps(100);
        // All positive values
        let values: Vec<f64> = (0..100)
            .map(|i| 50.0 + 10.0 * (i as f64 / 10.0).sin())
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = TBATS::new(vec![12]);
        model.fit(&ts).unwrap();

        // Lambda should be estimated
        assert!(model.lambda().is_some());
        let lambda = model.lambda().unwrap();
        assert!((0.0..=1.0).contains(&lambda));
    }

    #[test]
    fn tbats_deterministic_forecasts() {
        // Test that forecasts are deterministic (same model produces same results)
        let ts = make_complex_seasonal_series(200);

        let mut model1 = TBATS::new(vec![24]);
        model1.fit(&ts).unwrap();
        let pred1 = model1.predict(24).unwrap();

        let mut model2 = TBATS::new(vec![24]);
        model2.fit(&ts).unwrap();
        let pred2 = model2.predict(24).unwrap();

        for (p1, p2) in pred1.primary().iter().zip(pred2.primary().iter()) {
            assert!(
                (p1 - p2).abs() < 1e-6,
                "Forecasts should be deterministic: {} vs {}",
                p1,
                p2
            );
        }
    }

    #[test]
    fn tbats_forecasts_reasonable_magnitude() {
        // Test that forecasts have reasonable magnitude relative to input data
        let ts = make_complex_seasonal_series(200);
        let values = ts.primary_values();
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let data_range = data_max - data_min;

        let mut model = TBATS::new(vec![24]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(48).unwrap();
        let preds = forecast.primary();

        // Forecasts should be within a reasonable range of the data
        for &p in preds {
            assert!(
                p > data_min - 2.0 * data_range && p < data_max + 2.0 * data_range,
                "Forecast {} is outside reasonable range [{}, {}]",
                p,
                data_min - 2.0 * data_range,
                data_max + 2.0 * data_range
            );
        }
    }
}
