//! MFLES (Median Fourier Linear Exponential Smoothing).
//!
//! A gradient-boosted time series decomposition method that iteratively fits
//! median, Fourier (seasonal), linear trend, and exponential smoothing components.
//!
//! Reference: statsforecast MFLES implementation
//! - Uses multiplicative mode (log transform) by default for positive series
//! - In additive mode, standardizes data (subtract mean, divide by std)
//! - Boosting: seasonal on every iteration, linear on odd, SES on even (after round 4)
//!
//! Supports exogenous regressors via TimeSeries.regressors.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use std::collections::HashMap;

/// MFLES forecasting model.
///
/// MFLES applies gradient-boosted decomposition following the statsforecast implementation:
/// 1. Median baseline
/// 2. Fourier series for seasonality
/// 3. Linear/piecewise trend (odd rounds)
/// 4. Exponential smoothing ensemble for residuals (even rounds after round 4)
///
/// By default, uses multiplicative mode (log transform) for positive series.
#[derive(Debug, Clone)]
pub struct MFLES {
    /// Seasonal period.
    season_length: usize,
    /// Maximum boosting rounds.
    max_rounds: usize,
    /// Seasonal component learning rate (shrinkage factor).
    seasonal_lr: f64,
    /// Trend component learning rate.
    trend_lr: f64,
    /// Residual smoothing learning rate.
    rs_lr: f64,
    /// Enable robust mode (Siegel repeated medians).
    robust: bool,
    /// Whether to use multiplicative mode (auto-detected if None).
    multiplicative: Option<bool>,
    /// Apply trend penalty.
    trend_penalty: bool,
    /// Fourier order (None for auto).
    fourier_order: Option<usize>,

    // Fitted state
    /// Constant (min value) for multiplicative mode.
    const_val: Option<f64>,
    /// Mean for additive mode.
    mean: Option<f64>,
    /// Std for additive mode.
    std: Option<f64>,
    /// Trend: [last-1, last].
    trend: Option<[f64; 2]>,
    /// Fourier coefficients for seasonal forecasting.
    fourier_coeffs: Option<Vec<f64>>,
    /// Fourier order used.
    fitted_fourier_order: usize,
    /// Penalty for trend (R²).
    penalty: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Original series length.
    n: usize,
    /// Whether multiplicative mode was used.
    is_multiplicative: bool,
    /// Seasonal pattern for tiling (one cycle, statsforecast compatible).
    seasonality: Option<Vec<f64>>,
    /// OLS result for exogenous regressors (if any).
    exog_ols: Option<OLSResult>,
}

impl MFLES {
    /// Create a new MFLES model with given seasonal period.
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        let season_length = seasonal_periods.first().copied().unwrap_or(12);
        Self {
            season_length,
            max_rounds: 50,
            seasonal_lr: 0.9,
            trend_lr: 0.9,
            rs_lr: 1.0,
            robust: false,
            multiplicative: None,
            trend_penalty: true,
            fourier_order: None,
            const_val: None,
            mean: None,
            std: None,
            trend: None,
            fourier_coeffs: None,
            fitted_fourier_order: 0,
            penalty: None,
            fitted: None,
            residuals: None,
            n: 0,
            is_multiplicative: false,
            seasonality: None,
            exog_ols: None,
        }
    }

    /// Set maximum boosting rounds.
    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds.max(1);
        self
    }

    /// Set seasonal learning rate.
    pub fn with_seasonal_lr(mut self, lr: f64) -> Self {
        self.seasonal_lr = lr.clamp(0.01, 1.0);
        self
    }

    /// Set trend learning rate.
    pub fn with_trend_lr(mut self, lr: f64) -> Self {
        self.trend_lr = lr.clamp(0.01, 1.0);
        self
    }

    /// Enable robust mode.
    pub fn robust(mut self) -> Self {
        self.robust = true;
        self
    }

    /// Force multiplicative mode.
    pub fn multiplicative(mut self, value: bool) -> Self {
        self.multiplicative = Some(value);
        self
    }

    /// Debug: get internal state (for debugging)
    pub fn debug_state(&self) -> (Option<[f64; 2]>, Option<f64>, Option<&[f64]>, bool) {
        (
            self.trend,
            self.penalty,
            self.seasonality.as_deref(),
            self.is_multiplicative,
        )
    }

    /// Debug: get number of rounds completed
    pub fn debug_rounds(&self) -> usize {
        self.n // Using n as a proxy - actual rounds aren't stored
    }

    /// Extended debug: get component values
    pub fn debug_components(&self) -> Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
        // Returns: (median_val, linear_component_last3, seasonal_component_last3, ses_component_last3)
        // This requires storing components, which we don't currently do
        // For now return None
        None
    }

    /// Compute median of a slice.
    fn median_scalar(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = values.iter().filter(|v| v.is_finite()).copied().collect();
        if sorted.is_empty() {
            return 0.0;
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Compute median baseline for MFLES (statsforecast compatible).
    /// Without seasonal_period: returns single median for all values.
    /// With seasonal_period: computes median per season period.
    fn median(values: &[f64], seasonal_period: Option<usize>) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return vec![];
        }

        match seasonal_period {
            None => {
                // No seasonal: single median for all
                let med = Self::median_scalar(values);
                vec![med; n]
            }
            Some(period) if period > 0 => {
                // With seasonal: compute median per period, then tile
                let full_periods = n / period;
                let resid = n % period;

                let mut result = Vec::with_capacity(n);

                // Compute median for each full period
                for p in 0..full_periods {
                    let start = p * period;
                    let end = start + period;
                    let period_median = Self::median_scalar(&values[start..end]);
                    result.extend(std::iter::repeat_n(period_median, period));
                }

                // Handle remainder
                if resid > 0 {
                    // Use median of last seasonal_period values
                    let remainder_median = Self::median_scalar(&values[n - period..]);
                    result.extend(std::iter::repeat_n(remainder_median, resid));
                }

                result
            }
            _ => {
                // Invalid period: fallback to scalar median
                let med = Self::median_scalar(values);
                vec![med; n]
            }
        }
    }

    /// Auto-determine fourier order based on period (statsforecast compatible).
    fn set_fourier(period: usize) -> usize {
        // statsforecast implementation:
        // if period < 10: fourier = 5
        // elif period < 70: fourier = 10
        // else: fourier = 15
        if period < 10 {
            5
        } else if period < 70 {
            10
        } else {
            15
        }
    }

    /// Calculate coefficient of variation (statsforecast compatible).
    fn calc_cov(y: &[f64], multiplicative: bool) -> f64 {
        if y.is_empty() {
            return 0.0;
        }

        let std_val = {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            let variance = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64;
            variance.sqrt()
        };

        if multiplicative {
            // source http://medcraveonline.com/MOJPB/MOJPB-06-00200.pdf
            // res = np.sqrt(np.exp(np.log(10) * (np.std(y) ** 2) - 1))
            let log10 = 10.0_f64.ln();
            (log10 * std_val.powi(2) - 1.0).exp().sqrt().max(0.0)
        } else {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            if mean.abs() > 1e-10 {
                std_val / mean.abs()
            } else {
                std_val
            }
        }
    }

    /// Generate Fourier series matrix [n x (2*order)] (statsforecast compatible).
    fn get_fourier_series(n: usize, period: usize, order: usize) -> Vec<Vec<f64>> {
        let order = order.min(period / 2).max(1);
        let mut series = Vec::with_capacity(2 * order);

        // statsforecast uses:
        // x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
        // t = np.arange(1, length + 1)  # Note: starts at 1, not 0!
        // return np.hstack([np.cos(x), np.sin(x)])  # cos first, then sin

        for k in 1..=order {
            let freq = 2.0 * std::f64::consts::PI * k as f64 / period as f64;

            // Cosine component (statsforecast puts cos first)
            // t ranges from 1 to n (inclusive), not 0 to n-1
            let cos_component: Vec<f64> = (1..=n).map(|t| (freq * t as f64).cos()).collect();
            series.push(cos_component);

            // Sine component
            let sin_component: Vec<f64> = (1..=n).map(|t| (freq * t as f64).sin()).collect();
            series.push(sin_component);
        }

        series
    }

    /// OLS fit: X @ (X'X)^-1 @ X' @ y
    /// Returns (fitted_values, coefficients)
    fn ols_with_coeffs(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        if x.is_empty() || n == 0 {
            return (vec![0.0; n], vec![]);
        }

        let k = x.len();

        // Build X'X matrix
        let mut xtx = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                xtx[i][j] = x[i].iter().zip(x[j].iter()).map(|(a, b)| a * b).sum();
            }
        }

        // Build X'y vector
        let xty: Vec<f64> = x
            .iter()
            .map(|col| col.iter().zip(y.iter()).map(|(a, b)| a * b).sum())
            .collect();

        // Add regularization
        for i in 0..k {
            xtx[i][i] += 1e-8;
        }

        // Solve using Cholesky
        let coeffs = match Self::solve_symmetric(&xtx, &xty) {
            Some(c) => c,
            None => return (vec![0.0; n], vec![0.0; k]),
        };

        // Compute fitted values: X @ coeffs
        let mut fitted = vec![0.0; n];
        for (j, coef) in coeffs.iter().enumerate() {
            for (i, val) in x[j].iter().enumerate() {
                fitted[i] += coef * val;
            }
        }

        (fitted, coeffs)
    }

    /// OLS fit: X @ (X'X)^-1 @ X' @ y (fitted values only)
    #[allow(dead_code)]
    fn ols(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        Self::ols_with_coeffs(x, y).0
    }

    /// Solve symmetric positive definite system using Cholesky decomposition.
    fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        if n == 0 || a.len() != n {
            return None;
        }

        // Cholesky decomposition A = L L'
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }

                if i == j {
                    if sum <= 0.0 {
                        return None;
                    }
                    l[i][j] = sum.sqrt();
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }

        // Forward substitution: L y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i][j] * y[j];
            }
            y[i] = sum / l[i][i];
        }

        // Backward substitution: L' x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[j][i] * x[j];
            }
            x[i] = sum / l[i][i];
        }

        Some(x)
    }

    /// Fast OLS for linear trend.
    fn fast_ols(y: &[f64]) -> Vec<f64> {
        let n = y.len();
        if n < 2 {
            return y.to_vec();
        }

        let n_f = n as f64;
        // x = 0, 1, 2, ..., n-1
        let x_sum = n_f * (n_f - 1.0) / 2.0;
        let x_sq_sum = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
        let y_sum: f64 = y.iter().sum();
        let xy_sum: f64 = y.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();

        let denom = n_f * x_sq_sum - x_sum * x_sum;
        if denom.abs() < 1e-10 {
            return vec![y_sum / n_f; n];
        }

        let slope = (n_f * xy_sum - x_sum * y_sum) / denom;
        let intercept = (y_sum - slope * x_sum) / n_f;

        (0..n).map(|i| intercept + slope * i as f64).collect()
    }

    /// Siegel repeated medians for robust regression.
    fn siegel_repeated_medians(y: &[f64]) -> Vec<f64> {
        let n = y.len();
        if n < 2 {
            return y.to_vec();
        }

        // Sample for efficiency
        let max_samples = 100;
        let step = if n > max_samples { n / max_samples } else { 1 };

        let mut all_slopes = Vec::new();

        for i in (0..n).step_by(step) {
            let mut slopes_for_i = Vec::new();
            for j in (0..n).step_by(step) {
                if i != j {
                    let slope = (y[j] - y[i]) / (j as f64 - i as f64);
                    if slope.is_finite() {
                        slopes_for_i.push(slope);
                    }
                }
            }
            if !slopes_for_i.is_empty() {
                all_slopes.push(Self::median_scalar(&slopes_for_i));
            }
        }

        let slope = if all_slopes.is_empty() {
            0.0
        } else {
            Self::median_scalar(&all_slopes)
        };

        let intercepts: Vec<f64> = y
            .iter()
            .enumerate()
            .step_by(step)
            .map(|(i, &yi)| yi - slope * i as f64)
            .collect();
        let intercept = Self::median_scalar(&intercepts);

        (0..n).map(|i| intercept + slope * i as f64).collect()
    }

    /// Rolling mean with given window size.
    /// First `window` values are left unchanged (like statsforecast).
    fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
        let n = values.len();
        if n == 0 || window == 0 {
            return values.to_vec();
        }

        let mut result = values.to_vec();

        // First `window` values stay unchanged
        // Starting from index `window`, compute rolling mean
        for i in window..n {
            let sum: f64 = values[(i - window + 1)..=i].iter().sum();
            result[i] = sum / window as f64;
        }

        result
    }

    /// SES ensemble or rolling mean (statsforecast compatible).
    /// When smooth=true: uses EWM ensemble with multiple alpha values.
    /// When smooth=false: uses rolling mean (default behavior).
    fn ses_ensemble(
        values: &[f64],
        min_alpha: f64,
        max_alpha: f64,
        smooth: bool,
        order: usize,
    ) -> Vec<f64> {
        let n = values.len();
        if n == 0 {
            return Vec::new();
        }

        if smooth {
            // EWM ensemble mode
            // Generate alphas from min to max in steps of 0.05 (statsforecast style)
            let mut alphas = Vec::new();
            let mut alpha = min_alpha;
            while alpha < max_alpha {
                alphas.push(alpha);
                alpha += 0.05;
            }

            if alphas.is_empty() {
                alphas.push(min_alpha);
            }

            let mut ensemble = vec![0.0; n];

            // For each alpha, compute exponentially weighted mean
            for &alpha in &alphas {
                let mut ewm = vec![0.0; n];
                ewm[0] = values[0];
                for i in 1..n {
                    ewm[i] = alpha * values[i] + (1.0 - alpha) * ewm[i - 1];
                }

                for i in 0..n {
                    ensemble[i] += ewm[i];
                }
            }

            // Average across all alphas
            let num_alphas = alphas.len() as f64;
            for val in ensemble.iter_mut() {
                *val /= num_alphas;
            }

            ensemble
        } else {
            // Rolling mean mode (statsforecast default when smoother=False)
            let window = order + 1;
            let mut result = Self::rolling_mean(values, window);
            // First `window` values stay as original (statsforecast behavior)
            for i in 0..window.min(n) {
                result[i] = values[i];
            }
            result
        }
    }

    /// Calculate MSE.
    fn calc_mse(y: &[f64], fitted: &[f64]) -> f64 {
        if y.is_empty() {
            return 0.0;
        }
        y.iter()
            .zip(fitted.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y.len() as f64
    }

    /// Calculate R² for trend penalty.
    fn calc_rsq(residuals: &[f64], trend: &[f64]) -> f64 {
        let ss_res: f64 = residuals
            .iter()
            .zip(trend.iter())
            .map(|(r, t)| (r - t).powi(2))
            .sum();
        let mean_r: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let ss_tot: f64 = residuals.iter().map(|r| (r - mean_r).powi(2)).sum();

        if ss_tot < 1e-10 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    /// Resize/tile a vector to target length.
    #[allow(dead_code)]
    fn resize_vec(v: &[f64], target_len: usize) -> Vec<f64> {
        if v.is_empty() {
            return vec![0.0; target_len];
        }
        (0..target_len).map(|i| v[i % v.len()]).collect()
    }
}

impl Default for MFLES {
    fn default() -> Self {
        Self::new(vec![12])
    }
}

impl MFLES {
    /// Internal prediction method that handles both with and without exogenous cases.
    fn predict_internal(
        &self,
        horizon: usize,
        future_regressors: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<Forecast> {
        let trend = self.trend.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
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

            // Predict exogenous contribution
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

        // Calculate slope with penalty (matches statsforecast)
        let last_point = trend[1];
        let mut slope = last_point - trend[0];

        if self.trend_penalty {
            if let Some(penalty) = self.penalty {
                slope *= penalty.max(0.0);
            }
        }

        // Generate forecasts using tiling for seasonality (statsforecast compatible)
        for h in 1..=horizon {
            // Trend extrapolation: slope * h + last_point
            let trend_val = slope * h as f64 + last_point;

            // Seasonal component: tile the stored seasonality
            let seasonal_val = if let Some(ref seasonality) = self.seasonality {
                if !seasonality.is_empty() {
                    // np.resize behavior: cycle through seasonality
                    let idx = (h - 1) % seasonality.len();
                    seasonality[idx]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let pred = trend_val + seasonal_val;

            // Inverse transform
            let mut pred_original = if self.is_multiplicative {
                pred.exp()
            } else {
                let mean_val = self.mean.unwrap_or(0.0);
                let std_val = self.std.unwrap_or(1.0);
                mean_val + pred * std_val
            };

            // Add exogenous contribution
            if let Some(ref exog) = exog_contribution {
                pred_original += exog[h - 1];
            }

            predictions.push(pred_original);
        }

        Ok(Forecast::from_values(predictions))
    }
}

impl Forecaster for MFLES {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let raw_values = series.primary_values();
        let n = raw_values.len();
        self.n = n;

        if n < 4 {
            return Err(ForecastError::InsufficientData { needed: 4, got: n });
        }

        // Check for exogenous regressors
        let values: Vec<f64> = if series.has_regressors() {
            // Extract regressors from TimeSeries
            let regressors = series.all_regressors();

            // Fit OLS: y ~ X
            let ols_result = ols_fit(raw_values, &regressors)?;

            // Calculate residuals (y - OLS prediction)
            let adjusted = ols_residuals(raw_values, &ols_result, &regressors)?;

            // Store OLS result for prediction
            self.exog_ols = Some(ols_result);

            adjusted
        } else {
            self.exog_ols = None;
            raw_values.to_vec()
        };

        let values = &values;

        // Determine multiplicative mode
        let use_multiplicative = match self.multiplicative {
            Some(m) => m,
            None => {
                // Auto: multiplicative if positive and seasonal
                self.season_length > 0 && values.iter().all(|&v| v > 0.0)
            }
        };
        self.is_multiplicative = use_multiplicative;

        // Transform data
        let y: Vec<f64>;
        if use_multiplicative {
            let min_val = values.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
            self.const_val = Some(min_val);
            y = values.iter().map(|&v| v.ln()).collect();
        } else {
            self.const_val = None;
            let mean_val = values.iter().sum::<f64>() / n as f64;
            let std_val = (values.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>() / n as f64)
                .sqrt()
                .max(1e-10);
            self.mean = Some(mean_val);
            self.std = Some(std_val);
            y = values.iter().map(|&v| (v - mean_val) / std_val).collect();
        }

        // Check for constant series
        let all_same = y.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
        if all_same {
            let last_val = *y.last().unwrap();
            self.trend = Some([last_val, last_val]);
            // Zero Fourier coefficients for constant series
            let fourier_order = self
                .fourier_order
                .unwrap_or_else(|| Self::set_fourier(self.season_length));
            let k = fourier_order.min(self.season_length / 2).max(1);
            self.fourier_coeffs = Some(vec![0.0; 2 * k]);
            self.fitted_fourier_order = fourier_order;
            self.trend_penalty = false;

            // Inverse transform for fitted values
            let fitted: Vec<f64> = if use_multiplicative {
                vec![values.last().copied().unwrap_or(0.0); n]
            } else {
                values.to_vec()
            };
            self.fitted = Some(fitted);
            self.residuals = Some(vec![0.0; n]);
            return Ok(());
        }

        let _og_y = y.clone();

        // Fourier order for seasonal fitting
        let fourier_order = self
            .fourier_order
            .unwrap_or_else(|| Self::set_fourier(self.season_length));

        // Initialize components
        let mut fitted = vec![0.0; n];
        let mut median_component = vec![0.0; n];
        let mut seasonal_component = vec![0.0; n];
        let mut linear_component = vec![0.0; n];
        let mut ses_component = vec![0.0; n];

        // Initialize trend accumulator (statsforecast style: accumulates during boosting)
        // trend[0] and trend[1] start with median, then accumulate linear[-2:] and ses[-1]
        let mut trend_accum = [0.0f64; 2];

        // Initialize Fourier coefficients accumulator
        let k = fourier_order.min(self.season_length / 2).max(1);
        let mut fourier_coeffs_accum = vec![0.0; 2 * k];

        // Fit initial median
        // statsforecast defaults to moving_medians=False, so we use a single scalar median
        // This is important for matching statsforecast's penalty (R²) calculation
        let median_fitted = Self::median(&y, None);
        for i in 0..n {
            median_component[i] = median_fitted[i];
            fitted[i] = median_fitted[i];
        }

        // Initialize trend with median (last value, duplicated)
        trend_accum[0] = median_fitted[n - 1];
        trend_accum[1] = median_fitted[n - 1];

        // Fourier series for seasonal fitting
        let fourier_series = Self::get_fourier_series(n, self.season_length, fourier_order);

        let mut mse: Option<f64> = None;
        let mut equal_count = 0;
        let mut penalty_val: Option<f64> = None;

        // Robust mode: if not explicitly set, will be auto-detected after round 0
        // Default to false, but may be set to true based on coefficient of variation
        let mut robust_mode = self.robust;
        let cov_threshold = 0.7; // statsforecast default

        // Main boosting loop
        let mut final_round = 0;
        for round in 0..self.max_rounds {
            final_round = round;
            let resids: Vec<f64> = y.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();

            // Auto-detect robust mode after round 0 (statsforecast behavior)
            if round == 0 && !self.robust {
                let cov = Self::calc_cov(&resids, use_multiplicative);
                if cov > cov_threshold {
                    robust_mode = true;
                }
            }

            // Check convergence (statsforecast compatible)
            // IMPORTANT: statsforecast checks equal==6 BEFORE incrementing
            // This means 6 consecutive non-improvements break the loop,
            // but the 6th round still processes before the 7th round checks
            let current_mse = Self::calc_mse(&y, &fitted);
            match mse {
                None => mse = Some(current_mse),
                Some(prev_mse) => {
                    if prev_mse <= current_mse {
                        // Check BEFORE incrementing (statsforecast behavior)
                        if equal_count == 6 {
                            break;
                        }
                        equal_count += 1;
                    } else {
                        mse = Some(current_mse);
                        equal_count = 0;
                    }
                }
            }

            // Fit seasonal component
            if self.season_length > 0 && !fourier_series.is_empty() {
                let (seas, coeffs) = Self::ols_with_coeffs(&fourier_series, &resids);
                let seas_scaled: Vec<f64> = seas.iter().map(|&s| s * self.seasonal_lr).collect();
                let coeffs_scaled: Vec<f64> =
                    coeffs.iter().map(|&c| c * self.seasonal_lr).collect();

                let component_mse = Self::calc_mse(
                    &y,
                    &fitted
                        .iter()
                        .zip(seas_scaled.iter())
                        .map(|(f, s)| f + s)
                        .collect::<Vec<_>>(),
                );

                if mse.is_none_or(|m| m > component_mse) {
                    mse = Some(component_mse);
                    for i in 0..n {
                        fitted[i] += seas_scaled[i];
                        seasonal_component[i] += seas_scaled[i];
                    }

                    // Accumulate Fourier coefficients for forecasting
                    for (i, &c) in coeffs_scaled.iter().enumerate() {
                        if i < fourier_coeffs_accum.len() {
                            fourier_coeffs_accum[i] += c;
                        }
                    }
                }
            }

            // Recompute residuals after seasonal
            let resids: Vec<f64> = y.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();

            // Odd rounds: fit linear trend
            if round % 2 == 1 {
                let tren = if robust_mode {
                    Self::siegel_repeated_medians(&resids)
                } else {
                    Self::fast_ols(&resids)
                };

                let tren_scaled: Vec<f64> = tren.iter().map(|&t| t * self.trend_lr).collect();

                let component_mse = Self::calc_mse(
                    &y,
                    &fitted
                        .iter()
                        .zip(tren_scaled.iter())
                        .map(|(f, t)| f + t)
                        .collect::<Vec<_>>(),
                );

                if mse.is_none_or(|m| m > component_mse) {
                    mse = Some(component_mse);
                    for i in 0..n {
                        fitted[i] += tren_scaled[i];
                        linear_component[i] += tren_scaled[i];
                    }

                    // Accumulate linear component to trend (last 2 values)
                    // statsforecast: self.trend += tren[-2:]
                    if n >= 2 {
                        trend_accum[0] += tren_scaled[n - 2];
                        trend_accum[1] += tren_scaled[n - 1];
                    } else if n == 1 {
                        trend_accum[0] += tren_scaled[0];
                        trend_accum[1] += tren_scaled[0];
                    }

                    // Compute penalty on first linear fit
                    if round == 1 {
                        penalty_val = Some(Self::calc_rsq(&resids, &tren));
                    }
                }
            }
            // Even rounds after round 4: SES ensemble
            else if round > 4 && round % 2 == 0 {
                let resids: Vec<f64> = y.iter().zip(fitted.iter()).map(|(a, b)| a - b).collect();
                // statsforecast default: smoother=False, order=1 (uses rolling_mean with window=2)
                let ses = Self::ses_ensemble(&resids, 0.05, 1.0, false, 1);
                let ses_scaled: Vec<f64> = ses.iter().map(|&s| s * self.rs_lr).collect();

                let component_mse = Self::calc_mse(
                    &y,
                    &fitted
                        .iter()
                        .zip(ses_scaled.iter())
                        .map(|(f, s)| f + s)
                        .collect::<Vec<_>>(),
                );

                // Add round penalty to avoid overfitting
                let round_penalty = 0.0001;
                if mse.is_none_or(|m| m > component_mse + round_penalty * m) {
                    mse = Some(component_mse);
                    for i in 0..n {
                        fitted[i] += ses_scaled[i];
                        ses_component[i] += ses_scaled[i];
                    }

                    // Accumulate SES to trend (scalar added to both)
                    // statsforecast: self.trend += tren[-1]
                    let ses_last = ses_scaled[n - 1];
                    trend_accum[0] += ses_last;
                    trend_accum[1] += ses_last;
                }
            }
        }

        // Store fitted state
        // Use the accumulated trend (statsforecast style):
        // - Initialized with median[n-1]
        // - += linear[-2:] on each accepted linear fit
        // - += ses[-1] (scalar) on each accepted SES fit

        // Suppress unused variable warning
        let _ = final_round;

        self.trend = Some(trend_accum);
        self.fourier_coeffs = Some(fourier_coeffs_accum);
        self.fitted_fourier_order = fourier_order;
        self.penalty = penalty_val;

        // Store seasonality for tiling (statsforecast compatible)
        // statsforecast stores exactly one cycle from the LAST period of seasonal_component
        if self.season_length > 0 && n >= self.season_length {
            let period = self.season_length;
            // Take the last complete cycle
            let last_cycle: Vec<f64> = seasonal_component[n - period..].to_vec();
            self.seasonality = Some(last_cycle);
        } else if self.season_length > 0 {
            // If series is shorter than season, use what we have
            self.seasonality = Some(seasonal_component.clone());
        } else {
            self.seasonality = None;
        }

        // Inverse transform fitted values
        let fitted_original: Vec<f64> = if use_multiplicative {
            fitted.iter().map(|&f| f.exp()).collect()
        } else {
            let mean_val = self.mean.unwrap_or(0.0);
            let std_val = self.std.unwrap_or(1.0);
            fitted.iter().map(|&f| mean_val + f * std_val).collect()
        };

        // Compute residuals
        let residuals: Vec<f64> = values
            .iter()
            .zip(fitted_original.iter())
            .map(|(v, f)| v - f)
            .collect();

        self.fitted = Some(fitted_original);
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
        let point_forecast = self.predict_with_exog(horizon, future_regressors)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        // Compute variance from residuals
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let variance = if residuals.len() > 1 {
            let mean_resid: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
            residuals
                .iter()
                .map(|r| (r - mean_resid).powi(2))
                .sum::<f64>()
                / (residuals.len() - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        let z = crate::utils::quantile_normal(0.5 + level / 2.0);

        let preds = point_forecast.primary();
        let lower: Vec<f64> = preds
            .iter()
            .enumerate()
            .map(|(h, &f)| f - z * std_dev * ((h + 1) as f64).sqrt())
            .collect();
        let upper: Vec<f64> = preds
            .iter()
            .enumerate()
            .map(|(h, &f)| f + z * std_dev * ((h + 1) as f64).sqrt())
            .collect();

        Ok(Forecast::from_values_with_intervals(
            preds.to_vec(),
            lower,
            upper,
        ))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let point_forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        // Compute variance from residuals
        let residuals = self.residuals.as_ref().ok_or(ForecastError::FitRequired)?;
        let variance = if residuals.len() > 1 {
            let mean_resid: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
            residuals
                .iter()
                .map(|r| (r - mean_resid).powi(2))
                .sum::<f64>()
                / (residuals.len() - 1) as f64
        } else {
            1.0
        };
        let std_dev = variance.sqrt();

        let z = crate::utils::quantile_normal(0.5 + level / 2.0);

        // Widening intervals with horizon
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

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted.as_ref()?;
        let residuals = self.residuals.as_ref()?;

        // Compute variance from residuals
        let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

        if valid_residuals.is_empty() {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let n = valid_residuals.len() as f64;
        let variance = crate::simd::sum_of_squares(&valid_residuals) / n;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = crate::utils::quantile_normal(0.5 + level / 2.0);
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
        "MFLES"
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
            .map(|i| {
                let trend = 50.0 + 0.5 * i as f64;
                let seasonal =
                    10.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin();
                let noise = ((i * 17) % 7) as f64 * 0.1 - 0.3;
                trend + seasonal + noise
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn mfles_basic() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_captures_trend() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(24).unwrap();
        let preds = forecast.primary();

        // Should show increasing trend on average
        let first_half_avg: f64 = preds[..12].iter().sum::<f64>() / 12.0;
        let second_half_avg: f64 = preds[12..].iter().sum::<f64>() / 12.0;

        assert!(
            second_half_avg > first_half_avg,
            "Second half average {} should be > first half {}",
            second_half_avg,
            first_half_avg
        );
    }

    #[test]
    fn mfles_robust_mode() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]).robust();
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_confidence_intervals() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(12, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let preds = forecast.primary();
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        for i in 0..12 {
            assert!(lower[i] <= preds[i]);
            assert!(upper[i] >= preds[i]);
        }
    }

    #[test]
    fn mfles_fitted_and_residuals() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        assert!(model.fitted_values().is_some());
        assert!(model.residuals().is_some());
        assert_eq!(model.fitted_values().unwrap().len(), 100);
        assert_eq!(model.residuals().unwrap().len(), 100);
    }

    #[test]
    fn mfles_additive_mode() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]).multiplicative(false);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_requires_fit() {
        let model = MFLES::new(vec![12]);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn mfles_zero_horizon() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn mfles_insufficient_data() {
        let timestamps = make_timestamps(3);
        let values = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MFLES::new(vec![12]);
        assert!(matches!(
            model.fit(&ts),
            Err(ForecastError::InsufficientData { .. })
        ));
    }

    #[test]
    fn mfles_name() {
        let model = MFLES::new(vec![12]);
        assert_eq!(model.name(), "MFLES");
    }

    #[test]
    fn mfles_default() {
        let model = MFLES::default();
        assert_eq!(model.season_length, 12);
    }

    #[test]
    fn mfles_with_max_rounds() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]).with_max_rounds(5);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_with_learning_rates() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12])
            .with_seasonal_lr(0.5)
            .with_trend_lr(0.5);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_learning_rate_clamping() {
        let ts = make_seasonal_series(100, 12);
        // Test clamping of extreme values
        let mut model = MFLES::new(vec![12])
            .with_seasonal_lr(5.0) // Should clamp to 1.0
            .with_trend_lr(-1.0); // Should clamp to 0.01
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_max_rounds_clamping() {
        let ts = make_seasonal_series(100, 12);
        // with_max_rounds(0) should be clamped to 1
        let mut model = MFLES::new(vec![12]).with_max_rounds(0);
        model.fit(&ts).unwrap();

        let forecast = model.predict(6).unwrap();
        assert_eq!(forecast.horizon(), 6);
    }

    #[test]
    fn mfles_multiplicative_auto_detect() {
        // Create positive-only series for multiplicative auto-detection
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 100.0 + 0.5 * i as f64;
                let seasonal =
                    10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin() + 15.0;
                trend + seasonal
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        // Should auto-detect multiplicative mode for positive series
        let (_, _, _, is_mult) = model.debug_state();
        assert!(
            is_mult,
            "Should auto-detect multiplicative mode for positive series"
        );
    }

    #[test]
    fn mfles_constant_series() {
        let timestamps = make_timestamps(50);
        let values = vec![42.0; 50];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        // Constant series should predict constant values
        for &pred in forecast.primary() {
            assert!((pred - 42.0).abs() < 1.0, "Expected ~42, got {}", pred);
        }
    }

    #[test]
    fn mfles_with_negative_values() {
        // Series with negative values forces additive mode
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 0.5 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * (i % 12) as f64 / 12.0).sin();
                trend + seasonal - 30.0 // Ensure some negatives
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let (_, _, _, is_mult) = model.debug_state();
        assert!(
            !is_mult,
            "Should use additive mode for series with negatives"
        );

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_force_multiplicative() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]).multiplicative(true);
        model.fit(&ts).unwrap();

        let (_, _, _, is_mult) = model.debug_state();
        assert!(is_mult, "Should use multiplicative mode when forced");
    }

    #[test]
    fn mfles_debug_functions() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        // Test debug_state
        let (trend, penalty, seasonality, is_mult) = model.debug_state();
        assert!(trend.is_some(), "Trend should be set after fit");
        assert!(penalty.is_some(), "Penalty should be set after fit");
        assert!(seasonality.is_some(), "Seasonality should be set after fit");
        assert!(is_mult);

        // Test debug_rounds
        let rounds = model.debug_rounds();
        assert!(rounds > 0, "Should have processed data");

        // Test debug_components
        let components = model.debug_components();
        assert!(
            components.is_none(),
            "debug_components returns None currently"
        );
    }

    #[test]
    fn mfles_fitted_values_with_intervals() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        let forecast = model.fitted_values_with_intervals(0.95).unwrap();
        assert_eq!(forecast.horizon(), 100);
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let fitted = forecast.primary();
        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        for i in 0..100 {
            assert!(lower[i] <= fitted[i], "Lower should be <= fitted");
            assert!(upper[i] >= fitted[i], "Upper should be >= fitted");
        }
    }

    #[test]
    fn mfles_short_series_less_than_period() {
        // Series shorter than seasonal period
        let timestamps = make_timestamps(8);
        let values: Vec<f64> = (0..8).map(|i| 10.0 + i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MFLES::new(vec![12]); // period=12 but series is 8
        model.fit(&ts).unwrap();

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn mfles_long_horizon_forecast() {
        let ts = make_seasonal_series(100, 12);
        let mut model = MFLES::new(vec![12]);
        model.fit(&ts).unwrap();

        // Test very long horizon
        let forecast = model.predict(120).unwrap();
        assert_eq!(forecast.horizon(), 120);

        // Values should be finite
        for &val in forecast.primary() {
            assert!(val.is_finite(), "Forecast value should be finite");
        }
    }

    #[test]
    fn mfles_different_seasonal_periods() {
        // Test with different seasonal periods to trigger different fourier orders
        for period in [4, 7, 12, 24, 52, 100] {
            let timestamps = make_timestamps(period * 4);
            let values: Vec<f64> = (0..period * 4)
                .map(|i| {
                    let trend = 50.0 + 0.1 * i as f64;
                    let seasonal = 5.0
                        * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin();
                    trend + seasonal
                })
                .collect();
            let ts = TimeSeries::univariate(timestamps, values).unwrap();

            let mut model = MFLES::new(vec![period]);
            model.fit(&ts).unwrap();

            let forecast = model.predict(period).unwrap();
            assert_eq!(forecast.horizon(), period);
        }
    }

    #[test]
    fn mfles_median_scalar_edge_cases() {
        // Empty
        assert_eq!(MFLES::median_scalar(&[]), 0.0);

        // Single element
        assert_eq!(MFLES::median_scalar(&[5.0]), 5.0);

        // Two elements (even)
        assert_eq!(MFLES::median_scalar(&[2.0, 4.0]), 3.0);

        // Three elements (odd)
        assert_eq!(MFLES::median_scalar(&[1.0, 2.0, 3.0]), 2.0);

        // With NaN (should be filtered)
        assert_eq!(MFLES::median_scalar(&[1.0, f64::NAN, 3.0]), 2.0);

        // All NaN
        assert_eq!(MFLES::median_scalar(&[f64::NAN, f64::NAN]), 0.0);
    }

    #[test]
    fn mfles_median_with_seasonal_period() {
        // Test median with seasonal period
        let values: Vec<f64> = (0..24).map(|i| i as f64).collect();

        // Without period
        let med_no_period = MFLES::median(&values, None);
        assert_eq!(med_no_period.len(), 24);
        // Should be constant (global median)
        assert!((med_no_period[0] - med_no_period[12]).abs() < 0.01);

        // With period
        let med_with_period = MFLES::median(&values, Some(12));
        assert_eq!(med_with_period.len(), 24);

        // Invalid period (0)
        let med_zero_period = MFLES::median(&values, Some(0));
        assert_eq!(med_zero_period.len(), 24);

        // Empty values
        let med_empty = MFLES::median(&[], Some(12));
        assert!(med_empty.is_empty());
    }

    #[test]
    fn mfles_set_fourier_orders() {
        // Test auto fourier order determination
        assert_eq!(MFLES::set_fourier(4), 5); // period < 10
        assert_eq!(MFLES::set_fourier(9), 5); // period < 10
        assert_eq!(MFLES::set_fourier(10), 10); // period >= 10 and < 70
        assert_eq!(MFLES::set_fourier(69), 10); // period >= 10 and < 70
        assert_eq!(MFLES::set_fourier(70), 15); // period >= 70
        assert_eq!(MFLES::set_fourier(365), 15); // period >= 70
    }

    #[test]
    fn mfles_calc_cov() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Additive mode CoV
        let cov_add = MFLES::calc_cov(&data, false);
        assert!(cov_add > 0.0);

        // Multiplicative mode CoV
        let cov_mult = MFLES::calc_cov(&data, true);
        assert!(cov_mult >= 0.0);

        // Empty data
        assert_eq!(MFLES::calc_cov(&[], false), 0.0);

        // Data with mean near zero (additive)
        let zero_mean: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let cov_zero_mean = MFLES::calc_cov(&zero_mean, false);
        assert!(cov_zero_mean.is_finite());
    }

    #[test]
    fn mfles_fast_ols_edge_cases() {
        // Single element
        let single = MFLES::fast_ols(&[5.0]);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], 5.0);

        // Two elements
        let two = MFLES::fast_ols(&[1.0, 3.0]);
        assert_eq!(two.len(), 2);
        // Should fit line y = 1 + 2x (at x=0: 1, at x=1: 3)
        assert!((two[0] - 1.0).abs() < 0.01);
        assert!((two[1] - 3.0).abs() < 0.01);

        // Constant data
        let constant = MFLES::fast_ols(&[5.0, 5.0, 5.0, 5.0]);
        for &v in &constant {
            assert!((v - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn mfles_siegel_repeated_medians() {
        // Single element
        let single = MFLES::siegel_repeated_medians(&[5.0]);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], 5.0);

        // Linear data
        let linear: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let fitted = MFLES::siegel_repeated_medians(&linear);
        assert_eq!(fitted.len(), 20);
        // Should closely fit the line
        for (i, &v) in fitted.iter().enumerate() {
            let expected = 2.0 * i as f64 + 1.0;
            assert!(
                (v - expected).abs() < 1.0,
                "i={}: expected {}, got {}",
                i,
                expected,
                v
            );
        }
    }

    #[test]
    fn mfles_rolling_mean_edge_cases() {
        // Empty
        let empty = MFLES::rolling_mean(&[], 3);
        assert!(empty.is_empty());

        // Window = 0
        let zero_window = MFLES::rolling_mean(&[1.0, 2.0, 3.0], 0);
        assert_eq!(zero_window, vec![1.0, 2.0, 3.0]);

        // Normal case
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rolled = MFLES::rolling_mean(&data, 3);
        assert_eq!(rolled.len(), 5);
        // First 3 values unchanged
        assert_eq!(rolled[0], 1.0);
        assert_eq!(rolled[1], 2.0);
        assert_eq!(rolled[2], 3.0);
        // Rolling mean starts at index 3
        assert!((rolled[3] - 3.0).abs() < 0.01); // (2+3+4)/3 = 3
        assert!((rolled[4] - 4.0).abs() < 0.01); // (3+4+5)/3 = 4
    }

    #[test]
    fn mfles_ses_ensemble_modes() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();

        // Rolling mean mode (smooth=false)
        let rolling = MFLES::ses_ensemble(&data, 0.05, 1.0, false, 1);
        assert_eq!(rolling.len(), 20);

        // EWM ensemble mode (smooth=true)
        let ewm = MFLES::ses_ensemble(&data, 0.05, 1.0, true, 1);
        assert_eq!(ewm.len(), 20);

        // Empty data
        let empty = MFLES::ses_ensemble(&[], 0.05, 1.0, false, 1);
        assert!(empty.is_empty());
    }

    #[test]
    fn mfles_ols_edge_cases() {
        // Empty x matrix
        let (fitted, coeffs) = MFLES::ols_with_coeffs(&[], &[1.0, 2.0, 3.0]);
        assert_eq!(fitted.len(), 3);
        assert!(coeffs.is_empty());

        // Empty y
        let x = vec![vec![1.0, 2.0, 3.0]];
        let (fitted, coeffs) = MFLES::ols_with_coeffs(&x, &[]);
        assert!(fitted.is_empty());
        assert!(coeffs.is_empty());
    }

    #[test]
    fn mfles_solve_symmetric_edge_cases() {
        // Empty
        assert!(MFLES::solve_symmetric(&[], &[]).is_none());

        // Mismatched dimensions
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(MFLES::solve_symmetric(&a, &[1.0]).is_none());
    }

    #[test]
    fn mfles_robust_auto_detect_high_cov() {
        // Create a series with high coefficient of variation to trigger robust mode
        let timestamps = make_timestamps(100);
        // Use data that creates high residual variance
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.1;
                // Add large spikes to create high CoV
                if i % 10 == 0 {
                    base * 3.0
                } else {
                    base
                }
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = MFLES::new(vec![12]);
        // Should auto-detect robust mode if CoV > 0.7
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.horizon(), 12);
    }

    #[test]
    fn mfles_resize_vec() {
        // Empty
        let resized = MFLES::resize_vec(&[], 5);
        assert_eq!(resized, vec![0.0; 5]);

        // Resize smaller to larger (tile)
        let resized = MFLES::resize_vec(&[1.0, 2.0, 3.0], 7);
        assert_eq!(resized, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]);

        // Resize to exact multiple
        let resized = MFLES::resize_vec(&[1.0, 2.0], 4);
        assert_eq!(resized, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn mfles_empty_seasonal_periods() {
        let ts = make_seasonal_series(100, 12);
        // Empty seasonal periods should default to 12
        let mut model = MFLES::new(vec![]);
        // season_length would be 0 from unwrap_or(12) on first().copied()
        // Actually looking at the code: .first().copied().unwrap_or(12)
        // empty vec means first() returns None, so unwrap_or(12) gives 12
        model.fit(&ts).unwrap();
        let forecast = model.predict(6).unwrap();
        assert_eq!(forecast.horizon(), 6);
    }

    #[test]
    fn mfles_get_fourier_series() {
        // Test Fourier series generation
        let series = MFLES::get_fourier_series(10, 4, 2);
        // Should have 2 * min(2, 4/2) = 2 * 2 = 4 columns
        assert_eq!(series.len(), 4);
        // Each column should have 10 rows
        for col in &series {
            assert_eq!(col.len(), 10);
        }

        // Values should be bounded [-1, 1] for sin/cos
        for col in &series {
            for &val in col {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn mfles_calc_mse() {
        // Perfect fit
        assert_eq!(MFLES::calc_mse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);

        // Known MSE
        let mse = MFLES::calc_mse(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]);
        assert!((mse - 1.0).abs() < 0.01); // Each error is 1, MSE = 3/3 = 1

        // Empty
        assert_eq!(MFLES::calc_mse(&[], &[]), 0.0);
    }

    #[test]
    fn mfles_calc_rsq() {
        // Perfect fit (R² = 1)
        let rsq = MFLES::calc_rsq(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((rsq - 1.0).abs() < 0.01);

        // No fit (R² = 0 when trend is constant at mean)
        let rsq = MFLES::calc_rsq(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]);
        assert!((rsq - 0.0).abs() < 0.01);

        // Constant residuals (ss_tot near 0)
        let rsq = MFLES::calc_rsq(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0]);
        // When ss_tot < 1e-10, returns 0.0
        assert_eq!(rsq, 0.0);
    }
}
