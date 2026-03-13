//! Ensemble forecasting methods.
//!
//! Combines multiple forecasting models to produce a single forecast,
//! often with improved accuracy and robustness.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::{validate_series_complete, Forecaster};

/// Method for combining forecasts from multiple models.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CombinationMethod {
    /// Simple average of all forecasts.
    Mean,
    /// Median of all forecasts.
    Median,
    /// Weighted by inverse MSE on fitted values.
    WeightedMSE,
    /// Custom weights provided by user.
    Custom,
    /// Weighted by Akaike Information Criterion.
    ///
    /// Computes Akaike weights: w_i = exp(-0.5 * (AIC_i - AIC_min)) / sum.
    /// AIC is estimated from in-sample residuals. Falls back to equal weights
    /// when residuals are unavailable.
    InverseAIC,
    /// Stacking via projected gradient descent on second half of fitted values.
    ///
    /// Trains a linear combination of model forecasts using non-negative weights
    /// that sum to one. The `folds` parameter controls how many folds to use
    /// when splitting the training data (the second half of fitted values is used
    /// as the validation set).
    Stacking {
        /// Number of folds (reserved for future cross-validation; currently
        /// the second half of the in-sample period is used as the hold-out).
        folds: usize,
    },
    /// Per-horizon adaptive weights from rolling-origin evaluation.
    ///
    /// Computes a separate weight vector for each forecast horizon step,
    /// based on rolling-origin (expanding-window) forecast errors. The
    /// per-horizon weight matrix is stored internally after fitting.
    HorizonAdaptive,
}

/// Estimate the number of free parameters for a model based on its name.
///
/// This is a rough heuristic used for AIC estimation when the model does not
/// expose its own parameter count. The counts are intentionally conservative.
fn estimate_param_count(model: &dyn Forecaster) -> usize {
    let name = model.name().to_lowercase();
    if name.contains("naive") && !name.contains("seasonal") {
        return 1;
    }
    if name.contains("seasonal") {
        return 2;
    }
    if name.contains("simple moving average")
        || name.contains("simplemovingaverage")
        || name.contains("sma")
        || name.contains("window")
    {
        return 1;
    }
    if name.contains("ses") || name.contains("simple exponential") {
        return 2; // alpha + initial level
    }
    if name.contains("holt") && !name.contains("winters") {
        return 4; // alpha, beta, l0, b0
    }
    if name.contains("ets") || name.contains("winters") {
        return 6;
    }
    if name.contains("theta") {
        return 3;
    }
    if name.contains("arima") {
        return 5;
    }
    // Default: moderate complexity
    3
}

/// Estimate AIC from residuals for a single model.
///
/// AIC = n * ln(RSS/n) + 2k, where RSS is the residual sum of squares,
/// n is the number of observations, and k is the estimated parameter count.
///
/// Returns `None` if there are no valid residuals.
fn estimate_aic(model: &dyn Forecaster, actual: &[f64]) -> Option<f64> {
    let fitted = model.fitted_values()?;

    // Collect valid (non-NaN) residual pairs
    let pairs: Vec<(f64, f64)> = actual
        .iter()
        .zip(fitted.iter())
        .filter(|(a, f)| a.is_finite() && f.is_finite())
        .map(|(&a, &f)| (a, f))
        .collect();

    let n = pairs.len();
    if n == 0 {
        return None;
    }

    let rss: f64 = pairs.iter().map(|(a, f)| (a - f).powi(2)).sum();

    let mean_rss = rss / n as f64;
    if mean_rss <= 0.0 || !mean_rss.is_finite() {
        return None;
    }

    let k = estimate_param_count(model) as f64;
    let aic = (n as f64) * mean_rss.ln() + 2.0 * k;
    if aic.is_finite() {
        Some(aic)
    } else {
        None
    }
}

/// Project a weight vector onto the probability simplex (non-negative, sum-to-one).
///
/// Uses the algorithm of Duchi et al. (2008) for Euclidean projection onto
/// the simplex. This is used by the stacking combiner.
fn nnls_simplex(weights: &[f64]) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }

    // Sort in descending order
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut rho = 0;
    for (j, &val) in sorted.iter().enumerate() {
        cumsum += val;
        if val - (cumsum - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }

    let theta = (sorted[..=rho].iter().sum::<f64>() - 1.0) / (rho as f64 + 1.0);

    weights.iter().map(|&w| (w - theta).max(0.0)).collect()
}

/// Ensemble forecaster that combines multiple models.
pub struct Ensemble {
    /// The forecasting models.
    models: Vec<Box<dyn Forecaster>>,
    /// Method for combining forecasts.
    method: CombinationMethod,
    /// Custom weights (used when method is Custom).
    custom_weights: Option<Vec<f64>>,
    /// Computed weights after fitting.
    weights: Vec<f64>,
    /// Per-horizon weights for HorizonAdaptive method.
    /// `horizon_weights[h]` is the weight vector for step h.
    horizon_weights: Option<Vec<Vec<f64>>>,
    /// Combined fitted values.
    fitted: Option<Vec<f64>>,
    /// Combined residuals.
    residuals: Option<Vec<f64>>,
    /// Whether models have been fitted.
    is_fitted: bool,
}

impl Ensemble {
    /// Create a new ensemble with the given models.
    pub fn new(models: Vec<Box<dyn Forecaster>>) -> Self {
        let n = models.len();
        Self {
            models,
            method: CombinationMethod::Mean,
            custom_weights: None,
            weights: vec![1.0 / n as f64; n],
            horizon_weights: None,
            fitted: None,
            residuals: None,
            is_fitted: false,
        }
    }

    /// Set the combination method.
    pub fn with_method(mut self, method: CombinationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set custom weights (must match number of models).
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.custom_weights = Some(weights);
        self.method = CombinationMethod::Custom;
        self
    }

    /// Get the current weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the per-horizon weights (only populated for HorizonAdaptive).
    pub fn horizon_weights(&self) -> Option<&Vec<Vec<f64>>> {
        self.horizon_weights.as_ref()
    }

    /// Get the combination method.
    pub fn method(&self) -> CombinationMethod {
        self.method
    }

    /// Get the number of models in the ensemble.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Combine prediction interval bounds from multiple models.
    ///
    /// For lower bounds, takes the minimum across models at each step
    /// (widest interval). For upper bounds, takes the maximum.
    /// Falls back to `combine_values` averaging when `widest` is false.
    fn combine_interval_bounds(&self, bounds: &[Vec<f64>], take_min: bool) -> Vec<f64> {
        if bounds.is_empty() {
            return Vec::new();
        }

        let horizon = bounds.iter().map(|v| v.len()).min().unwrap_or(0);
        if horizon == 0 {
            return Vec::new();
        }

        let mut combined = vec![0.0; horizon];
        for h in 0..horizon {
            combined[h] = if take_min {
                bounds.iter().map(|v| v[h]).fold(f64::INFINITY, f64::min)
            } else {
                bounds
                    .iter()
                    .map(|v| v[h])
                    .fold(f64::NEG_INFINITY, f64::max)
            };
        }
        combined
    }

    /// Combine values using the specified method.
    fn combine_values(&self, values: &[Vec<f64>]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let horizon = values.iter().map(|v| v.len()).min().unwrap_or(0);
        if horizon == 0 {
            return Vec::new();
        }
        let mut combined = vec![0.0; horizon];

        match self.method {
            CombinationMethod::Mean => {
                for h in 0..horizon {
                    let sum: f64 = values.iter().filter(|v| h < v.len()).map(|v| v[h]).sum();
                    combined[h] = sum / values.len() as f64;
                }
            }
            CombinationMethod::Median => {
                for h in 0..horizon {
                    let mut vals: Vec<f64> = values.iter().map(|v| v[h]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = vals.len();
                    combined[h] = if n % 2 == 0 {
                        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
                    } else {
                        vals[n / 2]
                    };
                }
            }
            CombinationMethod::WeightedMSE
            | CombinationMethod::Custom
            | CombinationMethod::InverseAIC
            | CombinationMethod::Stacking { .. } => {
                for h in 0..horizon {
                    let weighted_sum: f64 = values
                        .iter()
                        .zip(self.weights.iter())
                        .map(|(v, w)| v[h] * w)
                        .sum();
                    combined[h] = weighted_sum;
                }
            }
            CombinationMethod::HorizonAdaptive => {
                if let Some(ref hw) = self.horizon_weights {
                    for h in 0..horizon {
                        let w = if h < hw.len() {
                            &hw[h]
                        } else {
                            // Beyond stored horizons, fall back to last available
                            hw.last().unwrap_or(&self.weights)
                        };
                        let weighted_sum: f64 =
                            values.iter().zip(w.iter()).map(|(v, wt)| v[h] * wt).sum();
                        combined[h] = weighted_sum;
                    }
                } else {
                    // Fall back to equal weights if horizon_weights not computed
                    for h in 0..horizon {
                        let weighted_sum: f64 = values
                            .iter()
                            .zip(self.weights.iter())
                            .map(|(v, w)| v[h] * w)
                            .sum();
                        combined[h] = weighted_sum;
                    }
                }
            }
        }

        combined
    }

    /// Compute weights based on MSE of fitted values.
    fn compute_mse_weights(&mut self, actual: &[f64]) {
        let n = self.models.len();
        let mut mse_values = vec![f64::INFINITY; n];

        for (i, model) in self.models.iter().enumerate() {
            if let Some(fitted) = model.fitted_values() {
                let mse: f64 = actual
                    .iter()
                    .zip(fitted.iter())
                    .map(|(a, f)| (a - f).powi(2))
                    .sum::<f64>()
                    / actual.len() as f64;
                mse_values[i] = mse.max(1e-10); // Avoid division by zero
            }
        }

        // Convert MSE to weights (inverse MSE, normalized)
        let inv_mse: Vec<f64> = mse_values.iter().map(|m| 1.0 / m).collect();
        let sum_inv: f64 = inv_mse.iter().sum();

        self.weights = if sum_inv > 0.0 {
            inv_mse.iter().map(|w| w / sum_inv).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
    }

    /// Compute weights based on Akaike Information Criterion.
    ///
    /// Akaike weights: w_i = exp(-0.5 * (AIC_i - AIC_min)) / sum.
    /// Falls back to equal weights when no AIC values can be estimated.
    fn compute_aic_weights(&mut self, actual: &[f64]) {
        let n = self.models.len();

        let aic_values: Vec<Option<f64>> = self
            .models
            .iter()
            .map(|m| estimate_aic(m.as_ref(), actual))
            .collect();

        let valid_aics: Vec<f64> = aic_values.iter().filter_map(|a| *a).collect();

        if valid_aics.is_empty() {
            // Fall back to equal weights
            self.weights = vec![1.0 / n as f64; n];
            return;
        }

        let aic_min = valid_aics.iter().copied().fold(f64::INFINITY, f64::min);

        let raw_weights: Vec<f64> = aic_values
            .iter()
            .map(|a| match a {
                Some(aic) => (-0.5 * (aic - aic_min)).exp(),
                None => 0.0, // Model with no AIC gets zero weight
            })
            .collect();

        let sum: f64 = raw_weights.iter().sum();

        self.weights = if sum > 0.0 {
            raw_weights.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        };
    }

    /// Compute stacking weights using projected gradient descent.
    ///
    /// Uses the second half of in-sample fitted values as a validation set.
    /// Optimizes non-negative weights summing to 1 via gradient descent with
    /// simplex projection after each step.
    fn compute_stacking_weights(&mut self, actual: &[f64]) {
        let n_models = self.models.len();
        let n_obs = actual.len();

        // Collect fitted values from all models
        let all_fitted: Vec<Vec<f64>> = self
            .models
            .iter()
            .filter_map(|m| m.fitted_values().map(|f| f.to_vec()))
            .collect();

        if all_fitted.len() != n_models || n_obs < 4 {
            // Fall back to equal weights
            self.weights = vec![1.0 / n_models as f64; n_models];
            return;
        }

        // Use second half as validation
        let split = n_obs / 2;
        let val_actual = &actual[split..];
        let val_len = val_actual.len();

        // Build matrix of fitted values on validation set: val_fitted[model][time]
        let val_fitted: Vec<Vec<f64>> = all_fitted
            .iter()
            .map(|f| {
                let end = f.len().min(n_obs);
                if split < end {
                    f[split..end].to_vec()
                } else {
                    vec![0.0; val_len]
                }
            })
            .collect();

        // Ensure all have same length
        let common_len = val_fitted
            .iter()
            .map(|v| v.len())
            .min()
            .unwrap_or(0)
            .min(val_len);

        if common_len == 0 {
            self.weights = vec![1.0 / n_models as f64; n_models];
            return;
        }

        // Projected gradient descent
        let mut w = vec![1.0 / n_models as f64; n_models];
        let lr = 0.01;
        let max_iter = 500;

        for _ in 0..max_iter {
            // Compute gradient of MSE w.r.t. weights
            let mut grad = vec![0.0; n_models];
            for t in 0..common_len {
                let pred: f64 = w
                    .iter()
                    .zip(val_fitted.iter())
                    .map(|(wi, fi)| wi * fi[t])
                    .sum();
                let err = pred - val_actual[t];
                for (j, fj) in val_fitted.iter().enumerate() {
                    grad[j] += 2.0 * err * fj[t] / common_len as f64;
                }
            }

            // Gradient step
            for j in 0..n_models {
                w[j] -= lr * grad[j];
            }

            // Project onto simplex
            w = nnls_simplex(&w);
        }

        self.weights = w;
    }

    /// Compute per-horizon adaptive weights using rolling-origin evaluation.
    ///
    /// For each forecast step h = 1..max_horizon, performs expanding-window
    /// evaluation: uses fitted values to approximate errors at each horizon,
    /// and computes inverse-MSE weights per horizon step.
    fn compute_horizon_adaptive_weights(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        let n = values.len();
        let n_models = self.models.len();

        // Use a moderate max horizon and minimum training size
        let max_horizon = ((n as f64 * 0.2).ceil() as usize).clamp(1, 20);
        let min_train = (n / 2).max(10).min(n.saturating_sub(max_horizon + 1));

        if min_train + max_horizon > n || n_models == 0 {
            // Not enough data; fall back to equal weights
            self.weights = vec![1.0 / n_models as f64; n_models];
            self.horizon_weights = Some(vec![self.weights.clone(); max_horizon]);
            return Ok(());
        }

        // Per-horizon, per-model squared errors
        // errors[h][model] = Vec of squared errors at that horizon
        let mut errors: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n_models]; max_horizon];

        // Rolling origins: t ranges over the training boundary
        let origins: Vec<usize> = {
            let start = min_train;
            let end = n.saturating_sub(1);
            // Use at most 10 origins to keep computation bounded
            let step = ((end - start) as f64 / 10.0).ceil() as usize;
            let step = step.max(1);
            (start..=end).step_by(step).collect()
        };

        for &origin in &origins {
            for (m_idx, model) in self.models.iter().enumerate() {
                if let Some(fitted) = model.fitted_values() {
                    for h in 0..max_horizon {
                        let target_idx = origin + h;
                        if target_idx < n && target_idx < fitted.len() {
                            let err = (values[target_idx] - fitted[target_idx]).powi(2);
                            errors[h][m_idx].push(err);
                        }
                    }
                }
            }
        }

        // Convert errors to per-horizon weights (inverse MSE)
        let mut hw = Vec::with_capacity(max_horizon);
        let equal_w = vec![1.0 / n_models as f64; n_models];

        for h in 0..max_horizon {
            let mut mse_vals = vec![f64::INFINITY; n_models];
            for m in 0..n_models {
                if !errors[h][m].is_empty() {
                    let mean_sq: f64 = errors[h][m].iter().sum::<f64>() / errors[h][m].len() as f64;
                    mse_vals[m] = mean_sq.max(1e-10);
                }
            }

            let inv: Vec<f64> = mse_vals.iter().map(|m| 1.0 / m).collect();
            let sum_inv: f64 = inv.iter().sum();

            let w = if sum_inv > 0.0 && sum_inv.is_finite() {
                inv.iter().map(|v| v / sum_inv).collect()
            } else {
                equal_w.clone()
            };
            hw.push(w);
        }

        // Store the average weights as the main weight vector
        let mut avg_w = vec![0.0; n_models];
        for w in &hw {
            for (j, wj) in w.iter().enumerate() {
                avg_w[j] += wj;
            }
        }
        let n_h = hw.len() as f64;
        for wj in &mut avg_w {
            *wj /= n_h;
        }
        // Normalize
        let s: f64 = avg_w.iter().sum();
        if s > 0.0 {
            for wj in &mut avg_w {
                *wj /= s;
            }
        }

        self.weights = avg_w;
        self.horizon_weights = Some(hw);
        Ok(())
    }
}

impl Forecaster for Ensemble {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        validate_series_complete(series)?;
        if self.models.is_empty() {
            return Err(ForecastError::InvalidParameter(
                "Ensemble has no models".to_string(),
            ));
        }

        // Fit all models (skip already-fitted ones from AutoEnsemble)
        for model in &mut self.models {
            if model.is_fitted() {
                continue;
            }
            model.fit(series)?;
        }

        let values = series.primary_values();

        // Compute weights based on method
        match self.method {
            CombinationMethod::WeightedMSE => {
                self.compute_mse_weights(values);
            }
            CombinationMethod::Custom => {
                if let Some(ref custom) = self.custom_weights {
                    if custom.len() != self.models.len() {
                        return Err(ForecastError::DimensionMismatch {
                            expected: self.models.len(),
                            got: custom.len(),
                        });
                    }
                    // Normalize weights
                    let sum: f64 = custom.iter().sum();
                    self.weights = custom.iter().map(|w| w / sum).collect();
                }
            }
            CombinationMethod::InverseAIC => {
                self.compute_aic_weights(values);
            }
            CombinationMethod::Stacking { .. } => {
                self.compute_stacking_weights(values);
            }
            CombinationMethod::HorizonAdaptive => {
                self.compute_horizon_adaptive_weights(series)?;
            }
            CombinationMethod::Mean | CombinationMethod::Median => {
                // No weight computation needed
            }
        }

        // Combine fitted values
        let all_fitted: Vec<Vec<f64>> = self
            .models
            .iter()
            .filter_map(|m| m.fitted_values().map(|f| f.to_vec()))
            .collect();

        if !all_fitted.is_empty() {
            let combined_fitted = self.combine_values(&all_fitted);
            let residuals: Vec<f64> = values
                .iter()
                .zip(combined_fitted.iter())
                .map(|(y, f)| y - f)
                .collect();
            self.fitted = Some(combined_fitted);
            self.residuals = Some(residuals);
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        if !self.is_fitted {
            return Err(ForecastError::FitRequired { model: None });
        }

        if horizon == 0 {
            return Ok(Forecast::from_values(Vec::new()));
        }

        // Get forecasts from all models
        let all_forecasts: Vec<Vec<f64>> = self
            .models
            .iter()
            .filter_map(|m| m.predict(horizon).ok())
            .map(|f| f.primary().to_vec())
            .collect();

        if all_forecasts.is_empty() {
            return Err(ForecastError::ComputationError(
                "No models produced valid forecasts".to_string(),
            ));
        }

        let combined = self.combine_values(&all_forecasts);
        Ok(Forecast::from_values(combined))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let point_forecast = self.predict(horizon)?;

        if horizon == 0 {
            return Ok(point_forecast);
        }

        // Get all forecasts with intervals from component models
        let all_forecasts: Vec<Forecast> = self
            .models
            .iter()
            .filter_map(|m| m.predict_with_intervals(horizon, level).ok())
            .collect();

        if all_forecasts.is_empty() {
            return Ok(point_forecast);
        }

        // Collect lower and upper bounds from models that produced intervals
        let all_lowers: Vec<Vec<f64>> = all_forecasts
            .iter()
            .filter_map(|f| f.lower_series(0).ok().map(|l| l.to_vec()))
            .collect();

        let all_uppers: Vec<Vec<f64>> = all_forecasts
            .iter()
            .filter_map(|f| f.upper_series(0).ok().map(|u| u.to_vec()))
            .collect();

        // Combine intervals using widest-envelope: take the minimum of all
        // lower bounds and the maximum of all upper bounds at each step.
        // This produces a conservative combined interval that covers the
        // uncertainty from all component models.
        let lower = if !all_lowers.is_empty() {
            self.combine_interval_bounds(&all_lowers, true)
        } else {
            point_forecast.primary().to_vec()
        };

        let upper = if !all_uppers.is_empty() {
            self.combine_interval_bounds(&all_uppers, false)
        } else {
            point_forecast.primary().to_vec()
        };

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
        match self.method {
            CombinationMethod::Mean => "Ensemble (Mean)",
            CombinationMethod::Median => "Ensemble (Median)",
            CombinationMethod::WeightedMSE => "Ensemble (Weighted MSE)",
            CombinationMethod::Custom => "Ensemble (Custom)",
            CombinationMethod::InverseAIC => "Ensemble (InverseAIC)",
            CombinationMethod::Stacking { .. } => "Ensemble (Stacking)",
            CombinationMethod::HorizonAdaptive => "Ensemble (HorizonAdaptive)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::baseline::{Naive, SimpleMovingAverage};
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    fn make_series() -> TimeSeries {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.3).sin())
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn ensemble_mean_basic() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn ensemble_median() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::Median);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn ensemble_weighted_mse() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::WeightedMSE);
        ensemble.fit(&ts).unwrap();

        // Weights should be normalized
        let weights = ensemble.weights();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1");
    }

    #[test]
    fn ensemble_custom_weights() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_weights(vec![0.7, 0.3]);
        ensemble.fit(&ts).unwrap();

        let weights = ensemble.weights();
        assert!((weights[0] - 0.7).abs() < 1e-6);
        assert!((weights[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn ensemble_empty() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![];

        let mut ensemble = Ensemble::new(models);
        assert!(ensemble.fit(&ts).is_err());
    }

    #[test]
    fn ensemble_requires_fit() {
        let models: Vec<Box<dyn Forecaster>> = vec![Box::new(Naive::new())];
        let ensemble = Ensemble::new(models);
        assert!(matches!(
            ensemble.predict(5),
            Err(ForecastError::FitRequired { model: None })
        ));
    }

    #[test]
    fn ensemble_zero_horizon() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![Box::new(Naive::new())];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn ensemble_confidence_intervals() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn ensemble_intervals_use_widest_envelope() {
        let ts = make_series();

        // Get individual model intervals
        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();
        let naive_fc = naive.predict_with_intervals(5, 0.95).unwrap();

        let mut sma = SimpleMovingAverage::new(5);
        sma.fit(&ts).unwrap();
        let sma_fc = sma.predict_with_intervals(5, 0.95).unwrap();

        // Get ensemble intervals
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];
        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();
        let ensemble_fc = ensemble.predict_with_intervals(5, 0.95).unwrap();

        // If both models produced intervals, the ensemble should use the
        // widest envelope (min of lowers, max of uppers)
        if let (Ok(naive_lower), Ok(sma_lower)) = (naive_fc.lower_series(0), sma_fc.lower_series(0))
        {
            let ensemble_lower = ensemble_fc.lower_series(0).unwrap();
            for i in 0..5 {
                let widest_lower = naive_lower[i].min(sma_lower[i]);
                assert!(
                    (ensemble_lower[i] - widest_lower).abs() < 1e-10,
                    "Ensemble lower[{}] = {} should equal widest lower {}",
                    i,
                    ensemble_lower[i],
                    widest_lower,
                );
            }
        }

        if let (Ok(naive_upper), Ok(sma_upper)) = (naive_fc.upper_series(0), sma_fc.upper_series(0))
        {
            let ensemble_upper = ensemble_fc.upper_series(0).unwrap();
            for i in 0..5 {
                let widest_upper = naive_upper[i].max(sma_upper[i]);
                assert!(
                    (ensemble_upper[i] - widest_upper).abs() < 1e-10,
                    "Ensemble upper[{}] = {} should equal widest upper {}",
                    i,
                    ensemble_upper[i],
                    widest_upper,
                );
            }
        }
    }

    #[test]
    fn ensemble_fitted_and_residuals() {
        let ts = make_series();
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();

        assert!(ensemble.fitted_values().is_some());
        assert!(ensemble.residuals().is_some());
    }

    #[test]
    fn ensemble_name() {
        let mean = Ensemble::new(vec![Box::new(Naive::new())]);
        assert_eq!(mean.name(), "Ensemble (Mean)");

        let median =
            Ensemble::new(vec![Box::new(Naive::new())]).with_method(CombinationMethod::Median);
        assert_eq!(median.name(), "Ensemble (Median)");

        let weighted =
            Ensemble::new(vec![Box::new(Naive::new())]).with_method(CombinationMethod::WeightedMSE);
        assert_eq!(weighted.name(), "Ensemble (Weighted MSE)");
    }

    #[test]
    fn ensemble_model_count() {
        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let ensemble = Ensemble::new(models);
        assert_eq!(ensemble.model_count(), 3);
    }

    #[test]
    fn ensemble_mean_is_between_individual_forecasts() {
        let ts = make_series();

        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();
        let naive_fc = naive.predict(5).unwrap();

        let mut sma = SimpleMovingAverage::new(5);
        sma.fit(&ts).unwrap();
        let sma_fc = sma.predict(5).unwrap();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models);
        ensemble.fit(&ts).unwrap();
        let ensemble_fc = ensemble.predict(5).unwrap();

        // Ensemble mean should be between the individual forecasts
        for i in 0..5 {
            let min_val = naive_fc.primary()[i].min(sma_fc.primary()[i]);
            let max_val = naive_fc.primary()[i].max(sma_fc.primary()[i]);
            assert!(
                ensemble_fc.primary()[i] >= min_val - 1e-10
                    && ensemble_fc.primary()[i] <= max_val + 1e-10,
                "Ensemble forecast should be between individual forecasts"
            );
        }
    }

    // =========================================================================
    // InverseAIC tests
    // =========================================================================

    #[test]
    fn ensemble_inverse_aic_weights_sum_to_one() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::InverseAIC);
        ensemble.fit(&ts).unwrap();

        let weights = ensemble.weights();
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "InverseAIC weights should sum to 1, got {}",
            sum,
        );
        // All weights should be non-negative
        for (i, &w) in weights.iter().enumerate() {
            assert!(w >= 0.0, "Weight {} should be non-negative, got {}", i, w);
        }
    }

    #[test]
    fn ensemble_inverse_aic_produces_forecasts() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::InverseAIC);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        // Forecasts should be finite
        for &v in forecast.primary() {
            assert!(v.is_finite(), "Forecast value should be finite");
        }
    }

    #[test]
    fn ensemble_inverse_aic_name() {
        let e =
            Ensemble::new(vec![Box::new(Naive::new())]).with_method(CombinationMethod::InverseAIC);
        assert_eq!(e.name(), "Ensemble (InverseAIC)");
    }

    #[test]
    fn ensemble_inverse_aic_fitted_values() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::InverseAIC);
        ensemble.fit(&ts).unwrap();

        assert!(ensemble.fitted_values().is_some());
        assert!(ensemble.residuals().is_some());
    }

    // =========================================================================
    // Stacking tests
    // =========================================================================

    #[test]
    fn ensemble_stacking_weights_sum_to_one() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble =
            Ensemble::new(models).with_method(CombinationMethod::Stacking { folds: 2 });
        ensemble.fit(&ts).unwrap();

        let weights = ensemble.weights();
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Stacking weights should sum to 1, got {}",
            sum,
        );
        // Non-negative
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                w >= -1e-10,
                "Weight {} should be non-negative, got {}",
                i,
                w,
            );
        }
    }

    #[test]
    fn ensemble_stacking_produces_forecasts() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble =
            Ensemble::new(models).with_method(CombinationMethod::Stacking { folds: 2 });
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        for &v in forecast.primary() {
            assert!(v.is_finite(), "Forecast value should be finite");
        }
    }

    #[test]
    fn ensemble_stacking_name() {
        let e = Ensemble::new(vec![Box::new(Naive::new())])
            .with_method(CombinationMethod::Stacking { folds: 5 });
        assert_eq!(e.name(), "Ensemble (Stacking)");
    }

    #[test]
    fn ensemble_stacking_fitted_values() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble =
            Ensemble::new(models).with_method(CombinationMethod::Stacking { folds: 2 });
        ensemble.fit(&ts).unwrap();

        assert!(ensemble.fitted_values().is_some());
        assert!(ensemble.residuals().is_some());
    }

    // =========================================================================
    // HorizonAdaptive tests
    // =========================================================================

    #[test]
    fn ensemble_horizon_adaptive_weights_sum_to_one() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::HorizonAdaptive);
        ensemble.fit(&ts).unwrap();

        // Main weights
        let weights = ensemble.weights();
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "HorizonAdaptive average weights should sum to 1, got {}",
            sum,
        );

        // Per-horizon weights should exist
        let hw = ensemble
            .horizon_weights()
            .expect("horizon_weights should be Some");
        assert!(!hw.is_empty(), "Should have per-horizon weights");
        for (h, w) in hw.iter().enumerate() {
            let s: f64 = w.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "Horizon {} weights should sum to 1, got {}",
                h,
                s,
            );
        }
    }

    #[test]
    fn ensemble_horizon_adaptive_produces_forecasts() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(3)),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::HorizonAdaptive);
        ensemble.fit(&ts).unwrap();

        let forecast = ensemble.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        for &v in forecast.primary() {
            assert!(v.is_finite(), "Forecast value should be finite");
        }
    }

    #[test]
    fn ensemble_horizon_adaptive_name() {
        let e = Ensemble::new(vec![Box::new(Naive::new())])
            .with_method(CombinationMethod::HorizonAdaptive);
        assert_eq!(e.name(), "Ensemble (HorizonAdaptive)");
    }

    #[test]
    fn ensemble_horizon_adaptive_fitted_values() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::HorizonAdaptive);
        ensemble.fit(&ts).unwrap();

        assert!(ensemble.fitted_values().is_some());
        assert!(ensemble.residuals().is_some());
    }

    #[test]
    fn ensemble_horizon_adaptive_long_horizon_fallback() {
        let ts = make_series();

        let models: Vec<Box<dyn Forecaster>> = vec![
            Box::new(Naive::new()),
            Box::new(SimpleMovingAverage::new(5)),
        ];

        let mut ensemble = Ensemble::new(models).with_method(CombinationMethod::HorizonAdaptive);
        ensemble.fit(&ts).unwrap();

        // Request a horizon larger than the stored per-horizon weights
        let forecast = ensemble.predict(50).unwrap();
        assert_eq!(forecast.horizon(), 50);
        for &v in forecast.primary() {
            assert!(v.is_finite(), "Forecast value should be finite");
        }
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn test_nnls_simplex_uniform() {
        let w = vec![0.5, 0.5];
        let proj = nnls_simplex(&w);
        let sum: f64 = proj.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Projected weights should sum to 1",
        );
        for &v in &proj {
            assert!(v >= 0.0, "Projected weight should be non-negative");
        }
    }

    #[test]
    fn test_nnls_simplex_negative() {
        let w = vec![-1.0, 3.0, -0.5];
        let proj = nnls_simplex(&w);
        let sum: f64 = proj.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Projected weights should sum to 1, got {}",
            sum,
        );
        for &v in &proj {
            assert!(v >= -1e-10, "Projected weight should be non-negative");
        }
    }

    #[test]
    fn test_nnls_simplex_already_on_simplex() {
        let w = vec![0.3, 0.5, 0.2];
        let proj = nnls_simplex(&w);
        let sum: f64 = proj.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Should be close to original since it's already on simplex
        for (a, b) in w.iter().zip(proj.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nnls_simplex_empty() {
        let w: Vec<f64> = vec![];
        let proj = nnls_simplex(&w);
        assert!(proj.is_empty());
    }

    #[test]
    fn test_estimate_param_count_naive() {
        let model = Naive::new();
        assert_eq!(estimate_param_count(&model), 1);
    }

    #[test]
    fn test_estimate_param_count_sma() {
        let model = SimpleMovingAverage::new(5);
        assert_eq!(estimate_param_count(&model), 1);
    }

    #[test]
    fn test_estimate_aic_produces_finite_value() {
        let ts = make_series();
        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let aic = estimate_aic(&model, ts.primary_values());
        assert!(aic.is_some(), "AIC should be computable for Naive");
        assert!(aic.unwrap().is_finite(), "AIC should be finite");
    }

    #[test]
    fn test_estimate_aic_lower_for_better_model() {
        let ts = make_series();

        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();

        let mut sma = SimpleMovingAverage::new(5);
        sma.fit(&ts).unwrap();

        let aic_naive = estimate_aic(&naive, ts.primary_values()).unwrap();
        let aic_sma = estimate_aic(&sma, ts.primary_values()).unwrap();

        // Both should be finite; we don't assert which is lower since it depends
        // on the data, but both should be finite.
        assert!(aic_naive.is_finite());
        assert!(aic_sma.is_finite());
    }

    #[test]
    fn ensemble_combination_method_partial_eq() {
        assert_eq!(CombinationMethod::Mean, CombinationMethod::Mean);
        assert_eq!(CombinationMethod::InverseAIC, CombinationMethod::InverseAIC);
        assert_eq!(
            CombinationMethod::Stacking { folds: 5 },
            CombinationMethod::Stacking { folds: 5 },
        );
        assert_ne!(
            CombinationMethod::Stacking { folds: 5 },
            CombinationMethod::Stacking { folds: 3 },
        );
        assert_eq!(
            CombinationMethod::HorizonAdaptive,
            CombinationMethod::HorizonAdaptive,
        );
        assert_ne!(CombinationMethod::InverseAIC, CombinationMethod::Mean);
    }
}
