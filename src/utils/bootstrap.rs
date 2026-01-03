//! Bootstrap methods for forecast uncertainty estimation.
//!
//! Provides residual bootstrap and block bootstrap methods for generating
//! empirical confidence intervals when analytical formulas are unavailable
//! or unreliable.

use crate::core::{Forecast, TimeSeries};
use crate::error::Result;
use crate::models::Forecaster;
use rand::prelude::*;
use rand::SeedableRng;

/// Configuration for bootstrap interval estimation.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap samples to generate.
    pub n_samples: usize,
    /// Block size for block bootstrap (None for residual bootstrap).
    pub block_size: Option<usize>,
    /// Random seed for reproducibility (None for random).
    pub seed: Option<u64>,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            block_size: None,
            seed: None,
        }
    }
}

impl BootstrapConfig {
    /// Create a new bootstrap config with specified number of samples.
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            ..Default::default()
        }
    }

    /// Use block bootstrap with specified block size.
    /// Preserves autocorrelation structure better than residual bootstrap.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of bootstrap interval estimation.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Lower bounds of confidence intervals (one per horizon step).
    pub lower: Vec<f64>,
    /// Upper bounds of confidence intervals (one per horizon step).
    pub upper: Vec<f64>,
    /// Confidence level used.
    pub level: f64,
    /// Number of bootstrap samples used.
    pub n_samples: usize,
}

/// Resample residuals with replacement (residual bootstrap).
fn resample_residuals(residuals: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    let n = residuals.len();
    (0..n).map(|_| residuals[rng.gen_range(0..n)]).collect()
}

/// Resample using block bootstrap (preserves autocorrelation).
fn resample_blocks(residuals: &[f64], block_size: usize, rng: &mut impl Rng) -> Vec<f64> {
    let n = residuals.len();
    if block_size == 0 || block_size > n {
        return resample_residuals(residuals, rng);
    }

    let mut result = Vec::with_capacity(n);
    let n_blocks = n / block_size + 1;

    for _ in 0..n_blocks {
        let start = rng.gen_range(0..=(n - block_size));
        for j in 0..block_size {
            if result.len() >= n {
                break;
            }
            result.push(residuals[start + j]);
        }
    }

    result.truncate(n);
    result
}

/// Generate bootstrap forecast intervals.
///
/// Uses residual bootstrap: resamples fitted residuals and generates
/// new synthetic series, fits the model, and collects forecast distributions.
///
/// # Arguments
/// * `model` - A fitted forecaster with residuals
/// * `series` - The original time series
/// * `horizon` - Forecast horizon
/// * `level` - Confidence level (e.g., 0.95 for 95% intervals)
/// * `config` - Bootstrap configuration
///
/// # Returns
/// Bootstrap result with lower and upper bounds for each horizon step.
///
/// # Example
/// ```ignore
/// use anofox_forecast::utils::bootstrap::{bootstrap_intervals, BootstrapConfig};
/// use anofox_forecast::models::baseline::Naive;
///
/// let mut model = Naive::new();
/// model.fit(&series).unwrap();
///
/// let config = BootstrapConfig::new(500).with_seed(42);
/// let result = bootstrap_intervals(&model, &series, 10, 0.95, &config).unwrap();
/// ```
pub fn bootstrap_intervals<M: Forecaster + Clone>(
    model: &M,
    series: &TimeSeries,
    horizon: usize,
    level: f64,
    config: &BootstrapConfig,
) -> Result<BootstrapResult> {
    let residuals = model
        .residuals()
        .ok_or(crate::error::ForecastError::FitRequired)?;

    let fitted = model
        .fitted_values()
        .ok_or(crate::error::ForecastError::FitRequired)?;

    // Filter out NaN residuals
    let valid_residuals: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();

    if valid_residuals.is_empty() {
        return Err(crate::error::ForecastError::ComputationError(
            "No valid residuals for bootstrap".to_string(),
        ));
    }

    // Initialize RNG
    let mut rng: StdRng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    // Collect forecast samples for each horizon step
    let mut forecast_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(config.n_samples); horizon];

    for _ in 0..config.n_samples {
        // Resample residuals
        let resampled = match config.block_size {
            Some(bs) => resample_blocks(&valid_residuals, bs, &mut rng),
            None => resample_residuals(&valid_residuals, &mut rng),
        };

        // Create synthetic series: fitted + resampled residuals
        let synthetic_values: Vec<f64> = fitted
            .iter()
            .zip(resampled.iter().cycle())
            .map(|(f, r)| f + r)
            .collect();

        // Create synthetic TimeSeries with same timestamps
        let synthetic_ts =
            TimeSeries::univariate(series.timestamps().to_vec(), synthetic_values.clone());

        if let Ok(ts) = synthetic_ts {
            // Fit model to synthetic series
            let mut bootstrap_model = model.clone();
            if bootstrap_model.fit(&ts).is_ok() {
                // Generate forecast
                if let Ok(forecast) = bootstrap_model.predict(horizon) {
                    for (h, &val) in forecast.primary().iter().enumerate() {
                        if val.is_finite() {
                            forecast_samples[h].push(val);
                        }
                    }
                }
            }
        }
    }

    // Calculate quantiles for each horizon
    let alpha = (1.0 - level) / 2.0;
    let mut lower = Vec::with_capacity(horizon);
    let mut upper = Vec::with_capacity(horizon);

    for samples in &mut forecast_samples {
        if samples.is_empty() {
            // Fallback to original forecast if no samples
            lower.push(f64::NAN);
            upper.push(f64::NAN);
            continue;
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();

        let lower_idx = ((alpha * n as f64).floor() as usize).min(n - 1);
        let upper_idx = (((1.0 - alpha) * n as f64).floor() as usize).min(n - 1);

        lower.push(samples[lower_idx]);
        upper.push(samples[upper_idx]);
    }

    Ok(BootstrapResult {
        lower,
        upper,
        level,
        n_samples: config.n_samples,
    })
}

/// Compute bootstrap forecast with intervals, returning a Forecast object.
///
/// Combines the point forecast from the original model with bootstrap intervals.
pub fn bootstrap_forecast<M: Forecaster + Clone>(
    model: &M,
    series: &TimeSeries,
    horizon: usize,
    level: f64,
    config: &BootstrapConfig,
) -> Result<Forecast> {
    let point_forecast = model.predict(horizon)?;
    let bootstrap_result = bootstrap_intervals(model, series, horizon, level, config)?;

    // Combine point forecast with bootstrap intervals
    Ok(Forecast::from_values_with_intervals(
        point_forecast.primary().to_vec(),
        bootstrap_result.lower,
        bootstrap_result.upper,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::baseline::Naive;
    use crate::models::exponential::SimpleExponentialSmoothing;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    #[test]
    fn bootstrap_config_default() {
        let config = BootstrapConfig::default();
        assert_eq!(config.n_samples, 1000);
        assert!(config.block_size.is_none());
        assert!(config.seed.is_none());
    }

    #[test]
    fn bootstrap_config_builder() {
        let config = BootstrapConfig::new(500).with_block_size(10).with_seed(42);

        assert_eq!(config.n_samples, 500);
        assert_eq!(config.block_size, Some(10));
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn resample_residuals_length() {
        let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = StdRng::seed_from_u64(42);
        let resampled = resample_residuals(&residuals, &mut rng);
        assert_eq!(resampled.len(), residuals.len());
    }

    #[test]
    fn resample_blocks_length() {
        let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut rng = StdRng::seed_from_u64(42);
        let resampled = resample_blocks(&residuals, 3, &mut rng);
        assert_eq!(resampled.len(), residuals.len());
    }

    #[test]
    fn bootstrap_intervals_naive() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let config = BootstrapConfig::new(100).with_seed(42);
        let result = bootstrap_intervals(&model, &ts, 5, 0.95, &config).unwrap();

        assert_eq!(result.lower.len(), 5);
        assert_eq!(result.upper.len(), 5);
        assert_eq!(result.level, 0.95);

        // Lower should be less than upper
        for i in 0..5 {
            assert!(
                result.lower[i] <= result.upper[i],
                "Lower {} > Upper {} at horizon {}",
                result.lower[i],
                result.upper[i],
                i
            );
        }
    }

    #[test]
    fn bootstrap_intervals_ses() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.3).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();

        let config = BootstrapConfig::new(100).with_seed(123);
        let result = bootstrap_intervals(&model, &ts, 5, 0.90, &config).unwrap();

        assert_eq!(result.lower.len(), 5);
        assert_eq!(result.upper.len(), 5);
        assert_eq!(result.level, 0.90);
    }

    #[test]
    fn bootstrap_forecast_contains_intervals() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let config = BootstrapConfig::new(50).with_seed(42);
        let forecast = bootstrap_forecast(&model, &ts, 5, 0.95, &config).unwrap();

        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn bootstrap_reproducible_with_seed() {
        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let config = BootstrapConfig::new(50).with_seed(42);

        let result1 = bootstrap_intervals(&model, &ts, 5, 0.95, &config).unwrap();
        let result2 = bootstrap_intervals(&model, &ts, 5, 0.95, &config).unwrap();

        for i in 0..5 {
            assert!(
                (result1.lower[i] - result2.lower[i]).abs() < 1e-10,
                "Results should be reproducible with seed"
            );
        }
    }

    #[test]
    fn bootstrap_block_vs_residual() {
        let timestamps = make_timestamps(100);
        let values: Vec<f64> = (0..100)
            .map(|i| 10.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let residual_config = BootstrapConfig::new(50).with_seed(42);
        let block_config = BootstrapConfig::new(50).with_block_size(5).with_seed(42);

        let residual_result = bootstrap_intervals(&model, &ts, 5, 0.95, &residual_config).unwrap();
        let block_result = bootstrap_intervals(&model, &ts, 5, 0.95, &block_config).unwrap();

        // Both should produce valid intervals
        assert_eq!(residual_result.lower.len(), 5);
        assert_eq!(block_result.lower.len(), 5);

        // Results may differ due to different resampling strategies
        // Just verify both are valid
        for i in 0..5 {
            assert!(residual_result.lower[i] <= residual_result.upper[i]);
            assert!(block_result.lower[i] <= block_result.upper[i]);
        }
    }
}
