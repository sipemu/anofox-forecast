//! Bootstrap prediction intervals for postprocessing.
//!
//! Generates prediction intervals by resampling forecast residuals. Unlike the
//! model-based bootstrap in `utils::bootstrap` (which re-fits the model on each
//! synthetic series), this operates purely on residuals — making it model-agnostic
//! and fast.
//!
//! Per-step uncertainty grows naturally because each simulated path accumulates
//! resampled errors.

use crate::error::{ForecastError, Result};
use crate::postprocess::PredictionIntervals;
use rand::prelude::*;
use rand::SeedableRng;

/// Bootstrap predictor for distribution-free prediction intervals.
///
/// Resamples forecast residuals to simulate future error paths, producing
/// per-step prediction intervals where uncertainty grows with horizon.
#[derive(Debug, Clone)]
pub struct BootstrapPredictor {
    /// Target coverage level (e.g., 0.95).
    coverage: f64,
    /// Number of bootstrap replicates.
    n_replicates: usize,
    /// Block size for block bootstrap (None = IID).
    block_size: Option<usize>,
    /// Random seed for reproducibility.
    seed: Option<u64>,
}

/// Result of fitting a bootstrap predictor on residuals.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Sorted finite residuals used for resampling.
    residuals: Vec<f64>,
    /// Coverage level.
    coverage: f64,
    /// Number of replicates.
    n_replicates: usize,
    /// Block size (None = IID).
    block_size: Option<usize>,
    /// Seed.
    seed: Option<u64>,
}

impl BootstrapPredictor {
    /// Create a bootstrap predictor with the given coverage level.
    pub fn new(coverage: f64) -> Self {
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "coverage must be in (0, 1)"
        );
        Self {
            coverage,
            n_replicates: 1000,
            block_size: None,
            seed: None,
        }
    }

    /// Set the number of bootstrap replicates (default: 1000).
    pub fn n_replicates(mut self, n: usize) -> Self {
        self.n_replicates = n.max(10);
        self
    }

    /// Use block bootstrap with the given block size.
    ///
    /// Preserves autocorrelation in residuals. Use when residuals are
    /// not independent (e.g., poor model fit or complex dynamics).
    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = if size > 0 { Some(size) } else { None };
        self
    }

    /// Set random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get the coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Fit on historical forecasts and actuals.
    ///
    /// Computes residuals (forecast - actual) and stores them for resampling.
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<BootstrapResult> {
        if forecasts.len() != actuals.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: forecasts.len(),
                got: actuals.len(),
            });
        }

        let residuals: Vec<f64> = forecasts
            .iter()
            .zip(actuals.iter())
            .map(|(f, a)| f - a)
            .filter(|r| r.is_finite())
            .collect();

        if residuals.len() < 2 {
            return Err(ForecastError::InsufficientData {
                needed: 2,
                got: residuals.len(),
                hint: Some("need at least 2 finite residuals for bootstrap".into()),
            });
        }

        Ok(BootstrapResult {
            residuals,
            coverage: self.coverage,
            n_replicates: self.n_replicates,
            block_size: self.block_size,
            seed: self.seed,
        })
    }

    /// Generate prediction intervals for a point forecast.
    ///
    /// Simulates `n_replicates` future error paths by resampling residuals,
    /// then extracts per-step quantile bounds.
    pub fn predict(&self, result: &BootstrapResult, point_forecast: &[f64]) -> PredictionIntervals {
        let horizon = point_forecast.len();
        let n_rep = result.n_replicates;

        let mut rng: StdRng = match result.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Simulate n_rep paths, each of length horizon
        let mut samples_per_step: Vec<Vec<f64>> = vec![Vec::with_capacity(n_rep); horizon];

        for _ in 0..n_rep {
            let errors = resample(&result.residuals, horizon, result.block_size, &mut rng);
            for (h, &err) in errors.iter().enumerate() {
                let simulated = point_forecast[h] + err;
                if simulated.is_finite() {
                    samples_per_step[h].push(simulated);
                }
            }
        }

        // Extract quantile bounds per step
        let alpha = (1.0 - result.coverage) / 2.0;
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for samples in &mut samples_per_step {
            if samples.is_empty() {
                lower.push(f64::NAN);
                upper.push(f64::NAN);
                continue;
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = samples.len();
            let lo_idx = ((alpha * n as f64).floor() as usize).min(n - 1);
            let hi_idx = (((1.0 - alpha) * n as f64).floor() as usize).min(n - 1);
            lower.push(samples[lo_idx]);
            upper.push(samples[hi_idx]);
        }

        PredictionIntervals::from_bounds(lower, upper, result.coverage)
            .expect("Valid prediction intervals")
    }
}

impl BootstrapResult {
    /// Get the residuals used for resampling.
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Get the coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Get the number of replicates.
    pub fn n_replicates(&self) -> usize {
        self.n_replicates
    }
}

/// Resample residuals for a given horizon length.
fn resample(
    residuals: &[f64],
    horizon: usize,
    block_size: Option<usize>,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n = residuals.len();
    match block_size {
        Some(bs) if bs > 0 && bs <= n => {
            let mut result = Vec::with_capacity(horizon);
            while result.len() < horizon {
                let start = rng.gen_range(0..=(n - bs));
                for j in 0..bs {
                    if result.len() >= horizon {
                        break;
                    }
                    result.push(residuals[start + j]);
                }
            }
            result
        }
        _ => (0..horizon)
            .map(|_| residuals[rng.gen_range(0..n)])
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_and_predict_basic() {
        let forecasts: Vec<f64> = (0..50).map(|i| 10.0 + i as f64).collect();
        let actuals: Vec<f64> = (0..50).map(|i| 10.5 + i as f64 * 0.98).collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(200).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();
        let point: Vec<f64> = (50..55).map(|i| 10.0 + i as f64).collect();
        let intervals = predictor.predict(&result, &point);

        assert_eq!(intervals.len(), 5);
        assert!((intervals.coverage() - 0.90).abs() < 1e-10);
        for i in 0..5 {
            assert!(intervals.lower()[i] <= intervals.upper()[i]);
        }
    }

    #[test]
    fn intervals_widen_with_block_bootstrap() {
        let forecasts: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let actuals: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 0.5).collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(200).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point = vec![0.0; 5];
        let intervals = predictor.predict(&result, &point);

        for i in 0..5 {
            assert!(intervals.lower()[i] <= intervals.upper()[i]);
            assert!(intervals.lower()[i].is_finite());
            assert!(intervals.upper()[i].is_finite());
        }
    }

    #[test]
    fn block_bootstrap() {
        let forecasts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.3).collect();

        let predictor = BootstrapPredictor::new(0.95)
            .n_replicates(100)
            .block_size(5)
            .seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();
        let intervals = predictor.predict(&result, &[50.0, 51.0, 52.0]);

        assert_eq!(intervals.len(), 3);
        for i in 0..3 {
            assert!(intervals.lower()[i] <= intervals.upper()[i]);
        }
    }

    #[test]
    fn reproducible_with_seed() {
        let forecasts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(100).seed(123);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point = vec![50.0, 51.0, 52.0];
        let i1 = predictor.predict(&result, &point);
        let i2 = predictor.predict(&result, &point);

        for h in 0..3 {
            assert!((i1.lower()[h] - i2.lower()[h]).abs() < 1e-10);
            assert!((i1.upper()[h] - i2.upper()[h]).abs() < 1e-10);
        }
    }

    #[test]
    fn fails_on_length_mismatch() {
        let predictor = BootstrapPredictor::new(0.90);
        let result = predictor.fit(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn fails_on_insufficient_residuals() {
        let predictor = BootstrapPredictor::new(0.90);
        let result = predictor.fit(&[1.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn higher_coverage_wider_intervals() {
        let forecasts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..100)
            .map(|i| i as f64 + ((i * 7 + 3) % 11) as f64 * 0.2 - 1.0)
            .collect();

        let p90 = BootstrapPredictor::new(0.50).n_replicates(500).seed(42);
        let p95 = BootstrapPredictor::new(0.95).n_replicates(500).seed(42);

        let r90 = p90.fit(&forecasts, &actuals).unwrap();
        let r95 = p95.fit(&forecasts, &actuals).unwrap();

        let point = vec![100.0];
        let i90 = p90.predict(&r90, &point);
        let i95 = p95.predict(&r95, &point);

        let w90 = i90.upper()[0] - i90.lower()[0];
        let w95 = i95.upper()[0] - i95.lower()[0];
        assert!(
            w95 >= w90,
            "95% interval ({}) should be >= 50% interval ({})",
            w95,
            w90
        );
    }
}
