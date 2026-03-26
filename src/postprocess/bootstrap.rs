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
use crate::postprocess::{PredictionIntervals, QuantileForecasts};
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
    /// then extracts per-step quantile bounds at `(1-coverage)/2` and
    /// `1-(1-coverage)/2`.
    pub fn predict(&self, result: &BootstrapResult, point_forecast: &[f64]) -> PredictionIntervals {
        let alpha = (1.0 - result.coverage) / 2.0;
        let quantiles = vec![alpha, 1.0 - alpha];
        let samples = simulate_paths(result, point_forecast);
        let values = extract_quantiles(&samples, &quantiles);

        PredictionIntervals::from_bounds(values[0].clone(), values[1].clone(), result.coverage)
            .expect("Valid prediction intervals")
    }

    /// Generate quantile forecasts at multiple quantile levels.
    ///
    /// Returns a `QuantileForecasts` with one column per requested quantile
    /// level, each of length `horizon`. Quantile levels must be in (0, 1).
    ///
    /// # Example
    /// ```ignore
    /// let quantiles = predictor.predict_quantiles(
    ///     &result,
    ///     &point_forecast,
    ///     &[0.10, 0.25, 0.50, 0.75, 0.90],
    /// );
    /// ```
    pub fn predict_quantiles(
        &self,
        result: &BootstrapResult,
        point_forecast: &[f64],
        quantile_levels: &[f64],
    ) -> QuantileForecasts {
        let samples = simulate_paths(result, point_forecast);
        let values = extract_quantiles(&samples, quantile_levels);

        // values[q][h] → need to reshape to QuantileForecasts format: [h][q]
        let horizon = point_forecast.len();
        let n_q = quantile_levels.len();
        let mut forecast_values = Vec::with_capacity(horizon);
        for h in 0..horizon {
            let row: Vec<f64> = (0..n_q).map(|q| values[q][h]).collect();
            forecast_values.push(row);
        }

        QuantileForecasts::from_values(quantile_levels.to_vec(), forecast_values)
            .expect("Valid quantile forecasts")
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

/// Simulate bootstrap paths: resample residuals, accumulate errors, add to point forecast.
///
/// Returns `samples_per_step[h]` = sorted Vec of simulated values at step h.
fn simulate_paths(result: &BootstrapResult, point_forecast: &[f64]) -> Vec<Vec<f64>> {
    let horizon = point_forecast.len();
    let n_rep = result.n_replicates;

    let mut rng: StdRng = match result.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut samples_per_step: Vec<Vec<f64>> = vec![Vec::with_capacity(n_rep); horizon];

    for _ in 0..n_rep {
        let errors = resample(&result.residuals, horizon, result.block_size, &mut rng);
        let mut cumulative_error = 0.0;
        for (h, &err) in errors.iter().enumerate() {
            cumulative_error += err;
            let simulated = point_forecast[h] + cumulative_error;
            if simulated.is_finite() {
                samples_per_step[h].push(simulated);
            }
        }
    }

    // Sort each step's samples for quantile extraction
    for samples in &mut samples_per_step {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    samples_per_step
}

/// Extract quantile values from sorted samples.
///
/// Returns `values[q][h]` for each quantile level q and horizon step h.
fn extract_quantiles(samples_per_step: &[Vec<f64>], quantile_levels: &[f64]) -> Vec<Vec<f64>> {
    let horizon = samples_per_step.len();
    let mut result = Vec::with_capacity(quantile_levels.len());

    for &q in quantile_levels {
        let mut col = Vec::with_capacity(horizon);
        for samples in samples_per_step {
            if samples.is_empty() {
                col.push(f64::NAN);
                continue;
            }
            let n = samples.len();
            let idx = ((q * n as f64).floor() as usize).min(n - 1);
            col.push(samples[idx]);
        }
        result.push(col);
    }

    result
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

    // =========================================================================
    // Interval-contains-point-forecast tests
    // =========================================================================

    #[test]
    fn intervals_contain_point_forecast() {
        // With moderate residuals the point forecast should lie within bounds
        // for most (not necessarily all) steps.
        let n = 200;
        let forecasts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..n)
            .map(|i| i as f64 + ((i * 13 + 7) % 17) as f64 * 0.3 - 2.5)
            .collect();

        let predictor = BootstrapPredictor::new(0.95).n_replicates(2000).seed(99);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point: Vec<f64> = (200..210).map(|i| i as f64).collect();
        let intervals = predictor.predict(&result, &point);

        let mut contained = 0;
        for i in 0..point.len() {
            if intervals.lower()[i] <= point[i] && point[i] <= intervals.upper()[i] {
                contained += 1;
            }
        }
        // With 95% coverage, most steps should contain the point forecast
        assert!(
            contained >= 7,
            "Expected at least 7/10 steps to contain the point forecast, got {}/10",
            contained
        );
    }

    // =========================================================================
    // Wider coverage produces wider intervals (50% vs 99%)
    // =========================================================================

    #[test]
    fn wider_coverage_produces_wider_intervals_50_vs_99() {
        let n = 200;
        let forecasts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..n)
            .map(|i| i as f64 + ((i * 11 + 5) % 23) as f64 * 0.4 - 4.0)
            .collect();

        let p50 = BootstrapPredictor::new(0.50).n_replicates(1000).seed(77);
        let p99 = BootstrapPredictor::new(0.99).n_replicates(1000).seed(77);

        let r50 = p50.fit(&forecasts, &actuals).unwrap();
        let r99 = p99.fit(&forecasts, &actuals).unwrap();

        let point = vec![250.0, 260.0, 270.0];
        let i50 = p50.predict(&r50, &point);
        let i99 = p99.predict(&r99, &point);

        for h in 0..3 {
            let w50 = i50.upper()[h] - i50.lower()[h];
            let w99 = i99.upper()[h] - i99.lower()[h];
            assert!(
                w99 > w50,
                "Step {}: 99% width ({}) should exceed 50% width ({})",
                h,
                w99,
                w50
            );
        }
    }

    // =========================================================================
    // Same seed → identical intervals (stability)
    // =========================================================================

    #[test]
    fn same_seed_produces_identical_intervals() {
        let forecasts: Vec<f64> = (0..80).map(|i| i as f64 * 1.5).collect();
        let actuals: Vec<f64> = (0..80)
            .map(|i| i as f64 * 1.5 + ((i * 3 + 1) % 9) as f64 * 0.2 - 0.8)
            .collect();

        let point = vec![120.0, 121.5, 123.0, 124.5, 126.0];

        // Run 1
        let p1 = BootstrapPredictor::new(0.90).n_replicates(500).seed(555);
        let r1 = p1.fit(&forecasts, &actuals).unwrap();
        let iv1 = p1.predict(&r1, &point);

        // Run 2 — same seed
        let p2 = BootstrapPredictor::new(0.90).n_replicates(500).seed(555);
        let r2 = p2.fit(&forecasts, &actuals).unwrap();
        let iv2 = p2.predict(&r2, &point);

        for h in 0..5 {
            assert!(
                (iv1.lower()[h] - iv2.lower()[h]).abs() < 1e-10,
                "Lower bounds differ at step {}",
                h
            );
            assert!(
                (iv1.upper()[h] - iv2.upper()[h]).abs() < 1e-10,
                "Upper bounds differ at step {}",
                h
            );
        }
    }

    // =========================================================================
    // Block vs IID bootstrap produce different intervals for autocorrelated residuals
    // =========================================================================

    #[test]
    fn block_vs_iid_differ_for_autocorrelated_residuals() {
        // Build residuals with strong structure: a sawtooth pattern where
        // nearby residuals are similar (autocorrelated). Block bootstrap
        // preserves this structure while IID destroys it.
        let n = 200;
        let forecasts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        // Sawtooth: residuals ramp from -5 to +5 in blocks of 20, then reset.
        // Adjacent residuals are very similar; IID will scramble this.
        let actuals: Vec<f64> = (0..n)
            .map(|i| {
                let phase = (i % 20) as f64;
                let err = phase * 0.5 - 5.0; // ranges from -5.0 to +4.5
                i as f64 + err
            })
            .collect();

        let point = vec![200.0, 201.0, 202.0, 203.0, 204.0];

        // Use different seeds to break any coincidental equality from
        // the RNG consuming the same number of random values.
        let iid = BootstrapPredictor::new(0.90).n_replicates(2000).seed(42);
        let block = BootstrapPredictor::new(0.90)
            .n_replicates(2000)
            .block_size(10)
            .seed(43);

        let r_iid = iid.fit(&forecasts, &actuals).unwrap();
        let r_block = block.fit(&forecasts, &actuals).unwrap();

        let iv_iid = iid.predict(&r_iid, &point);
        let iv_block = block.predict(&r_block, &point);

        // At least one step should show a meaningful difference in bounds
        let mut any_differ = false;
        for h in 0..5 {
            let lo_diff = (iv_iid.lower()[h] - iv_block.lower()[h]).abs();
            let hi_diff = (iv_iid.upper()[h] - iv_block.upper()[h]).abs();
            if lo_diff > 1e-6 || hi_diff > 1e-6 {
                any_differ = true;
                break;
            }
        }
        assert!(
            any_differ,
            "Block and IID bootstrap should produce different interval bounds for autocorrelated residuals"
        );
    }

    // =========================================================================
    // All NaN residuals → error
    // =========================================================================

    #[test]
    fn all_nan_residuals_errors() {
        let predictor = BootstrapPredictor::new(0.90);
        let forecasts = vec![f64::NAN, f64::NAN, f64::NAN];
        let actuals = vec![1.0, 2.0, 3.0];
        let result = predictor.fit(&forecasts, &actuals);
        assert!(result.is_err(), "All-NaN residuals should produce an error");
    }

    // =========================================================================
    // Residuals with some NaN values → filtered out, still works
    // =========================================================================

    #[test]
    fn some_nan_residuals_filtered_and_works() {
        let predictor = BootstrapPredictor::new(0.90).n_replicates(100).seed(42);
        // Mix of finite and NaN forecast values
        let mut forecasts = vec![f64::NAN; 5];
        forecasts.extend((0..30).map(|i| i as f64));
        let actuals: Vec<f64> = (0..35).map(|i| i as f64 * 0.9).collect();

        let result = predictor.fit(&forecasts, &actuals).unwrap();
        // Only 30 finite residuals should remain
        assert_eq!(result.residuals().len(), 30);

        let point = vec![35.0, 36.0];
        let intervals = predictor.predict(&result, &point);
        assert_eq!(intervals.len(), 2);
        for i in 0..2 {
            assert!(intervals.lower()[i].is_finite());
            assert!(intervals.upper()[i].is_finite());
            assert!(intervals.lower()[i] <= intervals.upper()[i]);
        }
    }

    // =========================================================================
    // Horizon = 1 works
    // =========================================================================

    #[test]
    fn horizon_one_works() {
        let forecasts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.3).collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(200).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let intervals = predictor.predict(&result, &[50.0]);
        assert_eq!(intervals.len(), 1);
        assert!(intervals.lower()[0] <= intervals.upper()[0]);
        assert!(intervals.lower()[0].is_finite());
        assert!(intervals.upper()[0].is_finite());
    }

    // =========================================================================
    // Large horizon (50 steps) works without panic
    // =========================================================================

    #[test]
    fn large_horizon_50_steps_works() {
        let forecasts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..100)
            .map(|i| i as f64 + ((i * 3 + 1) % 7) as f64 * 0.5 - 1.5)
            .collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(500).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point: Vec<f64> = (100..150).map(|i| i as f64).collect();
        assert_eq!(point.len(), 50);

        let intervals = predictor.predict(&result, &point);
        assert_eq!(intervals.len(), 50);
        for i in 0..50 {
            assert!(
                intervals.lower()[i] <= intervals.upper()[i],
                "lower > upper at step {}",
                i
            );
            assert!(intervals.lower()[i].is_finite(), "lower NaN at step {}", i);
            assert!(intervals.upper()[i].is_finite(), "upper NaN at step {}", i);
        }
    }

    // =========================================================================
    // Very small residuals → very narrow intervals
    // =========================================================================

    #[test]
    fn very_small_residuals_give_narrow_intervals() {
        let n = 100;
        let forecasts: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        // Actuals differ by at most 1e-8
        let actuals: Vec<f64> = (0..n)
            .map(|i| i as f64 * 10.0 + ((i % 3) as f64 - 1.0) * 1e-8)
            .collect();

        let predictor = BootstrapPredictor::new(0.95).n_replicates(500).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point = vec![1000.0, 1010.0, 1020.0];
        let intervals = predictor.predict(&result, &point);

        for h in 0..3 {
            let width = intervals.upper()[h] - intervals.lower()[h];
            assert!(
                width < 1e-6,
                "Step {}: interval width {} should be very narrow for near-zero residuals",
                h,
                width
            );
        }
    }

    // ── predict_quantiles tests ──────────────────────────────────

    #[test]
    fn predict_quantiles_returns_correct_shape() {
        let forecasts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(200).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point = vec![50.0, 51.0, 52.0, 53.0, 54.0];
        let levels = vec![0.10, 0.25, 0.50, 0.75, 0.90];
        let qf = predictor.predict_quantiles(&result, &point, &levels);

        assert_eq!(qf.quantiles().len(), 5);
        assert_eq!(qf.n_times(), 5); // 5 horizon steps
    }

    #[test]
    fn predict_quantiles_monotonically_ordered() {
        let forecasts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..100)
            .map(|i| i as f64 + ((i * 7 + 3) % 11) as f64 * 0.3 - 1.5)
            .collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(500).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point = vec![100.0, 101.0, 102.0];
        let levels = vec![0.05, 0.25, 0.50, 0.75, 0.95];
        let qf = predictor.predict_quantiles(&result, &point, &levels);

        // At each step, quantiles should be monotonically non-decreasing
        for h in 0..3 {
            let row = qf.at_time(h).unwrap();
            for i in 1..row.len() {
                assert!(
                    row[i] >= row[i - 1],
                    "Step {}: q[{}]={} < q[{}]={}",
                    h,
                    i,
                    row[i],
                    i - 1,
                    row[i - 1]
                );
            }
        }
    }

    #[test]
    fn predict_quantiles_median_near_point_forecast() {
        let forecasts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Symmetric residuals centered at 0
        let actuals: Vec<f64> = (0..100)
            .map(|i| i as f64 + ((i % 2) as f64 * 2.0 - 1.0) * 0.1)
            .collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(1000).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let point = vec![100.0];
        let qf = predictor.predict_quantiles(&result, &point, &[0.50]);

        // Median should be close to point forecast for symmetric residuals
        let median = qf.at_time(0).unwrap()[0];
        assert!(
            (median - 100.0).abs() < 2.0,
            "Median {} should be near point forecast 100.0",
            median
        );
    }

    #[test]
    fn predict_quantiles_single_level() {
        let forecasts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let actuals: Vec<f64> = (0..50).map(|i| i as f64 + 0.3).collect();

        let predictor = BootstrapPredictor::new(0.90).n_replicates(100).seed(42);
        let result = predictor.fit(&forecasts, &actuals).unwrap();

        let qf = predictor.predict_quantiles(&result, &[50.0, 51.0], &[0.50]);
        assert_eq!(qf.quantiles().len(), 1);
        assert_eq!(qf.n_times(), 2);
    }
}
