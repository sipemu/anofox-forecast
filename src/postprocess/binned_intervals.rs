//! Binned conformal prediction for heteroscedastic prediction intervals.
//!
//! Standard conformal prediction produces constant-width intervals. When
//! forecast errors scale with the predicted magnitude (heteroscedasticity),
//! binning residuals by predicted value yields intervals that are wider where
//! uncertainty is larger and tighter where it is smaller.
//!
//! # Algorithm
//!
//! 1. Compute absolute residuals from calibration forecasts and actuals.
//! 2. Sort (forecast, residual) pairs by forecast value.
//! 3. Split into quantile-based bins of equal count.
//! 4. Compute a per-bin conformal quantile (interval half-width).
//! 5. At prediction time, assign each new forecast to a bin and use that
//!    bin's quantile for the interval.
//!
//! # Fallback behaviour
//!
//! - If a bin has fewer than 3 residuals after splitting, it is merged with
//!   its neighbour.
//! - If total data is too small for the requested number of bins, a single
//!   global quantile is used (equivalent to standard conformal prediction).

use crate::error::{ForecastError, Result};
use crate::postprocess::PredictionIntervals;

/// Result of fitting a binned conformal predictor.
#[derive(Debug, Clone)]
pub struct BinnedConformalResult {
    /// Bin edges (n_bins + 1 values, sorted ascending).
    bin_edges: Vec<f64>,
    /// Per-bin conformal quantile (half-width).
    bin_quantiles: Vec<f64>,
    /// Global fallback quantile.
    global_quantile: f64,
    /// Coverage level used.
    coverage: f64,
}

impl BinnedConformalResult {
    /// Get the bin edges (n_bins + 1 values, sorted ascending).
    pub fn bin_edges(&self) -> &[f64] {
        &self.bin_edges
    }

    /// Get the per-bin conformal quantiles.
    pub fn bin_quantiles(&self) -> &[f64] {
        &self.bin_quantiles
    }

    /// Get the global fallback quantile.
    pub fn global_quantile(&self) -> f64 {
        self.global_quantile
    }

    /// Get the coverage level.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }

    /// Get the number of bins.
    pub fn n_bins(&self) -> usize {
        self.bin_quantiles.len()
    }
}

/// Binned conformal predictor for heteroscedastic prediction intervals.
///
/// Bins calibration residuals by predicted magnitude so that intervals
/// widen where errors are systematically larger.
#[derive(Debug, Clone)]
pub struct BinnedConformalPredictor {
    coverage: f64,
    n_bins: usize,
}

impl BinnedConformalPredictor {
    /// Create a new binned conformal predictor.
    ///
    /// # Arguments
    ///
    /// * `coverage` - Target coverage level in (0, 1)
    /// * `n_bins` - Number of bins to use (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if coverage is not in (0, 1) or n_bins < 1.
    pub fn new(coverage: f64, n_bins: usize) -> Self {
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "coverage must be in (0, 1)"
        );
        assert!(n_bins >= 1, "n_bins must be at least 1");
        Self { coverage, n_bins }
    }

    /// Create a binned conformal predictor with 3 bins (default).
    pub fn default_bins(coverage: f64) -> Self {
        Self::new(coverage, 3)
    }

    /// Fit the binned conformal predictor on historical forecasts and actuals.
    ///
    /// # Arguments
    ///
    /// * `forecasts` - Historical point forecasts
    /// * `actuals` - Corresponding actual values
    ///
    /// # Returns
    ///
    /// A `BinnedConformalResult` containing per-bin calibration information.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forecasts and actuals have different lengths
    /// - Data is empty
    pub fn fit(&self, forecasts: &[f64], actuals: &[f64]) -> Result<BinnedConformalResult> {
        if forecasts.len() != actuals.len() {
            return Err(ForecastError::DimensionMismatch {
                expected: forecasts.len(),
                got: actuals.len(),
            });
        }

        let n = forecasts.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }

        // 1. Compute absolute residuals paired with forecast values.
        let mut pairs: Vec<(f64, f64)> = forecasts
            .iter()
            .zip(actuals.iter())
            .map(|(&f, &a)| (f, (f - a).abs()))
            .collect();

        // 2. Sort by forecast value.
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 3. Compute the global quantile as fallback.
        let global_quantile = {
            let mut all_residuals: Vec<f64> = pairs.iter().map(|&(_, r)| r).collect();
            all_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            conformal_quantile(&all_residuals, self.coverage)
        };

        // 4. If data is too small for binning, fall back to a single global bin.
        let min_for_binning = self.n_bins * 3;
        if n < min_for_binning || self.n_bins == 1 {
            // Single-bin fallback: edges span full forecast range.
            let min_fc = pairs.first().unwrap().0;
            let max_fc = pairs.last().unwrap().0;
            return Ok(BinnedConformalResult {
                bin_edges: vec![min_fc, max_fc],
                bin_quantiles: vec![global_quantile],
                global_quantile,
                coverage: self.coverage,
            });
        }

        // 5. Compute quantile-based bin edges from forecast values.
        let forecast_values: Vec<f64> = pairs.iter().map(|&(f, _)| f).collect();
        let mut bin_edges = Vec::with_capacity(self.n_bins + 1);
        for i in 0..=self.n_bins {
            let frac = i as f64 / self.n_bins as f64;
            let idx = ((n as f64 - 1.0) * frac).round() as usize;
            let idx = idx.min(n - 1);
            bin_edges.push(forecast_values[idx]);
        }

        // 6. Assign residuals to bins and collect per-bin residuals.
        let mut bin_residuals: Vec<Vec<f64>> = vec![Vec::new(); self.n_bins];
        for &(fc, res) in &pairs {
            let bin_idx = find_bin(&bin_edges, fc);
            bin_residuals[bin_idx].push(res);
        }

        // 7. Merge bins with fewer than 3 residuals into neighbours.
        let (merged_edges, merged_residuals) = merge_small_bins(bin_edges, bin_residuals);

        // 8. Compute per-bin conformal quantiles.
        let bin_quantiles: Vec<f64> = merged_residuals
            .iter()
            .map(|residuals| {
                let mut sorted = residuals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                conformal_quantile(&sorted, self.coverage)
            })
            .collect();

        Ok(BinnedConformalResult {
            bin_edges: merged_edges,
            bin_quantiles,
            global_quantile,
            coverage: self.coverage,
        })
    }

    /// Generate prediction intervals for new point forecasts.
    ///
    /// # Arguments
    ///
    /// * `result` - The fitted binned conformal result
    /// * `point_forecasts` - New point forecasts to generate intervals for
    ///
    /// # Returns
    ///
    /// Prediction intervals with per-bin heteroscedastic widths.
    pub fn predict(
        &self,
        result: &BinnedConformalResult,
        point_forecasts: &[f64],
    ) -> PredictionIntervals {
        let mut lower = Vec::with_capacity(point_forecasts.len());
        let mut upper = Vec::with_capacity(point_forecasts.len());

        let edges = &result.bin_edges;
        let min_edge = edges.first().copied().unwrap_or(f64::NEG_INFINITY);
        let max_edge = edges.last().copied().unwrap_or(f64::INFINITY);

        for &fc in point_forecasts {
            let q = if fc < min_edge || fc > max_edge {
                // Extrapolation: use global quantile
                result.global_quantile
            } else {
                let bin_idx = find_bin(edges, fc);
                result.bin_quantiles[bin_idx]
            };
            lower.push(fc - q);
            upper.push(fc + q);
        }

        PredictionIntervals::from_bounds(lower, upper, self.coverage)
            .expect("Valid prediction intervals")
    }
}

/// Compute the conformal quantile from sorted residuals.
///
/// Uses the formula: index = ceil((n+1) * coverage) - 1, clamped to [0, n-1].
fn conformal_quantile(sorted_residuals: &[f64], coverage: f64) -> f64 {
    let n = sorted_residuals.len();
    if n == 0 {
        return 0.0;
    }
    let idx = (((n + 1) as f64 * coverage).ceil() as usize).saturating_sub(1);
    let idx = idx.min(n - 1);
    sorted_residuals[idx]
}

/// Find which bin a forecast value falls into using the bin edges.
///
/// Returns a bin index in [0, n_bins - 1]. Values at a boundary are
/// assigned to the lower bin (except for the last edge where they go to the
/// last bin).
fn find_bin(edges: &[f64], value: f64) -> usize {
    let n_bins = edges.len() - 1;
    if n_bins == 0 {
        return 0;
    }
    // Binary search: find the rightmost edge <= value
    match edges.binary_search_by(|e| e.partial_cmp(&value).unwrap()) {
        Ok(pos) => {
            // Exact match on an edge. If it's the last edge, use last bin.
            if pos >= n_bins {
                n_bins - 1
            } else if pos == 0 {
                0
            } else {
                // On an interior edge, assign to the lower bin.
                pos - 1
            }
        }
        Err(pos) => {
            // Value is between edges[pos-1] and edges[pos].
            if pos == 0 {
                0
            } else if pos >= edges.len() {
                n_bins - 1
            } else {
                (pos - 1).min(n_bins - 1)
            }
        }
    }
}

/// Merge bins that have fewer than 3 residuals with an adjacent bin.
///
/// Prefers merging into the neighbour with fewer residuals (to balance
/// bin sizes), but falls back to whichever neighbour exists.
fn merge_small_bins(mut edges: Vec<f64>, mut bins: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    const MIN_BIN_SIZE: usize = 3;

    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < bins.len() {
            if bins[i].len() < MIN_BIN_SIZE && bins.len() > 1 {
                // Choose merge direction
                let merge_into = if i == 0 {
                    // First bin: merge into right neighbour.
                    1
                } else if i == bins.len() - 1 {
                    // Last bin: merge into left neighbour.
                    i - 1
                } else {
                    // Interior: merge into the smaller neighbour.
                    if bins[i - 1].len() <= bins[i + 1].len() {
                        i - 1
                    } else {
                        i + 1
                    }
                };

                if merge_into > i {
                    // Merge i into merge_into (right). Remove the edge between them.
                    let removed = bins.remove(i);
                    // After removal, merge_into shifted down by 1 if it was > i.
                    let target = merge_into - 1;
                    bins[target].extend(removed);
                    // Remove the interior edge at position i+1 (boundary between i and merge_into).
                    // The edge at position `i + 1` in the original edges is the divider.
                    edges.remove(i + 1);
                } else {
                    // Merge i into merge_into (left). Remove the edge between them.
                    let removed = bins.remove(i);
                    bins[merge_into].extend(removed);
                    // Remove the interior edge at position i (boundary between merge_into and i).
                    edges.remove(i);
                }
                changed = true;
                // Don't increment i; re-check at the same position.
            } else {
                i += 1;
            }
        }
    }

    (edges, bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Construction tests
    // =========================================================================

    #[test]
    fn new_creates_predictor() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        assert!((pred.coverage - 0.90).abs() < 1e-10);
        assert_eq!(pred.n_bins, 3);
    }

    #[test]
    fn default_bins_uses_three() {
        let pred = BinnedConformalPredictor::default_bins(0.90);
        assert_eq!(pred.n_bins, 3);
    }

    #[test]
    #[should_panic(expected = "coverage must be in (0, 1)")]
    fn panics_on_invalid_coverage_zero() {
        BinnedConformalPredictor::new(0.0, 3);
    }

    #[test]
    #[should_panic(expected = "coverage must be in (0, 1)")]
    fn panics_on_invalid_coverage_one() {
        BinnedConformalPredictor::new(1.0, 3);
    }

    #[test]
    #[should_panic(expected = "n_bins must be at least 1")]
    fn panics_on_zero_bins() {
        BinnedConformalPredictor::new(0.90, 0);
    }

    // =========================================================================
    // Fit tests
    // =========================================================================

    #[test]
    fn fit_returns_result() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        // 30 data points, enough for 3 bins (3 * 3 = 9 minimum).
        let forecasts: Vec<f64> = (0..30).map(|i| i as f64 * 10.0).collect();
        let actuals: Vec<f64> = forecasts.iter().map(|&f| f + 1.0).collect();

        let result = pred.fit(&forecasts, &actuals).unwrap();

        assert!((result.coverage() - 0.90).abs() < 1e-10);
        assert!(!result.bin_quantiles().is_empty());
        assert!(result.global_quantile() > 0.0);
    }

    #[test]
    fn fit_fails_on_empty() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        let result = pred.fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn fit_fails_on_length_mismatch() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        let result = pred.fit(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn fit_bins_correct_count() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        let forecasts: Vec<f64> = (0..30).map(|i| i as f64 * 10.0).collect();
        let actuals: Vec<f64> = forecasts.iter().map(|&f| f + 1.0).collect();

        let result = pred.fit(&forecasts, &actuals).unwrap();

        // n_bins bins => n_bins + 1 edges
        assert_eq!(result.bin_edges().len(), result.n_bins() + 1);
        assert_eq!(result.bin_quantiles().len(), result.n_bins());
    }

    #[test]
    fn fit_small_data_falls_back_to_single_bin() {
        // 3 bins require at least 9 data points; provide only 5.
        let pred = BinnedConformalPredictor::new(0.90, 3);
        let forecasts = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actuals = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let result = pred.fit(&forecasts, &actuals).unwrap();

        // Fallback: single bin.
        assert_eq!(result.n_bins(), 1);
        assert_eq!(result.bin_edges().len(), 2);
    }

    // =========================================================================
    // Predict tests
    // =========================================================================

    #[test]
    fn predict_returns_intervals() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        let forecasts: Vec<f64> = (0..30).map(|i| i as f64 * 10.0).collect();
        let actuals: Vec<f64> = forecasts.iter().map(|&f| f + 1.0).collect();

        let result = pred.fit(&forecasts, &actuals).unwrap();
        let intervals = pred.predict(&result, &[50.0, 150.0, 250.0]);

        assert_eq!(intervals.len(), 3);
        assert!((intervals.coverage() - 0.90).abs() < 1e-10);
        // All intervals should have lower < upper (since residuals > 0).
        for i in 0..3 {
            assert!(intervals.lower()[i] < intervals.upper()[i]);
        }
    }

    #[test]
    fn predict_wider_for_high_magnitude() {
        // Construct data where residuals scale with forecast magnitude:
        // low forecasts => small errors, high forecasts => large errors.
        let pred = BinnedConformalPredictor::new(0.90, 3);
        let n = 60;
        let forecasts: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 10.0).collect();
        // Errors proportional to forecast value: error ~ forecast * 0.1
        let actuals: Vec<f64> = forecasts
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                f + sign * f * 0.1
            })
            .collect();

        let result = pred.fit(&forecasts, &actuals).unwrap();

        // Predict at a low value and a high value.
        let intervals = pred.predict(&result, &[50.0, 500.0]);
        let width_low = intervals.upper()[0] - intervals.lower()[0];
        let width_high = intervals.upper()[1] - intervals.lower()[1];

        // The interval for the high-magnitude forecast should be wider.
        assert!(
            width_high > width_low,
            "High-magnitude interval width ({}) should exceed low-magnitude width ({})",
            width_high,
            width_low
        );
    }

    #[test]
    fn predict_extrapolation_uses_global() {
        let pred = BinnedConformalPredictor::new(0.90, 3);
        // Training data spans [10, 300].
        let forecasts: Vec<f64> = (1..=30).map(|i| i as f64 * 10.0).collect();
        let actuals: Vec<f64> = forecasts.iter().map(|&f| f + 2.0).collect();

        let result = pred.fit(&forecasts, &actuals).unwrap();

        // Predict well outside the training range.
        let intervals = pred.predict(&result, &[1000.0]);
        let half_width = (intervals.upper()[0] - intervals.lower()[0]) / 2.0;

        assert!(
            (half_width - result.global_quantile()).abs() < 1e-10,
            "Extrapolation should use global quantile ({}) but got half-width {}",
            result.global_quantile(),
            half_width
        );
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn conformal_quantile_basic() {
        let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q = conformal_quantile(&residuals, 0.90);
        // (5+1)*0.9 = 5.4, ceil = 6, -1 = 5, min(4) = 4 => residuals[4] = 5.0
        assert!((q - 5.0).abs() < 1e-10);
    }

    #[test]
    fn conformal_quantile_empty() {
        let q = conformal_quantile(&[], 0.90);
        assert!((q - 0.0).abs() < 1e-10);
    }

    #[test]
    fn find_bin_interior_values() {
        let edges = vec![0.0, 10.0, 20.0, 30.0];
        assert_eq!(find_bin(&edges, 5.0), 0);
        assert_eq!(find_bin(&edges, 15.0), 1);
        assert_eq!(find_bin(&edges, 25.0), 2);
    }

    #[test]
    fn find_bin_boundary_values() {
        let edges = vec![0.0, 10.0, 20.0, 30.0];
        // At an interior edge, assign to the lower bin.
        assert_eq!(find_bin(&edges, 10.0), 0);
        assert_eq!(find_bin(&edges, 20.0), 1);
        // At the last edge, assign to the last bin.
        assert_eq!(find_bin(&edges, 30.0), 2);
        // At the first edge, assign to the first bin.
        assert_eq!(find_bin(&edges, 0.0), 0);
    }

    #[test]
    fn merge_small_bins_leaves_large_bins_alone() {
        let edges = vec![0.0, 10.0, 20.0, 30.0];
        let bins = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let (merged_edges, merged_bins) = merge_small_bins(edges, bins);
        assert_eq!(merged_bins.len(), 3);
        assert_eq!(merged_edges.len(), 4);
    }

    #[test]
    fn merge_small_bins_merges_tiny_bin() {
        let edges = vec![0.0, 10.0, 20.0, 30.0];
        let bins = vec![
            vec![1.0, 2.0], // Too small (< 3)
            vec![4.0, 5.0, 6.0, 7.0],
            vec![8.0, 9.0, 10.0],
        ];
        let (merged_edges, merged_bins) = merge_small_bins(edges, bins);
        // First bin should have been merged, resulting in 2 bins.
        assert_eq!(merged_bins.len(), 2);
        assert_eq!(merged_edges.len(), 3);
    }
}
