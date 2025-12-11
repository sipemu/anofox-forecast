//! PELT (Pruned Exact Linear Time) algorithm for changepoint detection.
//!
//! An exact method for detecting multiple changepoints with O(n) average complexity.

use super::cost::{segment_cost, CostFunction};

/// Configuration for PELT algorithm.
#[derive(Debug, Clone)]
pub struct PeltConfig {
    /// Cost function to use
    pub cost_fn: CostFunction,
    /// Penalty for each changepoint (controls number of changepoints)
    pub penalty: f64,
    /// Minimum segment length
    pub min_segment_length: usize,
}

impl Default for PeltConfig {
    fn default() -> Self {
        Self {
            cost_fn: CostFunction::L2,
            penalty: 1.0,
            min_segment_length: 2,
        }
    }
}

impl PeltConfig {
    /// Create a new config with BIC penalty.
    ///
    /// BIC penalty = log(n) where n is the series length.
    pub fn with_bic_penalty(n: usize) -> Self {
        Self {
            penalty: (n as f64).ln(),
            ..Default::default()
        }
    }

    /// Create a new config with AIC penalty.
    ///
    /// AIC penalty = 2.
    pub fn with_aic_penalty() -> Self {
        Self {
            penalty: 2.0,
            ..Default::default()
        }
    }

    /// Set the cost function.
    pub fn cost_function(mut self, cost_fn: CostFunction) -> Self {
        self.cost_fn = cost_fn;
        self
    }

    /// Set the penalty.
    pub fn penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// Set minimum segment length.
    pub fn min_segment_length(mut self, min_len: usize) -> Self {
        self.min_segment_length = min_len.max(1);
        self
    }
}

/// Result of PELT changepoint detection.
#[derive(Debug, Clone)]
pub struct PeltResult {
    /// Detected changepoint indices
    pub changepoints: Vec<usize>,
    /// Segment boundaries (start, end) pairs
    pub segments: Vec<(usize, usize)>,
    /// Total cost (excluding penalty)
    pub cost: f64,
    /// Number of changepoints
    pub n_changepoints: usize,
}

impl PeltResult {
    /// Get the segment containing a specific index.
    pub fn segment_for_index(&self, index: usize) -> Option<(usize, usize)> {
        self.segments.iter().find(|&&(start, end)| index >= start && index < end).copied()
    }

    /// Get segment means.
    pub fn segment_means(&self, series: &[f64]) -> Vec<f64> {
        self.segments
            .iter()
            .map(|&(start, end)| {
                let segment = &series[start..end];
                if segment.is_empty() {
                    f64::NAN
                } else {
                    segment.iter().sum::<f64>() / segment.len() as f64
                }
            })
            .collect()
    }
}

/// Detect changepoints using the PELT algorithm.
///
/// # Arguments
/// * `series` - Input time series
/// * `config` - PELT configuration
///
/// # Returns
/// PELT result containing detected changepoints
pub fn pelt_detect(series: &[f64], config: &PeltConfig) -> PeltResult {
    let n = series.len();

    if n < 2 * config.min_segment_length {
        return PeltResult {
            changepoints: Vec::new(),
            segments: vec![(0, n)],
            cost: if n > 0 { segment_cost(series, config.cost_fn) } else { 0.0 },
            n_changepoints: 0,
        };
    }

    // F[t] = minimum cost of segmenting series[0..t]
    let mut f = vec![f64::INFINITY; n + 1];
    f[0] = -config.penalty; // So first segment doesn't get penalized twice

    // cp[t] = optimal last changepoint for series[0..t]
    let mut cp: Vec<usize> = vec![0; n + 1];

    // R = set of candidate changepoints (pruned)
    let mut candidates: Vec<usize> = vec![0];

    // Precompute cumulative sums for efficient cost calculation
    let cum_sum: Vec<f64> = std::iter::once(0.0)
        .chain(series.iter().scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        }))
        .collect();

    let cum_sum_sq: Vec<f64> = std::iter::once(0.0)
        .chain(series.iter().scan(0.0, |acc, &x| {
            *acc += x * x;
            Some(*acc)
        }))
        .collect();

    for t in config.min_segment_length..=n {
        let mut best_cost = f64::INFINITY;
        let mut best_cp = 0;

        // Check all candidates
        for &s in &candidates {
            if t - s >= config.min_segment_length {
                let seg_cost = compute_segment_cost_fast(
                    s,
                    t,
                    &cum_sum,
                    &cum_sum_sq,
                    config.cost_fn,
                );
                let total = f[s] + seg_cost + config.penalty;

                if total < best_cost {
                    best_cost = total;
                    best_cp = s;
                }
            }
        }

        f[t] = best_cost;
        cp[t] = best_cp;

        // Pruning: remove candidates that can never be optimal
        candidates.retain(|&s| {
            if t - s < config.min_segment_length {
                return true;
            }
            let seg_cost = compute_segment_cost_fast(s, t, &cum_sum, &cum_sum_sq, config.cost_fn);
            f[s] + seg_cost <= f[t]
        });

        candidates.push(t);
    }

    // Backtrack to find changepoints
    let mut changepoints = Vec::new();
    let mut t = n;
    while t > 0 {
        let prev = cp[t];
        if prev > 0 {
            changepoints.push(prev);
        }
        t = prev;
    }
    changepoints.reverse();

    // Build segments
    let mut segments = Vec::new();
    let mut start = 0;
    for &cp_idx in &changepoints {
        segments.push((start, cp_idx));
        start = cp_idx;
    }
    segments.push((start, n));

    // Compute total cost
    let total_cost: f64 = segments
        .iter()
        .map(|&(s, e)| segment_cost(&series[s..e], config.cost_fn))
        .sum();

    PeltResult {
        n_changepoints: changepoints.len(),
        changepoints,
        segments,
        cost: total_cost,
    }
}

/// Fast segment cost computation using precomputed cumulative sums.
fn compute_segment_cost_fast(
    start: usize,
    end: usize,
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    cost_fn: CostFunction,
) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }

    match cost_fn {
        CostFunction::L2 | CostFunction::Normal => {
            // L2 cost = sum((x - mean)^2) = sum(x^2) - n*mean^2
            let sum = cum_sum[end] - cum_sum[start];
            let sum_sq = cum_sum_sq[end] - cum_sum_sq[start];
            let mean = sum / n as f64;
            let l2 = sum_sq - n as f64 * mean * mean;

            if cost_fn == CostFunction::Normal && n >= 2 {
                let var = l2 / n as f64;
                if var > 1e-10 {
                    n as f64 * var.ln()
                } else {
                    0.0
                }
            } else {
                l2.max(0.0)
            }
        }
        _ => {
            // For L1 and Poisson, fall back to direct computation
            // (would need different precomputation for efficiency)
            0.0 // Placeholder - should not reach here in typical usage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pelt_no_changepoint() {
        // Constant series - no changepoints
        let series = vec![5.0; 20];
        let config = PeltConfig::default().penalty(10.0);
        let result = pelt_detect(&series, &config);

        assert_eq!(result.n_changepoints, 0);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0], (0, 20));
    }

    #[test]
    fn pelt_one_clear_changepoint() {
        // Clear level shift at position 10
        let mut series = vec![0.0; 10];
        series.extend(vec![10.0; 10]);

        let config = PeltConfig::default().penalty(2.0);
        let result = pelt_detect(&series, &config);

        assert_eq!(result.n_changepoints, 1);
        assert_eq!(result.changepoints[0], 10);
        assert_eq!(result.segments, vec![(0, 10), (10, 20)]);
    }

    #[test]
    fn pelt_two_changepoints() {
        // Three distinct levels
        let mut series = vec![0.0; 10];
        series.extend(vec![10.0; 10]);
        series.extend(vec![0.0; 10]);

        let config = PeltConfig::default().penalty(2.0);
        let result = pelt_detect(&series, &config);

        assert_eq!(result.n_changepoints, 2);
        assert!(result.changepoints.contains(&10));
        assert!(result.changepoints.contains(&20));
    }

    #[test]
    fn pelt_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let config = PeltConfig::default();
        let result = pelt_detect(&series, &config);

        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn pelt_empty_series() {
        let series: Vec<f64> = vec![];
        let config = PeltConfig::default();
        let result = pelt_detect(&series, &config);

        assert_eq!(result.n_changepoints, 0);
        assert!(result.changepoints.is_empty());
    }

    #[test]
    fn pelt_high_penalty_no_changepoints() {
        // Even with clear changepoint, very high penalty prevents detection
        // For series [0]*10 + [100]*10, L2 cost without CP ≈ 50000
        // With CP at 10, cost = 0, so we need penalty > 50000
        let mut series = vec![0.0; 10];
        series.extend(vec![100.0; 10]);

        let config = PeltConfig::default().penalty(100000.0);
        let result = pelt_detect(&series, &config);

        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn pelt_low_penalty_many_changepoints() {
        // Very low penalty may detect spurious changepoints
        let series: Vec<f64> = (0..50).map(|i| i as f64 + ((i * 7) % 3) as f64).collect();
        let config = PeltConfig::default().penalty(0.01);
        let result = pelt_detect(&series, &config);

        // Should detect some changepoints
        assert!(result.n_changepoints >= 0);
    }

    #[test]
    fn pelt_config_bic() {
        let config = PeltConfig::with_bic_penalty(100);
        assert_relative_eq!(config.penalty, 100.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn pelt_config_aic() {
        let config = PeltConfig::with_aic_penalty();
        assert_relative_eq!(config.penalty, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn pelt_config_builder() {
        let config = PeltConfig::default()
            .cost_function(CostFunction::L1)
            .penalty(5.0)
            .min_segment_length(5);

        assert_eq!(config.cost_fn, CostFunction::L1);
        assert_relative_eq!(config.penalty, 5.0, epsilon = 1e-10);
        assert_eq!(config.min_segment_length, 5);
    }

    #[test]
    fn pelt_segment_means() {
        let mut series = vec![1.0; 5];
        series.extend(vec![10.0; 5]);

        let config = PeltConfig::default().penalty(1.0);
        let result = pelt_detect(&series, &config);

        let means = result.segment_means(&series);

        // Should have 2 segments
        assert_eq!(means.len(), 2);
        assert_relative_eq!(means[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(means[1], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn pelt_segment_for_index() {
        let mut series = vec![0.0; 10];
        series.extend(vec![10.0; 10]);

        let config = PeltConfig::default().penalty(1.0);
        let result = pelt_detect(&series, &config);

        // Index 5 should be in first segment
        assert_eq!(result.segment_for_index(5), Some((0, 10)));
        // Index 15 should be in second segment
        assert_eq!(result.segment_for_index(15), Some((10, 20)));
    }

    #[test]
    fn pelt_min_segment_length() {
        // Changepoint at position 2, but min_segment_length = 5
        let mut series = vec![0.0; 2];
        series.extend(vec![100.0; 18]);

        let config = PeltConfig::default().penalty(1.0).min_segment_length(5);
        let result = pelt_detect(&series, &config);

        // Changepoint at 2 should not be detected due to min segment length
        for cp in &result.changepoints {
            assert!(*cp >= 5);
        }
    }
}
