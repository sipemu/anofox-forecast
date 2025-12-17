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
        self.segments
            .iter()
            .find(|&&(start, end)| index >= start && index < end)
            .copied()
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
            cost: if n > 0 {
                segment_cost(series, config.cost_fn)
            } else {
                0.0
            },
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

    // Precompute cum_ixy for LinearTrend: cumulative sum of i * x[i]
    let cum_ixy: Vec<f64> = std::iter::once(0.0)
        .chain(series.iter().enumerate().scan(0.0, |acc, (i, &x)| {
            *acc += i as f64 * x;
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
                    &cum_ixy,
                    config.cost_fn,
                    series,
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
            let seg_cost = compute_segment_cost_fast(
                s,
                t,
                &cum_sum,
                &cum_sum_sq,
                &cum_ixy,
                config.cost_fn,
                series,
            );
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
///
/// For L2, Normal, MeanVariance, and LinearTrend cost functions, uses O(1) computation.
/// For other cost functions, falls back to direct computation.
fn compute_segment_cost_fast(
    start: usize,
    end: usize,
    cum_sum: &[f64],
    cum_sum_sq: &[f64],
    cum_ixy: &[f64],
    cost_fn: CostFunction,
    series: &[f64],
) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }
    let n_f64 = n as f64;

    match cost_fn {
        CostFunction::L2 | CostFunction::Normal | CostFunction::MeanVariance => {
            // L2 cost = sum((x - mean)^2) = sum(x^2) - n*mean^2
            let sum_y = cum_sum[end] - cum_sum[start];
            let sum_y2 = cum_sum_sq[end] - cum_sum_sq[start];
            let mean = sum_y / n_f64;
            let l2 = sum_y2 - n_f64 * mean * mean;

            match cost_fn {
                CostFunction::Normal if n >= 2 => {
                    let var = l2 / n_f64;
                    if var > 1e-10 {
                        n_f64 * var.ln()
                    } else {
                        0.0
                    }
                }
                CostFunction::MeanVariance if n >= 2 => {
                    let var = l2 / n_f64;
                    if var > 1e-10 {
                        n_f64 * (1.0 + var.ln())
                    } else {
                        n_f64
                    }
                }
                _ => l2.max(0.0),
            }
        }
        CostFunction::LinearTrend => {
            // Fast linear regression using cumulative sums
            // For segment [start, end), we fit y = a + bx where x = 0, 1, ..., n-1
            if n < 2 {
                return 0.0;
            }

            // sum_x = 0 + 1 + ... + (n-1) = n*(n-1)/2
            let sum_x = n_f64 * (n_f64 - 1.0) / 2.0;
            // sum_x2 = 0^2 + 1^2 + ... + (n-1)^2 = n*(n-1)*(2n-1)/6
            let sum_x2 = n_f64 * (n_f64 - 1.0) * (2.0 * n_f64 - 1.0) / 6.0;

            let sum_y = cum_sum[end] - cum_sum[start];
            let sum_y2 = cum_sum_sq[end] - cum_sum_sq[start];

            // sum_xy where x_i = i (relative to segment start)
            // = sum((j - start) * series[j]) for j in [start, end)
            // = sum(j * series[j]) - start * sum(series[j])
            let sum_xy =
                (cum_ixy[end] - cum_ixy[start]) - (start as f64) * (cum_sum[end] - cum_sum[start]);

            // Compute SS_xx, SS_yy, SS_xy
            let ss_xx = sum_x2 - sum_x * sum_x / n_f64;
            let ss_yy = sum_y2 - sum_y * sum_y / n_f64;
            let ss_xy = sum_xy - sum_x * sum_y / n_f64;

            // RSS = SS_yy - SS_xy^2 / SS_xx (if SS_xx > 0)
            if ss_xx.abs() < 1e-10 {
                ss_yy.max(0.0)
            } else {
                (ss_yy - ss_xy * ss_xy / ss_xx).max(0.0)
            }
        }
        // For L1, Poisson, Cusum, Periodicity: fall back to direct computation
        _ => segment_cost(&series[start..end], cost_fn),
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

        // Result should be valid (n_changepoints is usize, always >= 0)
        let _ = result.n_changepoints;
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

    // ==================== Integration tests for new cost functions ====================

    #[test]
    fn pelt_linear_trend_detects_slope_change() {
        // First segment: slope +1 (y = x)
        // Second segment: slope -1 (y = 100 - x)
        let mut series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        series.extend((0..50).map(|i| 100.0 - i as f64));

        let config = PeltConfig::default()
            .cost_function(CostFunction::LinearTrend)
            .penalty(100.0);
        let result = pelt_detect(&series, &config);

        // Should detect the slope change around index 50
        assert!(result.n_changepoints >= 1);
        let cp = result.changepoints[0];
        assert!(
            cp >= 45 && cp <= 55,
            "Expected changepoint near 50, got {}",
            cp
        );
    }

    #[test]
    fn pelt_linear_trend_no_change_for_constant_slope() {
        // Constant slope across entire series
        let series: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 5.0).collect();

        let config = PeltConfig::default()
            .cost_function(CostFunction::LinearTrend)
            .penalty(50.0);
        let result = pelt_detect(&series, &config);

        // No changepoints in a perfectly linear series
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn pelt_mean_variance_detects_variance_shift() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);

        // First segment: low variance (std = 1)
        // Second segment: high variance (std = 10)
        let mut series: Vec<f64> = (0..50).map(|_| 10.0 + rng.gen_range(-1.0..1.0)).collect();
        series.extend((0..50).map(|_| 10.0 + rng.gen_range(-10.0..10.0)));

        let config = PeltConfig::default()
            .cost_function(CostFunction::MeanVariance)
            .penalty(50.0);
        let result = pelt_detect(&series, &config);

        // Should detect variance change
        assert!(result.n_changepoints >= 1);
    }

    #[test]
    fn pelt_mean_variance_detects_joint_change() {
        // First segment: mean=0, variance=1
        // Second segment: mean=10, variance=4
        let mut series: Vec<f64> = vec![
            -0.5, 0.3, -0.2, 0.8, -0.1, 0.4, -0.6, 0.2, -0.3, 0.5, // std ~0.5
            -0.4, 0.1, -0.7, 0.6, -0.2, 0.3, -0.5, 0.4, -0.1, 0.2,
        ];
        series.extend(vec![
            8.0, 12.0, 7.0, 13.0, 9.0, 11.0, 6.0, 14.0, 8.0, 12.0, // mean 10, larger spread
            7.0, 13.0, 8.0, 12.0, 9.0, 11.0, 6.0, 14.0, 7.0, 13.0,
        ]);

        let config = PeltConfig::default()
            .cost_function(CostFunction::MeanVariance)
            .penalty(5.0);
        let result = pelt_detect(&series, &config);

        // Should detect change
        assert!(result.n_changepoints >= 1);
        let cp = result.changepoints[0];
        assert!(
            cp >= 15 && cp <= 25,
            "Expected changepoint near 20, got {}",
            cp
        );
    }

    #[test]
    fn pelt_cusum_detects_sustained_shift() {
        // First segment: centered around 0
        // Second segment: sustained positive shift to 5
        let mut series: Vec<f64> = vec![
            0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.0, 0.1, -0.1, 0.2, -0.2, 0.1,
            -0.3, 0.2, -0.1, 0.0,
        ];
        series.extend(vec![
            5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9,
            5.2, 4.8, 5.0,
        ]);

        let config = PeltConfig::default()
            .cost_function(CostFunction::Cusum)
            .penalty(2.0);
        let result = pelt_detect(&series, &config);

        // Should detect the sustained shift
        assert!(result.n_changepoints >= 1);
        let cp = result.changepoints[0];
        assert!(
            cp >= 15 && cp <= 25,
            "Expected changepoint near 20, got {}",
            cp
        );
    }

    #[test]
    fn pelt_cusum_no_change_for_balanced() {
        // Series that oscillates evenly around mean - no sustained shift
        let series: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let config = PeltConfig::default()
            .cost_function(CostFunction::Cusum)
            .penalty(5.0);
        let result = pelt_detect(&series, &config);

        // Balanced oscillations shouldn't trigger CUSUM changepoints
        // (might still detect some due to segment boundaries)
        assert!(result.n_changepoints <= 2);
    }

    #[test]
    fn pelt_periodicity_detects_period_change() {
        use std::f64::consts::PI;

        // First segment: period 8
        let mut series: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 8.0).sin()).collect();
        // Second segment: period 16
        series.extend((0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()));

        let config = PeltConfig::default()
            .cost_function(CostFunction::Periodicity)
            .penalty(5.0);
        let result = pelt_detect(&series, &config);

        // Should detect the period change
        assert!(result.n_changepoints >= 1);
    }

    #[test]
    fn pelt_periodicity_consistent_period() {
        use std::f64::consts::PI;

        // Consistent period throughout
        let series: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * i as f64 / 12.0).sin())
            .collect();

        let config = PeltConfig::default()
            .cost_function(CostFunction::Periodicity)
            .penalty(20.0);
        let result = pelt_detect(&series, &config);

        // Consistent period should have few or no changepoints
        assert!(result.n_changepoints <= 1);
    }
}
