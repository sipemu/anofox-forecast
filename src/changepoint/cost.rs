//! Cost functions for changepoint detection.
//!
//! Cost functions evaluate the "cost" of fitting a model to a segment of data.
//! Lower cost indicates a better fit.

use crate::detection::periodogram;

/// Cost function type.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CostFunction {
    /// L1 cost: sum of absolute deviations from median
    L1,
    /// L2 cost: sum of squared deviations from mean (normal likelihood)
    #[default]
    L2,
    /// Normal likelihood cost (equivalent to L2 with variance estimation)
    Normal,
    /// Poisson cost for count data
    Poisson,
    /// Linear trend cost: RSS from linear regression fit.
    /// Detects changes in slope/trend.
    LinearTrend,
    /// Mean-Variance joint cost: detects simultaneous mean AND variance changes.
    /// More sensitive than Normal for joint changes.
    MeanVariance,
    /// CUSUM cost: cumulative sum based detection for sustained mean shifts.
    /// Good for quality control / monitoring applications.
    Cusum,
    /// Periodicity cost: detects changes in seasonal patterns using FFT.
    Periodicity,
}

/// Compute the cost of a segment using the specified cost function.
///
/// # Arguments
/// * `segment` - The data segment
/// * `cost_fn` - The cost function to use
pub fn segment_cost(segment: &[f64], cost_fn: CostFunction) -> f64 {
    match cost_fn {
        CostFunction::L1 => l1_cost(segment),
        CostFunction::L2 => l2_cost(segment),
        CostFunction::Normal => normal_cost(segment),
        CostFunction::Poisson => poisson_cost(segment),
        CostFunction::LinearTrend => linear_trend_cost(segment),
        CostFunction::MeanVariance => mean_variance_cost(segment),
        CostFunction::Cusum => cusum_cost(segment),
        CostFunction::Periodicity => periodicity_cost(segment),
    }
}

/// L1 cost: sum of absolute deviations from median.
///
/// Robust to outliers.
pub fn l1_cost(segment: &[f64]) -> f64 {
    if segment.is_empty() {
        return 0.0;
    }

    let median = compute_median(segment);
    segment.iter().map(|x| (x - median).abs()).sum()
}

/// L2 cost: sum of squared deviations from mean.
///
/// Also known as residual sum of squares (RSS).
pub fn l2_cost(segment: &[f64]) -> f64 {
    if segment.is_empty() {
        return 0.0;
    }

    let mean = segment.iter().sum::<f64>() / segment.len() as f64;
    segment.iter().map(|x| (x - mean).powi(2)).sum()
}

/// Normal (Gaussian) cost: negative log-likelihood assuming constant mean and variance.
///
/// Cost = n * log(variance) (ignoring constant terms)
pub fn normal_cost(segment: &[f64]) -> f64 {
    let n = segment.len();
    if n < 2 {
        return 0.0;
    }

    let mean = segment.iter().sum::<f64>() / n as f64;
    let variance = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-10 {
        return 0.0; // Constant segment
    }

    n as f64 * variance.ln()
}

/// Poisson cost: negative log-likelihood for count data.
///
/// Assumes segment values are non-negative counts.
/// Cost = sum(x_i) - n * mean * log(mean) (simplified)
pub fn poisson_cost(segment: &[f64]) -> f64 {
    let n = segment.len();
    if n == 0 {
        return 0.0;
    }

    let sum: f64 = segment.iter().sum();
    let mean = sum / n as f64;

    if mean < 1e-10 {
        return 0.0;
    }

    // Negative log-likelihood (simplified, ignoring factorial terms)
    n as f64 * mean - sum * mean.ln()
}

/// Linear trend cost: residual sum of squares from linear regression.
///
/// Detects changes in slope/trend by fitting y = a + bx to the segment.
/// Cost = RSS = sum((y_i - (a + b*i))^2)
///
/// Lower cost indicates the segment fits well to a single linear trend.
pub fn linear_trend_cost(segment: &[f64]) -> f64 {
    let n = segment.len();
    if n < 2 {
        return 0.0;
    }

    let n_f64 = n as f64;

    // Compute sums for linear regression
    // x values are indices 0, 1, 2, ..., n-1
    let sum_x: f64 = (n * (n - 1)) as f64 / 2.0; // 0 + 1 + ... + (n-1)
    let sum_x2: f64 = ((n - 1) * n * (2 * n - 1)) as f64 / 6.0; // sum of i^2
    let sum_y: f64 = segment.iter().sum();
    let sum_xy: f64 = segment.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();

    let mean_x = sum_x / n_f64;
    let mean_y = sum_y / n_f64;

    let ss_xx = sum_x2 - n_f64 * mean_x * mean_x;
    let ss_xy = sum_xy - n_f64 * mean_x * mean_y;
    let ss_yy: f64 = segment.iter().map(|&y| (y - mean_y).powi(2)).sum();

    // Handle degenerate case
    if ss_xx.abs() < 1e-10 {
        return ss_yy.max(0.0); // Fall back to L2 cost
    }

    // RSS = SS_yy - SS_xy^2 / SS_xx
    let rss = ss_yy - (ss_xy * ss_xy) / ss_xx;

    rss.max(0.0)
}

/// Mean-Variance joint cost: detects simultaneous mean AND variance changes.
///
/// Based on normal likelihood with explicit mean+variance parameters.
/// More sensitive than Normal cost which uses just log(variance).
///
/// Cost = n * (1 + log(variance)) where variance = sum((x-mean)^2)/n
pub fn mean_variance_cost(segment: &[f64]) -> f64 {
    let n = segment.len();
    if n < 2 {
        return 0.0;
    }

    let n_f64 = n as f64;
    let mean = segment.iter().sum::<f64>() / n_f64;
    let variance = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_f64;

    if variance < 1e-10 {
        return 0.0; // Constant segment
    }

    // Cost based on negative log-likelihood of normal distribution
    // -2 * log(L) = n * log(2*pi) + n * log(sigma^2) + sum((x-mu)^2)/sigma^2
    //             = n * log(2*pi) + n * log(sigma^2) + n  (since sum/sigma^2 = n)
    // Ignoring constants: n * (1 + log(variance))
    n_f64 * (1.0 + variance.ln())
}

/// CUSUM cost: cumulative sum based detection for sustained mean shifts.
///
/// Based on CUSUM (Cumulative Sum) control charts.
/// Detects sustained shifts in mean level rather than random fluctuations.
/// Good for quality control and monitoring applications.
///
/// Cost is based on the maximum absolute cumulative deviation from the mean.
/// Lower cost = data stays close to mean throughout segment (no sustained shift).
pub fn cusum_cost(segment: &[f64]) -> f64 {
    let n = segment.len();
    if n < 2 {
        return 0.0;
    }

    let mean = segment.iter().sum::<f64>() / n as f64;

    // Compute maximum absolute cumulative deviation
    let mut cumulative = 0.0_f64;
    let mut max_cusum = 0.0_f64;

    for &x in segment {
        cumulative += x - mean;
        max_cusum = max_cusum.max(cumulative.abs());
    }

    max_cusum
}

/// Periodicity cost: detects changes in seasonal patterns using FFT.
///
/// Based on FFT/periodogram analysis of the segment.
/// Measures the deviation from the dominant periodic pattern.
/// Cost = total variance * (1 - periodicity_strength)
///
/// Lower cost = segment has strong, consistent periodicity.
/// Changes in seasonal pattern will create segment boundaries.
pub fn periodicity_cost(segment: &[f64]) -> f64 {
    let n = segment.len();
    if n < 8 {
        // Too short for meaningful periodicity analysis
        return l2_cost(segment); // Fall back to L2
    }

    let mean = segment.iter().sum::<f64>() / n as f64;
    let total_variance: f64 = segment.iter().map(|x| (x - mean).powi(2)).sum();

    if total_variance < 1e-10 {
        return 0.0; // Constant segment
    }

    // Compute periodogram to find dominant frequencies
    let psd = periodogram(segment);

    if psd.is_empty() {
        return l2_cost(segment);
    }

    // Sum of top k frequency powers
    let k_top = 3.min(psd.len());
    let mut powers: Vec<f64> = psd.iter().map(|(_, p)| *p).collect();
    powers.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let explained_power: f64 = powers.iter().take(k_top).sum();
    let total_power: f64 = powers.iter().sum();

    if total_power < 1e-10 {
        return total_variance;
    }

    // Periodicity strength: fraction of variance explained by dominant frequencies
    let periodicity_strength = explained_power / total_power;

    // Cost = unexplained variance
    // High periodicity_strength -> low cost (good periodic fit)
    total_variance * (1.0 - periodicity_strength * 0.9) // Scale factor to avoid zero cost
}

/// Compute the cost for the entire series given changepoint locations.
///
/// # Arguments
/// * `series` - The full time series
/// * `changepoints` - Indices where segments change (sorted)
/// * `cost_fn` - Cost function to use
pub fn total_cost(series: &[f64], changepoints: &[usize], cost_fn: CostFunction) -> f64 {
    if series.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    let mut start = 0;

    for &cp in changepoints {
        if cp > start && cp <= series.len() {
            total += segment_cost(&series[start..cp], cost_fn);
            start = cp;
        }
    }

    // Add cost of final segment
    if start < series.len() {
        total += segment_cost(&series[start..], cost_fn);
    }

    total
}

/// Helper: compute median
fn compute_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== l1_cost ====================

    #[test]
    fn l1_cost_empty() {
        assert_relative_eq!(l1_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l1_cost_single() {
        assert_relative_eq!(l1_cost(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l1_cost_constant() {
        let segment = vec![5.0; 10];
        assert_relative_eq!(l1_cost(&segment), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l1_cost_known() {
        // [1, 2, 3, 4, 5] -> median = 3
        // |1-3| + |2-3| + |3-3| + |4-3| + |5-3| = 2+1+0+1+2 = 6
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(l1_cost(&segment), 6.0, epsilon = 1e-10);
    }

    // ==================== l2_cost ====================

    #[test]
    fn l2_cost_empty() {
        assert_relative_eq!(l2_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_cost_single() {
        assert_relative_eq!(l2_cost(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_cost_constant() {
        let segment = vec![5.0; 10];
        assert_relative_eq!(l2_cost(&segment), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn l2_cost_known() {
        // [1, 2, 3, 4, 5] -> mean = 3
        // (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)² = 4+1+0+1+4 = 10
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(l2_cost(&segment), 10.0, epsilon = 1e-10);
    }

    // ==================== normal_cost ====================

    #[test]
    fn normal_cost_empty() {
        assert_relative_eq!(normal_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn normal_cost_single() {
        assert_relative_eq!(normal_cost(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn normal_cost_constant() {
        let segment = vec![5.0; 10];
        assert_relative_eq!(normal_cost(&segment), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn normal_cost_known() {
        // [1, 2, 3, 4, 5] -> mean = 3, variance = 10/5 = 2
        // cost = 5 * ln(2) ≈ 3.466
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 5.0 * 2.0_f64.ln();
        assert_relative_eq!(normal_cost(&segment), expected, epsilon = 1e-10);
    }

    // ==================== poisson_cost ====================

    #[test]
    fn poisson_cost_empty() {
        assert_relative_eq!(poisson_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn poisson_cost_zeros() {
        let segment = vec![0.0; 10];
        assert_relative_eq!(poisson_cost(&segment), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn poisson_cost_constant() {
        let segment = vec![5.0; 10];
        // mean = 5, cost = 10*5 - 50*ln(5) = 50 - 50*ln(5) ≈ 50 - 80.47 = -30.47
        let cost = poisson_cost(&segment);
        assert!(!cost.is_nan());
    }

    // ==================== segment_cost ====================

    #[test]
    fn segment_cost_l1() {
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(
            segment_cost(&segment, CostFunction::L1),
            6.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn segment_cost_l2() {
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(
            segment_cost(&segment, CostFunction::L2),
            10.0,
            epsilon = 1e-10
        );
    }

    // ==================== total_cost ====================

    #[test]
    fn total_cost_no_changepoints() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cost = total_cost(&series, &[], CostFunction::L2);
        assert_relative_eq!(cost, l2_cost(&series), epsilon = 1e-10);
    }

    #[test]
    fn total_cost_one_changepoint() {
        let series = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let cost_no_cp = total_cost(&series, &[], CostFunction::L2);
        let cost_with_cp = total_cost(&series, &[3], CostFunction::L2);

        // With changepoint at 3, each segment is constant -> cost = 0
        assert_relative_eq!(cost_with_cp, 0.0, epsilon = 1e-10);
        // Without changepoint, cost should be higher
        assert!(cost_no_cp > cost_with_cp);
    }

    #[test]
    fn total_cost_multiple_changepoints() {
        let series = vec![1.0, 1.0, 5.0, 5.0, 9.0, 9.0];
        let cost = total_cost(&series, &[2, 4], CostFunction::L2);
        // Each segment [1,1], [5,5], [9,9] is constant
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn total_cost_empty() {
        assert_relative_eq!(total_cost(&[], &[], CostFunction::L2), 0.0, epsilon = 1e-10);
    }

    // ==================== cost_function_default ====================

    #[test]
    fn cost_function_default_is_l2() {
        assert_eq!(CostFunction::default(), CostFunction::L2);
    }

    // ==================== linear_trend_cost ====================

    #[test]
    fn linear_trend_cost_empty() {
        assert_relative_eq!(linear_trend_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_trend_cost_single() {
        assert_relative_eq!(linear_trend_cost(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_trend_cost_perfect_line() {
        // Perfect linear data: y = 2x + 1
        let segment: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect();
        let cost = linear_trend_cost(&segment);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn linear_trend_cost_constant() {
        let segment = vec![5.0; 10];
        let cost = linear_trend_cost(&segment);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_trend_cost_with_noise() {
        // Linear with small noise
        let segment: Vec<f64> = (0..10)
            .map(|i| 2.0 * i as f64 + 1.0 + (i % 2) as f64 * 0.1)
            .collect();
        let cost = linear_trend_cost(&segment);
        assert!(cost > 0.0 && cost < 1.0); // Small residual
    }

    #[test]
    fn linear_trend_cost_less_than_l2_for_linear_data() {
        // Linear trend cost should be less than L2 for trending data
        let segment: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let lt_cost = linear_trend_cost(&segment);
        let l2_cost_val = l2_cost(&segment);
        assert!(lt_cost < l2_cost_val);
    }

    // ==================== mean_variance_cost ====================

    #[test]
    fn mean_variance_cost_empty() {
        assert_relative_eq!(mean_variance_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_variance_cost_single() {
        assert_relative_eq!(mean_variance_cost(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_variance_cost_constant() {
        let segment = vec![5.0; 10];
        assert_relative_eq!(mean_variance_cost(&segment), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_variance_cost_known() {
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // variance = 2.0
        // cost = 5 * (1 + ln(2))
        let expected = 5.0 * (1.0 + 2.0_f64.ln());
        assert_relative_eq!(mean_variance_cost(&segment), expected, epsilon = 1e-10);
    }

    #[test]
    fn mean_variance_cost_greater_than_normal_cost() {
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mv_cost = mean_variance_cost(&segment);
        let n_cost = normal_cost(&segment);
        // MeanVariance should be higher due to +n term
        assert!(mv_cost > n_cost);
    }

    // ==================== cusum_cost ====================

    #[test]
    fn cusum_cost_empty() {
        assert_relative_eq!(cusum_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cusum_cost_single() {
        assert_relative_eq!(cusum_cost(&[5.0]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cusum_cost_constant() {
        let segment = vec![5.0; 20];
        let cost = cusum_cost(&segment);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn cusum_cost_balanced() {
        // Alternating around mean - low CUSUM
        let segment: Vec<f64> = (0..20)
            .map(|i| 5.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let cost = cusum_cost(&segment);
        assert!(cost <= 1.0 + 1e-10); // Should be at most 1.0 (max deviation)
    }

    #[test]
    fn cusum_cost_sustained_shift() {
        // First half below mean, second half above - high CUSUM
        let mut segment: Vec<f64> = vec![0.0; 10];
        segment.extend(vec![10.0; 10]);
        let cost = cusum_cost(&segment);
        assert!(cost > 10.0); // Should be high due to sustained deviation
    }

    // ==================== periodicity_cost ====================

    #[test]
    fn periodicity_cost_empty() {
        assert_relative_eq!(periodicity_cost(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn periodicity_cost_short() {
        let segment = vec![1.0, 2.0, 3.0];
        // Should fall back to L2
        let cost = periodicity_cost(&segment);
        assert_relative_eq!(cost, l2_cost(&segment), epsilon = 1e-10);
    }

    #[test]
    fn periodicity_cost_constant() {
        let segment = vec![5.0; 64];
        assert_relative_eq!(periodicity_cost(&segment), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn periodicity_cost_pure_sine() {
        // Perfect periodicity should have low cost
        let segment: Vec<f64> = (0..64)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin())
            .collect();
        let cost = periodicity_cost(&segment);

        // Compare to L2 cost (no periodic model)
        let l2 = l2_cost(&segment);
        assert!(cost < l2); // Periodic cost should be lower
    }

    #[test]
    fn periodicity_cost_white_noise() {
        // Pseudo-random data should have higher cost
        let segment: Vec<f64> = (0..64).map(|i| ((i * 17 + 3) % 11) as f64 - 5.0).collect();
        let cost = periodicity_cost(&segment);
        // Cost should be positive for non-periodic data
        assert!(cost > 0.0);
    }

    // ==================== segment_cost with new functions ====================

    #[test]
    fn segment_cost_linear_trend() {
        let segment: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let cost = segment_cost(&segment, CostFunction::LinearTrend);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn segment_cost_mean_variance() {
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cost = segment_cost(&segment, CostFunction::MeanVariance);
        let expected = 5.0 * (1.0 + 2.0_f64.ln());
        assert_relative_eq!(cost, expected, epsilon = 1e-10);
    }

    #[test]
    fn segment_cost_cusum() {
        let segment = vec![5.0; 10];
        let cost = segment_cost(&segment, CostFunction::Cusum);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn segment_cost_periodicity() {
        let segment = vec![5.0; 64];
        let cost = segment_cost(&segment, CostFunction::Periodicity);
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
    }
}
