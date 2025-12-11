//! Cost functions for changepoint detection.
//!
//! Cost functions evaluate the "cost" of fitting a model to a segment of data.
//! Lower cost indicates a better fit.

/// Cost function type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostFunction {
    /// L1 cost: sum of absolute deviations from median
    L1,
    /// L2 cost: sum of squared deviations from mean (normal likelihood)
    L2,
    /// Normal likelihood cost (equivalent to L2 with variance estimation)
    Normal,
    /// Poisson cost for count data
    Poisson,
}

impl Default for CostFunction {
    fn default() -> Self {
        CostFunction::L2
    }
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
    if n % 2 == 0 {
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
        assert_relative_eq!(segment_cost(&segment, CostFunction::L1), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn segment_cost_l2() {
        let segment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(segment_cost(&segment, CostFunction::L2), 10.0, epsilon = 1e-10);
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
}
