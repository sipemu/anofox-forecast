//! Rolling and expanding window functions.
//!
//! Provides windowed statistics and transformations.

/// Compute rolling mean (moving average).
///
/// # Arguments
/// * `series` - Input time series
/// * `window` - Window size
/// * `center` - If true, center the window (default: false, trailing window)
pub fn rolling_mean(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    if series.is_empty() || window == 0 {
        return vec![f64::NAN; series.len()];
    }

    let n = series.len();
    let mut result = vec![f64::NAN; n];
    let offset = if center { window / 2 } else { window - 1 };

    for i in 0..n {
        let (start, end) = if center {
            let half = window / 2;
            let start = i.saturating_sub(half);
            let end = (i + window - half).min(n);
            (start, end)
        } else {
            if i + 1 < window {
                continue;
            }
            (i + 1 - window, i + 1)
        };

        if end > start {
            let sum: f64 = series[start..end].iter().sum();
            result[i] = sum / (end - start) as f64;
        }
    }

    result
}

/// Compute rolling standard deviation.
pub fn rolling_std(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    rolling_var(series, window, center)
        .iter()
        .map(|v| v.sqrt())
        .collect()
}

/// Compute rolling variance.
pub fn rolling_var(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    if series.is_empty() || window < 2 {
        return vec![f64::NAN; series.len()];
    }

    let n = series.len();
    let mut result = vec![f64::NAN; n];

    for i in 0..n {
        let (start, end) = if center {
            let half = window / 2;
            let start = i.saturating_sub(half);
            let end = (i + window - half).min(n);
            (start, end)
        } else {
            if i + 1 < window {
                continue;
            }
            (i + 1 - window, i + 1)
        };

        let segment = &series[start..end];
        if segment.len() >= 2 {
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            let var = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (segment.len() - 1) as f64;
            result[i] = var;
        }
    }

    result
}

/// Compute rolling minimum.
pub fn rolling_min(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    rolling_apply(series, window, center, |s| {
        s.iter().copied().fold(f64::INFINITY, f64::min)
    })
}

/// Compute rolling maximum.
pub fn rolling_max(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    rolling_apply(series, window, center, |s| {
        s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    })
}

/// Compute rolling sum.
pub fn rolling_sum(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    rolling_apply(series, window, center, |s| s.iter().sum())
}

/// Compute rolling median.
pub fn rolling_median(series: &[f64], window: usize, center: bool) -> Vec<f64> {
    rolling_apply(series, window, center, |s| {
        let mut sorted = s.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    })
}

/// Generic rolling window application.
fn rolling_apply<F>(series: &[f64], window: usize, center: bool, f: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if series.is_empty() || window == 0 {
        return vec![f64::NAN; series.len()];
    }

    let n = series.len();
    let mut result = vec![f64::NAN; n];

    for i in 0..n {
        let (start, end) = if center {
            let half = window / 2;
            let start = i.saturating_sub(half);
            let end = (i + window - half).min(n);
            (start, end)
        } else {
            if i + 1 < window {
                continue;
            }
            (i + 1 - window, i + 1)
        };

        if end > start {
            result[i] = f(&series[start..end]);
        }
    }

    result
}

/// Compute expanding mean (cumulative mean).
pub fn expanding_mean(series: &[f64]) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(series.len());
    let mut sum = 0.0;

    for (i, &x) in series.iter().enumerate() {
        sum += x;
        result.push(sum / (i + 1) as f64);
    }

    result
}

/// Compute expanding sum (cumulative sum).
pub fn expanding_sum(series: &[f64]) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(series.len());
    let mut sum = 0.0;

    for &x in series {
        sum += x;
        result.push(sum);
    }

    result
}

/// Compute expanding maximum (cumulative max).
pub fn expanding_max(series: &[f64]) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(series.len());
    let mut max = f64::NEG_INFINITY;

    for &x in series {
        max = max.max(x);
        result.push(max);
    }

    result
}

/// Compute expanding minimum (cumulative min).
pub fn expanding_min(series: &[f64]) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(series.len());
    let mut min = f64::INFINITY;

    for &x in series {
        min = min.min(x);
        result.push(min);
    }

    result
}

/// Compute exponentially weighted moving average (EWMA).
///
/// # Arguments
/// * `series` - Input time series
/// * `alpha` - Smoothing factor (0 < alpha <= 1)
///   - Higher alpha = more weight on recent values
///   - alpha = 2/(span+1) for span-based specification
pub fn ewm_mean(series: &[f64], alpha: f64) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let alpha = alpha.clamp(0.0, 1.0);
    let mut result = Vec::with_capacity(series.len());
    let mut ewm = series[0];

    result.push(ewm);

    for &x in series.iter().skip(1) {
        ewm = alpha * x + (1.0 - alpha) * ewm;
        result.push(ewm);
    }

    result
}

/// Compute exponentially weighted moving standard deviation.
pub fn ewm_std(series: &[f64], alpha: f64) -> Vec<f64> {
    ewm_var(series, alpha).iter().map(|v| v.sqrt()).collect()
}

/// Compute exponentially weighted moving variance.
pub fn ewm_var(series: &[f64], alpha: f64) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }

    let alpha = alpha.clamp(0.0, 1.0);
    let ewm = ewm_mean(series, alpha);

    let mut result = Vec::with_capacity(series.len());
    let mut ewm_sq = series[0] * series[0];

    result.push(0.0); // First variance is 0

    for (i, &x) in series.iter().enumerate().skip(1) {
        ewm_sq = alpha * x * x + (1.0 - alpha) * ewm_sq;
        let var = ewm_sq - ewm[i] * ewm[i];
        result.push(var.max(0.0));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== rolling_mean ====================

    #[test]
    fn rolling_mean_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&series, 3, false);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10); // (1+2+3)/3
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10); // (2+3+4)/3
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10); // (3+4+5)/3
    }

    #[test]
    fn rolling_mean_window_1() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&series, 1, false);

        for (i, &x) in series.iter().enumerate() {
            assert_relative_eq!(result[i], x, epsilon = 1e-10);
        }
    }

    #[test]
    fn rolling_mean_empty() {
        let result = rolling_mean(&[], 3, false);
        assert!(result.is_empty());
    }

    #[test]
    fn rolling_mean_centered() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&series, 3, true);

        // Centered window: [1,2,3], [2,3,4], [3,4,5] at indices 1, 2, 3
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 4.0, epsilon = 1e-10);
    }

    // ==================== rolling_std ====================

    #[test]
    fn rolling_std_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_std(&series, 3, false);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // std of [1,2,3] = 1.0
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn rolling_std_constant() {
        let series = vec![5.0; 10];
        let result = rolling_std(&series, 3, false);

        for i in 2..10 {
            assert_relative_eq!(result[i], 0.0, epsilon = 1e-10);
        }
    }

    // ==================== rolling_min / rolling_max ====================

    #[test]
    fn rolling_min_basic() {
        let series = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = rolling_min(&series, 3, false);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10); // min(3,1,4)
        assert_relative_eq!(result[3], 1.0, epsilon = 1e-10); // min(1,4,1)
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-10); // min(4,1,5)
    }

    #[test]
    fn rolling_max_basic() {
        let series = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = rolling_max(&series, 3, false);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 4.0, epsilon = 1e-10); // max(3,1,4)
        assert_relative_eq!(result[3], 4.0, epsilon = 1e-10); // max(1,4,1)
        assert_relative_eq!(result[4], 5.0, epsilon = 1e-10); // max(4,1,5)
    }

    // ==================== rolling_sum ====================

    #[test]
    fn rolling_sum_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_sum(&series, 3, false);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 6.0, epsilon = 1e-10);  // 1+2+3
        assert_relative_eq!(result[3], 9.0, epsilon = 1e-10);  // 2+3+4
        assert_relative_eq!(result[4], 12.0, epsilon = 1e-10); // 3+4+5
    }

    // ==================== rolling_median ====================

    #[test]
    fn rolling_median_basic() {
        let series = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let result = rolling_median(&series, 3, false);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10); // median(1,5,2)
        assert_relative_eq!(result[3], 5.0, epsilon = 1e-10); // median(5,2,8)
        assert_relative_eq!(result[4], 3.0, epsilon = 1e-10); // median(2,8,3)
    }

    // ==================== expanding functions ====================

    #[test]
    fn expanding_mean_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = expanding_mean(&series);

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.5, epsilon = 1e-10);
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 2.5, epsilon = 1e-10);
        assert_relative_eq!(result[4], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn expanding_sum_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = expanding_sum(&series);

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 10.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 15.0, epsilon = 1e-10);
    }

    #[test]
    fn expanding_max_basic() {
        let series = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let result = expanding_max(&series);

        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn expanding_min_basic() {
        let series = vec![5.0, 3.0, 4.0, 1.0, 2.0];
        let result = expanding_min(&series);

        assert_relative_eq!(result[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn expanding_empty() {
        assert!(expanding_mean(&[]).is_empty());
        assert!(expanding_sum(&[]).is_empty());
        assert!(expanding_max(&[]).is_empty());
        assert!(expanding_min(&[]).is_empty());
    }

    // ==================== ewm functions ====================

    #[test]
    fn ewm_mean_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewm_mean(&series, 0.5);

        // First value is just the first observation
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);

        // Each subsequent value: alpha * x + (1-alpha) * prev
        // ewm[1] = 0.5 * 2 + 0.5 * 1 = 1.5
        assert_relative_eq!(result[1], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn ewm_mean_alpha_1() {
        // Alpha = 1 means just the current value
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewm_mean(&series, 1.0);

        for (i, &x) in series.iter().enumerate() {
            assert_relative_eq!(result[i], x, epsilon = 1e-10);
        }
    }

    #[test]
    fn ewm_mean_alpha_0() {
        // Alpha = 0 means just the first value repeated
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewm_mean(&series, 0.0);

        for &r in &result {
            assert_relative_eq!(r, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn ewm_mean_empty() {
        assert!(ewm_mean(&[], 0.5).is_empty());
    }

    #[test]
    fn ewm_std_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ewm_std(&series, 0.5);

        // First value has 0 std
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);

        // Subsequent values should be non-negative
        for &r in result.iter().skip(1) {
            assert!(r >= 0.0);
        }
    }
}
