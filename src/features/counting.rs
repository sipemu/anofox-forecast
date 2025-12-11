//! Counting-based features for time series.
//!
//! Provides features based on counting specific patterns or values.

use super::basic::mean;

/// Returns the count of values strictly greater than the threshold.
pub fn count_above(series: &[f64], threshold: f64) -> usize {
    series.iter().filter(|&&x| x > threshold).count()
}

/// Returns the count of values strictly less than the threshold.
pub fn count_below(series: &[f64], threshold: f64) -> usize {
    series.iter().filter(|&&x| x < threshold).count()
}

/// Returns the count of values strictly greater than the mean.
pub fn count_above_mean(series: &[f64]) -> usize {
    if series.is_empty() {
        return 0;
    }
    let m = mean(series);
    series.iter().filter(|&&x| x > m).count()
}

/// Returns the count of values strictly less than the mean.
pub fn count_below_mean(series: &[f64]) -> usize {
    if series.is_empty() {
        return 0;
    }
    let m = mean(series);
    series.iter().filter(|&&x| x < m).count()
}

/// Returns the number of peaks in the time series.
///
/// A peak is a value that is higher than its n neighbors on both sides.
///
/// # Arguments
/// * `series` - Input time series
/// * `support` - Number of neighbors to consider on each side
pub fn number_peaks(series: &[f64], support: usize) -> usize {
    if series.len() < 2 * support + 1 || support == 0 {
        return 0;
    }

    let mut count = 0;
    for i in support..(series.len() - support) {
        let is_peak = (1..=support).all(|j| series[i] > series[i - j] && series[i] > series[i + j]);
        if is_peak {
            count += 1;
        }
    }
    count
}

/// Returns the number of times the series crosses the value m.
///
/// A crossing occurs when consecutive values are on opposite sides of m.
pub fn number_crossing_m(series: &[f64], m: f64) -> usize {
    if series.len() < 2 {
        return 0;
    }

    series
        .windows(2)
        .filter(|w| (w[0] <= m && w[1] > m) || (w[0] > m && w[1] <= m))
        .count()
}

/// Returns the longest consecutive run of values above the mean.
pub fn longest_strike_above_mean(series: &[f64]) -> usize {
    if series.is_empty() {
        return 0;
    }
    let m = mean(series);
    longest_strike(series, |x| x > m)
}

/// Returns the longest consecutive run of values below the mean.
pub fn longest_strike_below_mean(series: &[f64]) -> usize {
    if series.is_empty() {
        return 0;
    }
    let m = mean(series);
    longest_strike(series, |x| x < m)
}

/// Helper: find longest consecutive run satisfying a predicate.
fn longest_strike<F>(series: &[f64], predicate: F) -> usize
where
    F: Fn(f64) -> bool,
{
    let mut max_strike = 0;
    let mut current_strike = 0;

    for &x in series {
        if predicate(x) {
            current_strike += 1;
            max_strike = max_strike.max(current_strike);
        } else {
            current_strike = 0;
        }
    }
    max_strike
}

/// Returns the relative position of the first occurrence of the maximum.
///
/// Returns a value in [0, 1] representing the position as a fraction of the length.
pub fn first_location_of_maximum(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let max_val = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pos = series.iter().position(|&x| x == max_val).unwrap_or(0);
    pos as f64 / series.len() as f64
}

/// Returns the relative position of the first occurrence of the minimum.
pub fn first_location_of_minimum(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    let pos = series.iter().position(|&x| x == min_val).unwrap_or(0);
    pos as f64 / series.len() as f64
}

/// Returns the relative position of the last occurrence of the maximum.
pub fn last_location_of_maximum(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let max_val = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pos = series.iter().rposition(|&x| x == max_val).unwrap_or(0);
    pos as f64 / series.len() as f64
}

/// Returns the relative position of the last occurrence of the minimum.
pub fn last_location_of_minimum(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    let pos = series.iter().rposition(|&x| x == min_val).unwrap_or(0);
    pos as f64 / series.len() as f64
}

/// Returns whether the series has any duplicate values.
pub fn has_duplicate(series: &[f64]) -> bool {
    if series.len() < 2 {
        return false;
    }
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.windows(2).any(|w| (w[0] - w[1]).abs() < 1e-10)
}

/// Returns whether the maximum value appears more than once.
pub fn has_duplicate_max(series: &[f64]) -> bool {
    if series.len() < 2 {
        return false;
    }
    let max_val = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    series
        .iter()
        .filter(|&&x| (x - max_val).abs() < 1e-10)
        .count()
        > 1
}

/// Returns whether the minimum value appears more than once.
pub fn has_duplicate_min(series: &[f64]) -> bool {
    if series.len() < 2 {
        return false;
    }
    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    series
        .iter()
        .filter(|&&x| (x - min_val).abs() < 1e-10)
        .count()
        > 1
}

/// Returns the relative index where q% of the total mass is reached.
///
/// The mass is computed as the cumulative sum of absolute values.
///
/// # Arguments
/// * `series` - Input time series
/// * `q` - Quantile (0.0 to 1.0)
pub fn index_mass_quantile(series: &[f64], q: f64) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);

    let abs_values: Vec<f64> = series.iter().map(|x| x.abs()).collect();
    let total_mass: f64 = abs_values.iter().sum();

    if total_mass < 1e-10 {
        return 0.0;
    }

    let target = q * total_mass;
    let mut cumsum = 0.0;

    for (i, &v) in abs_values.iter().enumerate() {
        cumsum += v;
        if cumsum >= target {
            return (i + 1) as f64 / series.len() as f64;
        }
    }

    1.0
}

/// Returns the count of a specific value in the series.
///
/// Uses approximate equality with epsilon = 1e-10.
pub fn value_count(series: &[f64], value: f64) -> usize {
    series
        .iter()
        .filter(|&&x| (x - value).abs() < 1e-10)
        .count()
}

/// Returns the count of values within the specified range [min, max].
pub fn range_count(series: &[f64], min: f64, max: f64) -> usize {
    series.iter().filter(|&&x| x >= min && x <= max).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== count_above / count_below ====================

    #[test]
    fn count_above_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(count_above(&series, 3.0), 2); // 4, 5
        assert_eq!(count_above(&series, 0.0), 5);
        assert_eq!(count_above(&series, 10.0), 0);
    }

    #[test]
    fn count_below_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(count_below(&series, 3.0), 2); // 1, 2
        assert_eq!(count_below(&series, 10.0), 5);
        assert_eq!(count_below(&series, 0.0), 0);
    }

    #[test]
    fn count_above_below_empty() {
        assert_eq!(count_above(&[], 0.0), 0);
        assert_eq!(count_below(&[], 0.0), 0);
    }

    // ==================== count_above_mean / count_below_mean ====================

    #[test]
    fn count_above_mean_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean = 3
        assert_eq!(count_above_mean(&series), 2); // 4, 5
    }

    #[test]
    fn count_below_mean_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean = 3
        assert_eq!(count_below_mean(&series), 2); // 1, 2
    }

    #[test]
    fn count_above_below_mean_empty() {
        assert_eq!(count_above_mean(&[]), 0);
        assert_eq!(count_below_mean(&[]), 0);
    }

    #[test]
    fn count_above_below_mean_constant() {
        let series = vec![5.0; 10];
        assert_eq!(count_above_mean(&series), 0);
        assert_eq!(count_below_mean(&series), 0);
    }

    // ==================== number_peaks ====================

    #[test]
    fn number_peaks_basic() {
        // Peak at position 2 (value 5)
        let series = vec![1.0, 2.0, 5.0, 2.0, 1.0];
        assert_eq!(number_peaks(&series, 1), 1);
        assert_eq!(number_peaks(&series, 2), 1);
    }

    #[test]
    fn number_peaks_multiple() {
        let series = vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0];
        assert_eq!(number_peaks(&series, 1), 3); // peaks at positions 1, 3, 5
    }

    #[test]
    fn number_peaks_plateau() {
        // Plateau is not a peak (requires strict inequality)
        let series = vec![1.0, 3.0, 3.0, 3.0, 1.0];
        assert_eq!(number_peaks(&series, 1), 0);
    }

    #[test]
    fn number_peaks_short() {
        assert_eq!(number_peaks(&[], 1), 0);
        assert_eq!(number_peaks(&[1.0], 1), 0);
        assert_eq!(number_peaks(&[1.0, 2.0], 1), 0);
    }

    #[test]
    fn number_peaks_zero_support() {
        let series = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        assert_eq!(number_peaks(&series, 0), 0);
    }

    // ==================== number_crossing_m ====================

    #[test]
    fn number_crossing_m_basic() {
        let series = vec![-1.0, 1.0, -1.0, 1.0, -1.0];
        assert_eq!(number_crossing_m(&series, 0.0), 4);
    }

    #[test]
    fn number_crossing_m_no_crossing() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(number_crossing_m(&series, 0.0), 0);
    }

    #[test]
    fn number_crossing_m_short() {
        assert_eq!(number_crossing_m(&[], 0.0), 0);
        assert_eq!(number_crossing_m(&[1.0], 0.0), 0);
    }

    // ==================== longest_strike ====================

    #[test]
    fn longest_strike_above_mean_basic() {
        let series = vec![1.0, 5.0, 5.0, 5.0, 1.0, 5.0, 1.0]; // mean ≈ 3.29
                                                              // Above mean: positions 1, 2, 3 (strike of 3) and position 5 (strike of 1)
        assert_eq!(longest_strike_above_mean(&series), 3);
    }

    #[test]
    fn longest_strike_below_mean_basic() {
        let series = vec![5.0, 1.0, 1.0, 5.0, 5.0]; // mean = 3.4
                                                    // Below mean: positions 1, 2 (strike of 2)
        assert_eq!(longest_strike_below_mean(&series), 2);
    }

    #[test]
    fn longest_strike_empty() {
        assert_eq!(longest_strike_above_mean(&[]), 0);
        assert_eq!(longest_strike_below_mean(&[]), 0);
    }

    #[test]
    fn longest_strike_constant() {
        let series = vec![5.0; 10];
        assert_eq!(longest_strike_above_mean(&series), 0);
        assert_eq!(longest_strike_below_mean(&series), 0);
    }

    // ==================== first/last_location_of_maximum/minimum ====================

    #[test]
    fn first_location_of_maximum_basic() {
        let series = vec![1.0, 5.0, 3.0, 5.0, 2.0]; // max at position 1
        assert_relative_eq!(first_location_of_maximum(&series), 0.2, epsilon = 1e-10);
    }

    #[test]
    fn last_location_of_maximum_basic() {
        let series = vec![1.0, 5.0, 3.0, 5.0, 2.0]; // last max at position 3
        assert_relative_eq!(last_location_of_maximum(&series), 0.6, epsilon = 1e-10);
    }

    #[test]
    fn first_location_of_minimum_basic() {
        let series = vec![5.0, 1.0, 3.0, 1.0, 4.0]; // min at position 1
        assert_relative_eq!(first_location_of_minimum(&series), 0.2, epsilon = 1e-10);
    }

    #[test]
    fn last_location_of_minimum_basic() {
        let series = vec![5.0, 1.0, 3.0, 1.0, 4.0]; // last min at position 3
        assert_relative_eq!(last_location_of_minimum(&series), 0.6, epsilon = 1e-10);
    }

    #[test]
    fn location_empty() {
        assert!(first_location_of_maximum(&[]).is_nan());
        assert!(last_location_of_maximum(&[]).is_nan());
        assert!(first_location_of_minimum(&[]).is_nan());
        assert!(last_location_of_minimum(&[]).is_nan());
    }

    #[test]
    fn location_single() {
        assert_relative_eq!(first_location_of_maximum(&[5.0]), 0.0, epsilon = 1e-10);
        assert_relative_eq!(last_location_of_maximum(&[5.0]), 0.0, epsilon = 1e-10);
    }

    // ==================== has_duplicate ====================

    #[test]
    fn has_duplicate_true() {
        let series = vec![1.0, 2.0, 3.0, 2.0, 4.0];
        assert!(has_duplicate(&series));
    }

    #[test]
    fn has_duplicate_false() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(!has_duplicate(&series));
    }

    #[test]
    fn has_duplicate_short() {
        assert!(!has_duplicate(&[]));
        assert!(!has_duplicate(&[1.0]));
    }

    // ==================== has_duplicate_max / has_duplicate_min ====================

    #[test]
    fn has_duplicate_max_true() {
        let series = vec![1.0, 5.0, 3.0, 5.0, 2.0];
        assert!(has_duplicate_max(&series));
    }

    #[test]
    fn has_duplicate_max_false() {
        let series = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        assert!(!has_duplicate_max(&series));
    }

    #[test]
    fn has_duplicate_min_true() {
        let series = vec![5.0, 1.0, 3.0, 1.0, 2.0];
        assert!(has_duplicate_min(&series));
    }

    #[test]
    fn has_duplicate_min_false() {
        let series = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert!(!has_duplicate_min(&series));
    }

    #[test]
    fn has_duplicate_max_min_short() {
        assert!(!has_duplicate_max(&[]));
        assert!(!has_duplicate_max(&[1.0]));
        assert!(!has_duplicate_min(&[]));
        assert!(!has_duplicate_min(&[1.0]));
    }

    // ==================== index_mass_quantile ====================

    #[test]
    fn index_mass_quantile_uniform() {
        let series = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        // At q=0.5, we need 2.5 mass, reached at index 2 (3/5 = 0.6)
        assert_relative_eq!(index_mass_quantile(&series, 0.5), 0.6, epsilon = 1e-10);
    }

    #[test]
    fn index_mass_quantile_boundaries() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(index_mass_quantile(&series, 0.0), 0.2, epsilon = 1e-10);
        assert_relative_eq!(index_mass_quantile(&series, 1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn index_mass_quantile_empty() {
        assert!(index_mass_quantile(&[], 0.5).is_nan());
    }

    #[test]
    fn index_mass_quantile_zeros() {
        let series = vec![0.0; 5];
        assert_relative_eq!(index_mass_quantile(&series, 0.5), 0.0, epsilon = 1e-10);
    }

    // ==================== value_count ====================

    #[test]
    fn value_count_basic() {
        let series = vec![1.0, 2.0, 1.0, 3.0, 1.0];
        assert_eq!(value_count(&series, 1.0), 3);
        assert_eq!(value_count(&series, 2.0), 1);
        assert_eq!(value_count(&series, 5.0), 0);
    }

    #[test]
    fn value_count_empty() {
        assert_eq!(value_count(&[], 1.0), 0);
    }

    // ==================== range_count ====================

    #[test]
    fn range_count_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(range_count(&series, 2.0, 4.0), 3); // 2, 3, 4
        assert_eq!(range_count(&series, 0.0, 10.0), 5);
        assert_eq!(range_count(&series, 10.0, 20.0), 0);
    }

    #[test]
    fn range_count_empty() {
        assert_eq!(range_count(&[], 0.0, 10.0), 0);
    }

    #[test]
    fn range_count_boundaries() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(range_count(&series, 2.0, 2.0), 1); // Only exact 2.0
    }
}
