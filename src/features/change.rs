//! Change-based features for time series.
//!
//! Provides features related to changes and reoccurring patterns.

/// Computes change quantiles between corridors.
///
/// Aggregates changes where consecutive values are within specified quantile corridors.
///
/// # Arguments
/// * `series` - Input time series
/// * `q_low` - Lower quantile bound (0.0 to 1.0)
/// * `q_high` - Upper quantile bound (0.0 to 1.0)
/// * `is_abs` - Whether to use absolute changes
/// * `agg_func` - Aggregation: "mean", "var", "std", "median"
pub fn change_quantiles(
    series: &[f64],
    q_low: f64,
    q_high: f64,
    is_abs: bool,
    agg_func: &str,
) -> f64 {
    if series.len() < 2 {
        return f64::NAN;
    }

    let low = quantile(series, q_low);
    let high = quantile(series, q_high);

    // Collect changes where both values are in the corridor
    let mut changes: Vec<f64> = Vec::new();

    for w in series.windows(2) {
        if w[0] >= low && w[0] <= high && w[1] >= low && w[1] <= high {
            let change = if is_abs {
                (w[1] - w[0]).abs()
            } else {
                w[1] - w[0]
            };
            changes.push(change);
        }
    }

    if changes.is_empty() {
        return f64::NAN;
    }

    aggregate(&changes, agg_func)
}

/// Computes the energy ratio by chunks.
///
/// Divides the series into chunks and returns the ratio of energy in chunk i
/// to the total energy.
///
/// # Arguments
/// * `series` - Input time series
/// * `n_chunks` - Number of chunks
/// * `chunk_index` - Which chunk's ratio to return (0-indexed)
pub fn energy_ratio_by_chunks(series: &[f64], n_chunks: usize, chunk_index: usize) -> f64 {
    if series.is_empty() || n_chunks == 0 || chunk_index >= n_chunks {
        return f64::NAN;
    }

    let total_energy: f64 = series.iter().map(|x| x * x).sum();
    if total_energy < 1e-10 {
        return 0.0;
    }

    let chunk_size = series.len().div_ceil(n_chunks);
    let start = chunk_index * chunk_size;
    let end = ((chunk_index + 1) * chunk_size).min(series.len());

    if start >= series.len() {
        return 0.0;
    }

    let chunk_energy: f64 = series[start..end].iter().map(|x| x * x).sum();
    chunk_energy / total_energy
}

/// Returns the percentage of data points that appear more than once.
///
/// Counts unique values that occur more than once, divided by total data points.
pub fn percentage_of_reoccurring_datapoints_to_all_datapoints(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }

    let mut counts = std::collections::HashMap::new();
    for &x in series {
        // Discretize to handle floating point comparison
        let key = discretize(x);
        *counts.entry(key).or_insert(0) += 1;
    }

    let reoccurring_count: usize = counts.values().filter(|&&c| c > 1).copied().sum();

    reoccurring_count as f64 / series.len() as f64
}

/// Returns the percentage of unique values that appear more than once.
///
/// Counts unique values that occur more than once, divided by total unique values.
pub fn percentage_of_reoccurring_values_to_all_values(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }

    let mut counts = std::collections::HashMap::new();
    for &x in series {
        let key = discretize(x);
        *counts.entry(key).or_insert(0) += 1;
    }

    let total_unique = counts.len();
    if total_unique == 0 {
        return 0.0;
    }

    let reoccurring_unique = counts.values().filter(|&&c| c > 1).count();

    reoccurring_unique as f64 / total_unique as f64
}

/// Returns the ratio of unique values to the series length.
pub fn ratio_value_number_to_time_series_length(series: &[f64]) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }

    let mut unique = std::collections::HashSet::new();
    for &x in series {
        unique.insert(discretize(x));
    }

    unique.len() as f64 / series.len() as f64
}

/// Returns the sum of all data points that appear more than once.
pub fn sum_of_reoccurring_data_points(series: &[f64]) -> f64 {
    let mut counts = std::collections::HashMap::new();
    let mut sums = std::collections::HashMap::new();

    for &x in series {
        let key = discretize(x);
        *counts.entry(key).or_insert(0) += 1;
        *sums.entry(key).or_insert(0.0) += x;
    }

    counts
        .iter()
        .filter(|(_, &c)| c > 1)
        .map(|(k, _)| sums.get(k).unwrap_or(&0.0))
        .sum()
}

/// Returns the sum of unique values that appear more than once.
///
/// Each unique value is counted once, not by its frequency.
pub fn sum_of_reoccurring_values(series: &[f64]) -> f64 {
    let mut counts = std::collections::HashMap::new();
    let mut first_occurrence = std::collections::HashMap::new();

    for &x in series {
        let key = discretize(x);
        let count = counts.entry(key).or_insert(0);
        if *count == 0 {
            first_occurrence.insert(key, x);
        }
        *count += 1;
    }

    counts
        .iter()
        .filter(|(_, &c)| c > 1)
        .map(|(k, _)| first_occurrence.get(k).unwrap_or(&0.0))
        .sum()
}

/// Helper: discretize float for hashing
fn discretize(x: f64) -> i64 {
    (x * 1e10).round() as i64
}

/// Helper: compute quantile
fn quantile(series: &[f64], q: f64) -> f64 {
    if series.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let pos = q * (n - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let frac = pos - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Helper: aggregate values
fn aggregate(values: &[f64], func: &str) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    match func {
        "mean" => values.iter().sum::<f64>() / values.len() as f64,
        "var" => {
            if values.len() < 2 {
                return f64::NAN;
            }
            let m = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
        }
        "std" => {
            let var = aggregate(values, "var");
            var.sqrt()
        }
        "median" => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            if n.is_multiple_of(2) {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            }
        }
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== change_quantiles ====================

    #[test]
    fn change_quantiles_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = change_quantiles(&series, 0.0, 1.0, false, "mean");
        // All changes are 1.0
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn change_quantiles_abs() {
        let series = vec![5.0, 3.0, 7.0, 2.0, 8.0];
        let result = change_quantiles(&series, 0.0, 1.0, true, "mean");
        // Absolute changes: 2, 4, 5, 6 -> mean = 4.25
        assert!(!result.is_nan());
    }

    #[test]
    fn change_quantiles_narrow_corridor() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Only values between q25 and q75
        let result = change_quantiles(&series, 0.25, 0.75, false, "mean");
        assert!(!result.is_nan());
    }

    #[test]
    fn change_quantiles_empty() {
        assert!(change_quantiles(&[], 0.0, 1.0, false, "mean").is_nan());
        assert!(change_quantiles(&[1.0], 0.0, 1.0, false, "mean").is_nan());
    }

    // ==================== energy_ratio_by_chunks ====================

    #[test]
    fn energy_ratio_by_chunks_uniform() {
        let series = vec![1.0; 10];
        // Each chunk should have 20% of energy for 5 chunks
        let ratio = energy_ratio_by_chunks(&series, 5, 0);
        assert_relative_eq!(ratio, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn energy_ratio_by_chunks_concentrated() {
        let mut series = vec![0.0; 10];
        series[0] = 10.0; // All energy in first element
        let ratio0 = energy_ratio_by_chunks(&series, 5, 0);
        let ratio1 = energy_ratio_by_chunks(&series, 5, 1);
        assert_relative_eq!(ratio0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(ratio1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn energy_ratio_by_chunks_invalid() {
        assert!(energy_ratio_by_chunks(&[], 5, 0).is_nan());
        assert!(energy_ratio_by_chunks(&[1.0, 2.0], 0, 0).is_nan());
        assert!(energy_ratio_by_chunks(&[1.0, 2.0], 2, 5).is_nan());
    }

    // ==================== percentage_of_reoccurring_datapoints ====================

    #[test]
    fn percentage_reoccurring_datapoints_all_unique() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pct = percentage_of_reoccurring_datapoints_to_all_datapoints(&series);
        assert_relative_eq!(pct, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn percentage_reoccurring_datapoints_all_same() {
        let series = vec![5.0; 10];
        let pct = percentage_of_reoccurring_datapoints_to_all_datapoints(&series);
        assert_relative_eq!(pct, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn percentage_reoccurring_datapoints_mixed() {
        let series = vec![1.0, 2.0, 1.0, 3.0]; // 1.0 appears twice
        let pct = percentage_of_reoccurring_datapoints_to_all_datapoints(&series);
        // 2 out of 4 are reoccurring
        assert_relative_eq!(pct, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn percentage_reoccurring_datapoints_empty() {
        assert!(percentage_of_reoccurring_datapoints_to_all_datapoints(&[]).is_nan());
    }

    // ==================== percentage_of_reoccurring_values ====================

    #[test]
    fn percentage_reoccurring_values_all_unique() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pct = percentage_of_reoccurring_values_to_all_values(&series);
        assert_relative_eq!(pct, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn percentage_reoccurring_values_all_same() {
        let series = vec![5.0; 10];
        let pct = percentage_of_reoccurring_values_to_all_values(&series);
        // 1 unique value, and it occurs more than once
        assert_relative_eq!(pct, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn percentage_reoccurring_values_mixed() {
        let series = vec![1.0, 2.0, 1.0, 3.0]; // 3 unique, 1 reoccurs
        let pct = percentage_of_reoccurring_values_to_all_values(&series);
        assert_relative_eq!(pct, 1.0 / 3.0, epsilon = 1e-10);
    }

    // ==================== ratio_value_number_to_time_series_length ====================

    #[test]
    fn ratio_unique_all_unique() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ratio = ratio_value_number_to_time_series_length(&series);
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn ratio_unique_all_same() {
        let series = vec![5.0; 10];
        let ratio = ratio_value_number_to_time_series_length(&series);
        assert_relative_eq!(ratio, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn ratio_unique_mixed() {
        let series = vec![1.0, 1.0, 2.0, 2.0, 3.0]; // 3 unique out of 5
        let ratio = ratio_value_number_to_time_series_length(&series);
        assert_relative_eq!(ratio, 0.6, epsilon = 1e-10);
    }

    #[test]
    fn ratio_unique_empty() {
        assert!(ratio_value_number_to_time_series_length(&[]).is_nan());
    }

    // ==================== sum_of_reoccurring_data_points ====================

    #[test]
    fn sum_reoccurring_datapoints_all_unique() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = sum_of_reoccurring_data_points(&series);
        assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn sum_reoccurring_datapoints_with_repeats() {
        let series = vec![1.0, 2.0, 1.0, 3.0]; // 1.0 appears twice
        let sum = sum_of_reoccurring_data_points(&series);
        // Sum of all 1.0s = 2.0
        assert_relative_eq!(sum, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn sum_reoccurring_datapoints_multiple_repeats() {
        let series = vec![1.0, 2.0, 1.0, 2.0]; // 1.0 and 2.0 each appear twice
        let sum = sum_of_reoccurring_data_points(&series);
        // 1 + 1 + 2 + 2 = 6
        assert_relative_eq!(sum, 6.0, epsilon = 1e-10);
    }

    // ==================== sum_of_reoccurring_values ====================

    #[test]
    fn sum_reoccurring_values_all_unique() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = sum_of_reoccurring_values(&series);
        assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn sum_reoccurring_values_with_repeats() {
        let series = vec![1.0, 2.0, 1.0, 3.0]; // 1.0 appears twice
        let sum = sum_of_reoccurring_values(&series);
        // Sum of unique reoccurring = 1.0
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn sum_reoccurring_values_multiple_repeats() {
        let series = vec![1.0, 2.0, 1.0, 2.0]; // 1.0 and 2.0 each appear twice
        let sum = sum_of_reoccurring_values(&series);
        // 1 + 2 = 3
        assert_relative_eq!(sum, 3.0, epsilon = 1e-10);
    }
}
