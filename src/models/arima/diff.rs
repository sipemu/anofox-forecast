//! Differencing utilities for ARIMA models.

/// Apply differencing to a time series.
///
/// # Arguments
/// * `series` - The input series
/// * `d` - Differencing order (number of times to difference)
///
/// # Returns
/// The differenced series.
pub fn difference(series: &[f64], d: usize) -> Vec<f64> {
    if d == 0 || series.is_empty() {
        return series.to_vec();
    }

    let mut result = series.to_vec();
    for _ in 0..d {
        if result.len() <= 1 {
            break;
        }
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

/// Apply seasonal differencing to a time series.
///
/// # Arguments
/// * `series` - The input series
/// * `d` - Seasonal differencing order
/// * `period` - Seasonal period
///
/// # Returns
/// The seasonally differenced series.
pub fn seasonal_difference(series: &[f64], d: usize, period: usize) -> Vec<f64> {
    if d == 0 || period == 0 || series.len() <= period {
        return series.to_vec();
    }

    let mut result = series.to_vec();
    for _ in 0..d {
        if result.len() <= period {
            break;
        }
        result = result
            .iter()
            .skip(period)
            .zip(result.iter())
            .map(|(curr, prev)| curr - prev)
            .collect();
    }
    result
}

/// Integrate (reverse differencing) a differenced series.
///
/// # Arguments
/// * `differenced` - The differenced series
/// * `original` - The original series (needed for initial values)
/// * `d` - Differencing order used
///
/// # Returns
/// The integrated series.
pub fn integrate(differenced: &[f64], original: &[f64], d: usize) -> Vec<f64> {
    if d == 0 || differenced.is_empty() {
        return differenced.to_vec();
    }

    let mut result = differenced.to_vec();

    // We need to reverse the differencing d times
    for level in (0..d).rev() {
        // Get the initial value at this differencing level
        let init_value = if level == 0 {
            *original.last().unwrap_or(&0.0)
        } else {
            // For higher levels, we need the last value of the intermediate difference
            let intermediate = difference(original, level);
            *intermediate.last().unwrap_or(&0.0)
        };

        // Cumulative sum starting from the initial value
        let mut integrated = Vec::with_capacity(result.len());
        let mut cumsum = init_value;
        for &diff in &result {
            cumsum += diff;
            integrated.push(cumsum);
        }
        result = integrated;
    }

    result
}

/// Check if a series needs differencing using a simple variance ratio test.
///
/// # Arguments
/// * `series` - The input series
///
/// # Returns
/// Suggested differencing order (0, 1, or 2).
pub fn suggest_differencing(series: &[f64]) -> usize {
    if series.len() < 3 {
        return 0;
    }

    let var_0 = variance(series);
    let diff_1 = difference(series, 1);

    if diff_1.len() < 2 {
        return 0;
    }

    let var_1 = variance(&diff_1);

    // If variance decreases significantly after differencing, difference is needed
    if var_0 > 0.0 && var_1 / var_0 < 0.9 {
        // Check if second difference helps more
        let diff_2 = difference(&diff_1, 1);
        if diff_2.len() >= 2 {
            let var_2 = variance(&diff_2);
            if var_2 / var_1 < 0.9 && var_2 < var_0 {
                return 2;
            }
        }
        return 1;
    }

    0
}

/// Calculate variance of a series.
fn variance(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (series.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn difference_order_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = difference(&series, 0);
        assert_eq!(result, series);
    }

    #[test]
    fn difference_order_1() {
        let series = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = difference(&series, 1);
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn difference_order_2() {
        let series = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = difference(&series, 2);
        // First diff: [2, 3, 4, 5]
        // Second diff: [1, 1, 1]
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn difference_constant_series() {
        let series = vec![5.0, 5.0, 5.0, 5.0];
        let result = difference(&series, 1);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn difference_empty() {
        let series: Vec<f64> = vec![];
        let result = difference(&series, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn seasonal_difference_basic() {
        // Quarterly data: Q1 values increase by 10 each year
        let series = vec![
            100.0, 120.0, 80.0, 90.0, // Year 1
            110.0, 130.0, 90.0, 100.0, // Year 2
        ];
        let result = seasonal_difference(&series, 1, 4);
        // Each value minus same quarter previous year
        assert_eq!(result, vec![10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn seasonal_difference_no_change() {
        let series = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let result = seasonal_difference(&series, 1, 3);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn seasonal_difference_order_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0];
        let result = seasonal_difference(&series, 0, 2);
        assert_eq!(result, series);
    }

    #[test]
    fn integrate_reverses_difference() {
        let original = vec![10.0, 12.0, 15.0, 19.0, 24.0];
        let differenced = difference(&original, 1);
        let forecast_diff = vec![6.0, 7.0]; // Forecasted differences
        let integrated = integrate(&forecast_diff, &original, 1);

        // Should continue from last value: 24 + 6 = 30, 30 + 7 = 37
        assert_relative_eq!(integrated[0], 30.0, epsilon = 1e-10);
        assert_relative_eq!(integrated[1], 37.0, epsilon = 1e-10);
    }

    #[test]
    fn integrate_order_2() {
        let original = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let differenced = difference(&original, 2);
        // differenced = [1, 1, 1]

        let forecast_diff2 = vec![1.0, 1.0]; // Continue the pattern
        let integrated = integrate(&forecast_diff2, &original, 2);

        // The integration should produce reasonable continuation
        assert!(integrated.len() == 2);
    }

    #[test]
    fn suggest_differencing_stationary() {
        // White noise-like stationary series
        let series = vec![1.0, 0.5, 1.2, 0.8, 1.1, 0.9, 1.0, 1.1];
        let d = suggest_differencing(&series);
        assert_eq!(d, 0);
    }

    #[test]
    fn suggest_differencing_trend() {
        // Clear upward trend
        let series: Vec<f64> = (0..20).map(|i| 10.0 + 2.0 * i as f64).collect();
        let d = suggest_differencing(&series);
        assert!(d >= 1);
    }

    #[test]
    fn suggest_differencing_quadratic() {
        // Quadratic trend
        let series: Vec<f64> = (0..20).map(|i| (i * i) as f64).collect();
        let d = suggest_differencing(&series);
        assert!(d >= 1);
    }
}
