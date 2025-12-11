//! Autocorrelation-based features for time series.
//!
//! Provides features based on serial correlation patterns.

use super::basic::mean;

/// Returns the autocorrelation at a specific lag.
///
/// # Arguments
/// * `series` - Input time series
/// * `lag` - Lag value
pub fn autocorrelation(series: &[f64], lag: usize) -> f64 {
    if series.len() <= lag {
        return f64::NAN;
    }

    let m = mean(series);
    let n = series.len();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &x) in series.iter().enumerate() {
        denominator += (x - m).powi(2);
        if i >= lag {
            numerator += (x - m) * (series[i - lag] - m);
        }
    }

    if denominator < 1e-10 {
        return 0.0;
    }

    numerator / denominator
}

/// Returns the partial autocorrelation at a specific lag.
///
/// Uses the Durbin-Levinson algorithm.
///
/// # Arguments
/// * `series` - Input time series
/// * `lag` - Lag value (must be >= 1)
pub fn partial_autocorrelation(series: &[f64], lag: usize) -> f64 {
    if lag == 0 {
        return 1.0;
    }
    if series.len() <= lag {
        return f64::NAN;
    }

    // Compute autocorrelations up to lag
    let acf: Vec<f64> = (0..=lag).map(|k| autocorrelation(series, k)).collect();

    if acf.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }

    // Durbin-Levinson algorithm
    let mut phi = vec![vec![0.0; lag + 1]; lag + 1];

    phi[1][1] = acf[1];

    for k in 2..=lag {
        // Compute numerator
        let mut num = acf[k];
        for j in 1..k {
            num -= phi[k - 1][j] * acf[k - j];
        }

        // Compute denominator
        let mut denom = 1.0;
        for j in 1..k {
            denom -= phi[k - 1][j] * acf[j];
        }

        if denom.abs() < 1e-10 {
            return f64::NAN;
        }

        phi[k][k] = num / denom;

        // Update coefficients
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }

    phi[lag][lag]
}

/// Returns aggregated autocorrelation statistics.
///
/// Computes the autocorrelation for multiple lags and returns
/// an aggregation (mean, variance, etc.).
///
/// # Arguments
/// * `series` - Input time series
/// * `max_lag` - Maximum lag to consider
/// * `agg_func` - Aggregation function: "mean", "var", "std", "median"
pub fn agg_autocorrelation(series: &[f64], max_lag: usize, agg_func: &str) -> f64 {
    if series.len() <= max_lag || max_lag == 0 {
        return f64::NAN;
    }

    let acf_values: Vec<f64> = (1..=max_lag)
        .map(|lag| autocorrelation(series, lag))
        .filter(|x| !x.is_nan())
        .collect();

    if acf_values.is_empty() {
        return f64::NAN;
    }

    match agg_func {
        "mean" => acf_values.iter().sum::<f64>() / acf_values.len() as f64,
        "var" => {
            if acf_values.len() < 2 {
                return f64::NAN;
            }
            let m = acf_values.iter().sum::<f64>() / acf_values.len() as f64;
            let sum_sq: f64 = acf_values.iter().map(|x| (x - m).powi(2)).sum();
            sum_sq / (acf_values.len() - 1) as f64
        }
        "std" => {
            let var = agg_autocorrelation(series, max_lag, "var");
            var.sqrt()
        }
        "median" => {
            let mut sorted = acf_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            }
        }
        _ => f64::NAN,
    }
}

/// Returns the time reversal asymmetry statistic.
///
/// Measures whether the time series looks the same when reversed.
/// A value close to zero indicates time-reversible dynamics.
///
/// # Arguments
/// * `series` - Input time series
/// * `lag` - Lag value
pub fn time_reversal_asymmetry_statistic(series: &[f64], lag: usize) -> f64 {
    if series.len() <= 2 * lag {
        return f64::NAN;
    }

    let n = series.len();
    let mut sum = 0.0;

    for i in (2 * lag)..n {
        let x_i = series[i];
        let x_i_lag = series[i - lag];
        let x_i_2lag = series[i - 2 * lag];

        sum += x_i_lag.powi(2) * x_i - x_i_lag * x_i_2lag.powi(2);
    }

    sum / (n - 2 * lag) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== autocorrelation ====================

    #[test]
    fn autocorrelation_lag_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(autocorrelation(&series, 0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn autocorrelation_linear_trend() {
        // Linear trend should have high lag-1 autocorrelation
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let acf1 = autocorrelation(&series, 1);
        assert!(acf1 > 0.8, "Expected high ACF(1) for linear trend, got {}", acf1);
    }

    #[test]
    fn autocorrelation_alternating() {
        // Alternating series should have negative lag-1 autocorrelation
        let series: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let acf1 = autocorrelation(&series, 1);
        assert!(acf1 < -0.5, "Expected negative ACF(1) for alternating, got {}", acf1);
    }

    #[test]
    fn autocorrelation_seasonal() {
        // Series with period 4 should have high ACF at lag 4
        let series: Vec<f64> = (0..40)
            .map(|i| ((i % 4) as f64 * std::f64::consts::PI / 2.0).sin())
            .collect();
        let acf4 = autocorrelation(&series, 4);
        assert!(acf4 > 0.5, "Expected high ACF(4) for seasonal data, got {}", acf4);
    }

    #[test]
    fn autocorrelation_constant() {
        let series = vec![5.0; 10];
        assert_relative_eq!(autocorrelation(&series, 1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn autocorrelation_short() {
        assert!(autocorrelation(&[], 1).is_nan());
        assert!(autocorrelation(&[1.0], 1).is_nan());
        assert!(autocorrelation(&[1.0, 2.0], 5).is_nan());
    }

    // ==================== partial_autocorrelation ====================

    #[test]
    fn partial_autocorrelation_lag_0() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(partial_autocorrelation(&series, 0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn partial_autocorrelation_ar1() {
        // AR(1) process: only PACF(1) should be significant
        // x[t] = 0.8 * x[t-1] + noise
        let mut series = vec![0.0; 100];
        series[0] = 1.0;
        for i in 1..100 {
            series[i] = 0.8 * series[i - 1];
        }
        let pacf1 = partial_autocorrelation(&series, 1);
        assert!(pacf1 > 0.5, "Expected high PACF(1) for AR(1), got {}", pacf1);

        // PACF at higher lags should be smaller
        let pacf2 = partial_autocorrelation(&series, 2);
        assert!(
            pacf2.abs() < pacf1.abs(),
            "PACF(2) should be smaller than PACF(1)"
        );
    }

    #[test]
    fn partial_autocorrelation_short() {
        assert!(partial_autocorrelation(&[], 1).is_nan());
        assert!(partial_autocorrelation(&[1.0], 1).is_nan());
        assert!(partial_autocorrelation(&[1.0, 2.0], 5).is_nan());
    }

    // ==================== agg_autocorrelation ====================

    #[test]
    fn agg_autocorrelation_mean() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let agg = agg_autocorrelation(&series, 5, "mean");
        assert!(!agg.is_nan());
        // For trending series, mean ACF should be positive
        assert!(agg > 0.0);
    }

    #[test]
    fn agg_autocorrelation_var() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let agg = agg_autocorrelation(&series, 5, "var");
        assert!(!agg.is_nan());
        assert!(agg >= 0.0);
    }

    #[test]
    fn agg_autocorrelation_std() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let agg = agg_autocorrelation(&series, 5, "std");
        assert!(!agg.is_nan());
        assert!(agg >= 0.0);
    }

    #[test]
    fn agg_autocorrelation_median() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let agg = agg_autocorrelation(&series, 5, "median");
        assert!(!agg.is_nan());
    }

    #[test]
    fn agg_autocorrelation_invalid_func() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(agg_autocorrelation(&series, 5, "invalid").is_nan());
    }

    #[test]
    fn agg_autocorrelation_short() {
        assert!(agg_autocorrelation(&[1.0, 2.0], 5, "mean").is_nan());
        assert!(agg_autocorrelation(&[1.0, 2.0, 3.0], 0, "mean").is_nan());
    }

    // ==================== time_reversal_asymmetry_statistic ====================

    #[test]
    fn time_reversal_asymmetry_symmetric() {
        // A pure sine wave is time-reversible (after phase shift)
        let series: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        let tras = time_reversal_asymmetry_statistic(&series, 1);
        assert!(
            tras.abs() < 0.1,
            "Sine wave should be roughly symmetric, got {}",
            tras
        );
    }

    #[test]
    fn time_reversal_asymmetry_asymmetric() {
        // Sawtooth wave is not time-reversible
        let series: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let tras = time_reversal_asymmetry_statistic(&series, 1);
        // Should be non-zero but we don't know the sign
        assert!(!tras.is_nan());
    }

    #[test]
    fn time_reversal_asymmetry_constant() {
        let series = vec![5.0; 20];
        assert_relative_eq!(
            time_reversal_asymmetry_statistic(&series, 1),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn time_reversal_asymmetry_short() {
        assert!(time_reversal_asymmetry_statistic(&[], 1).is_nan());
        assert!(time_reversal_asymmetry_statistic(&[1.0, 2.0], 1).is_nan());
        assert!(time_reversal_asymmetry_statistic(&[1.0, 2.0, 3.0, 4.0], 3).is_nan());
    }

    #[test]
    fn time_reversal_asymmetry_different_lags() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let tras1 = time_reversal_asymmetry_statistic(&series, 1);
        let tras2 = time_reversal_asymmetry_statistic(&series, 2);
        // Both should be computable
        assert!(!tras1.is_nan());
        assert!(!tras2.is_nan());
    }
}
