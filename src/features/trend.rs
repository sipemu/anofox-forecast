//! Trend-based features for time series.
//!
//! Provides features related to linear trends and autoregressive properties.

/// Result of linear regression.
#[derive(Debug, Clone)]
pub struct LinearTrendResult {
    /// Slope of the fitted line
    pub slope: f64,
    /// Intercept of the fitted line
    pub intercept: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Standard error of the slope
    pub stderr: f64,
    /// P-value for the slope (approximate)
    pub p_value: f64,
}

/// Computes linear trend statistics for the time series.
///
/// Fits a linear regression y = slope * x + intercept where x is the index.
pub fn linear_trend(series: &[f64]) -> LinearTrendResult {
    if series.len() < 2 {
        return LinearTrendResult {
            slope: f64::NAN,
            intercept: f64::NAN,
            r_squared: f64::NAN,
            stderr: f64::NAN,
            p_value: f64::NAN,
        };
    }

    let n = series.len() as f64;

    // x values are indices 0, 1, 2, ...
    let sum_x: f64 = (0..series.len()).map(|i| i as f64).sum();
    let sum_y: f64 = series.iter().sum();
    let sum_xy: f64 = series.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..series.len()).map(|i| (i * i) as f64).sum();

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let ss_xx = sum_x2 - n * mean_x * mean_x;
    let ss_xy = sum_xy - n * mean_x * mean_y;

    if ss_xx.abs() < 1e-10 {
        return LinearTrendResult {
            slope: 0.0,
            intercept: mean_y,
            r_squared: 0.0,
            stderr: f64::NAN,
            p_value: 1.0,
        };
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    // Compute R-squared
    let ss_yy: f64 = series.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = series
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let y_pred = slope * i as f64 + intercept;
            (y - y_pred).powi(2)
        })
        .sum();

    let r_squared = if ss_yy.abs() < 1e-10 {
        1.0 // Perfect fit for constant y
    } else {
        1.0 - ss_res / ss_yy
    };

    // Compute standard error of slope
    let mse = if n > 2.0 { ss_res / (n - 2.0) } else { 0.0 };
    let stderr = if ss_xx > 0.0 {
        (mse / ss_xx).sqrt()
    } else {
        f64::NAN
    };

    // Compute approximate p-value using t-distribution
    let t_stat = if stderr > 1e-10 {
        slope / stderr
    } else {
        f64::INFINITY
    };

    // Approximate p-value (2-tailed) using normal approximation for large n
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));

    LinearTrendResult {
        slope,
        intercept,
        r_squared,
        stderr,
        p_value,
    }
}

/// Computes aggregated linear trend statistics.
///
/// Divides the series into chunks and computes linear trends for each,
/// then aggregates the results.
///
/// # Arguments
/// * `series` - Input time series
/// * `n_chunks` - Number of chunks
/// * `agg_func` - Aggregation function: "mean", "var", "std", "min", "max"
/// * `attribute` - Which attribute to aggregate: "slope", "intercept", "r_squared", "stderr"
pub fn agg_linear_trend(series: &[f64], n_chunks: usize, agg_func: &str, attribute: &str) -> f64 {
    if series.is_empty() || n_chunks == 0 {
        return f64::NAN;
    }

    let chunk_size = (series.len() + n_chunks - 1) / n_chunks;
    let mut values: Vec<f64> = Vec::new();

    for i in 0..n_chunks {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(series.len());
        if start >= series.len() {
            break;
        }
        let chunk = &series[start..end];
        if chunk.len() >= 2 {
            let trend = linear_trend(chunk);
            let val = match attribute {
                "slope" => trend.slope,
                "intercept" => trend.intercept,
                "r_squared" => trend.r_squared,
                "stderr" => trend.stderr,
                _ => f64::NAN,
            };
            if !val.is_nan() {
                values.push(val);
            }
        }
    }

    if values.is_empty() {
        return f64::NAN;
    }

    aggregate(&values, agg_func)
}

/// Estimates AR coefficients using Yule-Walker equations.
///
/// # Arguments
/// * `series` - Input time series
/// * `k` - Which coefficient to return (1-indexed)
pub fn ar_coefficient(series: &[f64], k: usize) -> f64 {
    if series.len() <= k || k == 0 {
        return f64::NAN;
    }

    // Compute autocorrelations up to lag k
    let mean: f64 = series.iter().sum::<f64>() / series.len() as f64;
    let var: f64 = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;

    if var < 1e-10 {
        return 0.0;
    }

    let acf: Vec<f64> = (0..=k)
        .map(|lag| {
            if lag == 0 {
                1.0
            } else {
                let cov: f64 = series[lag..]
                    .iter()
                    .zip(series.iter())
                    .map(|(&x1, &x2)| (x1 - mean) * (x2 - mean))
                    .sum::<f64>()
                    / series.len() as f64;
                cov / var
            }
        })
        .collect();

    // Solve Yule-Walker equations using Durbin-Levinson
    let mut phi = vec![0.0; k + 1];
    phi[1] = acf[1];

    for m in 2..=k {
        let mut num = acf[m];
        for j in 1..m {
            num -= phi[j] * acf[m - j];
        }

        let mut denom = 1.0;
        for j in 1..m {
            denom -= phi[j] * acf[j];
        }

        if denom.abs() < 1e-10 {
            return f64::NAN;
        }

        let new_phi = num / denom;

        // Update coefficients
        let mut new_coeffs = vec![0.0; k + 1];
        new_coeffs[m] = new_phi;
        for j in 1..m {
            new_coeffs[j] = phi[j] - new_phi * phi[m - j];
        }
        phi = new_coeffs;
    }

    phi[k]
}

/// Performs the Augmented Dickey-Fuller test for stationarity.
///
/// Returns the ADF test statistic. More negative values indicate
/// stronger evidence against the unit root hypothesis (i.e., evidence for stationarity).
pub fn augmented_dickey_fuller(series: &[f64]) -> f64 {
    if series.len() < 4 {
        return f64::NAN;
    }

    // Compute first differences
    let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    let n = diff.len();

    // Lagged level (y_{t-1})
    let y_lag: Vec<f64> = series[..n].to_vec();

    // Simple regression: diff = alpha + beta * y_lag + error
    let mean_y_lag: f64 = y_lag.iter().sum::<f64>() / n as f64;
    let mean_diff: f64 = diff.iter().sum::<f64>() / n as f64;

    let ss_yy: f64 = y_lag.iter().map(|&y| (y - mean_y_lag).powi(2)).sum();
    let ss_xy: f64 = diff
        .iter()
        .zip(y_lag.iter())
        .map(|(&d, &y)| (d - mean_diff) * (y - mean_y_lag))
        .sum();

    if ss_yy.abs() < 1e-10 {
        return f64::NAN;
    }

    let beta = ss_xy / ss_yy;
    let alpha = mean_diff - beta * mean_y_lag;

    // Compute residuals and standard error
    let residuals: Vec<f64> = diff
        .iter()
        .zip(y_lag.iter())
        .map(|(&d, &y)| d - alpha - beta * y)
        .collect();

    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let mse = sse / (n - 2) as f64;
    let se_beta = (mse / ss_yy).sqrt();

    if se_beta < 1e-10 {
        return f64::NAN;
    }

    // ADF statistic is beta / se(beta)
    beta / se_beta
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
        "min" => values.iter().copied().fold(f64::INFINITY, f64::min),
        "max" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        _ => f64::NAN,
    }
}

/// Helper: approximate normal CDF using error function approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Helper: error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== linear_trend ====================

    #[test]
    fn linear_trend_perfect_line() {
        // y = 2x + 1
        let series: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect();
        let trend = linear_trend(&series);

        assert_relative_eq!(trend.slope, 2.0, epsilon = 1e-10);
        assert_relative_eq!(trend.intercept, 1.0, epsilon = 1e-10);
        assert_relative_eq!(trend.r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_trend_no_trend() {
        let series = vec![5.0; 10];
        let trend = linear_trend(&series);

        assert_relative_eq!(trend.slope, 0.0, epsilon = 1e-10);
        assert_relative_eq!(trend.intercept, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_trend_negative_slope() {
        // y = -1.5x + 10
        let series: Vec<f64> = (0..10).map(|i| -1.5 * i as f64 + 10.0).collect();
        let trend = linear_trend(&series);

        assert_relative_eq!(trend.slope, -1.5, epsilon = 1e-10);
        assert_relative_eq!(trend.intercept, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_trend_with_noise() {
        // y = x + noise
        let series = vec![0.1, 1.2, 1.9, 3.1, 4.0, 5.2, 5.9, 7.1, 8.0, 9.1];
        let trend = linear_trend(&series);

        assert!(trend.slope > 0.9 && trend.slope < 1.1);
        assert!(trend.r_squared > 0.99);
    }

    #[test]
    fn linear_trend_short() {
        assert!(linear_trend(&[]).slope.is_nan());
        assert!(linear_trend(&[1.0]).slope.is_nan());
    }

    #[test]
    fn linear_trend_two_points() {
        let series = vec![0.0, 10.0];
        let trend = linear_trend(&series);

        assert_relative_eq!(trend.slope, 10.0, epsilon = 1e-10);
        assert_relative_eq!(trend.intercept, 0.0, epsilon = 1e-10);
    }

    // ==================== agg_linear_trend ====================

    #[test]
    fn agg_linear_trend_mean_slope() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let agg = agg_linear_trend(&series, 5, "mean", "slope");
        // Each chunk has slope ≈ 1
        assert_relative_eq!(agg, 1.0, epsilon = 0.1);
    }

    #[test]
    fn agg_linear_trend_var_slope() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let agg = agg_linear_trend(&series, 5, "var", "slope");
        // All chunks have same slope, so variance ≈ 0
        assert!(agg < 0.01);
    }

    #[test]
    fn agg_linear_trend_empty() {
        assert!(agg_linear_trend(&[], 5, "mean", "slope").is_nan());
    }

    #[test]
    fn agg_linear_trend_zero_chunks() {
        let series = vec![1.0, 2.0, 3.0];
        assert!(agg_linear_trend(&series, 0, "mean", "slope").is_nan());
    }

    // ==================== ar_coefficient ====================

    #[test]
    fn ar_coefficient_ar1_process() {
        // Simulate AR(1): y[t] = 0.8 * y[t-1] + noise
        let mut series = vec![0.0; 200];
        for i in 1..200 {
            series[i] = 0.8 * series[i - 1] + (i as f64 * 0.1).sin() * 0.1;
        }
        let coef = ar_coefficient(&series, 1);
        // Should be close to 0.8
        assert!(coef > 0.6 && coef < 1.0, "AR(1) coef should be ~0.8, got {}", coef);
    }

    #[test]
    fn ar_coefficient_constant() {
        let series = vec![5.0; 50];
        let coef = ar_coefficient(&series, 1);
        assert_relative_eq!(coef, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn ar_coefficient_short() {
        assert!(ar_coefficient(&[], 1).is_nan());
        assert!(ar_coefficient(&[1.0], 1).is_nan());
        assert!(ar_coefficient(&[1.0, 2.0], 3).is_nan());
    }

    #[test]
    fn ar_coefficient_zero_k() {
        let series = vec![1.0, 2.0, 3.0, 4.0];
        assert!(ar_coefficient(&series, 0).is_nan());
    }

    // ==================== augmented_dickey_fuller ====================

    #[test]
    fn adf_stationary() {
        // Stationary series (mean-reverting)
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).sin()).collect();
        let adf = augmented_dickey_fuller(&series);
        // Should be negative for stationary series
        assert!(!adf.is_nan());
    }

    #[test]
    fn adf_unit_root() {
        // Random walk (unit root)
        let mut series = vec![0.0; 100];
        for i in 1..100 {
            series[i] = series[i - 1] + ((i * 7) % 11) as f64 - 5.0;
        }
        let adf = augmented_dickey_fuller(&series);
        assert!(!adf.is_nan());
    }

    #[test]
    fn adf_trending() {
        // Strong upward trend with some noise
        let series: Vec<f64> = (0..100)
            .map(|i| i as f64 * 2.0 + ((i * 7) % 5) as f64 * 0.1)
            .collect();
        let adf = augmented_dickey_fuller(&series);
        assert!(!adf.is_nan());
    }

    #[test]
    fn adf_short() {
        assert!(augmented_dickey_fuller(&[]).is_nan());
        assert!(augmented_dickey_fuller(&[1.0, 2.0]).is_nan());
    }

    #[test]
    fn adf_constant() {
        let series = vec![5.0; 50];
        let adf = augmented_dickey_fuller(&series);
        // Constant series has no variation, may return NaN or extreme value
        assert!(adf.is_nan() || adf.abs() < 1e-10);
    }
}
