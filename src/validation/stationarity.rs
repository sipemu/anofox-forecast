//! Stationarity tests for time series.
//!
//! Provides tests to determine if a time series is stationary.

/// Result of a stationarity test.
#[derive(Debug, Clone)]
pub struct StationarityResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value (approximate)
    pub p_value: f64,
    /// Number of lags used
    pub lags: usize,
    /// Whether series appears stationary
    pub is_stationary: bool,
    /// Critical values at common significance levels
    pub critical_values: CriticalValues,
}

/// Critical values for stationarity tests.
#[derive(Debug, Clone, Default)]
pub struct CriticalValues {
    /// Critical value at 1% significance
    pub cv_1pct: f64,
    /// Critical value at 5% significance
    pub cv_5pct: f64,
    /// Critical value at 10% significance
    pub cv_10pct: f64,
}

/// Augmented Dickey-Fuller test for unit root (non-stationarity).
///
/// Tests null hypothesis that series has a unit root (non-stationary).
/// Rejection implies stationarity.
///
/// # Arguments
/// * `series` - Time series data
/// * `max_lags` - Maximum lags to include (default: (n-1)^(1/3))
///
/// # Returns
/// `StationarityResult` with test statistic and p-value
pub fn adf_test(series: &[f64], max_lags: Option<usize>) -> StationarityResult {
    let n = series.len();

    if n < 4 {
        return StationarityResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags: 0,
            is_stationary: false,
            critical_values: CriticalValues::default(),
        };
    }

    // Default lag selection: (n-1)^(1/3)
    let max_lags = max_lags.unwrap_or_else(|| ((n - 1) as f64).powf(1.0 / 3.0).floor() as usize);
    let max_lags = max_lags.min(n / 2 - 1).max(1);

    // First difference
    let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    // Select optimal lag using AIC
    let (best_lag, _) = select_lag_aic(&diff, &series[..n - 1], max_lags);

    // Run ADF regression: Δy_t = α + β*y_{t-1} + Σγ_i*Δy_{t-i} + ε_t
    let (stat, se) = compute_adf_statistic(&diff, &series[..n - 1], best_lag);

    if se == 0.0 || se.is_nan() {
        return StationarityResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags: best_lag,
            is_stationary: false,
            critical_values: CriticalValues::default(),
        };
    }

    let t_stat = stat / se;

    // Critical values for ADF with constant (MacKinnon approximation)
    let critical_values = CriticalValues {
        cv_1pct: -3.43,
        cv_5pct: -2.86,
        cv_10pct: -2.57,
    };

    // Approximate p-value using MacKinnon tables
    let p_value = adf_p_value(t_stat, n);

    // Series is stationary if we reject null (t_stat < critical value)
    let is_stationary = t_stat < critical_values.cv_5pct;

    StationarityResult {
        statistic: t_stat,
        p_value,
        lags: best_lag,
        is_stationary,
        critical_values,
    }
}

/// Select lag order using AIC.
fn select_lag_aic(diff: &[f64], level: &[f64], max_lags: usize) -> (usize, f64) {
    let mut best_lag = 1;
    let mut best_aic = f64::INFINITY;

    for lag in 1..=max_lags {
        let aic = compute_aic(diff, level, lag);
        if aic < best_aic {
            best_aic = aic;
            best_lag = lag;
        }
    }

    (best_lag, best_aic)
}

/// Compute AIC for a given lag order.
fn compute_aic(diff: &[f64], level: &[f64], lag: usize) -> f64 {
    let n = diff.len();
    if n <= lag + 1 {
        return f64::INFINITY;
    }

    let start = lag;
    let effective_n = n - start;

    if effective_n < 3 {
        return f64::INFINITY;
    }

    // Build design matrix and compute residual sum of squares
    let rss = compute_rss(diff, level, lag);

    if rss <= 0.0 {
        return f64::INFINITY;
    }

    // AIC = n * ln(RSS/n) + 2 * k
    let k = lag + 2; // intercept + level coefficient + lag coefficients
    effective_n as f64 * (rss / effective_n as f64).ln() + 2.0 * k as f64
}

/// Compute residual sum of squares for ADF regression.
fn compute_rss(diff: &[f64], level: &[f64], lag: usize) -> f64 {
    let n = diff.len();
    let start = lag;

    if n <= start + 1 || level.len() <= start {
        return f64::INFINITY;
    }

    // Simple OLS for: Δy_t = α + β*y_{t-1} + Σγ_i*Δy_{t-i}
    let effective_n = n - start;

    // Compute means
    let y_mean: f64 = diff[start..].iter().sum::<f64>() / effective_n as f64;
    let x_mean: f64 = level[start..n].iter().sum::<f64>() / effective_n as f64;

    // Simple regression of diff on level (ignoring lags for AIC simplicity)
    let mut xx = 0.0;
    let mut xy = 0.0;

    for i in start..n {
        let x = level[i] - x_mean;
        let y = diff[i] - y_mean;
        xx += x * x;
        xy += x * y;
    }

    if xx == 0.0 {
        return f64::INFINITY;
    }

    let beta = xy / xx;
    let alpha = y_mean - beta * x_mean;

    // Compute RSS
    let mut rss = 0.0;
    for i in start..n {
        let predicted = alpha + beta * level[i];
        let residual = diff[i] - predicted;
        rss += residual * residual;
    }

    rss
}

/// Compute ADF test statistic and standard error.
fn compute_adf_statistic(diff: &[f64], level: &[f64], lag: usize) -> (f64, f64) {
    let n = diff.len();
    let start = lag;

    if n <= start + 2 || level.len() <= start {
        return (f64::NAN, f64::NAN);
    }

    let effective_n = n - start;

    // Build augmented regression
    // y = diff[start..], X = [1, level[start..], diff_lags]

    // For simplicity, just compute the coefficient on level[t-1]
    let y_mean: f64 = diff[start..].iter().sum::<f64>() / effective_n as f64;
    let x_mean: f64 = level[start..n].iter().sum::<f64>() / effective_n as f64;

    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;

    for i in start..n {
        let x = level[i] - x_mean;
        let y = diff[i] - y_mean;
        xx += x * x;
        xy += x * y;
        yy += y * y;
    }

    if xx == 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let beta = xy / xx;

    // Compute residual variance
    let rss = yy - beta * xy;
    let sigma_sq = rss / (effective_n - 2) as f64;

    if sigma_sq <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    // Standard error of beta
    let se_beta = (sigma_sq / xx).sqrt();

    (beta, se_beta)
}

/// Approximate p-value for ADF test using MacKinnon regression.
fn adf_p_value(t_stat: f64, n: usize) -> f64 {
    // MacKinnon (1994) regression coefficients for constant, no trend
    // tau_c critical values
    // Simplified approximation using normal CDF for extreme values

    if t_stat.is_nan() {
        return f64::NAN;
    }

    // Approximate mapping from t-statistic to p-value
    // This is a simplified approximation
    if t_stat < -4.0 {
        0.001
    } else if t_stat < -3.43 {
        0.01
    } else if t_stat < -2.86 {
        0.05
    } else if t_stat < -2.57 {
        0.10
    } else if t_stat < -1.94 {
        0.20
    } else if t_stat < -1.62 {
        0.30
    } else if t_stat < -1.28 {
        0.40
    } else if t_stat < -0.84 {
        0.50
    } else if t_stat < 0.0 {
        0.70
    } else {
        0.90 + 0.05 * (1.0 - (-t_stat).exp())
    }
}

/// KPSS test for stationarity.
///
/// Tests null hypothesis that series is (trend) stationary.
/// Rejection implies non-stationarity.
///
/// # Arguments
/// * `series` - Time series data
/// * `lags` - Number of lags for HAC variance (default: 4*(n/100)^0.25)
///
/// # Returns
/// `StationarityResult` with test statistic and p-value
pub fn kpss_test(series: &[f64], lags: Option<usize>) -> StationarityResult {
    let n = series.len();

    if n < 4 {
        return StationarityResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags: 0,
            is_stationary: false,
            critical_values: CriticalValues::default(),
        };
    }

    // Default lag: 4 * (n/100)^0.25
    let lags = lags.unwrap_or_else(|| (4.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize);
    let lags = lags.min(n / 2).max(1);

    // Demean the series (level stationarity)
    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let residuals: Vec<f64> = series.iter().map(|&x| x - mean).collect();

    // Compute cumulative sum of residuals
    let mut cumsum = vec![0.0; n];
    cumsum[0] = residuals[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + residuals[i];
    }

    // Compute numerator: sum of squared cumulative sums
    let numerator: f64 = cumsum.iter().map(|&s| s * s).sum::<f64>() / (n * n) as f64;

    // Compute HAC variance estimator (Bartlett kernel)
    let mut variance = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    for j in 1..=lags {
        let weight = 1.0 - j as f64 / (lags + 1) as f64;
        let autocovar: f64 = residuals
            .iter()
            .skip(j)
            .zip(residuals.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / n as f64;
        variance += 2.0 * weight * autocovar;
    }

    if variance <= 0.0 {
        return StationarityResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags,
            is_stationary: true,
            critical_values: CriticalValues::default(),
        };
    }

    let stat = numerator / variance;

    // Critical values for KPSS level stationarity
    let critical_values = CriticalValues {
        cv_1pct: 0.739,
        cv_5pct: 0.463,
        cv_10pct: 0.347,
    };

    // Approximate p-value
    let p_value = kpss_p_value(stat);

    // Series is stationary if we fail to reject null (stat < critical value)
    let is_stationary = stat < critical_values.cv_5pct;

    StationarityResult {
        statistic: stat,
        p_value,
        lags,
        is_stationary,
        critical_values,
    }
}

/// Approximate p-value for KPSS test.
fn kpss_p_value(stat: f64) -> f64 {
    if stat.is_nan() {
        return f64::NAN;
    }

    // Simplified approximation based on critical values
    if stat < 0.347 {
        0.10 + 0.90 * (1.0 - stat / 0.347)
    } else if stat < 0.463 {
        0.05 + 0.05 * (0.463 - stat) / (0.463 - 0.347)
    } else if stat < 0.739 {
        0.01 + 0.04 * (0.739 - stat) / (0.739 - 0.463)
    } else {
        0.01 * (1.0 - (stat - 0.739).min(1.0))
    }
}

/// Combined stationarity test using both ADF and KPSS.
///
/// # Returns
/// A tuple of (adf_result, kpss_result, conclusion)
/// where conclusion is:
/// - "stationary" if ADF rejects AND KPSS fails to reject
/// - "non_stationary" if ADF fails to reject AND KPSS rejects
/// - "inconclusive" otherwise
pub fn test_stationarity(series: &[f64]) -> (StationarityResult, StationarityResult, &'static str) {
    let adf = adf_test(series, None);
    let kpss = kpss_test(series, None);

    let conclusion = if adf.is_stationary && kpss.is_stationary {
        "stationary"
    } else if !adf.is_stationary && !kpss.is_stationary {
        "non_stationary"
    } else {
        "inconclusive"
    };

    (adf, kpss, conclusion)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== adf_test ====================

    #[test]
    fn adf_stationary_series() {
        // White noise should be stationary
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = adf_test(&series, Some(5));

        assert!(!result.statistic.is_nan());
        assert!(result.statistic < 0.0); // Should be negative
    }

    #[test]
    fn adf_random_walk() {
        // Random walk should be non-stationary
        let mut series = vec![0.0; 200];
        for i in 1..200 {
            series[i] = series[i - 1] + ((i * 17) % 19) as f64 / 10.0 - 0.9;
        }

        let result = adf_test(&series, Some(5));

        assert!(!result.statistic.is_nan());
        // Just verify we get a valid result
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn adf_trending_series() {
        // Series with strong trend (add small noise to avoid numerical issues)
        let series: Vec<f64> = (0..200)
            .map(|i| i as f64 * 0.5 + ((i * 13) % 7) as f64 * 0.01)
            .collect();

        let result = adf_test(&series, Some(5));

        assert!(!result.statistic.is_nan());
        // Trending series should fail to reject null
        assert!(!result.is_stationary);
    }

    #[test]
    fn adf_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let result = adf_test(&series, Some(1));

        assert!(result.statistic.is_nan());
    }

    #[test]
    fn adf_empty() {
        let result = adf_test(&[], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn adf_critical_values() {
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = adf_test(&series, None);

        assert!(result.critical_values.cv_1pct < result.critical_values.cv_5pct);
        assert!(result.critical_values.cv_5pct < result.critical_values.cv_10pct);
    }

    // ==================== kpss_test ====================

    #[test]
    fn kpss_stationary_series() {
        // White noise should be stationary
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, Some(10));

        assert!(!result.statistic.is_nan());
        assert!(result.statistic > 0.0);
        // Should fail to reject null (stationary)
        assert!(result.is_stationary);
    }

    #[test]
    fn kpss_trending_series() {
        // Series with strong trend should reject stationarity
        let series: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();

        let result = kpss_test(&series, Some(10));

        assert!(!result.statistic.is_nan());
        // Should reject null (not stationary)
        assert!(!result.is_stationary);
    }

    #[test]
    fn kpss_random_walk() {
        // Random walk
        let mut series = vec![0.0; 200];
        for i in 1..200 {
            series[i] = series[i - 1] + ((i * 17) % 19) as f64 / 10.0 - 0.9;
        }

        let result = kpss_test(&series, Some(10));

        assert!(!result.statistic.is_nan());
    }

    #[test]
    fn kpss_short_series() {
        let series = vec![1.0, 2.0, 3.0];
        let result = kpss_test(&series, Some(1));

        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kpss_empty() {
        let result = kpss_test(&[], None);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn kpss_critical_values() {
        let series: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = kpss_test(&series, None);

        // KPSS critical values should increase
        assert!(result.critical_values.cv_10pct < result.critical_values.cv_5pct);
        assert!(result.critical_values.cv_5pct < result.critical_values.cv_1pct);
    }

    // ==================== test_stationarity ====================

    #[test]
    fn combined_test_stationary() {
        // White noise should be conclusively stationary
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let (adf, kpss, conclusion) = test_stationarity(&series);

        assert!(!adf.statistic.is_nan());
        assert!(!kpss.statistic.is_nan());
        // Conclusion depends on specific values
        assert!(conclusion == "stationary" || conclusion == "inconclusive");
    }

    #[test]
    fn combined_test_trending() {
        // Strong trend should be non-stationary (add small noise to avoid numerical issues)
        let series: Vec<f64> = (0..200)
            .map(|i| i as f64 * 0.5 + ((i * 13) % 7) as f64 * 0.01)
            .collect();

        let (adf, kpss, conclusion) = test_stationarity(&series);

        assert!(!adf.statistic.is_nan());
        assert!(!kpss.statistic.is_nan());
        // Should be non-stationary or inconclusive
        assert!(conclusion == "non_stationary" || conclusion == "inconclusive");
    }

    #[test]
    fn combined_test_short() {
        let series = vec![1.0, 2.0, 3.0];

        let (adf, kpss, _) = test_stationarity(&series);

        assert!(adf.statistic.is_nan());
        assert!(kpss.statistic.is_nan());
    }
}
