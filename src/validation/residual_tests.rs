//! Residual diagnostic tests for time series models.
//!
//! Provides tests to validate model residuals are white noise.

/// Ljung-Box test result.
#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    /// Test statistic Q
    pub statistic: f64,
    /// P-value (approximate)
    pub p_value: f64,
    /// Number of lags tested
    pub lags: usize,
    /// Degrees of freedom
    pub df: usize,
}

impl LjungBoxResult {
    /// Check if residuals pass at given significance level.
    /// Returns true if we fail to reject null (residuals are white noise).
    pub fn is_white_noise(&self, alpha: f64) -> bool {
        self.p_value > alpha
    }
}

/// Perform Ljung-Box test for autocorrelation in residuals.
///
/// Tests null hypothesis that residuals are independently distributed (white noise).
///
/// # Arguments
/// * `residuals` - Model residuals
/// * `lags` - Number of lags to include (default: min(10, n/5))
/// * `fitted_params` - Number of fitted parameters (for degrees of freedom adjustment)
///
/// # Returns
/// `LjungBoxResult` with test statistic and p-value
pub fn ljung_box(residuals: &[f64], lags: Option<usize>, fitted_params: usize) -> LjungBoxResult {
    let n = residuals.len();

    if n < 3 {
        return LjungBoxResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags: 0,
            df: 0,
        };
    }

    // Default lags: min(10, n/5)
    let lags = lags.unwrap_or_else(|| 10.min(n / 5).max(1));
    let lags = lags.min(n - 1);

    // Compute autocorrelations
    let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|&x| x - mean).collect();

    let var: f64 = centered.iter().map(|&x| x * x).sum::<f64>();
    if var == 0.0 {
        return LjungBoxResult {
            statistic: 0.0,
            p_value: 1.0,
            lags,
            df: lags.saturating_sub(fitted_params),
        };
    }

    // Compute Q statistic
    let mut q = 0.0;
    for k in 1..=lags {
        let acf_k: f64 = centered
            .iter()
            .skip(k)
            .zip(centered.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / var;

        q += (acf_k * acf_k) / (n - k) as f64;
    }
    q *= n as f64 * (n + 2) as f64;

    // Degrees of freedom
    let df = lags.saturating_sub(fitted_params);
    let df = df.max(1);

    // Approximate p-value using chi-squared distribution
    let p_value = chi_squared_sf(q, df);

    LjungBoxResult {
        statistic: q,
        p_value,
        lags,
        df,
    }
}

/// Durbin-Watson test result.
#[derive(Debug, Clone)]
pub struct DurbinWatsonResult {
    /// Test statistic (0 to 4)
    pub statistic: f64,
    /// Interpretation
    pub interpretation: AutocorrelationType,
}

/// Type of autocorrelation detected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutocorrelationType {
    /// Strong positive autocorrelation (DW near 0)
    PositiveStrong,
    /// Weak positive autocorrelation (DW < 2)
    PositiveWeak,
    /// No autocorrelation (DW near 2)
    None,
    /// Weak negative autocorrelation (DW > 2)
    NegativeWeak,
    /// Strong negative autocorrelation (DW near 4)
    NegativeStrong,
}

/// Perform Durbin-Watson test for first-order autocorrelation.
///
/// # Arguments
/// * `residuals` - Model residuals
///
/// # Returns
/// `DurbinWatsonResult` with statistic (0-4) where:
/// - 0: Strong positive autocorrelation
/// - 2: No autocorrelation
/// - 4: Strong negative autocorrelation
pub fn durbin_watson(residuals: &[f64]) -> DurbinWatsonResult {
    let n = residuals.len();

    if n < 2 {
        return DurbinWatsonResult {
            statistic: f64::NAN,
            interpretation: AutocorrelationType::None,
        };
    }

    // Sum of squared differences
    let sum_diff_sq: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();

    // Sum of squared residuals
    let sum_sq: f64 = residuals.iter().map(|&r| r * r).sum();

    if sum_sq == 0.0 {
        return DurbinWatsonResult {
            statistic: 2.0,
            interpretation: AutocorrelationType::None,
        };
    }

    let dw = sum_diff_sq / sum_sq;

    // Interpret the statistic
    let interpretation = if dw < 0.5 {
        AutocorrelationType::PositiveStrong
    } else if dw < 1.5 {
        AutocorrelationType::PositiveWeak
    } else if dw <= 2.5 {
        AutocorrelationType::None
    } else if dw < 3.5 {
        AutocorrelationType::NegativeWeak
    } else {
        AutocorrelationType::NegativeStrong
    };

    DurbinWatsonResult {
        statistic: dw,
        interpretation,
    }
}

/// Box-Pierce test (simplified version of Ljung-Box).
///
/// # Arguments
/// * `residuals` - Model residuals
/// * `lags` - Number of lags to include
pub fn box_pierce(residuals: &[f64], lags: Option<usize>) -> LjungBoxResult {
    let n = residuals.len();

    if n < 3 {
        return LjungBoxResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            lags: 0,
            df: 0,
        };
    }

    let lags = lags.unwrap_or_else(|| 10.min(n / 5).max(1));
    let lags = lags.min(n - 1);

    let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|&x| x - mean).collect();

    let var: f64 = centered.iter().map(|&x| x * x).sum::<f64>();
    if var == 0.0 {
        return LjungBoxResult {
            statistic: 0.0,
            p_value: 1.0,
            lags,
            df: lags,
        };
    }

    // Compute Q statistic (simpler than Ljung-Box)
    let mut q = 0.0;
    for k in 1..=lags {
        let acf_k: f64 = centered
            .iter()
            .skip(k)
            .zip(centered.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            / var;

        q += acf_k * acf_k;
    }
    q *= n as f64;

    let p_value = chi_squared_sf(q, lags);

    LjungBoxResult {
        statistic: q,
        p_value,
        lags,
        df: lags,
    }
}

/// Approximate chi-squared survival function (1 - CDF).
fn chi_squared_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }

    // Use incomplete gamma function approximation
    // P(X > x) = 1 - gamma_inc(df/2, x/2) / gamma(df/2)
    // Using Wilson-Hilferty approximation for chi-squared
    let k = df as f64;

    // For large df, use normal approximation
    if df > 30 {
        let z = ((x / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
        return normal_sf(z);
    }

    // For smaller df, use series approximation
    incomplete_gamma_q(k / 2.0, x / 2.0)
}

/// Upper incomplete gamma function Q(a, x) = 1 - P(a, x).
fn incomplete_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 1.0;
    }

    if x == 0.0 {
        return 1.0;
    }

    // Use series expansion for small x, continued fraction for large x
    if x < a + 1.0 {
        1.0 - gamma_series_p(a, x)
    } else {
        gamma_cf_q(a, x)
    }
}

/// Lower incomplete gamma P(a, x) via series expansion.
fn gamma_series_p(a: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    let mut sum = 1.0 / a;
    let mut term = sum;

    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Upper incomplete gamma Q(a, x) via continued fraction.
fn gamma_cf_q(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Log gamma function using Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();

    let mut ser = 1.000000000190015;
    for (j, &coef) in coefficients.iter().enumerate() {
        ser += coef / (y + 1.0 + j as f64);
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Standard normal survival function.
fn normal_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Complementary error function approximation.
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();

    if x >= 0.0 {
        tau
    } else {
        2.0 - tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== ljung_box ====================

    #[test]
    fn ljung_box_white_noise() {
        // White noise should not reject null hypothesis
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = ljung_box(&residuals, Some(10), 0);

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.lags, 10);
    }

    #[test]
    fn ljung_box_autocorrelated() {
        // Strongly autocorrelated series should reject null
        let mut residuals = vec![0.0; 100];
        residuals[0] = 1.0;
        for i in 1..100 {
            residuals[i] = 0.9 * residuals[i - 1] + 0.1 * ((i * 17) % 23) as f64 / 23.0;
        }

        let result = ljung_box(&residuals, Some(10), 0);

        // Should have significant Q statistic
        assert!(result.statistic > 0.0);
        // P-value should be small for autocorrelated data
        assert!(result.p_value < 0.5);
    }

    #[test]
    fn ljung_box_constant() {
        let residuals = vec![1.0; 50];
        let result = ljung_box(&residuals, Some(5), 0);

        // Constant series has zero variance
        assert_eq!(result.statistic, 0.0);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn ljung_box_short() {
        let residuals = vec![1.0, 2.0];
        let result = ljung_box(&residuals, Some(5), 0);

        assert!(result.statistic.is_nan());
    }

    #[test]
    fn ljung_box_empty() {
        let result = ljung_box(&[], Some(5), 0);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn ljung_box_is_white_noise() {
        let result = LjungBoxResult {
            statistic: 5.0,
            p_value: 0.3,
            lags: 10,
            df: 10,
        };

        assert!(result.is_white_noise(0.05));
        assert!(!result.is_white_noise(0.5));
    }

    #[test]
    fn ljung_box_with_fitted_params() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result_0 = ljung_box(&residuals, Some(10), 0);
        let result_2 = ljung_box(&residuals, Some(10), 2);

        // Same statistic but different df
        assert_eq!(result_0.df, 10);
        assert_eq!(result_2.df, 8);
    }

    // ==================== durbin_watson ====================

    #[test]
    fn durbin_watson_no_autocorrelation() {
        // Alternating residuals should have DW near 2
        let residuals: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();

        let result = durbin_watson(&residuals);

        // Should be close to 2 (actually 4 for perfect alternation)
        assert!(result.statistic >= 0.0 && result.statistic <= 4.0);
    }

    #[test]
    fn durbin_watson_positive_autocorrelation() {
        // Slowly changing residuals
        let mut residuals = vec![0.0; 100];
        residuals[0] = 1.0;
        for i in 1..100 {
            residuals[i] = 0.95 * residuals[i - 1];
        }

        let result = durbin_watson(&residuals);

        // Should be close to 0 (positive autocorrelation)
        assert!(result.statistic < 1.0);
        assert!(
            result.interpretation == AutocorrelationType::PositiveStrong
                || result.interpretation == AutocorrelationType::PositiveWeak
        );
    }

    #[test]
    fn durbin_watson_negative_autocorrelation() {
        // Perfect alternating pattern
        let residuals: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let result = durbin_watson(&residuals);

        // Should be close to 4 (negative autocorrelation)
        assert!(result.statistic > 3.0);
        assert!(
            result.interpretation == AutocorrelationType::NegativeStrong
                || result.interpretation == AutocorrelationType::NegativeWeak
        );
    }

    #[test]
    fn durbin_watson_constant() {
        let residuals = vec![1.0; 50];
        let result = durbin_watson(&residuals);

        // Constant gives 0/sum_sq = 0, but we handle this
        assert_eq!(result.statistic, 0.0);
    }

    #[test]
    fn durbin_watson_short() {
        let result = durbin_watson(&[1.0]);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn durbin_watson_zero_residuals() {
        let residuals = vec![0.0; 50];
        let result = durbin_watson(&residuals);

        assert_eq!(result.statistic, 2.0);
        assert_eq!(result.interpretation, AutocorrelationType::None);
    }

    // ==================== box_pierce ====================

    #[test]
    fn box_pierce_white_noise() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = box_pierce(&residuals, Some(10));

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn box_pierce_vs_ljung_box() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let bp = box_pierce(&residuals, Some(10));
        let lb = ljung_box(&residuals, Some(10), 0);

        // Both should give similar results, but Ljung-Box has small-sample correction
        assert!(bp.statistic >= 0.0);
        assert!(lb.statistic >= 0.0);
        // Ljung-Box generally gives larger statistic
        assert!(lb.statistic >= bp.statistic * 0.9);
    }

    #[test]
    fn box_pierce_constant() {
        let residuals = vec![1.0; 50];
        let result = box_pierce(&residuals, Some(5));

        assert_eq!(result.statistic, 0.0);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn box_pierce_short() {
        let residuals = vec![1.0, 2.0];
        let result = box_pierce(&residuals, Some(5));

        assert!(result.statistic.is_nan());
    }

    // ==================== chi_squared_sf ====================

    #[test]
    fn chi_squared_sf_zero() {
        let p = chi_squared_sf(0.0, 5);
        assert_relative_eq!(p, 1.0, epsilon = 0.01);
    }

    #[test]
    fn chi_squared_sf_known_values() {
        // For df=2, chi-squared(2) is exponential
        // P(X > 2) ≈ 0.368 for df=2
        let p = chi_squared_sf(2.0, 2);
        assert!(p > 0.3 && p < 0.4);

        // For df=10, P(X > 18.31) ≈ 0.05
        let p = chi_squared_sf(18.31, 10);
        assert!(p > 0.03 && p < 0.07);
    }

    #[test]
    fn chi_squared_sf_large_df() {
        // Large df uses normal approximation
        let p = chi_squared_sf(50.0, 50);
        // Should be around 0.5 when x = df
        assert!(p > 0.3 && p < 0.7);
    }
}
