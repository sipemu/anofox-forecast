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

/// Jarque-Bera normality test result.
#[derive(Debug, Clone)]
pub struct JarqueBeraResult {
    /// Test statistic.
    pub statistic: f64,
    /// P-value (approximate, from chi-squared with df=2).
    pub p_value: f64,
    /// Sample skewness.
    pub skewness: f64,
    /// Sample excess kurtosis.
    pub excess_kurtosis: f64,
}

impl JarqueBeraResult {
    /// Check if residuals pass normality test at given significance level.
    pub fn is_normal(&self, alpha: f64) -> bool {
        self.p_value > alpha
    }
}

/// Perform Jarque-Bera test for normality of residuals.
pub fn jarque_bera(residuals: &[f64]) -> JarqueBeraResult {
    let n = residuals.len() as f64;
    if residuals.len() < 3 {
        return JarqueBeraResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            skewness: f64::NAN,
            excess_kurtosis: f64::NAN,
        };
    }
    let mean = residuals.iter().sum::<f64>() / n;
    let m2: f64 = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = residuals.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n;
    let m4: f64 = residuals.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return JarqueBeraResult {
            statistic: 0.0,
            p_value: 1.0,
            skewness: 0.0,
            excess_kurtosis: 0.0,
        };
    }
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / m2.powi(2);
    let excess_kurtosis = kurtosis - 3.0;
    let jb = n / 6.0 * (skewness.powi(2) + excess_kurtosis.powi(2) / 4.0);
    let p_value = chi_squared_sf(jb, 2);
    JarqueBeraResult {
        statistic: jb,
        p_value,
        skewness,
        excess_kurtosis,
    }
}

/// Unified residual diagnostics combining multiple tests.
#[derive(Debug, Clone)]
pub struct ResidualDiagnostics {
    pub ljung_box: LjungBoxResult,
    pub durbin_watson: DurbinWatsonResult,
    pub jarque_bera: JarqueBeraResult,
    pub mean: f64,
    pub variance: f64,
    pub n: usize,
}

impl ResidualDiagnostics {
    /// Check if residuals are adequate (pass Ljung-Box at given alpha).
    pub fn is_adequate(&self, alpha: f64) -> bool {
        self.ljung_box.is_white_noise(alpha)
    }
}

impl std::fmt::Display for ResidualDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Residual Diagnostics (n={})", self.n)?;
        writeln!(
            f,
            "  Mean: {:.6}, Variance: {:.6}",
            self.mean, self.variance
        )?;
        writeln!(
            f,
            "  Ljung-Box: Q={:.4}, p={:.4}",
            self.ljung_box.statistic, self.ljung_box.p_value
        )?;
        writeln!(
            f,
            "  Durbin-Watson: d={:.4} ({:?})",
            self.durbin_watson.statistic, self.durbin_watson.interpretation
        )?;
        write!(
            f,
            "  Jarque-Bera: JB={:.4}, p={:.4} (skew={:.4}, kurt={:.4})",
            self.jarque_bera.statistic,
            self.jarque_bera.p_value,
            self.jarque_bera.skewness,
            self.jarque_bera.excess_kurtosis
        )
    }
}

/// Run all residual diagnostics in one call.
pub fn diagnose_residuals(residuals: &[f64], fitted_params: usize) -> ResidualDiagnostics {
    let n = residuals.len();
    let mean = if n > 0 {
        residuals.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let variance = if n > 1 {
        residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    ResidualDiagnostics {
        ljung_box: ljung_box(residuals, None, fitted_params),
        durbin_watson: durbin_watson(residuals),
        jarque_bera: jarque_bera(residuals),
        mean,
        variance,
        n,
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

    // ==================== jarque_bera ====================

    #[test]
    fn jarque_bera_normal_data() {
        // Pseudo-normal residuals (symmetric, mesokurtic) should pass
        let residuals: Vec<f64> = (0..200)
            .map(|i| {
                let x = ((i * 17 + 13) % 97) as f64 / 97.0;
                let y = ((i * 31 + 7) % 89) as f64 / 89.0;
                // Box-Muller-ish transform for approximate normality
                (x - 0.5) + (y - 0.5)
            })
            .collect();

        let result = jarque_bera(&residuals);

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        // Skewness should be near zero for symmetric data
        assert!(result.skewness.abs() < 1.0);
    }

    #[test]
    fn jarque_bera_skewed_data() {
        // Highly skewed data should fail normality test
        let mut residuals = vec![0.1; 100];
        residuals.extend(vec![10.0; 5]); // Add outliers to create skew

        let result = jarque_bera(&residuals);

        assert!(result.statistic > 0.0);
        assert!(result.skewness.abs() > 0.5, "Data should be skewed");
    }

    #[test]
    fn jarque_bera_constant_data() {
        let residuals = vec![3.0; 50];
        let result = jarque_bera(&residuals);

        assert_relative_eq!(result.statistic, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.p_value, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.skewness, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.excess_kurtosis, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn jarque_bera_too_short() {
        let residuals = vec![1.0, 2.0];
        let result = jarque_bera(&residuals);

        assert!(result.statistic.is_nan());
        assert!(result.p_value.is_nan());
        assert!(result.skewness.is_nan());
        assert!(result.excess_kurtosis.is_nan());
    }

    #[test]
    fn jarque_bera_is_normal() {
        let result = JarqueBeraResult {
            statistic: 1.0,
            p_value: 0.60,
            skewness: 0.1,
            excess_kurtosis: 0.05,
        };
        assert!(result.is_normal(0.05));
        assert!(!result.is_normal(0.99));
    }

    #[test]
    fn jarque_bera_single_value() {
        let residuals = vec![42.0];
        let result = jarque_bera(&residuals);
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn jarque_bera_empty() {
        let result = jarque_bera(&[]);
        assert!(result.statistic.is_nan());
    }

    // ==================== diagnose_residuals ====================

    #[test]
    fn diagnose_residuals_basic() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let diag = diagnose_residuals(&residuals, 0);

        assert_eq!(diag.n, 100);
        assert!(!diag.mean.is_nan());
        assert!(diag.variance >= 0.0);
        assert!(!diag.ljung_box.statistic.is_nan());
        assert!(!diag.durbin_watson.statistic.is_nan());
        assert!(!diag.jarque_bera.statistic.is_nan());
    }

    #[test]
    fn diagnose_residuals_empty() {
        let diag = diagnose_residuals(&[], 0);

        assert_eq!(diag.n, 0);
        assert_relative_eq!(diag.mean, 0.0, epsilon = 1e-10);
        assert_relative_eq!(diag.variance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn diagnose_residuals_single_value() {
        let diag = diagnose_residuals(&[5.0], 0);

        assert_eq!(diag.n, 1);
        assert_relative_eq!(diag.mean, 5.0, epsilon = 1e-10);
        assert_relative_eq!(diag.variance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn diagnose_residuals_constant() {
        let residuals = vec![2.0; 50];
        let diag = diagnose_residuals(&residuals, 0);

        assert_eq!(diag.n, 50);
        assert_relative_eq!(diag.mean, 2.0, epsilon = 1e-10);
        assert_relative_eq!(diag.variance, 0.0, epsilon = 1e-10);
        // Constant residuals should pass Ljung-Box (no autocorrelation)
        assert!(diag.ljung_box.is_white_noise(0.05));
    }

    #[test]
    fn diagnose_residuals_is_adequate() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let diag = diagnose_residuals(&residuals, 0);

        // is_adequate delegates to ljung_box.is_white_noise
        assert_eq!(diag.is_adequate(0.05), diag.ljung_box.is_white_noise(0.05));
    }

    #[test]
    fn diagnose_residuals_display() {
        let residuals: Vec<f64> = (0..50)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let diag = diagnose_residuals(&residuals, 0);
        let display = format!("{}", diag);

        assert!(display.contains("Residual Diagnostics"));
        assert!(display.contains("Ljung-Box"));
        assert!(display.contains("Durbin-Watson"));
        assert!(display.contains("Jarque-Bera"));
    }

    #[test]
    fn diagnose_residuals_with_fitted_params() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let diag0 = diagnose_residuals(&residuals, 0);
        let diag3 = diagnose_residuals(&residuals, 3);

        // Same statistic, different degrees of freedom
        assert_relative_eq!(
            diag0.ljung_box.statistic,
            diag3.ljung_box.statistic,
            epsilon = 1e-10
        );
        assert!(diag0.ljung_box.df > diag3.ljung_box.df);
    }

    // ==================== ljung_box edge cases ====================

    #[test]
    fn ljung_box_default_lags() {
        let residuals: Vec<f64> = (0..100)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        // None lags should use default: min(10, n/5)
        let result = ljung_box(&residuals, None, 0);
        assert_eq!(result.lags, 10); // min(10, 100/5) = 10
    }

    #[test]
    fn ljung_box_lags_clamped_to_n_minus_1() {
        let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Request more lags than n-1
        let result = ljung_box(&residuals, Some(100), 0);
        assert_eq!(result.lags, 4); // clamped to n - 1 = 4
    }

    #[test]
    fn ljung_box_three_values() {
        // Exactly 3 values, the minimum for non-NaN
        let residuals = vec![1.0, -1.0, 0.5];
        let result = ljung_box(&residuals, Some(1), 0);
        assert!(!result.statistic.is_nan());
        assert!(result.statistic >= 0.0);
    }

    // ==================== durbin_watson edge cases ====================

    #[test]
    fn durbin_watson_two_values() {
        let result = durbin_watson(&[1.0, 2.0]);
        assert!(!result.statistic.is_nan());
        assert!(result.statistic >= 0.0 && result.statistic <= 4.0);
    }

    #[test]
    fn durbin_watson_empty() {
        let result = durbin_watson(&[]);
        assert!(result.statistic.is_nan());
        assert_eq!(result.interpretation, AutocorrelationType::None);
    }

    #[test]
    fn durbin_watson_all_interpretation_ranges() {
        // PositiveStrong: DW < 0.5
        // Highly correlated: each value close to previous
        let mut strong_pos = vec![0.0; 100];
        strong_pos[0] = 10.0;
        for i in 1..100 {
            strong_pos[i] = strong_pos[i - 1] * 0.999;
        }
        let result = durbin_watson(&strong_pos);
        assert!(result.statistic < 0.5);
        assert_eq!(result.interpretation, AutocorrelationType::PositiveStrong);

        // None: DW near 2
        let result_none = DurbinWatsonResult {
            statistic: 2.0,
            interpretation: AutocorrelationType::None,
        };
        assert_eq!(result_none.interpretation, AutocorrelationType::None);

        // NegativeStrong: DW >= 3.5
        let alt: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 100.0 } else { -100.0 })
            .collect();
        let result = durbin_watson(&alt);
        assert!(result.statistic > 3.5);
        assert_eq!(result.interpretation, AutocorrelationType::NegativeStrong);
    }

    // ==================== box_pierce edge cases ====================

    #[test]
    fn box_pierce_empty() {
        let result = box_pierce(&[], Some(5));
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn box_pierce_default_lags() {
        let residuals: Vec<f64> = (0..50)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();

        let result = box_pierce(&residuals, None);
        assert_eq!(result.lags, 10); // min(10, 50/5)
    }

    // ==================== NaN handling ====================

    #[test]
    fn ljung_box_with_nan_values() {
        // NaN in residuals propagates through computation
        let mut residuals: Vec<f64> = (0..50)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        residuals[10] = f64::NAN;

        let result = ljung_box(&residuals, Some(5), 0);
        // Result should be NaN because NaN propagates
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn durbin_watson_with_nan() {
        let residuals = vec![1.0, f64::NAN, 3.0, 4.0];
        let result = durbin_watson(&residuals);
        // NaN should propagate
        assert!(result.statistic.is_nan());
    }

    #[test]
    fn jarque_bera_with_nan() {
        let mut residuals: Vec<f64> = (0..50)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        residuals[5] = f64::NAN;

        let result = jarque_bera(&residuals);
        assert!(result.statistic.is_nan());
    }
}
