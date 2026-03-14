//! Data profiling API for agent-based forecasting.
//!
//! Provides [`DataProfile`], a comprehensive summary of a time series
//! that an agent can use to select and configure forecasting models.

use crate::core::TimeSeries;
use crate::features::{
    approximate_entropy, autocorrelation, kurtosis, lempel_ziv_complexity, linear_trend, maximum,
    mean, minimum, partial_autocorrelation, skewness, standard_deviation, LinearTrendResult,
};
use crate::validation::stationarity::{adf_test, kpss_test};
use std::fmt;

/// Direction of the linear trend in a time series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Slope is positive and significant relative to std_dev.
    Rising,
    /// Slope is negative and significant relative to std_dev.
    Falling,
    /// Slope is near zero relative to std_dev.
    Flat,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendDirection::Rising => write!(f, "Rising"),
            TrendDirection::Falling => write!(f, "Falling"),
            TrendDirection::Flat => write!(f, "Flat"),
        }
    }
}

/// A comprehensive statistical profile of a time series.
///
/// Computed via [`DataProfile::from_series`], this struct captures everything
/// an agent needs to know about a series before selecting models: basic
/// statistics, stationarity test results, trend characteristics,
/// autocorrelation structure, distributional shape, complexity measures,
/// intermittent-demand indicators, and an overall quality score.
#[derive(Debug, Clone)]
pub struct DataProfile {
    // ---- Basic stats ----
    /// Number of observations (including NaN/Inf).
    pub n_observations: usize,
    /// Arithmetic mean (NaN values excluded from computation).
    pub mean: f64,
    /// Population standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,

    // ---- Missing / quality ----
    /// Count of NaN or infinite values.
    pub missing_count: usize,
    /// Fraction of observations that are NaN or infinite.
    pub missing_fraction: f64,
    /// Whether any value is strictly negative.
    pub has_negatives: bool,
    /// Whether any value is exactly zero.
    pub has_zeros: bool,
    /// Whether every finite value is integer-valued.
    pub is_integer: bool,

    // ---- Stationarity ----
    /// Augmented Dickey-Fuller test statistic.
    pub adf_statistic: f64,
    /// Approximate p-value for the ADF test.
    pub adf_p_value: f64,
    /// Whether the ADF test concludes stationarity at 5%.
    pub adf_is_stationary: bool,
    /// KPSS test statistic.
    pub kpss_statistic: f64,
    /// Approximate p-value for the KPSS test.
    pub kpss_p_value: f64,
    /// Whether the KPSS test concludes stationarity at 5%.
    pub kpss_is_stationary: bool,

    // ---- Trend ----
    /// Strength of the linear trend (R-squared, 0.0 to 1.0).
    pub trend_strength: f64,
    /// Slope of the fitted linear trend.
    pub trend_slope: f64,
    /// Classified direction of the trend.
    pub trend_direction: TrendDirection,

    // ---- Autocorrelation ----
    /// Autocorrelation at lag 1.
    pub acf_lag1: f64,
    /// Autocorrelation at lag 2.
    pub acf_lag2: f64,
    /// Partial autocorrelation at lag 1.
    pub partial_acf_lag1: f64,

    // ---- Distribution shape ----
    /// Skewness (third standardized moment).
    pub skewness: f64,
    /// Excess kurtosis (fourth standardized moment, normal = 0).
    pub kurtosis: f64,

    // ---- Complexity ----
    /// Approximate entropy (None if series is too short).
    pub approximate_entropy: Option<f64>,
    /// Lempel-Ziv complexity (normalized).
    pub lempel_ziv: f64,

    // ---- Intermittent demand ----
    /// Fraction of values that are exactly zero.
    pub zero_fraction: f64,
    /// Whether the series is classified as intermittent (zero_fraction > 0.1).
    pub is_intermittent: bool,

    // ---- Overall quality ----
    /// Heuristic data-quality score in [0.0, 1.0].
    pub quality_score: f64,
}

impl DataProfile {
    /// Build a [`DataProfile`] from a [`TimeSeries`], using the first dimension.
    pub fn from_series(ts: &TimeSeries) -> Self {
        let values: &[f64] = ts.values(0).unwrap_or(&[]);
        Self::from_values(values)
    }

    /// Build a [`DataProfile`] directly from a slice of values.
    pub fn from_values(values: &[f64]) -> Self {
        let n_observations = values.len();

        // ---- Missing / quality flags ----
        let missing_count = values
            .iter()
            .filter(|v| v.is_nan() || v.is_infinite())
            .count();
        let missing_fraction = if n_observations > 0 {
            missing_count as f64 / n_observations as f64
        } else {
            0.0
        };

        // Clean values (finite only) for downstream computations.
        let clean: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();

        let has_negatives = clean.iter().any(|&v| v < 0.0);
        let has_zeros = clean.contains(&0.0);
        let is_integer = clean.iter().all(|&v| v == v.floor());

        // ---- Basic stats ----
        let mn = mean(&clean);
        let std_dev = standard_deviation(&clean);
        let min_val = if clean.is_empty() {
            f64::NAN
        } else {
            minimum(&clean)
        };
        let max_val = if clean.is_empty() {
            f64::NAN
        } else {
            maximum(&clean)
        };

        // ---- Stationarity ----
        let adf = adf_test(&clean, None);
        let kpss = kpss_test(&clean, None);

        // ---- Trend ----
        let trend: LinearTrendResult = linear_trend(&clean);
        let trend_strength = if trend.r_squared.is_nan() {
            0.0
        } else {
            trend.r_squared.clamp(0.0, 1.0)
        };
        let trend_slope = if trend.slope.is_nan() {
            0.0
        } else {
            trend.slope
        };
        let trend_direction = classify_trend(trend_slope, std_dev);

        // ---- Autocorrelation ----
        let acf_lag1 = autocorrelation(&clean, 1);
        let acf_lag2 = autocorrelation(&clean, 2);
        let partial_acf_lag1 = partial_autocorrelation(&clean, 1);

        // ---- Distribution shape ----
        let skew = skewness(&clean);
        let kurt = kurtosis(&clean);

        // ---- Complexity ----
        let apen = {
            let r = 0.2 * std_dev;
            let val = approximate_entropy(&clean, 2, r);
            if val.is_nan() {
                None
            } else {
                Some(val)
            }
        };
        let lz = lempel_ziv_complexity(&clean, 10);

        // ---- Intermittent demand ----
        let zero_count = clean.iter().filter(|&&v| v == 0.0).count();
        let zero_fraction = if clean.is_empty() {
            0.0
        } else {
            zero_count as f64 / clean.len() as f64
        };
        let is_intermittent = zero_fraction > 0.1;

        // ---- Quality score ----
        // Simple heuristic: penalise missing data and outliers.
        let has_outliers = if std_dev > 0.0 && !std_dev.is_nan() {
            clean.iter().any(|&v| ((v - mn) / std_dev).abs() > 4.0)
        } else {
            false
        };
        let quality_score =
            (1.0 - missing_fraction - if has_outliers { 0.1 } else { 0.0 }).clamp(0.0, 1.0);

        // Sanitise NaN results for display convenience.
        let safe = |v: f64| if v.is_nan() { 0.0 } else { v };

        DataProfile {
            n_observations,
            mean: mn,
            std_dev,
            min: min_val,
            max: max_val,

            missing_count,
            missing_fraction,
            has_negatives,
            has_zeros,
            is_integer,

            adf_statistic: adf.statistic,
            adf_p_value: adf.p_value,
            adf_is_stationary: adf.is_stationary,
            kpss_statistic: kpss.statistic,
            kpss_p_value: kpss.p_value,
            kpss_is_stationary: kpss.is_stationary,

            trend_strength,
            trend_slope,
            trend_direction,

            acf_lag1: safe(acf_lag1),
            acf_lag2: safe(acf_lag2),
            partial_acf_lag1: safe(partial_acf_lag1),

            skewness: safe(skew),
            kurtosis: safe(kurt),

            approximate_entropy: apen,
            lempel_ziv: lz,

            zero_fraction,
            is_intermittent,

            quality_score,
        }
    }

    /// Combined stationarity conclusion.
    ///
    /// Returns `true` when the ADF test rejects the unit-root null **and**
    /// the KPSS test fails to reject the stationarity null. If either test
    /// produced NaN results the method falls back to whichever test is
    /// available, and returns `false` if both are NaN.
    pub fn is_stationary(&self) -> bool {
        let adf_nan = self.adf_statistic.is_nan();
        let kpss_nan = self.kpss_statistic.is_nan();

        if adf_nan && kpss_nan {
            false
        } else if adf_nan {
            self.kpss_is_stationary
        } else if kpss_nan {
            self.adf_is_stationary
        } else {
            self.adf_is_stationary && self.kpss_is_stationary
        }
    }

    /// Human-readable multi-line summary of the profile.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "DataProfile ({} observations)\n",
            self.n_observations
        ));
        s.push_str(&format!(
            "  Basic: mean={:.4}, std={:.4}, min={:.4}, max={:.4}\n",
            self.mean, self.std_dev, self.min, self.max
        ));
        s.push_str(&format!(
            "  Quality: missing={} ({:.1}%), quality_score={:.2}\n",
            self.missing_count,
            self.missing_fraction * 100.0,
            self.quality_score,
        ));
        s.push_str(&format!(
            "  Flags: negatives={}, zeros={}, integer={}, intermittent={}\n",
            self.has_negatives, self.has_zeros, self.is_integer, self.is_intermittent,
        ));
        s.push_str(&format!(
            "  Trend: direction={}, slope={:.6}, strength={:.4}\n",
            self.trend_direction, self.trend_slope, self.trend_strength,
        ));
        s.push_str(&format!(
            "  Stationarity: ADF(stat={:.4}, p={:.4}, stationary={}), KPSS(stat={:.4}, p={:.4}, stationary={})\n",
            self.adf_statistic, self.adf_p_value, self.adf_is_stationary,
            self.kpss_statistic, self.kpss_p_value, self.kpss_is_stationary,
        ));
        s.push_str(&format!(
            "  ACF: lag1={:.4}, lag2={:.4}, PACF lag1={:.4}\n",
            self.acf_lag1, self.acf_lag2, self.partial_acf_lag1,
        ));
        s.push_str(&format!(
            "  Distribution: skewness={:.4}, kurtosis={:.4}\n",
            self.skewness, self.kurtosis,
        ));
        s.push_str(&format!(
            "  Complexity: approx_entropy={}, lempel_ziv={:.4}\n",
            match self.approximate_entropy {
                Some(v) => format!("{:.4}", v),
                None => "N/A".to_string(),
            },
            self.lempel_ziv,
        ));
        s
    }
}

impl fmt::Display for DataProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Classify a trend slope relative to the series standard deviation.
fn classify_trend(slope: f64, std_dev: f64) -> TrendDirection {
    let threshold = 0.001
        * if std_dev.is_nan() || std_dev == 0.0 {
            1.0
        } else {
            std_dev
        };
    if slope.abs() < threshold {
        TrendDirection::Flat
    } else if slope > 0.0 {
        TrendDirection::Rising
    } else {
        TrendDirection::Falling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a DataProfile from a raw slice.
    fn profile_from(values: &[f64]) -> DataProfile {
        DataProfile::from_values(values)
    }

    // ---------- 1. Constant series ----------
    #[test]
    fn profile_constant_series() {
        let series = vec![5.0; 100];
        let p = profile_from(&series);

        assert_eq!(p.n_observations, 100);
        assert!((p.mean - 5.0).abs() < 1e-10);
        assert!((p.std_dev).abs() < 1e-10);
        assert_eq!(p.trend_direction, TrendDirection::Flat);
        assert!((p.trend_slope).abs() < 1e-10);
        // Constant series should be perfectly stationary or NaN
        // ACF should be 0 for constant series (no variance)
        assert!((p.acf_lag1).abs() < 1e-10);
        // Approximate entropy should be very low or zero for constant data
        if let Some(apen) = p.approximate_entropy {
            assert!(
                apen.abs() < 0.5,
                "constant series entropy should be near zero, got {}",
                apen
            );
        }
    }

    // ---------- 2. Trending series ----------
    #[test]
    fn profile_trending_series() {
        // Strong upward trend with small noise
        let series: Vec<f64> = (0..200)
            .map(|i| i as f64 * 2.0 + ((i * 13) % 7) as f64 * 0.01)
            .collect();
        let p = profile_from(&series);

        assert_eq!(p.trend_direction, TrendDirection::Rising);
        assert!(
            p.trend_slope > 1.0,
            "expected positive slope, got {}",
            p.trend_slope
        );
        assert!(
            p.trend_strength > 0.9,
            "expected high R^2, got {}",
            p.trend_strength
        );
    }

    // ---------- 3. Series with negatives ----------
    #[test]
    fn profile_with_negatives() {
        let series = vec![-3.0, -1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0];
        let p = profile_from(&series);

        assert!(p.has_negatives, "should detect negative values");
        assert!(p.has_zeros, "should detect zeros");
    }

    // ---------- 4. Series with zeros / intermittent ----------
    #[test]
    fn profile_with_zeros() {
        // 50% zeros => intermittent
        let mut series = vec![0.0; 50];
        series.extend(vec![10.0; 50]);
        let p = profile_from(&series);

        assert!(p.has_zeros, "should detect zeros");
        assert!(p.zero_fraction > 0.1);
        assert!(p.is_intermittent, "50% zeros should be intermittent");
    }

    // ---------- 5. Integer data ----------
    #[test]
    fn profile_integer_data() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let p = profile_from(&series);

        assert!(p.is_integer, "integer-valued data should set is_integer");
    }

    // ---------- 6. Quality score (no missing) ----------
    #[test]
    fn profile_quality_score_perfect() {
        // No missing, no extreme outliers => quality near 1.0
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let p = profile_from(&series);

        assert_eq!(p.missing_count, 0);
        assert!((p.missing_fraction).abs() < 1e-10);
        assert!(
            p.quality_score > 0.85,
            "expected high quality score, got {}",
            p.quality_score
        );
    }

    // ---------- 7. Series with missing values ----------
    #[test]
    fn profile_with_missing_values() {
        let mut series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Inject 10 NaNs
        for i in 0..10 {
            series[i * 10] = f64::NAN;
        }
        let p = profile_from(&series);

        assert_eq!(p.missing_count, 10);
        assert!((p.missing_fraction - 0.1).abs() < 1e-10);
        assert!(
            p.quality_score < 1.0,
            "quality should be reduced by missing data"
        );
    }

    // ---------- 8. Short series ----------
    #[test]
    fn profile_short_series() {
        // Only 3 observations -- many features will degrade gracefully
        let series = vec![1.0, 2.0, 3.0];
        let p = profile_from(&series);

        assert_eq!(p.n_observations, 3);
        // Trend still computable
        assert!(p.trend_slope > 0.0);
        // ADF / KPSS need >= 4 observations
        assert!(p.adf_statistic.is_nan() || p.adf_statistic.is_finite());
        // Should not panic
    }

    // ---------- 9. Display / summary ----------
    #[test]
    fn profile_display() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let p = profile_from(&series);
        let text = p.summary();

        assert!(
            text.contains("DataProfile"),
            "summary should contain header"
        );
        assert!(text.contains("mean="), "summary should contain mean");
        assert!(
            text.contains("Trend:"),
            "summary should contain trend section"
        );
        assert!(
            text.contains("Stationarity:"),
            "summary should contain stationarity"
        );
        assert!(
            text.contains("Complexity:"),
            "summary should contain complexity"
        );

        // Display delegates to summary
        let display_text = format!("{}", p);
        assert_eq!(display_text, text);
    }

    // ---------- 10. Combined is_stationary ----------
    #[test]
    fn profile_is_stationary_combined() {
        // White-noise-like series (pseudo-random, stationary)
        let series: Vec<f64> = (0..200)
            .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
            .collect();
        let p = profile_from(&series);

        // The combined method should agree with both tests when both are available
        if !p.adf_statistic.is_nan() && !p.kpss_statistic.is_nan() {
            assert_eq!(
                p.is_stationary(),
                p.adf_is_stationary && p.kpss_is_stationary,
                "is_stationary() should combine ADF and KPSS"
            );
        }

        // Strong trend should generally not be stationary
        let trending: Vec<f64> = (0..200)
            .map(|i| i as f64 * 0.5 + ((i * 13) % 7) as f64 * 0.01)
            .collect();
        let p2 = profile_from(&trending);
        // At minimum, not both tests should say stationary for a strong trend
        assert!(
            !p2.is_stationary() || p2.adf_statistic.is_nan(),
            "strong trend should not be stationary"
        );
    }

    // ---------- 11. Empty series ----------
    #[test]
    fn profile_empty_series() {
        let p = profile_from(&[]);

        assert_eq!(p.n_observations, 0);
        assert_eq!(p.missing_count, 0);
        assert!((p.missing_fraction).abs() < 1e-10);
        assert!(!p.has_negatives);
        assert!(!p.has_zeros);
        // Should not panic
    }

    // ---------- 12. Falling trend ----------
    #[test]
    fn profile_falling_trend() {
        let series: Vec<f64> = (0..200).map(|i| 1000.0 - i as f64 * 3.0).collect();
        let p = profile_from(&series);

        assert_eq!(p.trend_direction, TrendDirection::Falling);
        assert!(p.trend_slope < 0.0);
    }

    // ---------- 13. TrendDirection Display ----------
    #[test]
    fn trend_direction_display() {
        assert_eq!(format!("{}", TrendDirection::Rising), "Rising");
        assert_eq!(format!("{}", TrendDirection::Falling), "Falling");
        assert_eq!(format!("{}", TrendDirection::Flat), "Flat");
    }

    // ---------- 14. Non-integer float data ----------
    #[test]
    fn profile_non_integer_data() {
        let series = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        let p = profile_from(&series);

        assert!(!p.is_integer, "fractional data should not set is_integer");
    }

    // ---------- 15. Infinite values counted as missing ----------
    #[test]
    fn profile_infinite_values() {
        let series = vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY, 5.0];
        let p = profile_from(&series);

        assert_eq!(p.missing_count, 2, "infinities should count as missing");
        assert!((p.missing_fraction - 0.4).abs() < 1e-10);
    }
}
