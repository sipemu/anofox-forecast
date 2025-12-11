//! Seasonality detection utilities.
//!
//! Provides methods to detect the presence and period of seasonality in time series.

/// Result of seasonality detection.
#[derive(Debug, Clone)]
pub struct SeasonalityResult {
    /// Whether seasonality was detected.
    pub detected: bool,
    /// The detected seasonal period (if any).
    pub period: Option<usize>,
    /// Strength of seasonality (0 to 1).
    pub strength: f64,
    /// Candidate periods and their scores.
    pub candidates: Vec<(usize, f64)>,
}

impl SeasonalityResult {
    /// Check if strong seasonality was detected.
    pub fn is_strong(&self) -> bool {
        self.strength >= 0.7
    }

    /// Check if moderate or stronger seasonality was detected.
    pub fn is_moderate(&self) -> bool {
        self.strength >= 0.4
    }
}

/// Configuration for seasonality detection.
#[derive(Debug, Clone)]
pub struct SeasonalityConfig {
    /// Maximum period to consider.
    pub max_period: usize,
    /// Minimum period to consider.
    pub min_period: usize,
    /// Threshold for detection (0.0 to 1.0).
    pub threshold: f64,
}

impl Default for SeasonalityConfig {
    fn default() -> Self {
        Self {
            max_period: 365,
            min_period: 2,
            threshold: 0.3,
        }
    }
}

impl SeasonalityConfig {
    /// Set maximum period.
    pub fn with_max_period(mut self, max: usize) -> Self {
        self.max_period = max;
        self
    }

    /// Set minimum period.
    pub fn with_min_period(mut self, min: usize) -> Self {
        self.min_period = min;
        self
    }

    /// Set detection threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

/// Detect seasonality in a time series using autocorrelation.
pub fn detect_seasonality(series: &[f64], config: &SeasonalityConfig) -> SeasonalityResult {
    let n = series.len();

    if n < config.min_period * 2 {
        return SeasonalityResult {
            detected: false,
            period: None,
            strength: 0.0,
            candidates: Vec::new(),
        };
    }

    let max_lag = config.max_period.min(n / 2);
    let min_lag = config.min_period;

    // Compute autocorrelations
    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let variance: f64 = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-10 {
        return SeasonalityResult {
            detected: false,
            period: None,
            strength: 0.0,
            candidates: Vec::new(),
        };
    }

    let mut acf_values = Vec::new();
    for lag in min_lag..=max_lag {
        let acf = autocorrelation(series, lag, mean, variance);
        acf_values.push((lag, acf));
    }

    // Find local maxima in ACF
    let mut candidates: Vec<(usize, f64)> = Vec::new();

    for i in 1..acf_values.len() - 1 {
        let (lag, acf) = acf_values[i];
        let prev = acf_values[i - 1].1;
        let next = acf_values[i + 1].1;

        // Local maximum
        if acf > prev && acf > next && acf > config.threshold {
            candidates.push((lag, acf));
        }
    }

    // Sort by ACF value (highest first)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Determine best period
    let (detected, period, strength) = if let Some(&(best_period, best_acf)) = candidates.first() {
        (true, Some(best_period), best_acf.min(1.0).max(0.0))
    } else {
        (false, None, 0.0)
    };

    SeasonalityResult {
        detected,
        period,
        strength,
        candidates,
    }
}

/// Detect seasonality with default configuration.
pub fn detect_seasonality_auto(series: &[f64]) -> SeasonalityResult {
    detect_seasonality(series, &SeasonalityConfig::default())
}

/// Compute autocorrelation at a specific lag.
fn autocorrelation(series: &[f64], lag: usize, mean: f64, variance: f64) -> f64 {
    let n = series.len();
    if lag >= n {
        return 0.0;
    }

    let covariance: f64 = series
        .iter()
        .take(n - lag)
        .zip(series.iter().skip(lag))
        .map(|(x1, x2)| (x1 - mean) * (x2 - mean))
        .sum::<f64>()
        / n as f64;

    covariance / variance
}

/// Estimate the strength of seasonality.
/// Returns a value between 0 and 1.
pub fn seasonal_strength(trend: &[f64], seasonal: &[f64], remainder: &[f64]) -> f64 {
    let var_remainder = variance(remainder);
    let seasonal_plus_remainder: Vec<f64> = seasonal
        .iter()
        .zip(remainder.iter())
        .map(|(s, r)| s + r)
        .collect();
    let var_sr = variance(&seasonal_plus_remainder);

    if var_sr < 1e-10 {
        return 0.0;
    }

    (1.0 - var_remainder / var_sr).max(0.0).min(1.0)
}

/// Compute variance.
fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_seasonal_series(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                10.0 * ((2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            })
            .collect()
    }

    #[test]
    fn detect_seasonality_with_clear_pattern() {
        let period = 12;
        let series = generate_seasonal_series(120, period);

        let result = detect_seasonality_auto(&series);

        assert!(result.detected);
        assert!(result.period.is_some());
        // Should detect the period (or a harmonic)
        let detected_period = result.period.unwrap();
        assert!(
            detected_period == period || period % detected_period == 0,
            "Should detect period {} but got {}",
            period,
            detected_period
        );
    }

    #[test]
    fn detect_seasonality_no_pattern() {
        // Random-ish data with no clear seasonality
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.123).sin() * i as f64 % 7.0).collect();

        let config = SeasonalityConfig::default().with_threshold(0.5);
        let result = detect_seasonality(&series, &config);

        // Should not detect strong seasonality
        assert!(result.strength < 0.5);
    }

    #[test]
    fn detect_seasonality_constant_series() {
        let series = vec![5.0; 100];

        let result = detect_seasonality_auto(&series);

        assert!(!result.detected);
        assert!(result.period.is_none());
    }

    #[test]
    fn detect_seasonality_short_series() {
        let series = vec![1.0, 2.0, 3.0];

        let result = detect_seasonality_auto(&series);

        assert!(!result.detected);
    }

    #[test]
    fn detect_seasonality_with_config() {
        let period = 7;
        let series = generate_seasonal_series(100, period);

        let config = SeasonalityConfig::default()
            .with_min_period(3)
            .with_max_period(20)
            .with_threshold(0.2);

        let result = detect_seasonality(&series, &config);

        assert!(result.detected);
    }

    #[test]
    fn detect_seasonality_result_methods() {
        let result = SeasonalityResult {
            detected: true,
            period: Some(12),
            strength: 0.8,
            candidates: vec![(12, 0.8), (24, 0.6)],
        };

        assert!(result.is_strong());
        assert!(result.is_moderate());

        let weak = SeasonalityResult {
            detected: true,
            period: Some(12),
            strength: 0.3,
            candidates: vec![],
        };

        assert!(!weak.is_strong());
        assert!(!weak.is_moderate());
    }

    #[test]
    fn seasonal_strength_calculation() {
        // Simulated decomposition
        let trend = vec![10.0; 20];
        let seasonal: Vec<f64> = (0..20)
            .map(|i| 5.0 * ((2.0 * std::f64::consts::PI * i as f64 / 4.0).sin()))
            .collect();
        let remainder = vec![0.1; 20];

        let strength = seasonal_strength(&trend, &seasonal, &remainder);

        // Strong seasonal component should give high strength
        assert!(strength > 0.9);
    }

    #[test]
    fn config_default() {
        let config = SeasonalityConfig::default();
        assert_eq!(config.max_period, 365);
        assert_eq!(config.min_period, 2);
        assert!((config.threshold - 0.3).abs() < 1e-10);
    }

    #[test]
    fn detect_weekly_seasonality() {
        let period = 7;
        // Two full years of weekly data
        let series = generate_seasonal_series(52 * 2, period);

        let config = SeasonalityConfig::default()
            .with_min_period(2)
            .with_max_period(14);

        let result = detect_seasonality(&series, &config);

        assert!(result.detected);
    }
}
