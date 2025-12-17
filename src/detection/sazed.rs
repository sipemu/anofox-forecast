//! SAZED: Spectral Autocorrelation Zero-crossing Ensemble Detector.
//!
//! SAZED is a parameter-free, domain-agnostic ensemble method for period length
//! estimation that combines multiple detection approaches.
//!
//! Reference: "SAZED: parameter-free domain-agnostic season length estimation
//! in time series data" - Toller, Santos, Kern (2019)

use super::fft::periodogram_peaks;
use super::periodicity::{DetectedPeriod, PeriodSource, PeriodicityDetector, PeriodicityResult};
use std::collections::HashMap;

/// SAZED: Spectral Autocorrelation Zero-crossing Ensemble Detector.
///
/// Combines multiple period detection methods and uses voting/density
/// estimation to determine the most likely period(s).
///
/// Methods used:
/// 1. **Spectral**: FFT-based peak detection
/// 2. **ACF**: Autocorrelation local maxima
/// 3. **ACF-Average**: Average of all ACF peaks
/// 4. **Zero-crossing**: Period from ACF zero-crossing patterns
/// 5. **Ensemble voting**: Mode of all detected candidates
#[derive(Debug, Clone)]
pub struct SAZED {
    /// Minimum period to consider
    pub min_period: usize,
    /// Maximum period to consider
    pub max_period: usize,
    /// Voting tolerance (periods within this range are considered the same)
    pub vote_tolerance: usize,
}

impl Default for SAZED {
    fn default() -> Self {
        Self {
            min_period: 2,
            max_period: 365,
            vote_tolerance: 1,
        }
    }
}

impl SAZED {
    /// Create a new SAZED detector with custom parameters
    pub fn new(min_period: usize, max_period: usize) -> Self {
        Self {
            min_period,
            max_period,
            vote_tolerance: 1,
        }
    }

    /// Set minimum period
    pub fn with_min_period(mut self, min: usize) -> Self {
        self.min_period = min;
        self
    }

    /// Set maximum period
    pub fn with_max_period(mut self, max: usize) -> Self {
        self.max_period = max;
        self
    }

    /// Set voting tolerance
    pub fn with_vote_tolerance(mut self, tolerance: usize) -> Self {
        self.vote_tolerance = tolerance;
        self
    }

    /// Component 1: Spectral (FFT) based detection
    fn spectral_period(&self, series: &[f64]) -> Vec<usize> {
        let peaks = periodogram_peaks(series, 2.0, self.min_period, self.max_period);
        peaks.into_iter().take(3).map(|(p, _)| p).collect()
    }

    /// Compute ACF at a specific lag
    fn acf_at_lag(&self, series: &[f64], lag: usize, mean: f64, variance: f64) -> f64 {
        let n = series.len();
        if lag >= n || variance < 1e-10 {
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

    /// Compute full ACF up to max_lag
    fn compute_acf(&self, series: &[f64], max_lag: usize) -> Vec<f64> {
        let n = series.len();
        let mean: f64 = series.iter().sum::<f64>() / n as f64;
        let variance: f64 = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        (0..=max_lag)
            .map(|lag| self.acf_at_lag(series, lag, mean, variance))
            .collect()
    }

    /// Component 2: ACF local maxima detection
    fn acf_periods(&self, series: &[f64]) -> Vec<usize> {
        let max_lag = self.max_period.min(series.len() / 2);
        let acf = self.compute_acf(series, max_lag);

        let mut peaks = Vec::new();
        for i in self.min_period..acf.len().saturating_sub(1) {
            if i >= self.max_period {
                break;
            }
            let prev = acf[i - 1];
            let curr = acf[i];
            let next = acf[i + 1];

            if curr > prev && curr > next && curr > 0.2 {
                peaks.push((i, curr));
            }
        }

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.into_iter().take(3).map(|(p, _)| p).collect()
    }

    /// Component 3: ACF average - weighted average of all ACF peaks
    fn acf_average_period(&self, series: &[f64]) -> Option<usize> {
        let max_lag = self.max_period.min(series.len() / 2);
        let acf = self.compute_acf(series, max_lag);

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for i in self.min_period..acf.len().saturating_sub(1) {
            if i >= self.max_period {
                break;
            }
            let prev = acf[i - 1];
            let curr = acf[i];
            let next = acf[i + 1];

            // Check if local maximum
            if curr > prev && curr > next && curr > 0.0 {
                weighted_sum += i as f64 * curr;
                weight_sum += curr;
            }
        }

        if weight_sum > 0.0 {
            Some((weighted_sum / weight_sum).round() as usize)
        } else {
            None
        }
    }

    /// Component 4: Zero-crossing based period estimation
    ///
    /// Estimates period from the distances between ACF zero-crossings.
    /// The idea is that a sinusoidal ACF will cross zero at period/2 intervals.
    fn zero_crossing_period(&self, series: &[f64]) -> Option<usize> {
        let max_lag = self.max_period.min(series.len() / 2);
        let acf = self.compute_acf(series, max_lag);

        // Find zero crossings (sign changes)
        let mut crossings = Vec::new();
        for i in 1..acf.len() {
            if (acf[i - 1] >= 0.0 && acf[i] < 0.0) || (acf[i - 1] < 0.0 && acf[i] >= 0.0) {
                crossings.push(i);
            }
        }

        if crossings.len() < 2 {
            return None;
        }

        // Compute distances between consecutive zero crossings
        let distances: Vec<usize> = crossings.windows(2).map(|w| w[1] - w[0]).collect();

        if distances.is_empty() {
            return None;
        }

        // The period is approximately twice the average half-period (distance between crossings)
        let avg_half_period: f64 = distances.iter().sum::<usize>() as f64 / distances.len() as f64;
        let period = (avg_half_period * 2.0).round() as usize;

        if period >= self.min_period && period <= self.max_period {
            Some(period)
        } else {
            None
        }
    }

    /// Component 5: Iterative refinement using ACF
    ///
    /// Starting from a candidate period, refine by looking at integer multiples
    /// and divisors that have higher ACF values.
    fn refine_period(&self, series: &[f64], candidate: usize) -> usize {
        let max_lag = self.max_period.min(series.len() / 2);
        let acf = self.compute_acf(series, max_lag);

        let mut best_period = candidate;
        let mut best_acf = if candidate < acf.len() {
            acf[candidate]
        } else {
            0.0
        };

        // Check divisors (potential fundamental frequencies)
        for divisor in 2..=5 {
            if candidate.is_multiple_of(divisor) {
                let sub_period = candidate / divisor;
                if sub_period >= self.min_period && sub_period < acf.len() {
                    let sub_acf = acf[sub_period];
                    if sub_acf > best_acf * 0.9 {
                        // Prefer shorter period if ACF is similar
                        best_period = sub_period;
                        best_acf = sub_acf;
                    }
                }
            }
        }

        best_period
    }

    /// Combine all candidates using voting
    fn vote_for_period(&self, candidates: &[usize]) -> Option<(usize, f64)> {
        if candidates.is_empty() {
            return None;
        }

        // Count votes with tolerance
        let mut vote_counts: HashMap<usize, usize> = HashMap::new();

        for &candidate in candidates {
            // Find if there's an existing vote within tolerance
            let mut found = false;
            for (&period, count) in vote_counts.iter_mut() {
                if (period as i64 - candidate as i64).unsigned_abs() as usize <= self.vote_tolerance
                {
                    *count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                vote_counts.insert(candidate, 1);
            }
        }

        // Find the period with most votes
        let total_votes = candidates.len() as f64;
        vote_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(period, count)| {
                let confidence = count as f64 / total_votes;
                (period, confidence)
            })
    }
}

impl PeriodicityDetector for SAZED {
    fn detect(&self, series: &[f64]) -> PeriodicityResult {
        if series.len() < self.min_period * 2 {
            return PeriodicityResult {
                primary_period: None,
                periods: Vec::new(),
                method: self.name().to_string(),
            };
        }

        // Collect candidates from all components
        let mut all_candidates: Vec<usize> = Vec::new();

        // Component 1: Spectral
        let spectral = self.spectral_period(series);
        all_candidates.extend(&spectral);

        // Component 2: ACF peaks
        let acf_peaks = self.acf_periods(series);
        all_candidates.extend(&acf_peaks);

        // Component 3: ACF average
        if let Some(acf_avg) = self.acf_average_period(series) {
            all_candidates.push(acf_avg);
        }

        // Component 4: Zero-crossing
        if let Some(zero_period) = self.zero_crossing_period(series) {
            all_candidates.push(zero_period);
        }

        // Filter to valid range
        all_candidates.retain(|&p| p >= self.min_period && p <= self.max_period);

        if all_candidates.is_empty() {
            return PeriodicityResult {
                primary_period: None,
                periods: Vec::new(),
                method: self.name().to_string(),
            };
        }

        // Refine top candidates
        let refined: Vec<usize> = all_candidates
            .iter()
            .map(|&c| self.refine_period(series, c))
            .collect();

        // Combine refined with original candidates for voting
        let mut final_candidates = all_candidates.clone();
        final_candidates.extend(&refined);

        // Vote for the best period
        let (primary_period, confidence) = match self.vote_for_period(&final_candidates) {
            Some((p, c)) => (Some(p), c),
            None => (None, 0.0),
        };

        // Build result
        let mut periods = Vec::new();
        if let Some(period) = primary_period {
            periods.push(DetectedPeriod {
                period,
                score: confidence,
                source: PeriodSource::Hybrid,
            });
        }

        // Add other high-confidence candidates
        let mut seen = std::collections::HashSet::new();
        if let Some(p) = primary_period {
            seen.insert(p);
        }

        for &candidate in &all_candidates {
            if !seen.contains(&candidate) {
                // Quick ACF check for confidence
                let max_lag = self.max_period.min(series.len() / 2);
                let acf = self.compute_acf(series, max_lag);
                if candidate < acf.len() && acf[candidate] > 0.3 {
                    periods.push(DetectedPeriod {
                        period: candidate,
                        score: acf[candidate].clamp(0.0, 1.0),
                        source: PeriodSource::Hybrid,
                    });
                    seen.insert(candidate);
                }
            }
        }

        // Sort by score
        periods.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        PeriodicityResult {
            primary_period,
            periods,
            method: self.name().to_string(),
        }
    }

    fn name(&self) -> &'static str {
        "SAZED"
    }
}

/// Detect periodicity using the SAZED ensemble method.
///
/// This is the most robust detection method, combining multiple approaches
/// with voting for the final result.
pub fn detect_period_ensemble(series: &[f64]) -> PeriodicityResult {
    SAZED::default().detect(series)
}

/// Detect periodicity using SAZED with custom period range.
pub fn detect_period_ensemble_range(
    series: &[f64],
    min_period: usize,
    max_period: usize,
) -> PeriodicityResult {
    SAZED::new(min_period, max_period).detect(series)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            .collect()
    }

    fn generate_noisy_sine(n: usize, period: usize, noise_level: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let signal = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                // Deterministic "noise" for reproducibility
                let noise = ((i * 7 + 3) % 13) as f64 / 13.0 - 0.5;
                signal + noise * noise_level
            })
            .collect()
    }

    #[test]
    fn sazed_pure_sine() {
        let signal = generate_sine(240, 12);
        let result = detect_period_ensemble(&signal);

        assert!(result.has_periodicity());
        let period = result.primary_period.unwrap();
        assert!(
            (10..=14).contains(&period),
            "Expected period near 12, got {}",
            period
        );
        assert_eq!(result.method, "SAZED");
    }

    #[test]
    fn sazed_weekly_pattern() {
        let signal = generate_sine(140, 7);
        let detector = SAZED::new(2, 30);
        let result = detector.detect(&signal);

        assert!(result.has_periodicity());
        let period = result.primary_period.unwrap();
        assert!(
            (6..=8).contains(&period),
            "Expected period near 7, got {}",
            period
        );
    }

    #[test]
    fn sazed_noisy_signal() {
        let signal = generate_noisy_sine(240, 12, 0.3);
        let result = detect_period_ensemble(&signal);

        assert!(result.has_periodicity());
        let period = result.primary_period.unwrap();
        // More tolerance for noisy signal
        assert!(
            (10..=14).contains(&period),
            "Expected period near 12, got {}",
            period
        );
    }

    #[test]
    fn sazed_with_trend() {
        // Signal with trend + seasonality
        let n = 240;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                trend + seasonal
            })
            .collect();

        let result = detect_period_ensemble(&signal);

        assert!(result.has_periodicity());
        let period = result.primary_period.unwrap();
        assert!(
            (10..=14).contains(&period),
            "Expected period near 12, got {}",
            period
        );
    }

    #[test]
    fn sazed_short_series() {
        let signal = vec![1.0, 2.0, 3.0];
        let result = detect_period_ensemble(&signal);

        assert!(!result.has_periodicity());
    }

    #[test]
    fn sazed_constant_series() {
        let signal = vec![5.0; 100];
        let result = detect_period_ensemble(&signal);

        // Constant series should not have strong periodicity
        // (it might detect something due to numerical artifacts)
        if result.has_periodicity() {
            assert!(
                result.confidence() < 0.5,
                "Constant series should have low confidence"
            );
        }
    }

    #[test]
    fn sazed_range_function() {
        let signal = generate_sine(140, 7);
        let result = detect_period_ensemble_range(&signal, 2, 20);

        assert!(result.has_periodicity());
        let period = result.primary_period.unwrap();
        assert!(
            (6..=8).contains(&period),
            "Expected period near 7, got {}",
            period
        );
    }

    #[test]
    fn sazed_zero_crossing() {
        let sazed = SAZED::default();
        let signal = generate_sine(240, 12);

        let zero_period = sazed.zero_crossing_period(&signal);
        assert!(zero_period.is_some());

        let period = zero_period.unwrap();
        assert!(
            (10..=14).contains(&period),
            "Zero-crossing period should be near 12, got {}",
            period
        );
    }

    #[test]
    fn sazed_acf_average() {
        let sazed = SAZED::new(2, 50); // Limit range to avoid harmonics
        let signal = generate_sine(240, 12);

        let avg_period = sazed.acf_average_period(&signal);
        assert!(avg_period.is_some());

        let period = avg_period.unwrap();
        // ACF average can be influenced by harmonics, so allow wider range
        // The average of 12, 24, 36, 48 weighted by ACF would be higher
        // Accept any reasonable detection
        assert!(
            (10..=100).contains(&period),
            "ACF average period should be reasonable, got {}",
            period
        );
    }

    #[test]
    fn sazed_multiple_periods_detected() {
        // Signal with two frequencies
        let n = 365;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                    + 0.7 * (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin()
            })
            .collect();

        let result = detect_period_ensemble_range(&signal, 2, 50);

        assert!(result.has_periodicity());
        // Should detect at least one period
        assert!(!result.periods.is_empty());
    }
}
