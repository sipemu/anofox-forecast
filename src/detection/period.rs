//! Robust seasonal period detection.
//!
//! Provides [`detect_periods`] for multi-period detection and
//! [`detect_dominant_period`] as a convenience for the single strongest period.
//!
//! When the `seasonal-detection` feature is enabled, detection uses the
//! SAZED ensemble algorithm from [`fdars-core`](https://crates.io/crates/fdars-core)
//! (Spectral-ACF Zero-crossing Ensemble Detection), which combines five
//! independent estimators with majority voting for high robustness.
//!
//! Without the feature, detection falls back to the built-in Welch
//! periodogram with local-maxima extraction and spectral-leakage
//! suppression.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::detection::{detect_periods, detect_dominant_period, PeriodDetectionConfig};
//!
//! // Monthly data with annual seasonality
//! let signal: Vec<f64> = (0..144)
//!     .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
//!     .collect();
//!
//! let periods = detect_periods(&signal, &PeriodDetectionConfig::default());
//! assert!(!periods.is_empty());
//! assert_eq!(periods[0].period, 12);
//!
//! let dominant = detect_dominant_period(&signal);
//! assert_eq!(dominant, Some(12));
//! ```

use super::welch_periodogram;
use crate::features::autocorrelation::autocorrelation;
use crate::seasonality::seasonal_diff::seasonal_diff_strength;

/// Configuration for period detection.
#[derive(Debug, Clone)]
pub struct PeriodDetectionConfig {
    /// Minimum period to consider (default: 2).
    pub min_period: usize,
    /// Maximum period to consider (default: `signal.len() / 3`).
    pub max_period: Option<usize>,
    /// Maximum number of periods to return (default: 5).
    pub max_periods: usize,
    /// Minimum power ratio to the mean power for a peak to be significant
    /// (default: 3.0).
    pub min_power_ratio: f64,
    /// Welch window size. `None` picks a sensible default based on series
    /// length (default: `None`).
    pub window_size: Option<usize>,
    /// Minimum seasonal differencing strength for a period to be accepted.
    /// Ranges 0–1; values near 1 mean strong seasonality. Set to 0 to
    /// disable (default). Note: strength can be low for secondary periods
    /// in multi-period signals; use this only when you need a single
    /// dominant period.
    pub min_strength: f64,
    /// Minimum number of complete cycles required in the signal for a period
    /// to be considered reliable (default: 2).
    pub min_cycles: usize,
}

impl Default for PeriodDetectionConfig {
    fn default() -> Self {
        Self {
            min_period: 2,
            max_period: None,
            max_periods: 5,
            min_power_ratio: 3.0,
            window_size: None,
            min_strength: 0.0,
            min_cycles: 2,
        }
    }
}

/// A detected seasonal period with validation metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Period {
    /// The detected period (integer number of observations per cycle).
    pub period: usize,
    /// Spectral power at this period.
    pub power: f64,
    /// Seasonal differencing strength (0–1). Values > 0.6 indicate strong
    /// seasonality; < 0.3 is weak.
    pub strength: f64,
    /// Autocorrelation at the detected lag. Positive values confirm a
    /// repeating pattern.
    pub acf: f64,
    /// Number of complete cycles of this period in the signal.
    pub n_cycles: usize,
}

/// Detect seasonal periods in a time series.
///
/// Returns up to `config.max_periods` periods sorted by strength
/// (strongest first). Each candidate is validated with:
///
/// - **Seasonal differencing strength** — measures variance explained by the
///   period; candidates below `config.min_strength` are rejected.
/// - **Minimum cycles** — candidates where the signal contains fewer than
///   `config.min_cycles` complete cycles are rejected.
/// - **ACF confirmation** — the autocorrelation at the candidate lag is
///   recorded (positive values confirm a repeating pattern).
///
/// Spectral leakage side-lobes adjacent to stronger peaks are suppressed.
///
/// With feature `seasonal-detection`: uses SAZED ensemble from fdars-core.
/// Without: uses Welch periodogram with local-maxima + leakage suppression.
pub fn detect_periods(signal: &[f64], config: &PeriodDetectionConfig) -> Vec<Period> {
    if signal.len() < 6 {
        return Vec::new();
    }

    // With fdars-core available, use SAZED for primary detection and
    // fall back to Welch for additional periods.
    #[cfg(feature = "seasonal-detection")]
    let mut candidates = detect_periods_sazed(signal, config);

    #[cfg(not(feature = "seasonal-detection"))]
    let mut candidates = detect_periods_welch(signal, config);

    // Validate and enrich each candidate, then filter.
    validate_periods(signal, &mut candidates, config);
    candidates
}

/// Convenience: detect the single dominant period.
///
/// Returns `None` if no significant period is found.
pub fn detect_dominant_period(signal: &[f64]) -> Option<usize> {
    let config = PeriodDetectionConfig {
        max_periods: 1,
        ..Default::default()
    };
    detect_periods(signal, &config).first().map(|p| p.period)
}

// ── Validation ──────────────────────────────────────────────────────────────

/// Validate detected periods: compute strength / ACF / n_cycles,
/// then remove candidates that fail the thresholds.
fn validate_periods(signal: &[f64], periods: &mut Vec<Period>, config: &PeriodDetectionConfig) {
    let n = signal.len();

    for p in periods.iter_mut() {
        p.n_cycles = n / p.period;
        p.strength = seasonal_diff_strength(signal, p.period);
        p.acf = autocorrelation(signal, p.period);
    }

    periods.retain(|p| p.n_cycles >= config.min_cycles && p.strength >= config.min_strength);

    // Sort by strength first (most meaningful), then by power as tiebreaker.
    periods.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap()
            .then_with(|| b.power.partial_cmp(&a.power).unwrap())
    });
    periods.truncate(config.max_periods);
}

// ── fdars-core SAZED backend ────────────────────────────────────────────────

#[cfg(feature = "seasonal-detection")]
fn detect_periods_sazed(signal: &[f64], config: &PeriodDetectionConfig) -> Vec<Period> {
    let n = signal.len();
    let argvals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let max_period = config.max_period.unwrap_or(n / 3);

    // Run SAZED ensemble — most robust single-period estimator.
    let sazed_result = fdars_core::seasonal::sazed(signal, &argvals, None);

    let mut periods = Vec::new();

    // Accept the SAZED consensus if it looks valid.
    let sazed_period = sazed_result.period.round() as usize;
    if sazed_result.confidence > 0.0
        && sazed_period >= config.min_period
        && sazed_period <= max_period
    {
        periods.push(Period {
            period: sazed_period,
            power: sazed_result.confidence,
            strength: 0.0,
            acf: 0.0,
            n_cycles: 0,
        });
    }

    // Also run CFD-Autoperiod for additional multi-period detection.
    let cfd_result =
        fdars_core::seasonal::cfd_autoperiod(signal, &argvals, Some(0.1), Some(1));

    for (&p, &conf) in cfd_result.periods.iter().zip(cfd_result.confidences.iter()) {
        let p_int = p.round() as usize;
        if p_int < config.min_period || p_int > max_period || conf <= 0.0 {
            continue;
        }
        // Skip if too close to an already-accepted period (within ±20%).
        let dominated = periods.iter().any(|existing| {
            let ratio = p_int as f64 / existing.period as f64;
            (0.8..=1.2).contains(&ratio)
        });
        if !dominated {
            periods.push(Period {
                period: p_int,
                power: conf,
                strength: 0.0,
                acf: 0.0,
                n_cycles: 0,
            });
        }
    }

    // Supplement with Welch-based detection for additional periods
    // that SAZED/CFD may have missed (SAZED is single-period).
    if periods.len() < config.max_periods {
        let welch_periods = detect_periods_welch(signal, config);
        for wp in welch_periods {
            if periods.len() >= config.max_periods {
                break;
            }
            // Skip if too close to an already-accepted period (within ±20%).
            let dominated = periods.iter().any(|existing| {
                let ratio = wp.period as f64 / existing.period as f64;
                (0.8..=1.2).contains(&ratio)
            });
            if !dominated {
                periods.push(wp);
            }
        }
    }

    // If nothing was found at all, fall back entirely to Welch.
    if periods.is_empty() {
        return detect_periods_welch(signal, config);
    }

    periods.sort_by(|a, b| b.power.partial_cmp(&a.power).unwrap());
    periods.truncate(config.max_periods);
    periods
}

// ── Welch periodogram backend ───────────────────────────────────────────────

fn detect_periods_welch(signal: &[f64], config: &PeriodDetectionConfig) -> Vec<Period> {
    let n = signal.len();
    let max_period = config.max_period.unwrap_or(n / 3);

    // Pick a sensible window size: largest power of 2 ≤ n, capped at 256,
    // but at least 32.
    let window_size = config.window_size.unwrap_or_else(|| {
        let mut w = 32;
        while w * 2 <= n && w < 256 {
            w *= 2;
        }
        w
    });

    let raw = welch_periodogram(signal, window_size, 0.5);
    if raw.is_empty() {
        return Vec::new();
    }

    // Build a period → power map, filtered to [min_period, max_period].
    let mut spectrum: Vec<(usize, f64)> = raw
        .into_iter()
        .filter(|&(p, _)| p >= config.min_period && p <= max_period)
        .collect();

    if spectrum.is_empty() {
        return Vec::new();
    }

    // Sort by period ascending for local-maxima detection.
    spectrum.sort_by_key(|&(p, _)| p);

    // ── Step 1: Extract local maxima ────────────────────────────────────
    // A peak at index i is a local maximum if power[i] > power[i-1] and
    // power[i] > power[i+1].
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 0..spectrum.len() {
        let (period, power) = spectrum[i];
        let left = if i > 0 { spectrum[i - 1].1 } else { 0.0 };
        let right = if i + 1 < spectrum.len() {
            spectrum[i + 1].1
        } else {
            0.0
        };
        if power > left && power > right {
            peaks.push((period, power));
        }
    }

    if peaks.is_empty() {
        return Vec::new();
    }

    // ── Step 2: Filter by minimum power ratio ───────────────────────────
    let mean_power = spectrum.iter().map(|&(_, p)| p).sum::<f64>() / spectrum.len() as f64;
    let threshold = mean_power * config.min_power_ratio;
    peaks.retain(|&(_, power)| power >= threshold);

    // ── Step 3: Suppress spectral leakage ───────────────────────────────
    // Sort by power descending. Greedily keep the strongest peak and
    // suppress any weaker peak whose period is within ±20% of a kept peak.
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut kept: Vec<(usize, f64)> = Vec::new();
    for (period, power) in peaks {
        let is_leakage = kept.iter().any(|&(kept_p, _)| {
            let ratio = period as f64 / kept_p as f64;
            (0.8..=1.2).contains(&ratio)
        });
        if !is_leakage {
            kept.push((period, power));
        }
    }

    kept.truncate(config.max_periods);
    kept
        .into_iter()
        .map(|(period, power)| Period { period, power, strength: 0.0, acf: 0.0, n_cycles: 0 })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, period: usize) -> Vec<f64> {
        let two_pi = 2.0 * std::f64::consts::PI;
        (0..n)
            .map(|i| (two_pi * i as f64 / period as f64).sin())
            .collect()
    }

    fn airpassengers() -> Vec<f64> {
        vec![
            112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0,
            118.0, 115.0, 126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0,
            114.0, 140.0, 145.0, 150.0, 178.0, 163.0, 172.0, 178.0, 199.0, 199.0, 184.0,
            162.0, 146.0, 166.0, 171.0, 180.0, 193.0, 181.0, 183.0, 218.0, 230.0, 242.0,
            209.0, 191.0, 172.0, 194.0, 196.0, 196.0, 236.0, 235.0, 229.0, 243.0, 264.0,
            272.0, 237.0, 211.0, 180.0, 201.0, 204.0, 188.0, 235.0, 227.0, 234.0, 264.0,
            302.0, 293.0, 259.0, 229.0, 203.0, 229.0, 242.0, 233.0, 267.0, 269.0, 270.0,
            315.0, 364.0, 347.0, 312.0, 274.0, 237.0, 278.0, 284.0, 277.0, 317.0, 313.0,
            318.0, 374.0, 413.0, 405.0, 355.0, 306.0, 271.0, 306.0, 315.0, 301.0, 356.0,
            348.0, 355.0, 422.0, 465.0, 467.0, 404.0, 347.0, 305.0, 336.0, 340.0, 318.0,
            362.0, 348.0, 363.0, 435.0, 491.0, 505.0, 404.0, 359.0, 310.0, 337.0, 360.0,
            342.0, 406.0, 396.0, 420.0, 472.0, 548.0, 559.0, 463.0, 407.0, 362.0, 405.0,
            417.0, 391.0, 419.0, 461.0, 472.0, 535.0, 622.0, 606.0, 508.0, 461.0, 390.0,
            432.0,
        ]
    }

    // ── Issue #17 reproduction ──────────────────────────────────────────

    #[test]
    fn airpassengers_no_leakage_period_13() {
        let data = airpassengers();
        let periods = detect_periods(&data, &PeriodDetectionConfig::default());

        // Period 12 should be detected.
        assert!(
            periods.iter().any(|p| p.period == 12),
            "Should detect period 12, got {:?}",
            periods
        );

        // Period 13 (spectral leakage from 12) should NOT appear.
        assert!(
            !periods.iter().any(|p| p.period == 13),
            "Period 13 is spectral leakage and should be suppressed, got {:?}",
            periods
        );
    }

    #[test]
    fn airpassengers_dominant_period_is_12() {
        let data = airpassengers();
        let dominant = detect_dominant_period(&data);
        assert_eq!(dominant, Some(12));
    }

    // ── Standard seasonal periods ───────────────────────────────────────

    #[test]
    fn detects_period_7() {
        let signal: Vec<f64> = (0..365)
            .map(|i| 50.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
            .collect();
        assert_eq!(detect_dominant_period(&signal), Some(7));
    }

    #[test]
    fn detects_period_12() {
        let signal: Vec<f64> = (0..144)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        assert_eq!(detect_dominant_period(&signal), Some(12));
    }

    #[test]
    fn detects_period_24() {
        let signal: Vec<f64> = (0..720)
            .map(|i| 20.0 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin())
            .collect();
        assert_eq!(detect_dominant_period(&signal), Some(24));
    }

    #[test]
    fn detects_period_52() {
        let signal: Vec<f64> = (0..260)
            .map(|i| 30.0 + 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 52.0).sin())
            .collect();
        assert_eq!(detect_dominant_period(&signal), Some(52));
    }

    // ── Multiple periods ────────────────────────────────────────────────

    #[test]
    fn detects_period_12_and_6() {
        let signal: Vec<f64> = (0..240)
            .map(|i| {
                100.0
                    + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                    + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 6.0).sin()
            })
            .collect();
        let periods = detect_periods(&signal, &PeriodDetectionConfig::default());
        let period_vals: Vec<usize> = periods.iter().map(|p| p.period).collect();
        assert!(
            period_vals.contains(&12),
            "Should detect period 12, got {:?}",
            period_vals
        );
        assert!(
            period_vals.contains(&6),
            "Should detect period 6, got {:?}",
            period_vals
        );
    }

    // ── With trend and noise ────────────────────────────────────────────

    #[test]
    fn detects_period_with_trend_and_noise() {
        let signal: Vec<f64> = (0..240)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                let noise = ((i * 7 + 3) % 11) as f64 * 0.3 - 1.5;
                50.0 + trend + seasonal + noise
            })
            .collect();
        assert_eq!(detect_dominant_period(&signal), Some(12));
    }

    // ── Leakage suppression ─────────────────────────────────────────────

    #[test]
    fn no_adjacent_leakage_peaks() {
        // Pure sine at period 12 — should get only period 12, not 11 or 13.
        let signal = sine(256, 12);
        let periods = detect_periods(&signal, &PeriodDetectionConfig::default());

        for p in &periods {
            if p.period != 12 {
                let ratio = p.period as f64 / 12.0;
                assert!(
                    !(0.8..=1.2).contains(&ratio),
                    "Period {} is leakage from 12 and should be suppressed",
                    p.period
                );
            }
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn short_signal_returns_empty() {
        assert!(detect_periods(&[1.0, 2.0, 3.0], &PeriodDetectionConfig::default()).is_empty());
    }

    #[test]
    fn constant_signal_returns_empty() {
        let signal = vec![5.0; 100];
        assert!(detect_periods(&signal, &PeriodDetectionConfig::default()).is_empty());
    }

    #[test]
    fn config_min_period() {
        let signal: Vec<f64> = (0..144)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 4.0).sin())
            .collect();
        let config = PeriodDetectionConfig {
            min_period: 6,
            ..Default::default()
        };
        let periods = detect_periods(&signal, &config);
        for p in &periods {
            assert!(p.period >= 6, "Period {} is below min_period 6", p.period);
        }
    }

    #[test]
    fn config_max_period() {
        let signal: Vec<f64> = (0..240)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let config = PeriodDetectionConfig {
            max_period: Some(20),
            ..Default::default()
        };
        let periods = detect_periods(&signal, &config);
        for p in &periods {
            assert!(p.period <= 20, "Period {} is above max_period 20", p.period);
        }
    }

    // ── Validation fields ───────────────────────────────────────────────

    #[test]
    fn validation_fields_populated() {
        let signal: Vec<f64> = (0..144)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let periods = detect_periods(&signal, &PeriodDetectionConfig::default());

        let p12 = periods.iter().find(|p| p.period == 12).unwrap();
        assert!(p12.strength > 0.5, "strength should be high, got {}", p12.strength);
        assert!(p12.acf > 0.5, "acf(12) should be positive, got {}", p12.acf);
        assert_eq!(p12.n_cycles, 12); // 144 / 12
    }

    #[test]
    fn min_strength_filters_weak_periods() {
        // Period 12 with noise — add a spurious candidate by setting
        // min_power_ratio low, then use min_strength to filter.
        let signal: Vec<f64> = (0..144)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let config = PeriodDetectionConfig {
            min_strength: 0.5,
            ..Default::default()
        };
        let periods = detect_periods(&signal, &config);
        for p in &periods {
            assert!(
                p.strength >= 0.5,
                "Period {} has strength {} below threshold 0.5",
                p.period, p.strength
            );
        }
    }

    #[test]
    fn min_cycles_filters_unreliable_periods() {
        // 30 observations with period 12 → only 2.5 cycles.
        // With min_cycles=3, period 12 should be rejected.
        let signal: Vec<f64> = (0..30)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let config = PeriodDetectionConfig {
            min_cycles: 3,
            ..Default::default()
        };
        let periods = detect_periods(&signal, &config);
        for p in &periods {
            assert!(
                p.n_cycles >= 3,
                "Period {} has only {} cycles, below min_cycles=3",
                p.period, p.n_cycles
            );
        }
    }

    #[test]
    fn dominant_period_returns_strongest() {
        // Ensure detect_dominant_period returns the period with highest strength.
        let signal: Vec<f64> = (0..240)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let dominant = detect_dominant_period(&signal);
        assert_eq!(dominant, Some(12));

        let all = detect_periods(&signal, &PeriodDetectionConfig::default());
        assert_eq!(all[0].period, 12);
    }
}
