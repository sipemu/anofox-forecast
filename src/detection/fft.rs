//! Spectral analysis utilities.
//!
//! Provides Welch's periodogram for robust spectral estimation.

/// Compute the periodogram (power spectral density) of a signal.
///
/// Evaluates the DFT directly at each target integer period using the
/// Goertzel-style approach.  This avoids bin-to-period rounding errors
/// that plague FFT-based periodograms, because the spectral power is
/// computed at exactly the frequency `1/period`.
///
/// Returns (period, power) pairs sorted by period (largest first).
fn periodogram(signal: &[f64]) -> Vec<(usize, f64)> {
    let n = signal.len();
    if n < 4 {
        return Vec::new();
    }

    let n_f64 = n as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut result = Vec::with_capacity(n / 2);

    for period in 2..=n / 2 {
        let omega = two_pi / period as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &x) in signal.iter().enumerate() {
            let angle = omega * i as f64;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        let power = (re * re + im * im) / n_f64;
        result.push((period, power));
    }

    result.sort_by(|a, b| b.0.cmp(&a.0));
    result
}

/// Compute Welch's periodogram for more robust spectral estimation.
///
/// Uses overlapping windows with Hann windowing to reduce variance
/// in the power estimate. This method is more robust to noise than
/// a standard periodogram.
///
/// # Arguments
/// * `signal` - Input time series
/// * `window_size` - Size of each segment (should be power of 2 for efficiency)
/// * `overlap` - Overlap ratio between segments (0.0 to 0.9, typically 0.5)
///
/// # Returns
/// Vector of (period, power) tuples sorted by period (largest first)
///
/// # Example
/// ```
/// use anofox_forecast::detection::welch_periodogram;
///
/// let signal: Vec<f64> = (0..256)
///     .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
///     .collect();
///
/// let psd = welch_periodogram(&signal, 64, 0.5);
/// // Find the dominant period
/// if let Some((period, _)) = psd.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
///     println!("Dominant period: {}", period);
/// }
/// ```
pub fn welch_periodogram(signal: &[f64], window_size: usize, overlap: f64) -> Vec<(usize, f64)> {
    let n = signal.len();
    if n < window_size || window_size < 4 {
        return periodogram(signal);
    }

    let overlap = overlap.clamp(0.0, 0.9);
    let hop = ((1.0 - overlap) * window_size as f64).ceil() as usize;
    let hop = hop.max(1);

    // Collect segments
    let mut accumulated_psd: std::collections::HashMap<usize, (f64, usize)> =
        std::collections::HashMap::new();

    let mut start = 0;
    while start + window_size <= n {
        let segment = &signal[start..start + window_size];

        // Subtract segment mean before windowing to remove DC component
        let seg_mean = segment.iter().sum::<f64>() / window_size as f64;

        // Apply Hann window to demeaned segment
        let windowed: Vec<f64> = segment
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = 0.5
                    * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / window_size as f64).cos());
                (x - seg_mean) * w
            })
            .collect();

        // Compute periodogram for this segment
        let psd = periodogram(&windowed);

        // Accumulate
        for (period, power) in psd {
            let entry = accumulated_psd.entry(period).or_insert((0.0, 0));
            entry.0 += power;
            entry.1 += 1;
        }

        start += hop;
    }

    // Average the accumulated powers
    let mut result: Vec<(usize, f64)> = accumulated_psd
        .into_iter()
        .map(|(period, (sum, count))| (period, sum / count as f64))
        .collect();

    result.sort_by(|a, b| b.0.cmp(&a.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            .collect()
    }

    #[test]
    fn welch_periodogram_basic() {
        let signal = generate_sine(256, 12);

        // window_size=128 has enough frequency resolution to resolve period 12
        let psd = welch_periodogram(&signal, 128, 0.5);
        assert!(!psd.is_empty());
        let peak = psd.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (period, _) = peak.unwrap();
        assert_eq!(*period, 12, "Expected dominant period 12, got {}", period);

        // window_size=64: frequency resolution (1/64) can't distinguish
        // period 12 from 13, so accept either
        let psd = welch_periodogram(&signal, 64, 0.5);
        let peak = psd.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (period, _) = peak.unwrap();
        assert!(
            (11..=13).contains(period),
            "Expected period near 12 with window_size=64, got {}",
            period
        );
    }

    #[test]
    fn welch_short_signal() {
        let signal = generate_sine(32, 8);
        let psd = welch_periodogram(&signal, 64, 0.5);

        // Should fall back to regular periodogram
        assert!(!psd.is_empty());
    }

    #[test]
    fn welch_overlap_values() {
        let signal = generate_sine(256, 16);

        // Different overlaps should all work
        for overlap in [0.0, 0.25, 0.5, 0.75] {
            let psd = welch_periodogram(&signal, 64, overlap);
            assert!(!psd.is_empty(), "Failed with overlap {}", overlap);
        }
    }

    #[test]
    fn welch_finds_multiple_periods() {
        // Signal with two frequencies
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()
            })
            .collect();

        let psd = welch_periodogram(&signal, 128, 0.5);

        // Sort by power (descending) and take top 10 periods
        let mut by_power = psd.clone();
        by_power.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_periods: Vec<usize> = by_power.iter().take(10).map(|(p, _)| *p).collect();

        let has_16 = top_periods.iter().any(|p| (14..=18).contains(p));
        let has_32 = top_periods.iter().any(|p| (28..=36).contains(p));

        assert!(
            has_16 || has_32,
            "Should detect at least one period, got {:?}",
            top_periods
        );
    }

    /// Helper: find the dominant period from a welch_periodogram result.
    fn dominant_period(psd: &[(usize, f64)]) -> usize {
        psd.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    /// Helper: find the top-k periods by power.
    fn top_k_periods(psd: &[(usize, f64)], k: usize) -> Vec<usize> {
        let mut by_power = psd.to_vec();
        by_power.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        by_power.iter().take(k).map(|(p, _)| *p).collect()
    }

    #[test]
    fn seasonal_monthly_period_12() {
        // Monthly data: 144 observations (12 years), true period = 12
        let signal: Vec<f64> = (0..144)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
            .collect();
        let psd = welch_periodogram(&signal, 128, 0.5);
        assert_eq!(dominant_period(&psd), 12);
    }

    #[test]
    fn seasonal_quarterly_period_4() {
        let signal = generate_sine(200, 4);
        let psd = welch_periodogram(&signal, 128, 0.5);
        assert_eq!(dominant_period(&psd), 4);
    }

    #[test]
    fn seasonal_weekly_period_7() {
        let signal: Vec<f64> = (0..365)
            .map(|i| 50.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
            .collect();
        let psd = welch_periodogram(&signal, 128, 0.5);
        assert_eq!(dominant_period(&psd), 7);
    }

    #[test]
    fn seasonal_period_24() {
        // Hourly data with daily seasonality (period = 24)
        let signal: Vec<f64> = (0..720)
            .map(|i| 20.0 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin())
            .collect();
        let psd = welch_periodogram(&signal, 256, 0.5);
        assert_eq!(dominant_period(&psd), 24);
    }

    #[test]
    fn seasonal_period_52() {
        // Weekly data with annual seasonality (period = 52)
        let signal: Vec<f64> = (0..260)
            .map(|i| 30.0 + 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 52.0).sin())
            .collect();
        let psd = welch_periodogram(&signal, 128, 0.5);
        assert_eq!(dominant_period(&psd), 52);
    }

    #[test]
    fn seasonal_multiple_12_and_6() {
        // Monthly data with semi-annual and annual seasonality
        let signal: Vec<f64> = (0..240)
            .map(|i| {
                100.0
                    + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                    + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 6.0).sin()
            })
            .collect();
        let psd = welch_periodogram(&signal, 128, 0.5);
        let top = top_k_periods(&psd, 5);
        assert!(top.contains(&12), "Should detect period 12, top={:?}", top);
        assert!(top.contains(&6), "Should detect period 6, top={:?}", top);
    }

    #[test]
    fn seasonal_multiple_12_and_4() {
        // Monthly + quarterly seasonality
        let signal: Vec<f64> = (0..240)
            .map(|i| {
                50.0 + 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                    + 4.0 * (2.0 * std::f64::consts::PI * i as f64 / 4.0).sin()
            })
            .collect();
        let psd = welch_periodogram(&signal, 128, 0.5);
        let top = top_k_periods(&psd, 5);
        assert!(top.contains(&12), "Should detect period 12, top={:?}", top);
        assert!(top.contains(&4), "Should detect period 4, top={:?}", top);
    }

    #[test]
    fn seasonal_with_trend_and_noise() {
        // Period 12 signal + linear trend + noise
        let signal: Vec<f64> = (0..240)
            .map(|i| {
                let trend = 0.1 * i as f64;
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
                let noise = ((i * 7 + 3) % 11) as f64 * 0.3 - 1.5; // deterministic pseudo-noise
                50.0 + trend + seasonal + noise
            })
            .collect();
        let psd = welch_periodogram(&signal, 128, 0.5);
        assert_eq!(dominant_period(&psd), 12);
    }
}
