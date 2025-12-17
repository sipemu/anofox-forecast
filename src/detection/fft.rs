//! FFT utilities for periodicity detection.
//!
//! Provides Fast Fourier Transform based tools for spectral analysis
//! and periodogram computation.

use rustfft::{num_complex::Complex64, FftPlanner};

/// Compute the FFT of a real-valued signal.
///
/// Returns the complex frequency domain representation.
/// Only returns the first half (positive frequencies) since
/// the input is real-valued and the spectrum is symmetric.
///
/// # Arguments
/// * `signal` - Input time series (real values)
///
/// # Returns
/// Complex frequency components for frequencies 0 to N/2
pub fn fft_real(signal: &[f64]) -> Vec<Complex64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    // Convert to complex
    let mut buffer: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Return only positive frequencies (0 to N/2)
    buffer.truncate(n / 2 + 1);
    buffer
}

/// Compute the periodogram (power spectral density) of a signal.
///
/// Returns (period, power) pairs sorted by period, where power is the
/// squared magnitude of the FFT normalized by the signal length.
///
/// # Arguments
/// * `signal` - Input time series
///
/// # Returns
/// Vector of (period, power) tuples for periods >= 2
pub fn periodogram(signal: &[f64]) -> Vec<(usize, f64)> {
    let n = signal.len();
    if n < 4 {
        return Vec::new();
    }

    let fft_result = fft_real(signal);
    let n_f64 = n as f64;

    // Convert to power spectral density
    // Skip DC component (k=0) and frequencies beyond Nyquist
    let mut result = Vec::with_capacity(n / 2);

    for (k, complex) in fft_result.iter().enumerate().skip(1) {
        // Period = N / frequency_index
        let period = n / k;
        if period < 2 {
            break;
        }

        // Power = |X[k]|^2 / N
        let power = (complex.re * complex.re + complex.im * complex.im) / n_f64;

        result.push((period, power));
    }

    // Sort by period (largest first for consistency)
    result.sort_by(|a, b| b.0.cmp(&a.0));
    result
}

/// Compute the periodogram with peak detection.
///
/// Finds significant peaks in the power spectrum above the noise floor.
///
/// # Arguments
/// * `signal` - Input time series
/// * `threshold` - Multiplier for noise floor (e.g., 3.0 means peaks must be 3x above noise)
/// * `min_period` - Minimum period to consider
/// * `max_period` - Maximum period to consider
///
/// # Returns
/// Vector of (period, power) tuples for detected peaks, sorted by power (highest first)
pub fn periodogram_peaks(
    signal: &[f64],
    threshold: f64,
    min_period: usize,
    max_period: usize,
) -> Vec<(usize, f64)> {
    let psd = periodogram(signal);
    if psd.is_empty() {
        return Vec::new();
    }

    // Filter by period range
    let filtered: Vec<(usize, f64)> = psd
        .iter()
        .filter(|(p, _)| *p >= min_period && *p <= max_period)
        .copied()
        .collect();

    if filtered.is_empty() {
        return Vec::new();
    }

    // Estimate noise floor as median power
    let mut powers: Vec<f64> = filtered.iter().map(|(_, p)| *p).collect();
    powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_floor = if powers.len().is_multiple_of(2) {
        (powers[powers.len() / 2 - 1] + powers[powers.len() / 2]) / 2.0
    } else {
        powers[powers.len() / 2]
    };

    // Find peaks above threshold
    let peak_threshold = noise_floor * threshold;
    let mut peaks: Vec<(usize, f64)> = filtered
        .iter()
        .filter(|(_, power)| *power > peak_threshold)
        .copied()
        .collect();

    // Sort by power (highest first)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    peaks
}

/// Compute Welch's periodogram for more robust spectral estimation.
///
/// Uses overlapping windows to reduce variance in the power estimate.
///
/// # Arguments
/// * `signal` - Input time series
/// * `window_size` - Size of each segment (should be power of 2 for efficiency)
/// * `overlap` - Overlap ratio between segments (0.0 to 0.9, typically 0.5)
///
/// # Returns
/// Vector of (period, power) tuples
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

        // Apply Hann window
        let windowed: Vec<f64> = segment
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = 0.5
                    * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / window_size as f64).cos());
                x * w
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

/// Find the frequency index corresponding to a given period.
///
/// # Arguments
/// * `period` - The period to look up
/// * `n` - The length of the signal
///
/// # Returns
/// The frequency index k such that period ≈ n/k
pub fn period_to_frequency_index(period: usize, n: usize) -> usize {
    if period == 0 {
        return 0;
    }
    n / period
}

/// Convert a frequency index to the corresponding period.
///
/// # Arguments
/// * `freq_index` - The frequency index (k in the FFT)
/// * `n` - The length of the signal
///
/// # Returns
/// The period corresponding to this frequency
pub fn frequency_index_to_period(freq_index: usize, n: usize) -> usize {
    if freq_index == 0 {
        return n; // DC component corresponds to infinite period
    }
    n / freq_index
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
    fn fft_real_pure_sine() {
        let signal = generate_sine(128, 16);
        let fft_result = fft_real(&signal);

        // Should have strong peak at frequency 128/16 = 8
        assert!(!fft_result.is_empty());

        // Find the peak
        let powers: Vec<f64> = fft_result
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();
        let max_idx = powers
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(max_idx, 8); // frequency index 8 = period 16
    }

    #[test]
    fn fft_real_empty() {
        let result = fft_real(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn periodogram_pure_sine() {
        let signal = generate_sine(128, 12);
        let psd = periodogram(&signal);

        // Should detect period close to 12
        assert!(!psd.is_empty());

        // Find the peak
        let peak = psd.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        assert!(peak.is_some());

        let (period, _) = peak.unwrap();
        // Due to discretization, we might get 10-12
        assert!(
            (10..=14).contains(period),
            "Expected period near 12, got {}",
            period
        );
    }

    #[test]
    fn periodogram_multiple_frequencies() {
        // Signal with two frequencies: period 7 and period 30
        let n = 210;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                    + 0.8 * (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin()
            })
            .collect();

        let peaks = periodogram_peaks(&signal, 2.0, 2, 50);

        assert!(peaks.len() >= 2, "Should detect at least 2 peaks");

        // Check that we found periods near 7 and 30
        let periods: Vec<usize> = peaks.iter().map(|(p, _)| *p).collect();
        let has_7 = periods.iter().any(|p| (5..=10).contains(p));
        let has_30 = periods.iter().any(|p| (25..=35).contains(p));

        assert!(
            has_7 && has_30,
            "Should detect periods near 7 and 30, got {:?}",
            periods
        );
    }

    #[test]
    fn periodogram_peaks_threshold() {
        let signal = generate_sine(128, 16);
        let peaks = periodogram_peaks(&signal, 3.0, 2, 64);

        // Should find at least the main period
        assert!(!peaks.is_empty());

        // First peak should be near period 16
        let (period, _) = peaks[0];
        assert!(
            (14..=18).contains(&period),
            "Expected period near 16, got {}",
            period
        );
    }

    #[test]
    fn welch_periodogram_basic() {
        let signal = generate_sine(256, 12);
        let psd = welch_periodogram(&signal, 64, 0.5);

        assert!(!psd.is_empty());

        // Should find period near 12
        let peak = psd.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        assert!(peak.is_some());

        let (period, _) = peak.unwrap();
        assert!(
            (10..=14).contains(period),
            "Expected period near 12, got {}",
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
    fn period_frequency_conversion() {
        let n = 128;

        // Test round-trip
        for period in [4, 8, 12, 16, 32, 64] {
            let freq = period_to_frequency_index(period, n);
            let recovered = frequency_index_to_period(freq, n);
            // May not be exact due to integer division
            assert!((recovered as i32 - period as i32).abs() <= 1);
        }
    }

    #[test]
    fn periodogram_constant_signal() {
        let signal = vec![5.0; 64];
        let psd = periodogram(&signal);

        // Constant signal should have power concentrated at DC (period = N)
        // Other periods should have very low power
        for (period, power) in &psd {
            if *period < 64 {
                assert!(
                    *power < 0.01,
                    "Non-DC period {} has power {}",
                    period,
                    power
                );
            }
        }
    }

    #[test]
    fn periodogram_high_threshold_reduces_peaks() {
        // White noise-like signal (deterministic for reproducibility)
        let signal: Vec<f64> = (0..128).map(|i| ((i * 7 + 3) % 13) as f64 - 6.0).collect();

        // Compare different thresholds
        let peaks_low = periodogram_peaks(&signal, 1.5, 2, 64);
        let peaks_high = periodogram_peaks(&signal, 10.0, 2, 64);

        // Higher threshold should result in fewer or equal peaks
        assert!(
            peaks_high.len() <= peaks_low.len(),
            "Higher threshold should reduce or maintain peak count: low={}, high={}",
            peaks_low.len(),
            peaks_high.len()
        );
    }
}
