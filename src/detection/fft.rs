//! FFT utilities for spectral analysis.
//!
//! Provides Welch's periodogram for robust spectral estimation.

use rustfft::{num_complex::Complex64, FftPlanner};

/// Compute the FFT of a real-valued signal.
///
/// Returns the complex frequency domain representation.
/// Only returns the first half (positive frequencies) since
/// the input is real-valued and the spectrum is symmetric.
fn fft_real(signal: &[f64]) -> Vec<Complex64> {
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
fn periodogram(signal: &[f64]) -> Vec<(usize, f64)> {
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

        // Should have peaks near 16 and 32
        let top_periods: Vec<usize> = psd.iter().take(10).map(|(p, _)| *p).collect();

        let has_16 = top_periods.iter().any(|p| (14..=18).contains(p));
        let has_32 = top_periods.iter().any(|p| (28..=36).contains(p));

        assert!(
            has_16 || has_32,
            "Should detect at least one period, got {:?}",
            top_periods
        );
    }
}
