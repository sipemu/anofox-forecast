//! Phase-randomized surrogates for significance testing.
//!
//! Preserves the power spectrum (autocorrelation) of the original series
//! while destroying any nonlinear structure, phase relationships, and
//! specific temporal patterns. If the AMI of the original series exceeds
//! the surrogate band, the predictive structure is statistically
//! significant.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::fft_complex::{fft, next_pow2, C64};

/// Generate a single phase-randomized surrogate from a pre-computed
/// amplitude spectrum. Avoids storing all surrogates in memory.
fn generate_single_surrogate(
    spectrum: &[C64],
    amplitudes: &[f64],
    padded_n: usize,
    n: usize,
    base_seed: u64,
    surrogate_idx: usize,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(surrogate_idx as u64));
    let mut surr_spectrum: Vec<C64> = Vec::with_capacity(padded_n);

    for k in 0..padded_n {
        if k == 0 || (padded_n % 2 == 0 && k == padded_n / 2) {
            surr_spectrum.push(spectrum[k]);
        } else if k < padded_n.div_ceil(2) {
            let theta: f64 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            surr_spectrum.push((amplitudes[k] * theta.cos(), amplitudes[k] * theta.sin()));
        } else {
            let conj = surr_spectrum[padded_n - k];
            surr_spectrum.push((conj.0, -conj.1));
        }
    }

    fft(&mut surr_spectrum, true);
    surr_spectrum[..n].iter().map(|&(re, _)| re).collect()
}

/// Generate `n_surrogates` phase-randomized surrogate series.
///
/// Each surrogate has the same power spectrum as the original but randomized
/// phases, destroying any predictive structure that the original may have.
///
/// # Arguments
/// * `series` — the original time series
/// * `n_surrogates` — how many surrogates to produce
/// * `seed` — optional RNG seed for reproducibility
///
/// # Returns
/// A `Vec<Vec<f64>>` of `n_surrogates` surrogate series, each with the
/// same length as `series`.
pub fn phase_surrogates(series: &[f64], n_surrogates: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
    let n = series.len();
    if n < 4 || n_surrogates == 0 {
        return Vec::new();
    }

    let padded_n = next_pow2(n);
    let base_seed = seed.unwrap_or(0x42_43_44_45);

    // Forward FFT of the original (zero-padded to power of two).
    let mut spectrum: Vec<C64> = series
        .iter()
        .map(|&v| (v, 0.0))
        .chain(std::iter::repeat_n((0.0, 0.0), padded_n - n))
        .collect();
    fft(&mut spectrum, false);

    let amplitudes: Vec<f64> = spectrum
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect();

    (0..n_surrogates)
        .map(|s| generate_single_surrogate(&spectrum, &amplitudes, padded_n, n, base_seed, s))
        .collect()
}

/// Significance bands for a metric computed over phase surrogates.
#[derive(Debug, Clone)]
pub struct SignificanceBands {
    /// Lower (α/2) percentile of the surrogate distribution at each lag.
    pub lower: Vec<f64>,
    /// Upper (1 − α/2) percentile of the surrogate distribution at each lag.
    pub upper: Vec<f64>,
    /// Mean of the surrogate distribution at each lag.
    pub mean: Vec<f64>,
    /// Number of surrogates used.
    pub n_surrogates: usize,
    /// Significance level used.
    pub alpha: f64,
}

/// Compute significance bands for a lag-curve metric using phase surrogates.
///
/// Runs the supplied `metric_fn` on each surrogate to build a distribution
/// at every lag, then extracts the `(α/2, 1 − α/2)` percentile band.
///
/// # Arguments
/// * `series` — the original time series
/// * `metric_fn` — function `fn(&[f64], max_lag) → Vec<f64>` that computes
///   the lag curve (e.g. `ami_curve`, `gcmi_curve`)
/// * `max_lag` — number of lags to compute
/// * `n_surrogates` — how many surrogates to draw (default: 100)
/// * `alpha` — significance level (default: 0.05)
/// * `seed` — optional RNG seed
pub fn significance_bands<F>(
    series: &[f64],
    metric_fn: F,
    max_lag: usize,
    n_surrogates: usize,
    alpha: f64,
    seed: Option<u64>,
) -> SignificanceBands
where
    F: Fn(&[f64], usize) -> Vec<f64> + Sync + Send,
{
    // Generate surrogates one at a time (streaming) to avoid O(m×n) peak
    // memory from storing all surrogates simultaneously. The FFT amplitude
    // spectrum is pre-computed once and shared across all surrogates.
    let n = series.len();
    let padded_n = next_pow2(n);
    let base_seed = seed.unwrap_or(0x42_43_44_45);

    // Forward FFT of the original.
    let mut spectrum: Vec<C64> = series
        .iter()
        .map(|&v| (v, 0.0))
        .chain(std::iter::repeat_n((0.0, 0.0), padded_n - n))
        .collect();
    fft(&mut spectrum, false);

    let amplitudes: Vec<f64> = spectrum
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect();

    // Compute metric curves for all surrogates.
    #[cfg(feature = "parallel")]
    let surrogate_curves: Vec<Vec<f64>> = (0..n_surrogates)
        .into_par_iter()
        .map(|s| {
            let surr = generate_single_surrogate(&spectrum, &amplitudes, padded_n, n, base_seed, s);
            metric_fn(&surr, max_lag)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let surrogate_curves: Vec<Vec<f64>> = (0..n_surrogates)
        .map(|s| {
            let surr = generate_single_surrogate(&spectrum, &amplitudes, padded_n, n, base_seed, s);
            metric_fn(&surr, max_lag)
        })
        .collect();

    // Transpose: collect per-lag values across all surrogates.
    let mut lag_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_surrogates); max_lag];
    for curve in &surrogate_curves {
        for (lag_idx, &val) in curve.iter().enumerate() {
            if lag_idx < max_lag {
                lag_values[lag_idx].push(val);
            }
        }
    }

    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;

    let mut lower = Vec::with_capacity(max_lag);
    let mut upper = Vec::with_capacity(max_lag);
    let mut mean = Vec::with_capacity(max_lag);

    for vals in &mut lag_values {
        if vals.is_empty() {
            lower.push(0.0);
            upper.push(0.0);
            mean.push(0.0);
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        let lo_idx = ((lo_q * n as f64) as usize).min(n - 1);
        let hi_idx = ((hi_q * n as f64) as usize).min(n - 1);
        lower.push(vals[lo_idx]);
        upper.push(vals[hi_idx]);
        mean.push(vals.iter().sum::<f64>() / n as f64);
    }

    SignificanceBands {
        lower,
        upper,
        mean,
        n_surrogates,
        alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogates_preserve_length() {
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let surrs = phase_surrogates(&series, 5, Some(42));
        assert_eq!(surrs.len(), 5);
        for s in &surrs {
            assert_eq!(s.len(), 100);
        }
    }

    #[test]
    fn surrogates_deterministic_with_seed() {
        let series: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
        let a = phase_surrogates(&series, 3, Some(123));
        let b = phase_surrogates(&series, 3, Some(123));
        for (sa, sb) in a.iter().zip(b.iter()) {
            for (&va, &vb) in sa.iter().zip(sb.iter()) {
                assert!((va - vb).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn surrogates_have_similar_variance() {
        let series: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let var_orig = variance(&series);
        let surrs = phase_surrogates(&series, 10, Some(1));
        for s in &surrs {
            let var_s = variance(s);
            let ratio = var_s / var_orig;
            assert!(
                (0.5..2.0).contains(&ratio),
                "surrogate variance ratio {:.2} outside expected range",
                ratio
            );
        }
    }

    fn variance(x: &[f64]) -> f64 {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        x.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
    }
}
