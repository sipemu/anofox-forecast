//! Forecastability fingerprint — compact summary of a series' predictive
//! structure.
//!
//! Mirrors the "information geometry" workflow from `dependence-forecastability`:
//! compute AMI + GCMI curves, test significance against phase surrogates,
//! and distill into a handful of interpretable summary statistics.

use super::ami::{ami_curve, gcmi_curve};
use super::surrogates::significance_bands;

/// Compact forecastability fingerprint.
#[derive(Debug, Clone)]
pub struct ForecastabilityFingerprint {
    /// Sum of AMI values at lags where AMI exceeds the surrogate upper band.
    /// Higher = more total predictive signal.
    pub information_mass: f64,
    /// Last lag where AMI exceeds the surrogate upper band.
    /// Higher = deeper temporal dependence.
    pub information_horizon: usize,
    /// Entropy of the significant AMI profile (normalized).
    /// Higher = more evenly distributed dependence across lags.
    pub information_structure: f64,
    /// `1 − (sum of significant GCMI) / (sum of significant AMI)`.
    /// Higher = more nonlinear dependence (AMI captures it but GCMI doesn't).
    pub nonlinear_share: f64,
    /// Peak AMI / mean surrogate AMI at the same lag.
    /// Higher = stronger signal relative to noise.
    pub signal_to_noise: f64,
    /// `AMI(1) / information_mass` — how concentrated the dependence is at
    /// the first lag vs spread across all lags.
    pub directness_ratio: f64,
    /// Which lag indices (1-based) have AMI exceeding the surrogate band.
    pub informative_horizons: Vec<usize>,
    /// The raw AMI curve.
    pub ami: Vec<f64>,
    /// The raw GCMI curve.
    pub gcmi: Vec<f64>,
    /// Surrogate upper band for AMI.
    pub surrogate_upper: Vec<f64>,
    /// Surrogate mean for AMI.
    pub surrogate_mean: Vec<f64>,
}

impl ForecastabilityFingerprint {
    /// Compute the forecastability fingerprint for a series.
    ///
    /// # Arguments
    /// * `series` — the time series to analyze
    /// * `max_lag` — number of lags to probe (default: ~n/5 or 20)
    /// * `n_surrogates` — number of phase surrogates for significance testing
    ///   (default: 100)
    /// * `alpha` — significance level (default: 0.05)
    /// * `seed` — optional RNG seed for reproducibility
    pub fn compute(
        series: &[f64],
        max_lag: usize,
        n_surrogates: usize,
        alpha: f64,
        seed: Option<u64>,
    ) -> Self {
        let ami = ami_curve(series, max_lag);
        let gcmi = gcmi_curve(series, max_lag);

        // Compute surrogate significance bands for AMI.
        let bands = significance_bands(series, ami_curve, max_lag, n_surrogates, alpha, seed);

        // Identify informative horizons: where AMI > surrogate upper band.
        let informative_horizons: Vec<usize> = (0..max_lag)
            .filter(|&i| ami[i] > bands.upper[i])
            .map(|i| i + 1) // 1-based
            .collect();

        // Information mass: sum of significant AMI.
        let information_mass: f64 = informative_horizons.iter().map(|&h| ami[h - 1]).sum();

        // Information horizon: last significant lag.
        let information_horizon = informative_horizons.last().copied().unwrap_or(0);

        // Information structure: entropy of significant AMI profile.
        let information_structure = if information_mass > 1e-15 {
            let probs: Vec<f64> = informative_horizons
                .iter()
                .map(|&h| ami[h - 1] / information_mass)
                .collect();
            let k = probs.len() as f64;
            if k <= 1.0 {
                0.0
            } else {
                let h: f64 = probs
                    .iter()
                    .filter(|&&p| p > 1e-15)
                    .map(|&p| -p * p.ln())
                    .sum();
                h / k.ln() // normalize to [0, 1]
            }
        } else {
            0.0
        };

        // Nonlinear share: 1 − (sum significant GCMI) / (sum significant AMI).
        let gcmi_mass: f64 = informative_horizons
            .iter()
            .map(|&h| gcmi[h - 1].max(0.0))
            .sum();
        let nonlinear_share = if information_mass > 1e-15 {
            (1.0 - gcmi_mass / information_mass).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Signal-to-noise: peak AMI / mean surrogate AMI at the peak lag.
        let (peak_idx, peak_ami) = ami
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));
        let surr_mean_at_peak = bands.mean.get(peak_idx).copied().unwrap_or(1.0).max(1e-15);
        let signal_to_noise = peak_ami / surr_mean_at_peak;

        // Directness ratio.
        let directness_ratio = if information_mass > 1e-15 {
            ami[0] / information_mass
        } else {
            0.0
        };

        Self {
            information_mass,
            information_horizon,
            information_structure,
            nonlinear_share,
            signal_to_noise,
            directness_ratio,
            informative_horizons,
            ami,
            gcmi,
            surrogate_upper: bands.upper,
            surrogate_mean: bands.mean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn make_ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Vec::with_capacity(n);
        s.push(0.0);
        for _ in 1..n {
            let noise = (rng.gen::<f64>() - 0.5) * 2.0;
            s.push(phi * *s.last().unwrap() + noise);
        }
        s
    }

    #[test]
    fn fingerprint_ar1_has_signal() {
        let series = make_ar1(500, 0.8, 42);
        let fp = ForecastabilityFingerprint::compute(&series, 10, 30, 0.05, Some(1));
        assert!(
            fp.information_mass > 0.0,
            "AR(1) should have positive information mass"
        );
        assert!(
            fp.information_horizon >= 1,
            "AR(1) should have at least 1 informative horizon"
        );
        assert!(fp.signal_to_noise > 1.0, "SNR should exceed 1 for AR(1)");
    }

    #[test]
    fn fingerprint_white_noise_has_little_signal() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let series: Vec<f64> = (0..500).map(|_| (rng.gen::<f64>() - 0.5) * 2.0).collect();
        let fp = ForecastabilityFingerprint::compute(&series, 10, 50, 0.05, Some(2));
        // White noise may still have 1-2 false positives at α=0.05.
        assert!(
            fp.informative_horizons.len() <= 3,
            "white noise should have few informative horizons: {:?}",
            fp.informative_horizons
        );
    }
}
