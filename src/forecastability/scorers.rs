//! Scorer registry — 10 built-in forecastability scorers.
//!
//! Each scorer maps a time series (and optionally a lag) to a single
//! forecastability score.

use super::distance_correlation::distance_correlation;
use super::gcmi::gcmi;
use super::knn_mi::knn_mutual_information;
use super::transfer_entropy::transfer_entropy_curve;
use crate::features::entropy::permutation_entropy;

/// Available forecastability scorers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scorer {
    /// kNN mutual information at lag 1.
    Mi,
    /// Absolute Pearson correlation at lag 1.
    Pearson,
    /// Absolute Spearman rank correlation at lag 1.
    Spearman,
    /// Absolute Kendall tau-b at lag 1.
    Kendall,
    /// Distance correlation at lag 1 (Szekely/Rizzo).
    Distance,
    /// Transfer entropy at lag 1.
    TransferEntropy,
    /// Gaussian copula MI at lag 1.
    Gcmi,
    /// Normalized permutation entropy (lower = more regular / forecastable).
    PermutationEntropy,
    /// Spectral entropy (lower = more concentrated spectral energy).
    SpectralEntropy,
    /// Spectral predictability = 1 − spectral entropy.
    SpectralPredictability,
}

/// Compute a forecastability score for a given series.
///
/// For bivariate scorers (Mi, Pearson, Spearman, Kendall, Distance, TE,
/// GCMI), the score is computed at lag 1 (i.e., `(X_t, X_{t+1})`).
/// For univariate scorers (PermutationEntropy, SpectralEntropy,
/// SpectralPredictability), only the series itself is needed.
pub fn score(series: &[f64], scorer: Scorer) -> f64 {
    let n = series.len();
    match scorer {
        Scorer::Mi => {
            if n < 10 {
                return 0.0;
            }
            knn_mutual_information(&series[..n - 1], &series[1..], 8)
        }
        Scorer::Pearson => super::lag_correlations::pearson_curve(series, 1)
            .first()
            .copied()
            .unwrap_or(0.0),
        Scorer::Spearman => super::lag_correlations::spearman_curve(series, 1)
            .first()
            .copied()
            .unwrap_or(0.0),
        Scorer::Kendall => super::lag_correlations::kendall_curve(series, 1)
            .first()
            .copied()
            .unwrap_or(0.0),
        Scorer::Distance => {
            if n < 5 {
                return 0.0;
            }
            distance_correlation(&series[..n - 1], &series[1..])
        }
        Scorer::TransferEntropy => transfer_entropy_curve(series, series, 1)
            .first()
            .copied()
            .unwrap_or(0.0),
        Scorer::Gcmi => {
            if n < 4 {
                return 0.0;
            }
            gcmi(&series[..n - 1], &series[1..])
        }
        Scorer::PermutationEntropy => {
            // Auto-select embedding order m ∈ {3, 4, 5} based on series length.
            let m = if n >= 120 {
                5
            } else if n >= 24 {
                4
            } else {
                3
            };
            permutation_entropy(series, m, 1)
        }
        Scorer::SpectralEntropy => spectral_entropy(series),
        Scorer::SpectralPredictability => 1.0 - spectral_entropy(series),
    }
}

/// Normalized spectral entropy from Welch PSD.
fn spectral_entropy(series: &[f64]) -> f64 {
    use crate::detection::welch_periodogram;
    let n = series.len();
    if n < 16 {
        return f64::NAN;
    }
    let window_size = (n / 4).max(8).next_power_of_two().min(n);
    let psd = welch_periodogram(series, window_size, 0.5);
    if psd.is_empty() {
        return f64::NAN;
    }
    let total: f64 = psd.iter().map(|(_, p)| p).sum();
    if total < 1e-30 {
        return 0.0;
    }
    let n_bins = psd.len() as f64;
    let mut h = 0.0;
    for &(_, p) in &psd {
        let prob = p / total;
        if prob > 1e-30 {
            h -= prob * prob.ln();
        }
    }
    h / n_bins.ln() // normalized to [0, 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_scorers_return_finite_on_ar1() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 200;
        let mut s = vec![0.0; n];
        for i in 1..n {
            s[i] = 0.7 * s[i - 1] + (rng.gen::<f64>() - 0.5) * 2.0;
        }
        for scorer in [
            Scorer::Mi,
            Scorer::Pearson,
            Scorer::Spearman,
            Scorer::Kendall,
            Scorer::Distance,
            Scorer::TransferEntropy,
            Scorer::Gcmi,
            Scorer::PermutationEntropy,
            Scorer::SpectralEntropy,
            Scorer::SpectralPredictability,
        ] {
            let s_val = score(&s, scorer);
            assert!(
                s_val.is_finite(),
                "{:?} returned non-finite: {}",
                scorer,
                s_val
            );
        }
    }
}
