//! Likelihood-weighted mixture over leaves.
//!
//! Each leaf keeps a running cumulative one-step log-likelihood. Weights
//! are computed on demand as `softmax(cum_log_lik)`. `blend_horizon`
//! folds per-leaf horizon predictions into a single [`GaussianMixture`]
//! per step.

use super::dist::{Gaussian, GaussianMixture};

/// Turn cumulative log-likelihoods into normalised softmax weights.
///
/// Empty input returns empty. Non-finite entries (`NaN` / `-Inf`) are
/// treated as `f64::NEG_INFINITY` for numerical safety.
pub fn softmax(log_liks: &[f64]) -> Vec<f64> {
    if log_liks.is_empty() {
        return Vec::new();
    }
    let max = log_liks
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        // All leaves useless — fall back to uniform.
        let n = log_liks.len() as f64;
        return vec![1.0 / n; log_liks.len()];
    }
    let exps: Vec<f64> = log_liks
        .iter()
        .map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 })
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        let n = log_liks.len() as f64;
        return vec![1.0 / n; log_liks.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// In-place softmax — writes weights into `out` without allocating.
///
/// `out` is resized to `log_liks.len()`. Reuses caller-owned storage on
/// the fit hot path. Same semantics as [`softmax`] (uniform fallback on
/// degenerate inputs, `-Inf` for non-finite entries).
pub fn softmax_into(log_liks: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.resize(log_liks.len(), 0.0);
    if log_liks.is_empty() {
        return;
    }
    let max = log_liks
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        let uniform = 1.0 / log_liks.len() as f64;
        for w in out.iter_mut() {
            *w = uniform;
        }
        return;
    }
    let mut sum = 0.0;
    for (i, &l) in log_liks.iter().enumerate() {
        let e = if l.is_finite() { (l - max).exp() } else { 0.0 };
        out[i] = e;
        sum += e;
    }
    if sum <= 0.0 {
        let uniform = 1.0 / log_liks.len() as f64;
        for w in out.iter_mut() {
            *w = uniform;
        }
        return;
    }
    let inv = 1.0 / sum;
    for w in out.iter_mut() {
        *w *= inv;
    }
}

/// For a fixed horizon `h`, produce the mixture over `weights.len()` leaves.
///
/// `per_leaf_horizons[i]` is leaf `i`'s vector of per-horizon gaussians
/// (length `≥ h`). Returns an empty mixture if inputs are inconsistent.
pub fn blend_horizon(
    weights: &[f64],
    per_leaf_horizons: &[Vec<Gaussian>],
    h_index: usize,
) -> GaussianMixture {
    if weights.len() != per_leaf_horizons.len() {
        return GaussianMixture::default();
    }
    let pairs = weights
        .iter()
        .zip(per_leaf_horizons.iter())
        .filter_map(|(w, horizons)| horizons.get(h_index).map(|g| (*w, *g)));
    GaussianMixture::new(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_uniform_when_all_equal() {
        let w = softmax(&[-1.0, -1.0, -1.0]);
        for wi in w {
            assert!((wi - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_normalised_and_ordering_preserved() {
        let w = softmax(&[-10.0, -1.0, 0.0]);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w[0] < w[1] && w[1] < w[2]);
    }

    #[test]
    fn softmax_all_nan_falls_back_to_uniform() {
        let w = softmax(&[f64::NAN, f64::NAN]);
        assert!((w[0] - 0.5).abs() < 1e-12);
        assert!((w[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn blend_horizon_matches_expected_mixture() {
        let leaves = vec![
            vec![Gaussian::new(0.0, 1.0), Gaussian::new(0.0, 2.0)],
            vec![Gaussian::new(2.0, 1.0), Gaussian::new(2.0, 2.0)],
        ];
        let weights = vec![0.25, 0.75];
        let m = blend_horizon(&weights, &leaves, 0);
        assert_eq!(m.components.len(), 2);
        // Point forecast at h=1 = 0.25·0 + 0.75·2 = 1.5.
        assert!((m.mean() - 1.5).abs() < 1e-12);
    }
}
