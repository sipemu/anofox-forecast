//! Gaussian Copula Mutual Information (GCMI).
//!
//! Ince, R. A. A., et al. (2017). A statistical framework for neuroimaging
//! data analysis based on mutual information estimated via a Gaussian copula.
//! *Human Brain Mapping*, 38(3), 1541–1573.
//!
//! The idea: rank-transform both variables to uniform marginals, then map
//! through the inverse normal CDF (probit) to obtain Gaussian marginals.
//! The MI of a bivariate Gaussian with correlation ρ is:
//!
//! ```text
//! I(X; Y) = -0.5 * log₂(1 - ρ²)
//! ```
//!
//! This captures only **linear** dependence (since copula-Gaussian MI =
//! Pearson MI on rank-normalized data). Fully deterministic — no random state.

use statrs::distribution::{ContinuousCDF, Normal};

/// Compute the Gaussian Copula Mutual Information between `x` and `y`.
///
/// Returns MI in **bits** (base-2 logarithm), matching the convention in
/// Ince et al. (2017). Returns 0.0 for degenerate inputs (n < 3 or
/// zero variance).
pub fn gcmi(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");
    if n < 3 {
        return 0.0;
    }

    // Rank-transform → probit.
    let gx = rank_to_probit(x);
    let gy = rank_to_probit(y);

    // Pearson correlation on probit-transformed data.
    let rho = pearson(&gx, &gy);
    let rho2 = rho * rho;

    if rho2 >= 1.0 {
        return f64::INFINITY; // perfect dependence
    }

    -0.5 * (1.0 - rho2).log2()
}

/// Rank-transform a vector, then map ranks to Gaussian quantiles via the
/// probit (inverse normal CDF) of `rank / (n + 1)`.
fn rank_to_probit(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Compute ranks (1-based, average ties).
    let mut indexed: Vec<(f64, usize)> = values.iter().copied().zip(0..).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 0.5; // 1-based midrank
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.1] = avg_rank;
        }
        i = j;
    }

    // Probit: Φ⁻¹(rank / (n + 1)).
    let scale = 1.0 / (n as f64 + 1.0);
    ranks
        .iter()
        .map(|&r| normal.inverse_cdf(r * scale))
        .collect()
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx < 1e-30 || syy < 1e-30 {
        return 0.0;
    }
    sxy / (sxx * syy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn independent_variables_gcmi_near_zero() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.07).sin()).collect();
        let y: Vec<f64> = (0..200)
            .map(|i| ((i * 13 + 7) % 97) as f64 / 97.0)
            .collect();
        let mi = gcmi(&x, &y);
        assert!(
            mi.abs() < 0.15,
            "independent GCMI should be near 0, got {}",
            mi
        );
    }

    #[test]
    fn linearly_dependent_gcmi_is_positive() {
        let x: Vec<f64> = (0..300).map(|i| i as f64 * 0.1).collect();
        let noise: Vec<f64> = (0..300)
            .map(|i| ((i * 7 + 3) % 11) as f64 * 0.1 - 0.55)
            .collect();
        let y: Vec<f64> = x
            .iter()
            .zip(noise.iter())
            .map(|(&xi, &ni)| 3.0 * xi + ni)
            .collect();
        let mi = gcmi(&x, &y);
        assert!(
            mi > 1.0,
            "strong linear dependence GCMI should be >> 0, got {}",
            mi
        );
    }

    #[test]
    fn gcmi_deterministic_same_input_same_output() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).cos()).collect();
        let a = gcmi(&x, &y);
        let b = gcmi(&x, &y);
        assert_relative_eq!(a, b, epsilon = 1e-15);
    }
}
