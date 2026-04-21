//! Distance correlation (Szekely, Rizzo & Bakirov, 2007).
//!
//! Measures both linear *and* nonlinear dependence between two random
//! variables. Unlike Pearson correlation, `dCor(X, Y) = 0` if and only
//! if X and Y are independent (for finite-variance RVs).
//!
//! ```text
//! dCor(X, Y) = dCov(X, Y) / sqrt(dVar(X) * dVar(Y))
//! ```
//!
//! where `dCov` and `dVar` are computed from doubly-centered pairwise
//! Euclidean distance matrices.

/// Compute the distance correlation between `x` and `y`.
///
/// Returns a value in `[0, 1]`. Returns 0.0 for degenerate inputs (n < 4
/// or zero distance variance). Complexity: O(n²).
pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");
    if n < 4 {
        return 0.0;
    }

    let a = doubly_centered_distances(x);
    let b = doubly_centered_distances(y);

    let dcov2 = inner_product(&a, &b, n);
    let dvar_x = inner_product(&a, &a, n);
    let dvar_y = inner_product(&b, &b, n);

    if dvar_x <= 0.0 || dvar_y <= 0.0 {
        return 0.0;
    }

    let dcor2 = dcov2 / (dvar_x * dvar_y).sqrt();
    dcor2.max(0.0).sqrt()
}

/// Compute the doubly-centered distance matrix from a 1-D sample.
///
/// Returns a flattened n×n matrix where `A[i,j] = a_{ij} - a_{i.} - a_{.j} + a_{..}`,
/// with `a_{ij} = |x_i - x_j|`.
fn doubly_centered_distances(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let n_f = n as f64;

    // Pairwise distances.
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let dist = (x[i] - x[j]).abs();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }

    // Row means, column means, grand mean.
    let mut row_means = vec![0.0; n];
    let mut grand_mean = 0.0;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| d[i * n + j]).sum();
        row_means[i] = row_sum / n_f;
        grand_mean += row_sum;
    }
    grand_mean /= (n * n) as f64;

    // Double-center: A[i,j] = d[i,j] - row_mean[i] - row_mean[j] + grand_mean.
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = d[i * n + j] - row_means[i] - row_means[j] + grand_mean;
        }
    }

    d
}

/// Inner product of two doubly-centered matrices: (1/n²) Σ A[i,j] * B[i,j].
fn inner_product(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n * n {
        sum += a[i] * b[i];
    }
    sum / (n * n) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identical_variables_dcor_is_one() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let dc = distance_correlation(&x, &x);
        assert_relative_eq!(dc, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn independent_variables_dcor_near_zero() {
        // Two deterministic sequences with no shared structure.
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.07).sin()).collect();
        let y: Vec<f64> = (0..200)
            .map(|i| ((i * 13 + 7) % 97) as f64 / 97.0)
            .collect();
        let dc = distance_correlation(&x, &y);
        assert!(dc < 0.15, "independent dCor should be near 0, got {}", dc);
    }

    #[test]
    fn nonlinear_dependence_detected() {
        // Y = X² — Pearson ρ ≈ 0 for symmetric X, but dCor > 0.
        let x: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let dc = distance_correlation(&x, &y);
        assert!(
            dc > 0.3,
            "quadratic dependence dCor should be > 0.3, got {}",
            dc
        );
    }

    #[test]
    fn dcor_bounded_zero_one() {
        let x: Vec<f64> = (0..80).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..80).map(|i| (i as f64 * 0.13).cos()).collect();
        let dc = distance_correlation(&x, &y);
        assert!((0.0..=1.001).contains(&dc), "dCor out of range: {}", dc);
    }
}
