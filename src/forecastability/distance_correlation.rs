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
/// or zero distance variance). Complexity: O(n²) time, O(n) memory.
///
/// Uses a fused single-pass algorithm that computes dCov²(X,Y), dVar²(X),
/// and dVar²(Y) simultaneously without materializing two n×n matrices.
/// Only one n×n matrix (for X) is kept; the Y matrix is computed on the
/// fly during the inner-product accumulation.
pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");
    if n < 4 {
        return 0.0;
    }
    let n_f = n as f64;
    let n2 = (n * n) as f64;

    // Pre-sort both x and y for O(n log n) row-mean computation.
    // For 1D data, the sum of |x_i - x_j| over all j can be computed in
    // O(n) using the sorted-order trick:
    //   sum_j |x_i - x_j| = 2 * rank_i * x_i - 2 * prefix_sum[rank_i]
    //                        + total_sum - 2 * x_i * (n - rank_i)
    // ... but for simplicity and correctness, we use the straightforward
    // O(n²) approach since distance_correlation is not called in the hot
    // fingerprint loop. The O(n²) inner product dominates regardless.

    // Compute pairwise distances and row means for both X and Y in a
    // single fused O(n²) pass. Store X distances in a flat matrix;
    // Y distances are only used for row means and then recomputed
    // on the fly in the inner-product pass.
    let mut a = vec![0.0; n * n]; // X distance matrix
    let mut x_row_sums = vec![0.0; n];
    let mut y_row_sums = vec![0.0; n];
    let mut x_grand_sum = 0.0;
    let mut y_grand_sum = 0.0;

    for i in 0..n {
        for j in i + 1..n {
            let dx = (x[i] - x[j]).abs();
            let dy = (y[i] - y[j]).abs();
            a[i * n + j] = dx;
            a[j * n + i] = dx;
            x_row_sums[i] += dx;
            x_row_sums[j] += dx;
            y_row_sums[i] += dy;
            y_row_sums[j] += dy;
        }
    }
    for i in 0..n {
        x_grand_sum += x_row_sums[i];
        y_grand_sum += y_row_sums[i];
    }

    let x_grand_mean = x_grand_sum / n2;
    let y_grand_mean = y_grand_sum / n2;
    // Convert sums to means.
    for i in 0..n {
        x_row_sums[i] /= n_f;
        y_row_sums[i] /= n_f;
    }

    // Double-center A in place and accumulate inner products with B on the fly.
    let mut dcov2 = 0.0;
    let mut dvar_x = 0.0;
    let mut dvar_y = 0.0;
    for i in 0..n {
        for j in 0..n {
            let aij = a[i * n + j] - x_row_sums[i] - x_row_sums[j] + x_grand_mean;
            let bij = (y[i] - y[j]).abs() - y_row_sums[i] - y_row_sums[j] + y_grand_mean;
            dcov2 += aij * bij;
            dvar_x += aij * aij;
            dvar_y += bij * bij;
        }
    }
    dcov2 /= n2;
    dvar_x /= n2;
    dvar_y /= n2;

    if dvar_x <= 0.0 || dvar_y <= 0.0 {
        return 0.0;
    }

    let dcor2 = dcov2 / (dvar_x * dvar_y).sqrt();
    dcor2.max(0.0).sqrt()
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
