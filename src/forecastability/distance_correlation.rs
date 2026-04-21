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

    // Pre-compute row means and grand mean for Y (O(n²) time, O(n) memory).
    let mut y_row_means = vec![0.0; n];
    let mut y_grand_sum = 0.0;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += (y[i] - y[j]).abs();
        }
        y_row_means[i] = row_sum / n_f;
        y_grand_sum += row_sum;
    }
    let y_grand_mean = y_grand_sum / n2;

    // Build doubly-centered A (for X) and accumulate all three inner
    // products in a single pass over the A matrix + on-the-fly B elements.
    let mut x_row_means = vec![0.0; n];
    let mut x_grand_sum = 0.0;
    // First pass: pairwise distances for X, compute row means.
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let dist = (x[i] - x[j]).abs();
            a[i * n + j] = dist;
            a[j * n + i] = dist;
        }
    }
    for i in 0..n {
        let row_sum: f64 = a[i * n..i * n + n].iter().sum();
        x_row_means[i] = row_sum / n_f;
        x_grand_sum += row_sum;
    }
    let x_grand_mean = x_grand_sum / n2;

    // Double-center A in place and accumulate inner products with B on the fly.
    let mut dcov2 = 0.0;
    let mut dvar_x = 0.0;
    let mut dvar_y = 0.0;
    for i in 0..n {
        for j in 0..n {
            let aij = a[i * n + j] - x_row_means[i] - x_row_means[j] + x_grand_mean;
            let bij = (y[i] - y[j]).abs() - y_row_means[i] - y_row_means[j] + y_grand_mean;
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
