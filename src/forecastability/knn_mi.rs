//! k-Nearest-Neighbor mutual information estimator (Kraskov KSG1 algorithm).
//!
//! Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
//! *Estimating mutual information.* Physical Review E, 69(6), 066138.
//!
//! The KSG1 estimator uses the Chebyshev (L∞) distance in the joint
//! (X, Y) space to find the k-th nearest neighbor, then counts how many
//! marginal neighbors fall within the same ε-ball in each marginal
//! projection. The MI is:
//!
//! ```text
//! I(X; Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)
//! ```
//!
//! where `ψ` is the digamma function, `n_x` and `n_y` are the marginal
//! neighbor counts for each point, and `<·>` is the sample average.

/// Digamma function via Stirling + recurrence for small arguments.
///
/// Accurate to ~1e-12 for integer/half-integer args and ~1e-8 in general.
fn digamma(mut x: f64) -> f64 {
    // Shift to large-argument regime (x ≥ 8) via recurrence ψ(x) = ψ(x+1) - 1/x.
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Stirling series for ψ(x) when x ≥ 8.
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln()
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0)));
    result
}

/// Count how many values in `sorted` fall strictly within `(center - eps, center + eps)`.
///
/// Uses binary search for O(log n) per query.
fn count_within_eps(sorted: &[f64], center: f64, eps: f64) -> usize {
    if eps <= 0.0 {
        return 0;
    }
    let lo = center - eps;
    let hi = center + eps;
    let left = sorted.partition_point(|&v| v <= lo);
    let right = sorted.partition_point(|&v| v < hi);
    right.saturating_sub(left)
}

/// Compute the k-th nearest neighbor ε (Chebyshev / L∞ distance) for each
/// point in the joint (X, Y) space.
///
/// Returns one ε per point. Uses `select_nth_unstable_by` (introselect,
/// O(n) average) instead of a full sort, and reuses a single distance
/// buffer across all points to avoid n² small allocations.
fn kth_neighbor_eps(x: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut result = Vec::with_capacity(n);
    // Reusable buffer: one entry per point (including self).
    let mut dists = Vec::with_capacity(n);

    for i in 0..n {
        dists.clear();
        for j in 0..n {
            if j == i {
                // Self-distance: set to infinity so it's never selected as
                // a neighbor. Avoids the filter allocation + branch.
                dists.push(f64::INFINITY);
            } else {
                let dx = (x[i] - x[j]).abs();
                let dy = (y[i] - y[j]).abs();
                dists.push(dx.max(dy));
            }
        }
        // Partial selection: O(n) average to place the k-th smallest at
        // index k-1, without sorting the rest.
        let kth_idx = k - 1;
        dists.select_nth_unstable_by(kth_idx, |a, b| a.partial_cmp(b).unwrap());
        result.push(dists[kth_idx]);
    }
    result
}

/// Estimate the mutual information `I(X; Y)` using the KSG1 (Kraskov
/// Algorithm 1) k-nearest-neighbor estimator.
///
/// # Arguments
/// * `x` — first variable (length N)
/// * `y` — second variable (length N)
/// * `k` — number of neighbors (default in the reference package: 8)
///
/// # Returns
/// Estimated MI in nats. Returns 0.0 for degenerate inputs.
///
/// # Panics
/// Panics if `x.len() != y.len()` or `k == 0` or `k >= N`.
pub fn knn_mutual_information(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");
    assert!(k > 0 && k < n, "k must satisfy 0 < k < N");

    if n < k + 1 {
        return 0.0;
    }

    // Pre-sort marginals for binary-search counting.
    let mut x_sorted: Vec<f64> = x.to_vec();
    x_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut y_sorted: Vec<f64> = y.to_vec();
    y_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Find k-th neighbor ε for each point in the joint Chebyshev space.
    let eps_vec = kth_neighbor_eps(x, y, k);

    // For each point, count marginal neighbors within the ε-ball.
    let mut sum_psi = 0.0;
    for (i, &eps) in eps_vec.iter().enumerate() {
        // n_x = number of points j ≠ i with |x_j - x_i| < eps.
        // We count from the sorted array and subtract 1 for the point itself.
        let n_x = count_within_eps(&x_sorted, x[i], eps).saturating_sub(1);
        let n_y = count_within_eps(&y_sorted, y[i], eps).saturating_sub(1);
        sum_psi += digamma((n_x + 1) as f64) + digamma((n_y + 1) as f64);
    }

    let avg_psi = sum_psi / n as f64;
    let mi = digamma(k as f64) - avg_psi + digamma(n as f64);
    mi.max(0.0) // MI is non-negative; clamp numerical noise.
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn digamma_at_1_is_neg_euler() {
        // ψ(1) = -γ ≈ -0.5772156649
        assert_relative_eq!(digamma(1.0), -0.5772156649, epsilon = 1e-6);
    }

    #[test]
    fn digamma_at_integers() {
        // ψ(n) = -γ + Σ_{k=1}^{n-1} 1/k
        // ψ(2) = -γ + 1 ≈ 0.4228
        assert_relative_eq!(digamma(2.0), 0.42278, epsilon = 1e-4);
        // ψ(5) ≈ 1.5061
        assert_relative_eq!(digamma(5.0), 1.5061, epsilon = 1e-4);
    }

    #[test]
    fn mi_of_identical_variables_is_positive() {
        // I(X; X) = H(X) > 0 for non-degenerate X.
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let mi = knn_mutual_information(&x, &x, 5);
        assert!(mi > 0.5, "I(X;X) should be large, got {}", mi);
    }

    #[test]
    fn mi_of_independent_variables_is_near_zero() {
        // Independent X and Y (different deterministic sequences with no
        // shared structure) should have MI ≈ 0.
        let x: Vec<f64> = (0..300).map(|i| (i as f64 * 0.07).sin()).collect();
        let y: Vec<f64> = (0..300)
            .map(|i| ((i * 13 + 7) % 97) as f64 / 97.0)
            .collect();
        let mi = knn_mutual_information(&x, &y, 8);
        assert!(mi < 0.3, "I(independent X, Y) should be near 0, got {}", mi);
    }

    #[test]
    fn mi_detects_linear_dependence() {
        // Y = 2X + noise: should have positive MI.
        let x: Vec<f64> = (0..300)
            .map(|i| {
                let base = (i as f64 * 0.05).sin();
                base + ((i * 7 + 3) % 11) as f64 * 0.02 - 0.11
            })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.0 * xi + ((i * 13 + 5) % 17) as f64 * 0.02 - 0.17)
            .collect();
        let mi = knn_mutual_information(&x, &y, 8);
        assert!(mi > 0.3, "linear dependence MI should be > 0.3, got {}", mi);
    }
}
