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
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln()
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0)));
    result
}

/// Pre-computed digamma table: `DIGAMMA_TABLE[i] = ψ(i + 1)` for i = 0..n.
/// Avoids 2n calls to the recurrence-based `digamma()` per MI estimation.
fn digamma_table(max_arg: usize) -> Vec<f64> {
    (0..=max_arg).map(|i| digamma((i + 1) as f64)).collect()
}

/// Count how many values in `sorted` fall strictly within `(center - eps, center + eps)`.
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

// ---------------------------------------------------------------------------
// 2D KD-tree for Chebyshev (L∞) k-nearest-neighbor queries
// ---------------------------------------------------------------------------

/// A node in the 2D KD-tree. Stores the original index and coordinates.
#[derive(Clone)]
struct KdNode {
    idx: usize,
    xy: [f64; 2],
}

/// 2D KD-tree with Chebyshev distance for bulk k-NN queries.
///
/// Uses a flat sorted-subarray layout: the median at each level is stored
/// at the midpoint of the subarray, with left/right children in the
/// respective halves.
struct KdTree {
    tree: Vec<KdNode>,
}

impl KdTree {
    /// Build a KD-tree from n 2D points. O(n log n).
    fn build(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        let mut nodes: Vec<KdNode> = (0..n)
            .map(|i| KdNode {
                idx: i,
                xy: [x[i], y[i]],
            })
            .collect();
        let mut tree = vec![
            KdNode {
                idx: 0,
                xy: [0.0, 0.0]
            };
            n
        ];
        Self::build_recursive(&mut nodes, &mut tree, 0, n, 0);
        Self { tree }
    }

    fn build_recursive(
        nodes: &mut [KdNode],
        tree: &mut [KdNode],
        start: usize,
        end: usize,
        depth: usize,
    ) {
        if start >= end {
            return;
        }
        let dim = depth % 2;
        let mid = (start + end) / 2;
        // Partial sort to place the median at `mid`.
        nodes[start..end]
            .select_nth_unstable_by(mid - start, |a, b| a.xy[dim].total_cmp(&b.xy[dim]));
        tree[mid] = nodes[mid].clone();
        if mid > start {
            Self::build_recursive(nodes, tree, start, mid, depth + 1);
        }
        if mid + 1 < end {
            Self::build_recursive(nodes, tree, mid + 1, end, depth + 1);
        }
    }

    /// Find the k nearest neighbors (by Chebyshev L∞ distance) of query point,
    /// excluding the point at `exclude_idx`. Returns the k-th neighbor distance.
    fn kth_neighbor_chebyshev(&self, qx: f64, qy: f64, exclude_idx: usize, k: usize) -> f64 {
        let n = self.tree.len();
        if n == 0 {
            return f64::INFINITY;
        }
        // Max-heap of size k: stores (distance, idx). We want the k smallest.
        let mut heap: Vec<f64> = Vec::with_capacity(k + 1);
        self.search_recursive(qx, qy, exclude_idx, k, 0, n, 0, &mut heap);
        // The largest element in the heap is the k-th nearest distance.
        heap.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn search_recursive(
        &self,
        qx: f64,
        qy: f64,
        exclude_idx: usize,
        k: usize,
        start: usize,
        end: usize,
        depth: usize,
        heap: &mut Vec<f64>,
    ) {
        if start >= end {
            return;
        }
        let mid = (start + end) / 2;
        let node = &self.tree[mid];

        // Chebyshev distance.
        let dist = (qx - node.xy[0]).abs().max((qy - node.xy[1]).abs());

        if node.idx != exclude_idx {
            if heap.len() < k {
                heap.push(dist);
            } else {
                // Find max in heap.
                let max_pos = heap
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap();
                if dist < heap[max_pos] {
                    heap[max_pos] = dist;
                }
            }
        }

        let worst = if heap.len() < k {
            f64::INFINITY
        } else {
            heap.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        let dim = depth % 2;
        let q_dim = if dim == 0 { qx } else { qy };
        let split = node.xy[dim];
        let diff = q_dim - split;

        // Visit the side containing the query first.
        let (first_start, first_end, second_start, second_end) = if diff <= 0.0 {
            (start, mid, mid + 1, end)
        } else {
            (mid + 1, end, start, mid)
        };

        self.search_recursive(
            qx,
            qy,
            exclude_idx,
            k,
            first_start,
            first_end,
            depth + 1,
            heap,
        );

        // Prune: only visit the other side if the splitting plane is closer
        // than the worst distance in the heap. For Chebyshev distance,
        // the minimum distance to any point on the other side is |diff|.
        if diff.abs() < worst || heap.len() < k {
            self.search_recursive(
                qx,
                qy,
                exclude_idx,
                k,
                second_start,
                second_end,
                depth + 1,
                heap,
            );
        }
    }
}

/// Estimate the mutual information `I(X; Y)` using the KSG1 (Kraskov
/// Algorithm 1) k-nearest-neighbor estimator.
///
/// Uses a 2D KD-tree for O(n log n) k-NN queries instead of brute-force
/// O(n²). Combined with a pre-computed digamma table, this makes the
/// estimator fast for n up to ~100k.
///
/// # Arguments
/// * `x` — first variable (length N)
/// * `y` — second variable (length N)
/// * `k` — number of neighbors (default in the reference package: 8)
///
/// # Returns
/// Estimated MI in nats. Returns 0.0 for degenerate inputs.
pub fn knn_mutual_information(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have the same length");
    assert!(k > 0 && k < n, "k must satisfy 0 < k < N");

    if n < k + 1 {
        return 0.0;
    }

    // Pre-sort marginals for binary-search counting.
    let mut x_sorted: Vec<f64> = x.to_vec();
    x_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let mut y_sorted: Vec<f64> = y.to_vec();
    y_sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    // Pre-compute digamma table: ψ(1), ψ(2), …, ψ(n+1).
    let psi = digamma_table(n);

    // Build 2D KD-tree for Chebyshev k-NN.
    let tree = KdTree::build(x, y);

    // For each point, find k-th NN distance and count marginal neighbors.
    let mut sum_psi = 0.0;
    for i in 0..n {
        let eps = tree.kth_neighbor_chebyshev(x[i], y[i], i, k);

        let n_x = count_within_eps(&x_sorted, x[i], eps).saturating_sub(1);
        let n_y = count_within_eps(&y_sorted, y[i], eps).saturating_sub(1);
        // Use the lookup table: psi[j] = ψ(j+1), so ψ(n_x+1) = psi[n_x].
        sum_psi += psi[n_x.min(n)] + psi[n_y.min(n)];
    }

    let avg_psi = sum_psi / n as f64;
    let mi = digamma(k as f64) - avg_psi + digamma(n as f64);
    mi.max(0.0)
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
