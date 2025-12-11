//! Dynamic Time Warping (DTW) distance for time series.
//!
//! DTW is a distance measure that allows for elastic alignment between time series.

/// Compute the Dynamic Time Warping distance between two time series.
///
/// # Arguments
/// * `a` - First time series
/// * `b` - Second time series
///
/// # Returns
/// DTW distance (lower is more similar)
pub fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let n = a.len();
    let m = b.len();

    // DTW matrix
    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dtw[i][j] = cost + dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
        }
    }

    dtw[n][m]
}

/// Compute DTW distance with a Sakoe-Chiba band constraint.
///
/// The band constraint limits warping to within `window` positions,
/// which improves computational efficiency and can prevent pathological alignments.
///
/// # Arguments
/// * `a` - First time series
/// * `b` - Second time series
/// * `window` - Maximum warping window size
pub fn dtw_distance_windowed(a: &[f64], b: &[f64], window: usize) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let n = a.len();
    let m = b.len();

    // Ensure window is at least |n - m|
    let window = window.max(n.abs_diff(m));

    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        let j_start = 1.max(i.saturating_sub(window));
        let j_end = m.min(i + window);

        for j in j_start..=j_end {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dtw[i][j] = cost + dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
        }
    }

    dtw[n][m]
}

/// Compute the DTW alignment path.
///
/// Returns pairs of indices (i, j) showing how elements are aligned.
pub fn dtw_path(a: &[f64], b: &[f64]) -> Vec<(usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let n = a.len();
    let m = b.len();

    // Compute full DTW matrix
    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dtw[i][j] = cost + dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
        }
    }

    // Backtrack to find path
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));

        let diag = dtw[i - 1][j - 1];
        let left = dtw[i][j - 1];
        let up = dtw[i - 1][j];

        if diag <= left && diag <= up {
            i -= 1;
            j -= 1;
        } else if left < up {
            j -= 1;
        } else {
            i -= 1;
        }
    }

    path.reverse();
    path
}

/// Compute pairwise DTW distance matrix for multiple time series.
///
/// # Arguments
/// * `series` - Vector of time series
///
/// # Returns
/// Symmetric distance matrix
pub fn dtw_pairwise(series: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = series.len();
    let mut dist_matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dtw_distance(&series[i], &series[j]);
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }

    dist_matrix
}

/// Normalized DTW distance (divided by path length).
pub fn dtw_distance_normalized(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let path = dtw_path(a, b);
    if path.is_empty() {
        return f64::INFINITY;
    }

    let distance = dtw_distance(a, b);
    distance / path.len() as f64
}

/// Euclidean distance for same-length time series.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Manhattan (L1) distance for same-length time series.
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== dtw_distance ====================

    #[test]
    fn dtw_identical_series() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_relative_eq!(dtw_distance(&a, &b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn dtw_shifted_series() {
        // DTW should handle shifted series better than Euclidean
        let a = vec![0.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let b = vec![0.0, 1.0, 2.0, 1.0, 0.0, 0.0];

        let dtw_dist = dtw_distance(&a, &b);
        let eucl_dist = euclidean_distance(&a, &b);

        // DTW should be smaller because it can align the peaks
        assert!(dtw_dist <= eucl_dist);
    }

    #[test]
    fn dtw_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let dist = dtw_distance(&a, &b);
        assert!(!dist.is_nan());
        assert!(dist > 0.0);
    }

    #[test]
    fn dtw_empty() {
        assert_eq!(dtw_distance(&[], &[1.0, 2.0]), f64::INFINITY);
        assert_eq!(dtw_distance(&[1.0, 2.0], &[]), f64::INFINITY);
        assert_eq!(dtw_distance(&[], &[]), f64::INFINITY);
    }

    #[test]
    fn dtw_single_element() {
        let a = vec![5.0];
        let b = vec![3.0];
        assert_relative_eq!(dtw_distance(&a, &b), 2.0, epsilon = 1e-10);
    }

    // ==================== dtw_distance_windowed ====================

    #[test]
    fn dtw_windowed_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_relative_eq!(dtw_distance_windowed(&a, &b, 2), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn dtw_windowed_vs_full() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1, 2.1, 3.1, 4.1, 5.1];

        let full = dtw_distance(&a, &b);
        let windowed = dtw_distance_windowed(&a, &b, 1);

        // Windowed should be >= full DTW (more constrained)
        assert!(windowed >= full - 1e-10);
    }

    // ==================== dtw_path ====================

    #[test]
    fn dtw_path_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let path = dtw_path(&a, &b);

        // Should be diagonal path
        assert_eq!(path, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn dtw_path_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 2.5, 3.0];

        let path = dtw_path(&a, &b);

        // Path should start at (0,0) and end at (2,3)
        assert_eq!(path[0], (0, 0));
        assert_eq!(path[path.len() - 1], (2, 3));
    }

    #[test]
    fn dtw_path_empty() {
        assert!(dtw_path(&[], &[1.0]).is_empty());
        assert!(dtw_path(&[1.0], &[]).is_empty());
    }

    // ==================== dtw_pairwise ====================

    #[test]
    fn dtw_pairwise_basic() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let dist_matrix = dtw_pairwise(&series);

        // Self-distance should be 0
        assert_relative_eq!(dist_matrix[0][0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist_matrix[1][1], 0.0, epsilon = 1e-10);

        // Identical series should have 0 distance
        assert_relative_eq!(dist_matrix[0][1], 0.0, epsilon = 1e-10);

        // Symmetric
        assert_relative_eq!(dist_matrix[0][2], dist_matrix[2][0], epsilon = 1e-10);
    }

    #[test]
    fn dtw_pairwise_empty() {
        let series: Vec<Vec<f64>> = vec![];
        let dist_matrix = dtw_pairwise(&series);
        assert!(dist_matrix.is_empty());
    }

    // ==================== dtw_distance_normalized ====================

    #[test]
    fn dtw_normalized_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_relative_eq!(dtw_distance_normalized(&a, &b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn dtw_normalized_different() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];

        let norm_dist = dtw_distance_normalized(&a, &b);
        assert!(norm_dist > 0.0);
        // Normalized should be less than non-normalized for typical cases
        assert!(norm_dist <= dtw_distance(&a, &b));
    }

    // ==================== euclidean_distance ====================

    #[test]
    fn euclidean_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        assert_relative_eq!(euclidean_distance(&a, &b), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn euclidean_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        assert_relative_eq!(euclidean_distance(&a, &b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn euclidean_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        assert_eq!(euclidean_distance(&a, &b), f64::INFINITY);
    }

    // ==================== manhattan_distance ====================

    #[test]
    fn manhattan_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        assert_relative_eq!(manhattan_distance(&a, &b), 7.0, epsilon = 1e-10);
    }

    #[test]
    fn manhattan_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        assert_relative_eq!(manhattan_distance(&a, &b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn manhattan_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        assert_eq!(manhattan_distance(&a, &b), f64::INFINITY);
    }
}
