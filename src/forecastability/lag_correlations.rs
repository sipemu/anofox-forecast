//! Lag-curve correlation functions: Pearson, Spearman, Kendall.
//!
//! Each computes `|corr(X_t, X_{t+h})|` for h = 1..max_lag, returning a
//! vector of absolute correlation coefficients.

/// Pearson correlation lag curve: `|ρ(X_t, X_{t+h})|` for h = 1..max_lag.
pub fn pearson_curve(series: &[f64], max_lag: usize) -> Vec<f64> {
    lag_curve(series, max_lag, pearson_abs)
}

/// Spearman rank correlation lag curve: `|ρ_s(X_t, X_{t+h})|` for h = 1..max_lag.
pub fn spearman_curve(series: &[f64], max_lag: usize) -> Vec<f64> {
    lag_curve(series, max_lag, spearman_abs)
}

/// Kendall tau-b correlation lag curve: `|τ_b(X_t, X_{t+h})|` for h = 1..max_lag.
pub fn kendall_curve(series: &[f64], max_lag: usize) -> Vec<f64> {
    lag_curve(series, max_lag, kendall_abs)
}

fn lag_curve(series: &[f64], max_lag: usize, metric: fn(&[f64], &[f64]) -> f64) -> Vec<f64> {
    let n = series.len();
    (1..=max_lag)
        .map(|h| {
            if n <= h + 2 {
                0.0
            } else {
                metric(&series[..n - h], &series[h..])
            }
        })
        .collect()
}

fn pearson_abs(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
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
    (sxy / (sxx * syy).sqrt()).abs()
}

fn spearman_abs(x: &[f64], y: &[f64]) -> f64 {
    let rx = rank(x);
    let ry = rank(y);
    pearson_abs(&rx, &ry)
}

fn kendall_abs(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    // O(n log n) Kendall tau via merge-sort inversion count.
    // Sort pairs by x, then count inversions in the y-order.
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let y_vals: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let inversions = merge_sort_count_inversions(&y_vals);

    let n_pairs = n * (n - 1) / 2;
    if n_pairs == 0 {
        return 0.0;
    }
    let concordant = n_pairs as i64 - 2 * inversions as i64;
    (concordant as f64 / n_pairs as f64).abs()
}

/// Count inversions in a slice via merge sort. O(n log n).
/// An inversion is a pair (i, j) with i < j but a[i] > a[j].
fn merge_sort_count_inversions(arr: &[f64]) -> usize {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let mut left = arr[..mid].to_vec();
    let mut right = arr[mid..].to_vec();

    let mut count = 0;
    count += merge_sort_count_mut(&mut left);
    count += merge_sort_count_mut(&mut right);

    // Merge and count split inversions.
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    let mut merged = vec![0.0; n];
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            merged[k] = left[i];
            i += 1;
        } else {
            merged[k] = right[j];
            count += left.len() - i; // all remaining left elements are inversions
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        merged[k] = left[i];
        i += 1;
        k += 1;
    }
    while j < right.len() {
        merged[k] = right[j];
        j += 1;
        k += 1;
    }
    // Not needed externally but keeps the function self-contained.
    let _ = merged;
    count
}

fn merge_sort_count_mut(arr: &mut [f64]) -> usize {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let mut count = 0;
    count += merge_sort_count_mut(&mut arr[..mid]);
    count += merge_sort_count_mut(&mut arr[mid..]);

    // Merge in place via temp buffer.
    let left = arr[..mid].to_vec();
    let right = arr[mid..].to_vec();
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            arr[k] = left[i];
            i += 1;
        } else {
            arr[k] = right[j];
            count += left.len() - i;
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        arr[k] = left[i];
        i += 1;
        k += 1;
    }
    while j < right.len() {
        arr[k] = right[j];
        j += 1;
        k += 1;
    }
    count
}

/// Compute fractional ranks (1-based, average ties).
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(f64, usize)> = values.iter().copied().zip(0..).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-15 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 0.5;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.1] = avg;
        }
        i = j;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pearson_perfect_linear() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * v + 2.0).collect();
        assert_relative_eq!(pearson_abs(&x, &y), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn spearman_detects_monotonic() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.powi(3)).collect();
        assert_relative_eq!(spearman_abs(&x, &y), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn kendall_perfect_concordance() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y = x.clone();
        assert_relative_eq!(kendall_abs(&x, &y), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn pearson_curve_decays_for_ar1() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 500;
        let mut s = vec![0.0; n];
        for i in 1..n {
            s[i] = 0.7 * s[i - 1] + (rng.gen::<f64>() - 0.5) * 2.0;
        }
        let curve = pearson_curve(&s, 5);
        assert!(curve[0] > curve[4], "Pearson should decay for AR(1)");
    }
}
