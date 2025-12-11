//! Entropy-based features for time series.
//!
//! Provides features based on information-theoretic measures.

/// Returns the sample entropy of the time series.
///
/// Sample entropy measures the complexity/regularity of a time series.
/// Lower values indicate more regularity.
///
/// # Arguments
/// * `series` - Input time series
/// * `m` - Embedding dimension (typically 2)
/// * `r` - Tolerance (typically 0.2 * std)
pub fn sample_entropy(series: &[f64], m: usize, r: f64) -> f64 {
    if series.len() < m + 2 {
        return f64::NAN;
    }

    let n = series.len();

    // Count template matches for dimension m and m+1
    let count_m = count_matches(series, m, r);
    let count_m1 = count_matches(series, m + 1, r);

    if count_m == 0 || count_m1 == 0 {
        return f64::NAN;
    }

    // Sample entropy = -ln(A/B) where A = matches at m+1, B = matches at m
    -((count_m1 as f64) / (count_m as f64)).ln()
}

/// Returns the approximate entropy of the time series.
///
/// Similar to sample entropy but includes self-matches.
///
/// # Arguments
/// * `series` - Input time series
/// * `m` - Embedding dimension (typically 2)
/// * `r` - Tolerance (typically 0.2 * std)
pub fn approximate_entropy(series: &[f64], m: usize, r: f64) -> f64 {
    if series.len() < m + 2 {
        return f64::NAN;
    }

    let phi_m = phi(series, m, r);
    let phi_m1 = phi(series, m + 1, r);

    if phi_m.is_nan() || phi_m1.is_nan() {
        return f64::NAN;
    }

    phi_m - phi_m1
}

/// Helper: compute phi for approximate entropy
fn phi(series: &[f64], m: usize, r: f64) -> f64 {
    let n = series.len();
    if n < m {
        return f64::NAN;
    }

    let n_templates = n - m + 1;
    let mut sum = 0.0;

    for i in 0..n_templates {
        let mut count = 0;
        for j in 0..n_templates {
            if templates_match(series, i, j, m, r) {
                count += 1;
            }
        }
        if count > 0 {
            sum += (count as f64 / n_templates as f64).ln();
        }
    }

    sum / n_templates as f64
}

/// Helper: count matches for sample entropy (excluding self-matches)
fn count_matches(series: &[f64], m: usize, r: f64) -> usize {
    let n = series.len();
    if n < m {
        return 0;
    }

    let n_templates = n - m;
    let mut count = 0;

    for i in 0..n_templates {
        for j in (i + 1)..n_templates {
            if templates_match(series, i, j, m, r) {
                count += 2; // Count both (i,j) and (j,i)
            }
        }
    }

    count
}

/// Helper: check if two templates match within tolerance
fn templates_match(series: &[f64], i: usize, j: usize, m: usize, r: f64) -> bool {
    for k in 0..m {
        if (series[i + k] - series[j + k]).abs() > r {
            return false;
        }
    }
    true
}

/// Returns the permutation entropy of the time series.
///
/// Based on the frequency distribution of ordinal patterns.
///
/// # Arguments
/// * `series` - Input time series
/// * `order` - Order of permutation patterns (typically 3-7)
/// * `delay` - Time delay between elements (typically 1)
pub fn permutation_entropy(series: &[f64], order: usize, delay: usize) -> f64 {
    if order < 2 || series.len() < (order - 1) * delay + 1 {
        return f64::NAN;
    }

    let n_patterns = series.len() - (order - 1) * delay;
    let mut pattern_counts = std::collections::HashMap::new();

    for i in 0..n_patterns {
        let pattern = get_ordinal_pattern(series, i, order, delay);
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    // Compute entropy
    let mut entropy = 0.0;
    for &count in pattern_counts.values() {
        let p = count as f64 / n_patterns as f64;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    // Normalize by maximum possible entropy
    let max_entropy = (factorial(order) as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        entropy
    }
}

/// Helper: get ordinal pattern as a vector of ranks
fn get_ordinal_pattern(series: &[f64], start: usize, order: usize, delay: usize) -> Vec<usize> {
    let values: Vec<f64> = (0..order).map(|k| series[start + k * delay]).collect();

    // Get ranks
    let mut indices: Vec<usize> = (0..order).collect();
    indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0; order];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank;
    }

    ranks
}

/// Helper: compute factorial
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Returns the binned entropy of the time series.
///
/// Entropy of the histogram of values.
///
/// # Arguments
/// * `series` - Input time series
/// * `max_bins` - Maximum number of bins
pub fn binned_entropy(series: &[f64], max_bins: usize) -> f64 {
    if series.is_empty() || max_bins == 0 {
        return f64::NAN;
    }

    let min_val = series.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return 0.0; // Constant series has zero entropy
    }

    let n_bins = max_bins.min(series.len());
    let bin_width = (max_val - min_val) / n_bins as f64;

    let mut counts = vec![0usize; n_bins];

    for &x in series {
        let bin = ((x - min_val) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1); // Handle edge case where x == max_val
        counts[bin] += 1;
    }

    // Compute entropy
    let n = series.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Returns the spectral (Fourier) entropy of the time series.
///
/// Entropy of the power spectral density.
///
/// This is a simplified version using DFT computed manually.
pub fn fourier_entropy(series: &[f64]) -> f64 {
    if series.len() < 4 {
        return f64::NAN;
    }

    // Compute power spectral density using DFT
    let psd = compute_psd(series);

    if psd.is_empty() {
        return f64::NAN;
    }

    // Normalize PSD to get probability distribution
    let total: f64 = psd.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }

    // Compute entropy
    let mut entropy = 0.0;
    for &p in &psd {
        let prob = p / total;
        if prob > 1e-10 {
            entropy -= prob * prob.ln();
        }
    }

    entropy
}

/// Helper: compute power spectral density using DFT
fn compute_psd(series: &[f64]) -> Vec<f64> {
    let n = series.len();
    let mut psd = Vec::with_capacity(n / 2);

    for k in 0..n / 2 {
        let mut real = 0.0;
        let mut imag = 0.0;

        for (t, &x) in series.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
            real += x * angle.cos();
            imag += x * angle.sin();
        }

        psd.push((real * real + imag * imag) / n as f64);
    }

    psd
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== sample_entropy ====================

    #[test]
    fn sample_entropy_regular() {
        // Regular periodic signal should have low entropy
        let series: Vec<f64> = (0..100)
            .map(|i| ((i % 10) as f64 * std::f64::consts::PI / 5.0).sin())
            .collect();
        let se = sample_entropy(&series, 2, 0.2);
        assert!(!se.is_nan());
        // Regular signals have lower entropy
    }

    #[test]
    fn sample_entropy_random() {
        // More random signals should have higher entropy
        let series: Vec<f64> = (0..100).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let se = sample_entropy(&series, 2, 0.5);
        assert!(!se.is_nan());
    }

    #[test]
    fn sample_entropy_constant() {
        let series = vec![5.0; 50];
        let se = sample_entropy(&series, 2, 0.1);
        // Constant series: all templates match, so se should be 0 or very low
        assert!(se.is_nan() || se.abs() < 0.1);
    }

    #[test]
    fn sample_entropy_short() {
        assert!(sample_entropy(&[], 2, 0.2).is_nan());
        assert!(sample_entropy(&[1.0, 2.0], 2, 0.2).is_nan());
    }

    // ==================== approximate_entropy ====================

    #[test]
    fn approximate_entropy_regular() {
        let series: Vec<f64> = (0..50).map(|i| (i as f64 * 0.5).sin()).collect();
        let ae = approximate_entropy(&series, 2, 0.2);
        assert!(!ae.is_nan());
    }

    #[test]
    fn approximate_entropy_constant() {
        let series = vec![5.0; 50];
        let ae = approximate_entropy(&series, 2, 0.1);
        // Constant series should have low entropy
        assert!(!ae.is_nan());
        assert!(ae.abs() < 0.5);
    }

    #[test]
    fn approximate_entropy_short() {
        assert!(approximate_entropy(&[], 2, 0.2).is_nan());
        assert!(approximate_entropy(&[1.0, 2.0], 2, 0.2).is_nan());
    }

    // ==================== permutation_entropy ====================

    #[test]
    fn permutation_entropy_monotonic() {
        // Monotonically increasing: only one pattern
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let pe = permutation_entropy(&series, 3, 1);
        assert!(!pe.is_nan());
        assert!(pe < 0.1, "Monotonic should have low PE, got {}", pe);
    }

    #[test]
    fn permutation_entropy_alternating() {
        // Alternating: two patterns
        let series: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let pe = permutation_entropy(&series, 3, 1);
        assert!(!pe.is_nan());
    }

    #[test]
    fn permutation_entropy_random_like() {
        // More diverse patterns should have higher entropy
        let series: Vec<f64> = (0..50).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let pe = permutation_entropy(&series, 3, 1);
        assert!(!pe.is_nan());
        assert!(pe > 0.5, "Random-like should have higher PE, got {}", pe);
    }

    #[test]
    fn permutation_entropy_short() {
        assert!(permutation_entropy(&[1.0, 2.0], 3, 1).is_nan());
        assert!(permutation_entropy(&[1.0, 2.0, 3.0], 5, 1).is_nan());
    }

    #[test]
    fn permutation_entropy_invalid_order() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(permutation_entropy(&series, 1, 1).is_nan()); // Order must be >= 2
    }

    // ==================== binned_entropy ====================

    #[test]
    fn binned_entropy_uniform() {
        // Uniformly distributed should have high entropy
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let be = binned_entropy(&series, 10);
        assert!(!be.is_nan());
        assert!(be > 1.0, "Uniform should have high entropy, got {}", be);
    }

    #[test]
    fn binned_entropy_constant() {
        let series = vec![5.0; 100];
        let be = binned_entropy(&series, 10);
        assert_relative_eq!(be, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn binned_entropy_bimodal() {
        // Two clusters should have lower entropy than uniform
        let mut series = vec![0.0; 50];
        series.extend(vec![100.0; 50]);
        let be = binned_entropy(&series, 10);
        assert!(!be.is_nan());
    }

    #[test]
    fn binned_entropy_empty() {
        assert!(binned_entropy(&[], 10).is_nan());
    }

    #[test]
    fn binned_entropy_zero_bins() {
        let series = vec![1.0, 2.0, 3.0];
        assert!(binned_entropy(&series, 0).is_nan());
    }

    // ==================== fourier_entropy ====================

    #[test]
    fn fourier_entropy_sine() {
        // Pure sine wave should have low spectral entropy (concentrated at one frequency)
        let series: Vec<f64> = (0..64)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin())
            .collect();
        let fe = fourier_entropy(&series);
        assert!(!fe.is_nan());
    }

    #[test]
    fn fourier_entropy_white_noise_like() {
        // More random signal should have higher spectral entropy
        let series: Vec<f64> = (0..64).map(|i| ((i * 7 + 3) % 13) as f64 - 6.0).collect();
        let fe = fourier_entropy(&series);
        assert!(!fe.is_nan());
    }

    #[test]
    fn fourier_entropy_constant() {
        let series = vec![5.0; 64];
        let fe = fourier_entropy(&series);
        // DC component only
        assert!(!fe.is_nan());
    }

    #[test]
    fn fourier_entropy_short() {
        assert!(fourier_entropy(&[]).is_nan());
        assert!(fourier_entropy(&[1.0, 2.0]).is_nan());
    }
}
