//! SIMD-accelerated primitives via Trueno (f32 internal).
//!
//! This module provides high-performance vector operations using CPU SIMD
//! instructions (AVX2/SSE2/NEON). Functions accept `f64` slices (matching
//! the library's public API), convert internally to `f32` for SIMD computation,
//! and return `f64` results.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::simd;
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Basic operations
//! let total = simd::sum(&data);
//! let avg = simd::mean(&data);
//! let var = simd::variance(&data);
//!
//! // Distance calculations
//! let a = vec![1.0, 2.0, 3.0];
//! let b = vec![4.0, 5.0, 6.0];
//! let dist = simd::squared_distance(&a, &b).sqrt();
//! ```
//!
//! # Precision
//!
//! Using f32 internally provides ~7 decimal digits of precision, which is
//! sufficient for statistical aggregates. For precision-critical operations
//! (p-values, convergence checks), use the standard f64 implementations.

use trueno::Vector;

// ============================================================================
// Internal Helpers
// ============================================================================

/// Convert f64 slice to f32 Vec for SIMD processing
#[inline]
fn to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&x| x as f32).collect()
}

/// Convert f32 slice to f64 Vec for output
#[inline]
fn to_f64(data: &[f32]) -> Vec<f64> {
    data.iter().map(|&x| x as f64).collect()
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum of all elements.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::sum;
/// assert!((sum(&[1.0, 2.0, 3.0, 4.0]) - 10.0).abs() < 1e-6);
/// ```
#[inline]
pub fn sum(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .sum()
        .map(|s| s as f64)
        .unwrap_or_else(|_| data.iter().sum())
}

/// Sum of squared elements (x·x).
///
/// Equivalent to the dot product of a vector with itself.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::sum_of_squares;
/// assert!((sum_of_squares(&[1.0, 2.0, 3.0]) - 14.0).abs() < 1e-6); // 1 + 4 + 9
/// ```
#[inline]
pub fn sum_of_squares(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .sum_of_squares()
        .map(|s| s as f64)
        .unwrap_or_else(|_| data.iter().map(|x| x * x).sum())
}

/// Dot product of two vectors.
///
/// Returns the sum of element-wise products: Σ(a\[i\] * b\[i\]).
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::dot;
/// assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-5); // 4 + 10 + 18
/// ```
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have same length for dot product"
    );
    if a.is_empty() {
        return 0.0;
    }
    let a_f32 = to_f32(a);
    let b_f32 = to_f32(b);
    let va = Vector::from_slice(&a_f32);
    let vb = Vector::from_slice(&b_f32);
    va.dot(&vb)
        .map(|s| s as f64)
        .unwrap_or_else(|_| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Mean (average) of all elements.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::mean;
/// assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-6);
/// ```
#[inline]
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .mean()
        .map(|s| s as f64)
        .unwrap_or_else(|_| data.iter().sum::<f64>() / data.len() as f64)
}

/// Population variance.
///
/// Computes Σ(x - μ)² / n where μ is the mean.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::variance;
/// let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let var = variance(&data);
/// assert!((var - 4.0).abs() < 1e-5);
/// ```
#[inline]
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .variance()
        .map(|s| s as f64)
        .unwrap_or_else(|_| {
            let m = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
        })
}

/// Sample variance (with Bessel's correction).
///
/// Computes Σ(x - μ)² / (n-1) where μ is the mean.
#[inline]
pub fn variance_sample(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let n = data.len() as f64;
    variance(data) * n / (n - 1.0)
}

/// Population standard deviation.
#[inline]
pub fn stddev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Find the maximum value.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::max;
/// assert!((max(&[1.0, 5.0, 3.0, 2.0]) - 5.0).abs() < 1e-6);
/// ```
#[inline]
pub fn max(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .max()
        .map(|s| s as f64)
        .unwrap_or_else(|_| data.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}

/// Find the minimum value.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::min;
/// assert!((min(&[1.0, 5.0, 3.0, 2.0]) - 1.0).abs() < 1e-6);
/// ```
#[inline]
pub fn min(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::INFINITY;
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .min()
        .map(|s| s as f64)
        .unwrap_or_else(|_| data.iter().cloned().fold(f64::INFINITY, f64::min))
}

// ============================================================================
// Distance Operations
// ============================================================================

/// Squared Euclidean distance: Σ(a\[i\] - b\[i\])².
///
/// Use `.sqrt()` on the result for actual Euclidean distance.
///
/// Note: This uses scalar f64 implementation because the overhead of converting
/// two vectors from f64→f32 exceeds the SIMD performance gains.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::squared_distance;
/// let dist = squared_distance(&[0.0, 0.0], &[3.0, 4.0]);
/// assert!((dist - 25.0).abs() < 1e-5); // 9 + 16
/// assert!((dist.sqrt() - 5.0).abs() < 1e-5);
/// ```
#[inline]
pub fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have same length for distance"
    );
    // Use scalar f64 - conversion overhead for two vectors exceeds SIMD gains
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Manhattan (L1) distance: Σ|a\[i\] - b\[i\]|.
///
/// Note: This uses scalar f64 implementation because the overhead of converting
/// two vectors from f64→f32 exceeds the SIMD performance gains.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::l1_distance;
/// let dist = l1_distance(&[0.0, 0.0], &[3.0, 4.0]);
/// assert!((dist - 7.0).abs() < 1e-5); // 3 + 4
/// ```
#[inline]
pub fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "vectors must have same length for distance"
    );
    // Use scalar f64 - conversion overhead for two vectors exceeds SIMD gains
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise subtraction: a\[i\] - b\[i\].
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::sub;
/// let result = sub(&[5.0, 4.0, 3.0], &[1.0, 2.0, 3.0]);
/// assert!((result[0] - 4.0).abs() < 1e-6);
/// assert!((result[1] - 2.0).abs() < 1e-6);
/// assert!((result[2] - 0.0).abs() < 1e-6);
/// ```
#[inline]
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    if a.is_empty() {
        return Vec::new();
    }
    let a_f32 = to_f32(a);
    let b_f32 = to_f32(b);
    let va = Vector::from_slice(&a_f32);
    let vb = Vector::from_slice(&b_f32);
    va.sub(&vb)
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
}

/// Element-wise multiplication: a\[i\] * b\[i\].
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::mul;
/// let result = mul(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]);
/// assert!((result[0] - 10.0).abs() < 1e-5);
/// assert!((result[1] - 18.0).abs() < 1e-5);
/// assert!((result[2] - 28.0).abs() < 1e-5);
/// ```
#[inline]
pub fn mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    if a.is_empty() {
        return Vec::new();
    }
    let a_f32 = to_f32(a);
    let b_f32 = to_f32(b);
    let va = Vector::from_slice(&a_f32);
    let vb = Vector::from_slice(&b_f32);
    va.mul(&vb)
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

/// Element-wise division: a\[i\] / b\[i\].
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::div;
/// let result = div(&[10.0, 18.0, 28.0], &[2.0, 3.0, 4.0]);
/// assert!((result[0] - 5.0).abs() < 1e-5);
/// assert!((result[1] - 6.0).abs() < 1e-5);
/// assert!((result[2] - 7.0).abs() < 1e-5);
/// ```
#[inline]
pub fn div(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    if a.is_empty() {
        return Vec::new();
    }
    let a_f32 = to_f32(a);
    let b_f32 = to_f32(b);
    let va = Vector::from_slice(&a_f32);
    let vb = Vector::from_slice(&b_f32);
    va.div(&vb)
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
}

/// Element-wise addition: a\[i\] + b\[i\].
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::add;
/// let result = add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
/// assert!((result[0] - 5.0).abs() < 1e-6);
/// assert!((result[1] - 7.0).abs() < 1e-6);
/// assert!((result[2] - 9.0).abs() < 1e-6);
/// ```
#[inline]
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    if a.is_empty() {
        return Vec::new();
    }
    let a_f32 = to_f32(a);
    let b_f32 = to_f32(b);
    let va = Vector::from_slice(&a_f32);
    let vb = Vector::from_slice(&b_f32);
    va.add(&vb)
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

/// Scale all elements by a constant: data\[i\] * scalar.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::scale;
/// let result = scale(&[1.0, 2.0, 3.0], 2.0);
/// assert!((result[0] - 2.0).abs() < 1e-6);
/// assert!((result[1] - 4.0).abs() < 1e-6);
/// assert!((result[2] - 6.0).abs() < 1e-6);
/// ```
#[inline]
pub fn scale(data: &[f64], scalar: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let f32_data = to_f32(data);
    let scalar_f32 = scalar as f32;
    Vector::from_slice(&f32_data)
        .scale(scalar_f32)
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| data.iter().map(|x| x * scalar).collect())
}

// ============================================================================
// Correlation Operations
// ============================================================================

/// Pearson correlation coefficient between two vectors.
///
/// Computes `cov(x, y) / (std(x) * std(y))` using SIMD-accelerated
/// dot products and sums. Returns a value in [-1, 1].
///
/// # Panics
///
/// Panics if the vectors have different lengths.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::correlation;
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// assert!((correlation(&x, &y) - 1.0).abs() < 1e-5);
/// ```
#[inline]
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(
        x.len(),
        y.len(),
        "vectors must have same length for correlation"
    );
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);
    let n_f64 = n as f64;

    // cov = dot(x, y) / n - mean_x * mean_y
    let dot_xy = dot(x, y);
    let cov = dot_xy / n_f64 - mean_x * mean_y;

    // var_x = dot(x, x) / n - mean_x^2
    let var_x = sum_of_squares(x) / n_f64 - mean_x * mean_x;
    // var_y = dot(y, y) / n - mean_y^2
    let var_y = sum_of_squares(y) / n_f64 - mean_y * mean_y;

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Autocorrelation at a given lag.
///
/// Computes the Pearson autocorrelation of `data` with itself shifted by `lag`
/// positions, using SIMD-accelerated operations. Uses the standard normalization
/// where the denominator is the variance of the full series.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::autocorrelation;
/// let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
/// // Lag 0 should give 1.0
/// assert!((autocorrelation(&data, 0) - 1.0).abs() < 1e-5);
/// ```
#[inline]
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    let n = data.len();
    if n <= lag {
        return f64::NAN;
    }

    let m = mean(data);

    // Population variance of the full series via SIMD
    let var = sum_of_squares(data) / n as f64 - m * m;

    if var < 1e-10 {
        return 0.0;
    }

    // Compute the lagged cross-product sum using SIMD:
    // sum((data[lag..] - m) * (data[..n-lag] - m))
    let head = &data[..n - lag]; // data[i - lag] for i in lag..n
    let tail = &data[lag..]; // data[i] for i in lag..n

    let dot_val = dot(tail, head);
    let sum_tail = sum(tail);
    let sum_head = sum(head);
    let n_effective = (n - lag) as f64;

    // Expand (tail[i] - m)(head[i] - m) = tail[i]*head[i] - m*head[i] - m*tail[i] + m^2
    let cross_sum = dot_val - m * sum_head - m * sum_tail + m * m * n_effective;

    cross_sum / (n as f64 * var)
}

// ============================================================================
// Normalization Operations
// ============================================================================

/// Z-score normalization (standardization).
///
/// Transforms data to have mean 0 and standard deviation 1.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::zscore;
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let normalized = zscore(&data);
/// // Mean should be ~0
/// let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
/// assert!(mean.abs() < 1e-5);
/// ```
#[inline]
pub fn zscore(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return data.to_vec();
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .zscore()
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| {
            let m = mean(data);
            let s = stddev(data);
            if s.abs() < f64::EPSILON {
                data.to_vec()
            } else {
                data.iter().map(|x| (x - m) / s).collect()
            }
        })
}

/// Min-max normalization to [0, 1] range.
///
/// # Example
///
/// ```
/// use anofox_forecast::simd::minmax_normalize;
/// let data = vec![0.0, 50.0, 100.0];
/// let normalized = minmax_normalize(&data);
/// assert!((normalized[0] - 0.0).abs() < 1e-6);
/// assert!((normalized[1] - 0.5).abs() < 1e-5);
/// assert!((normalized[2] - 1.0).abs() < 1e-6);
/// ```
#[inline]
pub fn minmax_normalize(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let f32_data = to_f32(data);
    Vector::from_slice(&f32_data)
        .minmax_normalize()
        .map(|v| to_f64(v.as_slice()))
        .unwrap_or_else(|_| {
            let min_val = min(data);
            let max_val = max(data);
            let range = max_val - min_val;
            if range.abs() < f64::EPSILON {
                vec![0.0; data.len()]
            } else {
                data.iter().map(|x| (x - min_val) / range).collect()
            }
        })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Use looser tolerance for f32 conversions
    const EPSILON: f64 = 1e-5;

    fn assert_close(a: f64, b: f64, msg: &str) {
        assert!(
            (a - b).abs() < EPSILON,
            "{}: expected {}, got {}, diff = {}",
            msg,
            b,
            a,
            (a - b).abs()
        );
    }

    #[test]
    fn test_sum() {
        assert_close(sum(&[1.0, 2.0, 3.0, 4.0]), 10.0, "sum");
        assert_close(sum(&[]), 0.0, "empty sum");
        assert_close(sum(&[5.0]), 5.0, "single element");
        // Test with more elements
        let large: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_close(sum(&large), 5050.0, "large sum");
    }

    #[test]
    fn test_sum_of_squares() {
        assert_close(sum_of_squares(&[1.0, 2.0, 3.0]), 14.0, "sum_of_squares");
        assert_close(sum_of_squares(&[]), 0.0, "empty");
        assert_close(sum_of_squares(&[1.0, 2.0, 3.0, 4.0, 5.0]), 55.0, "larger");
    }

    #[test]
    fn test_dot() {
        assert_close(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0, "dot");
        assert_close(dot(&[], &[]), 0.0, "empty dot");
    }

    #[test]
    fn test_mean() {
        assert_close(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, "mean");
        assert!(mean(&[]).is_nan(), "empty mean should be NaN");
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_close(variance(&data), 4.0, "variance");
        assert!(variance(&[]).is_nan(), "empty variance");
        assert!(variance(&[1.0]).is_nan(), "single element variance");
    }

    #[test]
    fn test_squared_distance() {
        assert_close(
            squared_distance(&[0.0, 0.0], &[3.0, 4.0]),
            25.0,
            "squared_distance",
        );
        assert_close(squared_distance(&[], &[]), 0.0, "empty distance");
    }

    #[test]
    fn test_l1_distance() {
        assert_close(l1_distance(&[0.0, 0.0], &[3.0, 4.0]), 7.0, "l1_distance");
    }

    #[test]
    fn test_sub() {
        let result = sub(&[5.0, 4.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);
        assert_close(result[0], 4.0, "sub[0]");
        assert_close(result[1], 2.0, "sub[1]");
        assert_close(result[2], 0.0, "sub[2]");
    }

    #[test]
    fn test_mul() {
        let result = mul(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]);
        assert_close(result[0], 10.0, "mul[0]");
        assert_close(result[1], 18.0, "mul[1]");
        assert_close(result[2], 28.0, "mul[2]");
    }

    #[test]
    fn test_div() {
        let result = div(&[10.0, 18.0, 28.0], &[2.0, 3.0, 4.0]);
        assert_close(result[0], 5.0, "div[0]");
        assert_close(result[1], 6.0, "div[1]");
        assert_close(result[2], 7.0, "div[2]");
    }

    #[test]
    fn test_scale() {
        let result = scale(&[1.0, 2.0, 3.0], 2.0);
        assert_close(result[0], 2.0, "scale[0]");
        assert_close(result[1], 4.0, "scale[1]");
        assert_close(result[2], 6.0, "scale[2]");
    }

    #[test]
    fn test_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = zscore(&data);
        assert_eq!(normalized.len(), 5);
        // Mean should be close to 0
        let m: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(m.abs() < EPSILON, "zscore mean should be ~0, got {}", m);
    }

    #[test]
    fn test_minmax_normalize() {
        let result = minmax_normalize(&[0.0, 50.0, 100.0]);
        assert_close(result[0], 0.0, "minmax[0]");
        assert_close(result[1], 0.5, "minmax[1]");
        assert_close(result[2], 1.0, "minmax[2]");
    }

    #[test]
    fn test_max_min() {
        assert_close(max(&[1.0, 5.0, 3.0, 2.0]), 5.0, "max");
        assert_close(min(&[1.0, 5.0, 3.0, 2.0]), 1.0, "min");
    }

    #[test]
    fn test_variance_sample() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let pop_var = variance(&data);
        let sample_var = variance_sample(&data);
        // Sample variance = pop_variance * n / (n-1)
        let expected = pop_var * 8.0 / 7.0;
        assert_close(sample_var, expected, "variance_sample");

        // Edge cases
        assert!(variance_sample(&[]).is_nan(), "empty variance_sample");
        assert!(
            variance_sample(&[1.0]).is_nan(),
            "single element variance_sample"
        );
    }

    #[test]
    fn test_stddev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let expected = variance(&data).sqrt();
        assert_close(stddev(&data), expected, "stddev");
        assert_close(stddev(&data), 2.0, "stddev value");

        // Edge cases
        assert!(stddev(&[]).is_nan(), "empty stddev");
        assert!(stddev(&[1.0]).is_nan(), "single element stddev");
    }

    #[test]
    fn test_add() {
        let result = add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_eq!(result.len(), 3);
        assert_close(result[0], 5.0, "add[0]");
        assert_close(result[1], 7.0, "add[1]");
        assert_close(result[2], 9.0, "add[2]");

        // Empty case
        let empty_result = add(&[], &[]);
        assert!(empty_result.is_empty(), "empty add");
    }

    #[test]
    fn test_sub_empty() {
        let result = sub(&[], &[]);
        assert!(result.is_empty(), "empty sub");
    }

    #[test]
    fn test_mul_empty() {
        let result = mul(&[], &[]);
        assert!(result.is_empty(), "empty mul");
    }

    #[test]
    fn test_div_empty() {
        let result = div(&[], &[]);
        assert!(result.is_empty(), "empty div");
    }

    #[test]
    fn test_scale_empty() {
        let result = scale(&[], 2.0);
        assert!(result.is_empty(), "empty scale");
    }

    #[test]
    fn test_zscore_edge_cases() {
        // Single element returns unchanged
        let single = zscore(&[5.0]);
        assert_eq!(single.len(), 1);
        assert_close(single[0], 5.0, "zscore single");

        // Empty returns empty
        let empty = zscore(&[]);
        assert!(empty.is_empty(), "zscore empty");

        // Constant data (stddev = 0) returns unchanged
        let constant = zscore(&[3.0, 3.0, 3.0, 3.0]);
        assert_eq!(constant.len(), 4);
        // Should return original data when stddev is 0
        for &v in &constant {
            assert_close(v, 3.0, "zscore constant");
        }
    }

    #[test]
    fn test_minmax_normalize_edge_cases() {
        // Empty returns empty
        let empty = minmax_normalize(&[]);
        assert!(empty.is_empty(), "minmax empty");

        // Constant data returns zeros
        let constant = minmax_normalize(&[5.0, 5.0, 5.0]);
        assert_eq!(constant.len(), 3);
        for &v in &constant {
            assert_close(v, 0.0, "minmax constant");
        }

        // Single element
        let single = minmax_normalize(&[5.0]);
        assert_eq!(single.len(), 1);
        assert_close(single[0], 0.0, "minmax single");
    }

    #[test]
    fn test_max_empty() {
        assert_eq!(max(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_min_empty() {
        assert_eq!(min(&[]), f64::INFINITY);
    }

    #[test]
    fn test_single_element_sum_of_squares() {
        assert_close(sum_of_squares(&[3.0]), 9.0, "single sum_of_squares");
    }

    #[test]
    fn test_single_element_dot() {
        assert_close(dot(&[3.0], &[4.0]), 12.0, "single dot");
    }

    #[test]
    fn test_single_element_mean() {
        assert_close(mean(&[5.0]), 5.0, "single mean");
    }

    #[test]
    fn test_single_element_max_min() {
        assert_close(max(&[5.0]), 5.0, "single max");
        assert_close(min(&[5.0]), 5.0, "single min");
    }

    #[test]
    fn test_single_element_distances() {
        assert_close(
            squared_distance(&[3.0], &[5.0]),
            4.0,
            "single squared_distance",
        );
        assert_close(l1_distance(&[3.0], &[5.0]), 2.0, "single l1_distance");
    }

    #[test]
    fn test_single_element_elementwise_ops() {
        let sub_result = sub(&[5.0], &[3.0]);
        assert_close(sub_result[0], 2.0, "single sub");

        let mul_result = mul(&[5.0], &[3.0]);
        assert_close(mul_result[0], 15.0, "single mul");

        let div_result = div(&[6.0], &[3.0]);
        assert_close(div_result[0], 2.0, "single div");

        let add_result = add(&[5.0], &[3.0]);
        assert_close(add_result[0], 8.0, "single add");

        let scale_result = scale(&[5.0], 3.0);
        assert_close(scale_result[0], 15.0, "single scale");
    }

    #[test]
    fn test_large_vectors() {
        // Test with vectors larger than typical SIMD width
        let large_a: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let large_b: Vec<f64> = (0..1000).map(|x| (x * 2) as f64).collect();

        // Sum: 0 + 1 + 2 + ... + 999 = 999 * 1000 / 2 = 499500
        assert_close(sum(&large_a), 499500.0, "large sum");

        // Mean
        assert_close(mean(&large_a), 499.5, "large mean");

        // Max/Min
        assert_close(max(&large_a), 999.0, "large max");
        assert_close(min(&large_a), 0.0, "large min");

        // Element-wise operations
        let add_result = add(&large_a, &large_b);
        assert_eq!(add_result.len(), 1000);
        assert_close(add_result[500], 1500.0, "large add");

        let sub_result = sub(&large_b, &large_a);
        assert_eq!(sub_result.len(), 1000);
        assert_close(sub_result[500], 500.0, "large sub");
    }

    #[test]
    fn test_non_multiple_of_four_lengths() {
        // Test with lengths that are not multiples of 4 (typical SIMD width)
        let v3: Vec<f64> = vec![1.0, 2.0, 3.0];
        let v5: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v7: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        assert_close(sum(&v3), 6.0, "sum v3");
        assert_close(sum(&v5), 15.0, "sum v5");
        assert_close(sum(&v7), 28.0, "sum v7");

        assert_close(mean(&v3), 2.0, "mean v3");
        assert_close(mean(&v5), 3.0, "mean v5");
        assert_close(mean(&v7), 4.0, "mean v7");
    }

    #[test]
    fn test_negative_values() {
        let data = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];

        assert_close(sum(&data), 0.0, "negative sum");
        assert_close(mean(&data), 0.0, "negative mean");
        assert_close(max(&data), 5.0, "negative max");
        assert_close(min(&data), -5.0, "negative min");

        let squared = sum_of_squares(&data);
        assert_close(squared, 70.0, "negative sum_of_squares"); // 25+9+1+0+1+9+25

        let a = vec![-1.0, -2.0];
        let b = vec![3.0, 4.0];
        assert_close(l1_distance(&a, &b), 10.0, "negative l1"); // |(-1)-3| + |(-2)-4| = 4 + 6
        assert_close(squared_distance(&a, &b), 52.0, "negative squared"); // 16 + 36
    }

    // ==================== correlation ====================

    /// Naive Pearson correlation for test comparison
    fn naive_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let denom = (var_x * var_y).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_close(correlation(&x, &y), 1.0, "perfect positive correlation");
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_close(correlation(&x, &y), -1.0, "perfect negative correlation");
    }

    #[test]
    fn test_correlation_zero() {
        // Orthogonal-ish vectors
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        assert_close(correlation(&x, &y), 0.0, "zero correlation");
    }

    #[test]
    fn test_correlation_constant() {
        let x = vec![3.0, 3.0, 3.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        assert_close(correlation(&x, &y), 0.0, "constant x correlation");
    }

    #[test]
    fn test_correlation_matches_naive() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();
        let simd_r = correlation(&x, &y);
        let naive_r = naive_correlation(&x, &y);
        assert!(
            (simd_r - naive_r).abs() < 1e-4,
            "correlation mismatch: simd={}, naive={}, diff={}",
            simd_r,
            naive_r,
            (simd_r - naive_r).abs()
        );
    }

    #[test]
    fn test_correlation_large_vectors() {
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..1000).map(|i| (i * 2 + 3) as f64).collect();
        assert_close(correlation(&x, &y), 1.0, "large perfect correlation");
    }

    #[test]
    fn test_correlation_edge_cases() {
        assert!(correlation(&[1.0], &[2.0]).is_nan(), "single element");
        assert!(correlation(&[], &[]).is_nan(), "empty");
    }

    #[test]
    fn test_correlation_non_multiple_of_four() {
        // Test lengths 3, 5, 7 to exercise remainder handling
        for len in [3, 5, 7, 9, 13] {
            let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
            let y: Vec<f64> = (0..len).map(|i| (i * 2 + 1) as f64).collect();
            let simd_r = correlation(&x, &y);
            let naive_r = naive_correlation(&x, &y);
            assert!(
                (simd_r - naive_r).abs() < 1e-4,
                "correlation mismatch for len={}: simd={}, naive={}",
                len,
                simd_r,
                naive_r
            );
        }
    }

    // ==================== autocorrelation ====================

    /// Naive autocorrelation for test comparison
    fn naive_autocorrelation(data: &[f64], lag: usize) -> f64 {
        let n = data.len();
        if n <= lag {
            return f64::NAN;
        }
        let m = data.iter().sum::<f64>() / n as f64;
        let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-10 {
            return 0.0;
        }
        let mut cross_sum = 0.0;
        for i in lag..n {
            cross_sum += (data[i] - m) * (data[i - lag] - m);
        }
        cross_sum / (n as f64 * var)
    }

    #[test]
    fn test_autocorrelation_lag_0() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert_close(autocorrelation(&data, 0), 1.0, "acf lag 0");
    }

    #[test]
    fn test_autocorrelation_linear_trend() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let acf1 = autocorrelation(&data, 1);
        assert!(
            acf1 > 0.8,
            "Expected high ACF(1) for linear trend, got {}",
            acf1
        );
    }

    #[test]
    fn test_autocorrelation_alternating() {
        let data: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let acf1 = autocorrelation(&data, 1);
        assert!(
            acf1 < -0.5,
            "Expected negative ACF(1) for alternating, got {}",
            acf1
        );
    }

    #[test]
    fn test_autocorrelation_constant() {
        let data = vec![5.0; 10];
        assert_close(autocorrelation(&data, 1), 0.0, "constant acf");
    }

    #[test]
    fn test_autocorrelation_matches_naive() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        for lag in [1, 2, 5, 10] {
            let simd_acf = autocorrelation(&data, lag);
            let naive_acf = naive_autocorrelation(&data, lag);
            assert!(
                (simd_acf - naive_acf).abs() < 1e-4,
                "acf mismatch at lag {}: simd={}, naive={}, diff={}",
                lag,
                simd_acf,
                naive_acf,
                (simd_acf - naive_acf).abs()
            );
        }
    }

    #[test]
    fn test_autocorrelation_large() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.05).sin()).collect();
        let simd_acf = autocorrelation(&data, 3);
        let naive_acf = naive_autocorrelation(&data, 3);
        assert!(
            (simd_acf - naive_acf).abs() < 1e-4,
            "large acf mismatch: simd={}, naive={}",
            simd_acf,
            naive_acf
        );
    }

    #[test]
    fn test_autocorrelation_edge_cases() {
        assert!(autocorrelation(&[], 1).is_nan(), "empty acf");
        assert!(autocorrelation(&[1.0], 1).is_nan(), "too short for lag");
        assert!(
            autocorrelation(&[1.0, 2.0], 5).is_nan(),
            "lag exceeds length"
        );
    }

    #[test]
    fn test_autocorrelation_non_multiple_of_four() {
        // Lengths where the effective slice (n - lag) is not a multiple of 4
        let data: Vec<f64> = (0..23).map(|i| (i as f64 * 0.3).sin()).collect();
        for lag in [1, 2, 3, 4, 5] {
            let simd_acf = autocorrelation(&data, lag);
            let naive_acf = naive_autocorrelation(&data, lag);
            assert!(
                (simd_acf - naive_acf).abs() < 1e-4,
                "acf mismatch for len=23, lag={}: simd={}, naive={}",
                lag,
                simd_acf,
                naive_acf
            );
        }
    }
}
