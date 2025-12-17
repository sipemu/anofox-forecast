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
/// Returns the sum of element-wise products: Σ(a[i] * b[i]).
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

/// Squared Euclidean distance: Σ(a[i] - b[i])².
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

/// Manhattan (L1) distance: Σ|a[i] - b[i]|.
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

/// Element-wise subtraction: a[i] - b[i].
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

/// Element-wise multiplication: a[i] * b[i].
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

/// Element-wise division: a[i] / b[i].
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

/// Element-wise addition: a[i] + b[i].
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

/// Scale all elements by a constant: data[i] * scalar.
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
}
