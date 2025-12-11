//! Scaling and normalization transforms for time series.
//!
//! Provides methods to standardize, normalize, and scale data.

/// Result of a scaling transform, containing parameters for inverse transform.
#[derive(Debug, Clone)]
pub struct ScaleResult {
    /// Transformed data
    pub data: Vec<f64>,
    /// Center value used (mean or median)
    pub center: f64,
    /// Scale value used (std dev or IQR)
    pub scale: f64,
}

impl ScaleResult {
    /// Inverse transform to recover original scale.
    pub fn inverse(&self) -> Vec<f64> {
        self.data
            .iter()
            .map(|&x| x * self.scale + self.center)
            .collect()
    }

    /// Transform new data using the same parameters.
    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        if self.scale.abs() < 1e-10 {
            return vec![0.0; data.len()];
        }
        data.iter()
            .map(|&x| (x - self.center) / self.scale)
            .collect()
    }
}

/// Standardize data to zero mean and unit variance (z-score normalization).
///
/// x_scaled = (x - mean) / std
pub fn standardize(series: &[f64]) -> ScaleResult {
    if series.is_empty() {
        return ScaleResult {
            data: Vec::new(),
            center: 0.0,
            scale: 1.0,
        };
    }

    let n = series.len() as f64;
    let mean = series.iter().sum::<f64>() / n;

    let variance = if series.len() > 1 {
        series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let std = variance.sqrt();

    let scale = if std < 1e-10 { 1.0 } else { std };
    let data = series.iter().map(|&x| (x - mean) / scale).collect();

    ScaleResult {
        data,
        center: mean,
        scale,
    }
}

/// Normalize data to [0, 1] range (min-max normalization).
///
/// x_scaled = (x - min) / (max - min)
pub fn normalize(series: &[f64]) -> ScaleResult {
    if series.is_empty() {
        return ScaleResult {
            data: Vec::new(),
            center: 0.0,
            scale: 1.0,
        };
    }

    let min = series.iter().copied().fold(f64::INFINITY, f64::min);
    let max = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    let scale = if range < 1e-10 { 1.0 } else { range };
    let data = series.iter().map(|&x| (x - min) / scale).collect();

    ScaleResult {
        data,
        center: min,
        scale,
    }
}

/// Robust scaling using median and IQR.
///
/// x_scaled = (x - median) / IQR
///
/// More robust to outliers than standardization.
pub fn robust_scale(series: &[f64]) -> ScaleResult {
    if series.is_empty() {
        return ScaleResult {
            data: Vec::new(),
            center: 0.0,
            scale: 1.0,
        };
    }

    let median = compute_median(series);
    let q1 = compute_quantile(series, 0.25);
    let q3 = compute_quantile(series, 0.75);
    let iqr = q3 - q1;

    let scale = if iqr < 1e-10 { 1.0 } else { iqr };
    let data = series.iter().map(|&x| (x - median) / scale).collect();

    ScaleResult {
        data,
        center: median,
        scale,
    }
}

/// Scale data to a specific range [min_val, max_val].
pub fn scale_to_range(series: &[f64], min_val: f64, max_val: f64) -> Vec<f64> {
    if series.is_empty() || min_val >= max_val {
        return series.to_vec();
    }

    let normalized = normalize(series);
    let target_range = max_val - min_val;

    normalized
        .data
        .iter()
        .map(|&x| x * target_range + min_val)
        .collect()
}

/// Center data by subtracting the mean.
pub fn center(series: &[f64]) -> ScaleResult {
    if series.is_empty() {
        return ScaleResult {
            data: Vec::new(),
            center: 0.0,
            scale: 1.0,
        };
    }

    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let data = series.iter().map(|&x| x - mean).collect();

    ScaleResult {
        data,
        center: mean,
        scale: 1.0,
    }
}

/// Helper: compute median
fn compute_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Helper: compute quantile
fn compute_quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let frac = pos - lower as f64;

    if lower == upper || upper >= n {
        sorted[lower.min(n - 1)]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ==================== standardize ====================

    #[test]
    fn standardize_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = standardize(&series);

        // Mean should be 3, std should be sqrt(2.5)
        assert_relative_eq!(result.center, 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.scale, 2.5_f64.sqrt(), epsilon = 1e-10);

        // Standardized data should have mean ≈ 0
        let mean: f64 = result.data.iter().sum::<f64>() / result.data.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn standardize_constant() {
        let series = vec![5.0; 10];
        let result = standardize(&series);

        assert_relative_eq!(result.center, 5.0, epsilon = 1e-10);
        // Scale should be 1.0 (default) since std = 0
        assert_relative_eq!(result.scale, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn standardize_empty() {
        let result = standardize(&[]);
        assert!(result.data.is_empty());
    }

    #[test]
    fn standardize_inverse() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = standardize(&series);
        let recovered = result.inverse();

        for (orig, rec) in series.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    // ==================== normalize ====================

    #[test]
    fn normalize_basic() {
        let series = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let result = normalize(&series);

        // Should be in [0, 1]
        assert_relative_eq!(result.data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[4], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[2], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn normalize_negative_values() {
        let series = vec![-10.0, 0.0, 10.0];
        let result = normalize(&series);

        assert_relative_eq!(result.data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[2], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn normalize_constant() {
        let series = vec![5.0; 10];
        let result = normalize(&series);

        // All values should be 0 (x - min = 0, and we use scale=1)
        for &x in &result.data {
            assert_relative_eq!(x, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn normalize_empty() {
        let result = normalize(&[]);
        assert!(result.data.is_empty());
    }

    #[test]
    fn normalize_inverse() {
        let series = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = normalize(&series);
        let recovered = result.inverse();

        for (orig, rec) in series.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    // ==================== robust_scale ====================

    #[test]
    fn robust_scale_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = robust_scale(&series);

        // Median should be 5
        assert_relative_eq!(result.center, 5.0, epsilon = 1e-10);

        // Median value should map to 0
        assert_relative_eq!(result.data[4], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn robust_scale_with_outliers() {
        // Outliers should have less effect on robust scaling
        let mut series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        series.push(1000.0); // Extreme outlier

        let standard = standardize(&series);
        let robust = robust_scale(&series);

        // Robust scale should be much smaller than standardization scale
        assert!(robust.scale < standard.scale);
    }

    #[test]
    fn robust_scale_constant() {
        let series = vec![5.0; 10];
        let result = robust_scale(&series);

        assert_relative_eq!(result.center, 5.0, epsilon = 1e-10);
        assert_relative_eq!(result.scale, 1.0, epsilon = 1e-10); // Default when IQR = 0
    }

    #[test]
    fn robust_scale_empty() {
        let result = robust_scale(&[]);
        assert!(result.data.is_empty());
    }

    // ==================== scale_to_range ====================

    #[test]
    fn scale_to_range_basic() {
        let series = vec![0.0, 50.0, 100.0];
        let result = scale_to_range(&series, -1.0, 1.0);

        assert_relative_eq!(result[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn scale_to_range_custom() {
        let series = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let result = scale_to_range(&series, 10.0, 20.0);

        assert_relative_eq!(result[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn scale_to_range_empty() {
        let result = scale_to_range(&[], 0.0, 1.0);
        assert!(result.is_empty());
    }

    // ==================== center ====================

    #[test]
    fn center_basic() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = center(&series);

        assert_relative_eq!(result.center, 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.scale, 1.0, epsilon = 1e-10);

        // Sum should be 0
        let sum: f64 = result.data.iter().sum();
        assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn center_empty() {
        let result = center(&[]);
        assert!(result.data.is_empty());
    }

    // ==================== transform new data ====================

    #[test]
    fn transform_new_data() {
        let series = vec![0.0, 50.0, 100.0];
        let result = standardize(&series);

        let new_data = vec![25.0, 75.0];
        let transformed = result.transform(&new_data);

        // Should use same center and scale
        for (i, &x) in new_data.iter().enumerate() {
            let expected = (x - result.center) / result.scale;
            assert_relative_eq!(transformed[i], expected, epsilon = 1e-10);
        }
    }
}
