//! Outlier detection utilities.
//!
//! Provides methods to detect anomalous values in time series.

/// Method for outlier detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierMethod {
    /// IQR (Interquartile Range) method.
    IQR,
    /// Z-score method.
    ZScore,
    /// Modified Z-score using MAD.
    ModifiedZScore,
}

/// Result of outlier detection.
#[derive(Debug, Clone)]
pub struct OutlierResult {
    /// Indices of detected outliers.
    pub outlier_indices: Vec<usize>,
    /// Outlier scores for each point (higher = more anomalous).
    pub scores: Vec<f64>,
    /// Threshold used for detection.
    pub threshold: f64,
    /// Method used.
    pub method: OutlierMethod,
}

impl OutlierResult {
    /// Get the number of outliers detected.
    pub fn outlier_count(&self) -> usize {
        self.outlier_indices.len()
    }

    /// Check if a specific index is an outlier.
    pub fn is_outlier(&self, index: usize) -> bool {
        self.outlier_indices.contains(&index)
    }

    /// Get outlier percentage.
    pub fn outlier_percentage(&self) -> f64 {
        if self.scores.is_empty() {
            0.0
        } else {
            100.0 * self.outlier_indices.len() as f64 / self.scores.len() as f64
        }
    }
}

/// Configuration for outlier detection.
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Detection method.
    pub method: OutlierMethod,
    /// Threshold (interpretation depends on method).
    pub threshold: f64,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self {
            method: OutlierMethod::IQR,
            threshold: 1.5, // Standard IQR multiplier
        }
    }
}

impl OutlierConfig {
    /// Use IQR method with specified multiplier (default 1.5).
    pub fn iqr(multiplier: f64) -> Self {
        Self {
            method: OutlierMethod::IQR,
            threshold: multiplier,
        }
    }

    /// Use Z-score method with specified threshold (default 3.0).
    pub fn z_score(threshold: f64) -> Self {
        Self {
            method: OutlierMethod::ZScore,
            threshold,
        }
    }

    /// Use Modified Z-score method with specified threshold (default 3.5).
    pub fn modified_z_score(threshold: f64) -> Self {
        Self {
            method: OutlierMethod::ModifiedZScore,
            threshold,
        }
    }
}

/// Detect outliers in a time series.
pub fn detect_outliers(series: &[f64], config: &OutlierConfig) -> OutlierResult {
    if series.is_empty() {
        return OutlierResult {
            outlier_indices: Vec::new(),
            scores: Vec::new(),
            threshold: config.threshold,
            method: config.method,
        };
    }

    let (scores, threshold) = match config.method {
        OutlierMethod::IQR => compute_iqr_scores(series, config.threshold),
        OutlierMethod::ZScore => compute_z_scores(series, config.threshold),
        OutlierMethod::ModifiedZScore => compute_modified_z_scores(series, config.threshold),
    };

    let outlier_indices: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score > threshold)
        .map(|(i, _)| i)
        .collect();

    OutlierResult {
        outlier_indices,
        scores,
        threshold,
        method: config.method,
    }
}

/// Detect outliers with default configuration (IQR method).
pub fn detect_outliers_auto(series: &[f64]) -> OutlierResult {
    detect_outliers(series, &OutlierConfig::default())
}

/// Compute IQR-based outlier scores.
fn compute_iqr_scores(series: &[f64], multiplier: f64) -> (Vec<f64>, f64) {
    let mut sorted: Vec<f64> = series
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n < 4 {
        return (vec![0.0; series.len()], 1.0);
    }

    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    // Score: 0 if within bounds, distance from bound otherwise
    let scores: Vec<f64> = series
        .iter()
        .map(|&x| {
            if x < lower_bound {
                (lower_bound - x) / iqr.max(1e-10)
            } else if x > upper_bound {
                (x - upper_bound) / iqr.max(1e-10)
            } else {
                0.0
            }
        })
        .collect();

    (scores, 0.0) // Threshold is 0 for IQR method (anything > 0 is outlier)
}

/// Compute Z-score based outlier scores.
fn compute_z_scores(series: &[f64], threshold: f64) -> (Vec<f64>, f64) {
    let n = series.len();
    if n < 2 {
        return (vec![0.0; n], threshold);
    }

    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let variance: f64 = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return (vec![0.0; n], threshold);
    }

    let scores: Vec<f64> = series.iter().map(|x| ((x - mean) / std_dev).abs()).collect();

    (scores, threshold)
}

/// Compute Modified Z-score using MAD (Median Absolute Deviation).
fn compute_modified_z_scores(series: &[f64], threshold: f64) -> (Vec<f64>, f64) {
    let n = series.len();
    if n < 2 {
        return (vec![0.0; n], threshold);
    }

    let mut sorted: Vec<f64> = series
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // MAD: Median Absolute Deviation
    let mut abs_deviations: Vec<f64> = series.iter().map(|x| (x - median).abs()).collect();
    abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if abs_deviations.len() % 2 == 0 {
        (abs_deviations[abs_deviations.len() / 2 - 1] + abs_deviations[abs_deviations.len() / 2])
            / 2.0
    } else {
        abs_deviations[abs_deviations.len() / 2]
    };

    // Modified Z-score: 0.6745 is the 0.75th percentile of standard normal
    let k = 0.6745;
    let scaled_mad = mad / k;

    if scaled_mad < 1e-10 {
        return (vec![0.0; n], threshold);
    }

    let scores: Vec<f64> = series
        .iter()
        .map(|x| ((x - median) / scaled_mad).abs())
        .collect();

    (scores, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_outliers_iqr() {
        // Normal data with outliers
        let mut series: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect();
        series[50] = 100.0; // Outlier
        series[75] = -50.0; // Outlier

        let result = detect_outliers_auto(&series);

        assert!(result.outlier_count() >= 2);
        assert!(result.is_outlier(50));
        assert!(result.is_outlier(75));
    }

    #[test]
    fn detect_outliers_z_score() {
        let mut series: Vec<f64> = (0..100).map(|_| 10.0).collect();
        series[50] = 100.0; // Clear outlier

        let config = OutlierConfig::z_score(3.0);
        let result = detect_outliers(&series, &config);

        assert!(result.is_outlier(50));
    }

    #[test]
    fn detect_outliers_modified_z_score() {
        // Use a more varied base series
        let mut series: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect();
        series[25] = 100.0;
        series[50] = -50.0;

        let config = OutlierConfig::modified_z_score(3.0);
        let result = detect_outliers(&series, &config);

        // Should detect at least the extreme outliers
        assert!(result.outlier_count() >= 1);
        // Check that at least one of the extreme values is detected
        assert!(result.is_outlier(25) || result.is_outlier(50));
    }

    #[test]
    fn detect_outliers_no_outliers() {
        let series: Vec<f64> = (0..100).map(|i| 10.0 + 0.01 * i as f64).collect();

        let result = detect_outliers_auto(&series);

        // Should have very few or no outliers
        assert!(result.outlier_count() <= 2);
    }

    #[test]
    fn detect_outliers_empty_series() {
        let series: Vec<f64> = vec![];

        let result = detect_outliers_auto(&series);

        assert_eq!(result.outlier_count(), 0);
    }

    #[test]
    fn detect_outliers_constant_series() {
        let series = vec![5.0; 100];

        let result = detect_outliers_auto(&series);

        // No outliers in constant series
        assert_eq!(result.outlier_count(), 0);
    }

    #[test]
    fn outlier_result_methods() {
        let result = OutlierResult {
            outlier_indices: vec![10, 50, 90],
            scores: vec![0.0; 100],
            threshold: 3.0,
            method: OutlierMethod::ZScore,
        };

        assert_eq!(result.outlier_count(), 3);
        assert!(result.is_outlier(10));
        assert!(!result.is_outlier(11));
        assert!((result.outlier_percentage() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn config_methods() {
        let iqr = OutlierConfig::iqr(2.0);
        assert_eq!(iqr.method, OutlierMethod::IQR);
        assert!((iqr.threshold - 2.0).abs() < 1e-10);

        let z = OutlierConfig::z_score(2.5);
        assert_eq!(z.method, OutlierMethod::ZScore);
        assert!((z.threshold - 2.5).abs() < 1e-10);

        let mod_z = OutlierConfig::modified_z_score(3.0);
        assert_eq!(mod_z.method, OutlierMethod::ModifiedZScore);
    }

    #[test]
    fn default_config() {
        let config = OutlierConfig::default();
        assert_eq!(config.method, OutlierMethod::IQR);
        assert!((config.threshold - 1.5).abs() < 1e-10);
    }

    #[test]
    fn detect_outliers_extreme_values() {
        let mut series: Vec<f64> = vec![10.0; 100];
        series[0] = f64::MAX / 2.0;

        let result = detect_outliers_auto(&series);

        assert!(result.is_outlier(0));
    }
}
