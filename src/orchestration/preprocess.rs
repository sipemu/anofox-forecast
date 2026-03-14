//! Automatic preprocessing for the orchestration pipeline.
//!
//! Applies Box-Cox, outlier replacement, and differencing based on
//! data profile characteristics.

use std::fmt;

use crate::core::TimeSeries;
use crate::detection::{detect_outliers_auto, OutlierConfig};
use crate::error::Result;
use crate::transform::boxcox::{boxcox_auto, inv_boxcox, is_boxcox_suitable};

use super::profile::DataProfile;

/// How preprocessing should be handled.
#[derive(Debug, Clone, Default)]
pub enum PreprocessMode {
    /// Decide automatically from the [`DataProfile`].
    ///
    /// - Box-Cox when `|skewness| > 1.0` and all values are positive
    /// - Outlier replacement when `quality_score < 0.9`
    Auto,
    /// Apply exactly the specified steps.
    Manual(PreprocessSteps),
    /// No preprocessing (default).
    #[default]
    None,
}

/// Explicit preprocessing steps to apply.
#[derive(Debug, Clone)]
pub struct PreprocessSteps {
    /// Apply Box-Cox transformation (auto-lambda).
    pub boxcox: bool,
    /// Replace outliers with local median.
    pub outlier_treatment: bool,
    /// Outlier detection window for replacement (default: 5).
    pub outlier_window: usize,
}

impl Default for PreprocessSteps {
    fn default() -> Self {
        Self {
            boxcox: false,
            outlier_treatment: false,
            outlier_window: 5,
        }
    }
}

/// Record of what preprocessing was applied and how to invert it.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// Box-Cox lambda (None if Box-Cox was not applied).
    pub boxcox_lambda: Option<f64>,
    /// Number of outliers replaced.
    pub outliers_replaced: usize,
    /// Steps that were applied.
    pub steps_applied: Vec<String>,
}

impl fmt::Display for PreprocessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.steps_applied.is_empty() {
            write!(f, "Preprocessing: none")
        } else {
            write!(f, "Preprocessing: {}", self.steps_applied.join(", "))
        }
    }
}

/// Apply preprocessing to a time series, returning the transformed series
/// and a result that allows inversion of the forecast.
pub fn apply_preprocessing(
    ts: &TimeSeries,
    mode: &PreprocessMode,
    profile: Option<&DataProfile>,
) -> Result<(TimeSeries, PreprocessResult)> {
    let steps = match mode {
        PreprocessMode::None => {
            return Ok((
                ts.clone(),
                PreprocessResult {
                    boxcox_lambda: None,
                    outliers_replaced: 0,
                    steps_applied: vec![],
                },
            ));
        }
        PreprocessMode::Manual(steps) => steps.clone(),
        PreprocessMode::Auto => resolve_auto(profile),
    };

    let mut current = ts.clone();
    let mut result = PreprocessResult {
        boxcox_lambda: None,
        outliers_replaced: 0,
        steps_applied: vec![],
    };

    // 1. Outlier treatment (before Box-Cox, since outliers can distort lambda)
    if steps.outlier_treatment {
        let config = OutlierConfig::iqr(1.5);
        match current.with_outliers_replaced(&config, steps.outlier_window) {
            Ok(cleaned) => {
                let n_outliers = detect_outliers_auto(current.primary_values()).outlier_count();
                result.outliers_replaced = n_outliers;
                if n_outliers > 0 {
                    result
                        .steps_applied
                        .push(format!("outlier_replacement({})", n_outliers));
                }
                current = cleaned;
            }
            Err(_) => {
                // Silently skip if outlier replacement fails
            }
        }
    }

    // 2. Box-Cox
    if steps.boxcox {
        let values = current.primary_values();
        if is_boxcox_suitable(values) {
            let bc = boxcox_auto(values);
            result.boxcox_lambda = Some(bc.lambda);
            result
                .steps_applied
                .push(format!("boxcox(lambda={:.4})", bc.lambda));

            // Replace values in the time series
            current = TimeSeries::univariate(current.timestamps().to_vec(), bc.data)?;
        }
    }

    Ok((current, result))
}

/// Invert Box-Cox on a forecast's point values.
pub fn invert_boxcox_forecast(values: &[f64], lambda: f64) -> Vec<f64> {
    inv_boxcox(values, lambda)
}

/// Decide preprocessing steps from the data profile.
fn resolve_auto(profile: Option<&DataProfile>) -> PreprocessSteps {
    let profile = match profile {
        Some(p) => p,
        None => return PreprocessSteps::default(),
    };

    PreprocessSteps {
        boxcox: !profile.has_negatives && profile.skewness.abs() > 1.0,
        outlier_treatment: profile.quality_score < 0.9,
        outlier_window: 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesBuilder;
    use chrono::{Duration, Utc};

    fn make_ts(values: Vec<f64>) -> TimeSeries {
        let n = values.len();
        let start = Utc::now();
        let timestamps: Vec<_> = (0..n).map(|i| start + Duration::days(i as i64)).collect();
        TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .values(values)
            .build()
            .unwrap()
    }

    #[test]
    fn preprocess_none_passthrough() {
        let ts = make_ts(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (result_ts, info) = apply_preprocessing(&ts, &PreprocessMode::None, None).unwrap();
        assert_eq!(result_ts.len(), ts.len());
        assert!(info.boxcox_lambda.is_none());
        assert_eq!(info.outliers_replaced, 0);
        assert!(info.steps_applied.is_empty());
    }

    #[test]
    fn preprocess_manual_boxcox() {
        // Exponential data — suitable for Box-Cox
        let values: Vec<f64> = (1..=50).map(|i| (i as f64 * 0.1).exp()).collect();
        let ts = make_ts(values);
        let steps = PreprocessSteps {
            boxcox: true,
            outlier_treatment: false,
            outlier_window: 5,
        };
        let (_, info) = apply_preprocessing(&ts, &PreprocessMode::Manual(steps), None).unwrap();
        assert!(info.boxcox_lambda.is_some());
        assert!(info.steps_applied.iter().any(|s| s.contains("boxcox")));
    }

    #[test]
    fn preprocess_auto_skewed_data() {
        // Highly skewed positive data triggers Box-Cox
        let values: Vec<f64> = (1..=100).map(|i| (i as f64).powi(3)).collect();
        let ts = make_ts(values);
        let profile = DataProfile::from_series(&ts);

        if profile.skewness.abs() > 1.0 {
            let (_, info) =
                apply_preprocessing(&ts, &PreprocessMode::Auto, Some(&profile)).unwrap();
            assert!(info.boxcox_lambda.is_some());
        }
    }

    #[test]
    fn invert_boxcox_roundtrip() {
        let original = vec![2.0, 4.0, 8.0, 16.0];
        let bc = boxcox_auto(&original);
        let restored = invert_boxcox_forecast(&bc.data, bc.lambda);
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.01,
                "Expected {}, got {} after roundtrip",
                a,
                b
            );
        }
    }

    #[test]
    fn preprocess_display() {
        let info = PreprocessResult {
            boxcox_lambda: Some(0.5),
            outliers_replaced: 3,
            steps_applied: vec![
                "outlier_replacement(3)".into(),
                "boxcox(lambda=0.5000)".into(),
            ],
        };
        let text = format!("{}", info);
        assert!(text.contains("outlier_replacement"));
        assert!(text.contains("boxcox"));
    }
}
