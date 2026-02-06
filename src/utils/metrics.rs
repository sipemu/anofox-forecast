//! Accuracy metrics for forecast evaluation.

use crate::error::{ForecastError, Result};

/// Accuracy metrics for evaluating forecast performance.
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error (None if zeros in actual)
    pub mape: Option<f64>,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
    /// Mean Absolute Scaled Error (None if insufficient data)
    pub mase: Option<f64>,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
}

impl AccuracyMetrics {
    /// Create metrics with all zeros (for empty predictions).
    pub fn zero() -> Self {
        Self {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: Some(0.0),
            smape: 0.0,
            mase: Some(0.0),
            r_squared: 1.0,
        }
    }
}

/// Calculate accuracy metrics between actual and predicted values.
///
/// # Arguments
/// * `actual` - Actual observed values
/// * `predicted` - Predicted/forecast values
/// * `seasonal_period` - Optional seasonal period for MASE calculation
///
/// # Returns
/// `AccuracyMetrics` struct with all computed metrics
pub fn calculate_metrics(
    actual: &[f64],
    predicted: &[f64],
    seasonal_period: Option<usize>,
) -> Result<AccuracyMetrics> {
    if actual.is_empty() || predicted.is_empty() {
        return Err(ForecastError::EmptyData);
    }

    if actual.len() != predicted.len() {
        return Err(ForecastError::DimensionMismatch {
            expected: actual.len(),
            got: predicted.len(),
        });
    }

    let n = actual.len() as f64;

    // Pass 1: compute sum_ae, sum_se, sum_ape, sum_smape, sum_actual, has_zero
    let mut sum_ae = 0.0;
    let mut sum_se = 0.0;
    let mut sum_ape = 0.0;
    let mut sum_smape = 0.0;
    let mut sum_actual = 0.0;
    let mut has_zero = false;

    for (&a, &p) in actual.iter().zip(predicted.iter()) {
        let diff = a - p;
        let abs_diff = diff.abs();
        sum_ae += abs_diff;
        sum_se += diff * diff;
        sum_actual += a;

        if a == 0.0 {
            has_zero = true;
        } else {
            sum_ape += abs_diff / a.abs();
        }

        let denom = a.abs() + p.abs();
        if denom != 0.0 {
            sum_smape += 2.0 * abs_diff / denom;
        }
    }

    let mae = sum_ae / n;
    let mse = sum_se / n;
    let rmse = mse.sqrt();
    let mape = if has_zero {
        None
    } else {
        Some(100.0 * sum_ape / n)
    };
    let smape = 100.0 * sum_smape / n;

    // MASE
    let mase = calculate_mase(actual, predicted, seasonal_period);

    // Pass 2: R-squared (needs mean from pass 1)
    let mean_actual = sum_actual / n;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean_actual).powi(2)).sum();
    // ss_res == sum_se, reuse from pass 1
    let r_squared = if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - sum_se / ss_tot
    };

    Ok(AccuracyMetrics {
        mae,
        mse,
        rmse,
        mape,
        smape,
        mase,
        r_squared,
    })
}

/// Calculate Mean Absolute Scaled Error.
///
/// MASE = MAE / MAE_naive
/// where MAE_naive is the MAE of the naive (or seasonal naive) forecast.
fn calculate_mase(
    actual: &[f64],
    predicted: &[f64],
    seasonal_period: Option<usize>,
) -> Option<f64> {
    let n = actual.len();
    let period = seasonal_period.unwrap_or(1);

    if n <= period {
        return None;
    }

    // Calculate MAE of naive forecast
    let naive_mae: f64 = actual
        .iter()
        .skip(period)
        .zip(actual.iter())
        .map(|(curr, prev)| (curr - prev).abs())
        .sum::<f64>()
        / (n - period) as f64;

    if naive_mae == 0.0 {
        return None;
    }

    // Calculate forecast MAE
    let forecast_mae: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / n as f64;

    Some(forecast_mae / naive_mae)
}

/// Calculate MAE between two slices.
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }
    crate::simd::l1_distance(actual, predicted) / actual.len() as f64
}

/// Calculate MSE between two slices.
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }
    crate::simd::squared_distance(actual, predicted) / actual.len() as f64
}

/// Calculate RMSE between two slices.
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    mse(actual, predicted).sqrt()
}

/// Calculate SMAPE between two slices.
pub fn smape(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }
    let n = actual.len() as f64;
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| {
            let denom = a.abs() + p.abs();
            if denom == 0.0 {
                0.0
            } else {
                2.0 * (a - p).abs() / denom
            }
        })
        .sum::<f64>()
        * 100.0
        / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn calculate_metrics_perfect_prediction() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let metrics = calculate_metrics(&actual, &predicted, None).unwrap();

        assert_relative_eq!(metrics.mae, 0.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.mse, 0.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.rmse, 0.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.smape, 0.0, epsilon = 1e-10);
        assert_relative_eq!(metrics.r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn calculate_metrics_known_values() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![1.5, 2.5, 2.5, 4.5, 4.5];
        // Errors: 0.5, 0.5, 0.5, 0.5, 0.5

        let metrics = calculate_metrics(&actual, &predicted, None).unwrap();

        assert_relative_eq!(metrics.mae, 0.5, epsilon = 1e-10);
        assert_relative_eq!(metrics.mse, 0.25, epsilon = 1e-10);
        assert_relative_eq!(metrics.rmse, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn calculate_metrics_mape_with_zeros() {
        let actual = vec![0.0, 1.0, 2.0];
        let predicted = vec![0.1, 1.1, 2.1];

        let metrics = calculate_metrics(&actual, &predicted, None).unwrap();

        assert!(metrics.mape.is_none()); // Can't compute MAPE with zeros
        assert!(metrics.smape.is_finite()); // SMAPE should still work
    }

    #[test]
    fn calculate_metrics_dimension_mismatch() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0];

        let result = calculate_metrics(&actual, &predicted, None);
        assert!(matches!(
            result,
            Err(ForecastError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn calculate_metrics_empty_data() {
        let result = calculate_metrics(&[], &[], None);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn mase_with_seasonal_period() {
        // Non-perfect seasonal pattern so naive MAE is non-zero
        let actual = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];
        let predicted = vec![1.1, 2.1, 3.1, 4.1, 1.6, 2.6, 3.6, 4.6];

        let metrics = calculate_metrics(&actual, &predicted, Some(4)).unwrap();

        assert!(metrics.mase.is_some());
        // MASE should be finite and positive
        let mase = metrics.mase.unwrap();
        assert!(mase.is_finite() && mase > 0.0);
    }

    #[test]
    fn standalone_mae() {
        assert_relative_eq!(
            mae(&[1.0, 2.0, 3.0], &[1.5, 2.5, 3.5]),
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn standalone_rmse() {
        assert_relative_eq!(
            rmse(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn standalone_smape() {
        // For equal values, SMAPE should be 0
        assert_relative_eq!(
            smape(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn r_squared_negative_for_poor_model() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Inverted

        let metrics = calculate_metrics(&actual, &predicted, None).unwrap();

        assert!(metrics.r_squared < 0.0); // Worse than mean prediction
    }
}
