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

    if actual.iter().any(|v| !v.is_finite()) || predicted.iter().any(|v| !v.is_finite()) {
        return Err(ForecastError::MissingValues);
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

/// Weighted Absolute Percentage Error.
///
/// WAPE = sum(|actual - forecast|) / sum(|actual|)
///
/// Returns `NaN` for empty/mismatched slices or when the sum of |actual| is zero.
pub fn wape(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    let sum_abs_actual: f64 = actual.iter().map(|a| a.abs()).sum();
    if sum_abs_actual == 0.0 {
        return f64::NAN;
    }
    let sum_abs_error: f64 = actual
        .iter()
        .zip(forecast.iter())
        .map(|(a, f)| (a - f).abs())
        .sum();
    sum_abs_error / sum_abs_actual
}

/// Mean Directional Accuracy.
///
/// Fraction of times the forecast correctly predicts the direction of change.
/// Compares sign(forecast\[i\] - actual\[i-1\]) vs sign(actual\[i\] - actual\[i-1\]).
///
/// Returns `NaN` if fewer than 2 data points or mismatched lengths.
pub fn mda(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.len() < 2 {
        return f64::NAN;
    }
    let n = actual.len() - 1;
    let correct: usize = (1..actual.len())
        .filter(|&i| {
            let actual_dir = actual[i] - actual[i - 1];
            let forecast_dir = forecast[i] - actual[i - 1];
            // Same sign (both positive, both negative, or both zero)
            (actual_dir >= 0.0 && forecast_dir >= 0.0) || (actual_dir < 0.0 && forecast_dir < 0.0)
        })
        .count();
    correct as f64 / n as f64
}

/// Theil's U1 statistic.
///
/// U1 = RMSE / (sqrt(mean(actual²)) + sqrt(mean(forecast²)))
///
/// Returns `NaN` for empty/mismatched slices or when both RMS values are zero.
pub fn theils_u1(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    let n = actual.len() as f64;
    let rms_actual = (actual.iter().map(|a| a * a).sum::<f64>() / n).sqrt();
    let rms_forecast = (forecast.iter().map(|f| f * f).sum::<f64>() / n).sqrt();
    let denom = rms_actual + rms_forecast;
    if denom == 0.0 {
        return f64::NAN;
    }
    let rmse_val = rmse(actual, forecast);
    rmse_val / denom
}

/// Theil's U2 statistic.
///
/// U2 = RMSE(forecast) / RMSE(naive), where naive\[i\] = actual\[i-1\].
///
/// Returns `NaN` if fewer than 2 data points, mismatched lengths, or naive RMSE is zero.
pub fn theils_u2(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.len() < 2 {
        return f64::NAN;
    }
    // Work on the subset [1..n] since naive needs actual[i-1]
    let actual_sub = &actual[1..];
    let forecast_sub = &forecast[1..];
    let naive: Vec<f64> = actual[..actual.len() - 1].to_vec();

    let rmse_forecast = rmse(actual_sub, forecast_sub);
    let rmse_naive = rmse(actual_sub, &naive);

    if rmse_naive == 0.0 {
        return f64::NAN;
    }
    rmse_forecast / rmse_naive
}

/// Mean Scaled Interval Score (MSIS).
///
/// Computes the mean interval score, scaled by the mean absolute first-difference
/// of `actual`. The interval score for each point penalises width plus (2/alpha)
/// times the amount by which `actual` falls outside `[lower, upper]`.
///
/// Returns `NaN` for empty/mismatched slices or when the scaling denominator is zero.
pub fn msis(actual: &[f64], lower: &[f64], upper: &[f64], alpha: f64) -> f64 {
    let n = actual.len();
    if n < 2 || lower.len() != n || upper.len() != n {
        return f64::NAN;
    }

    // Interval scores
    let inv_alpha = 2.0 / alpha;
    let total_score: f64 = (0..n)
        .map(|i| {
            let width = upper[i] - lower[i];
            let lower_pen = if actual[i] < lower[i] {
                (lower[i] - actual[i]) * inv_alpha
            } else {
                0.0
            };
            let upper_pen = if actual[i] > upper[i] {
                (actual[i] - upper[i]) * inv_alpha
            } else {
                0.0
            };
            width + lower_pen + upper_pen
        })
        .sum();

    // Scaling factor: mean |diff(actual)|
    let mean_abs_diff: f64 =
        actual.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (n - 1) as f64;

    if mean_abs_diff == 0.0 {
        return f64::NAN;
    }

    (total_score / n as f64) / mean_abs_diff
}

/// Prediction interval coverage.
///
/// Fraction of actual values that fall within `[lower, upper]`.
///
/// Returns `NaN` for empty or mismatched slices.
pub fn coverage(actual: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let n = actual.len();
    if n == 0 || lower.len() != n || upper.len() != n {
        return f64::NAN;
    }
    let within: usize = (0..n)
        .filter(|&i| actual[i] >= lower[i] && actual[i] <= upper[i])
        .count();
    within as f64 / n as f64
}

/// Skill score.
///
/// `1 - metric_model / metric_baseline`.
///
/// A positive score means the model is better than the baseline.
/// Returns `NaN` if `metric_baseline` is zero.
pub fn skill_score(metric_model: f64, metric_baseline: f64) -> f64 {
    if metric_baseline == 0.0 {
        return f64::NAN;
    }
    1.0 - metric_model / metric_baseline
}

/// Forecast bias (mean error).
///
/// `bias = mean(forecast - actual)`
///
/// Positive bias indicates systematic over-forecasting, negative indicates
/// under-forecasting. Unlike `IntermittentDiagnostics::bias` which only
/// considers non-zero actual periods, this computes bias over all observations.
///
/// Returns `NaN` for empty or mismatched slices.
pub fn bias(actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() {
        return f64::NAN;
    }
    let n = actual.len() as f64;
    actual
        .iter()
        .zip(forecast.iter())
        .map(|(a, f)| f - a)
        .sum::<f64>()
        / n
}

/// Periods-In-Stock (PIS).
///
/// Cumulative sum of `(forecast[i] - actual[i])` at each time step.
/// Positive values indicate cumulative over-forecasting (overstock),
/// negative values indicate cumulative under-forecasting (stockout risk).
///
/// Returns an empty vector for empty or mismatched slices.
pub fn periods_in_stock(actual: &[f64], forecast: &[f64]) -> Vec<f64> {
    if actual.len() != forecast.len() || actual.is_empty() {
        return Vec::new();
    }
    let mut cum = 0.0;
    actual
        .iter()
        .zip(forecast.iter())
        .map(|(a, f)| {
            cum += f - a;
            cum
        })
        .collect()
}

/// Comprehensive forecast metrics combining point-forecast accuracy measures.
#[derive(Debug, Clone)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error (NaN if zeros in actual)
    pub mape: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
    /// Mean Absolute Scaled Error (NaN if insufficient data)
    pub mase: f64,
    /// Weighted Absolute Percentage Error
    pub wape: f64,
    /// Mean Directional Accuracy
    pub mda: f64,
    /// Theil's U1 statistic
    pub theils_u1: f64,
    /// Theil's U2 statistic
    pub theils_u2: f64,
}

impl ForecastMetrics {
    /// Compute all forecast metrics in one call.
    ///
    /// # Arguments
    /// * `actual` - Actual observed values
    /// * `forecast` - Predicted/forecast values
    /// * `seasonal_period` - Seasonal period for MASE calculation (use 1 for non-seasonal)
    pub fn compute(actual: &[f64], forecast: &[f64], seasonal_period: usize) -> Self {
        let mae_val = mae(actual, forecast);
        let mse_val = mse(actual, forecast);
        let rmse_val = mse_val.sqrt();
        let smape_val = smape(actual, forecast);

        // MAPE
        let mape_val = if actual.is_empty()
            || forecast.is_empty()
            || actual.len() != forecast.len()
            || actual.contains(&0.0)
        {
            f64::NAN
        } else {
            let n = actual.len() as f64;
            actual
                .iter()
                .zip(forecast.iter())
                .map(|(a, f)| ((a - f) / a).abs())
                .sum::<f64>()
                * 100.0
                / n
        };

        // MASE via the existing private helper
        let mase_val = calculate_mase(actual, forecast, Some(seasonal_period)).unwrap_or(f64::NAN);

        let wape_val = wape(actual, forecast);
        let mda_val = mda(actual, forecast);
        let u1 = theils_u1(actual, forecast);
        let u2 = theils_u2(actual, forecast);

        Self {
            mae: mae_val,
            mse: mse_val,
            rmse: rmse_val,
            mape: mape_val,
            smape: smape_val,
            mase: mase_val,
            wape: wape_val,
            mda: mda_val,
            theils_u1: u1,
            theils_u2: u2,
        }
    }
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

    #[test]
    fn calculate_metrics_nan_in_actual() {
        let actual = vec![1.0, f64::NAN, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];

        let result = calculate_metrics(&actual, &predicted, None);
        assert!(matches!(result, Err(ForecastError::MissingValues)));
    }

    #[test]
    fn calculate_metrics_nan_in_predicted() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, f64::NAN, 3.0];

        let result = calculate_metrics(&actual, &predicted, None);
        assert!(matches!(result, Err(ForecastError::MissingValues)));
    }

    #[test]
    fn calculate_metrics_inf_in_actual() {
        let actual = vec![1.0, f64::INFINITY, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];

        let result = calculate_metrics(&actual, &predicted, None);
        assert!(matches!(result, Err(ForecastError::MissingValues)));
    }

    // --- WAPE tests ---

    #[test]
    fn wape_known_values() {
        let actual = vec![10.0, 20.0, 30.0];
        let forecast = vec![12.0, 18.0, 33.0];
        // |errors| = 2 + 2 + 3 = 7, sum|actual| = 60
        assert_relative_eq!(wape(&actual, &forecast), 7.0 / 60.0, epsilon = 1e-10);
    }

    #[test]
    fn wape_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(wape(&actual, &actual), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn wape_empty() {
        assert!(wape(&[], &[]).is_nan());
    }

    #[test]
    fn wape_mismatched() {
        assert!(wape(&[1.0], &[1.0, 2.0]).is_nan());
    }

    #[test]
    fn wape_zero_actual() {
        assert!(wape(&[0.0, 0.0], &[1.0, 2.0]).is_nan());
    }

    // --- MDA tests ---

    #[test]
    fn mda_perfect_direction() {
        // actual goes up each step; forecast also goes up from prior actual
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let forecast = vec![0.0, 1.5, 2.5, 3.5]; // direction matches all 3 transitions
        assert_relative_eq!(mda(&actual, &forecast), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mda_wrong_direction() {
        let actual = vec![1.0, 2.0, 3.0, 4.0]; // always up
        let forecast = vec![0.0, 0.5, 1.0, 1.5]; // forecast[i] < actual[i-1] => predicts down
        assert_relative_eq!(mda(&actual, &forecast), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn mda_insufficient_points() {
        assert!(mda(&[1.0], &[1.0]).is_nan());
        assert!(mda(&[], &[]).is_nan());
    }

    // --- Theil's U1 tests ---

    #[test]
    fn theils_u1_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(theils_u1(&actual, &actual), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn theils_u1_known() {
        let actual = vec![1.0, 2.0, 3.0];
        let forecast = vec![1.5, 2.5, 3.5];
        let result = theils_u1(&actual, &forecast);
        assert!(result.is_finite());
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn theils_u1_all_zeros() {
        assert!(theils_u1(&[0.0, 0.0], &[0.0, 0.0]).is_nan());
    }

    #[test]
    fn theils_u1_empty() {
        assert!(theils_u1(&[], &[]).is_nan());
    }

    // --- Theil's U2 tests ---

    #[test]
    fn theils_u2_better_than_naive() {
        // Trending series; a close forecast should beat naive
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let forecast = vec![1.0, 2.1, 3.1, 4.1, 5.1];
        let u2 = theils_u2(&actual, &forecast);
        assert!(u2.is_finite());
        assert!(u2 < 1.0, "Good forecast should have U2 < 1, got {u2}");
    }

    #[test]
    fn theils_u2_naive_perfect() {
        // constant series => naive is perfect => denominator = 0 => NaN
        let actual = vec![5.0, 5.0, 5.0, 5.0];
        let forecast = vec![5.0, 5.0, 5.0, 5.0];
        assert!(theils_u2(&actual, &forecast).is_nan());
    }

    #[test]
    fn theils_u2_insufficient() {
        assert!(theils_u2(&[1.0], &[1.0]).is_nan());
    }

    // --- MSIS tests ---

    #[test]
    fn msis_all_covered() {
        let actual = vec![2.0, 3.0, 4.0, 5.0];
        let lower = vec![1.0, 2.0, 3.0, 4.0];
        let upper = vec![3.0, 4.0, 5.0, 6.0];
        let result = msis(&actual, &lower, &upper, 0.05);
        assert!(result.is_finite());
        assert!(result > 0.0); // interval has width so score > 0
    }

    #[test]
    fn msis_with_violations() {
        let actual = vec![0.0, 10.0, 5.0, 8.0];
        let lower = vec![1.0, 2.0, 3.0, 4.0];
        let upper = vec![3.0, 4.0, 6.0, 7.0];
        // actual[0]=0 < lower, actual[1]=10 > upper, actual[3]=8 > upper
        let result = msis(&actual, &lower, &upper, 0.05);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn msis_too_few_points() {
        assert!(msis(&[1.0], &[0.0], &[2.0], 0.05).is_nan());
    }

    #[test]
    fn msis_mismatched() {
        assert!(msis(&[1.0, 2.0], &[0.0], &[2.0, 3.0], 0.05).is_nan());
    }

    // --- Coverage tests ---

    #[test]
    fn coverage_all_within() {
        let actual = vec![2.0, 3.0, 4.0];
        let lower = vec![1.0, 2.0, 3.0];
        let upper = vec![3.0, 4.0, 5.0];
        assert_relative_eq!(coverage(&actual, &lower, &upper), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn coverage_none_within() {
        let actual = vec![0.0, 10.0, 20.0];
        let lower = vec![1.0, 2.0, 3.0];
        let upper = vec![0.5, 5.0, 10.0];
        assert_relative_eq!(coverage(&actual, &lower, &upper), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn coverage_partial() {
        let actual = vec![2.0, 10.0, 4.0];
        let lower = vec![1.0, 2.0, 3.0];
        let upper = vec![3.0, 5.0, 5.0];
        // 2 within [1,3], 10 NOT in [2,5], 4 within [3,5]
        assert_relative_eq!(
            coverage(&actual, &lower, &upper),
            2.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn coverage_empty() {
        assert!(coverage(&[], &[], &[]).is_nan());
    }

    // --- Skill score tests ---

    #[test]
    fn skill_score_better_model() {
        // model MAE = 1, baseline MAE = 4 => skill = 0.75
        assert_relative_eq!(skill_score(1.0, 4.0), 0.75, epsilon = 1e-10);
    }

    #[test]
    fn skill_score_equal() {
        assert_relative_eq!(skill_score(2.0, 2.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn skill_score_worse_model() {
        // model worse than baseline => negative
        assert!(skill_score(5.0, 2.0) < 0.0);
    }

    #[test]
    fn skill_score_zero_baseline() {
        assert!(skill_score(1.0, 0.0).is_nan());
    }

    // --- ForecastMetrics tests ---

    #[test]
    fn forecast_metrics_compute_basic() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let forecast = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let fm = ForecastMetrics::compute(&actual, &forecast, 1);

        assert_relative_eq!(fm.mae, 0.1, epsilon = 1e-10);
        assert!(fm.mse.is_finite());
        assert!(fm.rmse.is_finite());
        assert!(fm.mape.is_finite());
        assert!(fm.smape.is_finite());
        assert!(fm.mase.is_finite());
        assert!(fm.wape.is_finite());
        assert!(fm.mda.is_finite());
        assert!(fm.theils_u1.is_finite());
        assert!(fm.theils_u2.is_finite());
    }

    #[test]
    fn forecast_metrics_compute_perfect() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fm = ForecastMetrics::compute(&actual, &actual, 1);

        assert_relative_eq!(fm.mae, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fm.mse, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fm.rmse, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fm.smape, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fm.wape, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fm.mda, 1.0, epsilon = 1e-10);
        assert_relative_eq!(fm.theils_u1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn forecast_metrics_with_zeros_in_actual() {
        let actual = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let forecast = vec![0.1, 1.1, 2.1, 3.1, 4.1];
        let fm = ForecastMetrics::compute(&actual, &forecast, 1);

        assert!(fm.mape.is_nan()); // zero in actual
        assert!(fm.mae.is_finite());
        assert!(fm.smape.is_finite());
        assert!(fm.wape.is_finite());
    }

    // --- Bias tests ---

    #[test]
    fn bias_perfect() {
        let actual = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(bias(&actual, &actual), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn bias_overforecast() {
        let actual = vec![1.0, 2.0, 3.0];
        let forecast = vec![2.0, 3.0, 4.0]; // always +1
        assert_relative_eq!(bias(&actual, &forecast), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn bias_underforecast() {
        let actual = vec![5.0, 5.0, 5.0];
        let forecast = vec![3.0, 3.0, 3.0]; // always -2
        assert_relative_eq!(bias(&actual, &forecast), -2.0, epsilon = 1e-10);
    }

    #[test]
    fn bias_cancels_out() {
        let actual = vec![10.0, 10.0];
        let forecast = vec![12.0, 8.0]; // +2 then -2
        assert_relative_eq!(bias(&actual, &forecast), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn bias_empty() {
        assert!(bias(&[], &[]).is_nan());
    }

    #[test]
    fn bias_mismatched() {
        assert!(bias(&[1.0], &[1.0, 2.0]).is_nan());
    }

    // --- PIS tests ---

    #[test]
    fn pis_perfect() {
        let actual = vec![5.0, 3.0, 4.0];
        let pis = periods_in_stock(&actual, &actual);
        assert_eq!(pis.len(), 3);
        for &v in &pis {
            assert_relative_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn pis_constant_overforecast() {
        let actual = vec![3.0, 3.0, 3.0, 3.0];
        let forecast = vec![5.0, 5.0, 5.0, 5.0]; // +2 each step
        let pis = periods_in_stock(&actual, &forecast);
        assert_eq!(pis.len(), 4);
        assert_relative_eq!(pis[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(pis[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(pis[2], 6.0, epsilon = 1e-10);
        assert_relative_eq!(pis[3], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn pis_underforecast() {
        let actual = vec![5.0, 5.0, 5.0];
        let forecast = vec![4.0, 4.0, 4.0]; // -1 each step
        let pis = periods_in_stock(&actual, &forecast);
        assert_relative_eq!(pis[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(pis[1], -2.0, epsilon = 1e-10);
        assert_relative_eq!(pis[2], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn pis_mixed() {
        let actual = vec![10.0, 5.0, 8.0];
        let forecast = vec![8.0, 7.0, 6.0];
        // step 0: 8-10 = -2
        // step 1: -2 + (7-5) = 0
        // step 2: 0 + (6-8) = -2
        let pis = periods_in_stock(&actual, &forecast);
        assert_relative_eq!(pis[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(pis[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(pis[2], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn pis_empty() {
        assert!(periods_in_stock(&[], &[]).is_empty());
    }

    #[test]
    fn pis_mismatched() {
        assert!(periods_in_stock(&[1.0], &[1.0, 2.0]).is_empty());
    }
}
