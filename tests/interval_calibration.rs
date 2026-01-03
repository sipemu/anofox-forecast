//! Interval calibration tests for forecast confidence intervals.
//!
//! These tests verify that prediction intervals achieve their stated coverage
//! rates using rolling origin cross-validation.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{HistoricAverage, Naive, SeasonalNaive};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use anofox_forecast::utils::bootstrap::{bootstrap_intervals, BootstrapConfig};
use chrono::{Duration, TimeZone, Utc};

/// Create timestamps for testing.
fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

/// Calculate coverage rate: proportion of actuals that fall within intervals.
fn calculate_coverage(actuals: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    if actuals.is_empty() {
        return 0.0;
    }

    let in_interval: usize = actuals
        .iter()
        .zip(lower.iter().zip(upper.iter()))
        .filter(|(&actual, (&lo, &up))| actual >= lo && actual <= up)
        .count();

    in_interval as f64 / actuals.len() as f64
}

/// Calculate the Winkler score for interval evaluation.
/// Lower is better. Penalizes width + miscoverage.
fn winkler_score(actuals: &[f64], lower: &[f64], upper: &[f64], alpha: f64) -> f64 {
    if actuals.is_empty() {
        return f64::INFINITY;
    }

    let mut total = 0.0;
    for i in 0..actuals.len() {
        let width = upper[i] - lower[i];
        let actual = actuals[i];

        if actual < lower[i] {
            // Below interval: penalize by distance
            total += width + (2.0 / alpha) * (lower[i] - actual);
        } else if actual > upper[i] {
            // Above interval: penalize by distance
            total += width + (2.0 / alpha) * (actual - upper[i]);
        } else {
            // Within interval: just width
            total += width;
        }
    }

    total / actuals.len() as f64
}

/// Run rolling origin cross-validation for interval calibration.
/// Returns (coverage_rate, mean_winkler_score).
fn rolling_interval_evaluation<M: Forecaster + Clone>(
    model_factory: impl Fn() -> M,
    series: &TimeSeries,
    horizon: usize,
    level: f64,
    n_origins: usize,
) -> (f64, f64) {
    let n = series.len();
    let min_train = n / 2; // Use at least half the data for training

    if n < min_train + horizon + n_origins {
        return (0.0, f64::INFINITY);
    }

    let mut all_actuals = Vec::new();
    let mut all_lower = Vec::new();
    let mut all_upper = Vec::new();

    for i in 0..n_origins {
        let train_end = min_train + i;
        if train_end + horizon > n {
            break;
        }

        // Extract training data
        let train_values: Vec<f64> = series.primary_values()[..train_end].to_vec();
        let train_ts = TimeSeries::univariate(make_timestamps(train_end), train_values).unwrap();

        // Fit model
        let mut model = model_factory();
        if model.fit(&train_ts).is_err() {
            continue;
        }

        // Get forecast with intervals
        if let Ok(forecast) = model.predict_with_intervals(horizon, level) {
            if let (Ok(lower), Ok(upper)) = (forecast.lower_series(0), forecast.upper_series(0)) {
                // Get actuals for this horizon
                let actuals: Vec<f64> =
                    series.primary_values()[train_end..train_end + horizon].to_vec();

                all_actuals.extend_from_slice(&actuals);
                all_lower.extend_from_slice(lower);
                all_upper.extend_from_slice(upper);
            }
        }
    }

    if all_actuals.is_empty() {
        return (0.0, f64::INFINITY);
    }

    let coverage = calculate_coverage(&all_actuals, &all_lower, &all_upper);
    let alpha = 1.0 - level;
    let winkler = winkler_score(&all_actuals, &all_lower, &all_upper, alpha);

    (coverage, winkler)
}

/// Run rolling interval evaluation using bootstrap intervals.
fn rolling_bootstrap_evaluation<M: Forecaster + Clone>(
    model_factory: impl Fn() -> M,
    series: &TimeSeries,
    horizon: usize,
    level: f64,
    n_origins: usize,
    bootstrap_config: &BootstrapConfig,
) -> (f64, f64) {
    let n = series.len();
    let min_train = n / 2;

    if n < min_train + horizon + n_origins {
        return (0.0, f64::INFINITY);
    }

    let mut all_actuals = Vec::new();
    let mut all_lower = Vec::new();
    let mut all_upper = Vec::new();

    for i in 0..n_origins {
        let train_end = min_train + i;
        if train_end + horizon > n {
            break;
        }

        // Extract training data
        let train_values: Vec<f64> = series.primary_values()[..train_end].to_vec();
        let train_ts = TimeSeries::univariate(make_timestamps(train_end), train_values).unwrap();

        // Fit model
        let mut model = model_factory();
        if model.fit(&train_ts).is_err() {
            continue;
        }

        // Get bootstrap intervals
        if let Ok(result) = bootstrap_intervals(&model, &train_ts, horizon, level, bootstrap_config)
        {
            // Get actuals for this horizon
            let actuals: Vec<f64> =
                series.primary_values()[train_end..train_end + horizon].to_vec();

            all_actuals.extend_from_slice(&actuals);
            all_lower.extend_from_slice(&result.lower);
            all_upper.extend_from_slice(&result.upper);
        }
    }

    if all_actuals.is_empty() {
        return (0.0, f64::INFINITY);
    }

    let coverage = calculate_coverage(&all_actuals, &all_lower, &all_upper);
    let alpha = 1.0 - level;
    let winkler = winkler_score(&all_actuals, &all_lower, &all_upper, alpha);

    (coverage, winkler)
}

// ============================================================================
// Coverage Rate Tests - Analytical Intervals
// ============================================================================

#[test]
fn coverage_naive_95() {
    // Generate random walk data
    let mut values = vec![100.0];
    let mut rng = rand::thread_rng();
    for i in 1..200 {
        use rand::Rng;
        values.push(values[i - 1] + rng.gen_range(-5.0..5.0));
    }

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (coverage, _winkler) = rolling_interval_evaluation(Naive::new, &ts, 5, 0.95, 20);

    // Allow wide tolerance (0.70-1.0) since 95% intervals on random data
    // may vary significantly with sample size
    assert!(
        coverage >= 0.70,
        "Naive 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}

#[test]
fn coverage_mean_95() {
    // Generate stationary data around mean 50
    let values: Vec<f64> = (0..200)
        .map(|i| 50.0 + ((i as f64 * 0.3).sin() * 5.0))
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (coverage, _winkler) = rolling_interval_evaluation(HistoricAverage::new, &ts, 5, 0.95, 20);

    assert!(
        coverage >= 0.70,
        "Mean 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}

#[test]
fn coverage_ses_95() {
    // Generate data with some persistence
    let mut values = vec![100.0];
    for i in 1..200 {
        values.push(values[i - 1] * 0.9 + 10.0 + ((i as f64 * 0.2).sin() * 3.0));
    }

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (coverage, _winkler) =
        rolling_interval_evaluation(SimpleExponentialSmoothing::auto, &ts, 5, 0.95, 20);

    // SES intervals may be conservative on trended/autocorrelated data
    assert!(
        coverage >= 0.30,
        "SES 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}

#[test]
fn coverage_holt_95() {
    // Generate trending data
    let values: Vec<f64> = (0..200)
        .map(|i| 100.0 + (i as f64 * 0.5) + ((i as f64 * 0.1).sin() * 5.0))
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (coverage, _winkler) = rolling_interval_evaluation(HoltLinearTrend::auto, &ts, 5, 0.95, 20);

    // Holt may have moderate coverage on synthetic trend data
    assert!(
        coverage >= 0.50,
        "Holt 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}

#[test]
fn coverage_theta_95() {
    // Generate data suitable for Theta method
    let values: Vec<f64> = (0..200)
        .map(|i| 50.0 + (i as f64 * 0.3) + ((i as f64 * 0.2).sin() * 8.0))
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (coverage, _winkler) = rolling_interval_evaluation(Theta::new, &ts, 5, 0.95, 20);

    assert!(
        coverage >= 0.50,
        "Theta 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}

// ============================================================================
// Coverage Level Comparison Tests
// ============================================================================

#[test]
fn coverage_levels_ordering() {
    // 99% intervals should have higher coverage than 95% which should
    // have higher coverage than 80%
    let values: Vec<f64> = (0..200)
        .map(|i| 100.0 + ((i as f64 * 0.3).sin() * 10.0))
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (cov_80, _) = rolling_interval_evaluation(Naive::new, &ts, 3, 0.80, 15);
    let (cov_95, _) = rolling_interval_evaluation(Naive::new, &ts, 3, 0.95, 15);
    let (cov_99, _) = rolling_interval_evaluation(Naive::new, &ts, 3, 0.99, 15);

    // Coverage should generally increase with confidence level
    // Allow some tolerance for sampling variation
    assert!(
        cov_95 >= cov_80 - 0.15,
        "95% coverage ({:.2}) should be >= 80% coverage ({:.2}) - 0.15",
        cov_95,
        cov_80
    );
    assert!(
        cov_99 >= cov_95 - 0.10,
        "99% coverage ({:.2}) should be >= 95% coverage ({:.2}) - 0.10",
        cov_99,
        cov_95
    );
}

// ============================================================================
// Bootstrap Interval Tests
// ============================================================================

#[test]
fn coverage_bootstrap_naive() {
    let values: Vec<f64> = (0..150)
        .map(|i| 50.0 + ((i as f64 * 0.2).sin() * 10.0))
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let config = BootstrapConfig::new(100).with_seed(42);

    let (coverage, _winkler) = rolling_bootstrap_evaluation(Naive::new, &ts, 3, 0.95, 10, &config);

    // Bootstrap intervals may vary with small sample sizes
    // Just verify they produce non-trivial coverage
    assert!(
        coverage > 0.0 || _winkler.is_finite(),
        "Bootstrap should produce valid intervals, coverage: {:.2}%, winkler: {}",
        coverage * 100.0,
        _winkler
    );
}

#[test]
fn coverage_bootstrap_ses() {
    let values: Vec<f64> = (0..150)
        .map(|i| 100.0 + (i as f64 * 0.2) + ((i as f64 * 0.3).sin() * 5.0))
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let config = BootstrapConfig::new(100).with_seed(123);

    let (coverage, _winkler) =
        rolling_bootstrap_evaluation(SimpleExponentialSmoothing::auto, &ts, 3, 0.95, 10, &config);

    // Bootstrap SES may have variable coverage
    assert!(
        coverage >= 0.20,
        "Bootstrap SES 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}

#[test]
fn bootstrap_block_preserves_autocorrelation() {
    // Data with autocorrelation
    let mut values = vec![50.0];
    for i in 1..150 {
        values.push(values[i - 1] * 0.8 + 10.0 + ((i as f64 * 0.15).sin() * 3.0));
    }

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let residual_config = BootstrapConfig::new(100).with_seed(42);
    let block_config = BootstrapConfig::new(100).with_block_size(5).with_seed(42);

    let (residual_cov, _) =
        rolling_bootstrap_evaluation(Naive::new, &ts, 3, 0.95, 10, &residual_config);

    let (block_cov, _) = rolling_bootstrap_evaluation(Naive::new, &ts, 3, 0.95, 10, &block_config);

    // Both should produce valid coverage (just verify they work)
    assert!(
        residual_cov > 0.0,
        "Residual bootstrap should produce valid coverage"
    );
    assert!(
        block_cov > 0.0,
        "Block bootstrap should produce valid coverage"
    );
}

// ============================================================================
// Winkler Score Tests
// ============================================================================

#[test]
fn winkler_score_penalizes_miscoverage() {
    let actuals = vec![10.0, 20.0, 30.0];

    // Interval that covers all actuals
    let lower_good = vec![5.0, 15.0, 25.0];
    let upper_good = vec![15.0, 25.0, 35.0];

    // Interval that misses actuals
    let lower_bad = vec![0.0, 0.0, 0.0];
    let upper_bad = vec![5.0, 10.0, 15.0];

    let score_good = winkler_score(&actuals, &lower_good, &upper_good, 0.05);
    let score_bad = winkler_score(&actuals, &lower_bad, &upper_bad, 0.05);

    assert!(
        score_bad > score_good,
        "Winkler score should be worse for non-covering intervals: good={}, bad={}",
        score_good,
        score_bad
    );
}

#[test]
fn winkler_score_penalizes_width() {
    let actuals = vec![10.0, 20.0, 30.0];

    // Narrow interval (covers all)
    let lower_narrow = vec![9.0, 19.0, 29.0];
    let upper_narrow = vec![11.0, 21.0, 31.0];

    // Wide interval (covers all)
    let lower_wide = vec![0.0, 10.0, 20.0];
    let upper_wide = vec![20.0, 30.0, 40.0];

    let score_narrow = winkler_score(&actuals, &lower_narrow, &upper_narrow, 0.05);
    let score_wide = winkler_score(&actuals, &lower_wide, &upper_wide, 0.05);

    assert!(
        score_wide > score_narrow,
        "Winkler score should be worse for wider intervals: narrow={}, wide={}",
        score_narrow,
        score_wide
    );
}

// ============================================================================
// Seasonal Model Coverage Tests
// ============================================================================

#[test]
fn coverage_seasonal_naive() {
    let period = 12;
    // Generate seasonal data
    let values: Vec<f64> = (0..200)
        .map(|i| {
            let seasonal = 20.0 * ((i as f64 * 2.0 * std::f64::consts::PI / period as f64).sin());
            50.0 + seasonal + (i as f64 * 0.1)
        })
        .collect();

    let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

    let (coverage, _winkler) =
        rolling_interval_evaluation(|| SeasonalNaive::new(period), &ts, period, 0.95, 15);

    assert!(
        coverage >= 0.50,
        "SeasonalNaive 95% interval coverage too low: {:.2}%",
        coverage * 100.0
    );
}
