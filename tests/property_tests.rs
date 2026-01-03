//! Property-based tests for forecasting models.
//!
//! These tests verify invariants that should hold for all valid inputs,
//! using randomly generated time series data.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::ARIMA;
use anofox_forecast::models::baseline::{Naive, SeasonalNaive};
use anofox_forecast::models::exponential::{HoltLinearTrend, SimpleExponentialSmoothing};
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use proptest::prelude::*;

/// Create a TimeSeries from a vector of values.
fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::hours(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

/// Strategy for generating valid time series values.
/// Avoids extreme values that could cause numerical issues.
/// Adds small variation to avoid all-constant series which can cause NaN.
fn valid_values_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    (min_len..max_len).prop_flat_map(|len| {
        prop::collection::vec(1.0..1000.0_f64, len).prop_map(|mut v| {
            // Add small variation to ensure non-zero variance
            for (i, val) in v.iter_mut().enumerate() {
                *val += (i as f64) * 0.001;
            }
            v
        })
    })
}

/// Strategy for generating time series with trend.
fn trending_values_strategy(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    (min_len..max_len).prop_flat_map(|len| {
        (0.0..100.0_f64, 0.1..2.0_f64)
            .prop_map(move |(base, slope)| (0..len).map(|i| base + slope * i as f64).collect())
    })
}

/// Strategy for generating seasonal time series.
fn seasonal_values_strategy(
    min_len: usize,
    max_len: usize,
    period: usize,
) -> impl Strategy<Value = Vec<f64>> {
    (min_len..max_len).prop_flat_map(move |len| {
        (50.0..100.0_f64, 5.0..20.0_f64).prop_map(move |(base, amplitude)| {
            (0..len)
                .map(|i| {
                    base + amplitude * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                })
                .collect()
        })
    })
}

// =============================================================================
// Property: Forecast length matches requested horizon
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn naive_forecast_length_matches_horizon(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        prop_assert_eq!(forecast.horizon(), horizon);
    }

    #[test]
    fn ses_forecast_length_matches_horizon(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        prop_assert_eq!(forecast.horizon(), horizon);
    }

    #[test]
    fn holt_forecast_length_matches_horizon(
        values in trending_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = HoltLinearTrend::auto();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        prop_assert_eq!(forecast.horizon(), horizon);
    }

    #[test]
    fn theta_forecast_length_matches_horizon(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Theta::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        prop_assert_eq!(forecast.horizon(), horizon);
    }

    #[test]
    fn arima_forecast_length_matches_horizon(
        values in valid_values_strategy(30, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = ARIMA::new(1, 0, 1);
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        prop_assert_eq!(forecast.horizon(), horizon);
    }
}

// =============================================================================
// Property: Forecast values are finite (not NaN or Inf)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn naive_forecasts_are_finite(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        for val in forecast.primary() {
            prop_assert!(val.is_finite(), "Forecast contains non-finite value: {}", val);
        }
    }

    #[test]
    fn ses_forecasts_are_finite(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        for val in forecast.primary() {
            prop_assert!(val.is_finite(), "Forecast contains non-finite value: {}", val);
        }
    }

    #[test]
    fn theta_forecasts_are_finite(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Theta::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        for val in forecast.primary() {
            prop_assert!(val.is_finite(), "Forecast contains non-finite value: {}", val);
        }
    }

    #[test]
    fn arima_forecasts_are_finite(
        values in valid_values_strategy(30, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = ARIMA::new(1, 0, 1);
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();
        for val in forecast.primary() {
            prop_assert!(val.is_finite(), "Forecast contains non-finite value: {}", val);
        }
    }
}

// =============================================================================
// Property: Confidence intervals are properly ordered (lower <= point <= upper)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn naive_intervals_ordered(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict_with_intervals(horizon, 0.95).unwrap();

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let point = forecast.primary();

        for i in 0..horizon {
            prop_assert!(
                lower[i] <= point[i],
                "Lower bound {} > point {} at horizon {}",
                lower[i], point[i], i
            );
            prop_assert!(
                point[i] <= upper[i],
                "Point {} > upper bound {} at horizon {}",
                point[i], upper[i], i
            );
        }
    }

    #[test]
    fn ses_intervals_ordered(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();
        let forecast = model.predict_with_intervals(horizon, 0.95).unwrap();

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let point = forecast.primary();

        for i in 0..horizon {
            prop_assert!(
                lower[i] <= point[i],
                "Lower bound {} > point {} at horizon {}",
                lower[i], point[i], i
            );
            prop_assert!(
                point[i] <= upper[i],
                "Point {} > upper bound {} at horizon {}",
                point[i], upper[i], i
            );
        }
    }

    #[test]
    fn arima_intervals_ordered(
        values in valid_values_strategy(30, 100),
        horizon in 1usize..20
    ) {
        let ts = make_ts(&values);
        let mut model = ARIMA::new(1, 0, 1);
        model.fit(&ts).unwrap();
        let forecast = model.predict_with_intervals(horizon, 0.95).unwrap();

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let point = forecast.primary();

        for i in 0..horizon {
            prop_assert!(
                lower[i] <= point[i],
                "Lower bound {} > point {} at horizon {}",
                lower[i], point[i], i
            );
            prop_assert!(
                point[i] <= upper[i],
                "Point {} > upper bound {} at horizon {}",
                point[i], upper[i], i
            );
        }
    }
}

// =============================================================================
// Property: Intervals widen with horizon (monotonically non-decreasing width)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn naive_intervals_widen_with_horizon(
        values in valid_values_strategy(20, 100),
        horizon in 3usize..15
    ) {
        let ts = make_ts(&values);
        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict_with_intervals(horizon, 0.95).unwrap();

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        for i in 1..horizon {
            let prev_width = upper[i - 1] - lower[i - 1];
            let curr_width = upper[i] - lower[i];
            prop_assert!(
                curr_width >= prev_width - 1e-10,
                "Interval width decreased: {} -> {} at horizon {}",
                prev_width, curr_width, i
            );
        }
    }

    #[test]
    fn ses_intervals_widen_with_horizon(
        values in valid_values_strategy(20, 100),
        horizon in 3usize..15
    ) {
        let ts = make_ts(&values);
        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();
        let forecast = model.predict_with_intervals(horizon, 0.95).unwrap();

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();

        for i in 1..horizon {
            let prev_width = upper[i - 1] - lower[i - 1];
            let curr_width = upper[i] - lower[i];
            prop_assert!(
                curr_width >= prev_width - 1e-10,
                "Interval width decreased: {} -> {} at horizon {}",
                prev_width, curr_width, i
            );
        }
    }
}

// =============================================================================
// Property: Fitted values + residuals = original values (for SES)
// Note: Naive has complex indexing, so we only test SES here
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn ses_fitted_plus_residuals_equals_original(
        values in valid_values_strategy(20, 100)
    ) {
        let ts = make_ts(&values);
        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();

        if let (Some(fitted), Some(residuals)) = (model.fitted_values(), model.residuals()) {
            for i in 0..fitted.len().min(residuals.len()).min(values.len()) {
                let reconstructed = fitted[i] + residuals[i];
                let diff = (reconstructed - values[i]).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Reconstruction error {} at index {}",
                    diff, i
                );
            }
        }
    }
}

// =============================================================================
// Property: Re-fitting produces consistent results (idempotent)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn naive_fit_is_idempotent(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..10
    ) {
        let ts = make_ts(&values);

        let mut model1 = Naive::new();
        model1.fit(&ts).unwrap();
        let forecast1 = model1.predict(horizon).unwrap();

        let mut model2 = Naive::new();
        model2.fit(&ts).unwrap();
        model2.fit(&ts).unwrap(); // Fit twice
        let forecast2 = model2.predict(horizon).unwrap();

        for i in 0..horizon {
            let diff = (forecast1.primary()[i] - forecast2.primary()[i]).abs();
            prop_assert!(
                diff < 1e-10,
                "Forecasts differ after re-fitting: {} vs {} at horizon {}",
                forecast1.primary()[i], forecast2.primary()[i], i
            );
        }
    }

    #[test]
    fn ses_fit_is_idempotent(
        values in valid_values_strategy(20, 100),
        horizon in 1usize..10
    ) {
        let ts = make_ts(&values);

        let mut model1 = SimpleExponentialSmoothing::auto();
        model1.fit(&ts).unwrap();
        let forecast1 = model1.predict(horizon).unwrap();

        let mut model2 = SimpleExponentialSmoothing::auto();
        model2.fit(&ts).unwrap();
        model2.fit(&ts).unwrap(); // Fit twice
        let forecast2 = model2.predict(horizon).unwrap();

        for i in 0..horizon {
            let diff = (forecast1.primary()[i] - forecast2.primary()[i]).abs();
            prop_assert!(
                diff < 1e-10,
                "Forecasts differ after re-fitting: {} vs {} at horizon {}",
                forecast1.primary()[i], forecast2.primary()[i], i
            );
        }
    }
}

// =============================================================================
// Property: Seasonal naive respects period
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn seasonal_naive_uses_correct_period(
        values in seasonal_values_strategy(50, 100, 7),
        horizon in 7usize..21
    ) {
        let ts = make_ts(&values);
        let mut model = SeasonalNaive::new(7);
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();

        // For seasonal naive, forecast[i] should equal values[len - period + (i % period)]
        let n = values.len();
        for i in 0..horizon {
            let expected_idx = n - 7 + (i % 7);
            let expected = values[expected_idx];
            let actual = forecast.primary()[i];
            let diff = (expected - actual).abs();
            prop_assert!(
                diff < 1e-10,
                "Seasonal pattern broken at horizon {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }
}

// =============================================================================
// Property: Constant series produces constant forecast
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn constant_series_produces_constant_naive_forecast(
        constant in 1.0..1000.0_f64,
        length in 20usize..50,
        horizon in 1usize..20
    ) {
        let values: Vec<f64> = vec![constant; length];
        let ts = make_ts(&values);

        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();

        for (i, &val) in forecast.primary().iter().enumerate() {
            let diff = (val - constant).abs();
            prop_assert!(
                diff < 1e-10,
                "Constant series forecast should be constant: {} vs {} at {}",
                constant, val, i
            );
        }
    }

    #[test]
    fn constant_series_produces_constant_ses_forecast(
        constant in 1.0..1000.0_f64,
        length in 20usize..50,
        horizon in 1usize..20
    ) {
        let values: Vec<f64> = vec![constant; length];
        let ts = make_ts(&values);

        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();
        let forecast = model.predict(horizon).unwrap();

        for (i, &val) in forecast.primary().iter().enumerate() {
            let diff = (val - constant).abs();
            prop_assert!(
                diff < 1e-10,
                "Constant series forecast should be constant: {} vs {} at {}",
                constant, val, i
            );
        }
    }
}
