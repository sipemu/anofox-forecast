//! Integration tests for model serialization / persistence round-trips.
//!
//! For each major model type we:
//!   1. Fit the model
//!   2. Get predictions
//!   3. Serialize to JSON
//!   4. Verify serialized data is non-empty
//!   5. Deserialize and verify predictions match
//!
//! Note: Model types that use the `nan_vec` custom serializer for NaN-aware
//! JSON handling are NOT compatible with bincode. Bincode round-trips are
//! tested only for Forecast and TimeSeries, which work correctly with both
//! formats.

#[cfg(feature = "serde")]
mod serde_tests {
    use anofox_forecast::core::{Forecast, TimeSeries};
    use anofox_forecast::models::arima::ARIMA;
    use anofox_forecast::models::baseline::Naive;
    use anofox_forecast::models::exponential::{ETSSpec, SimpleExponentialSmoothing, ETS};
    use anofox_forecast::models::garch::GARCH;
    use anofox_forecast::models::theta::Theta;
    use anofox_forecast::models::Forecaster;
    use anofox_forecast::utils::persistence::{from_bincode, from_json, to_bincode, to_json};
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::days(i as i64)).collect()
    }

    /// Trend + seasonality + noise test data.
    fn make_test_data(n: usize) -> (Vec<chrono::DateTime<Utc>>, Vec<f64>) {
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 50.0 + 0.3 * i as f64;
                let season = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin();
                let noise = ((42u64.wrapping_mul(i as u64 + 1) % 1000) as f64 - 500.0) / 500.0;
                trend + season + noise
            })
            .collect();
        (timestamps, values)
    }

    fn make_ts(n: usize) -> TimeSeries {
        let (timestamps, values) = make_test_data(n);
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    /// Assert two f64 slices are approximately equal.
    fn assert_slices_approx_eq(a: &[f64], b: &[f64], tol: f64, context: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", context);
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff < tol,
                "{}: mismatch at index {}: {} vs {} (diff {})",
                context,
                i,
                va,
                vb,
                diff
            );
        }
    }

    // -----------------------------------------------------------------------
    // Naive model JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn naive_json_round_trip_predictions_match() {
        let ts = make_ts(50);
        let horizon = 5;

        let mut model = Naive::new();
        model.fit(&ts).unwrap();
        let original_forecast = model.predict(horizon).unwrap();

        let json = to_json(&model).unwrap();
        assert!(json.len() > 10, "JSON should be non-empty");

        let restored: Naive = from_json(&json).unwrap();
        let restored_forecast = restored.predict(horizon).unwrap();

        assert_eq!(original_forecast.horizon(), restored_forecast.horizon());
        assert_slices_approx_eq(
            original_forecast.primary(),
            restored_forecast.primary(),
            1e-10,
            "Naive JSON",
        );
    }

    // -----------------------------------------------------------------------
    // SES model JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn ses_json_round_trip_predictions_match() {
        let ts = make_ts(60);
        let horizon = 8;

        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();
        let original_forecast = model.predict(horizon).unwrap();

        let json = to_json(&model).unwrap();
        assert!(json.len() > 10, "SES JSON should be non-empty");

        let restored: SimpleExponentialSmoothing = from_json(&json).unwrap();
        let restored_forecast = restored.predict(horizon).unwrap();

        assert_eq!(original_forecast.horizon(), restored_forecast.horizon());
        assert_slices_approx_eq(
            original_forecast.primary(),
            restored_forecast.primary(),
            1e-10,
            "SES JSON",
        );
    }

    // -----------------------------------------------------------------------
    // ETS model JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn ets_json_round_trip_predictions_match() {
        let ts = make_ts(60);
        let horizon = 5;

        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();
        let original_forecast = model.predict(horizon).unwrap();

        let json = to_json(&model).unwrap();
        assert!(json.len() > 10, "ETS JSON should be non-empty");

        let restored: ETS = from_json(&json).unwrap();
        let restored_forecast = restored.predict(horizon).unwrap();

        assert_eq!(original_forecast.horizon(), restored_forecast.horizon());
        assert_slices_approx_eq(
            original_forecast.primary(),
            restored_forecast.primary(),
            1e-10,
            "ETS JSON",
        );
    }

    // -----------------------------------------------------------------------
    // ARIMA model JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn arima_json_round_trip_predictions_match() {
        let ts = make_ts(80);
        let horizon = 5;

        let mut model = ARIMA::new(1, 1, 0);
        model.fit(&ts).unwrap();
        let original_forecast = model.predict(horizon).unwrap();

        let json = to_json(&model).unwrap();
        assert!(json.len() > 10, "ARIMA JSON should be non-empty");

        let restored: ARIMA = from_json(&json).unwrap();
        let restored_forecast = restored.predict(horizon).unwrap();

        assert_eq!(original_forecast.horizon(), restored_forecast.horizon());
        assert_slices_approx_eq(
            original_forecast.primary(),
            restored_forecast.primary(),
            1e-10,
            "ARIMA JSON",
        );
    }

    // -----------------------------------------------------------------------
    // Theta model JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn theta_json_round_trip_predictions_match() {
        let ts = make_ts(60);
        let horizon = 5;

        let mut model = Theta::new();
        model.fit(&ts).unwrap();
        let original_forecast = model.predict(horizon).unwrap();

        let json = to_json(&model).unwrap();
        assert!(json.len() > 10, "Theta JSON should be non-empty");

        let restored: Theta = from_json(&json).unwrap();
        let restored_forecast = restored.predict(horizon).unwrap();

        assert_eq!(original_forecast.horizon(), restored_forecast.horizon());
        assert_slices_approx_eq(
            original_forecast.primary(),
            restored_forecast.primary(),
            1e-10,
            "Theta JSON",
        );
    }

    // -----------------------------------------------------------------------
    // GARCH model JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn garch_json_round_trip_predictions_match() {
        // GARCH needs enough data and variance
        let n = 100;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let base = 100.0;
                let vol = if (i / 20) % 2 == 0 { 5.0 } else { 15.0 };
                let noise = ((31u64.wrapping_mul(i as u64 + 7) % 1000) as f64 - 500.0) / 500.0;
                base + vol * noise
            })
            .collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();
        let horizon = 5;

        let mut model = GARCH::new(1, 1);
        model.fit(&ts).unwrap();
        let original_forecast = model.predict(horizon).unwrap();

        let json = to_json(&model).unwrap();
        assert!(json.len() > 10, "GARCH JSON should be non-empty");

        let restored: GARCH = from_json(&json).unwrap();
        let restored_forecast = restored.predict(horizon).unwrap();

        assert_eq!(original_forecast.horizon(), restored_forecast.horizon());
        assert_slices_approx_eq(
            original_forecast.primary(),
            restored_forecast.primary(),
            1e-10,
            "GARCH JSON",
        );
    }

    // -----------------------------------------------------------------------
    // JSON serialized output is non-empty for each model
    // -----------------------------------------------------------------------

    #[test]
    fn json_serialization_produces_substantial_output() {
        let ts = make_ts(60);

        // Naive
        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();
        let json = to_json(&naive).unwrap();
        assert!(
            json.len() > 50,
            "Naive JSON too small: {} bytes",
            json.len()
        );

        // SES
        let mut ses = SimpleExponentialSmoothing::auto();
        ses.fit(&ts).unwrap();
        let json = to_json(&ses).unwrap();
        assert!(json.len() > 50, "SES JSON too small: {} bytes", json.len());

        // ETS
        let mut ets = ETS::new(ETSSpec::ann(), 1);
        ets.fit(&ts).unwrap();
        let json = to_json(&ets).unwrap();
        assert!(json.len() > 50, "ETS JSON too small: {} bytes", json.len());

        // ARIMA
        let ts80 = make_ts(80);
        let mut arima = ARIMA::new(1, 1, 0);
        arima.fit(&ts80).unwrap();
        let json = to_json(&arima).unwrap();
        assert!(
            json.len() > 50,
            "ARIMA JSON too small: {} bytes",
            json.len()
        );

        // Theta
        let mut theta = Theta::new();
        theta.fit(&ts).unwrap();
        let json = to_json(&theta).unwrap();
        assert!(
            json.len() > 50,
            "Theta JSON too small: {} bytes",
            json.len()
        );
    }

    // -----------------------------------------------------------------------
    // TimeSeries serialization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn time_series_json_round_trip_preserves_values_and_timestamps() {
        let n = 30;
        let (timestamps, values) = make_test_data(n);
        let mut ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();
        ts.set_frequency(Duration::days(1));

        let json = to_json(&ts).unwrap();
        assert!(json.len() > 10, "TimeSeries JSON should be non-empty");

        let restored: TimeSeries = from_json(&json).unwrap();

        assert_eq!(restored.len(), n);
        assert_eq!(restored.timestamps(), &timestamps);
        assert_eq!(restored.frequency(), Some(Duration::days(1)));

        // Use approximate comparison for floating-point values
        assert_slices_approx_eq(
            restored.primary_values(),
            &values,
            1e-10,
            "TimeSeries JSON values",
        );
    }

    #[test]
    fn time_series_bincode_round_trip_preserves_values_and_timestamps() {
        let n = 30;
        let (timestamps, values) = make_test_data(n);
        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();

        let bytes = to_bincode(&ts).unwrap();
        assert!(!bytes.is_empty(), "TimeSeries bincode should be non-empty");

        let restored: TimeSeries = from_bincode(&bytes).unwrap();

        assert_eq!(restored.len(), n);
        assert_eq!(restored.primary_values(), &values);
        assert_eq!(restored.timestamps(), &timestamps);
    }

    // -----------------------------------------------------------------------
    // Forecast serialization round-trip (JSON + bincode)
    // -----------------------------------------------------------------------

    #[test]
    fn forecast_json_round_trip_preserves_point_and_intervals() {
        let point = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let lower = vec![8.0, 18.0, 28.0, 38.0, 48.0];
        let upper = vec![12.0, 22.0, 32.0, 42.0, 52.0];
        let forecast =
            Forecast::from_values_with_intervals(point.clone(), lower.clone(), upper.clone());

        let json = to_json(&forecast).unwrap();
        assert!(json.len() > 10, "Forecast JSON should be non-empty");

        let restored: Forecast = from_json(&json).unwrap();

        assert_eq!(restored.horizon(), 5);
        assert_eq!(restored.primary(), &point);
        assert!(restored.has_lower());
        assert!(restored.has_upper());

        let lower_restored = restored.lower_series(0).unwrap();
        let upper_restored = restored.upper_series(0).unwrap();
        assert_eq!(lower_restored, &lower);
        assert_eq!(upper_restored, &upper);
    }

    #[test]
    fn forecast_bincode_round_trip_preserves_all_data() {
        let point = vec![1.5, 2.5, 3.5];
        let lower = vec![1.0, 2.0, 3.0];
        let upper = vec![2.0, 3.0, 4.0];
        let forecast =
            Forecast::from_values_with_intervals(point.clone(), lower.clone(), upper.clone());

        let bytes = to_bincode(&forecast).unwrap();
        assert!(!bytes.is_empty(), "Forecast bincode should be non-empty");

        let restored: Forecast = from_bincode(&bytes).unwrap();
        assert_eq!(restored, forecast);
    }

    #[test]
    fn forecast_point_only_bincode_round_trip() {
        let forecast = Forecast::from_values(vec![100.0, 200.0, 300.0, 400.0]);

        let bytes = to_bincode(&forecast).unwrap();
        let restored: Forecast = from_bincode(&bytes).unwrap();

        assert_eq!(restored.horizon(), 4);
        assert_eq!(restored.primary(), forecast.primary());
        assert!(!restored.has_lower());
        assert!(!restored.has_upper());
    }

    // -----------------------------------------------------------------------
    // Bincode is more compact than JSON (for types that support both)
    // -----------------------------------------------------------------------

    #[test]
    fn bincode_is_more_compact_for_forecasts() {
        let forecast = Forecast::from_values_with_intervals(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![8.0, 18.0, 28.0, 38.0, 48.0],
            vec![12.0, 22.0, 32.0, 42.0, 52.0],
        );

        let json = to_json(&forecast).unwrap();
        let bincode_bytes = to_bincode(&forecast).unwrap();

        assert!(
            bincode_bytes.len() < json.len(),
            "Forecast: bincode ({} bytes) should be smaller than JSON ({} bytes)",
            bincode_bytes.len(),
            json.len()
        );
    }

    #[test]
    fn bincode_is_more_compact_for_time_series() {
        let (timestamps, values) = make_test_data(50);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let json = to_json(&ts).unwrap();
        let bincode_bytes = to_bincode(&ts).unwrap();

        assert!(
            bincode_bytes.len() < json.len(),
            "TimeSeries: bincode ({} bytes) should be smaller than JSON ({} bytes)",
            bincode_bytes.len(),
            json.len()
        );
    }

    // -----------------------------------------------------------------------
    // Round-trip preserves model fit state
    // -----------------------------------------------------------------------

    #[test]
    fn json_round_trip_preserves_fitted_state() {
        let ts = make_ts(50);

        // Naive should be fitted after restore
        let mut naive = Naive::new();
        naive.fit(&ts).unwrap();
        assert!(naive.is_fitted());

        let json = to_json(&naive).unwrap();
        let restored: Naive = from_json(&json).unwrap();
        assert!(
            restored.is_fitted(),
            "Naive should remain fitted after JSON round-trip"
        );
    }

    #[test]
    fn json_round_trip_unfitted_model_stays_unfitted() {
        let model = Naive::new();
        assert!(!model.is_fitted());

        let json = to_json(&model).unwrap();
        let restored: Naive = from_json(&json).unwrap();
        assert!(
            !restored.is_fitted(),
            "unfitted Naive should remain unfitted after JSON round-trip"
        );
    }
}
