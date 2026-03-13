//! Save/load convenience functions for serializable models.

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Serialize};

/// Serde helper module for `Option<Vec<f64>>` fields that may contain NaN values.
///
/// JSON does not support NaN, so `serde_json` serializes NaN as `null`.
/// This module provides custom (de)serialization that maps `null` back to `f64::NAN`.
#[cfg(feature = "serde")]
pub mod nan_vec {
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<Vec<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<Vec<Option<f64>>> = Option::deserialize(deserializer)?;
        Ok(opt.map(|v| v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect()))
    }
}

/// Serde helper module for `Option<chrono::Duration>` fields.
///
/// `chrono::Duration` does not implement `Serialize`/`Deserialize`, so this
/// module converts to and from an `Option<i64>` representing whole seconds.
#[cfg(feature = "serde")]
pub mod opt_duration_secs {
    use chrono::Duration;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(d) => serializer.serialize_some(&d.num_seconds()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<i64> = Option::deserialize(deserializer)?;
        Ok(opt.map(Duration::seconds))
    }
}

/// Save a serializable model to a JSON string.
#[cfg(feature = "serde")]
pub fn to_json<T: Serialize>(model: &T) -> crate::error::Result<String> {
    serde_json::to_string_pretty(model).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!("serialization failed: {}", e))
    })
}

/// Load a model from a JSON string.
#[cfg(feature = "serde")]
pub fn from_json<T: DeserializeOwned>(json: &str) -> crate::error::Result<T> {
    serde_json::from_str(json).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!("deserialization failed: {}", e))
    })
}

/// Save a model to a file.
#[cfg(feature = "serde")]
pub fn save_to_file<T: Serialize>(model: &T, path: &std::path::Path) -> crate::error::Result<()> {
    let json = to_json(model)?;
    std::fs::write(path, json).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!("file write failed: {}", e))
    })
}

/// Load a model from a file.
#[cfg(feature = "serde")]
pub fn load_from_file<T: DeserializeOwned>(path: &std::path::Path) -> crate::error::Result<T> {
    let json = std::fs::read_to_string(path).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!("file read failed: {}", e))
    })?;
    from_json(&json)
}

/// Serialize a model to a bincode byte vector.
#[cfg(feature = "serde")]
pub fn to_bincode<T: Serialize>(model: &T) -> crate::error::Result<Vec<u8>> {
    bincode::serialize(model).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!(
            "bincode serialization failed: {}",
            e
        ))
    })
}

/// Deserialize a model from a bincode byte slice.
#[cfg(feature = "serde")]
pub fn from_bincode<T: DeserializeOwned>(data: &[u8]) -> crate::error::Result<T> {
    bincode::deserialize(data).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!(
            "bincode deserialization failed: {}",
            e
        ))
    })
}

/// Save a model to a file in bincode format.
#[cfg(feature = "serde")]
pub fn save_to_bincode<T: Serialize>(
    model: &T,
    path: &std::path::Path,
) -> crate::error::Result<()> {
    let bytes = to_bincode(model)?;
    std::fs::write(path, bytes).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!("file write failed: {}", e))
    })
}

/// Load a model from a bincode file.
#[cfg(feature = "serde")]
pub fn load_from_bincode<T: DeserializeOwned>(path: &std::path::Path) -> crate::error::Result<T> {
    let bytes = std::fs::read(path).map_err(|e| {
        crate::error::ForecastError::SerializationError(format!("file read failed: {}", e))
    })?;
    from_bincode(&bytes)
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::core::{Forecast, TimeSeries};
    use crate::error::ForecastError;
    use crate::models::Forecaster;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    // ── JSON round-trip tests for models ─────────────────────────────────

    #[test]
    fn naive_round_trip() {
        use crate::models::baseline::Naive;

        let timestamps = make_timestamps(10);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let json = to_json(&model).unwrap();
        let restored: Naive = from_json(&json).unwrap();

        // Predictions should match
        let original_forecast = model.predict(3).unwrap();
        let restored_forecast = restored.predict(3).unwrap();
        assert_eq!(original_forecast.primary(), restored_forecast.primary());
    }

    #[test]
    fn arima_round_trip() {
        use crate::models::arima::ARIMA;

        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 0);
        model.fit(&ts).unwrap();

        let json = to_json(&model).unwrap();
        let restored: ARIMA = from_json(&json).unwrap();

        let original_forecast = model.predict(5).unwrap();
        let restored_forecast = restored.predict(5).unwrap();

        for (a, b) in original_forecast
            .primary()
            .iter()
            .zip(restored_forecast.primary().iter())
        {
            assert!(
                (a - b).abs() < 1e-10,
                "ARIMA forecasts should match after round-trip"
            );
        }
    }

    #[test]
    fn ets_round_trip() {
        use crate::models::exponential::{ETSSpec, ETS};

        let timestamps = make_timestamps(30);
        let values: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.3).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();

        let json = to_json(&model).unwrap();
        let restored: ETS = from_json(&json).unwrap();

        let original_forecast = model.predict(5).unwrap();
        let restored_forecast = restored.predict(5).unwrap();

        for (a, b) in original_forecast
            .primary()
            .iter()
            .zip(restored_forecast.primary().iter())
        {
            assert!(
                (a - b).abs() < 1e-10,
                "ETS forecasts should match after round-trip"
            );
        }
    }

    #[test]
    fn skipped_fields_are_none_after_deserialization() {
        use crate::models::arima::ARIMA;

        let timestamps = make_timestamps(50);
        let values: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = ARIMA::new(1, 1, 0);
        model.fit(&ts).unwrap();

        let json = to_json(&model).unwrap();
        let restored: ARIMA = from_json(&json).unwrap();

        // The exog_ols field is skipped during serialization,
        // so it should be None after deserialization
        assert!(
            !restored.has_exog(),
            "exog_ols should be None after deserialization"
        );
    }

    // ── File I/O round-trip tests ────────────────────────────────────────

    #[test]
    fn file_save_load_round_trip() {
        use crate::models::baseline::Naive;

        let timestamps = make_timestamps(10);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        // Use a temp file
        let dir = std::env::temp_dir();
        let path = dir.join("anofox_test_naive_model.json");

        save_to_file(&model, &path).unwrap();
        let restored: Naive = load_from_file(&path).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&path);

        let original_forecast = model.predict(3).unwrap();
        let restored_forecast = restored.predict(3).unwrap();
        assert_eq!(original_forecast.primary(), restored_forecast.primary());
    }

    // ── Bincode round-trip tests ─────────────────────────────────────────

    #[test]
    fn bincode_round_trip_forecast() {
        // Note: model types using the nan_vec custom serializer are not compatible
        // with bincode (they use JSON-specific null mapping for NaN). Use Forecast
        // and TimeSeries which work correctly with bincode.
        let forecast = Forecast::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let bytes = to_bincode(&forecast).unwrap();
        let restored: Forecast = from_bincode(&bytes).unwrap();
        assert_eq!(forecast, restored);
    }

    #[test]
    fn bincode_file_round_trip_forecast() {
        let forecast = Forecast::from_values_with_intervals(
            vec![10.0, 20.0, 30.0],
            vec![8.0, 18.0, 28.0],
            vec![12.0, 22.0, 32.0],
        );

        let dir = std::env::temp_dir();
        let path = dir.join("anofox_test_forecast.bin");

        save_to_bincode(&forecast, &path).unwrap();
        let restored: Forecast = load_from_bincode(&path).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&path);

        assert_eq!(forecast, restored);
    }

    #[test]
    fn bincode_file_round_trip_time_series() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join("anofox_test_timeseries.bin");

        save_to_bincode(&ts, &path).unwrap();
        let restored: TimeSeries = load_from_bincode(&path).unwrap();

        // Clean up
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.len(), 5);
        assert_eq!(restored.primary_values(), &values);
        assert_eq!(restored.timestamps(), &timestamps);
    }

    // ── Forecast and TimeSeries serialization ────────────────────────────

    #[test]
    fn forecast_json_round_trip() {
        let forecast = Forecast::from_values_with_intervals(
            vec![10.0, 20.0, 30.0],
            vec![8.0, 18.0, 28.0],
            vec![12.0, 22.0, 32.0],
        );

        let json = to_json(&forecast).unwrap();
        let restored: Forecast = from_json(&json).unwrap();
        assert_eq!(forecast, restored);
    }

    #[test]
    fn forecast_bincode_round_trip() {
        let forecast = Forecast::from_values_with_intervals(
            vec![1.5, 2.5, 3.5],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
        );

        let bytes = to_bincode(&forecast).unwrap();
        let restored: Forecast = from_bincode(&bytes).unwrap();
        assert_eq!(forecast, restored);
    }

    #[test]
    fn time_series_json_round_trip() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();
        ts.set_frequency(Duration::hours(1));

        let json = to_json(&ts).unwrap();
        let restored: TimeSeries = from_json(&json).unwrap();

        assert_eq!(restored.len(), 5);
        assert_eq!(restored.primary_values(), &values);
        assert_eq!(restored.timestamps(), &timestamps);
        assert_eq!(restored.frequency(), Some(Duration::hours(1)));
    }

    #[test]
    fn time_series_bincode_round_trip() {
        let timestamps = make_timestamps(5);
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ts = TimeSeries::univariate(timestamps.clone(), values.clone()).unwrap();

        let bytes = to_bincode(&ts).unwrap();
        let restored: TimeSeries = from_bincode(&bytes).unwrap();

        assert_eq!(restored.len(), 5);
        assert_eq!(restored.primary_values(), &values);
        assert_eq!(restored.timestamps(), &timestamps);
    }

    // ── opt_duration_secs helper tests ───────────────────────────────────

    #[test]
    fn opt_duration_secs_some_round_trip() {
        // Test with a struct that uses the opt_duration_secs helper
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct DurationWrapper {
            #[serde(with = "opt_duration_secs")]
            dur: Option<Duration>,
        }

        let wrapper = DurationWrapper {
            dur: Some(Duration::seconds(3600)),
        };

        let json = serde_json::to_string(&wrapper).unwrap();
        let restored: DurationWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.dur, Some(Duration::seconds(3600)));

        // Verify the JSON representation is an integer
        assert!(json.contains("3600"));
    }

    #[test]
    fn opt_duration_secs_none_round_trip() {
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct DurationWrapper {
            #[serde(with = "opt_duration_secs")]
            dur: Option<Duration>,
        }

        let wrapper = DurationWrapper { dur: None };

        let json = serde_json::to_string(&wrapper).unwrap();
        let restored: DurationWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.dur, None);

        // Verify the JSON has null
        assert!(json.contains("null"));
    }

    #[test]
    fn opt_duration_secs_negative_duration() {
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct DurationWrapper {
            #[serde(with = "opt_duration_secs")]
            dur: Option<Duration>,
        }

        let wrapper = DurationWrapper {
            dur: Some(Duration::seconds(-120)),
        };

        let json = serde_json::to_string(&wrapper).unwrap();
        let restored: DurationWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.dur, Some(Duration::seconds(-120)));
    }

    // ── nan_vec helper tests ─────────────────────────────────────────────

    #[test]
    fn nan_vec_round_trip_with_nans() {
        #[derive(serde::Serialize, serde::Deserialize, Debug)]
        struct NanWrapper {
            #[serde(with = "nan_vec")]
            data: Option<Vec<f64>>,
        }

        let wrapper = NanWrapper {
            data: Some(vec![1.0, f64::NAN, 3.0, f64::NAN]),
        };

        let json = serde_json::to_string(&wrapper).unwrap();
        let restored: NanWrapper = serde_json::from_str(&json).unwrap();

        let data = restored.data.unwrap();
        assert_eq!(data.len(), 4);
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!(data[1].is_nan());
        assert!((data[2] - 3.0).abs() < 1e-10);
        assert!(data[3].is_nan());
    }

    #[test]
    fn nan_vec_round_trip_none() {
        #[derive(serde::Serialize, serde::Deserialize, Debug)]
        struct NanWrapper {
            #[serde(with = "nan_vec")]
            data: Option<Vec<f64>>,
        }

        let wrapper = NanWrapper { data: None };

        let json = serde_json::to_string(&wrapper).unwrap();
        let restored: NanWrapper = serde_json::from_str(&json).unwrap();
        assert!(restored.data.is_none());
    }

    #[test]
    fn nan_vec_round_trip_no_nans() {
        #[derive(serde::Serialize, serde::Deserialize, Debug)]
        struct NanWrapper {
            #[serde(with = "nan_vec")]
            data: Option<Vec<f64>>,
        }

        let wrapper = NanWrapper {
            data: Some(vec![1.0, 2.0, 3.0]),
        };

        let json = serde_json::to_string(&wrapper).unwrap();
        let restored: NanWrapper = serde_json::from_str(&json).unwrap();

        let data = restored.data.unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    // ── Error case tests ─────────────────────────────────────────────────

    #[test]
    fn from_json_rejects_invalid_json() {
        let result: crate::error::Result<Forecast> = from_json("not valid json {{{");
        assert!(result.is_err());
        match result.unwrap_err() {
            ForecastError::SerializationError(msg) => {
                assert!(
                    msg.contains("deserialization failed"),
                    "unexpected error message: {}",
                    msg
                );
            }
            other => panic!("expected SerializationError, got {:?}", other),
        }
    }

    #[test]
    fn from_json_rejects_wrong_type() {
        // Valid JSON but wrong structure for Forecast
        let result: crate::error::Result<Forecast> = from_json(r#""just a string""#);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::SerializationError(_)
        ));
    }

    #[test]
    fn from_json_rejects_empty_string() {
        let result: crate::error::Result<Forecast> = from_json("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::SerializationError(_)
        ));
    }

    #[test]
    fn from_bincode_rejects_corrupted_data() {
        let corrupted = vec![0xFF, 0xFE, 0xFD, 0x00, 0x01, 0x02, 0x03];
        let result: crate::error::Result<Forecast> = from_bincode(&corrupted);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForecastError::SerializationError(msg) => {
                assert!(
                    msg.contains("bincode deserialization failed"),
                    "unexpected error message: {}",
                    msg
                );
            }
            other => panic!("expected SerializationError, got {:?}", other),
        }
    }

    #[test]
    fn from_bincode_rejects_empty_data() {
        let result: crate::error::Result<Forecast> = from_bincode(&[]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::SerializationError(_)
        ));
    }

    #[test]
    fn from_bincode_rejects_truncated_data() {
        use crate::models::baseline::Naive;

        let timestamps = make_timestamps(10);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let bytes = to_bincode(&model).unwrap();
        // Truncate the data to simulate corruption
        let truncated = &bytes[..bytes.len() / 2];
        let result: crate::error::Result<Naive> = from_bincode(truncated);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::SerializationError(_)
        ));
    }

    #[test]
    fn load_from_file_rejects_missing_file() {
        let path = std::path::Path::new("/tmp/anofox_nonexistent_file_12345.json");
        let result: crate::error::Result<Forecast> = load_from_file(path);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForecastError::SerializationError(msg) => {
                assert!(
                    msg.contains("file read failed"),
                    "unexpected error message: {}",
                    msg
                );
            }
            other => panic!("expected SerializationError, got {:?}", other),
        }
    }

    #[test]
    fn load_from_bincode_rejects_missing_file() {
        let path = std::path::Path::new("/tmp/anofox_nonexistent_file_12345.bin");
        let result: crate::error::Result<Forecast> = load_from_bincode(path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::SerializationError(_)
        ));
    }

    // ── Bincode is more compact than JSON ────────────────────────────────

    #[test]
    fn bincode_is_more_compact_than_json() {
        use crate::models::baseline::Naive;

        let timestamps = make_timestamps(10);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = Naive::new();
        model.fit(&ts).unwrap();

        let json = to_json(&model).unwrap();
        let bincode_bytes = to_bincode(&model).unwrap();

        assert!(
            bincode_bytes.len() < json.len(),
            "bincode ({} bytes) should be smaller than JSON ({} bytes)",
            bincode_bytes.len(),
            json.len()
        );
    }
}
