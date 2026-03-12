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

/// Save a serializable model to a JSON string.
#[cfg(feature = "serde")]
pub fn to_json<T: Serialize>(model: &T) -> crate::error::Result<String> {
    serde_json::to_string_pretty(model).map_err(|e| {
        crate::error::ForecastError::ComputationError(format!("serialization failed: {}", e))
    })
}

/// Load a model from a JSON string.
#[cfg(feature = "serde")]
pub fn from_json<T: DeserializeOwned>(json: &str) -> crate::error::Result<T> {
    serde_json::from_str(json).map_err(|e| {
        crate::error::ForecastError::ComputationError(format!("deserialization failed: {}", e))
    })
}

/// Save a model to a file.
#[cfg(feature = "serde")]
pub fn save_to_file<T: Serialize>(model: &T, path: &std::path::Path) -> crate::error::Result<()> {
    let json = to_json(model)?;
    std::fs::write(path, json).map_err(|e| {
        crate::error::ForecastError::ComputationError(format!("file write failed: {}", e))
    })
}

/// Load a model from a file.
#[cfg(feature = "serde")]
pub fn load_from_file<T: DeserializeOwned>(path: &std::path::Path) -> crate::error::Result<T> {
    let json = std::fs::read_to_string(path).map_err(|e| {
        crate::error::ForecastError::ComputationError(format!("file read failed: {}", e))
    })?;
    from_json(&json)
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::models::Forecaster;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

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
            assert!((a - b).abs() < 1e-10, "ARIMA forecasts should match after round-trip");
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
            assert!((a - b).abs() < 1e-10, "ETS forecasts should match after round-trip");
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
        assert!(!restored.has_exog(), "exog_ols should be None after deserialization");
    }

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
}
