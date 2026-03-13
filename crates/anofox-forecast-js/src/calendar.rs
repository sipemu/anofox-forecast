//! Calendar annotations wrapper for JavaScript.
//!
//! Exposes CalendarAnnotations from anofox-forecast for use in JS/WASM,
//! allowing users to define holidays, events, promotions, and named
//! regressors that can be attached to a TimeSeries for models that
//! support exogenous variables.

use anofox_forecast::core::CalendarAnnotations as InnerCalendarAnnotations;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Calendar annotations for holidays and external regressors.
///
/// Use this to define holidays, events, promotions, or any named
/// numeric regressors that can influence a forecast. Attach to a
/// `TimeSeries` via `TimeSeries.setCalendar()` before fitting a
/// model that supports exogenous variables.
///
/// @example
/// ```js
/// const cal = new CalendarAnnotations();
/// cal.addHoliday(new Date("2024-12-25").getTime());
/// cal.addHoliday(new Date("2025-01-01").getTime());
/// cal.addRegressor("promo", new Float64Array([0,0,1,1,0,0,0]));
///
/// const ts = TimeSeries.withTimestamps(values, timestamps);
/// ts.setCalendar(cal);
/// ```
#[wasm_bindgen]
pub struct CalendarAnnotations {
    inner: InnerCalendarAnnotations,
}

#[wasm_bindgen]
impl CalendarAnnotations {
    /// Create an empty CalendarAnnotations instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> CalendarAnnotations {
        CalendarAnnotations {
            inner: InnerCalendarAnnotations::new(),
        }
    }

    /// Add a single holiday date.
    ///
    /// @param timestamp_ms - Holiday date as milliseconds since Unix epoch
    #[wasm_bindgen(js_name = addHoliday)]
    pub fn add_holiday(&mut self, timestamp_ms: f64) {
        let secs = (timestamp_ms / 1000.0) as i64;
        let nanos = ((timestamp_ms % 1000.0) * 1_000_000.0) as u32;
        if let Some(dt) = chrono::DateTime::from_timestamp(secs, nanos) {
            let mut holidays = self.inner.holidays().to_vec();
            holidays.push(dt);
            self.inner = std::mem::take(&mut self.inner).with_holidays(holidays);
        }
    }

    /// Set multiple holiday dates at once (replaces any existing holidays).
    ///
    /// @param timestamps_ms - Array of holiday dates as milliseconds since Unix epoch
    #[wasm_bindgen(js_name = setHolidays)]
    pub fn set_holidays(&mut self, timestamps_ms: &[f64]) {
        let holidays: Vec<chrono::DateTime<chrono::Utc>> = timestamps_ms
            .iter()
            .filter_map(|&ms| {
                let secs = (ms / 1000.0) as i64;
                let nanos = ((ms % 1000.0) * 1_000_000.0) as u32;
                chrono::DateTime::from_timestamp(secs, nanos)
            })
            .collect();
        self.inner = std::mem::take(&mut self.inner).with_holidays(holidays);
    }

    /// Add a named regressor with values aligned to the time series.
    ///
    /// The values array must have the same length as the time series this
    /// will be attached to. Common examples: promotional flags (0/1),
    /// temperature, price changes, etc.
    ///
    /// @param name - Name of the regressor (e.g., "promo", "temperature")
    /// @param values - Array of numeric values, one per time step
    #[wasm_bindgen(js_name = addRegressor)]
    pub fn add_regressor(&mut self, name: &str, values: &[f64]) {
        self.inner =
            std::mem::take(&mut self.inner).with_regressor(name.to_string(), values.to_vec());
    }

    /// Get the number of holidays.
    #[wasm_bindgen(js_name = holidayCount)]
    pub fn holiday_count(&self) -> usize {
        self.inner.holidays().len()
    }

    /// Get the number of named regressors.
    #[wasm_bindgen(js_name = regressorCount)]
    pub fn regressor_count(&self) -> usize {
        self.inner.regressors().len()
    }

    /// Get holiday dates as milliseconds since Unix epoch.
    #[wasm_bindgen(js_name = getHolidays)]
    pub fn get_holidays(&self) -> Vec<f64> {
        self.inner
            .holidays()
            .iter()
            .map(|dt| dt.timestamp_millis() as f64)
            .collect()
    }

    /// Get the names of all registered regressors.
    #[wasm_bindgen(js_name = regressorNames)]
    pub fn regressor_names(&self) -> Result<JsValue, JsError> {
        let names: Vec<String> = self.inner.regressors().keys().cloned().collect();
        serde_wasm_bindgen::to_value(&names).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get values for a named regressor.
    ///
    /// @param name - Regressor name
    /// @returns Array of values, or undefined if the regressor does not exist
    #[wasm_bindgen(js_name = getRegressor)]
    pub fn get_regressor(&self, name: &str) -> Option<Vec<f64>> {
        self.inner.regressor(name).map(|v| v.to_vec())
    }

    /// Check whether any regressors have been added.
    #[wasm_bindgen(js_name = hasRegressors)]
    pub fn has_regressors(&self) -> bool {
        self.inner.has_regressors()
    }

    /// Check if a specific date is a holiday.
    ///
    /// @param timestamp_ms - Date to check as milliseconds since Unix epoch
    /// @returns true if the date matches any registered holiday
    #[wasm_bindgen(js_name = isHoliday)]
    pub fn is_holiday(&self, timestamp_ms: f64) -> bool {
        let secs = (timestamp_ms / 1000.0) as i64;
        let nanos = ((timestamp_ms % 1000.0) * 1_000_000.0) as u32;
        match chrono::DateTime::from_timestamp(secs, nanos) {
            Some(dt) => self.inner.is_holiday(&dt),
            None => false,
        }
    }

    /// Check if a specific date is a business day (not weekend, not holiday).
    ///
    /// @param timestamp_ms - Date to check as milliseconds since Unix epoch
    /// @returns true if the date is a weekday and not a holiday
    #[wasm_bindgen(js_name = isBusinessDay)]
    pub fn is_business_day(&self, timestamp_ms: f64) -> bool {
        let secs = (timestamp_ms / 1000.0) as i64;
        let nanos = ((timestamp_ms % 1000.0) * 1_000_000.0) as u32;
        match chrono::DateTime::from_timestamp(secs, nanos) {
            Some(dt) => self.inner.is_business_day(&dt),
            None => false,
        }
    }

    /// Serialize the calendar annotations to a JSON string.
    ///
    /// Useful for persisting or transferring annotations between contexts.
    ///
    /// @returns JSON string with `{ holidays: number[], regressors: { [name]: number[] } }`
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<String, JsError> {
        let repr = CalendarAnnotationsJson {
            holidays: self
                .inner
                .holidays()
                .iter()
                .map(|dt| dt.timestamp_millis() as f64)
                .collect(),
            regressors: self
                .inner
                .regressors()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        serde_json::to_string(&repr).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Deserialize calendar annotations from a JSON string.
    ///
    /// @param json - JSON string produced by `toJSON()`
    /// @returns A new CalendarAnnotations instance
    #[wasm_bindgen(js_name = fromJSON)]
    pub fn from_json(json: &str) -> Result<CalendarAnnotations, JsError> {
        let repr: CalendarAnnotationsJson =
            serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))?;

        let holidays: Vec<chrono::DateTime<chrono::Utc>> = repr
            .holidays
            .iter()
            .filter_map(|&ms| {
                let secs = (ms / 1000.0) as i64;
                let nanos = ((ms % 1000.0) * 1_000_000.0) as u32;
                chrono::DateTime::from_timestamp(secs, nanos)
            })
            .collect();

        let mut inner = InnerCalendarAnnotations::new().with_holidays(holidays);
        for (name, values) in repr.regressors {
            inner = inner.with_regressor(name, values);
        }

        Ok(CalendarAnnotations { inner })
    }

    /// Get the inner CalendarAnnotations (for internal use).
    pub(crate) fn inner(&self) -> &InnerCalendarAnnotations {
        &self.inner
    }
}

impl Default for CalendarAnnotations {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON-serializable representation of CalendarAnnotations.
#[derive(Serialize, Deserialize)]
struct CalendarAnnotationsJson {
    holidays: Vec<f64>,
    regressors: std::collections::HashMap<String, Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_calendar_annotations_new() {
        let cal = CalendarAnnotations::new();
        assert_eq!(cal.holiday_count(), 0);
        assert_eq!(cal.regressor_count(), 0);
        assert!(!cal.has_regressors());
    }

    #[wasm_bindgen_test]
    fn test_add_holiday() {
        let mut cal = CalendarAnnotations::new();
        // 2024-12-25 00:00:00 UTC in milliseconds
        let christmas = 1735084800000.0;
        cal.add_holiday(christmas);
        assert_eq!(cal.holiday_count(), 1);

        let holidays = cal.get_holidays();
        assert_eq!(holidays.len(), 1);
        assert!((holidays[0] - christmas).abs() < 1.0);
    }

    #[wasm_bindgen_test]
    fn test_set_holidays() {
        let mut cal = CalendarAnnotations::new();
        let holidays_ms = vec![1735084800000.0, 1735689600000.0]; // Dec 25, Jan 1
        cal.set_holidays(&holidays_ms);
        assert_eq!(cal.holiday_count(), 2);
    }

    #[wasm_bindgen_test]
    fn test_add_regressor() {
        let mut cal = CalendarAnnotations::new();
        let promo = vec![0.0, 0.0, 1.0, 1.0, 0.0];
        cal.add_regressor("promo", &promo);

        assert!(cal.has_regressors());
        assert_eq!(cal.regressor_count(), 1);

        let values = cal.get_regressor("promo").unwrap();
        assert_eq!(values, promo);
    }

    #[wasm_bindgen_test]
    fn test_multiple_regressors() {
        let mut cal = CalendarAnnotations::new();
        cal.add_regressor("promo", &[0.0, 1.0, 0.0]);
        cal.add_regressor("temperature", &[20.0, 22.0, 25.0]);

        assert_eq!(cal.regressor_count(), 2);
        assert!(cal.get_regressor("promo").is_some());
        assert!(cal.get_regressor("temperature").is_some());
        assert!(cal.get_regressor("nonexistent").is_none());
    }

    #[wasm_bindgen_test]
    fn test_is_holiday() {
        let mut cal = CalendarAnnotations::new();
        let christmas = 1735084800000.0; // 2024-12-25 00:00:00 UTC
        cal.add_holiday(christmas);

        // Same day at a different time should still be a holiday
        assert!(cal.is_holiday(christmas));
        assert!(cal.is_holiday(christmas + 3600000.0)); // +1 hour

        // Different day should not be a holiday
        assert!(!cal.is_holiday(christmas + 86400000.0)); // +1 day
    }

    #[wasm_bindgen_test]
    fn test_is_business_day() {
        let mut cal = CalendarAnnotations::new();
        let christmas = 1735084800000.0; // 2024-12-25 is a Wednesday
        cal.add_holiday(christmas);

        // Holiday (Wednesday) -> not a business day
        assert!(!cal.is_business_day(christmas));

        // Saturday (Dec 28) -> not a business day
        let saturday = christmas + 3.0 * 86400000.0;
        assert!(!cal.is_business_day(saturday));

        // Thursday Dec 26 (not a holiday, not weekend) -> business day
        let thursday = christmas + 86400000.0;
        assert!(cal.is_business_day(thursday));
    }

    #[wasm_bindgen_test]
    fn test_json_round_trip() {
        let mut cal = CalendarAnnotations::new();
        cal.add_holiday(1735084800000.0);
        cal.add_regressor("promo", &[0.0, 1.0, 0.0]);

        let json = cal.to_json().unwrap();
        let restored = CalendarAnnotations::from_json(&json).unwrap();

        assert_eq!(restored.holiday_count(), 1);
        assert_eq!(restored.regressor_count(), 1);
        assert_eq!(
            restored.get_regressor("promo").unwrap(),
            vec![0.0, 1.0, 0.0]
        );
    }

    #[wasm_bindgen_test]
    fn test_from_json_invalid() {
        let result = CalendarAnnotations::from_json("not valid json");
        assert!(result.is_err());
    }
}
