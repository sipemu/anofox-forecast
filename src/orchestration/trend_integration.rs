//! Trend integration for the forecasting pipeline.
//!
//! Provides two modes for incorporating a fitted trend component into
//! the forecasting workflow:
//!
//! - **Decompose**: subtract trend before fitting, add it back after prediction.
//!   Works with all models.
//! - **Regressor**: pass trend values as an exogenous feature named `"__trend"`.
//!   Requires models that support exogenous regressors.

use std::collections::HashMap;

use crate::core::{CalendarAnnotations, Forecast, TimeSeries};
use crate::error::Result;
use crate::seasonality::traits::TrendComponent;

/// The regressor name used when trend is passed as an exogenous feature.
pub const TREND_REGRESSOR_NAME: &str = "__trend";

/// Holds a fitted trend component and provides helpers for both integration modes.
#[derive(Debug)]
pub struct TrendIntegrationState {
    /// Fitted trend values (same length as training data).
    fitted_trend: Vec<f64>,
    /// Predicted future trend values (length = forecast horizon).
    future_trend: Vec<f64>,
}

impl TrendIntegrationState {
    /// Create a new state from a fitted trend component.
    ///
    /// Fits the component to the data and pre-computes the trend forecast.
    pub fn from_component(
        component: &mut dyn TrendComponent,
        values: &[f64],
        horizon: usize,
    ) -> Result<Self> {
        component.fit_trend(values)?;
        let fitted_trend = component.fitted_trend().to_vec();
        let future_trend = component.predict_trend(horizon);
        Ok(Self {
            fitted_trend,
            future_trend,
        })
    }

    /// Create a state from pre-computed trend values.
    pub fn from_values(fitted_trend: Vec<f64>, future_trend: Vec<f64>) -> Self {
        Self {
            fitted_trend,
            future_trend,
        }
    }

    /// Get the fitted trend values.
    pub fn fitted_trend(&self) -> &[f64] {
        &self.fitted_trend
    }

    /// Get the predicted future trend values.
    pub fn future_trend(&self) -> &[f64] {
        &self.future_trend
    }

    // ── Decompose mode helpers ───────────────────────────────────────

    /// Subtract the fitted trend from a TimeSeries, producing a detrended series.
    ///
    /// The resulting series has `values[i] - fitted_trend[i]` at each index.
    /// If the series is longer or shorter than the trend, slicing is applied
    /// to the overlapping portion.
    pub fn detrend_series(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let values = ts.values(0)?;
        let n = values.len().min(self.fitted_trend.len());
        let detrended: Vec<f64> = values[..n]
            .iter()
            .zip(self.fitted_trend[..n].iter())
            .map(|(v, t)| v - t)
            .collect();

        let mut builder = crate::core::TimeSeriesBuilder::new();
        builder = builder.timestamps(ts.timestamps()[..n].to_vec());
        builder = builder.values(detrended);

        // Preserve regressors
        if ts.has_regressors() {
            let mut cal = CalendarAnnotations::new();
            for (name, vals) in &ts.all_regressors() {
                cal = cal.with_regressor(name.clone(), vals[..n].to_vec());
            }
            builder = builder.calendar(cal);
        }

        builder.build()
    }

    /// Add the predicted trend back to a forecast (recompose).
    ///
    /// Modifies the forecast in-place: `point[i] += future_trend[i]`.
    /// Also adjusts lower/upper bounds if present.
    pub fn recompose_forecast(&self, forecast: &mut Forecast) {
        let horizon = forecast.primary().len().min(self.future_trend.len());

        for i in 0..horizon {
            forecast.primary_mut()[i] += self.future_trend[i];
        }

        // Adjust intervals if present
        if forecast.has_lower() {
            for i in 0..horizon {
                forecast.lower_series_mut(0)[i] += self.future_trend[i];
            }
        }
        if forecast.has_upper() {
            for i in 0..horizon {
                forecast.upper_series_mut(0)[i] += self.future_trend[i];
            }
        }
    }

    // ── Regressor mode helpers ───────────────────────────────────────

    /// Add the fitted trend as an exogenous regressor to a TimeSeries.
    ///
    /// The regressor is named `"__trend"` and has length equal to the series.
    pub fn add_trend_regressor(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let n = ts.len().min(self.fitted_trend.len());
        let trend_values = self.fitted_trend[..n].to_vec();

        let mut cal = ts
            .calendar()
            .cloned()
            .unwrap_or_else(CalendarAnnotations::new);
        cal = cal.with_regressor(TREND_REGRESSOR_NAME.to_string(), trend_values);

        let mut new_ts = ts.clone();
        new_ts.set_calendar(cal);
        Ok(new_ts)
    }

    /// Build the future regressors HashMap for `predict_with_exog()`.
    ///
    /// Returns a HashMap with `"__trend" → future_trend_values`.
    /// If additional future regressors are provided, they are merged in.
    pub fn future_regressors(
        &self,
        horizon: usize,
        existing: Option<&HashMap<String, Vec<f64>>>,
    ) -> HashMap<String, Vec<f64>> {
        let mut regs = existing.cloned().unwrap_or_default();
        let h = horizon.min(self.future_trend.len());
        regs.insert(
            TREND_REGRESSOR_NAME.to_string(),
            self.future_trend[..h].to_vec(),
        );
        regs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::seasonality::polynomial::PolynomialTrend;
    use crate::seasonality::traits::Recency;
    use approx::assert_relative_eq;
    use chrono::{TimeZone, Utc};

    fn make_ts(values: Vec<f64>) -> TimeSeries {
        let timestamps: Vec<_> = (0..values.len())
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 1, 1, i as u32 % 24, 0, 0)
                    .unwrap()
            })
            .collect();
        TimeSeries::univariate(timestamps, values).unwrap()
    }

    #[test]
    fn decompose_detrend_and_recompose() {
        // Linear trend: y = 2*t + 10
        let values: Vec<f64> = (0..20).map(|t| 2.0 * t as f64 + 10.0).collect();
        let ts = make_ts(values.clone());

        let mut poly = PolynomialTrend::new(1).with_recency(Recency::Full);
        let state = TrendIntegrationState::from_component(&mut poly, &values, 5).unwrap();

        // Detrend: residuals should be ~0
        let detrended_ts = state.detrend_series(&ts).unwrap();
        let detrended = detrended_ts.values(0).unwrap();
        for &v in detrended {
            assert!(v.abs() < 1e-8, "detrended value should be ~0, got {}", v);
        }

        // Recompose: add trend back to zero residuals
        let mut forecast = Forecast::from_values(vec![0.0; 5]);
        state.recompose_forecast(&mut forecast);
        for (i, &v) in forecast.primary().iter().enumerate() {
            let expected = 2.0 * (20 + i) as f64 + 10.0;
            assert_relative_eq!(v, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn regressor_adds_trend_to_series() {
        let values: Vec<f64> = (0..20).map(|t| 2.0 * t as f64 + 10.0).collect();
        let ts = make_ts(values.clone());

        let mut poly = PolynomialTrend::new(1).with_recency(Recency::Full);
        let state = TrendIntegrationState::from_component(&mut poly, &values, 5).unwrap();

        let ts_with_trend = state.add_trend_regressor(&ts).unwrap();
        assert!(ts_with_trend.has_regressors());
        let trend_reg = ts_with_trend.regressor(TREND_REGRESSOR_NAME).unwrap();
        assert_eq!(trend_reg.len(), 20);
        // Trend regressor should match fitted trend
        for (i, &v) in trend_reg.iter().enumerate() {
            assert_relative_eq!(v, state.fitted_trend()[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn future_regressors_builds_hashmap() {
        let values: Vec<f64> = (0..20).map(|t| 2.0 * t as f64 + 10.0).collect();

        let mut poly = PolynomialTrend::new(1).with_recency(Recency::Full);
        let state = TrendIntegrationState::from_component(&mut poly, &values, 10).unwrap();

        let future = state.future_regressors(10, None);
        assert_eq!(future.len(), 1);
        assert!(future.contains_key(TREND_REGRESSOR_NAME));
        assert_eq!(future[TREND_REGRESSOR_NAME].len(), 10);
    }

    #[test]
    fn future_regressors_merges_with_existing() {
        let values: Vec<f64> = (0..20).map(|t| t as f64).collect();

        let mut poly = PolynomialTrend::new(1).with_recency(Recency::Full);
        let state = TrendIntegrationState::from_component(&mut poly, &values, 5).unwrap();

        let mut existing = HashMap::new();
        existing.insert("other".to_string(), vec![1.0; 5]);

        let future = state.future_regressors(5, Some(&existing));
        assert_eq!(future.len(), 2);
        assert!(future.contains_key(TREND_REGRESSOR_NAME));
        assert!(future.contains_key("other"));
    }

    #[test]
    fn recompose_adjusts_intervals() {
        let future_trend = vec![10.0, 20.0, 30.0];
        let state = TrendIntegrationState::from_values(vec![], future_trend);

        let mut forecast =
            Forecast::from_values_with_intervals(vec![0.0; 3], vec![-1.0; 3], vec![1.0; 3]);
        state.recompose_forecast(&mut forecast);

        assert_relative_eq!(forecast.primary()[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(forecast.primary()[2], 30.0, epsilon = 1e-10);

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        assert_relative_eq!(lower[0], 9.0, epsilon = 1e-10);
        assert_relative_eq!(upper[2], 31.0, epsilon = 1e-10);
    }
}
