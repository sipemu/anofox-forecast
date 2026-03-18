//! Composable transform pipeline that chains reversible transforms around a
//! [`Forecaster`], and itself implements [`Forecaster`].
//!
//! # Example
//!
//! ```
//! use anofox_forecast::transform::pipeline::Pipeline;
//! use anofox_forecast::transform::transforms::{BoxCoxTransform, DifferenceTransform};
//! use anofox_forecast::models::baseline::Naive;
//!
//! let pipeline = Pipeline::builder()
//!     .transform(BoxCoxTransform::auto())
//!     .transform(DifferenceTransform::new(1))
//!     .model(Box::new(Naive::new()))
//!     .build();
//! ```

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::{BoxedForecaster, FittedParams, Forecaster};
use std::collections::HashMap;
use std::fmt;

// ── Transform trait ─────────────────────────────────────────────────────────

/// Whether inverse should anchor from the start or end of the training data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseMode {
    /// Anchor from start of training data (for reconstructing fitted values).
    Fitted,
    /// Anchor from end of training data (for forecasting into the future).
    Predict,
}

/// A reversible data transformation that can be chained in a [`Pipeline`].
pub trait Transform: fmt::Debug + Send + Sync {
    /// Learn parameters from `values` and return the transformed data.
    fn fit_transform(&mut self, values: &[f64]) -> Result<Vec<f64>>;

    /// Reverse the transform.
    ///
    /// `mode` selects the anchor point for length-changing transforms like
    /// differencing (`Fitted` uses the initial anchor, `Predict` uses the
    /// final anchor).
    fn inverse(&self, values: &[f64], mode: InverseMode) -> Result<Vec<f64>>;

    /// Number of observations consumed from the front (0 for length-preserving
    /// transforms, `d` for `Difference(d)`, `period` for seasonal differencing).
    fn offset(&self) -> usize;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Transform>;
}

impl Clone for Box<dyn Transform> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/// A chain of reversible transforms around a [`Forecaster`].
///
/// `Pipeline` itself implements [`Forecaster`], so it can be used anywhere a
/// model is expected (cross-validation, ensembles, batch operations).
pub struct Pipeline {
    transforms: Vec<Box<dyn Transform>>,
    model: BoxedForecaster,
    original_len: usize,
    total_offset: usize,
    fitted: Option<Vec<f64>>,
    residuals_cache: Option<Vec<f64>>,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("transforms", &self.transforms)
            .field("model", &self.model.name())
            .field("total_offset", &self.total_offset)
            .finish()
    }
}

impl Pipeline {
    /// Start building a pipeline.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder {
            transforms: Vec::new(),
            model: None,
        }
    }
}

// ── PipelineBuilder ─────────────────────────────────────────────────────────

/// Builder for constructing a [`Pipeline`].
pub struct PipelineBuilder {
    transforms: Vec<Box<dyn Transform>>,
    model: Option<BoxedForecaster>,
}

impl PipelineBuilder {
    /// Append a transform to the chain (applied in order during `fit`).
    pub fn transform(mut self, t: impl Transform + 'static) -> Self {
        self.transforms.push(Box::new(t));
        self
    }

    /// Set the inner forecasting model.
    pub fn model(mut self, model: BoxedForecaster) -> Self {
        self.model = Some(model);
        self
    }

    /// Consume the builder and produce a [`Pipeline`].
    ///
    /// # Panics
    ///
    /// Panics if no model was provided.
    pub fn build(self) -> Pipeline {
        Pipeline {
            transforms: self.transforms,
            model: self.model.expect("Pipeline requires a model"),
            original_len: 0,
            total_offset: 0,
            fitted: None,
            residuals_cache: None,
        }
    }
}

// ── Forecaster impl ─────────────────────────────────────────────────────────

impl Forecaster for Pipeline {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let original = series.primary_values();
        let n = original.len();
        if n == 0 {
            return Err(ForecastError::EmptyData);
        }

        self.original_len = n;
        self.total_offset = 0;
        self.fitted = None;
        self.residuals_cache = None;

        // ── Forward pass: apply transforms in order ─────────────────────
        let mut current = original.to_vec();
        for t in &mut self.transforms {
            current = t.fit_transform(&current)?;
            self.total_offset += t.offset();
        }

        // ── Build inner TimeSeries with transformed values ──────────────
        let inner_ts = series.slice(self.total_offset, n)?;
        // Replace primary values with the transformed ones.
        let inner_ts = TimeSeries::univariate(inner_ts.timestamps().to_vec(), current.clone())?;

        // ── Fit the inner model ─────────────────────────────────────────
        self.model.fit(&inner_ts)?;

        // ── Compute fitted values in original space ─────────────────────
        if let Some(inner_fitted) = self.model.fitted_values() {
            let mut inv = inner_fitted.to_vec();
            // Inverse transforms in reverse order, using Fitted mode.
            for t in self.transforms.iter().rev() {
                inv = t.inverse(&inv, InverseMode::Fitted)?;
            }

            // The inverse of offset-producing transforms may return more
            // values than the original series (the anchors are prepended).
            // Align to original length: take the last `n` values.
            let fitted_full = if inv.len() >= n {
                inv[inv.len() - n..].to_vec()
            } else {
                // Pad front with NaN if shorter (shouldn't normally happen).
                let mut padded = vec![f64::NAN; n - inv.len()];
                padded.extend_from_slice(&inv);
                padded
            };

            // Residuals: original - fitted, NaN where fitted is NaN.
            let residuals: Vec<f64> = original
                .iter()
                .zip(fitted_full.iter())
                .map(|(&o, &f)| if f.is_nan() { f64::NAN } else { o - f })
                .collect();

            self.fitted = Some(fitted_full);
            self.residuals_cache = Some(residuals);
        }

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let forecast = self.model.predict(horizon)?;
        self.inverse_forecast(forecast)
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let forecast = self.model.predict_with_intervals(horizon, level)?;
        self.inverse_forecast(forecast)
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals_cache.as_deref()
    }

    fn name(&self) -> &str {
        "Pipeline"
    }

    fn fitted_params(&self) -> Option<FittedParams> {
        self.model.fitted_params()
    }

    fn supports_exog(&self) -> bool {
        self.model.supports_exog()
    }

    fn has_exog(&self) -> bool {
        self.model.has_exog()
    }

    fn exog_names(&self) -> Option<&[String]> {
        self.model.exog_names()
    }

    fn predict_with_exog(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
    ) -> Result<Forecast> {
        let forecast = self.model.predict_with_exog(horizon, future_regressors)?;
        self.inverse_forecast(forecast)
    }

    fn predict_with_exog_intervals(
        &self,
        horizon: usize,
        future_regressors: &HashMap<String, Vec<f64>>,
        level: f64,
    ) -> Result<Forecast> {
        let forecast = self
            .model
            .predict_with_exog_intervals(horizon, future_regressors, level)?;
        self.inverse_forecast(forecast)
    }
}

impl Pipeline {
    /// Apply inverse transforms to all components of a forecast (point, lower, upper).
    fn inverse_forecast(&self, forecast: Forecast) -> Result<Forecast> {
        let point = self.inverse_series(forecast.primary())?;

        if forecast.has_lower() && forecast.has_upper() {
            let lower = self.inverse_series(forecast.lower_series(0)?)?;
            let upper = self.inverse_series(forecast.upper_series(0)?)?;
            Ok(Forecast::from_values_with_intervals(point, lower, upper))
        } else {
            Ok(Forecast::from_values(point))
        }
    }

    /// Apply inverse transforms in reverse order using Predict mode.
    fn inverse_series(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mut current = values.to_vec();
        for t in self.transforms.iter().rev() {
            current = t.inverse(&current, InverseMode::Predict)?;
        }
        Ok(current)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::baseline::Naive;
    use crate::transform::transforms::{
        BoxCoxTransform, DifferenceTransform, LogTransform, ScaleMethod, ScaleTransform,
    };
    use chrono::{TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }

    fn make_ts(values: Vec<f64>) -> TimeSeries {
        let n = values.len();
        TimeSeries::univariate(make_timestamps(n), values).unwrap()
    }

    // ── Basic pipeline ──────────────────────────────────────────────────

    #[test]
    fn pipeline_with_no_transforms() {
        let values: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let ts = make_ts(values);

        let mut pipeline = Pipeline::builder().model(Box::new(Naive::new())).build();

        pipeline.fit(&ts).unwrap();
        assert!(pipeline.is_fitted());

        let forecast = pipeline.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        // Naive repeats last value
        for &v in forecast.primary() {
            assert!((v - 30.0).abs() < 1e-10);
        }
    }

    #[test]
    fn pipeline_difference_naive() {
        // Linear data: [1, 2, 3, ..., 30]
        // Differenced: constant 1.0
        // Naive on differenced: predicts 1.0
        // Inverse: cumsum from 30 → [31, 32, 33, ...]
        let values: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let ts = make_ts(values);

        let mut pipeline = Pipeline::builder()
            .transform(DifferenceTransform::new(1))
            .model(Box::new(Naive::new()))
            .build();

        pipeline.fit(&ts).unwrap();
        let forecast = pipeline.predict(5).unwrap();

        let expected: Vec<f64> = (31..=35).map(|i| i as f64).collect();
        for (a, b) in forecast.primary().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "expected {}, got {}", b, a);
        }
    }

    #[test]
    fn pipeline_boxcox_difference_naive() {
        // Exponential-ish positive data
        let values: Vec<f64> = (1..=30).map(|i| (i as f64).powi(2)).collect();
        let ts = make_ts(values.clone());

        let mut pipeline = Pipeline::builder()
            .transform(BoxCoxTransform::auto())
            .transform(DifferenceTransform::new(1))
            .model(Box::new(Naive::new()))
            .build();

        pipeline.fit(&ts).unwrap();
        assert!(pipeline.is_fitted());

        let forecast = pipeline.predict(3).unwrap();
        assert_eq!(forecast.horizon(), 3);

        // Forecast should be in the original positive scale, beyond the last value (900).
        for &v in forecast.primary() {
            assert!(v > 800.0, "forecast {} should be > 800", v);
        }
    }

    #[test]
    fn pipeline_fitted_values_same_length_as_input() {
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ts = make_ts(values.clone());

        let mut pipeline = Pipeline::builder()
            .transform(DifferenceTransform::new(1))
            .model(Box::new(Naive::new()))
            .build();

        pipeline.fit(&ts).unwrap();

        let fitted = pipeline.fitted_values().unwrap();
        assert_eq!(fitted.len(), values.len());

        let residuals = pipeline.residuals().unwrap();
        assert_eq!(residuals.len(), values.len());
    }

    #[test]
    fn pipeline_with_intervals() {
        let values: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let ts = make_ts(values);

        let mut pipeline = Pipeline::builder()
            .transform(DifferenceTransform::new(1))
            .model(Box::new(Naive::new()))
            .build();

        pipeline.fit(&ts).unwrap();
        let forecast = pipeline.predict_with_intervals(5, 0.95).unwrap();
        assert_eq!(forecast.horizon(), 5);
        // Naive supports intervals
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());
    }

    #[test]
    fn pipeline_name() {
        let pipeline = Pipeline::builder().model(Box::new(Naive::new())).build();
        assert_eq!(pipeline.name(), "Pipeline");
    }

    // ── Scale transform in pipeline ─────────────────────────────────────

    #[test]
    fn pipeline_scale_naive() {
        let values: Vec<f64> = (1..=20).map(|i| i as f64 * 10.0).collect();
        let ts = make_ts(values.clone());

        let mut pipeline = Pipeline::builder()
            .transform(ScaleTransform::new(ScaleMethod::Standardize))
            .model(Box::new(Naive::new()))
            .build();

        pipeline.fit(&ts).unwrap();
        let forecast = pipeline.predict(3).unwrap();

        // Naive on standardized data predicts last standardized value,
        // which inverse-transforms back to 200.0.
        for &v in forecast.primary() {
            assert!((v - 200.0).abs() < 1e-8, "expected ~200, got {}", v);
        }
    }

    // ── Log transform in pipeline ───────────────────────────────────────

    #[test]
    fn pipeline_log_naive() {
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ts = make_ts(values);

        let mut pipeline = Pipeline::builder()
            .transform(LogTransform::new())
            .model(Box::new(Naive::new()))
            .build();

        pipeline.fit(&ts).unwrap();
        let forecast = pipeline.predict(3).unwrap();

        // Naive predicts last value in log space → exp(ln(20)) = 20
        for &v in forecast.primary() {
            assert!((v - 20.0).abs() < 1e-8, "expected ~20, got {}", v);
        }
    }

    // ── Fit-predict convenience ─────────────────────────────────────────

    #[test]
    fn pipeline_fit_predict() {
        let values: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let ts = make_ts(values);

        let mut pipeline = Pipeline::builder()
            .transform(DifferenceTransform::new(1))
            .model(Box::new(Naive::new()))
            .build();

        let forecast = pipeline.fit_predict(&ts, 5).unwrap();
        assert_eq!(forecast.horizon(), 5);
        assert!(pipeline.is_fitted());
    }

    // ── Error cases ─────────────────────────────────────────────────────

    #[test]
    fn pipeline_predict_before_fit() {
        let pipeline = Pipeline::builder()
            .transform(DifferenceTransform::new(1))
            .model(Box::new(Naive::new()))
            .build();

        assert!(!pipeline.is_fitted());
        assert!(pipeline.predict(5).is_err());
    }

    #[test]
    fn pipeline_insufficient_data_for_transform() {
        let values = vec![1.0, 2.0];
        let ts = make_ts(values);

        let mut pipeline = Pipeline::builder()
            .transform(DifferenceTransform::new(2))
            .model(Box::new(Naive::new()))
            .build();

        assert!(pipeline.fit(&ts).is_err());
    }

    // ── Integration: Pipeline used with ModelSpec ────────────────────────

    #[test]
    fn pipeline_as_model_spec() {
        use crate::models::ModelSpec;

        let spec = ModelSpec::new(
            "Pipeline(Diff→Naive)",
            || {
                Box::new(
                    Pipeline::builder()
                        .transform(DifferenceTransform::new(1))
                        .model(Box::new(Naive::new()))
                        .build(),
                )
            },
            true,
        );

        let mut model = spec.create();
        let values: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let ts = make_ts(values);

        model.fit(&ts).unwrap();
        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }
}
