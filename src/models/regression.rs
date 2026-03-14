//! Regression-based forecasting models.
//!
//! Bridges external regression estimators (e.g., OLS from `anofox-regression`)
//! into the [`Forecaster`] trait, enabling them to participate in pipelines,
//! model registries, ensembles, and cross-validation.
//!
//! # Feature engineering
//!
//! Time-series forecasting with regression requires features. The
//! [`RegressionFeatures`] builder configures which features are constructed
//! from a [`TimeSeries`] before fitting:
//!
//! | Feature          | Description |
//! |-----------------|-------------|
//! | Trend index      | Linear index `0, 1, …, n-1` |
//! | Lags             | `y[t-1], y[t-2], …, y[t-max_lag]` |
//! | Exogenous regressors | From `TimeSeries::all_regressors()` |
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_forecast::models::regression::{RegressionForecaster, RegressionFeatures};
//!
//! // OLS with trend + 3 lags + exogenous regressors
//! let mut model = RegressionForecaster::ols(
//!     RegressionFeatures::new().trend().lags(3),
//! );
//! model.fit(&ts)?;
//! let forecast = model.predict(12)?;
//! ```

#[cfg(feature = "postprocess")]
mod ols_impl {
    use std::collections::HashMap;

    use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
    use faer::{Col, Mat};

    use crate::core::{Forecast, TimeSeries};
    use crate::error::{ForecastError, Result};
    use crate::models::{validate_series_complete, Forecaster};

    // ── Feature specification ───────────────────────────────────────

    /// Configures which features are built from a [`TimeSeries`] for
    /// the regression model.
    #[derive(Debug, Clone)]
    pub struct RegressionFeatures {
        /// Include a linear trend index (0, 1, …, n-1).
        pub use_trend: bool,
        /// Number of autoregressive lags to include.
        pub max_lag: usize,
        /// Include exogenous regressors from the TimeSeries (if present).
        pub use_exog: bool,
    }

    impl Default for RegressionFeatures {
        fn default() -> Self {
            Self {
                use_trend: true,
                max_lag: 0,
                use_exog: true,
            }
        }
    }

    impl RegressionFeatures {
        /// Create a new feature configuration (trend only by default).
        pub fn new() -> Self {
            Self::default()
        }

        /// Include a linear trend index.
        pub fn trend(mut self) -> Self {
            self.use_trend = true;
            self
        }

        /// Do not include a trend index.
        pub fn no_trend(mut self) -> Self {
            self.use_trend = false;
            self
        }

        /// Include autoregressive lags `y[t-1] … y[t-max_lag]`.
        pub fn lags(mut self, max_lag: usize) -> Self {
            self.max_lag = max_lag;
            self
        }

        /// Include exogenous regressors from the TimeSeries.
        pub fn exog(mut self) -> Self {
            self.use_exog = true;
            self
        }

        /// Do not include exogenous regressors.
        pub fn no_exog(mut self) -> Self {
            self.use_exog = false;
            self
        }

        /// Number of observations lost to lagging.
        fn lag_offset(&self) -> usize {
            self.max_lag
        }

        /// Build feature column names for a given TimeSeries.
        fn feature_names(&self, exog_names: &[String]) -> Vec<String> {
            let mut names = Vec::new();
            if self.use_trend {
                names.push("__trend".to_string());
            }
            for lag in 1..=self.max_lag {
                names.push(format!("__lag_{}", lag));
            }
            if self.use_exog {
                for name in exog_names {
                    names.push(name.clone());
                }
            }
            names
        }

        /// Build the design matrix and target vector from a TimeSeries.
        ///
        /// Returns `(X, y, n_train, exog_names)` where `n_train` is the number
        /// of usable rows (= n - max_lag).
        fn build_matrices(
            &self,
            series: &TimeSeries,
        ) -> Result<(Mat<f64>, Col<f64>, usize, Vec<String>)> {
            let values = series.primary_values();
            let n = values.len();
            let offset = self.lag_offset();

            if n <= offset {
                return Err(ForecastError::InsufficientData {
                    needed: offset + 2,
                    got: n,
                    hint: Some(format!(
                        "need > {} observations for {} lags",
                        offset, self.max_lag
                    )),
                });
            }

            let n_train = n - offset;

            // Collect exogenous regressor names (sorted for determinism)
            let exog_names = if self.use_exog && series.has_regressors() {
                let mut names: Vec<String> = series.all_regressors().keys().cloned().collect();
                names.sort();
                names
            } else {
                Vec::new()
            };

            let feature_names = self.feature_names(&exog_names);
            let n_features = feature_names.len();

            if n_features == 0 {
                return Err(ForecastError::InvalidParameter(
                    "No features configured — enable at least one of: trend, lags, or exog"
                        .to_string(),
                ));
            }

            // Build design matrix
            let mut x = Mat::zeros(n_train, n_features);
            let mut y = Col::zeros(n_train);

            // Populate target
            for i in 0..n_train {
                y[i] = values[offset + i];
            }

            // Populate features
            let mut col_idx = 0;

            // Trend: index of the observation (relative to full series)
            if self.use_trend {
                for i in 0..n_train {
                    x[(i, col_idx)] = (offset + i) as f64;
                }
                col_idx += 1;
            }

            // Lags: y[t-1], y[t-2], …
            for lag in 1..=self.max_lag {
                for i in 0..n_train {
                    x[(i, col_idx)] = values[offset + i - lag];
                }
                col_idx += 1;
            }

            // Exogenous regressors (sliced to match after lag offset)
            if self.use_exog {
                let regressors = series.all_regressors();
                for name in &exog_names {
                    if let Some(reg_values) = regressors.get(name) {
                        for i in 0..n_train {
                            let idx = offset + i;
                            if idx < reg_values.len() {
                                x[(i, col_idx)] = reg_values[idx];
                            }
                        }
                    }
                    col_idx += 1;
                }
            }

            Ok((x, y, n_train, exog_names))
        }

        /// Build a design matrix for the forecast horizon.
        ///
        /// For lags: uses the last values from training + predicted values
        /// for multi-step recursive forecasting.
        fn build_future_matrix(
            &self,
            horizon: usize,
            n_total: usize,
            tail_values: &[f64],
            future_regressors: Option<&HashMap<String, Vec<f64>>>,
            exog_names: &[String],
        ) -> Result<Mat<f64>> {
            let feature_names = self.feature_names(exog_names);
            let n_features = feature_names.len();
            let mut x = Mat::zeros(horizon, n_features);

            let mut col_idx = 0;

            // Trend: continue the index
            if self.use_trend {
                for h in 0..horizon {
                    x[(h, col_idx)] = (n_total + h) as f64;
                }
                col_idx += 1;
            }

            // Lags: filled during recursive prediction (column indices stored)
            // Pre-fill from tail_values where possible
            for lag in 1..=self.max_lag {
                for h in 0..horizon {
                    if h >= lag {
                        // Will be filled recursively during prediction
                        x[(h, col_idx)] = f64::NAN; // placeholder
                    } else {
                        // Use known historical values
                        let idx = tail_values.len() as isize - lag as isize + h as isize;
                        if idx >= 0 {
                            x[(h, col_idx)] = tail_values[idx as usize];
                        }
                    }
                }
                col_idx += 1;
            }

            // Exogenous regressors
            if self.use_exog {
                for name in exog_names {
                    if let Some(regs) = future_regressors {
                        if let Some(vals) = regs.get(name) {
                            for h in 0..horizon.min(vals.len()) {
                                x[(h, col_idx)] = vals[h];
                            }
                        }
                    }
                    col_idx += 1;
                }
            }

            Ok(x)
        }
    }

    // ── Fitted state ────────────────────────────────────────────────

    /// Internal state stored after fitting.
    #[derive(Debug)]
    struct FittedState {
        /// The fitted OLS model from anofox-regression.
        model: anofox_regression::solvers::FittedOls,
        /// Feature configuration used.
        features: RegressionFeatures,
        /// Number of observations in the full series.
        n_total: usize,
        /// Last `max_lag` values for recursive prediction.
        tail_values: Vec<f64>,
        /// In-sample fitted values (full length, NaN-padded for lags).
        fitted_values: Vec<f64>,
        /// In-sample residuals (full length, NaN-padded for lags).
        residuals: Vec<f64>,
        /// Exogenous regressor names (sorted).
        exog_names: Vec<String>,
    }

    // ── RegressionForecaster ────────────────────────────────────────

    /// A forecasting model backed by an external regression estimator.
    ///
    /// Wraps `OlsRegressor` from `anofox-regression` behind the [`Forecaster`]
    /// trait, enabling it to participate in pipelines, registries, ensembles,
    /// and cross-validation just like any built-in model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anofox_forecast::models::regression::{RegressionForecaster, RegressionFeatures};
    ///
    /// let mut model = RegressionForecaster::ols(
    ///     RegressionFeatures::new().trend().lags(3),
    /// );
    /// model.fit(&ts)?;
    /// let forecast = model.predict(12)?;
    /// ```
    #[derive(Debug)]
    pub struct RegressionForecaster {
        features: RegressionFeatures,
        state: Option<FittedState>,
    }

    impl RegressionForecaster {
        /// Create a regression forecaster using OLS with the given features.
        pub fn ols(features: RegressionFeatures) -> Self {
            Self {
                features,
                state: None,
            }
        }

        /// Create a trend-only OLS forecaster (linear regression on time index).
        pub fn linear_trend() -> Self {
            Self::ols(RegressionFeatures::new().trend().no_exog())
        }

        /// Create an autoregressive OLS forecaster with the given number of lags.
        pub fn ar(lags: usize) -> Self {
            Self::ols(RegressionFeatures::new().no_trend().lags(lags).no_exog())
        }

        /// Create a trend + autoregressive OLS forecaster.
        pub fn trend_ar(lags: usize) -> Self {
            Self::ols(RegressionFeatures::new().trend().lags(lags))
        }

        /// Get the feature configuration.
        pub fn features(&self) -> &RegressionFeatures {
            &self.features
        }

        /// Get the fitted OLS result (coefficients, R², etc.) if fitted.
        pub fn fitted_ols(&self) -> Option<&anofox_regression::solvers::FittedOls> {
            self.state.as_ref().map(|s| &s.model)
        }

        /// Recursive multi-step prediction for models with lag features.
        fn predict_recursive(
            &self,
            state: &FittedState,
            horizon: usize,
            future_regressors: Option<&HashMap<String, Vec<f64>>>,
        ) -> Result<Vec<f64>> {
            let mut x_future = state.features.build_future_matrix(
                horizon,
                state.n_total,
                &state.tail_values,
                future_regressors,
                &state.exog_names,
            )?;

            if state.features.max_lag == 0 {
                // No lags — direct (non-recursive) prediction
                let preds = state.model.predict(&x_future);
                return Ok(preds.iter().copied().collect());
            }

            // Recursive: predict one step at a time, feeding predictions back as lags
            let trend_offset = if state.features.use_trend { 1 } else { 0 };
            let mut predictions = Vec::with_capacity(horizon);
            let mut recent: Vec<f64> = state.tail_values.clone();

            for h in 0..horizon {
                // Update lag columns with most recent known/predicted values
                for lag in 1..=state.features.max_lag {
                    let col = trend_offset + (lag - 1);
                    let idx = recent.len() as isize - lag as isize;
                    if idx >= 0 {
                        x_future[(h, col)] = recent[idx as usize];
                    }
                }

                // Predict this single step
                let row = x_future.submatrix(h, 0, 1, x_future.ncols());
                let row_mat = Mat::from_fn(1, row.ncols(), |r, c| row[(r, c)]);
                let pred = state.model.predict(&row_mat);
                let y_hat = pred[0];
                predictions.push(y_hat);
                recent.push(y_hat);
            }

            Ok(predictions)
        }
    }

    impl Clone for RegressionForecaster {
        fn clone(&self) -> Self {
            // State is not Clone (FittedOls), so we only clone config
            Self {
                features: self.features.clone(),
                state: None,
            }
        }
    }

    impl Forecaster for RegressionForecaster {
        fn fit(&mut self, series: &TimeSeries) -> Result<()> {
            validate_series_complete(series)?;
            let values = series.primary_values();
            let n = values.len();

            let (x, y, n_train, exog_names) = self.features.build_matrices(series)?;

            // Fit OLS via anofox-regression
            let ols = OlsRegressor::builder()
                .with_intercept(true)
                .build()
                .fit(&x, &y)
                .map_err(|e| ForecastError::ComputationError(format!("OLS fit failed: {}", e)))?;

            // In-sample predictions
            let in_sample_preds = ols.predict(&x);

            // Build full-length fitted values (NaN-padded for lag offset)
            let offset = self.features.lag_offset();
            let mut fitted_values = vec![f64::NAN; n];
            let mut residuals = vec![f64::NAN; n];
            for i in 0..n_train {
                fitted_values[offset + i] = in_sample_preds[i];
                residuals[offset + i] = values[offset + i] - in_sample_preds[i];
            }

            // Store tail values for recursive prediction
            let tail_len = self.features.max_lag.max(1);
            let tail_values = values[n.saturating_sub(tail_len)..].to_vec();

            self.state = Some(FittedState {
                model: ols,
                features: self.features.clone(),
                n_total: n,
                tail_values,
                fitted_values,
                residuals,
                exog_names,
            });

            Ok(())
        }

        fn predict(&self, horizon: usize) -> Result<Forecast> {
            let state = self
                .state
                .as_ref()
                .ok_or(ForecastError::FitRequired { model: None })?;

            if horizon == 0 {
                return Ok(Forecast::new());
            }

            // If model has exog and was fit with exog, require predict_with_exog
            if !state.exog_names.is_empty() {
                return Err(ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors; use predict_with_exog() \
                     to provide future regressor values"
                        .to_string(),
                ));
            }

            let predictions = self.predict_recursive(state, horizon, None)?;
            Ok(Forecast::from_values(predictions))
        }

        fn predict_with_intervals(&self, horizon: usize, _level: f64) -> Result<Forecast> {
            // TODO: leverage OLS prediction intervals from anofox-regression
            self.predict(horizon)
        }

        fn supports_exog(&self) -> bool {
            self.features.use_exog
        }

        fn has_exog(&self) -> bool {
            self.state
                .as_ref()
                .map(|s| !s.exog_names.is_empty())
                .unwrap_or(false)
        }

        fn exog_names(&self) -> Option<&[String]> {
            self.state
                .as_ref()
                .filter(|s| !s.exog_names.is_empty())
                .map(|s| s.exog_names.as_slice())
        }

        fn predict_with_exog(
            &self,
            horizon: usize,
            future_regressors: &HashMap<String, Vec<f64>>,
        ) -> Result<Forecast> {
            let state = self
                .state
                .as_ref()
                .ok_or(ForecastError::FitRequired { model: None })?;

            if horizon == 0 {
                return Ok(Forecast::new());
            }

            // Validate that all required regressors are provided
            for name in &state.exog_names {
                match future_regressors.get(name) {
                    None => {
                        return Err(ForecastError::InvalidParameter(format!(
                            "Missing future regressor '{}'. Required: {:?}",
                            name, state.exog_names
                        )));
                    }
                    Some(vals) if vals.len() < horizon => {
                        return Err(ForecastError::DimensionMismatch {
                            expected: horizon,
                            got: vals.len(),
                        });
                    }
                    _ => {}
                }
            }

            let predictions = self.predict_recursive(state, horizon, Some(future_regressors))?;
            Ok(Forecast::from_values(predictions))
        }

        fn fitted_values(&self) -> Option<&[f64]> {
            self.state.as_ref().map(|s| s.fitted_values.as_slice())
        }

        fn residuals(&self) -> Option<&[f64]> {
            self.state.as_ref().map(|s| s.residuals.as_slice())
        }

        fn name(&self) -> &str {
            "OLS"
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::core::{CalendarAnnotations, TimeSeriesBuilder};
        use approx::assert_relative_eq;
        use chrono::{Duration, TimeZone, Utc};

        fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
            let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
            (0..n).map(|i| start + Duration::days(i as i64)).collect()
        }

        fn make_linear_ts(n: usize) -> TimeSeries {
            // y = 2*t + 10 + small noise
            let values: Vec<f64> = (0..n)
                .map(|i| 2.0 * i as f64 + 10.0 + 0.01 * (i as f64 * 0.7).sin())
                .collect();
            TimeSeries::univariate(make_timestamps(n), values).unwrap()
        }

        #[test]
        fn ols_linear_trend_fit_predict() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::linear_trend();
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);

            // Should continue the linear trend: y ≈ 2*t + 10
            for (h, &pred) in forecast.primary().iter().enumerate() {
                let expected = 2.0 * (50 + h) as f64 + 10.0;
                assert_relative_eq!(pred, expected, epsilon = 0.5);
            }
        }

        #[test]
        fn ols_linear_trend_fitted_values() {
            let ts = make_linear_ts(30);
            let mut model = RegressionForecaster::linear_trend();
            model.fit(&ts).unwrap();

            let fitted = model.fitted_values().unwrap();
            assert_eq!(fitted.len(), 30);

            // All should be finite (no lags = no NaN padding)
            for &v in fitted {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn ols_ar_model() {
            // AR(1) process: y[t] = 0.8 * y[t-1] + 1.0
            let n = 100;
            let mut values = vec![10.0];
            for i in 1..n {
                values.push(0.8 * values[i - 1] + 1.0 + 0.01 * (i as f64).sin());
            }
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ar(1);
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);

            // Predictions should converge toward the stationary mean ≈ 5.0
            for &pred in forecast.primary() {
                assert!(pred.is_finite());
                assert!(pred > 0.0 && pred < 20.0);
            }
        }

        #[test]
        fn ols_ar_fitted_has_nan_padding() {
            let ts = make_linear_ts(30);
            let mut model = RegressionForecaster::ar(3);
            model.fit(&ts).unwrap();

            let fitted = model.fitted_values().unwrap();
            assert_eq!(fitted.len(), 30);

            // First 3 values should be NaN (lag offset)
            assert!(fitted[0].is_nan());
            assert!(fitted[1].is_nan());
            assert!(fitted[2].is_nan());
            // Rest should be finite
            assert!(fitted[3].is_finite());
        }

        #[test]
        fn ols_trend_ar_combined() {
            let ts = make_linear_ts(60);
            let mut model = RegressionForecaster::trend_ar(2);
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);

            // Trend+AR model should produce finite, increasing forecasts
            for &pred in forecast.primary() {
                assert!(pred.is_finite());
            }
            // Should be increasing (upward trend)
            assert!(forecast.primary()[4] > forecast.primary()[0]);
        }

        #[test]
        fn ols_with_exogenous_regressors() {
            // y = 3*x + 5 + trend
            let n = 50;
            let x_vals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
            let values: Vec<f64> = (0..n)
                .map(|i| 3.0 * x_vals[i] + 5.0 + 0.1 * i as f64)
                .collect();

            let cal = CalendarAnnotations::new()
                .with_regressor("temperature".to_string(), x_vals.clone());

            let ts = TimeSeriesBuilder::new()
                .timestamps(make_timestamps(n))
                .values(values)
                .calendar(cal)
                .build()
                .unwrap();

            let mut model = RegressionForecaster::ols(RegressionFeatures::new().trend().no_exog());
            // First verify it works without exog
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);

            // Now with exog
            let mut model_exog = RegressionForecaster::ols(RegressionFeatures::new().trend());
            model_exog.fit(&ts).unwrap();

            assert!(model_exog.supports_exog());
            assert!(model_exog.has_exog());
            assert_eq!(model_exog.exog_names().unwrap(), &["temperature"]);

            // predict() should error because exog regressors are needed
            assert!(model_exog.predict(5).is_err());

            // predict_with_exog() should work
            let future_x: Vec<f64> = (n..n + 5).map(|i| (i as f64 * 0.3).sin()).collect();
            let mut future_regs = HashMap::new();
            future_regs.insert("temperature".to_string(), future_x);
            let forecast = model_exog.predict_with_exog(5, &future_regs).unwrap();
            assert_eq!(forecast.primary().len(), 5);
        }

        #[test]
        fn ols_exog_missing_regressor_errors() {
            let n = 30;
            let cal = CalendarAnnotations::new().with_regressor("x".to_string(), vec![1.0; n]);
            let ts = TimeSeriesBuilder::new()
                .timestamps(make_timestamps(n))
                .values(vec![1.0; n])
                .calendar(cal)
                .build()
                .unwrap();

            let mut model = RegressionForecaster::ols(RegressionFeatures::new().trend());
            model.fit(&ts).unwrap();

            let future_regs = HashMap::new(); // missing "x"
            assert!(model.predict_with_exog(5, &future_regs).is_err());
        }

        #[test]
        fn ols_name() {
            let model = RegressionForecaster::linear_trend();
            assert_eq!(model.name(), "OLS");
        }

        #[test]
        fn ols_residuals_sum_near_zero() {
            let ts = make_linear_ts(40);
            let mut model = RegressionForecaster::linear_trend();
            model.fit(&ts).unwrap();

            let residuals = model.residuals().unwrap();
            let sum: f64 = residuals.iter().filter(|r| r.is_finite()).sum();
            assert!(sum.abs() < 1.0, "residuals sum = {}", sum);
        }

        #[test]
        fn ols_r_squared() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::linear_trend();
            model.fit(&ts).unwrap();

            let ols = model.fitted_ols().unwrap();
            let r2 = ols.r_squared();
            assert!(
                r2 > 0.99,
                "R² should be near 1.0 for linear data, got {}",
                r2
            );
        }

        #[test]
        fn ols_insufficient_data() {
            let ts = TimeSeries::univariate(make_timestamps(2), vec![1.0, 2.0]).unwrap();
            let mut model = RegressionForecaster::ar(3);
            assert!(model.fit(&ts).is_err());
        }

        #[test]
        fn ols_no_features_errors() {
            let ts = make_linear_ts(30);
            let mut model =
                RegressionForecaster::ols(RegressionFeatures::new().no_trend().no_exog());
            assert!(model.fit(&ts).is_err());
        }

        #[test]
        fn ols_zero_horizon() {
            let ts = make_linear_ts(30);
            let mut model = RegressionForecaster::linear_trend();
            model.fit(&ts).unwrap();
            let forecast = model.predict(0).unwrap();
            assert!(forecast.primary().is_empty());
        }

        #[test]
        fn ols_model_registry_integration() {
            use crate::models::{ModelRegistry, ModelSpec};

            let mut reg = ModelRegistry::new();
            reg.register(ModelSpec::new(
                "OLS(trend)",
                || Box::new(RegressionForecaster::linear_trend()),
                false,
            ));
            reg.register(ModelSpec::new(
                "OLS(AR3)",
                || Box::new(RegressionForecaster::ar(3)),
                false,
            ));

            assert_eq!(reg.len(), 2);

            let ts = make_linear_ts(50);
            for spec in reg.iter() {
                let mut model = spec.create();
                model.fit(&ts).unwrap();
                // AR model won't have exog, so predict should work
                if !model.has_exog() {
                    let fc = model.predict(5).unwrap();
                    assert_eq!(fc.primary().len(), 5);
                }
            }
        }
    }
}

#[cfg(feature = "postprocess")]
pub use ols_impl::{RegressionFeatures, RegressionForecaster};
