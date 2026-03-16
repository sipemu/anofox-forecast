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
    use std::sync::Arc;

    use anofox_regression::core::IntervalType;
    use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
    use faer::{Col, Mat};

    use crate::core::{Forecast, TimeSeries};
    use crate::error::{ForecastError, Result};
    use crate::models::{validate_series_complete, Forecaster};
    use crate::seasonality::dummy::DummySeasonality;
    use crate::seasonality::exponential_trend::ExponentialTrend;
    use crate::seasonality::fourier::fourier_terms;
    use crate::seasonality::polynomial::PolynomialTrend;
    use crate::seasonality::theilsen::TheilSenTrend;
    use crate::seasonality::traits::{Recency, SeasonalComponent, TrendComponent};

    // ── Feature safety classification ────────────────────────────────

    /// Classification of a feature by its data-leakage risk in cross-validation.
    ///
    /// | Level | Examples | CV requirement |
    /// |---|---|---|
    /// | `Deterministic` | Fourier terms, raw trend index, lags | None — always safe |
    /// | `DataDependent` | Fitted PolynomialTrend, DummySeasonal | Re-fit per fold |
    /// | `Structural` | Changepoint regime indicator | Re-detect per fold; flag if break in test |
    /// | `External` | User-provided exogenous regressors | User's responsibility |
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FeatureSafety {
        /// Computed from the time index only — no data leakage possible.
        Deterministic,
        /// Fitted from training data — must be re-fit per CV fold.
        DataDependent,
        /// Derived from structural analysis — forward-filled during prediction.
        Structural,
        /// User-provided exogenous regressors — user's responsibility.
        External,
    }

    // ── Structural feature trait ─────────────────────────────────────

    /// A feature derived from structural analysis that is forward-filled during prediction.
    ///
    /// Implementations compute one or more columns from training data. During
    /// prediction, each column is frozen at a constant value — the model
    /// never sees future structural changes.
    ///
    /// # Fill strategies
    ///
    /// The [`fill_values`](Self::fill_values) method returns the constant value
    /// to repeat for each column during prediction. Different features use
    /// different strategies:
    ///
    /// - **Forward-fill** (changepoints): return the value at the last training
    ///   index — the model continues in the last known regime.
    /// - **Constant fill** (outlier indicators): return a fixed default (e.g., 0.0)
    ///   — the model assumes no outliers in the forecast period.
    pub trait StructuralFeature: std::fmt::Debug + Send + Sync {
        /// Column names this feature produces.
        fn column_names(&self) -> Vec<String>;

        /// Number of output columns.
        fn n_columns(&self) -> usize {
            self.column_names().len()
        }

        /// Compute column values for observation indices `0..n`.
        ///
        /// Returns one `Vec<f64>` of length `n` per column.
        fn compute(&self, n: usize) -> Vec<Vec<f64>>;

        /// Constant values to use for each column during prediction.
        ///
        /// Returns one value per column. Each value is repeated for every
        /// forecast step.
        fn fill_values(&self, n_train: usize) -> Vec<f64>;

        /// Human-readable name for reporting.
        fn name(&self) -> &str;
    }

    // ── Changepoint encoding ─────────────────────────────────────────

    /// How changepoint locations are encoded as regression features.
    ///
    /// With `k` detected changepoints (e.g., at indices 50 and 120):
    ///
    /// | Encoding | Columns | Description |
    /// |---|---|---|
    /// | `StepFunctions` | `k` binary columns | Each column flips 0→1 at its CP |
    /// | `RegimeIndex` | 1 column | Values 0, 1, …, k per segment |
    /// | `CumulativeCount` | 1 column | Count of CPs at or before index |
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum ChangepointEncoding {
        /// One binary column per changepoint (0 before, 1 at-or-after).
        #[default]
        StepFunctions,
        /// Single column with ordinal regime index (0, 1, …, k).
        RegimeIndex,
        /// Single column counting changepoints at or before each index.
        CumulativeCount,
    }

    /// Changepoint-based structural feature with configurable encoding.
    ///
    /// Produces regression columns that encode the detected changepoint
    /// locations. During prediction, columns are forward-filled with
    /// their last training values (the model stays in the last known regime).
    #[derive(Debug, Clone)]
    pub struct ChangepointFeature {
        /// Sorted changepoint indices.
        indices: Vec<usize>,
        /// How to encode the changepoints.
        encoding: ChangepointEncoding,
    }

    impl ChangepointFeature {
        /// Create a new changepoint feature with the given encoding.
        pub fn new(mut indices: Vec<usize>, encoding: ChangepointEncoding) -> Self {
            indices.sort_unstable();
            Self { indices, encoding }
        }

        /// Create a changepoint feature with the default `StepFunctions` encoding.
        pub fn step_functions(indices: Vec<usize>) -> Self {
            Self::new(indices, ChangepointEncoding::StepFunctions)
        }
    }

    impl StructuralFeature for ChangepointFeature {
        fn column_names(&self) -> Vec<String> {
            match self.encoding {
                ChangepointEncoding::StepFunctions => self
                    .indices
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("__cp_step_{}", i + 1))
                    .collect(),
                ChangepointEncoding::RegimeIndex => vec!["__cp_regime".into()],
                ChangepointEncoding::CumulativeCount => vec!["__cp_count".into()],
            }
        }

        fn compute(&self, n: usize) -> Vec<Vec<f64>> {
            match self.encoding {
                ChangepointEncoding::StepFunctions => self
                    .indices
                    .iter()
                    .map(|&cp| (0..n).map(|i| if i >= cp { 1.0 } else { 0.0 }).collect())
                    .collect(),
                ChangepointEncoding::RegimeIndex | ChangepointEncoding::CumulativeCount => {
                    let col: Vec<f64> = (0..n)
                        .map(|i| self.indices.iter().filter(|&&cp| cp <= i).count() as f64)
                        .collect();
                    vec![col]
                }
            }
        }

        fn fill_values(&self, n_train: usize) -> Vec<f64> {
            match self.encoding {
                ChangepointEncoding::StepFunctions => self
                    .indices
                    .iter()
                    .map(|&cp| if n_train > 0 && cp < n_train { 1.0 } else { 0.0 })
                    .collect(),
                ChangepointEncoding::RegimeIndex | ChangepointEncoding::CumulativeCount => {
                    let count =
                        self.indices.iter().filter(|&&cp| cp < n_train).count() as f64;
                    vec![count]
                }
            }
        }

        fn name(&self) -> &str {
            "ChangepointFeature"
        }
    }

    // ── Component specifications ────────────────────────────────────

    /// Specifies a trend model to include as a regression feature column.
    ///
    /// Each trend type produces **one column** in the design matrix containing
    /// the fitted trend values. During prediction, `predict_trend(horizon)` is
    /// used to generate future values.
    ///
    /// All trend components are fitted with `Recency::Full` so the regression
    /// model sees the component's view of the entire training window.
    ///
    /// # Cross-validation caveat
    ///
    /// Trend features are functions of the training data only. If you use
    /// [`Recency::Auto`] (changepoint-based) on a manually created component,
    /// ensure the changepoint detection only sees the training fold — otherwise
    /// the changepoint location leaks future information into the features.
    #[derive(Debug, Clone)]
    pub enum TrendType {
        /// Linear trend via [`PolynomialTrend`](crate::seasonality::PolynomialTrend) degree 1.
        Linear,
        /// Quadratic trend via [`PolynomialTrend`](crate::seasonality::PolynomialTrend) degree 2.
        Quadratic,
        /// Cubic trend via [`PolynomialTrend`](crate::seasonality::PolynomialTrend) degree 3.
        Cubic,
        /// Exponential trend via [`ExponentialTrend`](crate::seasonality::ExponentialTrend).
        /// Requires positive values.
        Exponential,
        /// Theil-Sen robust linear trend via [`TheilSenTrend`](crate::seasonality::TheilSenTrend).
        TheilSen,
    }

    impl TrendType {
        /// Feature safety classification.
        pub fn safety(&self) -> FeatureSafety {
            FeatureSafety::DataDependent
        }
    }

    /// Specifies a seasonal component to include as regression feature column(s).
    ///
    /// - [`Fourier`](SeasonalSpec::Fourier): `2 * order` sin/cos columns — deterministic
    ///   functions of the time index, no fitting required (Prophet-style).
    /// - [`Dummy`](SeasonalSpec::Dummy): 1 column of per-period seasonal means,
    ///   fitted from training data.
    #[derive(Debug, Clone)]
    pub enum SeasonalSpec {
        /// Fourier seasonality: `2 * order` sin/cos columns at the given period.
        ///
        /// Period is in observation units (e.g., 7 for weekly with daily data,
        /// 12 for yearly with monthly data).
        Fourier {
            /// Seasonal period in observation units.
            period: usize,
            /// Number of Fourier pairs (total columns = 2 * order).
            order: usize,
        },
        /// Dummy seasonal encoding with the given period.
        ///
        /// One column of per-position means (averaged over all full cycles
        /// in the training data).
        Dummy(usize),
    }

    impl SeasonalSpec {
        /// Feature safety classification.
        pub fn safety(&self) -> FeatureSafety {
            match self {
                SeasonalSpec::Fourier { .. } => FeatureSafety::Deterministic,
                SeasonalSpec::Dummy(_) => FeatureSafety::DataDependent,
            }
        }
    }

    // ── Fitted component storage ────────────────────────────────────

    /// Internal storage for a fitted trend/seasonal component.
    #[derive(Debug)]
    enum FittedComponentState {
        Polynomial(PolynomialTrend),
        Exponential(ExponentialTrend),
        TheilSen(TheilSenTrend),
        Dummy(DummySeasonality),
        Fourier { period: usize, order: usize },
        /// Structural feature — forward-filled during prediction.
        Structural {
            /// Prediction fill values — one per column, repeated for every forecast step.
            fill_values: Vec<f64>,
        },
    }

    impl FittedComponentState {
        /// Feature safety classification.
        #[allow(dead_code)]
        fn safety(&self) -> FeatureSafety {
            match self {
                Self::Polynomial(_)
                | Self::Exponential(_)
                | Self::TheilSen(_)
                | Self::Dummy(_) => FeatureSafety::DataDependent,
                Self::Fourier { .. } => FeatureSafety::Deterministic,
                Self::Structural { .. } => FeatureSafety::Structural,
            }
        }

        /// Generate future feature columns for this component.
        ///
        /// Returns one `Vec<f64>` per column (most components produce 1 column,
        /// Fourier produces `2 * order`).
        fn predict(&self, horizon: usize, n_train: usize) -> Vec<Vec<f64>> {
            match self {
                Self::Polynomial(p) => vec![p.predict_trend(horizon)],
                Self::Exponential(e) => vec![e.predict_trend(horizon)],
                Self::TheilSen(t) => vec![t.predict_trend(horizon)],
                Self::Dummy(d) => vec![d.predict_seasonal(horizon)],
                Self::Fourier { period, order } => {
                    let timestamps: Vec<f64> =
                        (0..horizon).map(|h| (n_train + h) as f64).collect();
                    fourier_terms(&timestamps, *period as f64, *order).unwrap_or_default()
                }
                Self::Structural { fill_values } => fill_values
                    .iter()
                    .map(|&v| vec![v; horizon])
                    .collect(),
            }
        }
    }

    // ── Feature specification ───────────────────────────────────────

    /// Configures which features are built from a [`TimeSeries`] for
    /// the regression model.
    ///
    /// Features are added to the design matrix in this order:
    /// 1. Linear trend index (if `use_trend`)
    /// 2. Autoregressive lags
    /// 3. Trend component columns ([`TrendType`])
    /// 4. Seasonal component columns ([`SeasonalSpec`])
    /// 5. Structural feature columns ([`StructuralFeature`])
    /// 6. Exogenous regressors (if `use_exog`)
    #[derive(Debug, Clone)]
    pub struct RegressionFeatures {
        /// Include a linear trend index (0, 1, …, n-1).
        pub use_trend: bool,
        /// Number of autoregressive lags to include.
        pub max_lag: usize,
        /// Include exogenous regressors from the TimeSeries (if present).
        pub use_exog: bool,
        /// Trend components to include as feature columns.
        pub trend_components: Vec<TrendType>,
        /// Seasonal components to include as feature columns.
        pub seasonal_components: Vec<SeasonalSpec>,
        /// Structural features (forward-filled during prediction).
        pub structural_features: Vec<Arc<dyn StructuralFeature>>,
    }

    impl Default for RegressionFeatures {
        fn default() -> Self {
            Self {
                use_trend: true,
                max_lag: 0,
                use_exog: true,
                trend_components: Vec::new(),
                seasonal_components: Vec::new(),
                structural_features: Vec::new(),
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

        /// Add a trend component as a regression feature column.
        pub fn with_trend_component(mut self, trend: TrendType) -> Self {
            self.trend_components.push(trend);
            self
        }

        /// Add a seasonal component as regression feature column(s).
        pub fn with_seasonal(mut self, seasonal: SeasonalSpec) -> Self {
            self.seasonal_components.push(seasonal);
            self
        }

        /// Add Fourier seasonality features (shorthand for
        /// `with_seasonal(SeasonalSpec::Fourier { period, order })`).
        pub fn fourier(self, period: usize, order: usize) -> Self {
            self.with_seasonal(SeasonalSpec::Fourier { period, order })
        }

        /// Add dummy seasonal encoding (shorthand for
        /// `with_seasonal(SeasonalSpec::Dummy(period))`).
        pub fn dummy_seasonal(self, period: usize) -> Self {
            self.with_seasonal(SeasonalSpec::Dummy(period))
        }

        /// Add a structural feature (forward-filled during prediction).
        pub fn with_structural(mut self, feature: Arc<dyn StructuralFeature>) -> Self {
            self.structural_features.push(feature);
            self
        }

        /// Add changepoint step-function features (convenience).
        pub fn with_changepoint_steps(self, indices: Vec<usize>) -> Self {
            self.with_structural(Arc::new(ChangepointFeature::step_functions(indices)))
        }

        /// Add changepoint features with specified encoding (convenience).
        pub fn with_changepoints(
            self,
            indices: Vec<usize>,
            encoding: ChangepointEncoding,
        ) -> Self {
            self.with_structural(Arc::new(ChangepointFeature::new(indices, encoding)))
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
            // Trend component columns
            for trend in &self.trend_components {
                match trend {
                    TrendType::Linear => names.push("__linear_trend".to_string()),
                    TrendType::Quadratic => names.push("__quadratic_trend".to_string()),
                    TrendType::Cubic => names.push("__cubic_trend".to_string()),
                    TrendType::Exponential => names.push("__exp_trend".to_string()),
                    TrendType::TheilSen => names.push("__theilsen_trend".to_string()),
                }
            }
            // Seasonal component columns
            for seasonal in &self.seasonal_components {
                match seasonal {
                    SeasonalSpec::Fourier { period, order } => {
                        for k in 1..=*order {
                            names.push(format!("__fourier_p{}_sin_{}", period, k));
                            names.push(format!("__fourier_p{}_cos_{}", period, k));
                        }
                    }
                    SeasonalSpec::Dummy(period) => {
                        names.push(format!("__seasonal_{}", period));
                    }
                }
            }
            // Structural feature columns
            for sf in &self.structural_features {
                names.extend(sf.column_names());
            }
            if self.use_exog {
                for name in exog_names {
                    names.push(name.clone());
                }
            }
            names
        }

        /// Classify all feature columns by their data-leakage risk.
        ///
        /// Returns `(name, safety)` pairs in design-matrix column order.
        pub fn classify_features(&self, exog_names: &[String]) -> Vec<(String, FeatureSafety)> {
            let mut result = Vec::new();
            if self.use_trend {
                result.push(("__trend".to_string(), FeatureSafety::Deterministic));
            }
            for lag in 1..=self.max_lag {
                result.push((format!("__lag_{}", lag), FeatureSafety::Deterministic));
            }
            for trend in &self.trend_components {
                let name = match trend {
                    TrendType::Linear => "__linear_trend",
                    TrendType::Quadratic => "__quadratic_trend",
                    TrendType::Cubic => "__cubic_trend",
                    TrendType::Exponential => "__exp_trend",
                    TrendType::TheilSen => "__theilsen_trend",
                };
                result.push((name.to_string(), trend.safety()));
            }
            for seasonal in &self.seasonal_components {
                match seasonal {
                    SeasonalSpec::Fourier { period, order } => {
                        for k in 1..=*order {
                            result.push((
                                format!("__fourier_p{}_sin_{}", period, k),
                                FeatureSafety::Deterministic,
                            ));
                            result.push((
                                format!("__fourier_p{}_cos_{}", period, k),
                                FeatureSafety::Deterministic,
                            ));
                        }
                    }
                    SeasonalSpec::Dummy(period) => {
                        result.push((
                            format!("__seasonal_{}", period),
                            FeatureSafety::DataDependent,
                        ));
                    }
                }
            }
            for sf in &self.structural_features {
                for col_name in sf.column_names() {
                    result.push((col_name, FeatureSafety::Structural));
                }
            }
            if self.use_exog {
                for name in exog_names {
                    result.push((name.clone(), FeatureSafety::External));
                }
            }
            result
        }

        /// Build the design matrix and target vector from a TimeSeries.
        ///
        /// Returns `(X, y, n_train, exog_names, fitted_components)` where `n_train`
        /// is the number of usable rows (= n - max_lag) and `fitted_components`
        /// holds any trend/seasonal components fitted during matrix construction.
        fn build_matrices(
            &self,
            series: &TimeSeries,
        ) -> Result<(Mat<f64>, Col<f64>, usize, Vec<String>, Vec<FittedComponentState>)> {
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
                    "No features configured — enable at least one of: trend, lags, components, or exog"
                        .to_string(),
                ));
            }

            // ── Fit trend/seasonal components ───────────────────────────
            let mut fitted_components = Vec::new();

            for trend_type in &self.trend_components {
                let comp = match trend_type {
                    TrendType::Linear => {
                        let mut p = PolynomialTrend::new(1).with_recency(Recency::Full);
                        p.fit_trend(values)?;
                        FittedComponentState::Polynomial(p)
                    }
                    TrendType::Quadratic => {
                        let mut p = PolynomialTrend::new(2).with_recency(Recency::Full);
                        p.fit_trend(values)?;
                        FittedComponentState::Polynomial(p)
                    }
                    TrendType::Cubic => {
                        let mut p = PolynomialTrend::new(3).with_recency(Recency::Full);
                        p.fit_trend(values)?;
                        FittedComponentState::Polynomial(p)
                    }
                    TrendType::Exponential => {
                        let mut e = ExponentialTrend::new().with_recency(Recency::Full);
                        e.fit_trend(values)?;
                        FittedComponentState::Exponential(e)
                    }
                    TrendType::TheilSen => {
                        let mut t = TheilSenTrend::new().with_recency(Recency::Full);
                        t.fit_trend(values)?;
                        FittedComponentState::TheilSen(t)
                    }
                };
                fitted_components.push(comp);
            }

            for seasonal in &self.seasonal_components {
                match seasonal {
                    SeasonalSpec::Fourier { period, order } => {
                        fitted_components.push(FittedComponentState::Fourier {
                            period: *period,
                            order: *order,
                        });
                    }
                    SeasonalSpec::Dummy(period) => {
                        let mut d = DummySeasonality::new();
                        d.fit_seasonal(values, *period)?;
                        fitted_components.push(FittedComponentState::Dummy(d));
                    }
                }
            }

            // ── Build design matrix ─────────────────────────────────────
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

            // Trend/seasonal component columns
            for comp in &fitted_components {
                match comp {
                    FittedComponentState::Polynomial(p) => {
                        let fitted = p.fitted_trend();
                        for i in 0..n_train {
                            x[(i, col_idx)] = fitted[offset + i];
                        }
                        col_idx += 1;
                    }
                    FittedComponentState::Exponential(e) => {
                        let fitted = e.fitted_trend();
                        for i in 0..n_train {
                            x[(i, col_idx)] = fitted[offset + i];
                        }
                        col_idx += 1;
                    }
                    FittedComponentState::TheilSen(t) => {
                        let fitted = t.fitted_trend();
                        for i in 0..n_train {
                            x[(i, col_idx)] = fitted[offset + i];
                        }
                        col_idx += 1;
                    }
                    FittedComponentState::Dummy(d) => {
                        let fitted = d.fitted_seasonal();
                        for i in 0..n_train {
                            x[(i, col_idx)] = fitted[offset + i];
                        }
                        col_idx += 1;
                    }
                    FittedComponentState::Fourier { period, order } => {
                        let timestamps: Vec<f64> = (0..n).map(|i| i as f64).collect();
                        let basis = fourier_terms(&timestamps, *period as f64, *order)?;
                        for basis_vec in &basis {
                            for i in 0..n_train {
                                x[(i, col_idx)] = basis_vec[offset + i];
                            }
                            col_idx += 1;
                        }
                    }
                    FittedComponentState::Structural { .. } => {
                        // Handled below via structural_features iteration
                    }
                }
            }

            // Structural feature columns
            for sf in &self.structural_features {
                let columns = sf.compute(n);
                let fill = sf.fill_values(n);
                for col_vals in &columns {
                    for i in 0..n_train {
                        x[(i, col_idx)] = col_vals[offset + i];
                    }
                    col_idx += 1;
                }
                fitted_components.push(FittedComponentState::Structural {
                    fill_values: fill,
                });
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

            Ok((x, y, n_train, exog_names, fitted_components))
        }

        /// Build a design matrix for the forecast horizon.
        ///
        /// For lags: uses the last values from training + predicted values
        /// for multi-step recursive forecasting. Component columns are
        /// populated from the fitted components' `predict_trend` / `predict_seasonal`.
        fn build_future_matrix(
            &self,
            horizon: usize,
            n_total: usize,
            tail_values: &[f64],
            future_regressors: Option<&HashMap<String, Vec<f64>>>,
            exog_names: &[String],
            components: &[FittedComponentState],
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

            // Trend/seasonal component columns
            for comp in components {
                let future_cols = comp.predict(horizon, n_total);
                for col_vals in &future_cols {
                    for h in 0..horizon.min(col_vals.len()) {
                        x[(h, col_idx)] = col_vals[h];
                    }
                    col_idx += 1;
                }
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
        /// Fitted trend/seasonal components for generating future feature columns.
        components: Vec<FittedComponentState>,
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

        /// Create a trend + Fourier seasonality OLS forecaster.
        pub fn trend_fourier(period: usize, order: usize) -> Self {
            Self::ols(
                RegressionFeatures::new()
                    .trend()
                    .fourier(period, order)
                    .no_exog(),
            )
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
                &state.components,
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

            let (x, y, n_train, exog_names, components) = self.features.build_matrices(series)?;

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
                components,
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

        fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
            let state = self
                .state
                .as_ref()
                .ok_or(ForecastError::FitRequired { model: None })?;

            if horizon == 0 {
                return Ok(Forecast::new());
            }

            // OLS prediction intervals are only valid for direct (non-recursive)
            // forecasts.  Recursive models (max_lag > 0) feed predicted values
            // back as features, so the standard interval formula does not apply;
            // fall back to point-only predictions in that case.
            if state.features.max_lag > 0 {
                return self.predict(horizon);
            }

            if !state.exog_names.is_empty() {
                return Err(ForecastError::InvalidParameter(
                    "Model was fit with exogenous regressors; use predict_with_exog() \
                     to provide future regressor values"
                        .to_string(),
                ));
            }

            let x_future = state.features.build_future_matrix(
                horizon,
                state.n_total,
                &state.tail_values,
                None,
                &state.exog_names,
                &state.components,
            )?;

            let pred_result = state.model.predict_with_interval(
                &x_future,
                Some(IntervalType::Prediction),
                level,
            );

            let values: Vec<f64> = pred_result.fit.iter().copied().collect();
            let lower: Vec<f64> = pred_result.lower.iter().copied().collect();
            let upper: Vec<f64> = pred_result.upper.iter().copied().collect();

            // If intervals contain NaN (e.g. xtx_inverse unavailable), return
            // point-only to avoid misleading results.
            if lower.iter().any(|v| v.is_nan()) || upper.iter().any(|v| v.is_nan()) {
                return Ok(Forecast::from_values(values));
            }

            Ok(Forecast::from_values_with_intervals(values, lower, upper))
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
        fn ols_predict_with_intervals_linear_trend() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::linear_trend();
            model.fit(&ts).unwrap();

            let forecast = model.predict_with_intervals(5, 0.95).unwrap();
            assert_eq!(forecast.primary().len(), 5);

            // Intervals should be present for a non-recursive model
            let lower = forecast.lower().expect("lower bounds should be present");
            let upper = forecast.upper().expect("upper bounds should be present");
            assert_eq!(lower[0].len(), 5);
            assert_eq!(upper[0].len(), 5);

            // Lower < point < upper
            for h in 0..5 {
                assert!(
                    lower[0][h] < forecast.primary()[h],
                    "lower[{}] = {} should be < point = {}",
                    h, lower[0][h], forecast.primary()[h],
                );
                assert!(
                    forecast.primary()[h] < upper[0][h],
                    "point = {} should be < upper[{}] = {}",
                    forecast.primary()[h], h, upper[0][h],
                );
            }
        }

        #[test]
        fn ols_predict_with_intervals_ar_falls_back() {
            // AR models use recursive prediction; intervals should be absent
            let n = 100;
            let mut values = vec![10.0];
            for i in 1..n {
                values.push(0.8 * values[i - 1] + 1.0 + 0.01 * (i as f64).sin());
            }
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ar(1);
            model.fit(&ts).unwrap();

            let forecast = model.predict_with_intervals(5, 0.95).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            // Recursive model falls back to point-only
            assert!(forecast.lower().is_none());
            assert!(forecast.upper().is_none());
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

        // ── Component integration tests ─────────────────────────────

        fn make_seasonal_ts(n: usize, period: usize) -> TimeSeries {
            // y = 2*t + 10 * sin(2*pi*t/period) + 5
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let t = i as f64;
                    2.0 * t
                        + 10.0
                            * (2.0 * std::f64::consts::PI * t / period as f64).sin()
                        + 5.0
                })
                .collect();
            TimeSeries::univariate(make_timestamps(n), values).unwrap()
        }

        #[test]
        fn ols_fourier_seasonality() {
            let ts = make_seasonal_ts(100, 7);
            let mut model = RegressionForecaster::trend_fourier(7, 3);
            model.fit(&ts).unwrap();

            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Fourier prediction should be finite");
            }

            // R² should be high since y = trend + sin is well modeled by Fourier + trend
            let r2 = model.fitted_ols().unwrap().r_squared();
            assert!(
                r2 > 0.95,
                "R² should be > 0.95 for sinusoidal data, got {}",
                r2
            );
        }

        #[test]
        fn ols_dummy_seasonal() {
            let ts = make_seasonal_ts(56, 7); // 8 full weeks
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().trend().dummy_seasonal(7).no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn ols_theilsen_trend_component() {
            let ts = make_linear_ts(60);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .with_trend_component(TrendType::TheilSen)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            // Should produce increasing predictions for linear data
            assert!(forecast.primary()[4] > forecast.primary()[0]);
        }

        #[test]
        fn ols_exponential_trend_component() {
            // y = exp(0.05 * t)
            let n = 60;
            let values: Vec<f64> = (0..n)
                .map(|i| (0.05 * i as f64).exp())
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .with_trend_component(TrendType::Exponential)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite());
                assert!(v > 0.0, "exponential trend should predict positive values");
            }
        }

        #[test]
        fn ols_trend_plus_fourier_plus_lags() {
            let ts = make_seasonal_ts(100, 7);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .lags(2)
                    .fourier(7, 3)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(10).unwrap();
            assert_eq!(forecast.primary().len(), 10);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn ols_quadratic_trend_component() {
            // y = t^2
            let n = 50;
            let values: Vec<f64> = (0..n).map(|i| (i as f64).powi(2)).collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .with_trend_component(TrendType::Quadratic)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            for (h, &pred) in forecast.primary().iter().enumerate() {
                let expected = ((n + h) as f64).powi(2);
                assert_relative_eq!(pred, expected, epsilon = 5.0);
            }
        }

        #[test]
        fn ols_multiple_fourier_periods() {
            // Two seasonal components: period 7 + period 12
            let n = 120;
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let t = i as f64;
                    5.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin()
                        + 3.0 * (2.0 * std::f64::consts::PI * t / 12.0).cos()
                        + 0.5 * t
                })
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .fourier(7, 2)
                    .fourier(12, 2)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let r2 = model.fitted_ols().unwrap().r_squared();
            assert!(r2 > 0.85, "R² should be high for dual-seasonal data, got {}", r2);

            let forecast = model.predict(12).unwrap();
            assert_eq!(forecast.primary().len(), 12);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn ols_component_only_no_trend_index() {
            // Use TheilSen component as the only feature (no raw trend index)
            let ts = make_linear_ts(40);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .with_trend_component(TrendType::Linear)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(3).unwrap();
            assert_eq!(forecast.primary().len(), 3);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        // ── Feature safety classification tests ─────────────────────

        #[test]
        fn classify_features_deterministic_only() {
            let features = RegressionFeatures::new()
                .trend()
                .fourier(7, 2)
                .no_exog();
            let classified = features.classify_features(&[]);
            // __trend + 4 Fourier columns = 5 total
            assert_eq!(classified.len(), 5);
            assert!(classified.iter().all(|(_, s)| *s == FeatureSafety::Deterministic));
        }

        #[test]
        fn classify_features_mixed() {
            let features = RegressionFeatures::new()
                .trend()
                .lags(2)
                .with_trend_component(TrendType::TheilSen)
                .fourier(7, 1)
                .dummy_seasonal(12)
                .with_changepoint_steps(vec![10, 50])
                .no_exog();
            let exog: Vec<String> = vec![];
            let classified = features.classify_features(&exog);
            // __trend (D) + __lag_1 (D) + __lag_2 (D) + __theilsen (DD) +
            // __fourier_p7_sin_1 (D) + __fourier_p7_cos_1 (D) + __seasonal_12 (DD) +
            // __cp_step_1 (S) + __cp_step_2 (S) = 9
            assert_eq!(classified.len(), 9);
            assert_eq!(classified[0], ("__trend".into(), FeatureSafety::Deterministic));
            assert_eq!(classified[1], ("__lag_1".into(), FeatureSafety::Deterministic));
            assert_eq!(classified[3], ("__theilsen_trend".into(), FeatureSafety::DataDependent));
            assert_eq!(classified[4], ("__fourier_p7_sin_1".into(), FeatureSafety::Deterministic));
            assert_eq!(classified[6], ("__seasonal_12".into(), FeatureSafety::DataDependent));
            assert_eq!(classified[7], ("__cp_step_1".into(), FeatureSafety::Structural));
            assert_eq!(classified[8], ("__cp_step_2".into(), FeatureSafety::Structural));
        }

        #[test]
        fn classify_features_with_exog() {
            let features = RegressionFeatures::new().trend();
            let exog = vec!["temperature".to_string()];
            let classified = features.classify_features(&exog);
            assert_eq!(classified.last().unwrap().1, FeatureSafety::External);
        }

        // ── Changepoint feature tests ───────────────────────────────

        #[test]
        fn changepoint_step_functions_multiple_cps() {
            let cp = ChangepointFeature::step_functions(vec![10, 50]);
            assert_eq!(cp.column_names(), vec!["__cp_step_1", "__cp_step_2"]);
            let cols = cp.compute(100);
            assert_eq!(cols.len(), 2);
            // CP at 10: first 10 are 0, rest are 1
            assert_eq!(cols[0][9], 0.0);
            assert_eq!(cols[0][10], 1.0);
            assert_eq!(cols[0][99], 1.0);
            // CP at 50: first 50 are 0, rest are 1
            assert_eq!(cols[1][49], 0.0);
            assert_eq!(cols[1][50], 1.0);
        }

        #[test]
        fn changepoint_regime_index() {
            let cp = ChangepointFeature::new(vec![10, 50], ChangepointEncoding::RegimeIndex);
            assert_eq!(cp.column_names(), vec!["__cp_regime"]);
            let cols = cp.compute(100);
            assert_eq!(cols.len(), 1);
            assert_eq!(cols[0][0], 0.0);   // before any CP
            assert_eq!(cols[0][10], 1.0);  // after first CP
            assert_eq!(cols[0][49], 1.0);  // still in regime 1
            assert_eq!(cols[0][50], 2.0);  // after second CP
            assert_eq!(cols[0][99], 2.0);
        }

        #[test]
        fn changepoint_cumulative_count() {
            let cp = ChangepointFeature::new(vec![10, 50], ChangepointEncoding::CumulativeCount);
            assert_eq!(cp.column_names(), vec!["__cp_count"]);
            let cols = cp.compute(80);
            // Same numeric values as RegimeIndex
            assert_eq!(cols[0][5], 0.0);
            assert_eq!(cols[0][30], 1.0);
            assert_eq!(cols[0][60], 2.0);
        }

        #[test]
        fn changepoint_forward_fill_step_functions() {
            let cp = ChangepointFeature::step_functions(vec![10, 50]);
            let fill = cp.fill_values(80);
            assert_eq!(fill.len(), 2);
            // Both CPs are before n_train=80
            assert_eq!(fill[0], 1.0);
            assert_eq!(fill[1], 1.0);
        }

        #[test]
        fn changepoint_forward_fill_regime_index() {
            let cp = ChangepointFeature::new(vec![10, 50], ChangepointEncoding::RegimeIndex);
            let fill = cp.fill_values(80);
            assert_eq!(fill, vec![2.0]); // 2 CPs before index 80
        }

        #[test]
        fn changepoint_cp_after_training_end() {
            let cp = ChangepointFeature::step_functions(vec![100]);
            let fill = cp.fill_values(80);
            // CP at 100 is beyond n_train=80 → forward-fill is 0
            assert_eq!(fill, vec![0.0]);
        }

        #[test]
        fn changepoint_feature_in_regression() {
            // y = 2*t + 10 with a level shift of +20 at t=30
            let n = 60;
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let base = 2.0 * i as f64 + 10.0;
                    if i >= 30 { base + 20.0 } else { base }
                })
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .with_changepoint_steps(vec![30])
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let r2 = model.fitted_ols().unwrap().r_squared();
            assert!(r2 > 0.99, "R² should be high with changepoint step, got {}", r2);

            let forecast = model.predict(5).unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite());
                // Predictions should be on the shifted level (> 130 for t=60+)
                assert!(v > 120.0, "predicted {} should be on shifted level", v);
            }
        }

        #[test]
        fn changepoint_regime_index_in_regression() {
            // y = 5*regime + noise
            let n = 60;
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let regime = if i < 20 { 0.0 } else if i < 40 { 1.0 } else { 2.0 };
                    5.0 * regime + 0.01 * (i as f64 * 0.5).sin()
                })
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .with_changepoints(vec![20, 40], ChangepointEncoding::RegimeIndex)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite());
                // Should predict near 10.0 (regime 2 * 5.0)
                assert!((v - 10.0).abs() < 1.0, "predicted {} should be near 10.0", v);
            }
        }

        #[test]
        fn structural_feature_trait_custom() {
            // Custom outlier indicator: fills with 0.0 during prediction
            #[derive(Debug)]
            struct OutlierIndicator {
                outlier_indices: Vec<usize>,
            }
            impl StructuralFeature for OutlierIndicator {
                fn column_names(&self) -> Vec<String> {
                    vec!["__outlier".into()]
                }
                fn compute(&self, n: usize) -> Vec<Vec<f64>> {
                    let mut col = vec![0.0; n];
                    for &idx in &self.outlier_indices {
                        if idx < n {
                            col[idx] = 1.0;
                        }
                    }
                    vec![col]
                }
                fn fill_values(&self, _n_train: usize) -> Vec<f64> {
                    vec![0.0] // constant fill, not forward-fill
                }
                fn name(&self) -> &str {
                    "OutlierIndicator"
                }
            }

            let n = 50;
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let base = 0.5 * i as f64;
                    if i == 25 { base + 100.0 } else { base }
                })
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let indicator = Arc::new(OutlierIndicator {
                outlier_indices: vec![25],
            });

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .with_structural(indicator)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite());
                // Should not predict an outlier spike
                assert!(v < 50.0, "predicted {} should not include outlier", v);
            }
        }
    }
}

#[cfg(feature = "postprocess")]
pub use ols_impl::{
    ChangepointEncoding, ChangepointFeature, FeatureSafety, RegressionFeatures,
    RegressionForecaster, SeasonalSpec, StructuralFeature, TrendType,
};
