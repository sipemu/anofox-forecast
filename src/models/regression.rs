//! Regression-based forecasting models.
//!
//! Bridges external regression estimators (e.g., OLS from `anofox-regression`)
//! into the [`Forecaster`](crate::models::Forecaster) trait, enabling them to participate in pipelines,
//! model registries, ensembles, and cross-validation.
//!
//! # Feature engineering
//!
//! Time-series forecasting with regression requires features. The
//! [`RegressionFeatures`] builder configures which features are constructed
//! from a [`TimeSeries`](crate::core::TimeSeries) before fitting:
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
    use anofox_regression::solvers::{
        BlsRegressor, ElasticNetRegressor, FittedRegressor, InformationCriterion,
        LmDynamicRegressor, OlsRegressor, PoissonRegressor, QuantileRegressor, Regressor,
        RidgeRegressor, RlsRegressor, TweedieRegressor, WlsRegressor,
    };
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
                    .map(|&cp| {
                        if n_train > 0 && cp < n_train {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                ChangepointEncoding::RegimeIndex | ChangepointEncoding::CumulativeCount => {
                    let count = self.indices.iter().filter(|&&cp| cp < n_train).count() as f64;
                    vec![count]
                }
            }
        }

        fn name(&self) -> &str {
            "ChangepointFeature"
        }
    }

    // ── Regression backend ────────────────────────────────────────────

    /// Strategy for generating observation weights in WLS.
    #[derive(Debug, Clone)]
    pub enum WeightStrategy {
        /// Exponential decay: `w_i = decay^(n-1-i)`. Most recent observation
        /// gets weight 1.0, oldest gets `decay^(n-1)`.
        ExponentialDecay(f64),
        /// Custom weight vector (must match training row count after lag offset).
        Custom(Vec<f64>),
    }

    /// Specifies which regression estimator backs the forecaster.
    ///
    /// All backends share the same feature engineering ([`RegressionFeatures`]),
    /// design matrix construction, and recursive prediction logic. They differ
    /// only in the loss function / regularization applied during coefficient
    /// estimation.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use anofox_forecast::models::regression::*;
    ///
    /// // Ridge with λ = 0.1
    /// let model = RegressionForecaster::ridge(0.1, RegressionFeatures::new().trend().fourier(7, 3));
    ///
    /// // Quantile regression at the median
    /// let model = RegressionForecaster::quantile(0.5, RegressionFeatures::new().trend());
    /// ```
    #[derive(Debug, Clone, Default)]
    pub enum RegressionBackend {
        /// Ordinary Least Squares (default).
        #[default]
        Ols,
        /// Ridge regression (L2 regularization).
        Ridge {
            /// Regularization strength (λ ≥ 0).
            lambda: f64,
        },
        /// Elastic Net (L1 + L2 regularization).
        ElasticNet {
            /// Combined regularization strength.
            lambda: f64,
            /// Mixing parameter: 0 = pure Ridge, 1 = pure Lasso.
            alpha: f64,
        },
        /// Quantile regression — estimate a specific conditional quantile.
        Quantile {
            /// Quantile to estimate (0 < τ < 1). 0.5 = median.
            tau: f64,
        },
        /// Weighted Least Squares — down-weight older or less reliable observations.
        Wls {
            /// How observation weights are generated.
            strategy: WeightStrategy,
        },
        /// Recursive Least Squares — adaptive coefficients via forgetting factor.
        Rls {
            /// Exponential forgetting (0 < λ ≤ 1). 1.0 = equal weights, <1 = recent emphasis.
            forgetting_factor: f64,
        },
        /// Tweedie GLM — handles count, continuous, and zero-inflated data.
        Tweedie {
            /// Variance power: 0 = Gaussian, 1 = Poisson, 2 = Gamma, 3 = Inv-Gaussian.
            var_power: f64,
            /// Link function power: None = canonical. 0 = log, 1 = identity.
            link_power: Option<f64>,
        },
        /// Poisson GLM — for count data (non-negative integers).
        Poisson,
        /// Bounded Least Squares — box constraints on coefficients.
        Bls {
            /// Lower bound for all coefficients (None = unconstrained).
            lower: Option<f64>,
            /// Upper bound for all coefficients (None = unconstrained).
            upper: Option<f64>,
        },
        /// Dynamic linear model — time-varying parameters via IC-weighted model averaging.
        ///
        /// Automatically generates candidate models from variable subsets, fits each,
        /// and computes observation-level IC weights. Coefficients vary over time,
        /// giving the model the ability to adapt to structural changes.
        Dynamic {
            /// Information criterion for model weighting (default: AICc).
            ic: InformationCriterion,
            /// LOWESS smoothing span for weights (None = no smoothing).
            lowess_span: Option<f64>,
        },
    }

    impl RegressionBackend {
        /// Human-readable name for this backend.
        fn name(&self) -> &str {
            match self {
                Self::Ols => "OLS",
                Self::Ridge { .. } => "Ridge",
                Self::ElasticNet { .. } => "ElasticNet",
                Self::Quantile { .. } => "Quantile",
                Self::Wls { .. } => "WLS",
                Self::Rls { .. } => "RLS",
                Self::Tweedie { .. } => "Tweedie",
                Self::Poisson => "Poisson",
                Self::Bls { .. } => "BLS",
                Self::Dynamic { .. } => "Dynamic",
            }
        }

        /// Fit the backend to data, returning a boxed fitted regressor.
        fn fit_to(
            &self,
            x: &Mat<f64>,
            y: &Col<f64>,
        ) -> std::result::Result<Box<dyn FittedRegressor + Send>, String> {
            match self {
                Self::Ols => {
                    let model = OlsRegressor::builder().with_intercept(true).build();
                    let fitted = model.fit(x, y).map_err(|e| format!("OLS: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Ridge { lambda } => {
                    let model = RidgeRegressor::builder()
                        .with_intercept(true)
                        .lambda(*lambda)
                        .build();
                    let fitted = model.fit(x, y).map_err(|e| format!("Ridge: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::ElasticNet { lambda, alpha } => {
                    let model = ElasticNetRegressor::builder()
                        .with_intercept(true)
                        .lambda(*lambda)
                        .alpha(*alpha)
                        .build();
                    let fitted = model.fit(x, y).map_err(|e| format!("ElasticNet: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Quantile { tau } => {
                    let model = QuantileRegressor::builder()
                        .with_intercept(true)
                        .tau(*tau)
                        .build();
                    let fitted = model.fit(x, y).map_err(|e| format!("Quantile: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Wls { strategy } => {
                    let n = y.nrows();
                    let weights = match strategy {
                        WeightStrategy::ExponentialDecay(decay) => {
                            let mut w = Col::zeros(n);
                            for i in 0..n {
                                w[i] = decay.powi((n - 1 - i) as i32);
                            }
                            w
                        }
                        WeightStrategy::Custom(v) => {
                            if v.len() != n {
                                return Err(format!(
                                    "WLS: weight vector length {} != training rows {}",
                                    v.len(),
                                    n
                                ));
                            }
                            let mut w = Col::zeros(n);
                            for (i, &val) in v.iter().enumerate() {
                                w[i] = val;
                            }
                            w
                        }
                    };
                    let model = WlsRegressor::builder()
                        .with_intercept(true)
                        .weights(weights)
                        .build();
                    let fitted = model.fit(x, y).map_err(|e| format!("WLS: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Rls { forgetting_factor } => {
                    let model = RlsRegressor::builder()
                        .with_intercept(true)
                        .forgetting_factor(*forgetting_factor)
                        .build();
                    let fitted = model.fit(x, y).map_err(|e| format!("RLS: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Tweedie {
                    var_power,
                    link_power,
                } => {
                    let mut builder = TweedieRegressor::builder()
                        .with_intercept(true)
                        .var_power(*var_power);
                    if let Some(lp) = link_power {
                        builder = builder.link_power(*lp);
                    }
                    let model = builder.build();
                    let fitted = model.fit(x, y).map_err(|e| format!("Tweedie: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Poisson => {
                    let model = PoissonRegressor::builder().with_intercept(true).build();
                    let fitted = model.fit(x, y).map_err(|e| format!("Poisson: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Bls { lower, upper } => {
                    let mut builder = BlsRegressor::builder().with_intercept(true);
                    if let Some(lb) = lower {
                        builder = builder.lower_bound_all(*lb);
                    }
                    if let Some(ub) = upper {
                        builder = builder.upper_bound_all(*ub);
                    }
                    let model = builder.build();
                    let fitted = model.fit(x, y).map_err(|e| format!("BLS: {}", e))?;
                    Ok(Box::new(fitted))
                }
                Self::Dynamic { ic, lowess_span } => {
                    let mut builder = LmDynamicRegressor::builder().with_intercept(true).ic(*ic);
                    if let Some(span) = lowess_span {
                        builder = builder.lowess_span(*span);
                    } else {
                        builder = builder.no_smoothing();
                    }
                    let model = builder.build();
                    let fitted = model.fit(x, y).map_err(|e| format!("Dynamic: {}", e))?;
                    Ok(Box::new(fitted))
                }
            }
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
        Fourier {
            period: usize,
            order: usize,
        },
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
                Self::Polynomial(_) | Self::Exponential(_) | Self::TheilSen(_) | Self::Dummy(_) => {
                    FeatureSafety::DataDependent
                }
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
                    let timestamps: Vec<f64> = (0..horizon).map(|h| (n_train + h) as f64).collect();
                    fourier_terms(&timestamps, *period as f64, *order).unwrap_or_default()
                }
                Self::Structural { fill_values } => {
                    fill_values.iter().map(|&v| vec![v; horizon]).collect()
                }
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

    /// Criterion for automatic lag selection.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LagSelectionCriterion {
        /// Bayesian Information Criterion (default). Penalizes complexity more than AIC.
        Bic,
        /// Akaike Information Criterion. Penalizes complexity less than BIC.
        Aic,
    }

    #[derive(Debug, Clone)]
    pub struct RegressionFeatures {
        /// Include a linear trend index (0, 1, …, n-1).
        pub use_trend: bool,
        /// Number of autoregressive lags to include (legacy; prefer `lag_indices`).
        pub max_lag: usize,
        /// Specific lag indices to include (e.g. `vec![12]` for only lag-12).
        /// When non-empty, this takes precedence over `max_lag`.
        pub lag_indices: Vec<usize>,
        /// Automatic lag selection: (max_lag, criterion). When set, overrides max_lag/lag_indices.
        pub auto_lag_config: Option<(usize, LagSelectionCriterion)>,
        /// Include exogenous regressors from the TimeSeries (if present).
        pub use_exog: bool,
        /// Trend components to include as feature columns.
        pub trend_components: Vec<TrendType>,
        /// Seasonal components to include as feature columns.
        pub seasonal_components: Vec<SeasonalSpec>,
        /// Structural features (forward-filled during prediction).
        pub structural_features: Vec<Arc<dyn StructuralFeature>>,
        /// Regular differencing order (d). Applied before fitting, integrated after predict.
        pub diff_order: usize,
        /// Seasonal differencing specs: Vec of (order D, period s). Applied in order before fitting, integrated in reverse after predict.
        pub seasonal_diffs: Vec<(usize, usize)>,
    }

    impl Default for RegressionFeatures {
        fn default() -> Self {
            Self {
                use_trend: true,
                max_lag: 0,
                lag_indices: Vec::new(),
                auto_lag_config: None,
                use_exog: true,
                trend_components: Vec::new(),
                seasonal_components: Vec::new(),
                structural_features: Vec::new(),
                diff_order: 0,
                seasonal_diffs: Vec::new(),
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
            self.lag_indices = Vec::new(); // clear specific lags
            self
        }

        /// Automatically select the best lag order up to `max_lag` using BIC.
        ///
        /// Tries each lag order from 0 to `max_lag`, fits OLS, computes BIC, and
        /// selects the order with the lowest BIC. This is resolved during `fit()`.
        pub fn auto_lags(mut self, max_lag: usize) -> Self {
            self.auto_lag_config = Some((max_lag, LagSelectionCriterion::Bic));
            self.max_lag = 0;
            self.lag_indices = Vec::new();
            self
        }

        /// Automatically select the best lag order using the specified criterion.
        pub fn auto_lags_with(mut self, max_lag: usize, criterion: LagSelectionCriterion) -> Self {
            self.auto_lag_config = Some((max_lag, criterion));
            self.max_lag = 0;
            self.lag_indices = Vec::new();
            self
        }

        /// Include only the specified lag indices (e.g. `&[12]` for lag-12 only).
        pub fn specific_lags(mut self, lags: &[usize]) -> Self {
            let mut sorted = lags.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            self.lag_indices = sorted;
            self.max_lag = 0; // specific_lags takes precedence
            self
        }

        /// Return the effective list of lag indices.
        fn effective_lags(&self) -> Vec<usize> {
            if !self.lag_indices.is_empty() {
                self.lag_indices.clone()
            } else if self.max_lag > 0 {
                (1..=self.max_lag).collect()
            } else {
                Vec::new()
            }
        }

        /// Maximum lag value (for offset calculation and tail storage).
        fn max_effective_lag(&self) -> usize {
            if !self.lag_indices.is_empty() {
                self.lag_indices.iter().copied().max().unwrap_or(0)
            } else {
                self.max_lag
            }
        }

        /// Resolve auto-lag selection if configured.
        ///
        /// Tries each lag order 0..=max_lag, builds the feature matrix with that
        /// lag order, fits OLS, computes the information criterion, and selects
        /// the order with the lowest IC. Mutates `self.max_lag` with the result.
        fn resolve_auto_lags(&mut self, series: &TimeSeries) -> Result<()> {
            let (max_lag, criterion) = match self.auto_lag_config {
                Some(cfg) => cfg,
                None => return Ok(()),
            };

            let values = series.primary_values();
            let n = values.len();

            if max_lag == 0 || n < 4 {
                self.max_lag = 0;
                self.auto_lag_config = None;
                return Ok(());
            }

            let mut best_ic = f64::INFINITY;
            let mut best_order = 0_usize;

            for p in 0..=max_lag {
                // Temporarily set lag order
                self.max_lag = p;
                self.lag_indices.clear();

                let offset = self.lag_offset();
                if n <= offset + 1 {
                    continue; // Not enough data for this lag order
                }

                // Build matrices and fit OLS
                let result = self.build_matrices(series);
                let (x, y, n_train, _, _) = match result {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if n_train < 3 {
                    continue;
                }

                let fitted = match RegressionBackend::Ols.fit_to(&x, &y) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

                // Compute RSS from in-sample predictions
                let preds = fitted.predict(&x);
                let mut rss = 0.0_f64;
                for i in 0..n_train {
                    let r = y[i] - preds[i];
                    rss += r * r;
                }

                if !rss.is_finite() || rss <= 0.0 {
                    continue;
                }

                let n_f = n_train as f64;
                let k = x.ncols() as f64 + 1.0; // +1 for intercept

                let ic = match criterion {
                    LagSelectionCriterion::Bic => n_f * (rss / n_f).ln() + k * n_f.ln(),
                    LagSelectionCriterion::Aic => n_f * (rss / n_f).ln() + 2.0 * k,
                };

                if ic.is_finite() && ic < best_ic {
                    best_ic = ic;
                    best_order = p;
                }
            }

            // Set the winning lag order (keep auto_lag_config so re-fit works)
            self.max_lag = best_order;
            self.lag_indices.clear();

            Ok(())
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
        pub fn with_changepoints(self, indices: Vec<usize>, encoding: ChangepointEncoding) -> Self {
            self.with_structural(Arc::new(ChangepointFeature::new(indices, encoding)))
        }

        /// Apply regular differencing of order `d` before fitting.
        ///
        /// The model automatically integrates (undoes differencing) during predict.
        pub fn differencing(mut self, d: usize) -> Self {
            self.diff_order = d;
            self
        }

        /// Apply seasonal differencing of order `D` with the given period before fitting.
        ///
        /// Can be called multiple times for multi-seasonal series (e.g., weekly + yearly).
        /// Each call adds a differencing step; they are applied in order during fit
        /// and integrated in reverse order during predict.
        pub fn seasonal_differencing(mut self, d: usize, period: usize) -> Self {
            self.seasonal_diffs.push((d, period));
            self
        }

        /// Number of observations lost to lagging.
        fn lag_offset(&self) -> usize {
            self.max_effective_lag()
        }

        /// Build feature column names for a given TimeSeries.
        fn feature_names(&self, exog_names: &[String]) -> Vec<String> {
            let mut names = Vec::new();
            if self.use_trend {
                names.push("__trend".to_string());
            }
            for lag in self.effective_lags() {
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
            for lag in self.effective_lags() {
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
        ) -> Result<(
            Mat<f64>,
            Col<f64>,
            usize,
            Vec<String>,
            Vec<FittedComponentState>,
        )> {
            let values = series.primary_values();
            let n = values.len();
            let offset = self.lag_offset();

            if n <= offset {
                return Err(ForecastError::InsufficientData {
                    needed: offset + 2,
                    got: n,
                    hint: Some(format!(
                        "need > {} observations for lags {:?}",
                        offset,
                        self.effective_lags()
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

            // Lags: y[t-k] for each specified lag k
            for lag in self.effective_lags() {
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
                fitted_components.push(FittedComponentState::Structural { fill_values: fill });
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
            for lag in self.effective_lags() {
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
    struct FittedState {
        /// The fitted regression model (any backend).
        model: Box<dyn FittedRegressor + Send>,
        /// Feature configuration used.
        features: RegressionFeatures,
        /// Number of observations in the full series (before differencing).
        n_total: usize,
        /// Last `max_lag` values for recursive prediction (from differenced series).
        tail_values: Vec<f64>,
        /// In-sample fitted values (full length, NaN-padded for lags/differencing).
        fitted_values: Vec<f64>,
        /// In-sample residuals (full length, NaN-padded for lags/differencing).
        residuals: Vec<f64>,
        /// Exogenous regressor names (sorted).
        exog_names: Vec<String>,
        /// Fitted trend/seasonal components for generating future feature columns.
        components: Vec<FittedComponentState>,
        /// Original series values (stored when differencing is used, for integration).
        original_values: Option<Vec<f64>>,
    }

    impl std::fmt::Debug for FittedState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("FittedState")
                .field("n_total", &self.n_total)
                .field(
                    "n_features",
                    &self.features.feature_names(&self.exog_names).len(),
                )
                .field("r_squared", &self.model.r_squared())
                .finish()
        }
    }

    // ── RegressionForecaster ────────────────────────────────────────

    /// A forecasting model backed by an external regression estimator.
    ///
    /// Wraps regression estimators from `anofox-regression` behind the
    /// [`Forecaster`] trait, enabling them to participate in pipelines,
    /// registries, ensembles, and cross-validation.
    ///
    /// # Backends
    ///
    /// | Backend | Use case |
    /// |---|---|
    /// | [`Ols`](RegressionBackend::Ols) | Default — unregularized |
    /// | [`Ridge`](RegressionBackend::Ridge) | L2 regularization — many features |
    /// | [`ElasticNet`](RegressionBackend::ElasticNet) | L1+L2 — feature selection |
    /// | [`Quantile`](RegressionBackend::Quantile) | Conditional quantile estimation |
    /// | [`Wls`](RegressionBackend::Wls) | Observation weighting / recency |
    /// | [`Rls`](RegressionBackend::Rls) | Adaptive / online coefficients |
    /// | [`Tweedie`](RegressionBackend::Tweedie) | GLM for count / continuous data |
    /// | [`Poisson`](RegressionBackend::Poisson) | GLM for count data |
    /// | [`Bls`](RegressionBackend::Bls) | Box-constrained coefficients |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anofox_forecast::models::regression::{RegressionForecaster, RegressionFeatures};
    ///
    /// // OLS with trend + 3 lags
    /// let mut model = RegressionForecaster::ols(
    ///     RegressionFeatures::new().trend().lags(3),
    /// );
    ///
    /// // Ridge with Fourier seasonality
    /// let mut model = RegressionForecaster::ridge(
    ///     0.1,
    ///     RegressionFeatures::new().trend().fourier(7, 3),
    /// );
    /// ```
    #[derive(Debug)]
    pub struct RegressionForecaster {
        features: RegressionFeatures,
        backend: RegressionBackend,
        state: Option<FittedState>,
    }

    impl RegressionForecaster {
        /// Create a regression forecaster with the given backend and features.
        pub fn new(backend: RegressionBackend, features: RegressionFeatures) -> Self {
            Self {
                features,
                backend,
                state: None,
            }
        }

        /// Create a regression forecaster using OLS with the given features.
        pub fn ols(features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::Ols, features)
        }

        /// Create a Ridge regression forecaster (L2 regularization).
        pub fn ridge(lambda: f64, features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::Ridge { lambda }, features)
        }

        /// Create an Elastic Net forecaster (L1 + L2 regularization).
        ///
        /// `alpha` controls the L1/L2 mix: 0 = pure Ridge, 1 = pure Lasso.
        pub fn elastic_net(lambda: f64, alpha: f64, features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::ElasticNet { lambda, alpha }, features)
        }

        /// Create a quantile regression forecaster.
        ///
        /// `tau` is the quantile to estimate (0.5 = median).
        pub fn quantile(tau: f64, features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::Quantile { tau }, features)
        }

        /// Create a WLS forecaster with exponential decay weighting.
        ///
        /// `decay` controls recency emphasis (e.g., 0.95 = moderate, 0.99 = mild).
        pub fn wls_decay(decay: f64, features: RegressionFeatures) -> Self {
            Self::new(
                RegressionBackend::Wls {
                    strategy: WeightStrategy::ExponentialDecay(decay),
                },
                features,
            )
        }

        /// Create a WLS forecaster with custom weights.
        pub fn wls(weights: Vec<f64>, features: RegressionFeatures) -> Self {
            Self::new(
                RegressionBackend::Wls {
                    strategy: WeightStrategy::Custom(weights),
                },
                features,
            )
        }

        /// Create a Recursive Least Squares forecaster (adaptive coefficients).
        ///
        /// `forgetting_factor` controls adaptation speed (0 < λ ≤ 1).
        /// 1.0 = equal weights, 0.95 = moderate adaptation.
        pub fn rls(forgetting_factor: f64, features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::Rls { forgetting_factor }, features)
        }

        /// Create a Tweedie GLM forecaster.
        ///
        /// `var_power`: 0 = Gaussian, 1 = Poisson, 2 = Gamma, 3 = Inv-Gaussian.
        pub fn tweedie(var_power: f64, features: RegressionFeatures) -> Self {
            Self::new(
                RegressionBackend::Tweedie {
                    var_power,
                    link_power: None,
                },
                features,
            )
        }

        /// Create a Poisson GLM forecaster (for count data).
        pub fn poisson(features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::Poisson, features)
        }

        /// Create a Bounded Least Squares forecaster (box-constrained coefficients).
        pub fn bls(lower: Option<f64>, upper: Option<f64>, features: RegressionFeatures) -> Self {
            Self::new(RegressionBackend::Bls { lower, upper }, features)
        }

        /// Create a Non-Negative Least Squares forecaster (all coefficients ≥ 0).
        pub fn nnls(features: RegressionFeatures) -> Self {
            Self::bls(Some(0.0), None, features)
        }

        /// Create a Dynamic Linear Model forecaster (time-varying parameters).
        ///
        /// Automatically generates candidate models from variable subsets and
        /// computes observation-level IC weights. Use `lowess_span` to smooth
        /// the weights over time (e.g., 0.3), or `None` for no smoothing.
        pub fn dynamic(features: RegressionFeatures) -> Self {
            Self::new(
                RegressionBackend::Dynamic {
                    ic: InformationCriterion::AICc,
                    lowess_span: None,
                },
                features,
            )
        }

        /// Create a Dynamic Linear Model with LOWESS-smoothed weights.
        pub fn dynamic_smoothed(lowess_span: f64, features: RegressionFeatures) -> Self {
            Self::new(
                RegressionBackend::Dynamic {
                    ic: InformationCriterion::AICc,
                    lowess_span: Some(lowess_span),
                },
                features,
            )
        }

        // ── OLS convenience constructors (backward-compatible) ──────

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

        /// Get the backend configuration.
        pub fn backend(&self) -> &RegressionBackend {
            &self.backend
        }

        /// Get the regression result (coefficients, R², etc.) if fitted.
        pub fn fitted_result(&self) -> Option<&anofox_regression::core::RegressionResult> {
            self.state.as_ref().map(|s| s.model.result())
        }

        /// Get R² of the fitted model.
        pub fn r_squared(&self) -> Option<f64> {
            self.state.as_ref().map(|s| s.model.r_squared())
        }

        /// Apply configured differencing to a series.
        fn apply_differencing(&self, values: &[f64]) -> Vec<f64> {
            use crate::models::arima::{difference, seasonal_difference};

            let mut result = values.to_vec();

            // Seasonal differencing first (standard ARIMA convention), in order
            for &(d, period) in &self.features.seasonal_diffs {
                result = seasonal_difference(&result, d, period);
            }

            // Then regular differencing
            if self.features.diff_order > 0 {
                result = difference(&result, self.features.diff_order);
            }

            result
        }

        /// Integrate (undo differencing) forecast values back to original scale.
        fn apply_integration(&self, state: &FittedState, predictions: &[f64]) -> Vec<f64> {
            let original = match &state.original_values {
                Some(v) => v,
                None => return predictions.to_vec(),
            };

            use crate::models::arima::{integrate, seasonal_integrate};

            let mut result = predictions.to_vec();

            // Undo regular differencing first (reverse of application order)
            if self.features.diff_order > 0 {
                // The reference for regular integration is the original after all seasonal diffs
                let mut reference = original.clone();
                for &(d, period) in &self.features.seasonal_diffs {
                    reference = crate::models::arima::seasonal_difference(&reference, d, period);
                }
                result = integrate(&result, &reference, self.features.diff_order);
            }

            // Then undo seasonal differencing in reverse order
            for &(d, period) in self.features.seasonal_diffs.iter().rev() {
                // Reference for each level is original with all prior seasonal diffs applied
                // For the last applied (first undone), reference is original with all-but-last
                // For simplicity, use original — seasonal_integrate handles the seed correctly
                result = seasonal_integrate(&result, original, d, period);
            }

            result
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

            let eff_lags = state.features.effective_lags();
            if eff_lags.is_empty() {
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
                for (col_offset, &lag) in eff_lags.iter().enumerate() {
                    let col = trend_offset + col_offset;
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
            // State is not Clone (Box<dyn FittedRegressor>), so we only clone config
            Self {
                features: self.features.clone(),
                backend: self.backend.clone(),
                state: None,
            }
        }
    }

    impl Forecaster for RegressionForecaster {
        fn fit(&mut self, series: &TimeSeries) -> Result<()> {
            validate_series_complete(series)?;

            // Resolve auto-lag selection before anything else
            self.features.resolve_auto_lags(series)?;

            let values = series.primary_values();
            let n_original = values.len();

            // Apply differencing if configured
            let uses_diff =
                self.features.diff_order > 0 || !self.features.seasonal_diffs.is_empty();
            let original_values = if uses_diff {
                Some(values.to_vec())
            } else {
                None
            };

            let working_values = self.apply_differencing(values);

            // Build a temporary TimeSeries from differenced values for matrix construction
            let fit_series = if uses_diff {
                // Trim timestamps to match differenced length
                let diff_offset = n_original - working_values.len();
                let trimmed_ts = series.slice(diff_offset, n_original)?;
                // Replace values with differenced values
                TimeSeries::univariate(trimmed_ts.timestamps().to_vec(), working_values.clone())?
            } else {
                series.clone()
            };

            let n = fit_series.primary_values().len();
            let (x, y, n_train, exog_names, components) =
                self.features.build_matrices(&fit_series)?;

            // Fit via the configured backend
            let fitted = self.backend.fit_to(&x, &y).map_err(|e| {
                ForecastError::ComputationError(format!(
                    "{} fit failed: {}",
                    self.backend.name(),
                    e
                ))
            })?;

            // In-sample predictions (on differenced scale)
            let in_sample_preds = fitted.predict(&x);

            // Build full-length fitted values (NaN-padded for lag offset + differencing)
            let diff_offset = n_original - n;
            let lag_offset = self.features.lag_offset();
            let total_offset = diff_offset + lag_offset;
            let mut fitted_values = vec![f64::NAN; n_original];
            let mut residuals = vec![f64::NAN; n_original];
            let diff_values = fit_series.primary_values();
            for i in 0..n_train {
                fitted_values[total_offset + i] = in_sample_preds[i];
                residuals[total_offset + i] = diff_values[lag_offset + i] - in_sample_preds[i];
            }

            // Store tail values for recursive prediction (from differenced series)
            let tail_len = self.features.max_effective_lag().max(1);
            let tail_values =
                working_values[working_values.len().saturating_sub(tail_len)..].to_vec();

            self.state = Some(FittedState {
                model: fitted,
                features: self.features.clone(),
                n_total: n,
                tail_values,
                fitted_values,
                residuals,
                exog_names,
                components,
                original_values,
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
            let predictions = self.apply_integration(state, &predictions);
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
            if !state.features.effective_lags().is_empty() {
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

            let pred_result =
                state
                    .model
                    .predict_with_interval(&x_future, Some(IntervalType::Prediction), level);

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
            let predictions = self.apply_integration(state, &predictions);
            Ok(Forecast::from_values(predictions))
        }

        fn fitted_values(&self) -> Option<&[f64]> {
            self.state.as_ref().map(|s| s.fitted_values.as_slice())
        }

        fn residuals(&self) -> Option<&[f64]> {
            self.state.as_ref().map(|s| s.residuals.as_slice())
        }

        fn name(&self) -> &str {
            self.backend.name()
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
                    h,
                    lower[0][h],
                    forecast.primary()[h],
                );
                assert!(
                    forecast.primary()[h] < upper[0][h],
                    "point = {} should be < upper[{}] = {}",
                    forecast.primary()[h],
                    h,
                    upper[0][h],
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

            let r2 = model.r_squared().unwrap();
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
                    2.0 * t + 10.0 * (2.0 * std::f64::consts::PI * t / period as f64).sin() + 5.0
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
            let r2 = model.r_squared().unwrap();
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
                RegressionFeatures::new()
                    .trend()
                    .dummy_seasonal(7)
                    .no_exog(),
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
            let values: Vec<f64> = (0..n).map(|i| (0.05 * i as f64).exp()).collect();
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

            let r2 = model.r_squared().unwrap();
            assert!(
                r2 > 0.85,
                "R² should be high for dual-seasonal data, got {}",
                r2
            );

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
            let features = RegressionFeatures::new().trend().fourier(7, 2).no_exog();
            let classified = features.classify_features(&[]);
            // __trend + 4 Fourier columns = 5 total
            assert_eq!(classified.len(), 5);
            assert!(classified
                .iter()
                .all(|(_, s)| *s == FeatureSafety::Deterministic));
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
            assert_eq!(
                classified[0],
                ("__trend".into(), FeatureSafety::Deterministic)
            );
            assert_eq!(
                classified[1],
                ("__lag_1".into(), FeatureSafety::Deterministic)
            );
            assert_eq!(
                classified[3],
                ("__theilsen_trend".into(), FeatureSafety::DataDependent)
            );
            assert_eq!(
                classified[4],
                ("__fourier_p7_sin_1".into(), FeatureSafety::Deterministic)
            );
            assert_eq!(
                classified[6],
                ("__seasonal_12".into(), FeatureSafety::DataDependent)
            );
            assert_eq!(
                classified[7],
                ("__cp_step_1".into(), FeatureSafety::Structural)
            );
            assert_eq!(
                classified[8],
                ("__cp_step_2".into(), FeatureSafety::Structural)
            );
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
            assert_eq!(cols[0][0], 0.0); // before any CP
            assert_eq!(cols[0][10], 1.0); // after first CP
            assert_eq!(cols[0][49], 1.0); // still in regime 1
            assert_eq!(cols[0][50], 2.0); // after second CP
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
                    if i >= 30 {
                        base + 20.0
                    } else {
                        base
                    }
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

            let r2 = model.r_squared().unwrap();
            assert!(
                r2 > 0.99,
                "R² should be high with changepoint step, got {}",
                r2
            );

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
                    let regime = if i < 20 {
                        0.0
                    } else if i < 40 {
                        1.0
                    } else {
                        2.0
                    };
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
                assert!(
                    (v - 10.0).abs() < 1.0,
                    "predicted {} should be near 10.0",
                    v
                );
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
                    if i == 25 {
                        base + 100.0
                    } else {
                        base
                    }
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

        // ── Backend tests ───────────────────────────────────────────

        #[test]
        fn backend_ridge_fit_predict() {
            let ts = make_linear_ts(50);
            let mut model =
                RegressionForecaster::ridge(0.1, RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "Ridge");

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            // Ridge on clean linear data should track closely
            for (h, &pred) in forecast.primary().iter().enumerate() {
                let expected = 2.0 * (50 + h) as f64 + 10.0;
                assert_relative_eq!(pred, expected, epsilon = 2.0);
            }
        }

        #[test]
        fn backend_ridge_fourier() {
            let ts = make_seasonal_ts(100, 7);
            let mut model = RegressionForecaster::ridge(
                0.01,
                RegressionFeatures::new().trend().fourier(7, 3).no_exog(),
            );
            model.fit(&ts).unwrap();
            let r2 = model.r_squared().unwrap();
            assert!(r2 > 0.90, "Ridge R² = {} should be high", r2);
        }

        #[test]
        fn backend_elastic_net_fit_predict() {
            let ts = make_linear_ts(60);
            let mut model = RegressionForecaster::elastic_net(
                0.01,
                0.5,
                RegressionFeatures::new().trend().fourier(7, 2).no_exog(),
            );
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "ElasticNet");

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn backend_quantile_median() {
            let ts = make_linear_ts(60);
            let mut model =
                RegressionForecaster::quantile(0.5, RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "Quantile");

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            // Median of near-linear data should track the trend
            for (h, &pred) in forecast.primary().iter().enumerate() {
                let expected = 2.0 * (60 + h) as f64 + 10.0;
                assert_relative_eq!(pred, expected, epsilon = 3.0);
            }
        }

        #[test]
        fn backend_quantile_upper() {
            let ts = make_linear_ts(60);
            let mut m50 =
                RegressionForecaster::quantile(0.5, RegressionFeatures::new().trend().no_exog());
            let mut m90 =
                RegressionForecaster::quantile(0.9, RegressionFeatures::new().trend().no_exog());
            m50.fit(&ts).unwrap();
            m90.fit(&ts).unwrap();

            let f50 = m50.predict(1).unwrap();
            let f90 = m90.predict(1).unwrap();
            // For near-linear data, q90 ≥ q50 (approximately)
            assert!(
                f90.primary()[0] >= f50.primary()[0] - 1.0,
                "q90={} should be ≥ q50={}",
                f90.primary()[0],
                f50.primary()[0],
            );
        }

        #[test]
        fn backend_wls_decay() {
            let ts = make_linear_ts(60);
            let mut model =
                RegressionForecaster::wls_decay(0.95, RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "WLS");

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn backend_rls_fit_predict() {
            let ts = make_linear_ts(60);
            let mut model =
                RegressionForecaster::rls(0.99, RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "RLS");

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn backend_tweedie_gaussian() {
            // var_power=0 is Gaussian — should behave like OLS
            let ts = make_linear_ts(50);
            let mut model =
                RegressionForecaster::tweedie(0.0, RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "Tweedie");

            let forecast = model.predict(5).unwrap();
            for (h, &pred) in forecast.primary().iter().enumerate() {
                let expected = 2.0 * (50 + h) as f64 + 10.0;
                assert_relative_eq!(pred, expected, epsilon = 2.0);
            }
        }

        #[test]
        fn backend_poisson_count_data() {
            // y = exp(0.02*t + 1) — count-like data
            let n = 60;
            let values: Vec<f64> = (0..n)
                .map(|i| (0.02 * i as f64 + 1.0).exp().round().max(1.0))
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model =
                RegressionForecaster::poisson(RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "Poisson");

            let forecast = model.predict(5).unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite());
                assert!(v > 0.0, "Poisson should predict positive, got {}", v);
            }
        }

        #[test]
        fn backend_bls_nonnegative() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::nnls(RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "BLS");

            let forecast = model.predict(5).unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn backend_new_generic_constructor() {
            let ts = make_linear_ts(40);
            let mut model = RegressionForecaster::new(
                RegressionBackend::Ridge { lambda: 0.5 },
                RegressionFeatures::new().trend().no_exog(),
            );
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "Ridge");
            assert!(model.r_squared().unwrap() > 0.9);
        }

        #[test]
        fn backend_model_registry_integration() {
            use crate::models::{ModelRegistry, ModelSpec};

            let mut reg = ModelRegistry::new();
            reg.register(ModelSpec::new(
                "OLS",
                || Box::new(RegressionForecaster::linear_trend()),
                false,
            ));
            reg.register(ModelSpec::new(
                "Ridge(0.1)",
                || {
                    Box::new(RegressionForecaster::ridge(
                        0.1,
                        RegressionFeatures::new().trend().no_exog(),
                    ))
                },
                false,
            ));
            reg.register(ModelSpec::new(
                "Quantile(0.5)",
                || {
                    Box::new(RegressionForecaster::quantile(
                        0.5,
                        RegressionFeatures::new().trend().no_exog(),
                    ))
                },
                false,
            ));

            let ts = make_linear_ts(50);
            for spec in reg.iter() {
                let mut model = spec.create();
                model.fit(&ts).unwrap();
                if !model.has_exog() {
                    let fc = model.predict(5).unwrap();
                    assert_eq!(fc.primary().len(), 5);
                }
            }
        }

        #[test]
        fn backend_dynamic_fit_predict() {
            let ts = make_linear_ts(60);
            let mut model =
                RegressionForecaster::dynamic(RegressionFeatures::new().trend().no_exog());
            model.fit(&ts).unwrap();
            assert_eq!(model.name(), "Dynamic");

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn backend_dynamic_smoothed() {
            let ts = make_seasonal_ts(100, 7);
            let mut model = RegressionForecaster::dynamic_smoothed(
                0.3,
                RegressionFeatures::new().trend().fourier(7, 2).no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        // ── Differencing tests ───────────────────────────────────────

        #[test]
        fn differencing_d1_produces_finite_forecast() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().trend().differencing(1).no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Forecast should be finite, got {}", v);
            }
        }

        #[test]
        fn differencing_d1_continues_trend() {
            // Linear trend: values 1..=30
            let ts = make_linear_ts(30);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().trend().differencing(1).no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();

            // After d=1 differencing of a linear trend, the differenced series
            // is constant. Integration should continue the trend roughly.
            let last = ts.primary_values().last().copied().unwrap();
            for &v in forecast.primary() {
                assert!(
                    v > last * 0.5,
                    "Forecast {} should continue trend from {}",
                    v,
                    last
                );
            }
        }

        #[test]
        fn differencing_d1_with_lags() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .differencing(1)
                    .lags(2)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn seasonal_differencing_produces_finite_forecast() {
            let ts = make_seasonal_ts(60, 7);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .seasonal_differencing(1, 7)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Forecast should be finite, got {}", v);
            }
        }

        #[test]
        fn both_differencing_and_seasonal() {
            let ts = make_seasonal_ts(80, 7);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .differencing(1)
                    .seasonal_differencing(1, 7)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn multiple_seasonal_differencing_periods() {
            // Daily data with weekly seasonality (simplified — use enough observations)
            let n = 120;
            let ts = make_seasonal_ts(n, 7);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .seasonal_differencing(1, 7)
                    .seasonal_differencing(1, 14)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Forecast should be finite, got {}", v);
            }
        }

        #[test]
        fn seasonal_differencing_chains_not_overwrites() {
            let features = RegressionFeatures::new()
                .seasonal_differencing(1, 7)
                .seasonal_differencing(1, 365);
            assert_eq!(features.seasonal_diffs.len(), 2);
            assert_eq!(features.seasonal_diffs[0], (1, 7));
            assert_eq!(features.seasonal_diffs[1], (1, 365));
        }

        #[test]
        fn no_differencing_is_default() {
            let features = RegressionFeatures::new();
            assert_eq!(features.diff_order, 0);
            assert!(features.seasonal_diffs.is_empty());
        }

        // ── Auto-lag selection tests ─────────────────────────────────

        #[test]
        fn auto_lags_selects_reasonable_order() {
            // AR(2) process: y[t] = 0.5*y[t-1] + 0.3*y[t-2] + noise
            let mut values = vec![0.0; 100];
            values[0] = 1.0;
            values[1] = 0.5;
            for i in 2..100 {
                let noise = ((i * 7 + 3) % 11) as f64 * 0.02 - 0.11;
                values[i] = 0.5 * values[i - 1] + 0.3 * values[i - 2] + noise;
            }
            let ts = TimeSeries::univariate(make_timestamps(100), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().no_trend().auto_lags(10).no_exog(),
            );
            model.fit(&ts).unwrap();

            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn auto_lags_with_aic() {
            let ts = make_linear_ts(50);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .auto_lags_with(5, LagSelectionCriterion::Aic)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        #[test]
        fn auto_lags_max_zero_selects_no_lags() {
            let ts = make_linear_ts(50);
            let mut model =
                RegressionForecaster::ols(RegressionFeatures::new().trend().auto_lags(0).no_exog());
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
        }

        #[test]
        fn auto_lags_default_is_none() {
            let features = RegressionFeatures::new();
            assert!(features.auto_lag_config.is_none());
        }

        #[test]
        fn auto_lags_with_differencing() {
            let ts = make_linear_ts(60);
            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .auto_lags(5)
                    .differencing(1)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();
            assert_eq!(forecast.primary().len(), 5);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        // ── Extended auto-lag selection tests ────────────────────────

        #[test]
        fn auto_lags_ar1_selects_order_ge_1() {
            // Pure AR(1): y[t] = 0.8 * y[t-1] + noise
            let n = 200;
            let mut values = vec![0.0_f64; n];
            values[0] = 1.0;
            for i in 1..n {
                let noise = ((i * 13 + 7) % 17) as f64 * 0.01 - 0.085;
                values[i] = 0.8 * values[i - 1] + noise;
            }
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().no_trend().auto_lags(10).no_exog(),
            );
            model.fit(&ts).unwrap();

            assert!(
                model.features().max_lag >= 1,
                "AR(1) process: auto_lags should select order >= 1, got {}",
                model.features().max_lag
            );
        }

        #[test]
        fn auto_lags_ar3_selects_order_ge_3() {
            // Pure AR(3): y[t] = 0.5*y[t-1] + 0.2*y[t-2] + 0.15*y[t-3] + noise
            let n = 300;
            let mut values = vec![0.0_f64; n];
            values[0] = 1.0;
            values[1] = 0.5;
            values[2] = 0.7;
            for i in 3..n {
                let noise = ((i * 11 + 5) % 19) as f64 * 0.005 - 0.0475;
                values[i] =
                    0.5 * values[i - 1] + 0.2 * values[i - 2] + 0.15 * values[i - 3] + noise;
            }
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().no_trend().auto_lags(10).no_exog(),
            );
            model.fit(&ts).unwrap();

            assert!(
                model.features().max_lag >= 3,
                "AR(3) process: auto_lags should select order >= 3, got {}",
                model.features().max_lag
            );
        }

        #[test]
        fn auto_lags_white_noise_selects_low_order() {
            // White noise (no autocorrelation): BIC should select a low lag order.
            // Use a deterministic PRNG (xorshift-style) to avoid sequential patterns.
            let n = 200;
            let mut state: u64 = 123456789;
            let values: Vec<f64> = (0..n)
                .map(|_| {
                    // xorshift64
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // Map to [-1, 1]
                    (state as f64 / u64::MAX as f64) * 2.0 - 1.0
                })
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().no_trend().auto_lags(10).no_exog(),
            );
            model.fit(&ts).unwrap();

            // For white noise, BIC should select 0 or a very low order (<=2).
            // The exact result depends on the PRNG realization, but it should
            // not select a high order like 8-10 on uncorrelated data.
            assert!(
                model.features().max_lag <= 2,
                "White noise: auto_lags should select a low order (<= 2), got {}",
                model.features().max_lag
            );
        }

        #[test]
        fn auto_lags_aic_selects_ge_bic() {
            // AIC penalizes less than BIC, so AIC should select >= BIC order.
            // Use an AR(2) process with moderate signal.
            let n = 200;
            let mut values = vec![0.0_f64; n];
            values[0] = 1.0;
            values[1] = 0.5;
            for i in 2..n {
                let noise = ((i * 7 + 3) % 11) as f64 * 0.02 - 0.11;
                values[i] = 0.5 * values[i - 1] + 0.3 * values[i - 2] + noise;
            }

            let ts_bic = TimeSeries::univariate(make_timestamps(n), values.clone()).unwrap();
            let ts_aic = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model_bic = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .auto_lags_with(10, LagSelectionCriterion::Bic)
                    .no_exog(),
            );
            model_bic.fit(&ts_bic).unwrap();

            let mut model_aic = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .auto_lags_with(10, LagSelectionCriterion::Aic)
                    .no_exog(),
            );
            model_aic.fit(&ts_aic).unwrap();

            let bic_order = model_bic.features().max_lag;
            let aic_order = model_aic.features().max_lag;
            assert!(
                aic_order >= bic_order,
                "AIC order ({}) should be >= BIC order ({})",
                aic_order,
                bic_order
            );
        }

        #[test]
        fn auto_lags_short_series_does_not_panic() {
            // Barely enough data: 6 observations, max_lag = 4
            let n = 6;
            let values: Vec<f64> = (0..n).map(|i| i as f64 * 2.0 + 1.0).collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().no_trend().auto_lags(4).no_exog(),
            );
            // Should not panic, even if it selects order 0
            model.fit(&ts).unwrap();
            let forecast = model.predict(3).unwrap();
            assert_eq!(forecast.primary().len(), 3);
            for &v in forecast.primary() {
                assert!(v.is_finite());
            }
        }

        // ── Extended differencing round-trip tests ───────────────────

        #[test]
        fn differencing_d1_then_integrate_continues_original_scale() {
            // Linear trend: 1, 2, 3, ..., 50
            let n = 50;
            let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values.clone()).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().trend().differencing(1).no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(5).unwrap();

            // After integration, forecast should be on the original scale (around 51-55)
            let last = *values.last().unwrap(); // 50.0
            for (h, &v) in forecast.primary().iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Forecast step {} should be finite, got {}",
                    h,
                    v
                );
                // Should be roughly continuing the trend (within 10% of expected)
                let expected = last + (h + 1) as f64;
                assert!(
                    (v - expected).abs() < expected * 0.2,
                    "Forecast step {}: expected ~{}, got {}",
                    h,
                    expected,
                    v
                );
            }
        }

        #[test]
        fn linear_trend_differencing_gives_constant_diffs() {
            // 1, 2, 3, ..., 50 — first difference should be all 1.0
            use crate::models::arima::difference;
            let values: Vec<f64> = (1..=50).map(|i| i as f64).collect();
            let diffed = difference(&values, 1);
            assert_eq!(diffed.len(), 49);
            for &d in &diffed {
                assert_relative_eq!(d, 1.0, epsilon = 1e-12);
            }
        }

        #[test]
        fn forecast_after_integration_is_on_original_scale() {
            // Exponential-ish growth: values in 100..~250 range
            let n = 60;
            let values: Vec<f64> = (0..n).map(|i| 100.0 + 2.5 * i as f64).collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values.clone()).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new().trend().differencing(1).no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(10).unwrap();

            let last = *values.last().unwrap(); // 100 + 2.5*59 = 247.5
            for &v in forecast.primary() {
                // Forecasts should be on the original scale (> 200), not on
                // the differenced scale (which would be around 2.5)
                assert!(
                    v > last * 0.5,
                    "Forecast {} should be on original scale (last train = {}), not differenced",
                    v,
                    last
                );
            }
        }

        // ── Extended seasonal differencing round-trip tests ──────────

        #[test]
        fn seasonal_diff_weekly_integration_recovers_scale() {
            // Weekly pattern repeating over 10 weeks
            let period = 7;
            let n_weeks = 10;
            let n = period * n_weeks;
            let weekly_pattern = [10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 8.0];
            let values: Vec<f64> = (0..n)
                .map(|i| weekly_pattern[i % period] + 0.5 * i as f64)
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values.clone()).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .seasonal_differencing(1, 7)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(7).unwrap();

            // Forecasts should be on the original scale, not the differenced scale
            let _last = *values.last().unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Seasonal forecast should be finite");
                // Original scale is roughly 10-80 range; differenced scale is ~3.5
                assert!(
                    v > 5.0,
                    "Forecast {} should be on original scale (not seasonal-differenced)",
                    v
                );
            }
        }

        #[test]
        fn seasonal_diff_monthly_integration_recovers_scale() {
            // Monthly pattern (period=12) with linear trend
            let period = 12;
            let n_years = 5;
            let n = period * n_years;
            let monthly_pattern = [
                5.0, 8.0, 12.0, 18.0, 22.0, 25.0, 24.0, 22.0, 18.0, 12.0, 8.0, 5.0,
            ];
            let values: Vec<f64> = (0..n)
                .map(|i| monthly_pattern[i % period] + 1.0 * i as f64)
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values.clone()).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .seasonal_differencing(1, 12)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(12).unwrap();

            let last = *values.last().unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Monthly seasonal forecast should be finite");
                // Should be on the original scale (last train ~ 64)
                assert!(
                    v > last * 0.3,
                    "Forecast {} should be on original scale (last train = {})",
                    v,
                    last
                );
            }
        }

        #[test]
        fn multiple_seasonal_diffs_chain_correctly() {
            // Two seasonal diffs: weekly (7) and bi-weekly (14)
            let features = RegressionFeatures::new()
                .seasonal_differencing(1, 7)
                .seasonal_differencing(1, 365);
            assert_eq!(features.seasonal_diffs.len(), 2);
            assert_eq!(features.seasonal_diffs[0], (1, 7));
            assert_eq!(features.seasonal_diffs[1], (1, 365));

            // Verify that chaining actually works end-to-end with a model
            // Use enough data for both periods (need > 365 + 7 for both diffs)
            let n = 400;
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let t = i as f64;
                    // Weekly component + small trend
                    5.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin() + 0.1 * t + 50.0
                })
                .collect();
            let ts = TimeSeries::univariate(make_timestamps(n), values).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .trend()
                    .seasonal_differencing(1, 7)
                    .seasonal_differencing(1, 14)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();
            let forecast = model.predict(7).unwrap();
            assert_eq!(forecast.primary().len(), 7);
            for &v in forecast.primary() {
                assert!(
                    v.is_finite(),
                    "Multi-seasonal forecast should be finite, got {}",
                    v
                );
            }
        }

        // ── Combined differencing + auto_lags tests ─────────────────

        #[test]
        fn differencing_d1_auto_lags_fits_on_differenced_data() {
            // Linear trend with AR structure: after differencing, should have lag structure
            let n = 100;
            let mut values = vec![0.0_f64; n];
            values[0] = 10.0;
            for i in 1..n {
                let noise = ((i * 13 + 7) % 17) as f64 * 0.01 - 0.085;
                values[i] = values[i - 1] + 2.0 + 0.3 * noise;
            }
            let ts = TimeSeries::univariate(make_timestamps(n), values.clone()).unwrap();

            let mut model = RegressionForecaster::ols(
                RegressionFeatures::new()
                    .no_trend()
                    .differencing(1)
                    .auto_lags(5)
                    .no_exog(),
            );
            model.fit(&ts).unwrap();

            // auto_lag_config is preserved for re-fit; max_lag is set to the selected order
            assert!(
                model.features().max_lag <= 5,
                "selected lag should be <= max_lag"
            );

            let forecast = model.predict(10).unwrap();
            assert_eq!(forecast.primary().len(), 10);

            let last = *values.last().unwrap();
            for &v in forecast.primary() {
                assert!(v.is_finite(), "Forecast should be finite");
                // Forecast should be on the original scale, not differenced
                assert!(
                    v > last * 0.5,
                    "Forecast {} should be on original scale (last train = {})",
                    v,
                    last
                );
            }
        }
    }
}

#[cfg(feature = "postprocess")]
pub use ols_impl::{
    ChangepointEncoding, ChangepointFeature, FeatureSafety, LagSelectionCriterion,
    RegressionBackend, RegressionFeatures, RegressionForecaster, SeasonalSpec, StructuralFeature,
    TrendType, WeightStrategy,
};
// Re-export InformationCriterion so users can configure Dynamic backend without
// depending on anofox-regression directly.
#[cfg(feature = "postprocess")]
pub use anofox_regression::solvers::InformationCriterion;
