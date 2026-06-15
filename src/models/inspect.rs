//! Fit-state inspection trait + per-model typed payloads.
//!
//! `Inspectable` is the typed sibling to [`Explainable`](super::explain::Explainable).
//! Where `Explainable` decomposes a *forecast* into level / trend /
//! seasonal so callers can reconstruct the prediction, `Inspectable`
//! packages the *fit*: the model's internal state and interpretable
//! parameters as a single per-fit snapshot consumers can serialise,
//! cache, and surface to UIs.
//!
//! Issue #107. Builds on the primitive accessors from #106
//! (`fitted_values`, `trend_component`, `seasonal_component`,
//! `residual_component`, `training_values`).
//!
//! # Usage
//!
//! ```ignore
//! use anofox_forecast::models::{Forecaster, Inspectable, Explanation};
//!
//! let mut model = AutoETS::new();
//! model.fit(&ts)?;
//! match model.explanation()? {
//!     Explanation::Ets(e) => println!("spec {} alpha {:.3}", e.spec, e.alpha),
//!     _ => unreachable!(),
//! }
//! ```

use crate::error::Result;

/// Snapshot of a model's most recent fit.
///
/// Each variant carries the universal spine (fitted values, residuals)
/// plus model-specific scalars and structural components. Variants are
/// owned (no borrows) so `Explanation` can be serialised, cached, and
/// sent across process boundaries.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Explanation {
    /// Regression models (`RegressionForecaster`, OLS / Ridge / WLS / etc.).
    Regression(RegressionExplanation),
    /// ETS family (`SimpleES`, `HoltLinear`, `HoltWinters`, `SeasonalES`, `AutoETS`).
    Ets(EtsExplanation),
    /// ARIMA family (`AutoARIMA`).
    Arima(ArimaExplanation),
    /// MFLES (the boosted MFLES forecaster).
    Mfles(MflesExplanation),
    /// Theta family (`AutoTheta`, `DynamicTheta`, `OptimizedTheta`).
    Theta(ThetaExplanation),
    /// TBATS family (`AutoTBATS`).
    Tbats(TbatsExplanation),
    /// MSTL (`MSTLForecaster`).
    Mstl(MstlExplanation),
}

/// Interpretable state for regression-backed forecasters.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegressionExplanation {
    /// Named feature columns in the order they appear in `coefficients`.
    pub feature_names: Vec<String>,
    /// Coefficients aligned with `feature_names`.
    pub coefficients: Vec<f64>,
    /// Intercept term (separated out from `coefficients`).
    pub intercept: f64,
    /// In-sample R² (1 − RSS / TSS).
    pub r_squared: f64,
    /// Backend identifier, e.g. `"ols"`, `"ridge"`, `"wls_logistic"`.
    pub backend: String,
    /// Per-coefficient standard errors, aligned with `coefficients`.
    /// Empty when the backend doesn't compute inference (e.g. Poisson,
    /// Tweedie, BLS, RLS). Nominal under regularised backends (Ridge,
    /// ElasticNet) and weighted backends (WLS) — useful for display but
    /// don't read them as exact frequentist SEs.
    pub coef_std_errors: Vec<f64>,
    /// Standard error of the intercept. `NaN` when not available.
    pub intercept_std_error: f64,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
}

/// Interpretable state for ETS-family forecasters.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EtsExplanation {
    /// ETS spec identifier (e.g. `"AAA"`, `"ANN"`, `"MAM"`).
    pub spec: String,
    /// Level smoothing parameter.
    pub alpha: f64,
    /// Trend smoothing parameter; `None` for level-only specs.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter; `None` for non-seasonal specs.
    pub gamma: Option<f64>,
    /// Damping parameter for damped-trend specs.
    pub phi: Option<f64>,
    /// Seasonal period (0 for non-seasonal fits).
    pub seasonal_period: usize,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// Trend component (from the #106 accessor). Same length as
    /// `fitted_values` or shorter (warmup-dropped).
    pub trend_component: Vec<f64>,
    /// Seasonal component if available; `None` for non-seasonal fits.
    pub seasonal_component: Option<Vec<f64>>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
}

/// Interpretable state for ARIMA / SARIMA forecasters.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArimaExplanation {
    /// Non-seasonal order `(p, d, q)`.
    pub order: (usize, usize, usize),
    /// Seasonal order `(P, D, Q, s)` if present.
    pub seasonal_order: Option<(usize, usize, usize, usize)>,
    /// Fitted AR/MA coefficients.
    pub coefficients: Vec<f64>,
    /// AIC of the selected model.
    pub aic: f64,
    /// BIC of the selected model.
    pub bic: f64,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
}

/// Interpretable state for MFLES.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MflesExplanation {
    /// Seasonal period (0 for non-seasonal).
    pub seasonal_period: usize,
    /// Number of boosting rounds actually executed.
    pub max_rounds: usize,
    /// Whether multiplicative mode was used.
    pub multiplicative: bool,
    /// Trend penalty (R²-based shrinkage), if computed.
    pub penalty: Option<f64>,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// Per-row trend component.
    pub trend_component: Vec<f64>,
    /// Per-row seasonal contribution (defined as `fitted - trend` so the
    /// invariant `trend + seasonal + residual = training` holds in both
    /// additive and multiplicative modes).
    pub seasonal_component: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
}

/// Interpretable state for Theta-family forecasters.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThetaExplanation {
    /// Selected Theta variant identifier (e.g. `"StandardTheta"`,
    /// `"OptimizedTheta"`, `"DynamicStandardTheta"`).
    pub variant: String,
    /// Theta parameter (default 2.0 in the standard model).
    pub theta: f64,
    /// Level smoothing parameter; `None` if the variant doesn't expose it.
    pub alpha: Option<f64>,
    /// Seasonal period (0 for non-seasonal fits).
    pub seasonal_period: usize,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
}

/// Interpretable state for TBATS.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TbatsExplanation {
    /// Seasonal periods used in the fit.
    pub seasonal_periods: Vec<usize>,
    /// Box-Cox λ if the transformation was applied.
    pub box_cox_lambda: Option<f64>,
    /// Description of the selected configuration (trend / damping / Box-Cox flags).
    pub selected_config: String,
    /// AIC of the selected model.
    pub aic: f64,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
}

/// Interpretable state for MSTL.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MstlExplanation {
    /// Seasonal periods used in the decomposition.
    pub seasonal_periods: Vec<usize>,
    /// Number of MSTL iterations.
    pub iterations: usize,
    /// In-sample fitted values.
    pub fitted_values: Vec<f64>,
    /// Per-row trend component (STL trend).
    pub trend_component: Vec<f64>,
    /// Per-row sum of all seasonal components.
    pub seasonal_component: Vec<f64>,
    /// Per-row STL remainder.
    pub residuals: Vec<f64>,
}

/// Fit-state inspection trait.
///
/// Models that implement this expose a typed [`Explanation`] snapshot
/// of their most recent fit. Not every forecaster implements this —
/// baselines (`Naive`, `RandomWalkDrift`, …) and the intermittent-demand
/// family don't have interpretable structure worth packaging.
///
/// `Box<dyn Inspectable>` works because `Explanation` is owned (no
/// borrows in any variant).
pub trait Inspectable {
    /// Snapshot of the most recent fit.
    ///
    /// # Errors
    /// Returns `Err(FitRequired)` if `fit()` hasn't been called yet, or
    /// `Err(InvalidParameter)` if the model lacks the state needed to
    /// build the snapshot.
    fn explanation(&self) -> Result<Explanation>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explanation_variants_are_owned_and_clonable() {
        // Smoke test: variants are owned, no lifetime parameters, so
        // they Clone trivially.
        let e = Explanation::Regression(RegressionExplanation {
            feature_names: vec!["x".into()],
            coefficients: vec![1.0],
            intercept: 0.5,
            r_squared: 0.9,
            backend: "ols".into(),
            coef_std_errors: vec![0.05],
            intercept_std_error: 0.02,
            fitted_values: vec![1.0, 2.0],
            residuals: vec![0.1, -0.1],
        });
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn regression_explanation_round_trips_through_serde_json() {
        let e = RegressionExplanation {
            feature_names: vec!["intercept".into(), "lag_1".into()],
            coefficients: vec![1.5, 0.8],
            intercept: 1.5,
            r_squared: 0.92,
            backend: "ridge".into(),
            coef_std_errors: vec![0.10, 0.04],
            intercept_std_error: 0.07,
            fitted_values: vec![1.0, 2.0, 3.0],
            residuals: vec![0.0, 0.1, -0.1],
        };
        let json = serde_json::to_string(&e).unwrap();
        let e2: RegressionExplanation = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn explanation_enum_round_trips_through_serde_json() {
        let e = Explanation::Ets(EtsExplanation {
            spec: "AAA".into(),
            alpha: 0.3,
            beta: Some(0.1),
            gamma: Some(0.2),
            phi: None,
            seasonal_period: 12,
            fitted_values: vec![10.0; 5],
            trend_component: vec![1.0; 5],
            seasonal_component: Some(vec![0.5; 5]),
            residuals: vec![0.0; 5],
        });
        let json = serde_json::to_string(&e).unwrap();
        let e2: Explanation = serde_json::from_str(&json).unwrap();
        assert_eq!(e, e2);
    }
}
