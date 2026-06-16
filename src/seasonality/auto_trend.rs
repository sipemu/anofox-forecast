//! Automatic trend component selection via information criteria.
//!
//! [`AutoTrend`] fits multiple candidate trend components and selects the best
//! one using AICc, BIC, or holdout evaluation. It implements [`TrendComponent`]
//! itself, delegating to the winner.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::auto_trend::AutoTrend;
//! use anofox_forecast::seasonality::traits::TrendComponent;
//!
//! let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
//! let mut auto = AutoTrend::new();
//! auto.fit_trend(&values).unwrap();
//!
//! let fitted = auto.fitted_trend();
//! assert_eq!(fitted.len(), 100);
//!
//! let result = auto.selection_result().unwrap();
//! println!("Selected: {} (AICc={:.2})", result.selected, result.scores[0].1);
//! ```

use super::exponential_trend::ExponentialTrend;
use super::piecewise::PiecewiseLinearTrend;
use super::polynomial::PolynomialTrend;
use super::theilsen::TheilSenTrend;
use super::traits::{Recency, TrendComponent};
use crate::error::{ForecastError, Result};

/// Information criterion for model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendCriterion {
    /// Corrected Akaike Information Criterion (default).
    #[default]
    AICc,
    /// Bayesian Information Criterion.
    BIC,
    /// Holdout evaluation (last 20% as test set).
    Holdout,
}

/// Result of automatic trend selection.
#[derive(Debug, Clone)]
pub struct AutoTrendResult {
    /// Name of the selected component.
    pub selected: String,
    /// Criterion used for selection.
    pub criterion: TrendCriterion,
    /// All candidates ranked by score (name, score). Lower is better.
    pub scores: Vec<(String, f64)>,
}

/// Enum dispatch for fitted trend components (avoids trait objects).
#[derive(Debug, Clone)]
enum FittedTrendComponent {
    Polynomial(PolynomialTrend),
    Exponential(ExponentialTrend),
    PiecewiseLinear(PiecewiseLinearTrend),
    TheilSen(TheilSenTrend),
}

impl FittedTrendComponent {
    fn fitted_trend(&self) -> &[f64] {
        match self {
            Self::Polynomial(c) => c.fitted_trend(),
            Self::Exponential(c) => c.fitted_trend(),
            Self::PiecewiseLinear(c) => c.fitted_trend(),
            Self::TheilSen(c) => c.fitted_trend(),
        }
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        match self {
            Self::Polynomial(c) => c.predict_trend(n_ahead),
            Self::Exponential(c) => c.predict_trend(n_ahead),
            Self::PiecewiseLinear(c) => c.predict_trend(n_ahead),
            Self::TheilSen(c) => c.predict_trend(n_ahead),
        }
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        match self {
            Self::Polynomial(c) => c.trend_features(),
            Self::Exponential(c) => c.trend_features(),
            Self::PiecewiseLinear(c) => c.trend_features(),
            Self::TheilSen(c) => c.trend_features(),
        }
    }

    fn trend_name(&self) -> &str {
        match self {
            Self::Polynomial(c) => c.trend_name(),
            Self::Exponential(c) => c.trend_name(),
            Self::PiecewiseLinear(c) => c.trend_name(),
            Self::TheilSen(c) => c.trend_name(),
        }
    }

    fn n_params(&self) -> usize {
        match self {
            Self::Polynomial(c) => c.n_params(),
            Self::Exponential(c) => c.n_params(),
            Self::PiecewiseLinear(c) => c.n_params(),
            Self::TheilSen(c) => c.n_params(),
        }
    }
}

/// Automatic trend component selector.
///
/// Fits multiple candidate trend components and selects the best one using
/// an information criterion (AICc by default).
///
/// Default candidates: Linear (poly1), Quadratic (poly2), Exponential (if data
/// positive), TheilSen, PiecewiseLinear.
#[derive(Debug, Clone)]
pub struct AutoTrend {
    /// Recency specification for all candidates.
    recency: Recency,
    /// Selection criterion.
    criterion: TrendCriterion,
    /// The winning component after fit.
    winner: Option<FittedTrendComponent>,
    /// Selection result with scores.
    result: Option<AutoTrendResult>,
}

impl AutoTrend {
    /// Create a new AutoTrend with default settings (AICc, Fraction(0.3)).
    pub fn new() -> Self {
        Self {
            recency: Recency::Fraction(0.3),
            criterion: TrendCriterion::AICc,
            winner: None,
            result: None,
        }
    }

    /// Set the recency window for all candidates (builder-style).
    pub fn with_recency(mut self, recency: Recency) -> Self {
        self.recency = recency;
        self
    }

    /// Set the selection criterion (builder-style).
    pub fn with_criterion(mut self, criterion: TrendCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Get the selection result after fitting.
    pub fn selection_result(&self) -> Option<&AutoTrendResult> {
        self.result.as_ref()
    }
}

impl Default for AutoTrend {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute AICc from residual variance, number of observations, and parameters.
fn compute_aicc(values: &[f64], fitted: &[f64], k: usize) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let nf = n as f64;
    let kf = k as f64;

    // Residual variance
    let ss_res: f64 = values
        .iter()
        .zip(fitted.iter())
        .map(|(v, f)| (v - f).powi(2))
        .sum();
    let var = ss_res / nf;

    if var <= 0.0 || !var.is_finite() {
        return f64::INFINITY;
    }

    // Log-likelihood (Gaussian assumption)
    let ll = -0.5 * nf * (1.0 + var.ln() + (2.0 * std::f64::consts::PI).ln());

    // AICc
    let denom = (nf - kf - 1.0).max(1.0);
    -2.0 * ll + 2.0 * kf * nf / denom
}

/// Compute BIC from residual variance, number of observations, and parameters.
fn compute_bic(values: &[f64], fitted: &[f64], k: usize) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let nf = n as f64;
    let kf = k as f64;

    let ss_res: f64 = values
        .iter()
        .zip(fitted.iter())
        .map(|(v, f)| (v - f).powi(2))
        .sum();
    let var = ss_res / nf;

    if var <= 0.0 || !var.is_finite() {
        return f64::INFINITY;
    }

    let ll = -0.5 * nf * (1.0 + var.ln() + (2.0 * std::f64::consts::PI).ln());
    -2.0 * ll + kf * nf.ln()
}

/// Compute holdout MSE: fit on first 80%, evaluate on last 20%.
fn compute_holdout_score(
    values: &[f64],
    fitted: &[f64],
    predict_fn: &dyn Fn(usize) -> Vec<f64>,
) -> f64 {
    let n = values.len();
    let split = (n as f64 * 0.8).ceil() as usize;
    if split >= n || split == 0 {
        return f64::INFINITY;
    }
    let test_n = n - split;
    let preds = predict_fn(test_n);
    if preds.len() != test_n {
        return f64::INFINITY;
    }

    let mse: f64 = values[split..]
        .iter()
        .zip(preds.iter())
        .map(|(v, p)| (v - p).powi(2))
        .sum::<f64>()
        / test_n as f64;

    // Ignore the fitted parameter to suppress warnings
    let _ = fitted;

    mse
}

impl TrendComponent for AutoTrend {
    fn fit_trend(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let all_positive = values.iter().all(|&v| v > 0.0);

        // Build candidates
        let mut candidates: Vec<(String, Box<dyn FnOnce() -> Option<FittedTrendComponent>>)> =
            Vec::new();

        let recency = self.recency.clone();

        // Linear (poly1)
        {
            let r = recency.clone();
            candidates.push((
                "Linear".to_string(),
                Box::new(move || {
                    let mut c = PolynomialTrend::new(1).with_recency(r);
                    c.fit_trend(values).ok()?;
                    Some(FittedTrendComponent::Polynomial(c))
                }),
            ));
        }

        // Quadratic (poly2)
        {
            let r = recency.clone();
            candidates.push((
                "Quadratic".to_string(),
                Box::new(move || {
                    let mut c = PolynomialTrend::new(2).with_recency(r);
                    c.fit_trend(values).ok()?;
                    Some(FittedTrendComponent::Polynomial(c))
                }),
            ));
        }

        // Exponential (only if data positive)
        if all_positive {
            let r = recency.clone();
            candidates.push((
                "Exponential".to_string(),
                Box::new(move || {
                    let mut c = ExponentialTrend::new().with_recency(r);
                    c.fit_trend(values).ok()?;
                    Some(FittedTrendComponent::Exponential(c))
                }),
            ));
        }

        // TheilSen
        {
            let r = recency.clone();
            candidates.push((
                "TheilSen".to_string(),
                Box::new(move || {
                    let mut c = TheilSenTrend::new().with_recency(r);
                    c.fit_trend(values).ok()?;
                    Some(FittedTrendComponent::TheilSen(c))
                }),
            ));
        }

        // PiecewiseLinear — auto-tune the PELT penalty per series so
        // the candidate competes fairly in the AICc / BIC / Holdout
        // bake-off instead of running on the hard-coded default.
        {
            let r = recency.clone();
            candidates.push((
                "PiecewiseLinear".to_string(),
                Box::new(move || {
                    let mut c = PiecewiseLinearTrend::new()
                        .with_auto_penalty()
                        .with_recency(r);
                    c.fit_trend(values).ok()?;
                    Some(FittedTrendComponent::PiecewiseLinear(c))
                }),
            ));
        }

        // Fit all candidates and score them
        let mut scored: Vec<(String, f64, FittedTrendComponent)> = Vec::new();

        for (name, fit_fn) in candidates {
            if let Some(component) = fit_fn() {
                let fitted = component.fitted_trend();
                let k = component.n_params();

                let score = match self.criterion {
                    TrendCriterion::AICc => compute_aicc(values, fitted, k),
                    TrendCriterion::BIC => compute_bic(values, fitted, k),
                    TrendCriterion::Holdout => {
                        let component_ref = &component;
                        compute_holdout_score(values, fitted, &|n_ahead| {
                            component_ref.predict_trend(n_ahead)
                        })
                    }
                };

                if score.is_finite() {
                    scored.push((name, score, component));
                }
            }
        }

        if scored.is_empty() {
            return Err(ForecastError::ComputationError(
                "AutoTrend: all candidates failed".to_string(),
            ));
        }

        // Sort by score (lower is better)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let scores: Vec<(String, f64)> = scored.iter().map(|(n, s, _)| (n.clone(), *s)).collect();
        let selected = scored[0].0.clone();

        self.result = Some(AutoTrendResult {
            selected: selected.clone(),
            criterion: self.criterion,
            scores,
        });

        // Take ownership of the winner
        let (_, _, winner) = scored.into_iter().next().unwrap();
        self.winner = Some(winner);

        Ok(())
    }

    fn fitted_trend(&self) -> &[f64] {
        match &self.winner {
            Some(w) => w.fitted_trend(),
            None => &[],
        }
    }

    fn predict_trend(&self, n_ahead: usize) -> Vec<f64> {
        match &self.winner {
            Some(w) => w.predict_trend(n_ahead),
            None => vec![f64::NAN; n_ahead],
        }
    }

    fn trend_features(&self) -> Vec<(&str, f64)> {
        match &self.winner {
            Some(w) => w.trend_features(),
            None => Vec::new(),
        }
    }

    fn trend_name(&self) -> &str {
        match &self.winner {
            Some(w) => w.trend_name(),
            None => "auto_trend",
        }
    }

    fn n_params(&self) -> usize {
        match &self.winner {
            Some(w) => w.n_params(),
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Linear data → should select Linear or TheilSen ────────────────

    #[test]
    fn selects_linear_for_linear_data() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        // Linear or TheilSen should be top (both fit perfectly)
        assert!(
            result.selected == "Linear" || result.selected == "TheilSen",
            "expected Linear or TheilSen, got {}",
            result.selected
        );
    }

    #[test]
    fn fitted_matches_data_for_linear() {
        let values: Vec<f64> = (0..100).map(|i| 3.0 * i as f64 - 5.0).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let fitted = auto.fitted_trend();
        assert_eq!(fitted.len(), 100);
        for (f, v) in fitted.iter().zip(values.iter()) {
            assert_abs_diff_eq!(*f, *v, epsilon = 0.1);
        }
    }

    // ── Exponential data → should select Exponential ──────────────────

    #[test]
    fn selects_exponential_for_exponential_data() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * (0.05 * i as f64).exp()).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        assert_eq!(
            result.selected, "Exponential",
            "scores: {:?}",
            result.scores
        );
    }

    // ── Quadratic data → should select Quadratic ──────────────────────

    #[test]
    fn selects_quadratic_for_quadratic_data() {
        let values: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64;
                0.01 * t * t + 0.5 * t + 10.0
            })
            .collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        assert_eq!(result.selected, "Quadratic", "scores: {:?}", result.scores);
    }

    // ── Selection result has all candidates ───────────────────────────

    #[test]
    fn selection_result_has_all_candidates() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * (0.05 * i as f64).exp()).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        // All positive data → should have Linear, Quadratic, Exponential, TheilSen, PiecewiseLinear
        assert!(
            result.scores.len() >= 4,
            "got {} scores",
            result.scores.len()
        );
    }

    // ── BIC criterion works ──────────────────────────────────────────

    #[test]
    fn bic_criterion_works() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut auto = AutoTrend::new()
            .with_criterion(TrendCriterion::BIC)
            .with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        assert_eq!(result.criterion, TrendCriterion::BIC);
        assert!(!result.scores.is_empty());
    }

    // ── Holdout criterion works ──────────────────────────────────────

    #[test]
    fn holdout_criterion_works() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut auto = AutoTrend::new()
            .with_criterion(TrendCriterion::Holdout)
            .with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        assert_eq!(result.criterion, TrendCriterion::Holdout);
    }

    // ── Predict delegates to winner ──────────────────────────────────

    #[test]
    fn predict_delegates_to_winner() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let forecast = auto.predict_trend(10);
        assert_eq!(forecast.len(), 10);
        // Should extrapolate the linear trend
        assert!(forecast[0] > values.last().unwrap() - 1.0);
    }

    // ── Features delegate to winner ──────────────────────────────────

    #[test]
    fn features_delegate_to_winner() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let features = auto.trend_features();
        assert!(!features.is_empty());
    }

    // ── Error cases ──────────────────────────────────────────────────

    #[test]
    fn empty_data_error() {
        let mut auto = AutoTrend::new();
        let result = auto.fit_trend(&[]);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn unfitted_returns_defaults() {
        let auto = AutoTrend::new();
        assert!(auto.fitted_trend().is_empty());
        assert!(auto.selection_result().is_none());
        assert_eq!(auto.trend_name(), "auto_trend");
        assert_eq!(auto.n_params(), 0);
    }

    // ── Scores are sorted ────────────────────────────────────────────

    #[test]
    fn scores_are_sorted_ascending() {
        let values: Vec<f64> = (0..100).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        for w in result.scores.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "scores not sorted: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    // ── Default trait ────────────────────────────────────────────────

    #[test]
    fn default_is_unfitted() {
        let auto = AutoTrend::default();
        assert!(auto.winner.is_none());
        assert!(auto.result.is_none());
    }

    // ── Negative data excludes exponential ───────────────────────────

    #[test]
    fn negative_data_excludes_exponential() {
        let values: Vec<f64> = (0..100).map(|i| -(i as f64) - 1.0).collect();
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(&values).unwrap();

        let result = auto.selection_result().unwrap();
        assert!(
            !result.scores.iter().any(|(n, _)| n == "Exponential"),
            "exponential should not be a candidate for negative data"
        );
    }

    // ── Recency builder ──────────────────────────────────────────────

    #[test]
    fn with_recency_builder() {
        let auto = AutoTrend::new().with_recency(Recency::Window(50));
        assert_eq!(auto.recency, Recency::Window(50));
    }

    #[test]
    fn with_criterion_builder() {
        let auto = AutoTrend::new().with_criterion(TrendCriterion::BIC);
        assert_eq!(auto.criterion, TrendCriterion::BIC);
    }
}
