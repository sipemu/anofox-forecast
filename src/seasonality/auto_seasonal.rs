//! Automatic seasonal component selection via information criteria.
//!
//! [`AutoSeasonal`] fits multiple candidate seasonal components and selects
//! the best one using AICc, BIC, or holdout evaluation.
//!
//! # Example
//!
//! ```
//! use anofox_forecast::seasonality::auto_seasonal::AutoSeasonal;
//! use anofox_forecast::seasonality::traits::SeasonalComponent;
//!
//! let values: Vec<f64> = (0..48).map(|i| (i % 12) as f64 * 2.0 + 10.0).collect();
//! let mut auto = AutoSeasonal::new();
//! auto.fit_seasonal(&values, 12).unwrap();
//!
//! let result = auto.selection_result().unwrap();
//! println!("Selected: {}", result.selected);
//! ```

use super::dummy::DummySeasonality;
use super::seasonal_diff::SeasonalDifference;
use super::traits::SeasonalComponent;
use crate::error::{ForecastError, Result};

/// Information criterion for seasonal model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeasonalCriterion {
    /// Corrected Akaike Information Criterion (default).
    #[default]
    AICc,
    /// Bayesian Information Criterion.
    BIC,
}

/// Result of automatic seasonal selection.
#[derive(Debug, Clone)]
pub struct AutoSeasonalResult {
    /// Name of the selected component.
    pub selected: String,
    /// Criterion used for selection.
    pub criterion: SeasonalCriterion,
    /// All candidates ranked by score (name, score). Lower is better.
    pub scores: Vec<(String, f64)>,
}

/// Enum dispatch for fitted seasonal components.
#[derive(Debug, Clone)]
enum FittedSeasonalComponent {
    Dummy(DummySeasonality),
    Difference(SeasonalDifference),
}

impl FittedSeasonalComponent {
    fn fitted_seasonal(&self) -> &[f64] {
        match self {
            Self::Dummy(c) => c.fitted_seasonal(),
            Self::Difference(c) => c.fitted_seasonal(),
        }
    }

    fn predict_seasonal(&self, n_ahead: usize) -> Vec<f64> {
        match self {
            Self::Dummy(c) => c.predict_seasonal(n_ahead),
            Self::Difference(c) => c.predict_seasonal(n_ahead),
        }
    }

    fn seasonal_features(&self) -> Vec<(&str, f64)> {
        match self {
            Self::Dummy(c) => c.seasonal_features(),
            Self::Difference(c) => c.seasonal_features(),
        }
    }

    fn seasonal_name(&self) -> &str {
        match self {
            Self::Dummy(c) => c.seasonal_name(),
            Self::Difference(c) => c.seasonal_name(),
        }
    }

    fn n_params(&self) -> usize {
        match self {
            Self::Dummy(c) => c.n_params(),
            Self::Difference(c) => c.n_params(),
        }
    }
}

/// Automatic seasonal component selector.
///
/// Fits `DummySeasonality` and `SeasonalDifference` and selects the best one.
#[derive(Debug, Clone)]
pub struct AutoSeasonal {
    criterion: SeasonalCriterion,
    winner: Option<FittedSeasonalComponent>,
    result: Option<AutoSeasonalResult>,
}

impl AutoSeasonal {
    /// Create a new AutoSeasonal with default settings (AICc).
    pub fn new() -> Self {
        Self {
            criterion: SeasonalCriterion::AICc,
            winner: None,
            result: None,
        }
    }

    /// Set the selection criterion (builder-style).
    pub fn with_criterion(mut self, criterion: SeasonalCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Get the selection result after fitting.
    pub fn selection_result(&self) -> Option<&AutoSeasonalResult> {
        self.result.as_ref()
    }
}

impl Default for AutoSeasonal {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute AICc from residual variance.
fn compute_aicc(values: &[f64], fitted: &[f64], k: usize) -> f64 {
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
    let denom = (nf - kf - 1.0).max(1.0);
    -2.0 * ll + 2.0 * kf * nf / denom
}

/// Compute BIC.
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

impl SeasonalComponent for AutoSeasonal {
    fn fit_seasonal(&mut self, values: &[f64], period: usize) -> Result<()> {
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let mut scored: Vec<(String, f64, FittedSeasonalComponent)> = Vec::new();

        // Candidate 1: DummySeasonality
        {
            let mut c = DummySeasonality::new();
            if c.fit_seasonal(values, period).is_ok() {
                let fitted = c.fitted_seasonal();
                let k = c.n_params();
                let score = match self.criterion {
                    SeasonalCriterion::AICc => compute_aicc(values, fitted, k),
                    SeasonalCriterion::BIC => compute_bic(values, fitted, k),
                };
                if score.is_finite() {
                    scored.push((
                        "DummySeasonality".to_string(),
                        score,
                        FittedSeasonalComponent::Dummy(c),
                    ));
                }
            }
        }

        // Candidate 2: SeasonalDifference
        {
            if let Ok(mut c) = SeasonalDifference::new(period) {
                if c.fit_seasonal(values, period).is_ok() {
                    let fitted = c.fitted_seasonal();
                    let k = c.n_params();
                    let score = match self.criterion {
                        SeasonalCriterion::AICc => compute_aicc(values, fitted, k),
                        SeasonalCriterion::BIC => compute_bic(values, fitted, k),
                    };
                    if score.is_finite() {
                        scored.push((
                            "SeasonalDifference".to_string(),
                            score,
                            FittedSeasonalComponent::Difference(c),
                        ));
                    }
                }
            }
        }

        if scored.is_empty() {
            return Err(ForecastError::ComputationError(
                "AutoSeasonal: all candidates failed".to_string(),
            ));
        }

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let scores: Vec<(String, f64)> = scored.iter().map(|(n, s, _)| (n.clone(), *s)).collect();
        let selected = scored[0].0.clone();

        self.result = Some(AutoSeasonalResult {
            selected,
            criterion: self.criterion,
            scores,
        });

        let (_, _, winner) = scored.into_iter().next().unwrap();
        self.winner = Some(winner);

        Ok(())
    }

    fn fitted_seasonal(&self) -> &[f64] {
        match &self.winner {
            Some(w) => w.fitted_seasonal(),
            None => &[],
        }
    }

    fn predict_seasonal(&self, n_ahead: usize) -> Vec<f64> {
        match &self.winner {
            Some(w) => w.predict_seasonal(n_ahead),
            None => Vec::new(),
        }
    }

    fn seasonal_features(&self) -> Vec<(&str, f64)> {
        match &self.winner {
            Some(w) => w.seasonal_features(),
            None => Vec::new(),
        }
    }

    fn seasonal_name(&self) -> &str {
        match &self.winner {
            Some(w) => w.seasonal_name(),
            None => "auto_seasonal",
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

    fn seasonal_data(period: usize, n_cycles: usize) -> Vec<f64> {
        (0..period * n_cycles)
            .map(|i| (i % period) as f64 * 3.0 + 10.0)
            .collect()
    }

    #[test]
    fn selects_a_component() {
        let values = seasonal_data(4, 10);
        let mut auto = AutoSeasonal::new();
        auto.fit_seasonal(&values, 4).unwrap();

        let result = auto.selection_result().unwrap();
        assert!(!result.selected.is_empty());
        assert!(!result.scores.is_empty());
    }

    #[test]
    fn fitted_same_length_as_input() {
        let values = seasonal_data(4, 10);
        let mut auto = AutoSeasonal::new();
        auto.fit_seasonal(&values, 4).unwrap();

        assert_eq!(auto.fitted_seasonal().len(), values.len());
    }

    #[test]
    fn predict_returns_values() {
        let values = seasonal_data(4, 10);
        let mut auto = AutoSeasonal::new();
        auto.fit_seasonal(&values, 4).unwrap();

        let forecast = auto.predict_seasonal(8);
        assert_eq!(forecast.len(), 8);
    }

    #[test]
    fn features_non_empty() {
        let values = seasonal_data(4, 10);
        let mut auto = AutoSeasonal::new();
        auto.fit_seasonal(&values, 4).unwrap();

        assert!(!auto.seasonal_features().is_empty());
    }

    #[test]
    fn bic_criterion_works() {
        let values = seasonal_data(4, 10);
        let mut auto = AutoSeasonal::new().with_criterion(SeasonalCriterion::BIC);
        auto.fit_seasonal(&values, 4).unwrap();

        let result = auto.selection_result().unwrap();
        assert_eq!(result.criterion, SeasonalCriterion::BIC);
    }

    #[test]
    fn empty_data_error() {
        let mut auto = AutoSeasonal::new();
        let result = auto.fit_seasonal(&[], 4);
        assert!(matches!(result, Err(ForecastError::EmptyData)));
    }

    #[test]
    fn unfitted_defaults() {
        let auto = AutoSeasonal::new();
        assert!(auto.fitted_seasonal().is_empty());
        assert!(auto.selection_result().is_none());
        assert_eq!(auto.seasonal_name(), "auto_seasonal");
    }

    #[test]
    fn scores_sorted_ascending() {
        let values = seasonal_data(4, 10);
        let mut auto = AutoSeasonal::new();
        auto.fit_seasonal(&values, 4).unwrap();

        let result = auto.selection_result().unwrap();
        for w in result.scores.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }
}
