//! Statistical confidence in model selection.
//!
//! Uses the Diebold-Mariano test for pairwise forecast accuracy comparison,
//! the Model Confidence Set (MCS) for identifying the set of best models,
//! and the Superior Predictive Ability (SPA) test for quality floor checks.
//!
//! Backed by the [`anofox_statistics`] crate.

use std::fmt;

use anofox_statistics::{
    diebold_mariano, model_confidence_set, spa_test, Alternative, LossFunction, MCSStatistic,
    VarEstimator,
};

// ---------------------------------------------------------------------------
// SelectionVerdict
// ---------------------------------------------------------------------------

/// How confident we are that the selected model is genuinely the best.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionVerdict {
    /// Best model is statistically significantly better (DM p < 0.05).
    SignificantWinner,
    /// Best model is better but the difference is marginal (0.05 <= p < 0.20).
    MarginalWinner,
    /// Models are effectively equivalent (DM p >= 0.20).
    Indistinguishable,
}

impl fmt::Display for SelectionVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SelectionVerdict::SignificantWinner => "SignificantWinner",
            SelectionVerdict::MarginalWinner => "MarginalWinner",
            SelectionVerdict::Indistinguishable => "Indistinguishable",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// SelectionConfidence (pairwise, Diebold-Mariano)
// ---------------------------------------------------------------------------

/// Pairwise confidence analysis using the Diebold-Mariano test.
///
/// Compares two models' per-fold CV errors to determine whether the best
/// model is statistically significantly more accurate than the runner-up.
#[derive(Debug, Clone)]
pub struct SelectionConfidence {
    /// Name of the best model (lowest mean score).
    pub best_model: String,
    /// Mean metric value for the best model (lower is better).
    pub best_score: f64,
    /// Name of the runner-up model.
    pub runner_up_model: String,
    /// Mean metric value for the runner-up.
    pub runner_up_score: f64,
    /// Score gap: `best_score - runner_up_score` (negative means best is better).
    pub score_gap: f64,
    /// Relative gap: `|gap| / runner_up_score`.
    pub relative_gap: f64,
    /// Statistical verdict derived from the DM test.
    pub verdict: SelectionVerdict,
    /// Diebold-Mariano test statistic.
    pub dm_statistic: f64,
    /// Diebold-Mariano p-value (one-sided: is best *lower* than runner-up?).
    pub dm_p_value: f64,
    /// Per-fold errors for the best model.
    pub best_fold_scores: Vec<f64>,
    /// Per-fold errors for the runner-up model.
    pub runner_up_fold_scores: Vec<f64>,
}

impl SelectionConfidence {
    /// Compare two models using their per-fold CV errors.
    ///
    /// `best_errors` and `runner_up_errors` are per-fold metric values (lower
    /// is better, e.g., MAE or MASE). They must have the same length.
    ///
    /// Uses the Diebold-Mariano test with absolute error loss and a one-sided
    /// alternative (best < runner-up).
    pub fn compare(
        best_name: impl Into<String>,
        best_errors: Vec<f64>,
        runner_up_name: impl Into<String>,
        runner_up_errors: Vec<f64>,
    ) -> Self {
        let best_name = best_name.into();
        let runner_up_name = runner_up_name.into();

        let best_mean = mean(&best_errors);
        let runner_up_mean = mean(&runner_up_errors);
        let gap = best_mean - runner_up_mean;
        let relative_gap = if runner_up_mean.abs() > f64::EPSILON {
            gap.abs() / runner_up_mean.abs()
        } else {
            0.0
        };

        // Run the Diebold-Mariano test.
        // DM takes forecast *errors* (actual - forecast), but here we already
        // have loss values per fold (e.g. MAE per fold). We treat them as
        // pre-computed errors and use AbsoluteError loss so the DM loss
        // differential is simply |e1| - |e2| = e1 - e2 (since inputs are
        // already absolute errors / losses).
        //
        // One-sided test: Alternative::Less tests whether model 1 has
        // *lower* loss than model 2.
        let (dm_stat, dm_p, verdict) = match diebold_mariano(
            &best_errors,
            &runner_up_errors,
            LossFunction::AbsoluteError,
            1,
            Alternative::Less,
            VarEstimator::Acf,
        ) {
            Ok(result) => {
                let v = p_to_verdict(result.p_value);
                (result.statistic, result.p_value, v)
            }
            Err(_) => {
                // DM requires >= 3 observations; fall back to simple comparison
                let v = if best_errors.len() < 2 {
                    SelectionVerdict::Indistinguishable
                } else if gap < -f64::EPSILON {
                    SelectionVerdict::MarginalWinner
                } else {
                    SelectionVerdict::Indistinguishable
                };
                (f64::NAN, f64::NAN, v)
            }
        };

        Self {
            best_model: best_name,
            best_score: best_mean,
            runner_up_model: runner_up_name,
            runner_up_score: runner_up_mean,
            score_gap: gap,
            relative_gap,
            verdict,
            dm_statistic: dm_stat,
            dm_p_value: dm_p,
            best_fold_scores: best_errors,
            runner_up_fold_scores: runner_up_errors,
        }
    }

    /// Whether the verdict is [`SelectionVerdict::SignificantWinner`].
    pub fn is_significant(&self) -> bool {
        self.verdict == SelectionVerdict::SignificantWinner
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        if self.dm_p_value.is_nan() {
            format!(
                "{} (mean={:.4}) vs {} (mean={:.4}): gap={:.4} ({:.1}% relative) — {} (DM: insufficient data)",
                self.best_model, self.best_score, self.runner_up_model, self.runner_up_score,
                self.score_gap, self.relative_gap * 100.0, self.verdict,
            )
        } else {
            format!(
                "{} (mean={:.4}) vs {} (mean={:.4}): gap={:.4} ({:.1}% relative) — {} (DM: stat={:.3}, p={:.4})",
                self.best_model, self.best_score, self.runner_up_model, self.runner_up_score,
                self.score_gap, self.relative_gap * 100.0, self.verdict,
                self.dm_statistic, self.dm_p_value,
            )
        }
    }
}

impl fmt::Display for SelectionConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// ModelConfidenceSet
// ---------------------------------------------------------------------------

/// The Model Confidence Set: the subset of models that contains the best
/// model at a given confidence level.
///
/// Unlike pairwise comparison, MCS simultaneously considers all models and
/// sequentially eliminates those that are significantly worse until only the
/// statistically indistinguishable "best" models remain.
#[derive(Debug, Clone)]
pub struct ModelConfidenceSet {
    /// Names of models included in the confidence set.
    pub included: Vec<String>,
    /// Names of models eliminated, in elimination order.
    pub eliminated: Vec<String>,
    /// MCS p-value (p-value when elimination stopped).
    pub mcs_p_value: f64,
    /// Significance level used.
    pub alpha: f64,
}

impl ModelConfidenceSet {
    /// Compute the Model Confidence Set from per-fold CV loss values.
    ///
    /// `model_scores` is a list of `(model_name, per_fold_losses)`. All fold
    /// vectors must have the same length. Returns `None` if fewer than two
    /// models are provided.
    ///
    /// Uses the Range statistic with 1000 bootstrap samples at the given
    /// `alpha` (typically 0.10 or 0.15).
    pub fn from_cv_scores(model_scores: Vec<(String, Vec<f64>)>, alpha: f64) -> Option<Self> {
        if model_scores.len() < 2 {
            return None;
        }

        let names: Vec<String> = model_scores.iter().map(|(n, _)| n.clone()).collect();
        let losses: Vec<Vec<f64>> = model_scores.into_iter().map(|(_, s)| s).collect();

        match model_confidence_set(
            &losses,
            alpha,
            MCSStatistic::Range,
            1000,
            0.0, // automatic block length
            Some(42),
        ) {
            Ok(result) => {
                let included = result
                    .included_models
                    .iter()
                    .map(|&i| names[i].clone())
                    .collect();
                let eliminated = result
                    .eliminated_models
                    .iter()
                    .map(|&i| names[i].clone())
                    .collect();

                Some(Self {
                    included,
                    eliminated,
                    mcs_p_value: result.mcs_p_value,
                    alpha,
                })
            }
            Err(_) => None,
        }
    }

    /// Number of models in the confidence set.
    pub fn len(&self) -> usize {
        self.included.len()
    }

    /// Whether the confidence set is empty.
    pub fn is_empty(&self) -> bool {
        self.included.is_empty()
    }

    /// Whether the set contains a single clear winner.
    pub fn has_single_winner(&self) -> bool {
        self.included.len() == 1
    }
}

impl fmt::Display for ModelConfidenceSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MCS (alpha={:.2}): {} model(s) in set {:?}, eliminated {:?} (p={:.4})",
            self.alpha,
            self.included.len(),
            self.included,
            self.eliminated,
            self.mcs_p_value,
        )
    }
}

// ---------------------------------------------------------------------------
// QualityFloor (SPA test)
// ---------------------------------------------------------------------------

/// Quality floor check using the Superior Predictive Ability (SPA) test.
///
/// Tests whether at least one candidate model significantly outperforms a
/// benchmark (typically Naive). Unlike a simple point comparison, SPA
/// accounts for data snooping bias when testing multiple models against a
/// single benchmark.
#[derive(Debug, Clone)]
pub struct QualityFloor {
    /// Name of the benchmark model.
    pub benchmark_name: String,
    /// SPA consistent p-value.
    pub spa_p_value: f64,
    /// SPA upper p-value (more conservative).
    pub spa_p_value_upper: f64,
    /// Whether at least one model significantly outperforms the benchmark
    /// (consistent p-value < 0.05).
    pub is_outperformed: bool,
    /// Index of the best alternative model (if any outperforms).
    pub best_alternative_idx: Option<usize>,
}

impl QualityFloor {
    /// Run the SPA test.
    ///
    /// `benchmark_losses` are per-observation losses from the benchmark.
    /// `model_losses` are per-observation losses from each candidate model.
    /// All vectors must have the same length.
    pub fn test(
        benchmark_name: impl Into<String>,
        benchmark_losses: &[f64],
        model_losses: &[Vec<f64>],
    ) -> Option<Self> {
        if model_losses.is_empty() || benchmark_losses.is_empty() {
            return None;
        }

        match spa_test(benchmark_losses, model_losses, 1000, 0.0, Some(42)) {
            Ok(result) => Some(Self {
                benchmark_name: benchmark_name.into(),
                spa_p_value: result.p_value_consistent,
                spa_p_value_upper: result.p_value_upper,
                is_outperformed: result.p_value_consistent < 0.05,
                best_alternative_idx: result.best_model_idx,
            }),
            Err(_) => None,
        }
    }
}

impl fmt::Display for QualityFloor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quality floor (vs {}): {} (SPA p={:.4})",
            self.benchmark_name,
            if self.is_outperformed {
                "PASSED"
            } else {
                "FAILED"
            },
            self.spa_p_value,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn p_to_verdict(p: f64) -> SelectionVerdict {
    if p < 0.05 {
        SelectionVerdict::SignificantWinner
    } else if p < 0.20 {
        SelectionVerdict::MarginalWinner
    } else {
        SelectionVerdict::Indistinguishable
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dm_significant_winner() {
        // Best model clearly lower on every fold.
        let best = vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.97];
        let runner_up = vec![3.0, 3.2, 2.8, 3.1, 3.0, 2.9, 3.15, 3.05, 2.95, 3.1];

        let conf = SelectionConfidence::compare("ModelA", best, "ModelB", runner_up);

        assert_eq!(conf.verdict, SelectionVerdict::SignificantWinner);
        assert!(conf.is_significant());
        assert!(conf.best_score < conf.runner_up_score);
        assert!(conf.score_gap < 0.0);
        assert!(!conf.dm_statistic.is_nan());
        assert!(conf.dm_p_value < 0.05);
    }

    #[test]
    fn dm_indistinguishable() {
        // Nearly identical scores — noise dominates.
        let best = vec![2.00, 2.01, 1.99, 2.00, 2.01, 1.99, 2.00, 2.01, 1.99, 2.00];
        let runner_up = vec![2.01, 2.00, 2.00, 2.01, 2.00, 2.00, 2.01, 2.00, 2.00, 2.01];

        let conf = SelectionConfidence::compare("ModelA", best, "ModelB", runner_up);

        assert!(!conf.is_significant());
    }

    #[test]
    fn dm_fallback_on_too_few_folds() {
        // Only 2 folds — DM requires >= 3, should not panic.
        let best = vec![1.0, 1.1];
        let runner_up = vec![3.0, 3.2];

        let conf = SelectionConfidence::compare("A", best, "B", runner_up);

        assert!(conf.dm_statistic.is_nan());
        // Should still recognise the gap direction.
        assert_eq!(conf.verdict, SelectionVerdict::MarginalWinner);
    }

    #[test]
    fn relative_gap_correct() {
        let best = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let runner_up = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

        let conf = SelectionConfidence::compare("A", best, "B", runner_up);

        assert!((conf.score_gap - (-1.0)).abs() < 1e-10);
        assert!((conf.relative_gap - 0.5).abs() < 1e-10);
    }

    #[test]
    fn display_contains_dm() {
        let best = vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.97];
        let runner_up = vec![3.0, 3.2, 2.8, 3.1, 3.0, 2.9, 3.15, 3.05, 2.95, 3.1];

        let conf = SelectionConfidence::compare("ModelA", best, "ModelB", runner_up);
        let display = format!("{}", conf);

        assert!(display.contains("ModelA"));
        assert!(display.contains("ModelB"));
        assert!(display.contains("DM:"));
    }

    #[test]
    fn verdict_display() {
        assert_eq!(
            format!("{}", SelectionVerdict::SignificantWinner),
            "SignificantWinner"
        );
        assert_eq!(
            format!("{}", SelectionVerdict::MarginalWinner),
            "MarginalWinner"
        );
        assert_eq!(
            format!("{}", SelectionVerdict::Indistinguishable),
            "Indistinguishable"
        );
    }

    // ── MCS tests ────────────────────────────────────────────────────────

    #[test]
    fn mcs_clearly_inferior_eliminated() {
        let scores = vec![
            ("Good".into(), vec![1.0; 50]),
            ("Bad".into(), vec![10.0; 50]),
        ];

        let mcs = ModelConfidenceSet::from_cv_scores(scores, 0.10).unwrap();

        assert!(mcs.included.contains(&"Good".to_string()));
        assert!(mcs.eliminated.contains(&"Bad".to_string()));
        assert!(mcs.has_single_winner());
    }

    #[test]
    fn mcs_equivalent_models_all_included() {
        let base: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.1).sin().abs() + 1.0)
            .collect();
        let scores = vec![
            ("A".into(), base.clone()),
            ("B".into(), base.clone()),
            ("C".into(), base.clone()),
        ];

        let mcs = ModelConfidenceSet::from_cv_scores(scores, 0.10).unwrap();

        assert_eq!(mcs.len(), 3);
        assert!(mcs.eliminated.is_empty());
        assert!(!mcs.has_single_winner());
    }

    #[test]
    fn mcs_single_model_returns_none() {
        let scores = vec![("Only".into(), vec![1.0, 2.0, 3.0])];

        assert!(ModelConfidenceSet::from_cv_scores(scores, 0.10).is_none());
    }

    #[test]
    fn mcs_display() {
        let scores = vec![("A".into(), vec![1.0; 50]), ("B".into(), vec![10.0; 50])];
        let mcs = ModelConfidenceSet::from_cv_scores(scores, 0.10).unwrap();
        let text = format!("{}", mcs);

        assert!(text.contains("MCS"));
        assert!(text.contains("alpha=0.10"));
    }

    // ── QualityFloor tests ───────────────────────────────────────────────

    #[test]
    fn quality_floor_outperformed() {
        let benchmark = vec![10.0; 50];
        let models = vec![vec![1.0; 50]];

        let qf = QualityFloor::test("Naive", &benchmark, &models).unwrap();

        assert!(qf.is_outperformed);
        assert!(qf.spa_p_value < 0.05);
        assert_eq!(qf.best_alternative_idx, Some(0));
    }

    #[test]
    fn quality_floor_not_outperformed() {
        // All models have similar losses.
        let base: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.1).sin().abs() + 1.0)
            .collect();
        let model: Vec<f64> = base.iter().map(|x| x + 0.001).collect();
        let models = vec![model];

        let qf = QualityFloor::test("Naive", &base, &models).unwrap();

        assert!(!qf.is_outperformed);
    }

    #[test]
    fn quality_floor_display() {
        let qf = QualityFloor {
            benchmark_name: "Naive".into(),
            spa_p_value: 0.02,
            spa_p_value_upper: 0.03,
            is_outperformed: true,
            best_alternative_idx: Some(0),
        };
        let text = format!("{}", qf);

        assert!(text.contains("PASSED"));
        assert!(text.contains("Naive"));
    }

    #[test]
    fn quality_floor_empty_returns_none() {
        assert!(QualityFloor::test("Naive", &[], &[]).is_none());
    }
}
