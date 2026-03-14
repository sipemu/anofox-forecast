//! Data-aware metric strategy for model selection.
//!
//! Instead of hard-coding MAE as the sole selection metric, a
//! [`MetricStrategy`] selects and combines metrics based on the
//! characteristics of the data (e.g. intermittent vs. non-negative).

use std::fmt;

use crate::utils::metrics::{mae, mda, mse, rmse, smape, wape};

/// Individual metrics available for model scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    MAE,
    MSE,
    RMSE,
    SMAPE,
    WAPE,
    MDA,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Metric::MAE => "MAE",
            Metric::MSE => "MSE",
            Metric::RMSE => "RMSE",
            Metric::SMAPE => "SMAPE",
            Metric::WAPE => "WAPE",
            Metric::MDA => "MDA",
        };
        write!(f, "{}", s)
    }
}

/// How to select and combine metrics for model ranking.
#[derive(Debug, Clone)]
pub enum MetricStrategy {
    /// Automatically select based on data profile characteristics.
    /// - Intermittent → MAE(0.5) + SMAPE(0.5)
    /// - Non-negative → MAE(0.5) + WAPE(0.5)
    /// - General → MAE(0.4) + SMAPE(0.3) + MDA(0.3)
    Auto,
    /// Use a single metric for ranking.
    Single(Metric),
    /// Weighted combination of metrics. Weights are normalized internally.
    Composite(Vec<(Metric, f64)>),
}

impl Default for MetricStrategy {
    fn default() -> Self {
        MetricStrategy::Single(Metric::MAE)
    }
}

/// Scores for a single model on multiple metrics.
#[derive(Debug, Clone)]
pub struct MetricScores {
    /// Combined ranking score (lower is better, except MDA where higher is better).
    pub primary: f64,
    /// Individual metric values.
    pub components: Vec<(Metric, f64)>,
}

impl fmt::Display for MetricScores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={:.4} [", self.primary)?;
        for (i, (m, v)) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}={:.4}", m, v)?;
        }
        write!(f, "]")
    }
}

impl MetricStrategy {
    /// Resolve `Auto` into a concrete `Composite` based on data characteristics.
    pub fn resolve(&self, is_intermittent: bool, has_negatives: bool) -> Vec<(Metric, f64)> {
        match self {
            MetricStrategy::Auto => {
                if is_intermittent {
                    vec![(Metric::MAE, 0.5), (Metric::SMAPE, 0.5)]
                } else if !has_negatives {
                    vec![(Metric::MAE, 0.5), (Metric::WAPE, 0.5)]
                } else {
                    vec![(Metric::MAE, 0.4), (Metric::SMAPE, 0.3), (Metric::MDA, 0.3)]
                }
            }
            MetricStrategy::Single(m) => vec![(*m, 1.0)],
            MetricStrategy::Composite(pairs) => pairs.clone(),
        }
    }

    /// Compute individual metric values for actual vs predicted.
    pub fn compute_metric(metric: Metric, actual: &[f64], predicted: &[f64]) -> f64 {
        match metric {
            Metric::MAE => mae(actual, predicted),
            Metric::MSE => mse(actual, predicted),
            Metric::RMSE => rmse(actual, predicted),
            Metric::SMAPE => smape(actual, predicted),
            Metric::WAPE => wape(actual, predicted),
            Metric::MDA => mda(actual, predicted),
        }
    }

    /// Score a model's predictions against actuals using the resolved metric weights.
    ///
    /// For MDA (higher is better), the contribution is `weight * (1 - mda)` so
    /// that the composite score remains lower-is-better.
    pub fn score(
        &self,
        actual: &[f64],
        predicted: &[f64],
        is_intermittent: bool,
        has_negatives: bool,
    ) -> MetricScores {
        let weights = self.resolve(is_intermittent, has_negatives);
        let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();

        let mut primary = 0.0;
        let mut components = Vec::with_capacity(weights.len());

        for (metric, weight) in &weights {
            let raw = Self::compute_metric(*metric, actual, predicted);
            components.push((*metric, raw));

            let normalized_weight = if total_weight > 0.0 {
                weight / total_weight
            } else {
                0.0
            };

            // MDA is higher-is-better; invert it for the composite score
            let contribution = match metric {
                Metric::MDA => (1.0 - raw) * normalized_weight,
                _ => raw * normalized_weight,
            };
            primary += contribution;
        }

        MetricScores {
            primary,
            components,
        }
    }

    /// Format the resolved strategy for logging.
    pub fn description(&self, is_intermittent: bool, has_negatives: bool) -> String {
        let weights = self.resolve(is_intermittent, has_negatives);
        let parts: Vec<String> = weights
            .iter()
            .map(|(m, w)| format!("{}({:.0}%)", m, w * 100.0))
            .collect();
        parts.join(" + ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_actual() -> Vec<f64> {
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    }

    fn sample_predicted() -> Vec<f64> {
        vec![12.0, 18.0, 33.0, 38.0, 52.0, 58.0, 72.0]
    }

    #[test]
    fn auto_intermittent() {
        let strat = MetricStrategy::Auto;
        let resolved = strat.resolve(true, false);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].0, Metric::MAE);
        assert_eq!(resolved[1].0, Metric::SMAPE);
    }

    #[test]
    fn auto_non_negative() {
        let strat = MetricStrategy::Auto;
        let resolved = strat.resolve(false, false);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].0, Metric::MAE);
        assert_eq!(resolved[1].0, Metric::WAPE);
    }

    #[test]
    fn auto_general() {
        let strat = MetricStrategy::Auto;
        let resolved = strat.resolve(false, true);
        assert_eq!(resolved.len(), 3);
    }

    #[test]
    fn single_metric_score() {
        let strat = MetricStrategy::Single(Metric::MAE);
        let scores = strat.score(&sample_actual(), &sample_predicted(), false, false);
        let expected = mae(&sample_actual(), &sample_predicted());
        assert!((scores.primary - expected).abs() < 1e-10);
        assert_eq!(scores.components.len(), 1);
        assert_eq!(scores.components[0].0, Metric::MAE);
    }

    #[test]
    fn composite_score_lower_is_better() {
        let strat = MetricStrategy::Composite(vec![(Metric::MAE, 0.5), (Metric::SMAPE, 0.5)]);
        let good = strat.score(&sample_actual(), &sample_predicted(), false, false);

        // Worse predictions should have higher score
        let bad_pred = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let bad = strat.score(&sample_actual(), &bad_pred, false, false);
        assert!(bad.primary > good.primary);
    }

    #[test]
    fn mda_inverted_in_composite() {
        // MDA is higher-is-better, so composite uses (1 - MDA)
        let strat = MetricStrategy::Single(Metric::MDA);
        let scores = strat.score(&sample_actual(), &sample_predicted(), false, false);
        let raw_mda = mda(&sample_actual(), &sample_predicted());
        assert!((scores.primary - (1.0 - raw_mda)).abs() < 1e-10);
    }

    #[test]
    fn description_format() {
        let strat = MetricStrategy::Auto;
        let desc = strat.description(false, true);
        assert!(desc.contains("MAE"));
        assert!(desc.contains("SMAPE"));
        assert!(desc.contains("MDA"));
    }

    #[test]
    fn metric_display() {
        assert_eq!(format!("{}", Metric::MAE), "MAE");
        assert_eq!(format!("{}", Metric::WAPE), "WAPE");
    }

    #[test]
    fn metric_scores_display() {
        let scores = MetricScores {
            primary: 2.5,
            components: vec![(Metric::MAE, 3.0), (Metric::SMAPE, 5.2)],
        };
        let text = format!("{}", scores);
        assert!(text.contains("score=2.5000"));
        assert!(text.contains("MAE=3.0000"));
    }
}
