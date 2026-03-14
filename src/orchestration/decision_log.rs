use std::fmt;
use std::time::Duration;

/// A single decision made during the forecasting pipeline.
#[derive(Debug, Clone)]
pub struct Decision {
    /// Sequential step number.
    pub step: usize,
    /// Category of the decision.
    pub category: DecisionCategory,
    /// What was done, e.g. "Fitted ARIMA(1,1,1)".
    pub action: String,
    /// Outcome of the decision.
    pub outcome: DecisionOutcome,
    /// Extra info, e.g. "MASE=0.82".
    pub detail: Option<String>,
    /// How long the step took.
    pub duration: Option<Duration>,
}

/// Category of a pipeline decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionCategory {
    DataProfiling,
    Preprocessing,
    ModelSelection,
    ModelFitting,
    CrossValidation,
    Ensembling,
    Postprocessing,
    Constraint,
    Fallback,
    TrendSelection,
    SeasonalSelection,
    ChangepointAdaptation,
}

/// Outcome of a pipeline decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionOutcome {
    Success,
    Failed,
    Skipped,
    FallbackUsed,
}

/// Ordered log of all decisions made during a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct DecisionLog {
    decisions: Vec<Decision>,
}

impl DecisionLog {
    /// Create an empty decision log.
    pub fn new() -> Self {
        Self {
            decisions: Vec::new(),
        }
    }

    /// Record a decision with auto-incremented step, no detail or duration.
    pub fn record(
        &mut self,
        category: DecisionCategory,
        action: impl Into<String>,
        outcome: DecisionOutcome,
    ) {
        self.record_full(category, action, outcome, None, None);
    }

    /// Record a decision with detail.
    pub fn record_with_detail(
        &mut self,
        category: DecisionCategory,
        action: impl Into<String>,
        outcome: DecisionOutcome,
        detail: impl Into<String>,
    ) {
        self.record_full(category, action, outcome, Some(detail.into()), None);
    }

    /// Record a decision with duration.
    pub fn record_timed(
        &mut self,
        category: DecisionCategory,
        action: impl Into<String>,
        outcome: DecisionOutcome,
        duration: Duration,
    ) {
        self.record_full(category, action, outcome, None, Some(duration));
    }

    /// Record a decision with full control over all fields.
    pub fn record_full(
        &mut self,
        category: DecisionCategory,
        action: impl Into<String>,
        outcome: DecisionOutcome,
        detail: Option<String>,
        duration: Option<Duration>,
    ) {
        let step = self.decisions.len() + 1;
        self.decisions.push(Decision {
            step,
            category,
            action: action.into(),
            outcome,
            detail,
            duration,
        });
    }

    /// Read access to all decisions.
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Number of recorded decisions.
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Return all decisions with a `Failed` outcome.
    pub fn failures(&self) -> Vec<&Decision> {
        self.decisions
            .iter()
            .filter(|d| d.outcome == DecisionOutcome::Failed)
            .collect()
    }

    /// Return all decisions matching the given category.
    pub fn by_category(&self, cat: DecisionCategory) -> Vec<&Decision> {
        self.decisions
            .iter()
            .filter(|d| d.category == cat)
            .collect()
    }

    /// Human-readable multi-line summary of the log.
    pub fn summary(&self) -> String {
        if self.decisions.is_empty() {
            return "DecisionLog: (empty)".to_string();
        }

        let mut lines = Vec::with_capacity(self.decisions.len() + 2);
        lines.push(format!("DecisionLog: {} decisions", self.decisions.len()));

        for d in &self.decisions {
            let mut line = format!(
                "  [{}] {} | {} -> {}",
                d.step, d.category, d.action, d.outcome
            );
            if let Some(ref detail) = d.detail {
                line.push_str(&format!(" ({})", detail));
            }
            if let Some(dur) = d.duration {
                line.push_str(&format!(" [{:.3}s]", dur.as_secs_f64()));
            }
            lines.push(line);
        }

        let total = self.total_duration();
        if total > Duration::ZERO {
            lines.push(format!("  Total time: {:.3}s", total.as_secs_f64()));
        }

        let fail_count = self.failures().len();
        if fail_count > 0 {
            lines.push(format!("  Failures: {}", fail_count));
        }

        lines.join("\n")
    }

    /// Sum of all recorded durations.
    pub fn total_duration(&self) -> Duration {
        self.decisions.iter().filter_map(|d| d.duration).sum()
    }
}

impl fmt::Display for DecisionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DecisionCategory::DataProfiling => "Profiling",
            DecisionCategory::Preprocessing => "Preprocessing",
            DecisionCategory::ModelSelection => "Model Selection",
            DecisionCategory::ModelFitting => "Model Fitting",
            DecisionCategory::CrossValidation => "Cross-Validation",
            DecisionCategory::Ensembling => "Ensembling",
            DecisionCategory::Postprocessing => "Postprocessing",
            DecisionCategory::Constraint => "Constraint",
            DecisionCategory::Fallback => "Fallback",
            DecisionCategory::TrendSelection => "Trend Selection",
            DecisionCategory::SeasonalSelection => "Seasonal Selection",
            DecisionCategory::ChangepointAdaptation => "Changepoint Adaptation",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for DecisionOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DecisionOutcome::Success => "OK",
            DecisionOutcome::Failed => "FAILED",
            DecisionOutcome::Skipped => "SKIPPED",
            DecisionOutcome::FallbackUsed => "FALLBACK",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for DecisionLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log() {
        let log = DecisionLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.decisions().is_empty());
        assert!(log.failures().is_empty());
        assert_eq!(log.total_duration(), Duration::ZERO);
        assert!(log.summary().contains("empty"));
    }

    #[test]
    fn record_basic() {
        let mut log = DecisionLog::new();
        log.record(
            DecisionCategory::ModelFitting,
            "Fitted ARIMA(1,1,1)",
            DecisionOutcome::Success,
        );
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());

        let d = &log.decisions()[0];
        assert_eq!(d.step, 1);
        assert_eq!(d.category, DecisionCategory::ModelFitting);
        assert_eq!(d.action, "Fitted ARIMA(1,1,1)");
        assert_eq!(d.outcome, DecisionOutcome::Success);
        assert!(d.detail.is_none());
        assert!(d.duration.is_none());
    }

    #[test]
    fn record_with_detail() {
        let mut log = DecisionLog::new();
        log.record_with_detail(
            DecisionCategory::CrossValidation,
            "Evaluated ETS",
            DecisionOutcome::Success,
            "MASE=0.82",
        );
        assert_eq!(log.len(), 1);

        let d = &log.decisions()[0];
        assert_eq!(d.detail.as_deref(), Some("MASE=0.82"));
    }

    #[test]
    fn record_timed() {
        let mut log = DecisionLog::new();
        let dur = Duration::from_millis(150);
        log.record_timed(
            DecisionCategory::ModelFitting,
            "Fitted Theta",
            DecisionOutcome::Success,
            dur,
        );
        assert_eq!(log.len(), 1);

        let d = &log.decisions()[0];
        assert_eq!(d.duration, Some(dur));
    }

    #[test]
    fn failures_filtered() {
        let mut log = DecisionLog::new();
        log.record(
            DecisionCategory::ModelFitting,
            "Fitted ARIMA",
            DecisionOutcome::Success,
        );
        log.record(
            DecisionCategory::ModelFitting,
            "Fitted ETS",
            DecisionOutcome::Failed,
        );
        log.record(
            DecisionCategory::Fallback,
            "Used naive",
            DecisionOutcome::FallbackUsed,
        );
        log.record(
            DecisionCategory::Preprocessing,
            "Log transform",
            DecisionOutcome::Failed,
        );

        let failures = log.failures();
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].action, "Fitted ETS");
        assert_eq!(failures[1].action, "Log transform");
    }

    #[test]
    fn by_category() {
        let mut log = DecisionLog::new();
        log.record(
            DecisionCategory::DataProfiling,
            "Detected seasonality",
            DecisionOutcome::Success,
        );
        log.record(
            DecisionCategory::ModelFitting,
            "Fitted ARIMA",
            DecisionOutcome::Success,
        );
        log.record(
            DecisionCategory::ModelFitting,
            "Fitted ETS",
            DecisionOutcome::Failed,
        );
        log.record(
            DecisionCategory::Postprocessing,
            "Clamped negatives",
            DecisionOutcome::Success,
        );

        let fitting = log.by_category(DecisionCategory::ModelFitting);
        assert_eq!(fitting.len(), 2);

        let profiling = log.by_category(DecisionCategory::DataProfiling);
        assert_eq!(profiling.len(), 1);

        let ensembling = log.by_category(DecisionCategory::Ensembling);
        assert!(ensembling.is_empty());
    }

    #[test]
    fn total_duration() {
        let mut log = DecisionLog::new();
        log.record_timed(
            DecisionCategory::ModelFitting,
            "Fitted ARIMA",
            DecisionOutcome::Success,
            Duration::from_millis(100),
        );
        log.record(
            DecisionCategory::ModelFitting,
            "Fitted ETS",
            DecisionOutcome::Failed,
        );
        log.record_timed(
            DecisionCategory::CrossValidation,
            "CV fold 1",
            DecisionOutcome::Success,
            Duration::from_millis(250),
        );

        assert_eq!(log.total_duration(), Duration::from_millis(350));
    }

    #[test]
    fn display_contains_info() {
        let mut log = DecisionLog::new();
        log.record_with_detail(
            DecisionCategory::ModelFitting,
            "Fitted ARIMA(1,1,1)",
            DecisionOutcome::Success,
            "MASE=0.82",
        );
        log.record(
            DecisionCategory::Preprocessing,
            "Differenced series",
            DecisionOutcome::Skipped,
        );

        let display = format!("{}", log);
        assert!(display.contains("2 decisions"));
        assert!(display.contains("Model Fitting"));
        assert!(display.contains("Fitted ARIMA(1,1,1)"));
        assert!(display.contains("OK"));
        assert!(display.contains("MASE=0.82"));
        assert!(display.contains("SKIPPED"));

        // Also check individual Display impls
        assert_eq!(format!("{}", DecisionCategory::DataProfiling), "Profiling");
        assert_eq!(format!("{}", DecisionOutcome::FallbackUsed), "FALLBACK");
    }

    #[test]
    fn step_auto_increments() {
        let mut log = DecisionLog::new();
        log.record(
            DecisionCategory::DataProfiling,
            "Step A",
            DecisionOutcome::Success,
        );
        log.record(
            DecisionCategory::Preprocessing,
            "Step B",
            DecisionOutcome::Success,
        );
        log.record(
            DecisionCategory::ModelFitting,
            "Step C",
            DecisionOutcome::Success,
        );

        assert_eq!(log.decisions()[0].step, 1);
        assert_eq!(log.decisions()[1].step, 2);
        assert_eq!(log.decisions()[2].step, 3);
    }
}
