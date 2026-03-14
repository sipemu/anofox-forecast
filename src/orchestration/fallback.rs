//! Error recovery and fallback chain for model selection.
//!
//! A [`FallbackChain`] wraps a list of model factory closures. When
//! [`FallbackChain::execute`] is called with a [`TimeSeries`], it tries each
//! model in order until one succeeds (fit + predict). Successes and failures
//! can be recorded in a [`DecisionLog`](super::DecisionLog).

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Result of a fallback chain execution.
#[derive(Debug)]
pub struct FallbackResult {
    /// Name of the model that ultimately succeeded.
    pub model_name: String,
    /// The forecast produced by the successful model.
    pub forecast: Forecast,
    /// How many models were tried (including the successful one).
    pub attempts: usize,
    /// Names of models that failed before a success was found.
    pub failed_models: Vec<String>,
}

/// A chain of model factories tried in order until one succeeds.
pub struct FallbackChain {
    entries: Vec<FallbackEntry>,
}

struct FallbackEntry {
    name: String,
    factory: Box<dyn Fn() -> Box<dyn Forecaster>>,
}

impl FallbackChain {
    /// Create a new, empty fallback chain.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a model to the chain.
    ///
    /// Models are tried in the order they are added. The `factory` closure is
    /// called each time the chain is executed, producing a fresh model instance.
    pub fn add(
        mut self,
        name: impl Into<String>,
        factory: impl Fn() -> Box<dyn Forecaster> + 'static,
    ) -> Self {
        self.entries.push(FallbackEntry {
            name: name.into(),
            factory: Box::new(factory),
        });
        self
    }

    /// Number of models in the chain.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the chain contains no models.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Try each model in order; return the first successful forecast.
    ///
    /// If all models fail, the last error is returned.
    pub fn execute(&self, ts: &TimeSeries, horizon: usize) -> Result<FallbackResult> {
        if self.entries.is_empty() {
            return Err(ForecastError::ComputationError(
                "fallback chain is empty — no models to try".to_string(),
            ));
        }

        let mut failed_models = Vec::new();
        let mut last_error: Option<ForecastError> = None;

        for entry in &self.entries {
            let mut model = (entry.factory)();
            match model.fit(ts).and_then(|()| model.predict(horizon)) {
                Ok(forecast) => {
                    return Ok(FallbackResult {
                        model_name: entry.name.clone(),
                        forecast,
                        attempts: failed_models.len() + 1,
                        failed_models,
                    });
                }
                Err(e) => {
                    failed_models.push(entry.name.clone());
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Try each model in order with prediction intervals; return the first success.
    ///
    /// If all models fail, the last error is returned.
    pub fn execute_with_intervals(
        &self,
        ts: &TimeSeries,
        horizon: usize,
        level: f64,
    ) -> Result<FallbackResult> {
        if self.entries.is_empty() {
            return Err(ForecastError::ComputationError(
                "fallback chain is empty — no models to try".to_string(),
            ));
        }

        let mut failed_models = Vec::new();
        let mut last_error: Option<ForecastError> = None;

        for entry in &self.entries {
            let mut model = (entry.factory)();
            match model
                .fit(ts)
                .and_then(|()| model.predict_with_intervals(horizon, level))
            {
                Ok(forecast) => {
                    return Ok(FallbackResult {
                        model_name: entry.name.clone(),
                        forecast,
                        attempts: failed_models.len() + 1,
                        failed_models,
                    });
                }
                Err(e) => {
                    failed_models.push(entry.name.clone());
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Try each model in order and record decisions in the provided log.
    ///
    /// If all models fail, the last error is returned.
    pub fn execute_with_log(
        &self,
        ts: &TimeSeries,
        horizon: usize,
        log: &mut super::decision_log::DecisionLog,
    ) -> Result<FallbackResult> {
        use super::decision_log::{DecisionCategory, DecisionOutcome};

        if self.entries.is_empty() {
            log.record(
                DecisionCategory::Fallback,
                "Fallback chain is empty",
                DecisionOutcome::Failed,
            );
            return Err(ForecastError::ComputationError(
                "fallback chain is empty — no models to try".to_string(),
            ));
        }

        let mut failed_models = Vec::new();
        let mut last_error: Option<ForecastError> = None;

        for entry in &self.entries {
            let mut model = (entry.factory)();
            match model.fit(ts).and_then(|()| model.predict(horizon)) {
                Ok(forecast) => {
                    let outcome = if failed_models.is_empty() {
                        DecisionOutcome::Success
                    } else {
                        DecisionOutcome::FallbackUsed
                    };
                    log.record(
                        DecisionCategory::Fallback,
                        format!("Tried {}", entry.name),
                        outcome,
                    );
                    return Ok(FallbackResult {
                        model_name: entry.name.clone(),
                        forecast,
                        attempts: failed_models.len() + 1,
                        failed_models,
                    });
                }
                Err(e) => {
                    log.record_with_detail(
                        DecisionCategory::Fallback,
                        format!("Tried {}", entry.name),
                        DecisionOutcome::Failed,
                        e.to_string(),
                    );
                    failed_models.push(entry.name.clone());
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap())
    }
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesBuilder;
    use crate::models::baseline::{HistoricAverage, Naive, SimpleMovingAverage};
    use crate::orchestration::decision_log::DecisionLog;
    use chrono::{Duration, Utc};

    /// A model that always fails during fitting.
    struct AlwaysFailsModel;

    impl Forecaster for AlwaysFailsModel {
        fn fit(&mut self, _: &TimeSeries) -> Result<()> {
            Err(ForecastError::ComputationError(
                "intentional failure".to_string(),
            ))
        }

        fn predict(&self, _: usize) -> Result<Forecast> {
            Err(ForecastError::FitRequired { model: None })
        }

        fn predict_with_intervals(&self, _: usize, _: f64) -> Result<Forecast> {
            Err(ForecastError::FitRequired { model: None })
        }

        fn fitted_values(&self) -> Option<&[f64]> {
            None
        }

        fn residuals(&self) -> Option<&[f64]> {
            None
        }

        fn name(&self) -> &str {
            "AlwaysFails"
        }

        fn is_fitted(&self) -> bool {
            false
        }
    }

    fn make_test_series() -> TimeSeries {
        let n = 50;
        let values: Vec<f64> = (0..n)
            .map(|i| 10.0 + (i as f64 * 0.3).sin() * 3.0)
            .collect();
        let start = Utc::now();
        let timestamps: Vec<_> = (0..n).map(|i| start + Duration::days(i as i64)).collect();
        TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .values(values)
            .build()
            .unwrap()
    }

    #[test]
    fn first_model_succeeds() {
        let chain = FallbackChain::new().add("Naive", || Box::new(Naive::new()));

        let ts = make_test_series();
        let result = chain.execute(&ts, 5).unwrap();

        assert_eq!(result.model_name, "Naive");
        assert_eq!(result.attempts, 1);
        assert!(result.failed_models.is_empty());
        assert_eq!(result.forecast.horizon(), 5);
    }

    #[test]
    fn fallback_to_second() {
        let chain = FallbackChain::new()
            .add("AlwaysFails", || Box::new(AlwaysFailsModel))
            .add("Naive", || Box::new(Naive::new()));

        let ts = make_test_series();
        let result = chain.execute(&ts, 5).unwrap();

        assert_eq!(result.model_name, "Naive");
        assert_eq!(result.attempts, 2);
        assert_eq!(result.failed_models, vec!["AlwaysFails"]);
    }

    #[test]
    fn all_fail() {
        let chain = FallbackChain::new()
            .add("Fail1", || Box::new(AlwaysFailsModel))
            .add("Fail2", || Box::new(AlwaysFailsModel));

        let ts = make_test_series();
        let err = chain.execute(&ts, 5).unwrap_err();

        // Should return the last error
        assert!(
            matches!(err, ForecastError::ComputationError(_)),
            "expected ComputationError, got {:?}",
            err
        );
    }

    #[test]
    fn empty_chain() {
        let chain = FallbackChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        let ts = make_test_series();
        let err = chain.execute(&ts, 5).unwrap_err();
        assert!(
            matches!(err, ForecastError::ComputationError(_)),
            "expected ComputationError for empty chain, got {:?}",
            err
        );
    }

    #[test]
    fn with_intervals() {
        let chain = FallbackChain::new().add("Naive", || Box::new(Naive::new()));

        let ts = make_test_series();
        let result = chain.execute_with_intervals(&ts, 5, 0.95).unwrap();

        assert_eq!(result.model_name, "Naive");
        assert_eq!(result.forecast.horizon(), 5);
        assert!(result.forecast.has_lower());
        assert!(result.forecast.has_upper());
    }

    #[test]
    fn failed_models_tracked() {
        let chain = FallbackChain::new()
            .add("Fail1", || Box::new(AlwaysFailsModel))
            .add("Fail2", || Box::new(AlwaysFailsModel))
            .add("BigSMA", || Box::new(SimpleMovingAverage::new(999)))
            .add("HistAvg", || Box::new(HistoricAverage::new()));

        let ts = make_test_series();
        let result = chain.execute(&ts, 5).unwrap();

        assert_eq!(result.model_name, "HistAvg");
        assert_eq!(result.failed_models, vec!["Fail1", "Fail2", "BigSMA"]);
    }

    #[test]
    fn attempts_count() {
        let chain = FallbackChain::new()
            .add("Fail1", || Box::new(AlwaysFailsModel))
            .add("Fail2", || Box::new(AlwaysFailsModel))
            .add("Naive", || Box::new(Naive::new()));

        let ts = make_test_series();
        let result = chain.execute(&ts, 5).unwrap();

        assert_eq!(result.attempts, 3);
        assert_eq!(result.failed_models.len(), 2);
    }

    #[test]
    fn with_log() {
        let chain = FallbackChain::new()
            .add("AlwaysFails", || Box::new(AlwaysFailsModel))
            .add("Naive", || Box::new(Naive::new()));

        let ts = make_test_series();
        let mut log = DecisionLog::new();
        let result = chain.execute_with_log(&ts, 5, &mut log).unwrap();

        assert_eq!(result.model_name, "Naive");
        assert!(!log.is_empty());
        assert_eq!(log.len(), 2); // one failure + one success

        // The first decision should be a failure
        let decisions = log.decisions();
        assert_eq!(
            decisions[0].outcome,
            super::super::decision_log::DecisionOutcome::Failed
        );
        // The second should reflect fallback success
        assert_eq!(
            decisions[1].outcome,
            super::super::decision_log::DecisionOutcome::FallbackUsed
        );
    }
}
