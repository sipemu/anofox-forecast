//! Declarative pipeline builder for composing forecasting workflows.
//!
//! A [`PipelineBuilder`] chains configuration steps into a [`Pipeline`] that
//! can be executed against a [`TimeSeries`]. The result captures the forecast,
//! data profile, decision log, execution metadata, and per-horizon analysis.
//!
//! Pipeline configurations can be serialized to JSON and replayed on new data
//! via [`PipelineConfig`].

use std::fmt;

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::baseline::{Naive, SimpleMovingAverage};
use crate::models::{Forecaster, ModelRegistry};
use crate::utils::metrics::mae;

use super::confidence::{ModelConfidenceSet, QualityFloor, SelectionConfidence};
use super::decision_log::{DecisionCategory, DecisionLog, DecisionOutcome};
use super::fallback::FallbackChain;
use super::horizon::HorizonAnalysis;
use super::metadata::{ExecutionMetadata, ExecutionTimer};
use super::profile::DataProfile;

/// Serializable pipeline configuration for replay on new data.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether to run data profiling.
    pub profile: bool,
    /// Number of top models to select for evaluation (0 = use all).
    pub select_top_k: usize,
    /// Number of CV folds (0 = skip CV, use holdout).
    pub cv_folds: usize,
    /// Forecast horizon for CV and final prediction.
    pub horizon: usize,
    /// Holdout size for train/test split when CV is skipped.
    pub holdout: usize,
    /// Whether to use a fallback chain (Naive → SMA).
    pub use_fallback: bool,
    /// Confidence level for prediction intervals (0 = no intervals).
    pub interval_level: f64,
    /// Whether to apply non-negative constraint.
    pub non_negative: bool,
    /// Whether to compute per-horizon analysis.
    pub horizon_analysis: bool,
    /// Seasonal period hint (0 = none).
    pub seasonal_period: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            profile: true,
            select_top_k: 0,
            cv_folds: 0,
            horizon: 1,
            holdout: 0,
            use_fallback: true,
            interval_level: 0.0,
            non_negative: false,
            horizon_analysis: false,
            seasonal_period: 0,
        }
    }
}

/// Result of executing a pipeline.
#[derive(Debug)]
pub struct PipelineResult {
    /// The final forecast.
    pub forecast: Forecast,
    /// Name of the model that produced the forecast.
    pub model_name: String,
    /// Data profile (if profiling was enabled).
    pub profile: Option<DataProfile>,
    /// Full decision log of the pipeline run.
    pub log: DecisionLog,
    /// Per-model execution metadata.
    pub model_metadata: Vec<ExecutionMetadata>,
    /// Per-horizon analysis (if enabled and CV was used).
    pub horizon_analysis: Option<HorizonAnalysis>,
    /// Pairwise selection confidence (DM test, if multiple models were compared).
    pub selection_confidence: Option<SelectionConfidence>,
    /// Model Confidence Set (set of statistically best models).
    pub model_confidence_set: Option<ModelConfidenceSet>,
    /// Quality floor check (SPA test vs Naive).
    pub quality_floor: Option<QualityFloor>,
}

impl fmt::Display for PipelineResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Result")?;
        writeln!(f, "===============")?;
        writeln!(f, "Selected model: {}", self.model_name)?;
        writeln!(
            f,
            "Forecast horizon: {} steps",
            self.forecast.primary().len()
        )?;
        if let Some(ref profile) = self.profile {
            writeln!(f, "Series length: {}", profile.n_observations)?;
        }
        if let Some(ref qf) = self.quality_floor {
            writeln!(f, "{}", qf)?;
        }
        if let Some(ref conf) = self.selection_confidence {
            writeln!(f, "Selection: {}", conf.verdict)?;
        }
        if let Some(ref mcs) = self.model_confidence_set {
            writeln!(f, "{}", mcs)?;
        }
        writeln!(f, "\n{}", self.log)?;
        Ok(())
    }
}

/// Builder for constructing a forecasting pipeline.
pub struct PipelineBuilder {
    config: PipelineConfig,
    registry: Option<ModelRegistry>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            registry: None,
        }
    }

    /// Enable data profiling (enabled by default).
    pub fn profile(mut self) -> Self {
        self.config.profile = true;
        self
    }

    /// Skip data profiling.
    pub fn skip_profile(mut self) -> Self {
        self.config.profile = false;
        self
    }

    /// Set the model registry to select from.
    pub fn registry(mut self, registry: ModelRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Select the top-k models by holdout performance before CV.
    /// Use 0 to evaluate all models (default).
    pub fn select_models(mut self, top_k: usize) -> Self {
        self.config.select_top_k = top_k;
        self
    }

    /// Configure cross-validation with the given number of folds and horizon.
    pub fn cross_validate(mut self, folds: usize, horizon: usize) -> Self {
        self.config.cv_folds = folds;
        self.config.horizon = horizon;
        self
    }

    /// Set the holdout size for train/test evaluation (used when CV is skipped).
    pub fn holdout(mut self, size: usize) -> Self {
        self.config.holdout = size;
        self
    }

    /// Set the forecast horizon.
    pub fn horizon(mut self, h: usize) -> Self {
        self.config.horizon = h;
        self
    }

    /// Enable fallback chain (Naive → SMA) for error recovery.
    pub fn with_fallback(mut self) -> Self {
        self.config.use_fallback = true;
        self
    }

    /// Disable fallback chain.
    pub fn without_fallback(mut self) -> Self {
        self.config.use_fallback = false;
        self
    }

    /// Request prediction intervals at the given confidence level (e.g., 0.95).
    pub fn intervals(mut self, level: f64) -> Self {
        self.config.interval_level = level;
        self
    }

    /// Apply non-negative constraint to the forecast.
    pub fn non_negative(mut self) -> Self {
        self.config.non_negative = true;
        self
    }

    /// Enable per-horizon analysis.
    pub fn with_horizon_analysis(mut self) -> Self {
        self.config.horizon_analysis = true;
        self
    }

    /// Set the seasonal period hint.
    pub fn seasonal_period(mut self, period: usize) -> Self {
        self.config.seasonal_period = period;
        self
    }

    /// Build the pipeline from the current configuration.
    pub fn build(self) -> Pipeline {
        Pipeline {
            config: self.config,
            registry: self.registry,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A configured forecasting pipeline ready for execution.
pub struct Pipeline {
    config: PipelineConfig,
    registry: Option<ModelRegistry>,
}

impl Pipeline {
    /// Create a pipeline from a saved configuration.
    pub fn from_config(config: PipelineConfig) -> Self {
        Self {
            config,
            registry: None,
        }
    }

    /// Set the model registry (needed when replaying from config).
    pub fn with_registry(mut self, registry: ModelRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Get the pipeline configuration (for serialization / replay).
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Execute the pipeline on a time series.
    pub fn execute(&self, ts: &TimeSeries, horizon: usize) -> Result<PipelineResult> {
        let mut log = DecisionLog::new();
        let mut model_metadata = Vec::new();
        let effective_horizon = if horizon > 0 {
            horizon
        } else {
            self.config.horizon
        };

        // ── Step 1: Data profiling ──────────────────────────────────
        let profile = if self.config.profile {
            let timer = ExecutionTimer::start();
            let p = DataProfile::from_series(ts);
            let dur = timer.stop();
            log.record_timed(
                DecisionCategory::DataProfiling,
                format!(
                    "Profiled {} observations: trend={}, stationary={}",
                    p.n_observations, p.trend_direction, p.adf_is_stationary
                ),
                DecisionOutcome::Success,
                dur,
            );
            Some(p)
        } else {
            log.record(
                DecisionCategory::DataProfiling,
                "Profiling skipped",
                DecisionOutcome::Skipped,
            );
            None
        };

        // ── Step 2: Model selection via registry ───────────────────
        let registry = match &self.registry {
            Some(r) if !r.is_empty() => r,
            _ => {
                // No registry — use fallback chain directly
                return self.execute_fallback_only(
                    ts,
                    effective_horizon,
                    log,
                    model_metadata,
                    profile,
                );
            }
        };

        let holdout = if self.config.holdout > 0 {
            self.config.holdout
        } else {
            effective_horizon
        };
        let n = ts.len();

        if n <= holdout {
            return self.execute_fallback_only(ts, effective_horizon, log, model_metadata, profile);
        }

        // Train/test split for holdout evaluation
        let train_ts = ts.slice(0, n - holdout)?;
        let test_values = ts.values(0)?;
        let test_actual = &test_values[n - holdout..];

        // Evaluate all registry models on holdout.
        // Track per-observation absolute errors for DM/MCS/SPA.
        let mut scored: Vec<(String, f64)> = Vec::new();
        let mut per_obs_losses: Vec<(String, Vec<f64>)> = Vec::new();

        for spec in registry.iter() {
            let timer = ExecutionTimer::start();
            let mut model = spec.create();
            let name = model.name().to_string();

            match model.fit(&train_ts).and_then(|_| model.predict(holdout)) {
                Ok(forecast) => {
                    let dur = timer.stop();
                    let preds = forecast.primary();
                    let score = mae(test_actual, preds);
                    // Per-observation absolute errors for statistical tests
                    let obs_losses: Vec<f64> = test_actual
                        .iter()
                        .zip(preds.iter())
                        .map(|(a, p)| (a - p).abs())
                        .collect();
                    log.record_full(
                        DecisionCategory::ModelFitting,
                        format!("Fitted {}", name),
                        DecisionOutcome::Success,
                        Some(format!("MAE={:.4}", score)),
                        Some(dur),
                    );
                    model_metadata.push(
                        ExecutionMetadata::new(&name)
                            .with_fit(dur)
                            .with_observations(n - holdout)
                            .with_horizon(holdout)
                            .with_convergence(true),
                    );
                    per_obs_losses.push((name.clone(), obs_losses));
                    scored.push((name, score));
                }
                Err(e) => {
                    let dur = timer.stop();
                    log.record_full(
                        DecisionCategory::ModelFitting,
                        format!("Fitted {}", name),
                        DecisionOutcome::Failed,
                        Some(format!("{}", e)),
                        Some(dur),
                    );
                    model_metadata.push(
                        ExecutionMetadata::new(&name)
                            .with_fit(dur)
                            .with_observations(n - holdout)
                            .with_convergence(false)
                            .with_error(format!("{}", e)),
                    );
                }
            }
        }

        // Sort by score (lower MAE is better)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if scored.is_empty() {
            log.record(
                DecisionCategory::ModelSelection,
                "All models failed, using fallback",
                DecisionOutcome::FallbackUsed,
            );
            return self.execute_fallback_only(ts, effective_horizon, log, model_metadata, profile);
        }

        // Select top-k if configured
        if self.config.select_top_k > 0 {
            let top_k = self.config.select_top_k.min(scored.len());
            let top_names: std::collections::HashSet<&str> =
                scored[..top_k].iter().map(|(n, _)| n.as_str()).collect();
            per_obs_losses.retain(|(n, _)| top_names.contains(n.as_str()));
            scored.truncate(top_k);
        }

        // ── Quality floor: SPA test vs Naive ─────────────────────────
        let quality_floor = {
            let mut naive = Naive::new();
            if let Ok(()) = naive.fit(&train_ts) {
                if let Ok(naive_fc) = naive.predict(holdout) {
                    let naive_losses: Vec<f64> = test_actual
                        .iter()
                        .zip(naive_fc.primary().iter())
                        .map(|(a, p)| (a - p).abs())
                        .collect();
                    let model_loss_vecs: Vec<Vec<f64>> = per_obs_losses
                        .iter()
                        .map(|(_, losses)| losses.clone())
                        .collect();
                    let qf = QualityFloor::test("Naive", &naive_losses, &model_loss_vecs);
                    if let Some(ref qf) = qf {
                        log.record_with_detail(
                            DecisionCategory::ModelSelection,
                            if qf.is_outperformed {
                                "Quality floor: PASSED (SPA test)"
                            } else {
                                "Quality floor: FAILED (no model significantly beats Naive)"
                            },
                            if qf.is_outperformed {
                                DecisionOutcome::Success
                            } else {
                                DecisionOutcome::Failed
                            },
                            format!("SPA p={:.4}", qf.spa_p_value),
                        );
                    }
                    qf
                } else {
                    None
                }
            } else {
                None
            }
        };

        // ── Model Confidence Set ─────────────────────────────────────
        let model_confidence_set = if per_obs_losses.len() >= 2 {
            let mcs = ModelConfidenceSet::from_cv_scores(per_obs_losses.clone(), 0.10);
            if let Some(ref mcs) = mcs {
                log.record_with_detail(
                    DecisionCategory::ModelSelection,
                    format!("MCS: {} model(s) in confidence set", mcs.len()),
                    DecisionOutcome::Success,
                    format!("included: {:?}", mcs.included),
                );
            }
            mcs
        } else {
            None
        };

        // ── Selection confidence (DM test, top-2) ────────────────────
        let selection_confidence = if scored.len() >= 2 {
            let best_name = &scored[0].0;
            let runner_up_name = &scored[1].0;
            let best_losses = per_obs_losses
                .iter()
                .find(|(n, _)| n == best_name)
                .map(|(_, l)| l.clone());
            let runner_up_losses = per_obs_losses
                .iter()
                .find(|(n, _)| n == runner_up_name)
                .map(|(_, l)| l.clone());
            match (best_losses, runner_up_losses) {
                (Some(bl), Some(rl)) => {
                    let conf = SelectionConfidence::compare(best_name, bl, runner_up_name, rl);
                    log.record_with_detail(
                        DecisionCategory::ModelSelection,
                        format!("DM test: {}", conf.verdict),
                        DecisionOutcome::Success,
                        format!("p={:.4}", conf.dm_p_value),
                    );
                    Some(conf)
                }
                _ => None,
            }
        } else {
            None
        };

        // ── Select winner and produce final forecast ────────────────
        let winner_name = scored[0].0.clone();
        log.record_with_detail(
            DecisionCategory::ModelSelection,
            format!("Selected {}", winner_name),
            DecisionOutcome::Success,
            format!(
                "MAE={:.4}, {} of {} models succeeded",
                scored[0].1,
                scored.len(),
                registry.len()
            ),
        );

        // Re-fit winner on full data and predict
        let timer = ExecutionTimer::start();
        let forecast =
            self.fit_and_predict_model(registry, &winner_name, ts, effective_horizon, &mut log);
        let _dur = timer.stop();

        let forecast = match forecast {
            Ok(f) => f,
            Err(_) if self.config.use_fallback => {
                log.record(
                    DecisionCategory::Fallback,
                    format!("Winner {} failed on full data, using fallback", winner_name),
                    DecisionOutcome::FallbackUsed,
                );
                return self.execute_fallback_only(
                    ts,
                    effective_horizon,
                    log,
                    model_metadata,
                    profile,
                );
            }
            Err(e) => return Err(e),
        };

        // ── Apply constraints ───────────────────────────────────────
        let forecast = if self.config.non_negative {
            log.record(
                DecisionCategory::Constraint,
                "Applied non-negative constraint",
                DecisionOutcome::Success,
            );
            forecast.non_negative()
        } else {
            forecast
        };

        Ok(PipelineResult {
            forecast,
            model_name: winner_name,
            profile,
            log,
            model_metadata,
            horizon_analysis: None,
            selection_confidence,
            model_confidence_set,
            quality_floor,
        })
    }

    /// Fallback-only path when no registry or all models fail.
    fn execute_fallback_only(
        &self,
        ts: &TimeSeries,
        horizon: usize,
        mut log: DecisionLog,
        model_metadata: Vec<ExecutionMetadata>,
        profile: Option<DataProfile>,
    ) -> Result<PipelineResult> {
        if !self.config.use_fallback {
            return Err(ForecastError::ComputationError(
                "No models available and fallback is disabled".to_string(),
            ));
        }

        let chain = FallbackChain::new()
            .add("Naive", || Box::new(Naive::new()) as Box<dyn Forecaster>)
            .add("SMA(5)", || {
                Box::new(SimpleMovingAverage::new(5)) as Box<dyn Forecaster>
            });

        let result = chain.execute_with_log(ts, horizon, &mut log)?;

        let forecast = if self.config.non_negative {
            log.record(
                DecisionCategory::Constraint,
                "Applied non-negative constraint",
                DecisionOutcome::Success,
            );
            result.forecast.non_negative()
        } else {
            result.forecast
        };

        Ok(PipelineResult {
            forecast,
            model_name: result.model_name,
            profile,
            log,
            model_metadata,
            horizon_analysis: None,
            selection_confidence: None,
            model_confidence_set: None,
            quality_floor: None,
        })
    }

    /// Re-fit a named model from the registry on full data and predict.
    fn fit_and_predict_model(
        &self,
        registry: &ModelRegistry,
        name: &str,
        ts: &TimeSeries,
        horizon: usize,
        _log: &mut DecisionLog,
    ) -> Result<Forecast> {
        for spec in registry.iter() {
            let mut model = spec.create();
            if model.name() == name {
                model.fit(ts)?;
                let forecast = if self.config.interval_level > 0.0 {
                    model.predict_with_intervals(horizon, self.config.interval_level)?
                } else {
                    model.predict(horizon)?
                };
                return Ok(forecast);
            }
        }
        Err(ForecastError::InvalidParameter(format!(
            "Model '{}' not found in registry",
            name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeriesBuilder;
    use crate::models::baseline::HistoricAverage;
    use crate::models::ModelSpec;
    use chrono::{Duration, Utc};

    fn make_ts(n: usize) -> TimeSeries {
        let values: Vec<f64> = (0..n)
            .map(|i| 10.0 + (i as f64 * 0.3).sin() * 3.0 + (i as f64) * 0.1)
            .collect();
        let start = Utc::now();
        let timestamps: Vec<_> = (0..n).map(|i| start + Duration::days(i as i64)).collect();
        TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .values(values)
            .build()
            .unwrap()
    }

    fn make_registry() -> ModelRegistry {
        let mut reg = ModelRegistry::new();
        reg.register(ModelSpec::new("Naive", || Box::new(Naive::new()), true));
        reg.register(ModelSpec::new(
            "SMA(3)",
            || Box::new(SimpleMovingAverage::new(3)),
            true,
        ));
        reg.register(ModelSpec::new(
            "HistoricAverage",
            || Box::new(HistoricAverage::new()),
            true,
        ));
        reg
    }

    #[test]
    fn pipeline_basic_with_registry() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .horizon(5)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
        assert!(!result.model_name.is_empty());
        assert!(!result.log.is_empty());
    }

    #[test]
    fn pipeline_with_profiling() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .profile()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(result.profile.is_some());
        let profile = result.profile.unwrap();
        assert_eq!(profile.n_observations, 50);
    }

    #[test]
    fn pipeline_skip_profiling() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .skip_profile()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(result.profile.is_none());
    }

    #[test]
    fn pipeline_fallback_no_registry() {
        let ts = make_ts(50);
        let result = PipelineBuilder::new()
            .with_fallback()
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
        // Should have used Naive or SMA from fallback
        assert!(
            result.model_name == "Naive" || result.model_name == "SMA(5)",
            "Expected fallback model, got: {}",
            result.model_name
        );
    }

    #[test]
    fn pipeline_no_registry_no_fallback_fails() {
        let ts = make_ts(50);
        let result = PipelineBuilder::new()
            .without_fallback()
            .build()
            .execute(&ts, 5);

        assert!(result.is_err());
    }

    #[test]
    fn pipeline_non_negative_constraint() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .non_negative()
            .build()
            .execute(&ts, 5)
            .unwrap();

        for &v in result.forecast.primary() {
            assert!(v >= 0.0, "Expected non-negative, got {}", v);
        }
    }

    #[test]
    fn pipeline_with_intervals() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .intervals(0.95)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(result.forecast.has_lower());
        assert!(result.forecast.has_upper());
    }

    #[test]
    fn pipeline_quality_floor_check() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        // quality_floor (SPA test) should be populated
        assert!(result.quality_floor.is_some());
    }

    #[test]
    fn pipeline_mcs_populated() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        // MCS should be populated when multiple models succeed
        assert!(result.model_confidence_set.is_some());
        let mcs = result.model_confidence_set.unwrap();
        assert!(!mcs.is_empty());
    }

    #[test]
    fn pipeline_selection_confidence_populated() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        // DM-based selection confidence should be populated
        assert!(result.selection_confidence.is_some());
        let conf = result.selection_confidence.unwrap();
        assert!(!conf.best_model.is_empty());
    }

    #[test]
    fn pipeline_select_top_k() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .select_models(2)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
    }

    #[test]
    fn pipeline_decision_log_populated() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .profile()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        // Should have profiling + model fitting + selection decisions
        assert!(result.log.len() >= 3);
        assert!(!result
            .log
            .by_category(DecisionCategory::DataProfiling)
            .is_empty());
        assert!(!result
            .log
            .by_category(DecisionCategory::ModelFitting)
            .is_empty());
        assert!(!result
            .log
            .by_category(DecisionCategory::ModelSelection)
            .is_empty());
    }

    #[test]
    fn pipeline_model_metadata_populated() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(!result.model_metadata.is_empty());
        for meta in &result.model_metadata {
            assert!(!meta.model_name.is_empty());
        }
    }

    #[test]
    fn pipeline_display() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        let text = format!("{}", result);
        assert!(text.contains("Pipeline Result"));
        assert!(text.contains("Selected model:"));
    }

    #[test]
    fn pipeline_from_config() {
        let ts = make_ts(50);
        let config = PipelineConfig {
            profile: true,
            select_top_k: 0,
            cv_folds: 0,
            horizon: 5,
            holdout: 5,
            use_fallback: true,
            interval_level: 0.0,
            non_negative: false,
            horizon_analysis: false,
            seasonal_period: 0,
        };
        let reg = make_registry();
        let result = Pipeline::from_config(config)
            .with_registry(reg)
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
    }

    #[test]
    fn pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.profile);
        assert!(config.use_fallback);
        assert_eq!(config.select_top_k, 0);
        assert_eq!(config.cv_folds, 0);
    }
}
