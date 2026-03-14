//! Declarative pipeline builder for composing forecasting workflows.
//!
//! A [`PipelineBuilder`] chains configuration steps into a [`Pipeline`] that
//! can be executed against a [`TimeSeries`]. The result captures the forecast,
//! data profile, decision log, execution metadata, per-horizon analysis,
//! preprocessing info, ensemble weights, and multi-metric scores.
//!
//! Pipeline configurations can be replayed on new data via [`PipelineConfig`].

use std::collections::HashMap;
use std::fmt;

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::baseline::{Naive, SimpleMovingAverage};
use crate::models::ensemble::{CombinationMethod, Ensemble};
use crate::models::{Forecaster, ModelRegistry};
use crate::seasonality::auto_trend::AutoTrend;
use crate::seasonality::polynomial::PolynomialTrend;
use crate::seasonality::traits::Recency;

use super::confidence::{ModelConfidenceSet, QualityFloor, SelectionConfidence};
use super::decision_log::{DecisionCategory, DecisionLog, DecisionOutcome};
use super::fallback::FallbackChain;
use super::horizon::HorizonAnalysis;
use super::metadata::{ExecutionMetadata, ExecutionTimer};
use super::metric_strategy::{MetricScores, MetricStrategy};
use super::preprocess::{
    apply_preprocessing, invert_boxcox_forecast, PreprocessMode, PreprocessResult,
};
use super::profile::DataProfile;
use super::report::PipelineReport;
use super::trend_integration::TrendIntegrationState;

/// How trend decomposition should be handled in the pipeline.
#[derive(Debug, Clone, Default)]
pub enum TrendMode {
    /// No trend decomposition (default).
    #[default]
    None,
    /// Automatically select the best trend component via IC criteria.
    Auto,
    /// Use a specific named trend component.
    Fixed(String),
}

/// How changepoint detection should inform model training.
///
/// When enabled, the pipeline uses detected changepoints to adapt how
/// models are trained. The adaptation is context-aware: models are only
/// trained on post-changepoint data when there is enough data for the
/// model class (e.g., seasonal models need at least 2 full cycles).
///
/// # Example
///
/// ```ignore
/// use anofox_forecast::orchestration::ChangepointMode;
///
/// let pipeline = PipelineBuilder::new()
///     .changepoint(ChangepointMode::Auto)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub enum ChangepointMode {
    /// Ignore changepoints — train on all data (default).
    #[default]
    None,
    /// Automatically adapt training based on detected changepoints.
    ///
    /// The pipeline will:
    /// - Detect changepoints during profiling (via PELT with LinearTrend cost)
    /// - If enough post-changepoint data exists (≥ `min_observations` and
    ///   ≥ `2 × seasonal_period`), train models on post-changepoint data only
    /// - Otherwise, train on all data (preserving seasonal model viability)
    /// - Log the decision in the audit trail
    Auto,
    /// Train models only on data after the given index.
    ///
    /// Use this when you know the changepoint location from domain knowledge.
    /// The pipeline will still check that enough data remains for training.
    FitFrom(usize),
}

/// How seasonal decomposition should be handled in the pipeline.
#[derive(Debug, Clone, Default)]
pub enum SeasonalMode {
    /// No seasonal decomposition (default).
    #[default]
    None,
    /// Automatically select the best seasonal component via IC criteria.
    Auto,
    /// Use a specific named seasonal component.
    Fixed(String),
}

/// How a fitted trend component integrates with the forecasting models.
///
/// When a trend component is selected (via `TrendMode::Auto` or `TrendMode::Fixed`),
/// `TrendIntegration` controls whether the trend is removed before forecasting
/// (classical decomposition) or passed as a feature to the models.
///
/// # Example
///
/// ```ignore
/// use anofox_forecast::orchestration::{TrendMode, TrendIntegration, PipelineBuilder};
///
/// // Decompose: detrend → forecast residuals → recompose
/// let pipeline = PipelineBuilder::new()
///     .trend(TrendMode::Auto)
///     .trend_integration(TrendIntegration::Decompose)
///     .build();
///
/// // Regressor: pass trend as exogenous feature (models with exog support)
/// let pipeline = PipelineBuilder::new()
///     .trend(TrendMode::Auto)
///     .trend_integration(TrendIntegration::Regressor)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub enum TrendIntegration {
    /// No trend integration — trend component is computed but not used in forecasting (default).
    /// Useful when you only want trend features for analysis.
    #[default]
    None,
    /// Detrend → forecast residuals → recompose.
    ///
    /// The fitted trend is subtracted from the series before model training.
    /// After forecasting, the predicted trend is added back. Works with **all**
    /// models regardless of exogenous variable support.
    Decompose,
    /// Pass the fitted trend as an exogenous regressor.
    ///
    /// The trend values are added to the TimeSeries as a regressor named `"__trend"`.
    /// Models with `supports_exog()` will use it during fitting and receive
    /// predicted trend values during forecasting. Models without exog support
    /// will ignore it.
    Regressor,
}

/// How ensemble construction should be handled.
#[derive(Debug, Clone, Default)]
pub enum EnsembleMode {
    /// Ensemble the models in the Model Confidence Set (if > 1 model).
    Auto,
    /// Always ensemble top-k with the given method.
    Fixed(CombinationMethod),
    /// No ensemble — pick the single best model (default).
    #[default]
    None,
}

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
    /// Preprocessing mode.
    pub preprocess: PreprocessMode,
    /// Metric strategy for model selection.
    pub metric_strategy: MetricStrategy,
    /// Ensemble construction mode.
    pub ensemble_mode: EnsembleMode,
    /// Conformal postprocessing coverage (0 = disabled).
    pub postprocess_coverage: f64,
    /// Trend decomposition mode.
    pub trend_mode: TrendMode,
    /// How the trend integrates with forecasting models.
    pub trend_integration: TrendIntegration,
    /// Seasonal decomposition mode.
    pub seasonal_mode: SeasonalMode,
    /// Changepoint adaptation mode.
    pub changepoint_mode: ChangepointMode,
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
            preprocess: PreprocessMode::None,
            metric_strategy: MetricStrategy::default(),
            ensemble_mode: EnsembleMode::None,
            postprocess_coverage: 0.0,
            trend_mode: TrendMode::None,
            trend_integration: TrendIntegration::None,
            seasonal_mode: SeasonalMode::None,
            changepoint_mode: ChangepointMode::None,
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
    /// Preprocessing info (if preprocessing was applied).
    pub preprocess: Option<PreprocessResult>,
    /// Ensemble model weights (if ensemble mode was used).
    pub ensemble_weights: Option<Vec<(String, f64)>>,
    /// Per-model metric scores from multi-metric evaluation.
    pub metric_scores: Option<Vec<(String, MetricScores)>>,
    /// Trend component selection result (if trend mode was Auto).
    pub trend_selection: Option<TrendSelectionResult>,
    /// Seasonal component selection result (if seasonal mode was Auto).
    pub seasonal_selection: Option<SeasonalSelectionResult>,
}

/// Result of automatic trend component selection.
#[derive(Debug, Clone)]
pub struct TrendSelectionResult {
    /// Name of the selected component.
    pub selected: String,
    /// Criterion used for selection.
    pub criterion: String,
    /// All candidates ranked by score (name, score).
    pub scores: Vec<(String, f64)>,
}

/// Result of automatic seasonal component selection.
#[derive(Debug, Clone)]
pub struct SeasonalSelectionResult {
    /// Name of the selected component.
    pub selected: String,
    /// Criterion used for selection.
    pub criterion: String,
    /// All candidates ranked by score (name, score).
    pub scores: Vec<(String, f64)>,
}

impl PipelineResult {
    /// Build a structured report from this result.
    pub fn report(&self) -> PipelineReport {
        PipelineReport::from_result(self)
    }
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
        if let Some(ref pp) = self.preprocess {
            if !pp.steps_applied.is_empty() {
                writeln!(f, "{}", pp)?;
            }
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
        if let Some(ref weights) = self.ensemble_weights {
            let parts: Vec<String> = weights
                .iter()
                .map(|(n, w)| format!("{}({:.2})", n, w))
                .collect();
            writeln!(f, "Ensemble: {}", parts.join(" + "))?;
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

    /// Set the preprocessing mode.
    pub fn preprocess(mut self, mode: PreprocessMode) -> Self {
        self.config.preprocess = mode;
        self
    }

    /// Set the metric strategy for model selection.
    pub fn metric(mut self, strategy: MetricStrategy) -> Self {
        self.config.metric_strategy = strategy;
        self
    }

    /// Set the ensemble mode.
    pub fn ensemble(mut self, mode: EnsembleMode) -> Self {
        self.config.ensemble_mode = mode;
        self
    }

    /// Set the trend decomposition mode.
    pub fn trend(mut self, mode: TrendMode) -> Self {
        self.config.trend_mode = mode;
        self
    }

    /// Set how a fitted trend component integrates with forecasting models.
    ///
    /// Requires `TrendMode::Auto` or `TrendMode::Fixed` to have an effect.
    pub fn trend_integration(mut self, mode: TrendIntegration) -> Self {
        self.config.trend_integration = mode;
        self
    }

    /// Set the seasonal decomposition mode.
    pub fn seasonal(mut self, mode: SeasonalMode) -> Self {
        self.config.seasonal_mode = mode;
        self
    }

    /// Set the changepoint adaptation mode.
    ///
    /// When set to `ChangepointMode::Auto`, the pipeline will detect changepoints
    /// during profiling and train models only on post-changepoint data when
    /// sufficient data is available.
    pub fn changepoint(mut self, mode: ChangepointMode) -> Self {
        self.config.changepoint_mode = mode;
        self
    }

    /// Enable conformal postprocessing at the given coverage level (e.g. 0.90).
    #[cfg(feature = "postprocess")]
    pub fn postprocess(mut self, coverage: f64) -> Self {
        self.config.postprocess_coverage = coverage;
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
            let cp_info = if let Some(last_cp) = p.last_changepoint {
                format!(
                    ", changepoints={} (last at {})",
                    p.changepoints.len(),
                    last_cp
                )
            } else {
                String::new()
            };
            log.record_timed(
                DecisionCategory::DataProfiling,
                format!(
                    "Profiled {} observations: trend={}, stationary={}{}",
                    p.n_observations, p.trend_direction, p.adf_is_stationary, cp_info
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

        // ── Step 2: Preprocessing ───────────────────────────────────
        let (working_ts, preprocess_result) =
            apply_preprocessing(ts, &self.config.preprocess, profile.as_ref())?;
        let has_preprocess = !preprocess_result.steps_applied.is_empty();
        if has_preprocess {
            log.record_with_detail(
                DecisionCategory::Preprocessing,
                format!("Applied {}", preprocess_result),
                DecisionOutcome::Success,
                preprocess_result.steps_applied.join(", "),
            );
        }

        // ── Step 2b: Changepoint-aware data adaptation ──────────────
        let working_ts =
            self.apply_changepoint_adaptation(&working_ts, profile.as_ref(), &mut log)?;

        // ── Step 2c: Trend integration ──────────────────────────────
        let (working_ts, trend_state, trend_selection) =
            self.apply_trend_integration(&working_ts, effective_horizon, &mut log)?;

        // ── Step 3: Model selection via registry ────────────────────
        let registry = match &self.registry {
            Some(r) if !r.is_empty() => r,
            _ => {
                return self.execute_fallback_only(
                    &working_ts,
                    effective_horizon,
                    log,
                    model_metadata,
                    profile,
                    if has_preprocess {
                        Some(preprocess_result)
                    } else {
                        None
                    },
                );
            }
        };

        let holdout = if self.config.holdout > 0 {
            self.config.holdout
        } else {
            effective_horizon
        };
        let n = working_ts.len();

        if n <= holdout {
            return self.execute_fallback_only(
                &working_ts,
                effective_horizon,
                log,
                model_metadata,
                profile,
                if has_preprocess {
                    Some(preprocess_result)
                } else {
                    None
                },
            );
        }

        // Train/test split for holdout evaluation
        let train_ts = working_ts.slice(0, n - holdout)?;
        let test_values = working_ts.values(0)?;
        let test_actual = &test_values[n - holdout..];

        // Resolve metric strategy
        let is_intermittent = profile.as_ref().is_some_and(|p| p.is_intermittent);
        let has_negatives = profile.as_ref().is_some_and(|p| p.has_negatives);
        let metric_desc = self
            .config
            .metric_strategy
            .description(is_intermittent, has_negatives);
        log.record_with_detail(
            DecisionCategory::ModelSelection,
            "Metric strategy resolved",
            DecisionOutcome::Success,
            &metric_desc,
        );

        // Evaluate all registry models on holdout
        let mut scored: Vec<(String, f64, MetricScores)> = Vec::new();
        let mut per_obs_losses: Vec<(String, Vec<f64>)> = Vec::new();

        // Build future regressors for holdout evaluation in Regressor mode
        let holdout_future_regs = trend_state.as_ref().and_then(|ts_state| {
            if matches!(self.config.trend_integration, TrendIntegration::Regressor) {
                // For holdout eval, the "future" trend is the fitted trend for the holdout period
                // We need to use the tail of the fitted trend for the test portion
                let fitted = ts_state.fitted_trend();
                if fitted.len() >= holdout {
                    let holdout_trend = fitted[fitted.len() - holdout..].to_vec();
                    let mut regs = HashMap::new();
                    regs.insert(
                        super::trend_integration::TREND_REGRESSOR_NAME.to_string(),
                        holdout_trend,
                    );
                    Some(regs)
                } else {
                    None
                }
            } else {
                None
            }
        });

        for spec in registry.iter() {
            let timer = ExecutionTimer::start();
            let mut model = spec.create();
            let name = model.name().to_string();

            let predict_result = model.fit(&train_ts).and_then(|_| {
                if let Some(ref regs) = holdout_future_regs {
                    if model.supports_exog() && model.has_exog() {
                        model.predict_with_exog(holdout, regs)
                    } else {
                        model.predict(holdout)
                    }
                } else {
                    model.predict(holdout)
                }
            });

            match predict_result {
                Ok(forecast) => {
                    let dur = timer.stop();
                    let preds = forecast.primary();
                    let ms = self.config.metric_strategy.score(
                        test_actual,
                        preds,
                        is_intermittent,
                        has_negatives,
                    );
                    let obs_losses: Vec<f64> = test_actual
                        .iter()
                        .zip(preds.iter())
                        .map(|(a, p)| (a - p).abs())
                        .collect();
                    log.record_full(
                        DecisionCategory::ModelFitting,
                        format!("Fitted {}", name),
                        DecisionOutcome::Success,
                        Some(format!("{}", ms)),
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
                    scored.push((name, ms.primary, ms));
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

        // Sort by composite score (lower is better)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if scored.is_empty() {
            log.record(
                DecisionCategory::ModelSelection,
                "All models failed, using fallback",
                DecisionOutcome::FallbackUsed,
            );
            return self.execute_fallback_only(
                &working_ts,
                effective_horizon,
                log,
                model_metadata,
                profile,
                if has_preprocess {
                    Some(preprocess_result)
                } else {
                    None
                },
            );
        }

        // Collect metric scores before truncation
        let all_metric_scores: Vec<(String, MetricScores)> = scored
            .iter()
            .map(|(name, _, ms)| (name.clone(), ms.clone()))
            .collect();

        // Select top-k if configured
        if self.config.select_top_k > 0 {
            let top_k = self.config.select_top_k.min(scored.len());
            let top_names: std::collections::HashSet<&str> =
                scored[..top_k].iter().map(|(n, _, _)| n.as_str()).collect();
            per_obs_losses.retain(|(n, _)| top_names.contains(n.as_str()));
            scored.truncate(top_k);
        }

        // ── Quality floor: SPA test vs Naive ────────────────────────
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

        // ── Model Confidence Set ────────────────────────────────────
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

        // ── Selection confidence (DM test, top-2) ───────────────────
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

        // ── Decide: ensemble or single winner ───────────────────────
        let (forecast, final_model_name, ensemble_weights) = self.produce_forecast(
            registry,
            &working_ts,
            effective_horizon,
            &scored,
            &model_confidence_set,
            trend_state.as_ref(),
            &mut log,
        )?;

        // Handle forecast failure with fallback
        let (forecast, final_model_name, ensemble_weights) = match forecast {
            Some(f) => (f, final_model_name, ensemble_weights),
            None if self.config.use_fallback => {
                log.record(
                    DecisionCategory::Fallback,
                    "Winner failed on full data, using fallback",
                    DecisionOutcome::FallbackUsed,
                );
                return self.execute_fallback_only(
                    &working_ts,
                    effective_horizon,
                    log,
                    model_metadata,
                    profile,
                    if has_preprocess {
                        Some(preprocess_result)
                    } else {
                        None
                    },
                );
            }
            None => {
                return Err(ForecastError::ComputationError(
                    "Selected model failed on full data".into(),
                ));
            }
        };

        // ── Recompose trend (Decompose mode) ─────────────────────────
        let forecast = if matches!(self.config.trend_integration, TrendIntegration::Decompose) {
            if let Some(ref state) = trend_state {
                let mut f = forecast;
                state.recompose_forecast(&mut f);
                log.record(
                    DecisionCategory::TrendSelection,
                    "Recomposed trend onto forecast",
                    DecisionOutcome::Success,
                );
                f
            } else {
                forecast
            }
        } else {
            forecast
        };

        // ── Postprocessing (conformal calibration) ──────────────────
        #[cfg(feature = "postprocess")]
        let forecast = if self.config.postprocess_coverage > 0.0 {
            self.apply_postprocessing(
                &working_ts,
                &forecast,
                &final_model_name,
                registry,
                &mut log,
            )
        } else {
            forecast
        };

        // ── Invert preprocessing on forecast ────────────────────────
        let forecast = if let Some(lambda) = preprocess_result.boxcox_lambda {
            let inverted = invert_boxcox_forecast(forecast.primary(), lambda);
            log.record(
                DecisionCategory::Postprocessing,
                "Inverted Box-Cox on forecast",
                DecisionOutcome::Success,
            );
            Forecast::from_values(inverted)
        } else {
            forecast
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
            model_name: final_model_name,
            profile,
            log,
            model_metadata,
            horizon_analysis: None,
            selection_confidence,
            model_confidence_set,
            quality_floor,
            preprocess: if has_preprocess {
                Some(preprocess_result)
            } else {
                None
            },
            ensemble_weights,
            metric_scores: Some(all_metric_scores),
            trend_selection,
            seasonal_selection: None,
        })
    }

    /// Produce the final forecast — either single model or ensemble.
    fn produce_forecast(
        &self,
        registry: &ModelRegistry,
        ts: &TimeSeries,
        horizon: usize,
        scored: &[(String, f64, MetricScores)],
        mcs: &Option<ModelConfidenceSet>,
        trend_state: Option<&TrendIntegrationState>,
        log: &mut DecisionLog,
    ) -> Result<(Option<Forecast>, String, Option<Vec<(String, f64)>>)> {
        // Build future regressors for Regressor mode
        let future_regs = trend_state.and_then(|state| {
            if matches!(self.config.trend_integration, TrendIntegration::Regressor) {
                Some(state.future_regressors(horizon, None))
            } else {
                None
            }
        });

        match &self.config.ensemble_mode {
            EnsembleMode::None => {
                let winner_name = scored[0].0.clone();
                log.record_with_detail(
                    DecisionCategory::ModelSelection,
                    format!("Selected {}", winner_name),
                    DecisionOutcome::Success,
                    format!(
                        "{}, {} of {} models succeeded",
                        scored[0].2,
                        scored.len(),
                        registry.len()
                    ),
                );

                let forecast = self.fit_and_predict_model(
                    registry,
                    &winner_name,
                    ts,
                    horizon,
                    future_regs.as_ref(),
                    log,
                );
                match forecast {
                    Ok(f) => Ok((Some(f), winner_name, None)),
                    Err(_) => Ok((None, winner_name, None)),
                }
            }
            EnsembleMode::Auto => {
                // Use MCS included models if > 1, otherwise single winner
                let ensemble_names: Vec<&str> = match mcs {
                    Some(mcs) if mcs.len() > 1 => mcs.included.iter().map(|s| s.as_str()).collect(),
                    _ => vec![scored[0].0.as_str()],
                };

                if ensemble_names.len() <= 1 {
                    let winner = ensemble_names[0].to_string();
                    log.record_with_detail(
                        DecisionCategory::ModelSelection,
                        format!("Selected {} (MCS single winner)", winner),
                        DecisionOutcome::Success,
                        format!("{}", scored[0].2),
                    );
                    let forecast = self.fit_and_predict_model(
                        registry,
                        &winner,
                        ts,
                        horizon,
                        future_regs.as_ref(),
                        log,
                    );
                    match forecast {
                        Ok(f) => Ok((Some(f), winner, None)),
                        Err(_) => Ok((None, winner, None)),
                    }
                } else {
                    self.build_ensemble(
                        registry,
                        ts,
                        horizon,
                        &ensemble_names,
                        CombinationMethod::WeightedMSE,
                        future_regs.as_ref(),
                        log,
                    )
                }
            }
            EnsembleMode::Fixed(method) => {
                let names: Vec<&str> = scored.iter().map(|(n, _, _)| n.as_str()).collect();
                self.build_ensemble(
                    registry,
                    ts,
                    horizon,
                    &names,
                    *method,
                    future_regs.as_ref(),
                    log,
                )
            }
        }
    }

    /// Build and run an ensemble from named models.
    fn build_ensemble(
        &self,
        registry: &ModelRegistry,
        ts: &TimeSeries,
        horizon: usize,
        names: &[&str],
        method: CombinationMethod,
        future_regs: Option<&HashMap<String, Vec<f64>>>,
        log: &mut DecisionLog,
    ) -> Result<(Option<Forecast>, String, Option<Vec<(String, f64)>>)> {
        let mut models: Vec<Box<dyn Forecaster>> = Vec::new();
        let mut used_names: Vec<String> = Vec::new();

        for &name in names {
            for spec in registry.iter() {
                let mut model = spec.create();
                if model.name() == name {
                    if model.fit(ts).is_ok() {
                        used_names.push(name.to_string());
                        models.push(model);
                    }
                    break;
                }
            }
        }

        if models.is_empty() {
            return Ok((None, "Ensemble(empty)".into(), None));
        }

        if models.len() == 1 {
            let name = used_names[0].clone();
            let model = &models[0];
            let forecast = if let Some(regs) = future_regs {
                if model.supports_exog() && model.has_exog() {
                    model.predict_with_exog(horizon, regs)
                } else {
                    model.predict(horizon)
                }
            } else {
                model.predict(horizon)
            };
            match forecast {
                Ok(f) => return Ok((Some(f), name, None)),
                Err(_) => return Ok((None, name, None)),
            }
        }

        let ensemble_name = format!("Ensemble({})", used_names.join("+"));

        let n_models = models.len();
        let mut ensemble = Ensemble::new(models).with_method(method);
        ensemble.fit(ts)?;
        // Note: Ensemble.predict() delegates to individual model predict() calls.
        // For Regressor mode, regressors are already embedded in the TimeSeries,
        // so models that picked them up during fit will use them via their internal state.
        let forecast = ensemble.predict(horizon)?;

        let weights: Vec<(String, f64)> = used_names
            .iter()
            .zip(ensemble.weights().iter())
            .map(|(n, &w)| (n.clone(), w))
            .collect();

        log.record_with_detail(
            DecisionCategory::Ensembling,
            format!("Built ensemble of {} models", n_models),
            DecisionOutcome::Success,
            weights
                .iter()
                .map(|(n, w)| format!("{}={:.3}", n, w))
                .collect::<Vec<_>>()
                .join(", "),
        );

        Ok((Some(forecast), ensemble_name, Some(weights)))
    }

    /// Apply conformal postprocessing to a forecast.
    #[cfg(feature = "postprocess")]
    fn apply_postprocessing(
        &self,
        ts: &TimeSeries,
        forecast: &Forecast,
        model_name: &str,
        registry: &ModelRegistry,
        log: &mut DecisionLog,
    ) -> Forecast {
        use crate::postprocess::{ConformalPredictor, PointForecasts};

        // Re-fit model and get in-sample residuals for calibration
        let n = ts.len();
        let horizon = forecast.primary().len();
        if n <= horizon + 10 {
            return forecast.clone();
        }

        // Use a holdout to generate calibration residuals
        let cal_size = (n / 4).max(horizon).min(n / 2);
        let cal_train = match ts.slice(0, n - cal_size) {
            Ok(t) => t,
            Err(_) => return forecast.clone(),
        };

        // Fit the model on calibration training set
        for spec in registry.iter() {
            let mut model = spec.create();
            if model.name() == model_name {
                if model.fit(&cal_train).is_ok() {
                    if let Ok(cal_fc) = model.predict(cal_size) {
                        let cal_actual = match ts.values(0) {
                            Ok(v) => &v[n - cal_size..],
                            Err(_) => return forecast.clone(),
                        };

                        let cp = ConformalPredictor::split(self.config.postprocess_coverage);
                        if let Ok(result) = cp.fit(cal_fc.primary(), cal_actual) {
                            let pf = PointForecasts::from_values(forecast.primary().to_vec());
                            let intervals = cp.predict(&result, &pf);

                            log.record_with_detail(
                                DecisionCategory::Postprocessing,
                                format!(
                                    "Applied conformal calibration ({:.0}%)",
                                    self.config.postprocess_coverage * 100.0
                                ),
                                DecisionOutcome::Success,
                                format!("quantile={:.4}", result.quantile_value()),
                            );

                            return Forecast::from_values_with_intervals(
                                forecast.primary().to_vec(),
                                intervals.lower().to_vec(),
                                intervals.upper().to_vec(),
                            );
                        }
                    }
                }
                break;
            }
        }

        forecast.clone()
    }

    /// Apply trend integration to the working time series.
    ///
    /// Depending on `TrendMode` and `TrendIntegration`, this:
    /// - Fits a trend component (Auto or Fixed)
    /// - For Decompose: detrends the series (subtract fitted trend)
    /// - For Regressor: adds fitted trend as an exogenous regressor `"__trend"`
    /// - Returns the (possibly modified) series, the trend state, and selection result
    fn apply_trend_integration(
        &self,
        ts: &TimeSeries,
        horizon: usize,
        log: &mut DecisionLog,
    ) -> Result<(
        TimeSeries,
        Option<TrendIntegrationState>,
        Option<TrendSelectionResult>,
    )> {
        // Check if trend mode is enabled
        if matches!(self.config.trend_mode, TrendMode::None) {
            return Ok((ts.clone(), None, None));
        }

        let values = ts.values(0)?;
        let timer = ExecutionTimer::start();

        // Fit the trend component
        let (state, selection_result) = match &self.config.trend_mode {
            TrendMode::Auto => {
                let mut auto = AutoTrend::new();
                let trend_state =
                    TrendIntegrationState::from_component(&mut auto, values, horizon)?;
                let dur = timer.stop();

                let sel = auto.selection_result().map(|r| {
                    let scores_desc: String = r
                        .scores
                        .iter()
                        .take(3)
                        .map(|(n, s)| format!("{}={:.1}", n, s))
                        .collect::<Vec<_>>()
                        .join(", ");
                    log.record_full(
                        DecisionCategory::TrendSelection,
                        format!("Selected {} ({:?})", r.selected, r.criterion),
                        DecisionOutcome::Success,
                        Some(scores_desc),
                        Some(dur),
                    );
                    TrendSelectionResult {
                        selected: r.selected.clone(),
                        criterion: format!("{:?}", r.criterion),
                        scores: r.scores.clone(),
                    }
                });

                (trend_state, sel)
            }
            TrendMode::Fixed(name) => {
                // For now, Fixed mode supports polynomial trends by degree
                // e.g., "linear" → degree 1, "quadratic" → degree 2
                let degree = match name.to_lowercase().as_str() {
                    "linear" | "poly1" => 1,
                    "quadratic" | "poly2" => 2,
                    "cubic" | "poly3" => 3,
                    _ => {
                        // Try parsing as a number for degree
                        name.parse::<usize>().unwrap_or(1)
                    }
                };
                let mut poly = PolynomialTrend::new(degree).with_recency(Recency::Fraction(0.3));
                let trend_state =
                    TrendIntegrationState::from_component(&mut poly, values, horizon)?;
                let dur = timer.stop();
                log.record_full(
                    DecisionCategory::TrendSelection,
                    format!("Fitted fixed trend: {} (degree {})", name, degree),
                    DecisionOutcome::Success,
                    None,
                    Some(dur),
                );
                (
                    trend_state,
                    Some(TrendSelectionResult {
                        selected: name.clone(),
                        criterion: "Fixed".to_string(),
                        scores: vec![],
                    }),
                )
            }
            TrendMode::None => unreachable!(),
        };

        // Apply integration mode
        let modified_ts = match &self.config.trend_integration {
            TrendIntegration::None => {
                log.record(
                    DecisionCategory::TrendSelection,
                    "Trend fitted but not integrated (analysis only)",
                    DecisionOutcome::Success,
                );
                ts.clone()
            }
            TrendIntegration::Decompose => {
                let detrended = state.detrend_series(ts)?;
                log.record_with_detail(
                    DecisionCategory::TrendSelection,
                    "Detrended series (Decompose mode)",
                    DecisionOutcome::Success,
                    format!("Removed trend from {} observations", detrended.len()),
                );
                detrended
            }
            TrendIntegration::Regressor => {
                let with_reg = state.add_trend_regressor(ts)?;
                log.record_with_detail(
                    DecisionCategory::TrendSelection,
                    "Added trend as regressor (Regressor mode)",
                    DecisionOutcome::Success,
                    format!(
                        "Added '{}' regressor ({} values)",
                        super::trend_integration::TREND_REGRESSOR_NAME,
                        with_reg.len()
                    ),
                );
                with_reg
            }
        };

        Ok((modified_ts, Some(state), selection_result))
    }

    /// Apply changepoint-aware data adaptation.
    ///
    /// If a changepoint was detected and there is enough post-changepoint data,
    /// returns a sliced TimeSeries starting from the last changepoint.
    /// Otherwise returns the input unchanged.
    ///
    /// "Enough data" means:
    /// - At least 30 observations (absolute minimum for most models)
    /// - At least `2 × seasonal_period` observations (if seasonal period is set)
    /// - At least `horizon + holdout` observations (need room for evaluation)
    fn apply_changepoint_adaptation(
        &self,
        ts: &TimeSeries,
        profile: Option<&DataProfile>,
        log: &mut DecisionLog,
    ) -> Result<TimeSeries> {
        let fit_from = match &self.config.changepoint_mode {
            ChangepointMode::None => {
                return Ok(ts.clone());
            }
            ChangepointMode::Auto => {
                // Use profiled changepoints
                match profile.and_then(|p| p.last_changepoint) {
                    Some(cp) => cp,
                    None => {
                        log.record(
                            DecisionCategory::ChangepointAdaptation,
                            "No changepoints detected, using full data",
                            DecisionOutcome::Skipped,
                        );
                        return Ok(ts.clone());
                    }
                }
            }
            ChangepointMode::FitFrom(idx) => *idx,
        };

        let n = ts.len();
        if fit_from >= n {
            log.record_with_detail(
                DecisionCategory::ChangepointAdaptation,
                "Changepoint index beyond data length, using full data",
                DecisionOutcome::Skipped,
                format!("fit_from={}, n={}", fit_from, n),
            );
            return Ok(ts.clone());
        }

        let post_cp_len = n - fit_from;

        // Minimum data requirements
        let min_abs = 30; // Absolute minimum for robust fitting
        let min_seasonal = if self.config.seasonal_period > 0 {
            2 * self.config.seasonal_period // Need at least 2 full cycles
        } else {
            0
        };
        let holdout = if self.config.holdout > 0 {
            self.config.holdout
        } else {
            self.config.horizon
        };
        let min_for_eval = holdout + 10; // Need holdout + some training data
        let min_required = min_abs.max(min_seasonal).max(min_for_eval);

        if post_cp_len < min_required {
            log.record_with_detail(
                DecisionCategory::ChangepointAdaptation,
                format!(
                    "Changepoint at {} but only {} post-CP observations (need ≥{}), using full data",
                    fit_from, post_cp_len, min_required
                ),
                DecisionOutcome::Skipped,
                format!(
                    "min_abs={}, min_seasonal={}, min_eval={}",
                    min_abs, min_seasonal, min_for_eval
                ),
            );
            return Ok(ts.clone());
        }

        // Enough data — truncate
        let sliced = ts.slice(fit_from, n)?;
        log.record_with_detail(
            DecisionCategory::ChangepointAdaptation,
            format!(
                "Training from changepoint at index {} ({} observations, was {})",
                fit_from, post_cp_len, n
            ),
            DecisionOutcome::Success,
            format!(
                "Dropped {} pre-changepoint observations ({:.0}% of data)",
                fit_from,
                fit_from as f64 / n as f64 * 100.0
            ),
        );

        Ok(sliced)
    }

    /// Fallback-only path when no registry or all models fail.
    fn execute_fallback_only(
        &self,
        ts: &TimeSeries,
        horizon: usize,
        mut log: DecisionLog,
        model_metadata: Vec<ExecutionMetadata>,
        profile: Option<DataProfile>,
        preprocess: Option<PreprocessResult>,
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
            preprocess,
            ensemble_weights: None,
            metric_scores: None,
            trend_selection: None,
            seasonal_selection: None,
        })
    }

    /// Re-fit a named model from the registry on full data and predict.
    fn fit_and_predict_model(
        &self,
        registry: &ModelRegistry,
        name: &str,
        ts: &TimeSeries,
        horizon: usize,
        future_regs: Option<&HashMap<String, Vec<f64>>>,
        _log: &mut DecisionLog,
    ) -> Result<Forecast> {
        for spec in registry.iter() {
            let mut model = spec.create();
            if model.name() == name {
                model.fit(ts)?;
                let use_exog = future_regs.is_some() && model.supports_exog() && model.has_exog();
                let forecast = if self.config.interval_level > 0.0 {
                    // predict_with_intervals doesn't have an exog variant,
                    // so we fall back to predict_with_exog when exog is needed
                    if use_exog {
                        model.predict_with_exog(horizon, future_regs.unwrap())?
                    } else {
                        model.predict_with_intervals(horizon, self.config.interval_level)?
                    }
                } else if use_exog {
                    model.predict_with_exog(horizon, future_regs.unwrap())?
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
    use crate::orchestration::metric_strategy::Metric;
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
        let config = PipelineConfig::default();
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

    #[test]
    fn pipeline_multi_metric_scoring() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .metric(MetricStrategy::Composite(vec![
                (Metric::MAE, 0.5),
                (Metric::SMAPE, 0.5),
            ]))
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(result.metric_scores.is_some());
        let scores = result.metric_scores.unwrap();
        assert!(!scores.is_empty());
        // Each model should have component scores
        for (_, ms) in &scores {
            assert_eq!(ms.components.len(), 2);
        }
    }

    #[test]
    fn pipeline_auto_metric_strategy() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .metric(MetricStrategy::Auto)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(result.metric_scores.is_some());
    }

    #[test]
    fn pipeline_ensemble_fixed() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .ensemble(EnsembleMode::Fixed(CombinationMethod::Mean))
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
        // Should have ensemble weights
        assert!(result.ensemble_weights.is_some());
    }

    #[test]
    fn pipeline_ensemble_auto() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .ensemble(EnsembleMode::Auto)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
    }

    #[test]
    fn pipeline_report() {
        let ts = make_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .profile()
            .registry(reg)
            .build()
            .execute(&ts, 5)
            .unwrap();

        let report = result.report();
        assert!(report.title.contains(&result.model_name));
        let text = format!("{}", report);
        assert!(text.contains("Summary"));
        assert!(text.contains("Forecast"));
    }

    #[test]
    fn pipeline_preprocess_manual() {
        // Exponential data suitable for Box-Cox
        let values: Vec<f64> = (1..=50).map(|i| (i as f64 * 0.1).exp()).collect();
        let start = Utc::now();
        let timestamps: Vec<_> = (0..50).map(|i| start + Duration::days(i as i64)).collect();
        let ts = TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .values(values)
            .build()
            .unwrap();
        let reg = make_registry();

        let result = PipelineBuilder::new()
            .registry(reg)
            .preprocess(PreprocessMode::Manual(
                crate::orchestration::preprocess::PreprocessSteps {
                    boxcox: true,
                    outlier_treatment: false,
                    outlier_window: 5,
                },
            ))
            .build()
            .execute(&ts, 5)
            .unwrap();

        // Should have preprocess info
        assert!(result.preprocess.is_some());
        let pp = result.preprocess.unwrap();
        assert!(pp.boxcox_lambda.is_some());
    }

    /// Helper: make a trending time series for trend integration tests.
    fn make_trending_ts(n: usize) -> TimeSeries {
        // y = 2*t + 10 + small noise
        let values: Vec<f64> = (0..n)
            .map(|i| 2.0 * i as f64 + 10.0 + 0.1 * (i as f64 * 0.7).sin())
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
    fn pipeline_trend_decompose_mode() {
        let ts = make_trending_ts(80);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .trend(TrendMode::Auto)
            .trend_integration(TrendIntegration::Decompose)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
        // With strong upward trend, forecast should continue upward
        let last_actual = ts.values(0).unwrap().last().copied().unwrap();
        let first_pred = result.forecast.primary()[0];
        // Forecast should be in a reasonable range of the last value
        assert!(
            (first_pred - last_actual).abs() < 20.0,
            "Decompose forecast too far from last value: pred={}, last={}",
            first_pred,
            last_actual
        );
        // Trend selection should be populated
        assert!(result.trend_selection.is_some());
        // Decision log should contain trend entries
        let trend_decisions = result.log.by_category(DecisionCategory::TrendSelection);
        assert!(
            trend_decisions.len() >= 2,
            "Expected trend selection + detrend + recompose decisions, got {}",
            trend_decisions.len()
        );
    }

    #[test]
    fn pipeline_trend_regressor_mode() {
        let ts = make_trending_ts(80);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .trend(TrendMode::Auto)
            .trend_integration(TrendIntegration::Regressor)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
        assert!(result.trend_selection.is_some());
        let trend_decisions = result.log.by_category(DecisionCategory::TrendSelection);
        assert!(
            !trend_decisions.is_empty(),
            "Expected trend selection decisions"
        );
    }

    #[test]
    fn pipeline_trend_fixed_linear() {
        let ts = make_trending_ts(80);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .trend(TrendMode::Fixed("linear".to_string()))
            .trend_integration(TrendIntegration::Decompose)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert_eq!(result.forecast.primary().len(), 5);
        let sel = result.trend_selection.unwrap();
        assert_eq!(sel.selected, "linear");
        assert_eq!(sel.criterion, "Fixed");
    }

    #[test]
    fn pipeline_trend_none_integration() {
        let ts = make_trending_ts(80);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .trend(TrendMode::Auto)
            .trend_integration(TrendIntegration::None)
            .build()
            .execute(&ts, 5)
            .unwrap();

        // Should still have selection result but series was not modified
        assert!(result.trend_selection.is_some());
        let trend_decisions = result.log.by_category(DecisionCategory::TrendSelection);
        // Should have "analysis only" entry
        let analysis_only = trend_decisions
            .iter()
            .any(|d| d.action.contains("analysis only"));
        assert!(analysis_only, "Expected 'analysis only' decision");
    }

    #[test]
    fn pipeline_trend_mode_none_skips() {
        let ts = make_trending_ts(50);
        let reg = make_registry();
        let result = PipelineBuilder::new()
            .registry(reg)
            .trend(TrendMode::None)
            .build()
            .execute(&ts, 5)
            .unwrap();

        assert!(result.trend_selection.is_none());
    }
}
