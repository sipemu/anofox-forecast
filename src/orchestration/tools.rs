//! Structured tool functions for MCP / agent integration.
//!
//! Each function has well-defined input/output types and wraps existing
//! orchestration functionality. An MCP server serializes the I/O types;
//! the functions themselves are pure Rust with no network dependency.

use crate::core::TimeSeries;
use crate::error::Result;
use crate::models::ModelRegistry;

use super::pipeline::{PipelineBuilder, PipelineConfig, PipelineResult};
use super::profile::DataProfile;
use super::report::PipelineReport;

// ---------------------------------------------------------------------------
// profile_data
// ---------------------------------------------------------------------------

/// Input for the `profile_data` tool.
#[derive(Debug, Clone)]
pub struct ProfileDataInput<'a> {
    pub series: &'a TimeSeries,
}

/// Output of the `profile_data` tool.
#[derive(Debug, Clone)]
pub struct ProfileDataOutput {
    pub profile: DataProfile,
}

/// Profile a time series and return a comprehensive data summary.
pub fn profile_data(input: ProfileDataInput<'_>) -> ProfileDataOutput {
    ProfileDataOutput {
        profile: DataProfile::from_series(input.series),
    }
}

// ---------------------------------------------------------------------------
// select_models
// ---------------------------------------------------------------------------

/// Input for the `select_models` tool.
#[derive(Debug)]
pub struct SelectModelsInput<'a> {
    pub profile: &'a DataProfile,
    pub available_models: &'a [String],
}

/// Output of the `select_models` tool.
#[derive(Debug, Clone)]
pub struct SelectModelsOutput {
    pub recommended: Vec<String>,
    pub reasoning: Vec<String>,
}

/// Recommend model families based on data profile characteristics.
///
/// This is a heuristic recommendation — not a replacement for actual
/// evaluation. It narrows the search space before the pipeline runs.
pub fn select_models(input: SelectModelsInput<'_>) -> SelectModelsOutput {
    let p = input.profile;
    let mut recommended = Vec::new();
    let mut reasoning = Vec::new();

    // Intermittent demand
    if p.is_intermittent {
        recommended.extend_from_slice(&[
            "Croston".to_string(),
            "TSB".to_string(),
            "ADIDA".to_string(),
            "SBA".to_string(),
        ]);
        reasoning.push(format!(
            "Intermittent demand detected (zero_fraction={:.1}%): recommending intermittent models",
            p.zero_fraction * 100.0
        ));
    }

    // Strong trend
    if p.trend_strength > 0.7 {
        recommended.extend_from_slice(&[
            "ARIMA".to_string(),
            "Theta".to_string(),
            "Holt".to_string(),
            "HoltWinters".to_string(),
        ]);
        reasoning.push(format!(
            "Strong trend detected (R²={:.2}, direction={}): recommending trend-capable models",
            p.trend_strength, p.trend_direction
        ));
    }

    // High autocorrelation
    if p.acf_lag1.abs() > 0.7 {
        recommended.extend_from_slice(&["ARIMA".to_string(), "ETS".to_string()]);
        reasoning.push(format!(
            "High autocorrelation (ACF lag1={:.2}): recommending ARIMA/ETS",
            p.acf_lag1
        ));
    }

    // Non-stationary
    if !p.is_stationary() {
        recommended.push("ARIMA".to_string());
        reasoning.push("Non-stationary series: ARIMA with differencing recommended".into());
    }

    // Low complexity / short series — simple models work well
    if p.n_observations < 30 {
        recommended.extend_from_slice(&["Naive".to_string(), "SES".to_string(), "SMA".to_string()]);
        reasoning.push(format!(
            "Short series ({} obs): recommending simple models",
            p.n_observations
        ));
    }

    // Always include baselines
    if !recommended.contains(&"Naive".to_string()) {
        recommended.push("Naive".to_string());
    }
    if !recommended.contains(&"SES".to_string()) {
        recommended.push("SES".to_string());
    }
    if !recommended.contains(&"ETS".to_string()) && p.n_observations >= 20 {
        recommended.push("ETS".to_string());
        reasoning.push("ETS added as general-purpose model".into());
    }

    // Deduplicate
    recommended.sort();
    recommended.dedup();

    // Filter to only available models if provided
    if !input.available_models.is_empty() {
        recommended.retain(|m| input.available_models.contains(m));
    }

    SelectModelsOutput {
        recommended,
        reasoning,
    }
}

// ---------------------------------------------------------------------------
// run_pipeline
// ---------------------------------------------------------------------------

/// Input for the `run_pipeline` tool.
pub struct RunPipelineInput<'a> {
    pub series: &'a TimeSeries,
    pub horizon: usize,
    pub registry: ModelRegistry,
    pub config: Option<PipelineConfig>,
}

/// Run a forecasting pipeline end-to-end and return the result.
pub fn run_pipeline(input: RunPipelineInput<'_>) -> Result<PipelineResult> {
    match input.config {
        Some(config) => crate::orchestration::Pipeline::from_config(config)
            .with_registry(input.registry)
            .execute(input.series, input.horizon),
        None => PipelineBuilder::new()
            .profile()
            .registry(input.registry)
            .with_fallback()
            .build()
            .execute(input.series, input.horizon),
    }
}

// ---------------------------------------------------------------------------
// explain_result
// ---------------------------------------------------------------------------

/// Verbosity level for explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainVerbosity {
    Brief,
    Normal,
    Detailed,
}

/// Input for the `explain_result` tool.
#[derive(Debug)]
pub struct ExplainResultInput<'a> {
    pub result: &'a PipelineResult,
    pub verbosity: ExplainVerbosity,
}

/// Output of the `explain_result` tool.
#[derive(Debug, Clone)]
pub struct ExplainResultOutput {
    pub summary: String,
    pub sections: Vec<(String, String)>,
    pub report: PipelineReport,
}

/// Generate a human-readable explanation of a pipeline result.
pub fn explain_result(input: ExplainResultInput<'_>) -> ExplainResultOutput {
    let r = input.result;
    let report = PipelineReport::from_result(r);

    let summary = match input.verbosity {
        ExplainVerbosity::Brief => {
            format!(
                "Selected {} for {}-step forecast.",
                r.model_name,
                r.forecast.primary().len()
            )
        }
        ExplainVerbosity::Normal | ExplainVerbosity::Detailed => {
            let mut parts = vec![format!(
                "Selected {} for {}-step forecast.",
                r.model_name,
                r.forecast.primary().len()
            )];
            if let Some(ref qf) = r.quality_floor {
                parts.push(format!(
                    "Quality floor: {} (SPA p={:.4}).",
                    if qf.is_outperformed {
                        "passed"
                    } else {
                        "failed"
                    },
                    qf.spa_p_value,
                ));
            }
            if let Some(ref conf) = r.selection_confidence {
                parts.push(format!(
                    "Selection confidence: {} (DM p={:.4}).",
                    conf.verdict, conf.dm_p_value
                ));
            }
            if let Some(ref mcs) = r.model_confidence_set {
                parts.push(format!(
                    "MCS contains {} model(s): {:?}.",
                    mcs.len(),
                    mcs.included
                ));
            }
            parts.join(" ")
        }
    };

    let mut sections = Vec::new();
    if input.verbosity == ExplainVerbosity::Detailed {
        if let Some(ref profile) = r.profile {
            sections.push(("Data Profile".into(), profile.summary()));
        }
        sections.push(("Decision Log".into(), r.log.summary()));
    }

    ExplainResultOutput {
        summary,
        sections,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Forecast, TimeSeriesBuilder};
    use crate::orchestration::decision_log::DecisionLog;
    use chrono::{Duration, Utc};

    fn make_ts(n: usize) -> TimeSeries {
        let values: Vec<f64> = (0..n).map(|i| 10.0 + i as f64 * 0.5).collect();
        let start = Utc::now();
        let timestamps: Vec<_> = (0..n).map(|i| start + Duration::days(i as i64)).collect();
        TimeSeriesBuilder::new()
            .timestamps(timestamps)
            .values(values)
            .build()
            .unwrap()
    }

    #[test]
    fn profile_data_tool() {
        let ts = make_ts(50);
        let output = profile_data(ProfileDataInput { series: &ts });
        assert_eq!(output.profile.n_observations, 50);
    }

    #[test]
    fn select_models_intermittent() {
        let mut values = vec![0.0; 50];
        values.extend(vec![10.0; 50]);
        let profile = DataProfile::from_values(&values);

        let output = select_models(SelectModelsInput {
            profile: &profile,
            available_models: &[],
        });

        if profile.is_intermittent {
            assert!(output.recommended.contains(&"Croston".to_string()));
        }
    }

    #[test]
    fn select_models_filters_available() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let profile = DataProfile::from_values(&values);

        let available = vec!["Naive".to_string(), "SES".to_string()];
        let output = select_models(SelectModelsInput {
            profile: &profile,
            available_models: &available,
        });

        for model in &output.recommended {
            assert!(available.contains(model));
        }
    }

    #[test]
    fn explain_result_brief() {
        let result = PipelineResult {
            forecast: Forecast::from_values(vec![1.0, 2.0, 3.0]),
            model_name: "SES".into(),
            profile: None,
            log: DecisionLog::new(),
            model_metadata: vec![],
            horizon_analysis: None,
            selection_confidence: None,
            model_confidence_set: None,
            quality_floor: None,
            preprocess: None,
            ensemble_weights: None,
            metric_scores: None,
        };

        let output = explain_result(ExplainResultInput {
            result: &result,
            verbosity: ExplainVerbosity::Brief,
        });

        assert!(output.summary.contains("SES"));
        assert!(output.summary.contains("3-step"));
    }
}
