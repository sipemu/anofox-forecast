//! Orchestration layer for agent-based forecasting.
//!
//! Provides the building blocks for autonomous forecasting pipelines:
//! data profiling, declarative pipeline construction, decision logging,
//! fallback chains, per-horizon analysis, execution metadata,
//! selection confidence, preprocessing, multi-metric selection,
//! ensemble construction, abstract storage, structured tool functions,
//! and unified reporting.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_forecast::orchestration::prelude::*;
//!
//! // Profile the data
//! let profile = DataProfile::from_series(&ts);
//! println!("{}", profile);
//!
//! // Build and run a pipeline
//! let result = PipelineBuilder::new()
//!     .profile()
//!     .preprocess()
//!     .metric(MetricStrategy::Auto)
//!     .ensemble(EnsembleMode::Auto)
//!     .select_models(3)
//!     .cross_validate(5, 12)
//!     .with_fallback()
//!     .build()
//!     .execute(&ts, 12)?;
//!
//! println!("{}", result.report());
//! ```

pub mod confidence;
pub mod decision_log;
pub mod fallback;
pub mod horizon;
pub mod metadata;
pub mod metric_strategy;
pub mod pipeline;
pub mod preprocess;
pub mod profile;
pub mod report;
pub mod store;
pub mod tools;

// Re-export primary types
pub use confidence::{ModelConfidenceSet, QualityFloor, SelectionConfidence, SelectionVerdict};
pub use decision_log::{Decision, DecisionCategory, DecisionLog, DecisionOutcome};
pub use fallback::{FallbackChain, FallbackResult};
pub use horizon::{HorizonAnalysis, HorizonStep};
pub use metadata::{ExecutionMetadata, ExecutionTimer};
pub use metric_strategy::{Metric, MetricScores, MetricStrategy};
pub use pipeline::{EnsembleMode, Pipeline, PipelineBuilder, PipelineConfig, PipelineResult};
pub use preprocess::{PreprocessMode, PreprocessResult, PreprocessSteps};
pub use profile::{DataProfile, TrendDirection};
pub use report::PipelineReport;
pub use store::{InMemoryStore, PipelineRecord, PipelineStore, RecordKind, Storable, Value};

/// Convenience prelude for orchestration types.
pub mod prelude {
    pub use super::confidence::{
        ModelConfidenceSet, QualityFloor, SelectionConfidence, SelectionVerdict,
    };
    pub use super::decision_log::{DecisionCategory, DecisionLog, DecisionOutcome};
    pub use super::fallback::{FallbackChain, FallbackResult};
    pub use super::horizon::HorizonAnalysis;
    pub use super::metadata::{ExecutionMetadata, ExecutionTimer};
    pub use super::metric_strategy::{Metric, MetricScores, MetricStrategy};
    pub use super::pipeline::{
        EnsembleMode, Pipeline, PipelineBuilder, PipelineConfig, PipelineResult,
    };
    pub use super::preprocess::{PreprocessMode, PreprocessResult};
    pub use super::profile::{DataProfile, TrendDirection};
    pub use super::report::PipelineReport;
    pub use super::store::{PipelineStore, Storable, Value};
}
