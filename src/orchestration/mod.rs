//! Orchestration layer for agent-based forecasting.
//!
//! Provides the building blocks for autonomous forecasting pipelines:
//! data profiling, declarative pipeline construction, decision logging,
//! fallback chains, per-horizon analysis, execution metadata,
//! selection confidence, and pipeline persistence.
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
//!     .select_models(3)
//!     .cross_validate(5, 12)
//!     .with_fallback()
//!     .build()
//!     .execute(&ts, 12)?;
//!
//! println!("{}", result.log);
//! ```

pub mod confidence;
pub mod decision_log;
pub mod fallback;
pub mod horizon;
pub mod metadata;
pub mod pipeline;
pub mod profile;

// Re-export primary types
pub use confidence::{ModelConfidenceSet, QualityFloor, SelectionConfidence, SelectionVerdict};
pub use decision_log::{Decision, DecisionCategory, DecisionLog, DecisionOutcome};
pub use fallback::{FallbackChain, FallbackResult};
pub use horizon::{HorizonAnalysis, HorizonStep};
pub use metadata::{ExecutionMetadata, ExecutionTimer};
pub use pipeline::{Pipeline, PipelineBuilder, PipelineConfig, PipelineResult};
pub use profile::{DataProfile, TrendDirection};

/// Convenience prelude for orchestration types.
pub mod prelude {
    pub use super::confidence::{
        ModelConfidenceSet, QualityFloor, SelectionConfidence, SelectionVerdict,
    };
    pub use super::decision_log::{DecisionCategory, DecisionLog, DecisionOutcome};
    pub use super::fallback::{FallbackChain, FallbackResult};
    pub use super::horizon::HorizonAnalysis;
    pub use super::metadata::{ExecutionMetadata, ExecutionTimer};
    pub use super::pipeline::{Pipeline, PipelineBuilder, PipelineConfig, PipelineResult};
    pub use super::profile::{DataProfile, TrendDirection};
}
