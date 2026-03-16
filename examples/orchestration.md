# Orchestration

**Run:** `cargo run --example orchestration`

## What this example demonstrates

Walks through the full orchestration module: data profiling, decision logging, fallback chains, execution timing, statistical model selection confidence, horizon analysis, end-to-end pipeline execution, config replay, pipeline reports, multi-metric selection, structured tools, and abstract storage.

## Sections

1. **Data profiling** -- Runs `DataProfile::from_series` to get stationarity, trend direction, ACF, and quality score.
2. **Decision log** -- Records decisions across categories (profiling, fitting, CV) with outcomes and optional timing, then queries failures and categories.
3. **Fallback chain** -- Builds a chain of Naive, SMA, and HistoricAverage; executes with automatic fallback on failure.
4. **Execution metadata** -- Times fit and predict steps with `ExecutionTimer`, bundles into `ExecutionMetadata`.
5. **Selection confidence** -- Runs a Diebold-Mariano pairwise test, builds a Model Confidence Set from CV scores, and checks a Quality Floor (SPA test).
6. **Horizon analysis** -- Computes per-step RMSE across CV folds; identifies hardest/easiest horizons and error growth rate.
7. **Pipeline (end-to-end)** -- Configures a `PipelineBuilder` with profiling, preprocessing, metric strategy, ensemble mode, model registry, fallback, and non-negative constraint; executes and prints results.
8. **Pipeline replay** -- Serializes a `PipelineConfig` and replays it on fresh data with a different registry.
9. **Pipeline report** -- Generates a `PipelineReport` from the pipeline result.
10. **Multi-metric selection** -- Shows `MetricStrategy::Auto` descriptions for different data types and scores a custom composite metric.
11. **Structured tools** -- Calls `tools::profile_data` and `tools::select_models` (MCP-ready tool functions).
12. **Abstract storage** -- Saves and loads a `PipelineRecord` via `InMemoryStore`.

## Key types

- `DataProfile` -- statistical summary of a time series
- `DecisionLog`, `DecisionCategory`, `DecisionOutcome` -- structured audit trail
- `FallbackChain` -- ordered model fallback
- `ExecutionTimer`, `ExecutionMetadata` -- timing instrumentation
- `SelectionConfidence`, `ModelConfidenceSet`, `QualityFloor` -- statistical selection tests
- `HorizonAnalysis` -- per-horizon error decomposition
- `PipelineBuilder`, `Pipeline`, `PipelineConfig`, `PipelineResult` -- end-to-end forecasting pipeline
- `PipelineReport` -- human-readable pipeline summary
- `MetricStrategy`, `Metric` -- pluggable accuracy metrics
- `ModelRegistry`, `ModelSpec` -- model registration
- `InMemoryStore`, `PipelineStore`, `PipelineRecord` -- abstract persistence
- `tools::profile_data`, `tools::select_models` -- structured tool functions
