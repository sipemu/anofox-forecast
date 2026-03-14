# Agent-Based Forecasting: Framework Analysis & Roadmap

## Background

Analysis of two agent-based forecasting frameworks — [TimeCopilot](https://github.com/TimeCopilot/timecopilot) and [Harbor](https://github.com/harbor-framework/harbor) — compared against `anofox-forecast` to identify gaps and design an autonomous forecasting agent path.

## Harbor: Agent Evaluation Harness (Not Forecasting)

Harbor is an **agent evaluation platform**, not a forecasting framework. It runs AI coding agents (Claude Code, OpenHands, Codex, etc.) against tasks inside Docker containers and collects rewards. No forecasting models, no time series handling, no preprocessing.

**Relevant architectural patterns worth borrowing:**
- Job → Orchestrator → Trial pipeline with hook system
- Retry with exponential backoff at multiple levels
- Job resumption from disk
- Structured concurrency via `asyncio.TaskGroup`
- ATIF trajectory interchange format for audit trails

## TimeCopilot: LLM-Driven Forecasting Agent

Uses **pydantic-ai** with a three-agent architecture:

| Agent | Role |
|-------|------|
| **forecasting_agent** | LLM selects models based on tsfeatures, runs CV, picks winner |
| **decision_agent** | Routes follow-up queries (re-analyze vs. use cached results) |
| **query_agent** | Answers follow-up questions using stored DataFrames |

### How It Works

1. LLM extracts parameters from natural language (horizon, frequency, seasonality)
2. Calls `tsfeatures_tool` → 17 time series features
3. LLM reasons about features and calls `cross_validation_tool` with chosen models
4. MASE-based selection with a hard floor: selected model must beat SeasonalNaive (enforced via `ModelRetry` validator)
5. Calls `forecast_tool` with the winner
6. Optional `detect_anomalies_tool` (z-score on CV residuals)

**30+ models** across statsforecast, Prophet, foundation models (Chronos, Moirai, TimesFM, TimeGPT), neural (NHITS, TFT), and ML (LightGBM).

### Advantages

| Advantage | Detail |
|-----------|--------|
| Agentic model selection | LLM reasons about features before choosing models — not a fixed algorithm |
| Massive model breadth | Foundation models + statistical + neural + ML in one interface |
| Natural language interface | Users describe what they want; agent parses parameters |
| Quality floor | Must beat SeasonalNaive or the agent retries with different models |
| Unified quantile interface | `QuantileConverter` with isotonic regression for monotonic enforcement |
| GIFT-Eval integration | Standardized benchmarking with 11 metrics |
| Conversation memory | Follow-up queries use cached results without re-running |

### Disadvantages

| Disadvantage | Detail | How to Avoid |
|-------------|--------|--------------|
| LLM dependency for every run | Cannot forecast without API calls — adds cost, latency, non-determinism | Provide a purely algorithmic fast path alongside the agent path |
| Single evaluation metric | Agent uses only MASE for selection; misses cases where MASE is inappropriate (e.g., intermittent demand) | Use multiple metrics with data-aware metric selection |
| No preprocessing pipeline | No automated outlier treatment, differencing, or Box-Cox before fitting | Build a declarative preprocessing pipeline with audit trail |
| No exogenous variables | Multiple `# TODO` comments; CV raises `NotImplementedError` for >3 columns | Support regressors from day one |
| Sequential model execution | Models run one-at-a-time, even GPU models | Parallel model fitting with rayon |
| Enormous dependencies | PyTorch, Lightning, Ray, Wandb, transformers — fragile installs | Rust crate with zero Python dependency; optional features for heavy deps |
| No caching/warm-starting | Every `analyze()` starts from scratch | Fitted parameter persistence + warm-starting (already in anofox-forecast) |
| Shallow testing | Agent tests use stub LLM, skip actual tool-calling flow | Integration tests that exercise the full pipeline |
| Z-score-only anomaly detection | Single method for all data | anofox-forecast already has IQR, z-score, modified z-score + AID + changepoint detection |
| No multivariate support | Only `unique_id/ds/y` panel format | anofox-forecast already has VAR + Kalman + multivariate TimeSeries |

## Gaps in anofox-forecast for Agent-Based Forecasting

The crate has strong foundations (35+ models, 76+ features, comprehensive CV, postprocessing) but lacks the **orchestration layer** an agent needs.

### Tier 1 — Critical Missing Pieces

**1. Data Profiling API**
No single call that returns: stationarity (ADF/KPSS exist but are manual), seasonality period (delegated to fdars), trend strength/direction, missing value patterns, outlier statistics, intermittent classification, data quality score. An agent needs a `DataProfile` struct it can reason about.

**2. Feature → Model Mapping** *(deferred — hard problem)*
76+ features exist but nothing connects them to model suitability. AutoForecast ignores features entirely.

**3. Declarative Pipeline Builder**
No way to compose: profile → preprocess → select models → CV → ensemble → postprocess → constrain. Each step must be manually wired.

**4. Structured Decision Log / Audit Trail**
No way to record and export: "tried ARIMA(1,1,1), failed with ConvergenceFailure; tried ETS(A,A,A), MASE=0.82; tried Theta, MASE=0.91; selected ETS".

**5. Error Recovery & Fallbacks**
When a model fails, there's no automatic fallback. Batch operations return per-item Results but don't suggest alternatives. An agent needs: "ARIMA failed → try ETS → try Theta → fall back to SeasonalNaive".

**6. Per-Horizon Analysis**
No easy way to see which forecast horizons are hardest. CV results are aggregated across all horizons. An agent needs horizon-specific metrics to explain where uncertainty grows.

**7. Automatic Preprocessing Selection** *(deferred — related to feature→model mapping)*
Box-Cox, scaling, differencing, outlier treatment exist as separate tools but nothing decides which to apply based on data characteristics.

**8. Execution Metadata**
No timing information (how long did fit take?), no memory tracking, no model complexity metrics. Agents need this for cost-aware model selection.

**9. Confidence in Selection**
AutoForecast returns scores but no confidence intervals. Is the best model significantly better than #2, or within noise?

**10. Pipeline Persistence**
Cannot save a pipeline configuration and replay it on new data. Important for production deployment.

## What anofox-forecast Already Has That TimeCopilot Doesn't

| Capability | anofox-forecast | TimeCopilot |
|---|---|---|
| Exogenous variables | Across model families | Not supported |
| Multivariate (VAR, Kalman) | Yes | No |
| Warm-starting | ETS, SES, ARIMA, Theta | No |
| Forecast constraints | 6 types | No |
| Hierarchical reconciliation | 5 methods | No |
| Demand classification | Syntetos-Boylan + AID | No |
| Forecast explainability | ETS/Theta/MSTL decomposition | No |
| Postprocessing | Conformal, IDR, QRA, historical sim | No |
| Model serialization | JSON + bincode | No |
| WASM/browser support | Full npm package | No |
| Zero Python dependency | Pure Rust | Requires Python + PyTorch + many GB |

## Implementation Roadmap

Items selected for implementation (feature→model mapping deferred as hard problem):

### Phase 1: Foundational Tooling

1. **Data Profiling API** (`DataProfile` struct)
   - Stationarity (ADF/KPSS integrated)
   - Trend strength/direction
   - Missing value patterns
   - Outlier statistics
   - Intermittent classification (Syntetos-Boylan + AID)
   - Data quality score
   - Single `DataProfile::from_series(&ts)` entry point

3. **Declarative Pipeline Builder**
   - `PipelineBuilder` with chained configuration
   - Steps: profile → preprocess → select → CV → ensemble → postprocess → constrain
   - `Pipeline::execute(&ts)` returns `PipelineResult`
   - Intermediate step inspection

4. **Structured Decision Log / Audit Trail**
   - `DecisionLog` with timestamped entries
   - Decision types: ModelTried, ModelSelected, ModelFailed, FallbackUsed, PreprocessingApplied
   - `Display` and JSON export
   - Attached to pipeline results

5. **Error Recovery & Fallbacks**
   - `FallbackChain` for model selection
   - Configurable retry strategies
   - Automatic degradation: complex model → simpler model → baseline
   - Recovery suggestions in error context

6. **Per-Horizon Analysis**
   - Per-step metrics from CV results
   - Horizon difficulty profile
   - Confidence decay curves
   - `HorizonAnalysis` struct with per-h MAE/RMSE/coverage

### Phase 2: Metadata & Persistence

8. **Execution Metadata**
   - `ExecutionMetadata`: fit duration, predict duration, memory estimate
   - Attached to every model fit/predict result
   - Aggregated in pipeline results

9. **Confidence in Selection**
   - Statistical comparison of top-k model scores
   - Paired t-test or Wilcoxon on CV fold results
   - `SelectionConfidence`: significant_winner vs. within_noise
   - Margin of victory reporting

10. **Pipeline Persistence**
    - Serialize `PipelineConfig` to JSON/YAML
    - Replay on new data: `Pipeline::from_config(config).execute(&new_ts)`
    - Version tracking

## Target Architecture

```
┌──────────────────────────────────────────────────┐
│  Agent Interface (MCP / CLI / API)               │
│  Natural language → structured ForecastRequest    │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  Orchestration Layer (NEW module)                 │
│                                                   │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ DataProfiler │  │ DecisionLog  │               │
│  │ - stationarity│ │ - decisions  │               │
│  │ - seasonality │ │ - timings    │               │
│  │ - intermittent│ │ - fallbacks  │               │
│  │ - quality     │ │ - explain()  │               │
│  └──────┬──────┘  └──────────────┘               │
│         │                                         │
│  ┌──────▼──────────────────────────┐             │
│  │ PipelineBuilder                  │             │
│  │ .profile()                       │             │
│  │ .preprocess(Auto)                │             │
│  │ .select_models(top_k)            │             │
│  │ .cross_validate(strategy=Auto)   │             │
│  │ .ensemble(method=Auto)           │             │
│  │ .postprocess(conformal)          │             │
│  │ .constrain(non_negative)         │             │
│  │ .build() → Pipeline              │             │
│  └──────┬──────────────────────────┘             │
│         │                                         │
│  ┌──────▼──────┐  ┌──────────────┐               │
│  │ FallbackMgr │  │ ReportBuilder│               │
│  │ model chain  │  │ JSON/Display │               │
│  │ error→retry  │  │ audit trail  │               │
│  └─────────────┘  └──────────────┘               │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  anofox-forecast (existing)                       │
│  35+ models, 76+ features, CV, postprocessing     │
└──────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Algorithmic-first, agent-optional.** Unlike TimeCopilot which requires an LLM for every run, build `DataProfiler` + `PipelineBuilder` as pure Rust logic. An LLM agent can use this as tools, but the system works standalone.

2. **Quality floor.** Borrow TimeCopilot's "must beat SeasonalNaive" validator but implement it as a Rust check, not an LLM retry loop.

3. **Multi-metric selection.** Instead of MASE-only, select based on a weighted combination that adapts to data type:
   - Intermittent → MASE + coverage
   - Non-negative → MASE + WAPE
   - General → MASE + SMAPE + MDA

4. **Structured tools for MCP.** Expose `profile_data`, `select_models`, `run_pipeline`, `explain_result` as MCP tools for any LLM agent.

5. **Everything serializable.** Pipelines, profiles, decisions, results — all JSON-serializable for persistence and debugging.
