# Phase 6: Multiplicative-Guard Bug-Class Audit - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Inventory and audit every model with a data-driven auto-multiplicative/log selection path
(the #10/#219 too-loose-guard failure class), fix any additional offenders with the Phase 5
guard discipline, and record already-safe models with evidence. Produces a complete, auditable
sweep. Covers MULT-05 (inventory + audit) and MULT-06 (fix + regression for any offender).
Depends on Phase 5 (reuses the tightened-guard pattern and regression-test shape). Public
`Forecaster` API unchanged; npm/WASM keeps building; clippy `-D warnings` + audit/deny green.

</domain>

<decisions>
## Implementation Decisions

### Audit Scope & Methodology
- **In scope** — data-driven auto mode/transform selection reachable via the public API:
  - AutoETS (multiplicative error/seasonal auto-selection — the #10 lineage) and the ETS family.
  - Theta / TBATS log (Box-Cox) options and any auto log/level selection.
  - Auto Box-Cox / Yeo-Johnson λ selection in `src/transform/` (transform-driven mode selection).
  - Distributional Laplace leaves with log/multiplicative behavior (`lognormal`, `seasonal_mult`, `yj_wrapper`, standardize wrappers) — **audited and recorded, flagged as `distributional`-gated**.
  - MFLES is already fixed (Phase 5) — record as fixed/pass with a pointer to #219.
- **Out of scope**: non-auto internal `.ln()`/multiplicative sites that are not a data-driven selection reachable through the public forecasting API (avoid noise). SMA/Naive/RandomWalk baselines that have no multiplicative auto-path are noted as N/A.
- **Repro series**: reuse Phase 5's shape (level ≈ 2000, one near-zero month) for at-risk models, adapted to each model's minimum data requirements.
- **Fail criterion**: a model "fails" the audit when a near-zero-relative-to-level input produces a forecast blowing up beyond a bounded multiple of level (same spirit as Phase 5's ~2.5–10× level threshold). Document the before (blow-up) / after (~level) number.
- **Fix discipline for offenders**: reuse the Phase 5 pattern — min/level threshold guard on mode selection, floored/winsorized transform, clamped back-transform — adapted per model. Constants private; no public API change.

### Inventory Artifact
- Committed **markdown inventory** at `docs/audits/multiplicative-guard-audit.md` (create `docs/audits/` if absent) — a table with one row per audited model/path.
- **Verdict granularity per model**: `pass` (safe — with a short rationale for *why* the existing guard is tight enough) or `fail` (offender — fixed, with the before→after number and the regression test name).
- **Already-safe models** (SC4): recorded in the inventory with evidence, AND backed by a passing guard-assertion test where practical (a near-zero series that must forecast ~level); rely on the existing suite only where a dedicated test adds no signal.
- Any offender fixed under MULT-06 gets a regression test proving the pre-fix blow-up no longer occurs (Phase 5 test shape).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 5 guard pattern** (`src/models/mfles.rs`): `MULT_AUTO_TAU=0.10` (min/median mode guard), `MULT_LOG_FLOOR_FRAC=0.01` (ln floor), `MULT_BACK_CLAMP_K=10.0` (back-transform clamp), `insample_max: Option<f64>` fitted field, `Self::median_scalar` helper. Copy the discipline, not necessarily the exact constants.
- **Phase 5 regression test** (`tests/issue_219_mfles_multiplicative_runaway.rs`): the near-zero repro + bounded-multiple assertion shape to clone per offender.
- Candidate auto-selection sites surfaced by scout (to be confirmed by research):
  - `src/models/exponential/auto_ets.rs`, `ets.rs`, `holt_winters.rs`, `seasonal_es.rs`, `global_ets.rs` (mult error/seasonal — #10 lineage).
  - `src/models/theta/{model,optimized,dynamic}.rs`, `src/models/tbats/{model,auto}.rs` (log/Box-Cox options).
  - `src/transform/{boxcox,yeo_johnson,transforms,pipeline}.rs` (auto-λ selection).
  - `src/models/laplace/leaves/{lognormal,seasonal_mult,yj_wrapper,standardize,slow_standardize}.rs`, `dist.rs`, `multiscale.rs` (distributional-gated).

### Established Patterns
- Models validate input at `fit()` start (`validate_series_complete`); additive/multiplicative decided in `fit()`; back-transform in `predict`/`predict_internal`.
- No production logging (library); private const for guard params; builder pattern but avoid new public knobs.

### Integration Points
- Audit tests live under `tests/` following the existing model-validation shape; the inventory doc under `docs/audits/`.
- Any fix ripples only internally (private consts / fitted fields) — no signature changes.

</code_context>

<specifics>
## Specific Ideas

- The inventory must be *complete and auditable*: every in-scope model/path appears with a verdict (pass/fail/N-A), not just the offenders — SC4 requires already-safe models be recorded with evidence.
- Where a model reuses a shared transform (Box-Cox), audit the transform once and reference it from dependent models to avoid duplicate verdicts.
- Cross-reference #10 and #219 in the inventory as the lineage motivating the sweep.

</specifics>

<deferred>
## Deferred Ideas

- Making guard thresholds user-configurable (carried from Phase 5 — still out of scope).
- Auditing non-auto internal `.ln()` sites unreachable from the public forecasting API.
- Any performance re-tuning of the guards beyond correctness (that is the broader perf milestone, not this audit).

</deferred>
