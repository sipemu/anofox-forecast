# Phase 8: Auto-λ & Grouped/Crossed Validation - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete the ERM reconciliation feature: add a Ledoit-Wolf-style auto-λ default (matching
`MinTraceShrink` ergonomics) selectable alongside the fixed-λ path, and prove ERM end-to-end on a
grouped/crossed hierarchy with a committed before/after accuracy delta vs unreconciled and MinTrace
baselines. Covers ERM-04 (auto-λ) and ERM-06 (grouped/crossed validation). Builds on Phase 7
(`Erm` variant + ridge solve). Public API stays backward-compatible within v1.1; npm/WASM keeps
building; clippy `-D warnings` + audit/deny green. This is the final v1.1 phase.

</domain>

<decisions>
## Implementation Decisions

### Auto-λ API Shape
- **Change the Phase 7 variant `Erm { lambda: f64 }` → `Erm { lambda: Option<f64> }`**: `None` = Ledoit-Wolf-style auto-λ (the default, matching `MinTraceShrink` ergonomics); `Some(x)` = caller-supplied fixed λ. Single variant, both selectable via the public API (ERM-04). v1.1 is unreleased, so this signature change is acceptable; **Phase 7's ERM tests must be updated to `Some(λ)`** where they used a bare `f64`.
- **Auto-λ estimator: Ledoit-Wolf-style shrinkage intensity** adapted to the ERM ridge/Gram context. Reuse/adapt the existing LW machinery in `src/hierarchy/mod.rs` (`min_trace_shrink` ~1140, the LW-intensity helper ~1662) — the research pins the exact formula for the ERM `(ŶŶᵀ + λI)` setting. Must behave sensibly on real crossed data (SC4).
- **Auto-λ is the default** when `None` is passed (no explicit opt-in needed) — mirrors `MinTraceShrink` picking a sensible shrink automatically.
- **Fixed-λ fully supported** via `Some(λ)`; the Phase 7 fixed-λ correctness proofs are retained (updated to `Some(λ)`), and the Phase 7 λ-guards (finite, non-negative) still apply to the `Some` path.

### End-to-End Grouped/Crossed Validation (ERM-06)
- **Grouped/crossed hierarchy** with ≥2 crossing dimensions (e.g. region × product), synthetic but realistic; built via `from_summing_matrix` (grouped hierarchies are already supported there). No customer/client identity in the data (public crate — describe structure only).
- **Two baselines**: unreconciled (base forecasts) AND MinTrace (the incumbent optimal-combination method) — shows ERM vs no-reconciliation and vs MinTrace.
- **Accuracy metric: RMSSE** (or MASE) — scale-free, standard for hierarchical forecasting — measured across ALL aggregation levels, aggregated to a headline number per method.
- **Auto-λ (None) is the ERM configuration exercised in the end-to-end validation** (SC4 — demonstrate the shrink default on real crossed data), not only the fixed-λ path.
- **Committed before/after**: a test asserting the ERM-vs-baseline deltas (ERM error ≤ baseline, or within a documented tolerance) PLUS a short committed results note (metric table) — honoring the Core Value before/after discipline. Results note lives under `docs/audits/` or a phase artifact (planner's discretion).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 7 ERM: `ReconciliationMethod::Erm` variant, `with_erm_training`, `erm_reconcile` (`src/hierarchy/mod.rs` ~791), reusing in-house `cholesky`/`cholesky_solve_vec`. Phase 7 ERM tests (`erm_correctness_reference_formula`, `erm_correctness_asymmetric`, error-path guards) — update these to `Some(λ)`.
- **Ledoit-Wolf machinery already present**: `min_trace_shrink` (~1140) and the LW optimal-shrinkage-intensity helper (~1662) — the reference/adaptable code for ERM auto-λ.
- `from_summing_matrix` (~248) — build grouped/crossed hierarchies from an explicit S matrix (the ERM training + validation path already operates on S).
- MinTrace methods (`min_trace_ols`, `min_trace_shrink`, diagonal variants) — the baselines for the accuracy comparison; `reconcile()` provides all of them.
- Existing accuracy/metric utilities: check `src/utils/` and `src/validation/` for RMSSE/MASE helpers before writing a new one.

### Established Patterns
- `reconcile()` enum dispatch; per-method private fns returning a coherent full-node vector.
- Tests inline `#[cfg(test)]` in `src/hierarchy/mod.rs` and/or `tests/` integration files; before/after numbers committed as asserting tests (Phase 5/7 pattern).
- No production logging; private consts; `ForecastError` with hints.

### Integration Points
- The `Option<f64>` signature change ripples to: the enum arm, `erm_reconcile` (branch on None → compute auto-λ), Phase 7 tests, and any exhaustive match / JS binding `parse_method()` (catch-all, compiles — but consider adding `"erm"` string parsing now that ERM is feature-complete).
- Serde: the arm becomes `Erm { lambda: Option<f64> }` — confirm round-trip.

</code_context>

<specifics>
## Specific Ideas

- The auto-λ path (`None`) must reduce to a well-conditioned, sensible ridge on crossed data — the end-to-end test is the evidence it "behaves sensibly" (SC4), not just that it runs.
- Keep the auto-λ computation in its own helper so the fixed-λ solve path is unchanged when `Some(λ)` is supplied.
- The results note should present a small table: method (unreconciled / MinTrace / ERM auto-λ) × RMSSE (headline + per-level), making the before/after delta explicit and auditable.
- Consider adding an `"erm"` arm to the JS `parse_method()` now that ERM is complete (Phase 7 deferred it) — optional polish, keep non-blocking.

</specifics>

<deferred>
## Deferred Ideas

- Further ERM performance tuning beyond correctness + the accuracy proof (broader perf milestone).
- Additional reconciliation methods or shrinkage targets beyond Ledoit-Wolf.
- Exposing auto-λ tuning knobs to callers (keep the default parameter-free like MinTraceShrink).
