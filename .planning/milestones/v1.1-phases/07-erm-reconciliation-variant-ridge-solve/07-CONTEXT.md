# Phase 7: ERM Reconciliation Variant & Ridge Solve - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `ReconciliationMethod::Erm { lambda }` to the hierarchy reconciliation subsystem
(`src/hierarchy/mod.rs`) as a backward-compatible additive variant, fed by a training-history
API, computing the projection via the regularized ridge solve `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹` and
returning reconciled bottom `P·ŷ` and reconciled all `S·P·ŷ`. Prove correctness against the
reference formula on a known small hierarchy. Covers ERM-01 (variant), ERM-02 (training-history
API), ERM-03 (ridge solve), ERM-05 (correctness proof). Auto-λ (ERM-04) and grouped/crossed
end-to-end validation (ERM-06) are Phase 8. Fixed-λ only here. Public API additive-only; npm/WASM
keeps building; clippy `-D warnings` + audit/deny green.

</domain>

<decisions>
## Implementation Decisions

### API & Variant Shape
- New enum arm **`Erm { lambda: f64 }`** (required fixed λ; auto-λ is Phase 8) added to `ReconciliationMethod` alongside BottomUp/TopDown/MiddleOut/MinTrace* — purely additive, existing hierarchy code compiles unchanged (ERM-01).
- **Training history via a builder setter on `HierarchyTree`** (e.g. `with_erm_training(base_forecast_history, leaf_actuals_history)` — name at planner discretion), mirroring the existing per-node historical-actuals / residuals storage pattern. `reconcile(base, ReconciliationMethod::Erm { lambda })` then dispatches through the existing `reconcile()` match (ERM-02).
- **Missing training history when `Erm` selected → return a clear `ForecastError`** with an actionable hint (mirrors the `MinTraceShrink` "requires residuals" behavior). No panic, no silent fallback.
- **Validate matrix shapes at solve time**: base-forecast history is nodes×T, leaf actuals is leaves×T, with T ≥ a documented minimum (enough columns for a well-posed `Ŷ'Ŷ`); mismatched shapes → `ForecastError`.

### Solve Backend & Correctness
- **Reuse the module's in-house `cholesky` / `cholesky_solve_vec` helpers** (`src/hierarchy/mod.rs:1389/1414`) — the ridge-regularized Gram matrix `Ŷ'Ŷ + λI` is SPD, ideal for Cholesky. **No faer, no feature gate** — ERM is always available like the tree-based methods (NOT behind `postprocess`).
- Ridge solve: `P = B'Ŷ (Ŷ'Ŷ + λI)⁻¹` — form the SPD system `(Ŷ'Ŷ + λI)`, Cholesky-factor once, solve for each row/column as needed to assemble `P` (leaves × nodes). Reconciled bottom = `P·ŷ`; reconciled all = `S·P·ŷ` (coherent full vector, same return shape as other methods).
- **Correctness proof (ERM-05):** a known small 2-level hierarchy (root + 2 leaves) with a hand-computed `P` and reconciled output; assert ERM output matches the reference formula within numerical tolerance — a real correctness check, not just "it runs". Also assert coherence (`S·bottom == all`).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ReconciliationMethod` enum (`src/hierarchy/mod.rs:61`) — add the `Erm { lambda: f64 }` arm; wire a dispatch case in `reconcile()` (`mod.rs:481`, match around `mod.rs:524-530`).
- In-house linalg: `cholesky(n, a)` (`mod.rs:1389`) and `cholesky_solve_vec(n, l, b)` (`mod.rs:1414`) — reuse for the SPD ridge solve. MinTrace methods (`min_trace_ols`/`min_trace_shrink`) show the pattern of assembling S, solving, and returning a coherent vector.
- `HierarchyTree` already stores per-node historical actuals (TopDown proportions) and residuals (MinTraceShrink) — add ERM training-history fields in the same style; `from_summing_matrix` (`mod.rs:248`) provides the S matrix for grouped hierarchies.
- Summing matrix `S`, node/leaf indexing, and coherence helpers already exist and are used by the MinTrace paths.

### Established Patterns
- `reconcile()` returns a coherent full-node vector; per-method private fns (`min_trace_*`) do the work. ERM follows the same shape: a private `erm(&base_map, horizon, lambda)` fn.
- Errors via `ForecastError` with hints; shape validation up front. Feature-gated code uses `#[cfg(feature = ...)]` — ERM needs NONE (pure in-house linalg).
- `serde` derive is applied conditionally to public types — the new enum arm must serialize/deserialize cleanly under the `serde` feature (add the arm; it derives automatically). No `serde(default)` needed on an enum arm, but confirm round-trip.

### Integration Points
- Enum is re-exported via `src/hierarchy/mod.rs` → `src/lib.rs`; a new arm is source-compatible. No signature changes to `reconcile()`.
- Tests live inline in `src/hierarchy/mod.rs` `#[cfg(test)]` (see `mint_ols_coherent` etc.) and/or a `tests/` integration file — follow the existing mint-test shape for the ERM correctness proof.

</code_context>

<specifics>
## Specific Ideas

- Reference formula to encode verbatim in the correctness test: `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹`, reconciled bottom `P·ŷ`, reconciled all `S·P·ŷ`, where `B` = leaf actuals history (leaves×T), `Ŷ` = base-forecast history (nodes×T), `ŷ` = current base forecast (nodes).
- Keep the `Erm` solve as its own private fn (`erm_reconcile` / `min_trace`-sibling) so Phase 8 auto-λ can layer on top without restructuring.
- Document the λ meaning in the variant doc-comment (ridge regularization strength; larger λ → more shrinkage toward the unregularized least-squares P).

</specifics>

<deferred>
## Deferred Ideas

- **Auto-λ (Ledoit-Wolf-style) default (ERM-04)** — Phase 8.
- **Grouped/crossed end-to-end accuracy validation vs MinTrace/unreconciled baseline (ERM-06)** — Phase 8.
- Any performance tuning of the solve beyond correctness (broader perf milestone, not this phase).

</deferred>
