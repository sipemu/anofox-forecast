# Phase 5: MFLES Multiplicative-Guard Fix - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the MFLES auto-multiplicative runaway (#219) at all three failure points in
`src/models/mfles.rs` — (1) auto mode selection, (2) the `ln()` transform, and
(3) the `exp()` back-transform — and prove the fix with a committed before/after
regression on the #219 repro series (near-zero month, level ≈ 2k). Scope is MFLES
only; the broader bug-class audit across other models is Phase 6. Covers
MULT-01..04. Public `Forecaster` API stays unchanged; npm/WASM package keeps
building; clippy `-D warnings` and cargo-audit/deny stay green.

</domain>

<decisions>
## Implementation Decisions

### Mode-Selection Guard (MULT-01)
- Threshold τ = **0.10**: when `min / level < τ`, auto mode selects **additive** instead of multiplicative.
- "Level" is the **median** of the series (robust to the near-zero outlier itself).
- Guard is an **internal constant** — no new builder method, `Forecaster` API unchanged (SC5).
- Guard trips **silently** (fall back to additive; library avoids logging per conventions).
- Applies to the auto path only (`multiplicative: None`, currently `mfles.rs:998-1004`); an explicit `.multiplicative(true)` still honors the user's choice but is still protected by the floor + clamp below.

### Log-Transform Floor (MULT-02)
- Before `ln()`, winsorize each value: `v.max(floor_frac × level)` with **floor_frac = 0.01** and `level = median`.
- Prevents a single near-zero observation from opening a log-space crater (currently `mfles.rs:1012`).
- Applied **consistently** at fit-time and any internal transform that maps to log space.

### Back-Transform Clamp (MULT-03)
- Clamp the `exp()` back-transform (currently `mfles.rs:937-938`) to a bounded range:
  - **Upper cap = 10 × in-sample max** (runaway backstop; comfortably above legitimate growth, kills the ~13× blow-up).
  - **Floor at 0** for positive series.
- Store the in-sample max (or bound) as fitted state so `predict()` can clamp.

### Before/After Proof (MULT-04)
- Regression test reproduces the #219 series (near-zero month, level ≈ 2k), asserts the forecast is within a bounded multiple of level (no ~13× blow-up).
- Commit the before (~13×) / after (~level) delta as the regression guard, honoring the Core Value before/after discipline.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/models/mfles.rs` (2415 lines) — single-file model. Failure points located:
  - Auto mode selection: `mfles.rs:998-1004` (`season_length > 0 && all values > 0`).
  - `ln()` transform: `mfles.rs:1009-1012` (also sets `const_val = min`).
  - `exp()` back-transform: `mfles.rs:937-943` in the prediction loop.
- Fitted-state fields already present: `const_val`, `is_multiplicative`, `n` — add an in-sample-max/bound field alongside.
- `calc_cov` (`mfles.rs:422`) already reasons in log space for multiplicative mode — floor must be consistent with it.

### Established Patterns
- Builder pattern (`MFLESBuilder`, `mfles.rs:108`); avoid adding public knobs to keep API unchanged.
- Additive mode already computes `mean`/`std` with a `.max(1e-10)` guard — mirror that defensive style.
- Validation via `validate_series_complete()` at top of `fit()`.

### Integration Points
- MFLES is reachable via `auto_forecast.rs`, `batch.rs`, `inspect.rs` — internal-constant approach means no signature changes ripple out.
- Regression test belongs in `tests/` following existing model-validation test shape (e.g. `ets_validation.rs`).

</code_context>

<specifics>
## Specific Ideas

- Numeric constants: τ = 0.10, floor_frac = 0.01, back-transform cap K = 10× in-sample max, all as named `const` with a doc comment referencing #219.
- The regression test must assert an upper bound (e.g. `forecast_max < K_test × level` with a margin below the raw pre-fix ~13×) so it fails on the old behavior and passes on the new.

</specifics>

<deferred>
## Deferred Ideas

- Auditing other models (Theta, ETS, TBATS, etc.) for the same too-loose-guard failure class — that is Phase 6 (MULT-05/06).
- Making the guard thresholds user-configurable via the builder — deferred unless a real need arises (keeps API surface minimal for v1.1).

</deferred>
