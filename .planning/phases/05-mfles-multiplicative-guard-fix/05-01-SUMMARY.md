---
phase: 05-mfles-multiplicative-guard-fix
plan: 01
subsystem: forecasting
tags: [mfles, log-transform, multiplicative, bug-fix, issue-219]

requires:
  - phase: 04
    provides: v1.0 baselines/regression conventions and test shapes
provides:
  - MFLES auto-multiplicative runaway (#219) fixed at all three failure points (mode selection, ln() floor, exp() back-transform clamp)
  - Committed before/after regression guard for #219 (~13× → ~level)
  - insample_max fitted-state field for predict-time clamping
affects: [phase-06-multiplicative-guard-bug-class-audit]

actuals:
  tokens: 27000
  tasks: 4
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Tightened multiplicative-guard pattern: min/median mode threshold + log-space winsorization floor + bounded back-transform clamp, all private const"

key-files:
  created:
    - tests/issue_219_mfles_multiplicative_runaway.rs
  modified:
    - src/models/mfles.rs

key-decisions:
  - "Mode guard τ=0.10 on min/median (median = robust level), internal const, silent additive fallback"
  - "Log floor = 0.01×median winsorization applied before ln()"
  - "Back-transform clamp = [0, 10×in-sample max], applied ONLY in predict_internal() to preserve the fit() decomposable invariant (issue_106 conformance)"
  - "All guard parameters private const — public Forecaster API unchanged"

patterns-established:
  - "Guard-composition: mode guard routes near-zero series to additive (floor+clamp inert there); when multiplicative is forced, floor+clamp protect independently"

requirements-completed: [MULT-01, MULT-02, MULT-03, MULT-04]

coverage:
  - id: D1
    description: "MFLES auto mode selects additive when min/median < τ=0.10 (near-zero relative to level)"
    requirement: "MULT-01"
    verification:
      - kind: integration
        ref: "tests/issue_219_mfles_multiplicative_runaway.rs#issue_219_mfles_auto_mode_selects_additive_for_near_zero_series"
        status: pass
    human_judgment: false
  - id: D2
    description: "Multiplicative ln() transform winsorizes input to 0.01×median (no log-space crater)"
    requirement: "MULT-02"
    verification:
      - kind: integration
        ref: "tests/issue_219_mfles_multiplicative_runaway.rs#issue_219_mfles_explicit_multiplicative_with_floor_and_clamp"
        status: pass
    human_judgment: false
  - id: D3
    description: "predict() clamps multiplicative forecast to [0, 10×in-sample max]; fit() decomposition invariant preserved"
    requirement: "MULT-03"
    verification:
      - kind: integration
        ref: "tests/issue_219_mfles_multiplicative_runaway.rs#issue_219_mfles_explicit_multiplicative_with_floor_and_clamp"
        status: pass
      - kind: integration
        ref: "tests/issue_106_decomposable_conformance.rs#mfles_conforms_to_decomposable_contract"
        status: pass
    human_judgment: false
  - id: D4
    description: "#219 repro series forecasts at ~level, not ~13× (committed before/after regression guard)"
    requirement: "MULT-04"
    verification:
      - kind: integration
        ref: "tests/issue_219_mfles_multiplicative_runaway.rs#issue_219_mfles_no_multiplicative_runaway"
        status: pass
    human_judgment: false

duration: ~15min
completed: 2026-09-09
status: complete
---

# Phase 5 / Plan 01: MFLES Multiplicative-Guard Fix Summary

**The MFLES auto-multiplicative runaway (#219) is fixed at all three failure points and locked behind a committed before/after regression — a near-zero-month series that used to forecast ~13× level now forecasts at ~level.**

## Performance

- **Duration:** ~15 min (executor) + orchestrator verification/finalization
- **Completed:** 2026-09-09
- **Tasks:** 4 (TDD: RED tests → fix; guard implementation covers Tasks 1–3; gate is Task 4)
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- Mode-selection guard (MULT-01): auto mode now selects additive when `min/median < MULT_AUTO_TAU (0.10)`, using the robust median as the level. `src/models/mfles.rs`.
- Log-transform floor (MULT-02): multiplicative `ln()` input is winsorized to `v.max(MULT_LOG_FLOOR_FRAC × median)` with `MULT_LOG_FLOOR_FRAC = 0.01`, closing the log-space crater a single near-zero observation opened.
- Back-transform clamp (MULT-03): multiplicative forecasts are clamped to `[0, MULT_BACK_CLAMP_K × insample_max]` (`K = 10`) **only in `predict_internal()`** — a new `insample_max: Option<f64>` fitted-state field is stored at fit time (and before the constant-series early return). The `fit()` inverse transform is deliberately left unclamped so the `trend + seasonal + residual == training` decomposable invariant is preserved.
- Before/after regression guard (MULT-04): `tests/issue_219_mfles_multiplicative_runaway.rs` reproduces the #219 series (24-pt monthly, level ≈ 2000, one near-zero month) and asserts the forecast stays below 2.5× level — a threshold the pre-fix ~13× blow-up easily violated.

## Task Commits

TDD flow (test → fix), 2 commits:

1. **RED regression tests** — `2d6eb9b` (test)
2. **Three-guard fix** — `d36de76` (feat) — implements mode guard, log floor, back-transform clamp; trims the tests to their final asserting form.

**Follow-on cleanup (separate, pre-existing debt):** `da0d92d` (style) — gated distributional-only examples/tests behind `required-features` / `#![cfg(feature)]` so bare `cargo test` compiles (see Deviations).

## Files Created/Modified
- `tests/issue_219_mfles_multiplicative_runaway.rs` (created) — 3 regression tests covering MULT-01..04.
- `src/models/mfles.rs` (modified) — three private consts, `insample_max` field, guard logic at mode-selection / ln() / predict-time back-transform.

## Decisions Made
Followed the plan and CONTEXT-locked parameters exactly (τ=0.10, floor 0.01×median, clamp 10×max + floor 0, internal consts, silent fallback, API unchanged). Reused the existing `Self::median_scalar` helper (NaN-safe). Clamp confined to `predict_internal()` per RESEARCH pitfall #1.

## Deviations from Plan

### Accepted deviation — pre-existing example/test gating debt (out of original scope, user-approved)

- **Found during:** Task 4 (full-suite gate).
- **Issue:** Bare `cargo test` failed to **compile** eight unrelated targets — examples importing `distributional`-gated modules without `required-features` (`leaf_init_pathology_sweep`, `issue_195_amplitude_decline`, `issue_195_intermittent`, `issue_198_seasonal_underuse`, `monthly_48_seasonal_diagnostic`, `synthetic_bakeoff`, `wql_outlier_diagnosis`, `m5_wape`), plus `tests/laplace_component_robustness.rs` missing the top-level `#![cfg(feature = "distributional")]` guard its siblings use. This debt predates v1.1; CI runs `cargo test --all-features` (ci.yml), so it never surfaced there.
- **Fix:** Added `[[example]] … required-features = ["distributional"]` declarations and the missing test cfg-guard, matching the existing project pattern. Metadata/gating only — no behavior change.
- **Committed in:** `da0d92d` (separate `style:` commit, kept out of the MFLES feat commit).
- **Verification:** Bare `cargo test` now green — 36 test binaries, 3299 tests pass.

**Impact on plan:** Necessary to make the phase's own `cargo test` verify command runnable; scoped to build-metadata, no source-behavior change, committed separately from the MFLES fix.

## Issues Encountered

- **Executor stalled** during Task 4's full-suite compile (600s watchdog) — the orchestrator finished the tail (verification, example-gating cleanup, this SUMMARY) via the documented spot-check/close-out path. The RED-test and fix commits were already on disk and verified.
- **`cargo test --all-features`** (the CI gate) OOM'd the linker in this sandbox (`ld … signal 7 [Bus error]`) — an environment resource limit, not a code defect. Verified green instead via three independent runs: `cargo clippy --all-targets --all-features -- -D warnings` (clean, full compile), `cargo test` default-features (3299 pass), and the targeted #219 + `issue_106` decomposable-conformance tests.

## Verification

- MULT-01..04: all covered by `tests/issue_219_mfles_multiplicative_runaway.rs` (3 tests) — pass.
- Decomposable invariant: `issue_106_decomposable_conformance` (12 tests incl. `mfles_conforms_to_decomposable_contract`) — pass.
- clippy `-D warnings` (`--all-features`): clean.
- Public `Forecaster` API: unchanged (guard params are private const; no new builder methods).
