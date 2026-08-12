---
phase: 04-prioritized-improvement-backlog-top-value-fixes
plan: "01"
subsystem: testing
tags: [nan-guard, coverage, global-models, robustness, rust]

requires:
  - phase: 03-numerical-robustness-coverage-baseline
    provides: GAP-INVENTORY with V-01/V-02/V-03/G-01 findings; coverage.json baseline (91.30%, floor 90.3%); VAR::fit guard pattern

provides:
  - "V-01: NaN/Inf guard in GlobalETS::fit returning ForecastError::InvalidParameter"
  - "V-02: NaN/Inf guard in GlobalCroston::fit returning ForecastError::InvalidParameter"
  - "V-03: NaN/Inf guard in GlobalTheta::fit returning ForecastError::InvalidParameter"
  - "G-01: GlobalTheta smoke test (0% → covered); tests/global_theta_smoke.rs with 5 tests"
  - "Guard-path tests for GlobalETS and GlobalCroston in tests/global_model_nan_guards.rs"
  - "coverage.json ratcheted: 91.30% → 91.44%, floor 90.3% → 90.4%"

affects:
  - 04-02-accuracy-investigation
  - 04-03-backlog

actuals:
  tokens: 5200
  tasks: 4
  commits: 4

tech-stack:
  added: []
  patterns:
    - "per-series !v.is_finite() guard pattern for raw &[Vec<f64>] fit() APIs (matches VAR::fit at var.rs:119-124)"

key-files:
  created:
    - tests/global_theta_smoke.rs
    - tests/global_model_nan_guards.rs
  modified:
    - src/models/theta/global_theta.rs
    - src/models/exponential/global_ets.rs
    - src/models/intermittent/global_croston.rs
    - .planning/baselines/coverage.json

key-decisions:
  - "Used !v.is_finite() predicate (clippy-preferred) over v.is_nan() || v.is_infinite() — same semantics, cleaner form"
  - "Guard inserted after existing is_empty/length guards and before estimation step in each model"
  - "coverage.json baseline update committed as a SEPARATE deliberate commit (84f9863) per IMPR-03 — not bundled with code/test commits"
  - "V-04 (VAR error-variant divergence) left as document-only — refactor deferred to 04-03 backlog"

patterns-established:
  - "NaN/Inf guard for raw-vec fit() APIs: iterate with index, check !v.is_finite(), return ForecastError::InvalidParameter(format!(\"series {} contains NaN or Inf values\", i))"
  - "Smoke tests for 0%-coverage models: constructor, with_param(), accessor, fit/predict happy path, NaN guard, Inf guard, empty-input guard"

requirements-completed: [IMPR-02, IMPR-03]

coverage:
  - id: D1
    description: "V-01 guard: GlobalETS::fit returns ForecastError::InvalidParameter on NaN/Inf series element"
    requirement: IMPR-02
    verification:
      - kind: integration
        ref: "tests/global_model_nan_guards.rs#global_ets_nan_guard"
        status: pass
      - kind: integration
        ref: "tests/global_model_nan_guards.rs#global_ets_inf_guard"
        status: pass
      - kind: integration
        ref: "tests/global_model_nan_guards.rs#global_ets_valid_panel_guard_does_not_fire"
        status: pass
    human_judgment: false
  - id: D2
    description: "V-02 guard: GlobalCroston::fit returns ForecastError::InvalidParameter on NaN/Inf series element"
    requirement: IMPR-02
    verification:
      - kind: integration
        ref: "tests/global_model_nan_guards.rs#global_croston_nan_guard"
        status: pass
      - kind: integration
        ref: "tests/global_model_nan_guards.rs#global_croston_inf_guard"
        status: pass
      - kind: integration
        ref: "tests/global_model_nan_guards.rs#global_croston_valid_panel_guard_does_not_fire"
        status: pass
    human_judgment: false
  - id: D3
    description: "V-03 guard: GlobalTheta::fit returns ForecastError::InvalidParameter on NaN/Inf series element"
    requirement: IMPR-02
    verification:
      - kind: integration
        ref: "tests/global_theta_smoke.rs#global_theta_nan_guard"
        status: pass
      - kind: integration
        ref: "tests/global_theta_smoke.rs#global_theta_inf_guard"
        status: pass
    human_judgment: false
  - id: D4
    description: "G-01 closed: GlobalTheta smoke test covers new(), with_theta(), alpha(), fit(), predict(), and guard paths (was 0% coverage)"
    requirement: IMPR-02
    verification:
      - kind: integration
        ref: "tests/global_theta_smoke.rs#global_theta_basic_fit_predict"
        status: pass
      - kind: integration
        ref: "tests/global_theta_smoke.rs#global_theta_empty_input_guard"
        status: pass
      - kind: integration
        ref: "tests/global_theta_smoke.rs#global_theta_with_theta_constructor"
        status: pass
    human_judgment: false
  - id: D5
    description: "coverage.json ratcheted: 91.30% → 91.44% (+0.14 pp), floor 90.3% → 90.4%, committed as a separate deliberate commit"
    requirement: IMPR-03
    verification:
      - kind: other
        ref: "jq '.coverage | {lines_percent, ratchet_floor_percent}' .planning/baselines/coverage.json"
        status: pass
    human_judgment: false

duration: 75min
completed: 2026-08-11
status: complete
---

# Phase 4 Plan 01: P1 NaN/Inf Guards + GlobalTheta Smoke Test Summary

**Three raw-vec NaN/Inf guards (V-01/V-02/V-03) land in GlobalETS, GlobalCroston, and GlobalTheta fit() paths; GlobalTheta smoke test closes the G-01 0%-coverage void; coverage ratcheted from 91.30% to 91.44% (floor 90.3% to 90.4%)**

## Performance

- **Duration:** ~75 min
- **Started:** 2026-08-11T19:30:05Z
- **Completed:** 2026-08-11T20:45:30Z
- **Tasks:** 4 (Task 3 was a verification gate — no files modified)
- **Files modified:** 6

## Accomplishments

- Inserted per-series `!v.is_finite()` NaN/Inf guard in `GlobalETS::fit` (V-01), `GlobalCroston::fit` (V-02), and `GlobalTheta::fit` (V-03), each returning `ForecastError::InvalidParameter` naming the offending series index — matching the canonical `VAR::fit` pattern at `var.rs:119-124`.
- Created `tests/global_theta_smoke.rs` with 5 integration tests (happy path, NaN guard, Inf guard, empty-input guard, `with_theta()` constructor), closing the G-01 void: GlobalTheta went from 0% to fully exercised.
- Created `tests/global_model_nan_guards.rs` with 6 integration tests covering NaN guard, Inf guard, and valid-panel happy paths for both GlobalETS and GlobalCroston.
- Full suite (`cargo test --package anofox-forecast --all-features --tests`) green — all test result lines show `ok`, zero regressions from the guards.
- Captured before/after coverage delta and ratcheted the CI floor in a separate deliberate commit: Before 91.30%/90.3%, After 91.44%/90.4% (+0.14 pp coverage, +0.1 pp floor).

## Coverage Delta (IMPR-03)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| lines_total | 77,743 | 77,767 | +24 (new test lines counted) |
| lines_covered | 70,982 | 71,112 | +130 |
| lines_percent | 91.30% | 91.44% | +0.14 pp |
| ratchet_floor_percent | 90.3% | 90.4% | +0.1 pp |

Measurement: `bash scripts/update_coverage.sh` (scope: `--package anofox-forecast --all-features`, tool: cargo-llvm-cov 0.8.4).

## Task Commits

1. **Task 1 (Tracer): GlobalTheta V-03 guard + G-01 smoke test** - `e2d2e73` (feat)
2. **Task 2: GlobalETS V-01 + GlobalCroston V-02 guards + guard-path tests** - `3fc82aa` (feat)
3. **Task 3: Full-suite green + clippy verification** - no commit (verification gate)
4. **Task 4: Coverage delta capture + floor ratchet** - `84f9863` (chore, separate deliberate commit)

## Files Created/Modified

- `src/models/theta/global_theta.rs` — V-03 guard inserted after is_empty() check, before nelder_mead()
- `src/models/exponential/global_ets.rs` — V-01 guard inserted after length-check, before states initialization
- `src/models/intermittent/global_croston.rs` — V-02 guard inserted after is_empty() check, before extract_demands
- `tests/global_theta_smoke.rs` — 5-test smoke file covering full GlobalTheta API + guard paths (G-01)
- `tests/global_model_nan_guards.rs` — 6-test guard-path file for GlobalETS (V-01) and GlobalCroston (V-02)
- `.planning/baselines/coverage.json` — ratcheted floor, separate deliberate commit

## Decisions Made

- Used `!v.is_finite()` predicate (clippy-preferred, equivalent to `is_nan() || is_infinite()`) — matches the VAR::fit intent, shorter form.
- Guard placement: inserted between the existing empty/length check and the estimation step, so the only new code is the scan loop.
- V-04 (VAR `MissingValues` vs `InvalidParameter` divergence) left as document-only per plan — no refactor this phase; deferred to 04-03 backlog.
- Coverage baseline committed with `--no-verify` (cargo-fmt pre-commit hook blocks planning-doc commits) per project memory.

## Deviations from Plan

### Non-Blocking Pre-existing Issues Observed (Out of Scope)

**1. [Scope Boundary — Out of Scope] Pre-existing clippy warnings (59 errors in all-targets run)**
- **Observed during:** Task 3 verification
- **Issue:** `cargo clippy --all-targets --all-features -- -D warnings` reports 59 errors on the baseline commit (before any of our changes). Count is identical after our changes (59 before, 59 after).
- **Action:** None. All errors are in files not touched by this plan. Logged here for visibility; deferred to `deferred-items.md`.

**2. [Scope Boundary — Out of Scope] `examples/skaters_m5_full_auto.rs` compile error**
- **Observed during:** Task 3 verification
- **Issue:** Pre-existing non-exhaustive match error (`SelectedFamily::AutoETSStructural` and `AutoThetaShortHistory` not covered). Exists before and after our changes.
- **Action:** None. Not introduced by this plan.

### Coverage Run Environment Note

The first two attempts at `bash scripts/update_coverage.sh` failed with "could not execute process" errors for test binaries in the llvm-cov-target directory. This was a stale cache issue from a prior partial run. Clearing `target/llvm-cov-target/` and re-running produced a clean measurement on the third attempt.

---

**Total deviations:** None from plan — all three guards inserted exactly as specified, smoke test and guard-path tests match the plan's behavior descriptions, coverage committed separately. Pre-existing issues observed are out of scope and not introduced by this plan.

## Issues Encountered

- `cargo-llvm-cov` failed twice due to stale `target/llvm-cov-target/` cache from a prior partial run. Resolution: `rm -rf target/llvm-cov-target` then re-run script.

## Known Stubs

None — all deliverables are fully implemented and verified.

## Threat Flags

None — the three guards close the T-04-01 threat (silent NaN propagation into fitted parameters) by adding explicit input validation. No new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

Verified:
- `src/models/theta/global_theta.rs` — guard present (FOUND)
- `src/models/exponential/global_ets.rs` — guard present (FOUND)
- `src/models/intermittent/global_croston.rs` — guard present (FOUND)
- `tests/global_theta_smoke.rs` — file exists (FOUND)
- `tests/global_model_nan_guards.rs` — file exists (FOUND)
- `.planning/baselines/coverage.json` — ratchet_floor_percent=90.4 > 90.3 (VERIFIED)
- Commits `e2d2e73`, `3fc82aa`, `84f9863` — exist in git log (VERIFIED)
- Test suite: all 20 test result lines show `ok`, zero `FAILED` (VERIFIED)
- Coverage baseline commit `84f9863` is separate from code commits (VERIFIED via `git log --oneline -4`)

## Next Phase Readiness

- 04-02 (time-boxed accuracy investigation) can proceed independently — the guards and smoke test are the guaranteed IMPR-02/03 deliverables.
- 04-03 (consolidated backlog) should document V-04 (VAR error-variant divergence) as a known inconsistency.
- CI floor is now 90.4% — any future coverage drop below this level will be caught by `ci.yml --fail-under-lines`.

---
*Phase: 04-prioritized-improvement-backlog-top-value-fixes*
*Completed: 2026-08-11*
