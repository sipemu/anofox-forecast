---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 04
current_phase_name: prioritized-improvement-backlog-top-value-fixes
status: executing
stopped_at: "Completed 04-02-PLAN.md (defer branch: accuracy.json deferred, period=12 fix retained, +0.0290 gap handed to 04-03)"
last_updated: "2026-08-11T21:21:31.587Z"
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 13
  completed_plans: 12
---

# Project State: anofox-forecast — Performance & Validation Hardening

**Last Updated:** 2026-08-10
**Session:** Phase 1 verified & complete

---

## Project Reference

**Core Value:** Every claimed capability is measured, and every improvement is proven with a before/after number.

**Current Focus:** Phase 04 — prioritized-improvement-backlog-top-value-fixes

---

## Current Position

**Phase:** 04 (prioritized-improvement-backlog-top-value-fixes) — EXECUTING
**Plan:** 3 of 3
**Status:** Ready to execute

```
Progress: [█████████░] 92%

Phase 1 [COMPLETE]     █████
Phase 2 [NOT STARTED]  ░░░░░
Phase 3 [NOT STARTED]  ░░░░░
Phase 4 [NOT STARTED]  ░░░░░
```

---

## Performance Metrics

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| Requirements mapped | 28/28 | 28/28 | — |
| Phases complete | 0/4 | 0/4 | — |
| Plans complete | — | — | — |

---
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 9 | 4 tasks | 7 files |
| Phase 01 P02 | 10 | 3 tasks | 5 files |
| Phase 01 P03 | 30 | 3 tasks | 7 files |
| Phase 02 P01 | 330 | 3 tasks | 4 files |
| Phase 02 P02 | 518 | 3 tasks | 3 files |
| Phase 02-accuracy-harness-statistical-methodology P03 | 678 | 3 tasks | 5 files |
| Phase 02 P04 | 12 | 2 tasks | 2 files |
| Phase 03 P01 | 7 | 3 tasks | 4 files |
| Phase 03 P02 | 43 | 3 tasks | 2 files |
| Phase 03 P03 | ~15min | 4 tasks | 5 files |
| Phase 04 P01 | 75 | 4 tasks | 6 files |
| Phase 04 P02 | 15 | 3 tasks | 1 files |

## Accumulated Context

### Key Decisions

| Decision | Rationale | Phase |
|----------|-----------|-------|
| PERF-06 (dead-code cleanup) assigned to Phase 1 before PERF-05 (WASM size baseline) | Baseline would overstate current size if captured before cleanup; sequencing constraint from research | Phase 1 |
| accuracy.yml is workflow_dispatch-only from the start | Must never gate PR merges per explicit requirement MEAS-03 | Phase 2 |
| BENCH-01/BENCH-02 assigned to Phase 2, not Phase 4 | Cross-library comparison depends on a correct accuracy harness (ACCUR-08 validates against reference first); accuracy and benchmarking form one coherent delivery boundary | Phase 2 |
| ROBUST/COVER assigned to Phase 3, after accuracy harness | Gap inventory (COVER-02) benefits from knowing which code paths the harness exercises; no hard dependency but logical ordering improves coverage relevance | Phase 3 |
| IMPR-* assigned to Phase 4 (last) | Cannot rank backlog without real baseline numbers from all three preceding phases; research and requirements both mandate measurement-first | Phase 4 |

### Critical Constraints to Honor

- All measurement code lives in `benches/`, `tests/`, `scripts/`, harness crate — nothing new enters `src/`
- Criterion baselines captured on a quiet local machine, NOT on GitHub Actions (wall-clock noise)
- `accuracy.yml` must be `workflow_dispatch`-only — never gates PR merges
- wee_alloc must NOT be used (archived Aug 2025, known memory leaks)
- WASM target forbids `parallel` (Rayon) feature — native and WASM profiles must be measured separately
- No customer/client names in code, comments, or test names

### Todos

- [ ] Plan Phase 2 via `/gsd-plan-phase 2`

### Blockers

- ⚠️ [Phase 1] `iai.json` / `criterion.json` baselines committed as structural placeholders — real numbers require maintainer capture on a valgrind ≥ 3.20 machine (`scripts/update_iai.sh`) and a quiet local machine (`scripts/update_criterion.sh`). Harness/gates complete; numeric population is a documented manual step.

---
- accuracy.json lock deferred (ACCUR-07 partial): ACCUR-08 anchor failed (anofox MASE=1.0452 vs ref=0.8633, gap=+21%, outside ±0.02). Must re-run emit_accuracy_baseline_if_write_flag_set with ANOFOX_WRITE_ACCURACY_BASELINE=1 after Phase 4 improvements close the MASE gap.

## Session Continuity

**Last session:** 2026-08-11T21:21:31.577Z
**Stopped at:** Completed 04-02-PLAN.md (defer branch: accuracy.json deferred, period=12 fix retained, +0.0290 gap handed to 04-03)
**Resume file:** None

### What Was Done This Session

- Resumed Phase 1 UAT; marked tests 2 (iai.json) and 3 (criterion.json) passed — 3/3 UAT passed, 0 issues
- Canonicalized 01-VERIFICATION.md status human_needed → passed
- Ran phase.complete: ROADMAP + STATE advanced to Phase 2; evolved PROJECT.md (2 requirements → Validated, 3 decisions logged)

### Resume Point

Start `/gsd-plan-phase 2` — Phase 2: Accuracy Harness & Statistical Methodology covers ACCUR-01..08 and BENCH-01..02.

---

*State initialized: 2026-08-09*

## Decisions

- [Phase ?]: No [[bench]] in harness Cargo.toml yet — avoids compile error until source files exist in Plans 02-03
- [Phase ?]: wasm-size.yml uses quoted 'on' key to avoid YAML boolean coercion; zero baseline write steps (MEAS-01 CI-read-only)
- [Phase ?]: PERF-06 guard in update_wasm_size.sh enforces PERF-06-before-PERF-05 sequencing at runtime
- [Phase ?]: LibraryBenchmarkConfig.tool() not .callgrind() — verified against iai-callgrind 0.16.1 source
- [Phase ?]: iai.json placeholder values (instruction_count=0) — valgrind absent on dev machine; regenerate via update_iai.sh
- [Phase ?]: Doc comments (///) rejected by #[library_benchmark] proc-macro — use // inline comments on bench fns
- [Phase ?]: LaplaceForecaster gated behind cfg(distributional) in baseline_suite.rs — criterion_group! duplicated for cfg/no-cfg build variants
- [Phase ?]: All 7 dhat families in one #[test] fn to avoid overlapping dhat::Profiler instances (dhat panics on concurrent profilers)
- [Phase ?]: criterion.json committed as 0.0 placeholder (local-only capture, D-03); dhat.json committed with real values (native test, no env deps)
- [Phase ?]: Period-1 naive fallback on seasonal MASE collapse (D-03/D-04): fix in src/utils/metrics.rs keeps series in aggregate instead of dropping with None/NaN — matches statsforecast behavior
- [Phase ?]: mase_scale() training-denominator placed in loader.rs (harness) to keep competition-correct MASE separate from library calculate_mase which scales on test slice (Pitfall 1)
- [Phase ?]: MSIS scoped to monthly only to bound runtime; monthly is ACCUR-08 anchor frequency with richest interval evaluation
- [Phase ?]: msis() uses period-1 first-diff scaling (A4/Pitfall 4) — not seasonal-lag as M4 competition; documented in code; ACCUR-08 anchor = MASE only
- [Phase ?]: DM gate exercised via synthetic vectors (fixture has only aggregate MASE, not per-step forecasts) — documented as fixture scope limitation; Plan 04 narrative must state DM gate data scope honestly
- [Phase ?]: statsforecast 2.0.3 monthly MASE=0.8633 (not 0.93 from 1.x): provenance makes the version difference auditable; ACCUR-08 comparison is within the pinned env
- [Phase ?]: CARGO_MANIFEST_DIR used for cross_library.rs fixture path: cargo integration tests for workspace member crates CWD is the crate dir, not workspace root
- [Phase ?]: defer-lock: ACCUR-08 anchor failed (anofox MASE=1.0452 vs statsforecast 2.0.3 ref=0.8633, gap=+21%, outside ±0.02 tolerance); accuracy.json not committed until Phase 4 improvements close the gap
- [Phase ?]: accuracy.yml (MEAS-03) delivered workflow_dispatch-only with contents:read permissions, read-only harness runner, never gates PR merges
- [Phase ?]: emit_accuracy_json() helper committed with dual-key guard (write flag + anchor assertion); Phase 4 can lock baseline by running with ANOFOX_WRITE_ACCURACY_BASELINE=1 after anchor passes
- [Phase ?]: ROBUST-01: One representative model per family (Naive, AutoETS, ARIMA(1,0,1), Theta, TBATS([12]), Croston, MSTLForecaster([12]), GARCH(1,1), VARForecaster(1), LaplaceForecaster) driven through constant/n=2/zeros/intermittent/NaN/Inf/empty/extreme edge-case inputs; 61 tests pass, zero panics
- [Phase ?]: ROBUST-02: validate_series_complete(series)? added at fit() entry in gpd_tails.rs and multiscale.rs (delegation wrapper boundary guard); 4 raw-vec global models deferred to P1 gap inventory
- [Phase ?]: Three separate proptest blocks used (50/30/100 cases each subsystem) to permit different runtime bounds per block
- [Phase ?]: MSTLResult.seasonal_components field (not seasonal) accessed for property assertions — confirmed in mstl.rs:17
- [Phase ?]: update_coverage.sh scoped to --package anofox-forecast --all-features (not workspace-wide) to avoid bench-harness distorting line %
- [Phase ?]: ratchet_floor_percent = lines_percent - 1.0 (91.30% measured → 90.3% floor); mirrors Phase 2 accuracy-lock baseline-lock pattern
- [Phase ?]: Lock coverage baseline as-measured: 91.30% measured, floor 90.3%, scope --package anofox-forecast --all-features; CI enforces via --fail-under-lines in existing coverage: job (COVER-01)
- [Phase ?]: Gap inventory 03-GAP-INVENTORY.md filed with 5 P1 / 9 P2 / 6 P3 rows (20 total) for Phase 4 backlog; highest-value P1 targets are V-01 to V-04 raw-vec NaN guards and G-01 GlobalTheta 0% coverage (COVER-02)
- [Phase ?]: Used !v.is_finite() predicate for NaN/Inf guards in GlobalETS/GlobalCroston/GlobalTheta fit() paths — clippy-preferred, matches VAR::fit pattern
- [Phase ?]: V-04 (VAR MissingValues vs InvalidParameter variant divergence) deferred to 04-03 backlog — document-only, no refactor this phase
- [Phase ?]: Coverage baseline committed separately per IMPR-03: 91.30% → 91.44%, floor 90.3% → 90.4%
- [Phase ?]: L3/L4 root-cause confirmed: AutoETS::new() without period=12 suppressed all seasonal candidates; period=12 fix reduces MASE from 1.0452 to 0.8923 (−14.6%) but anchor FAILED (0.8923 outside [0.8433,0.8833]); accuracy.json deferred pending checkpoint decision
- [Phase ?]: L3/L4 root-cause confirmed: AutoETS::new() without period=12 suppressed all seasonal candidates; fix is harness-only (AutoETS::with_period(period))
- [Phase ?]: MASE improved 1.0452 → 0.8923 (−14.6%), but anchor FAILED (0.8923 > 0.8833 upper tolerance); decision=DEFER: accuracy.json stays absent, residual +0.0290 gap is #1 backlog item for 04-03
