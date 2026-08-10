---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 02
current_phase_name: accuracy-harness-statistical-methodology
status: verifying
stopped_at: "Completed 02-04-PLAN.md (defer-lock: accuracy.json deferred; accuracy.yml MEAS-03 delivered)"
last_updated: "2026-08-10T21:48:16.305Z"
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
---

# Project State: anofox-forecast — Performance & Validation Hardening

**Last Updated:** 2026-08-10
**Session:** Phase 1 verified & complete

---

## Project Reference

**Core Value:** Every claimed capability is measured, and every improvement is proven with a before/after number.

**Current Focus:** Phase 02 — accuracy-harness-statistical-methodology

---

## Current Position

**Phase:** 02 (accuracy-harness-statistical-methodology) — EXECUTING
**Plan:** 4 of 4
**Status:** Phase complete — ready for verification

```
Progress: [██████████] 100%

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

**Last session:** 2026-08-10T21:48:07.747Z
**Stopped at:** Completed 02-04-PLAN.md (defer-lock: accuracy.json deferred; accuracy.yml MEAS-03 delivered)
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
