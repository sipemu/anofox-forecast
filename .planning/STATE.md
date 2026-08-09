# Project State: anofox-forecast — Performance & Validation Hardening

**Last Updated:** 2026-08-09
**Session:** Roadmap initialization

---

## Project Reference

**Core Value:** Every claimed capability is measured, and every improvement is proven with a before/after number.

**Current Focus:** Phase 1 — Measurement Infrastructure & Compute Baselines

---

## Current Position

**Phase:** 1 — Measurement Infrastructure & Compute Baselines
**Plan:** None yet (planning not started)
**Status:** Not started

```
Progress: [░░░░░░░░░░░░░░░░░░░░] 0%

Phase 1 [NOT STARTED] ░░░░░
Phase 2 [NOT STARTED] ░░░░░
Phase 3 [NOT STARTED] ░░░░░
Phase 4 [NOT STARTED] ░░░░░
```

---

## Performance Metrics

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| Requirements mapped | 28/28 | 28/28 | — |
| Phases complete | 0/4 | 0/4 | — |
| Plans complete | — | — | — |

---

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

- [ ] Plan Phase 1 via `/gsd-plan-phase 1`

### Blockers

None at this time.

---

## Session Continuity

### What Was Done This Session

- Loaded PROJECT.md, REQUIREMENTS.md, research/SUMMARY.md, config.json, codebase/ARCHITECTURE.md
- Derived 4 phases from 28 v1 requirements (MEAS×4 + PERF×6 + ACCUR×8 + ROBUST×3 + COVER×2 + BENCH×2 + IMPR×3)
- Applied sequencing constraints: PERF-06 before PERF-05; ACCUR validation before BENCH claims; IMPR last
- Wrote ROADMAP.md with 4 phases, success criteria, and progress table
- Updated REQUIREMENTS.md traceability section

### Resume Point

Start `/gsd-plan-phase 1` — Phase 1: Measurement Infrastructure & Compute Baselines covers MEAS-01..04 and PERF-01..06.

---

*State initialized: 2026-08-09*
