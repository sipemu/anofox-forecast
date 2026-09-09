---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Robustness Fixes & ERM Reconciliation
current_phase: 7
current_phase_name: ERM Reconciliation Variant & Ridge Solve
status: planning
stopped_at: Phase 06 complete, ready to plan Phase 7
last_updated: "2026-09-09T07:49:05.666Z"
last_activity: 2026-09-09
last_activity_desc: Phase 06 complete, transitioned to Phase 7
state_head: 8758efd4747aa0663230f3d17d4520bb3142a521
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
  percent: 50
---

# Project State: anofox-forecast — Performance & Validation Hardening

**Last Updated:** 2026-09-08
**Session:** v1.1 roadmap created (Phases 5–8)

---

## Project Reference

**Core Value:** Every claimed capability is measured, and every improvement is proven with a before/after number.

**Current Focus:** Phase 06 — Multiplicative-Guard Bug-Class Audit

---

## Current Position

Phase: 7 — ERM Reconciliation Variant & Ridge Solve
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-09 — Phase 06 complete, transitioned to Phase 7

## Performance Metrics

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| v1.1 requirements mapped | 12/12 | 12/12 | — |
| v1.1 phases complete | 0/4 | 0/4 | — |
| v1.1 plans complete | — | — | — |

---
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 06-multiplicative-guard-bug-class-audit P01 | 4 | 3 tasks | 2 files |

## Accumulated Context

### Key Decisions (v1.1)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| MFLES fix (MULT-01..04) is its own phase (5), separate from the broader bug-class audit (6) | The #219 fix has a concrete repro + before/after proof; the audit is an open-ended sweep. Separating keeps each phase independently verifiable and keeps the proven MFLES fix from being blocked on audit scope | Phase 5/6 |
| Audit (MULT-05/06) depends on Phase 5 | The audit reuses the tightened-guard pattern (min/level threshold, floored transform, clamped back-transform) and regression-test shape established while fixing MFLES | Phase 6 |
| ERM variant + solve + correctness (ERM-01/02/03/05) grouped in Phase 7; auto-λ + grouped validation (ERM-04/06) in Phase 8 | Phase 7 delivers a fixed-λ ERM proven correct against the reference formula (a testable unit); Phase 8 layers the Ledoit-Wolf auto-λ and the end-to-end accuracy before/after on top | Phase 7/8 |
| ERM-05 (correctness vs reference formula) placed in Phase 7, not Phase 8 | Correctness of the ridge solve is a property of the solve itself and should be proven where the solve lands, before auto-λ and grouped validation build on it | Phase 7 |
| Workstreams A and B kept technically independent (no forced coupling) | Different subsystems (src/models/mfles.rs + model audit vs hierarchy reconciliation); numeric order 5→6→7→8 is default sequencing, not a hard dependency between workstreams | — |

### Critical Constraints to Honor

- Public `Forecaster` / hierarchy API stays backward-compatible; ERM is an additive enum variant + new API surface only
- The published npm package `@sipemu/anofox-forecast` must keep building (WASM target forbids the `parallel` feature)
- Every robustness fix needs a before/after proof (repro series ~13× → ~level); ERM needs a before/after accuracy proof vs a MinTrace/unreconciled baseline — no unquantified improvements
- clippy `-D warnings` and cargo-audit/deny gates must stay green
- Respect existing feature gates (`distributional`, `postprocess`, `anomaly`, `forecastability`, `seasonal-detection`, `parallel`, `serde`, `js`)
- No customer/client names in code, comments, or test names (public crate)
- Do not add automatic seasonal-period detection integration (out of scope, carried from v1.0)

### Todos

- [ ] Plan Phase 5 via `/gsd-plan-phase 5`

### Blockers

- None for v1.1.
- ⚠️ (carried, NOT v1.1 scope) v1.0 `iai.json` / `criterion.json` baselines remain structural placeholders pending maintainer hardware capture; `accuracy.json` deferred (ACC-01 gap). Tracked in `baselines/BACKLOG.md`.

---

## Session Continuity

**Last session:** 2026-09-09T07:32:31.038Z
**Stopped at:** Phase 06 complete, ready to plan Phase 7
**Resume file:** None

### What Was Done This Session

- Created v1.1 roadmap: Phases 5–8, continuing numbering from v1.0 (ended at Phase 4)
- Mapped all 12 v1.1 requirements to phases (Phase 5: MULT-01..04; Phase 6: MULT-05/06; Phase 7: ERM-01/02/03/05; Phase 8: ERM-04/06)
- Derived 5 observable success criteria per phase, each honoring the before/after proof discipline and backward-compat/CI constraints
- Populated REQUIREMENTS.md traceability (no TBD rows remaining)
- Collapsed v1.0 into a `<details>` summary in ROADMAP.md; v1.1 expanded

### Resume Point

Start `/gsd-plan-phase 5` — Phase 5: MFLES Multiplicative-Guard Fix covers MULT-01..04 (the #219 fix with the ~13× → ~level repro before/after).

---

*State initialized: 2026-08-09. Reset for v1.1 planning: 2026-09-08.*

## v1.0 History (archived)

Full v1.0 phase history, per-plan metrics, and decision log are preserved in
[`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md), [`v1.0-MILESTONE-AUDIT.md`](v1.0-MILESTONE-AUDIT.md),
and [`baselines/BACKLOG.md`](baselines/BACKLOG.md). v1.0 shipped 2026-08-12 with 28/28 requirements
satisfied across Phases 1–4.

## Decisions

- [Phase 06]: All 19 in-scope auto-multiplicative/log paths are PASS or N/A — MULT-06 satisfied by evidence, no production code change required
- [Phase 06]: MFLES ln→boosting→exp blow-up class is architecturally unique; guard-assertion tests empirically confirm Theta and AutoETS stay at level for near-zero series
