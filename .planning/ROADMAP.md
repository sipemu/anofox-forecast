# Roadmap: anofox-forecast — Performance & Validation Hardening

**Core Value:** Every claimed capability is measured, and every improvement is proven with a
before/after number.

## Milestones

- ✅ **v1.0 — Performance & Validation Hardening** — Phases 1–4 (shipped 2026-08-12)
- 🚧 **v1.1 — Robustness Fixes & ERM Reconciliation** — Phases 5–8 (in progress)

## Phases

<details>
<summary>✅ v1.0 — Performance & Validation Hardening (Phases 1–4) — SHIPPED 2026-08-12</summary>

28/28 requirements satisfied, milestone audit INTEGRATED. Delivered: committed measurement
baselines per dimension (speed/memory/WASM-size/coverage) with CI guards; a statistically correct
accuracy harness (competition MASE, Naive2, Diebold-Mariano, pinned statsforecast cross-library
reference); numerical-robustness edge-case + property suites with per-family NaN/Inf guards; a
CI-enforced coverage floor (90.4%); and a ranked improvement backlog with top-value fixes landed
(each proven by a before/after delta).

Full detail: [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md) and
[`v1.0-MILESTONE-AUDIT.md`](v1.0-MILESTONE-AUDIT.md).

**Carried forward (`baselines/BACKLOG.md`, not v1.1 scope):** ACC-01 AutoETS M3-monthly accuracy
gap (MASE 0.8923 vs 0.8633, +0.0290 above anchor tolerance — `accuracy.json` deferred), MEM-01,
WSZ-01, coverage gaps, iai/criterion manual hardware capture.

</details>

### 🚧 v1.1 — Robustness Fixes & ERM Reconciliation (In Progress)

**Milestone Goal:** Close a silent multiplicative-mode over-forecast bug class and add ERM
hierarchical reconciliation — each proven against a baseline, keeping the "every improvement has a
before/after number" discipline.

Two technically independent workstreams:

- **Workstream A — Multiplicative-guard robustness** (Phases 5–6): fix the MFLES
  auto-multiplicative runaway (#219), then audit every model with an auto-multiplicative/log path
  for the same too-loose-guard failure class.
- **Workstream B — ERM reconciliation** (Phases 7–8): add `ReconciliationMethod::Erm { lambda }`
  (Ben Taieb & Koo 2019) with a training-history API and ridge solve, then a Ledoit-Wolf-style
  auto-λ and end-to-end grouped/crossed validation against a baseline.

- [x] **Phase 5: MFLES Multiplicative-Guard Fix** - Tighten the MFLES min/level guard, floor the ln() transform, clamp the back-transform; prove the #219 repro series drops from ~13× to ~level (completed 2026-09-09)
- [x] **Phase 6: Multiplicative-Guard Bug-Class Audit** - Inventory every auto-multiplicative/log selection path across models; fix any exhibiting the too-loose-guard failure class with regression guards (completed 2026-09-09)
- [x] **Phase 7: ERM Reconciliation Variant & Ridge Solve** - Add the `Erm { lambda }` variant, training-history API, and the `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹` fixed-λ solve, verified against the reference formula (completed 2026-09-09)
- [ ] **Phase 8: Auto-λ & Grouped/Crossed Validation** - Add the Ledoit-Wolf-style auto-λ default and validate ERM end-to-end on a grouped/crossed hierarchy against a MinTrace/unreconciled baseline

## Phase Details

### Phase 5: MFLES Multiplicative-Guard Fix

**Goal**: The MFLES auto-multiplicative runaway (#219) is fixed at all three failure points — mode
selection, the log transform, and the back-transform — and the fix is proven with a committed
before/after on the repro series.
**Depends on**: Phase 4 (v1.0 baselines/regression conventions)
**Requirements**: MULT-01, MULT-02, MULT-03, MULT-04
**Success Criteria** (what must be TRUE):

  1. When a series has a value near-zero relative to its level (min/level below threshold τ), MFLES selects additive mode instead of multiplicative — verified by a unit test on a synthetic near-zero series
  2. MFLES multiplicative mode floors/winsorizes the `ln()` transform so a single near-zero observation can no longer open a log-space crater, and the multiplicative back-transform is clamped to a bounded multiple of the in-sample level
  3. The #219 repro series (near-zero month, level ≈ 2k) forecasts at ~level; a regression test asserts the forecast is within a bounded multiple of level (no ~13× blow-up)
  4. The before/after delta (~13× → ~level) is committed as the regression guard, honoring the Core Value's before/after proof discipline
  5. Public `Forecaster` API is unchanged, the npm/WASM package still builds, and clippy `-D warnings` + cargo-audit/deny remain green

**Plans**: 1/1 plans executed

- [x] 05-01-PLAN.md — Mode guard + log floor + back-transform clamp in mfles.rs, with committed #219 before/after regression (MULT-01..04)

### Phase 6: Multiplicative-Guard Bug-Class Audit

**Goal**: Every model with an auto-multiplicative/log selection path is inventoried and audited for
the same too-loose-guard failure class as MFLES/#219 (the #10 lineage); any additional offender is
fixed and guarded against regression.
**Depends on**: Phase 5 (reuses the tightened-guard pattern and regression-test shape)
**Requirements**: MULT-05, MULT-06
**Success Criteria** (what must be TRUE):

  1. A committed inventory lists every model with an auto-multiplicative or log selection path (e.g. AutoETS/#10 lineage, Theta/TBATS log options, any transform-driven mode selection) with a pass/fail audit verdict for the too-loose-guard failure class
  2. Each model flagged as at-risk is exercised with a near-zero-relative-to-level series; models that blow up before the fix and forecast at ~level after are documented with a before/after number
  3. Any additional offending model is fixed with the same guard discipline (min/level threshold, floored transform, clamped back-transform as applicable) and a regression test proving the pre-fix blow-up no longer occurs
  4. Models audited and found already-safe are recorded as such with the evidence (why the existing guard is tight enough), so the inventory is a complete, auditable sweep
  5. Public `Forecaster` API is unchanged, the npm/WASM package still builds, and clippy `-D warnings` + cargo-audit/deny remain green

**Plans**: 1/1 plans executed

- [x] 06-01-PLAN.md — Commit the MULT-05 19-row inventory doc + Theta/AutoETS near-zero guard-assertion tests; record MULT-06 zero-offender satisfied-by-evidence

### Phase 7: ERM Reconciliation Variant & Ridge Solve

**Goal**: `ReconciliationMethod::Erm { lambda }` exists as a backward-compatible variant fed by a
training-history API, computes the regularized ridge projection, and is verified numerically against
the reference formula on a known small hierarchy.
**Depends on**: Phase 4 (existing hierarchy/reconciliation subsystem); independent of Phases 5–6
**Requirements**: ERM-01, ERM-02, ERM-03, ERM-05
**Success Criteria** (what must be TRUE):

  1. `ReconciliationMethod::Erm { lambda }` exists alongside BottomUp/TopDown/MiddleOut/MinTrace*; existing hierarchy code compiles unchanged (backward-compatible additive variant)
  2. A training-history API accepts base forecasts + leaf actuals across all nodes over T periods and feeds them to the ERM solve
  3. ERM computes the projection via the regularized ridge solve `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹`, with reconciled bottom = `P·ŷ` and reconciled all = `S·P·ŷ`
  4. On a known small hierarchy with a hand-computable answer, ERM's output matches the reference formula within numerical tolerance (correctness proof, not just "it runs")
  5. Work respects `serde`/feature gates as applicable, the npm/WASM package still builds, and clippy `-D warnings` + cargo-audit/deny remain green

**Plans**: 1/1 plans executed

Plans:

- [x] 07-01-PLAN.md — ERM variant + Eq-drop, training-history API, ridge solve `P = B'Ŷ(Ŷ'Ŷ+λI)⁻¹`, hand-computed correctness + coherence tests (ERM-01/02/03/05)

### Phase 8: Auto-λ & Grouped/Crossed Validation

**Goal**: ERM ships with a Ledoit-Wolf-style auto-λ default matching `MinTraceShrink` ergonomics
(fixed-λ path still available) and is validated end-to-end on a grouped/crossed hierarchy against a
MinTrace/unreconciled baseline with the accuracy before/after committed.
**Depends on**: Phase 7 (auto-λ and validation build on the variant + solve)
**Requirements**: ERM-04, ERM-06
**Success Criteria** (what must be TRUE):

  1. A Ledoit-Wolf-style auto-λ default is available with the same ergonomics as `MinTraceShrink`, and a caller-supplied fixed-λ path is still supported (both selectable through the public API)
  2. ERM is run end-to-end on a grouped/crossed hierarchy (multiple crossing dimensions), producing coherent reconciled forecasts across all aggregation levels
  3. ERM accuracy is measured against a MinTrace/unreconciled baseline on the same hierarchy and horizon; the before/after accuracy delta is committed, honoring the Core Value's before/after proof discipline
  4. The auto-λ default is exercised in the end-to-end validation (not only the fixed-λ path), demonstrating the shrink default behaves sensibly on real crossed data
  5. Public API stays backward-compatible, the npm/WASM package still builds, and clippy `-D warnings` + cargo-audit/deny remain green

**Plans**: TBD

## Progress

**Execution Order:** Phases execute in numeric order: 5 → 6 → 7 → 8. Workstream A (5–6) and
Workstream B (7–8) are technically independent and could progress in parallel; the numeric order is
the default sequencing.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Measurement Infrastructure & Compute Baselines | v1.0 | 3/3 | Complete | 2026-08-10 |
| 2. Accuracy Harness & Statistical Methodology | v1.0 | 4/4 | Complete | 2026-08-11 |
| 3. Numerical Robustness & Coverage Baseline | v1.0 | 3/3 | Complete | 2026-08-11 |
| 4. Prioritized Improvement Backlog & Top-Value Fixes | v1.0 | 3/3 | Complete | 2026-08-12 |
| 5. MFLES Multiplicative-Guard Fix | v1.1 | 1/1 | Complete    | 2026-09-09 |
| 6. Multiplicative-Guard Bug-Class Audit | v1.1 | 1/1 | Complete    | 2026-09-09 |
| 7. ERM Reconciliation Variant & Ridge Solve | v1.1 | 1/1 | Complete    | 2026-09-09 |
| 8. Auto-λ & Grouped/Crossed Validation | v1.1 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-08-09*
*Last updated: 2026-09-08 — added v1.1 (Phases 5–8: Robustness Fixes & ERM Reconciliation)*
