# Roadmap: anofox-forecast — Performance & Validation Hardening

**Core Value:** Every claimed capability is measured, and every improvement is proven with a
before/after number.

## Milestones

- ✅ **v1.0 — Performance & Validation Hardening** — Phases 1–4 (shipped 2026-08-12)
- ✅ **v1.1 — Robustness Fixes & ERM Reconciliation** — Phases 5–8 (shipped 2026-09-09)

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

<details>
<summary>✅ v1.1 — Robustness Fixes & ERM Reconciliation (Phases 5–8) — SHIPPED 2026-09-09</summary>

12/12 requirements satisfied, milestone audit PASSED. Two technically independent workstreams:

- **Workstream A — Multiplicative-guard robustness** (Phases 5–6): fixed the MFLES
  auto-multiplicative runaway (#219) at all three failure points (mode guard, ln() floor,
  back-transform clamp) proven with a committed ~13×→~level regression; then a complete 19-model
  auditable sweep found zero new offenders — the #10/#219 ln→boosting→exp blow-up class is
  architecturally unique to MFLES (Theta/AutoETS confirmed safe by guard-assertion tests).
- **Workstream B — ERM reconciliation** (Phases 7–8): added `ReconciliationMethod::Erm { lambda:
  Option<f64> }` (Ben Taieb & Koo 2019) with a training-history API and an in-house Cholesky ridge
  solve `P = BŶᵀ(ŶŶᵀ+λI)⁻¹`, proven correct against a hand-computed reference (and an asymmetric
  independent Gauss-Jordan oracle); then a Ledoit-Wolf-style auto-λ default validated end-to-end on
  a 9-node 2×2 grouped/crossed hierarchy — coherence hard-asserted, RMSSE −59.7% vs unreconciled.

- [x] Phase 5: MFLES Multiplicative-Guard Fix (1/1 plans) — completed 2026-09-09
- [x] Phase 6: Multiplicative-Guard Bug-Class Audit (1/1 plans) — completed 2026-09-09
- [x] Phase 7: ERM Reconciliation Variant & Ridge Solve (1/1 plans) — completed 2026-09-09
- [x] Phase 8: Auto-λ & Grouped/Crossed Validation (1/1 plans) — completed 2026-09-09

Full detail: [`milestones/v1.1-ROADMAP.md`](milestones/v1.1-ROADMAP.md) and
[`v1.1-MILESTONE-AUDIT.md`](milestones/v1.1-MILESTONE-AUDIT.md).

**Carried forward (not v1.1 scope):** Nyquist VALIDATION.md files remain `status: draft`
(NOT-VALIDATED — run `/gsd-validate-phase 5..8` if formal compliance is wanted); stale tracked
`src/hierarchy/mod.rs.bak` recommended for deletion; `cargo test --all-features` OOMs the sandbox
linker (CI gate exercised via clippy --all-features + default-feature suite).

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Measurement Infrastructure & Compute Baselines | v1.0 | 3/3 | Complete | 2026-08-10 |
| 2. Accuracy Harness & Statistical Methodology | v1.0 | 4/4 | Complete | 2026-08-11 |
| 3. Numerical Robustness & Coverage Baseline | v1.0 | 3/3 | Complete | 2026-08-11 |
| 4. Prioritized Improvement Backlog & Top-Value Fixes | v1.0 | 3/3 | Complete | 2026-08-12 |
| 5. MFLES Multiplicative-Guard Fix | v1.1 | 1/1 | Complete | 2026-09-09 |
| 6. Multiplicative-Guard Bug-Class Audit | v1.1 | 1/1 | Complete | 2026-09-09 |
| 7. ERM Reconciliation Variant & Ridge Solve | v1.1 | 1/1 | Complete | 2026-09-09 |
| 8. Auto-λ & Grouped/Crossed Validation | v1.1 | 1/1 | Complete | 2026-09-09 |

---
*Roadmap created: 2026-08-09*
*Last updated: 2026-09-09 — v1.1 shipped (Phases 5–8: Robustness Fixes & ERM Reconciliation)*
