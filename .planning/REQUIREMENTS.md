# Requirements: anofox-forecast — v1.1 Robustness Fixes & ERM Reconciliation

**Defined:** 2026-09-08
**Core Value:** Every claimed capability is measured, and every improvement is proven with a before/after number.

## v1.1 Requirements

Requirements for milestone v1.1. Each maps to exactly one roadmap phase.

### Multiplicative-Guard Robustness

Fixes the silent multiplicative-mode over-forecast bug class (GitHub #219, same failure class as
the closed #10). Every fix is proven with a before/after on a repro series and guarded against regression.

- [x] **MULT-01**: MFLES no longer selects multiplicative mode when a series value is near-zero relative to its level (min/level below a threshold τ); it falls back to additive.
- [x] **MULT-02**: MFLES multiplicative mode floors/winsorizes the `ln()` transform so a single near-zero observation cannot create a log-space crater.
- [x] **MULT-03**: MFLES multiplicative back-transform is clamped so a forecast cannot exceed a bounded multiple of the in-sample level (runaway backstop).
- [x] **MULT-04**: The #219 repro series (near-zero month, level ≈ 2k) forecasts at ~level instead of ~13×; the before/after delta is committed as a regression guard.
- [x] **MULT-05**: Every model with an auto-multiplicative/log selection path is inventoried and audited for the same too-loose-guard failure class (the #10/#219 lineage).
- [x] **MULT-06**: Any additional model exhibiting the failure class is fixed and guarded by a regression test proving the pre-fix blow-up no longer occurs.

### ERM Reconciliation

Adds ERM (empirical-risk-minimization) hierarchical reconciliation (Ben Taieb & Koo 2019, KDD) as a
`ReconciliationMethod` variant (GitHub #216). Backward-compatible; validated against a baseline.

- [ ] **ERM-01**: `ReconciliationMethod::Erm { lambda }` exists alongside BottomUp/TopDown/MiddleOut/MinTrace*, fully backward-compatible (existing hierarchy code compiles unchanged).
- [ ] **ERM-02**: A training-history API accepts base forecasts + leaf actuals across all nodes (T periods) for ERM to consume.
- [ ] **ERM-03**: ERM computes the projection via the regularized ridge solve `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹`; reconciled bottom = `P·ŷ`, reconciled all = `S·P·ŷ`.
- [ ] **ERM-04**: A Ledoit-Wolf-style auto-λ default is available (matching `MinTraceShrink` ergonomics), with a caller-supplied fixed-λ path also supported.
- [ ] **ERM-05**: ERM correctness is verified against the reference formula on a known small hierarchy (numerical agreement within tolerance).
- [ ] **ERM-06**: ERM is validated end-to-end on a grouped/crossed hierarchy against a MinTrace/unreconciled baseline, with the accuracy before/after committed.

## Future Requirements

Deferred to a later milestone. Tracked but not in the current roadmap.

### v1.0 Backlog (carried, see `baselines/BACKLOG.md`)

- **ACC-01**: Close residual AutoETS M3-monthly MASE gap (+0.0290 above reference); lock `accuracy.json`.
- **MEM-01**: Reduce AutoETS peak-memory outlier (290 KB vs ~190–200 KB for other families).
- **WSZ-01**: WASM binary size reduction from the 2.84 MB baseline.
- **Coverage gaps G-02, G-04…G-10**: SmartForecaster dispatch, batch Err arms, CQR boundaries, Inspectable, GPD tails, etc.
- **A-01…A-05**: Add assertions to cross-library comparison tests that currently only print.
- **V-04**: Align VAR error-variant divergence across fit paths.
- **Manual-capture-pending**: Populate iai/criterion baselines (valgrind ≥3.20 / quiet machine).

## Out of Scope

Explicitly excluded from v1.1. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New forecasting models or model families | Project remains hardening-oriented; ERM is a reconciliation method over existing forecasts, not a new forecaster |
| Public `Forecaster`/hierarchy API redesigns | Improvements stay backward-compatible; ERM is an additive enum variant + new API surface only |
| Automatic seasonal-period-detection integration into models | Deliberately excluded (carried from v1.0) |
| The v1.0 accuracy/memory/WASM/coverage backlog | Real work, but not this milestone's focus — tracked in Future Requirements above |
| New Python bindings | Out of scope for this cycle |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MULT-01 | Phase 5 | Complete |
| MULT-02 | Phase 5 | Complete |
| MULT-03 | Phase 5 | Complete |
| MULT-04 | Phase 5 | Complete |
| MULT-05 | Phase 6 | Complete |
| MULT-06 | Phase 6 | Complete |
| ERM-01 | Phase 7 | Pending |
| ERM-02 | Phase 7 | Pending |
| ERM-03 | Phase 7 | Pending |
| ERM-04 | Phase 8 | Pending |
| ERM-05 | Phase 7 | Pending |
| ERM-06 | Phase 8 | Pending |

**Coverage:**

- v1.1 requirements: 12 total
- Mapped to phases: 12 (Phase 5: 4, Phase 6: 2, Phase 7: 4, Phase 8: 2) ✓
- Unmapped: 0

---
*Requirements defined: 2026-09-08*
*Last updated: 2026-09-08 — traceability populated during v1.1 roadmap creation*
