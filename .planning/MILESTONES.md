# Milestones

## v1.1 Robustness Fixes & ERM Reconciliation (Shipped: 2026-09-09)

**Phases completed:** 4 phases, 4 plans, 10 tasks

**Key accomplishments:**

- The MFLES auto-multiplicative runaway (#219) is fixed at all three failure points and locked behind a committed before/after regression — a near-zero-month series that used to forecast ~13× level now forecasts at ~level.
- Complete 19-model auditable sweep finds zero new offenders: the #10/#219 ln→boosting→exp blow-up class is architecturally unique to MFLES (already fixed Phase 5); Theta and AutoETS guard-assertion tests empirically confirm both models stay at level for a near-zero series.
- Fixed-λ ERM hierarchical reconciliation via Cholesky ridge solve (P=B'Ŷ(Ŷ'Ŷ+λI)⁻¹), proven correct against hand-computed reference values within 1e-10 and coherent across all horizon steps
- Ledoit-Wolf auto-λ ERM via Option<f64> variant, proven coherent on a 9-node 2×2 crossed hierarchy with RMSSE −59.7% vs unreconciled (0.922 vs 2.287, seeded deterministic)

---
