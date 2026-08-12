# Roadmap: anofox-forecast — Performance & Validation Hardening

**Core Value:** Every claimed capability is measured, and every improvement is proven with a
before/after number.

## Shipped Milestones

- ✅ **v1.0 — Performance & Validation Hardening** (Phases 1–4, 13 plans, 28/28 requirements) —
  shipped 2026-08-12. Measurement harnesses + committed baselines per dimension, a statistically
  correct accuracy harness (Naive2 + Diebold-Mariano + cross-library), numerical-robustness
  edge/property suites, a CI-enforced coverage floor (90.4%), and a ranked improvement backlog with
  top-value fixes landed (each proven by a before/after delta). See
  [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md) and
  [`v1.0-MILESTONE-AUDIT.md`](v1.0-MILESTONE-AUDIT.md).

  **Carried forward (`baselines/BACKLOG.md`):** #1 AutoETS M3-monthly accuracy gap (MASE 0.8923 vs
  0.8633, +0.0290 above anchor tolerance — `accuracy.json` deferred), iai/criterion baseline manual
  hardware capture, optional Nyquist validation for Phases 3–4.

## Next Milestone

Start with `/gsd-new-milestone` (defines fresh requirements → research → roadmap). The v1.0 backlog
(`.planning/baselines/BACKLOG.md`) is the natural seed for the next cycle.
