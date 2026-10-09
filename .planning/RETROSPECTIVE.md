# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 — Robustness Fixes & ERM Reconciliation

**Shipped:** 2026-09-09
**Phases:** 4 (5–8) | **Plans:** 4 | **Sessions:** 1 (autonomous)

### What Was Built
- MFLES auto-multiplicative runaway (#219) fixed at all three failure points (min/median mode guard τ=0.10, `ln()` winsorized to 0.01×median, back-transform clamped to 10×in-sample max) — proven by a committed ~13×→~level regression.
- A 19-model auditable sweep of every auto-multiplicative/log selection path — zero new offenders; the ln→boosting→exp blow-up class is architecturally unique to MFLES. Theta/AutoETS confirmed safe by two-sided guard-assertion tests.
- ERM hierarchical reconciliation: `ReconciliationMethod::Erm { lambda: Option<f64> }` with a training-history API and an in-house Cholesky ridge solve `P = BŶᵀ(ŶŶᵀ+λI)⁻¹`, proven correct against a hand-computed reference and an asymmetric independent Gauss-Jordan oracle.
- Ledoit-Wolf-style auto-λ default (self-consistent with the uncentered Gram) validated end-to-end on a 9-node 2×2 grouped/crossed hierarchy — coherence hard-asserted, RMSSE −59.7% vs unreconciled (drift-locked).

### What Worked
- **Tracer-first planning** on every phase: the tracer task threaded the whole feature (variant → solve → correctness proof) before expansion, so each phase had an end-to-end proof early.
- **Adversarial code review caught real correctness bugs** that tests alone missed: Phase 7's negative/NaN-λ and T=0 silent-corruption guards, Phase 8's floor-bypass and the centered-vs-uncentered-Gram λ scale mismatch, and — most valuably — the "tautological test" findings (Phase 6 one-sided bounds, Phase 8's cherry-picked baseline). Each was fixed and re-verified.
- **Independent-oracle testing** (an in-test Gauss-Jordan solver vs the production Cholesky path) proved the ERM ridge solve had no transposition/ordering bug — a class a single symmetric test would have missed.
- **The before/after discipline held throughout**: every robustness fix and the ERM accuracy claim is backed by a committed, drift-locked number, honoring the project's Core Value.

### What Was Inefficient
- **Executor stalls / API drops** on the two largest phases (5 and 7): the executor hit the 600s watchdog / a connection drop mid-run. Recovered via the documented filesystem-spot-check + re-dispatch path, but cost wall-clock. Long single-file edits (mfles.rs 2.4k lines, hierarchy/mod.rs 3.4k lines) are stall-prone.
- **`cargo test --all-features` OOMs the sandbox linker** (bus error) — the intended CI gate couldn't run whole; had to substitute clippy --all-features + default-feature suite + targeted --features runs. An environment limitation, but it forced per-run workarounds.
- **The dispatch-isolation sentinel needed manual re-persistence** (`--force-isolation none`) before every executor dispatch because HEAD had diverged from origin/HEAD (feature branch), and any plain resolve overwrote it back to harness-worktree.
- **Pre-existing example/test feature-gating debt** surfaced during Phase 5 verification (bare `cargo test` never compiled 7 distributional examples + 1 test) — absorbed as an in-scope cleanup, but it was unrelated to the MFLES fix.

### Patterns Established
- **Guard-composition discipline** for multiplicative/log models: min/level mode guard + floored transform + clamped back-transform, with the clamp applied ONLY in predict (never fit) to preserve the decomposable invariant.
- **Auditable-sweep artifact**: a committed `docs/audits/*.md` inventory table (per-item verdict + rationale) plus guard-assertion tests for already-safe items — a reusable shape for bug-class audits.
- **Auto-parameter helpers self-consistent with the matrix they regularize** (auto-λ estimated on the same uncentered Gram it's added to), and drift-locked committed numbers (±1% assertions) so "measured" means measured.

### Key Lessons
1. **Adversarial review earns its keep on numerical code** — the highest-value findings this milestone (silent λ corruption, tautological tests, cherry-picked baseline) were review-only; no test would have flagged them.
2. **Long single-file executor tasks are stall-prone** — prefer decomposing edits or expect to recover via spot-check + re-dispatch; the partial work is usually salvageable from the working tree.
3. **"Honest numbers" needs enforcement, not just intent** — a passing test with a one-sided bound or a zero-data baseline can look like a proof while proving little; two-sided bounds, incoherence spot-checks, independent oracles, and drift-locks are what make a before/after trustworthy.
4. **Environment gates can diverge from CI gates** — when `--all-features` can't run locally, name the substitute evidence explicitly rather than claiming the CI gate passed.

### Cost Observations
- Model mix: planning on opus, research/execution/review/verification on sonnet, plan-checking on haiku (per configured model profile).
- Sessions: 1 autonomous run across all 4 phases (discuss → plan → execute → review → verify each), plus milestone lifecycle.
- Notable: 46 commits over ~14h wall-clock; two executor recoveries (Phases 5, 7) handled without losing committed work.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | — | 4 (1–4) | Measurement-first hardening: baselines + CI gates established |
| v1.1 | 1 | 4 (5–8) | Fully autonomous discuss→plan→execute→review→verify per phase; adversarial review + independent-oracle testing added |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | property + edge + accuracy suites | 90.4% floor (CI-enforced) | measurement harnesses |
| v1.1 | +ERM (13) +#219 (3) +audit (2) tests; 2890 default-feature tests green | maintained (no regression) | ERM ridge solve (in-house Cholesky), inline LCG for tests — no new crates |

### Top Lessons (Verified Across Milestones)

1. Every improvement needs a committed before/after number — and the number needs a *falsifiable* test behind it, not just an observation.
2. Backward-compatibility on a published crate is a hard constraint — additive enum arms + builder methods, and serde-default on new fields, keep it intact.
