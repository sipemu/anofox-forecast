# Phase 4: Prioritized Improvement Backlog & Top-Value Fixes - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Real baseline numbers from Phases 1–3 drive a consolidated, ranked improvement backlog; the
highest-value, lowest-risk improvements are LANDED this phase, each proven by a documented
before/after delta in the relevant baseline file; and regression guards are tightened to reflect
the new thresholds. This is the closing phase of the performance & validation hardening milestone.

It is NOT a new-feature or algorithmic-research phase. Improvements are only "done" when backed by
a reproducible before/after measurement (the project core value). Baseline updates are committed as
separate, deliberate changes — never auto-written by CI.

</domain>

<decisions>
## Implementation Decisions

### What to LAND this phase (IMPR-02)
- **P1 raw-vec NaN/Inf guards (from Phase 3 gap inventory V-01/V-02/V-03):** add per-series,
  per-element NaN/Inf scans at the top of `GlobalETS::fit`, `GlobalCroston::fit`, and
  `GlobalTheta::fit` (raw `&[Vec<f64>]` APIs) returning `ForecastError::InvalidParameter` (or the
  established variant) before the estimation loop. These close real silent-NaN correctness risks.
- **G-01 — GlobalTheta 0% coverage:** add at least a smoke test (constructor + basic fit/predict on
  valid data + the new NaN-guard path) so the model is no longer entirely uncovered.
- **V-04 (VAR variant divergence): DOCUMENT ONLY.** The guard already exists; record the
  `MissingValues` vs `InvalidParameter` variant divergence as a known inconsistency in the backlog —
  do not refactor error types this phase.
- **Proof:** each landed fix is proven with a **before/after `coverage.json` delta** (re-run
  `scripts/update_coverage.sh`; the new tests raise line coverage). Capture the delta explicitly.

### Accuracy MASE gap — time-boxed best-effort (RESOLVED after conflicting answers)
- Add a **bounded, best-effort investigation** into the AutoETS M3-monthly MASE gap (~1.045 vs the
  ~0.93 anchor / 0.8633 statsforecast 2.x reference).
- **IF** a change measurably improves MASE (verified via the Phase 2 harness / accuracy anchor),
  land it and document the **before/after accuracy delta**.
- **IF it does not converge within the time-box, STOP** — keep `accuracy.json` deferred/UNLOCKED
  (do NOT force a lock with a failing anchor), and leave the gap as the **#1 ranked backlog item**.
  No rushed or forced accuracy.json lock. This preserves the Phase 2 defer-lock discipline.

### Backlog + ranking (IMPR-01)
- Consolidated backlog ranks findings from **all 8 measurement dimensions** by value/effort using
  the **real captured numbers** from Phases 1–3.
- **Explicitly flag placeholder baselines:** `iai.json` and `criterion.json` are structural
  placeholders requiring manual capture on specific hardware (valgrind ≥3.20 machine /
  quiet local machine per the Phase 1 blocker). Do NOT fabricate perf deltas for them — mark those
  backlog items "manual-capture-pending" with the capture command.
- Rank rationale documented (value, effort, risk, evidence/baseline source per item).
- Inputs include: Phase 3 gap inventory (20 rows: 5 P1 / 9 P2 / 6 P3), the deferred accuracy MASE
  gap (ranked #1 as fallback), assertion-density gaps, wasm-size, and any perf placeholders.

### Regression guard tightening (IMPR-03)
- After the fixes land and coverage rises, **ratchet the `coverage.json` floor UP** to reflect the
  new (higher) coverage; commit the baseline update as a **separate, deliberate change** — never
  CI-auto-written. The existing `ci.yml --fail-under-lines` gate then enforces the tighter floor.
- Any other tightened guard follows the same rule: measured, committed deliberately, enforced in CI.

### Claude's Discretion
- Exact backlog document location/format (recommend `.planning/baselines/BACKLOG.md` or a phase-dir
  markdown), the precise NaN-guard error variant per model (match each model's existing convention),
  the accuracy investigation's specific hypotheses, and the time-box size are at Claude's discretion.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 3 `03-GAP-INVENTORY.md` — the ranked-candidate source (V-01..V-04 P1 guards, G-01
  GlobalTheta 0% coverage, assertion-free tests A-01..A-05).
- `scripts/update_coverage.sh` — re-run to capture the after-delta coverage number.
- Phase 2 accuracy harness (`crates/anofox-bench-harness/tests/accuracy.rs`, ACCUR-08 anchor,
  `emit_accuracy_baseline_if_write_flag_set`) — the measurement rig for any accuracy delta.
- `.planning/baselines/*.json` — coverage/wasm_size have real numbers; iai/criterion are placeholders.
- `ForecastError::InvalidParameter` — the variant VAR/global models use for raw-vec NaN.

### Established Patterns
- Every improvement proven with a before/after baseline delta; baselines carry provenance blocks.
- Defer-lock discipline (Phase 2): never lock a baseline that fails its validation anchor.
- Coverage floor is an up-only ratchet enforced by ci.yml `--fail-under-lines`; publish `needs: coverage`.

### Integration Points
- `src/models/exponential/global_ets.rs::fit`, `src/models/intermittent/global_croston.rs::fit`,
  `src/models/theta/global_theta.rs::fit` — the three raw-vec fit paths to guard.
- `.planning/baselines/coverage.json` — before/after delta + ratchet target.
- `.planning/baselines/accuracy.json` — stays absent/deferred unless the MASE attempt succeeds.
- A new consolidated backlog document (ranked, value/effort, evidence-linked).

</code_context>

<specifics>
## Specific Ideas

- Land the P1 NaN guards + GlobalTheta smoke test FIRST (safe, measurable coverage delta) — this is
  the guaranteed IMPR-02/03 deliverable regardless of the accuracy attempt's outcome.
- The accuracy MASE attempt is time-boxed and non-blocking: a measured improvement is a bonus delta;
  no improvement means the gap stays #1 backlog and accuracy.json stays deferred. Do not force a lock.
- Do not fabricate perf deltas for placeholder iai/criterion baselines — mark manual-capture-pending.

</specifics>

<deferred>
## Deferred Ideas

- Full closure of the AutoETS MASE gap (if the time-boxed attempt does not converge) — ranked #1
  backlog item for a future dedicated accuracy effort.
- iai.json / criterion.json real-number capture — requires specific hardware (valgrind ≥3.20 / quiet
  machine); documented as manual-capture-pending, not attempted here.
- V-04 VAR error-variant unification — documented as a known inconsistency, not refactored this phase.

</deferred>
