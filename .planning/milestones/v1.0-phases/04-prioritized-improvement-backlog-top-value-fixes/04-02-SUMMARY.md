---
phase: 04-prioritized-improvement-backlog-top-value-fixes
plan: "02"
subsystem: bench-harness / accuracy-investigation
status: complete
tags: [accuracy, autoets, mase, m3-monthly, period-fix, defer-lock]

requires:
  - phase: 04-01
    provides: baseline harness infrastructure, coverage ratchet, input-guard fixes

provides:
  - L3/L4 root-cause confirmed and documented: AutoETS harness now passes period=12 for monthly data
  - Measured before/after MASE delta (1.0452 → 0.8923, −14.6%) committed to history
  - Defer-lock discipline preserved: accuracy.json stays absent; residual +0.0290 gap handed to 04-03

affects: [04-03]

actuals:
  tokens: 16000
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Defer-lock discipline: accuracy.json is locked ONLY when the ACCUR-08 anchor passes; a failing anchor keeps it absent and routes the gap to the next plan"

key-files:
  created: []
  modified:
    - crates/anofox-bench-harness/tests/accuracy.rs

key-decisions:
  - "L3/L4 root-cause confirmed: AutoETS::new() without period=12 suppressed all seasonal candidates for monthly data; fix is harness-only (AutoETS::with_period(period))"
  - "MASE improved 1.0452 → 0.8923 (−0.1529, −14.6%), closing 84% of the original gap, but anchor FAILED: 0.8923 outside ±0.02 upper tolerance of 0.8833 by +0.0090"
  - "Decision = DEFER: accuracy.json stays absent; residual +0.0290 gap (reference 0.8633, upper tolerance 0.8833) becomes the #1 backlog item for plan 04-03"
  - "Period=12 harness fix RETAINED as a documented partial improvement (commit b6a75cd) — real, measured, non-regressing"
  - "Two untried low-risk levers (L1: lower seasonal minimum 3×→2×period; L2: relax F-ratio gate) deferred to 04-03 to avoid overfitting the library to one benchmark within this time-box"

patterns-established:
  - "Root-cause lever table: L3/L4 (harness period) confirmed; L1/L2 (src seasonal thresholds) remain as next levers"

requirements-completed: [IMPR-02]

coverage:
  - id: D1
    description: "AutoETS harness call uses AutoETS::with_period(period) for monthly data (period=12), quarterly (period=4), yearly (period=1)"
    requirement: IMPR-02
    verification:
      - kind: unit
        ref: "crates/anofox-bench-harness/tests/accuracy.rs — all 5 tests pass (anchor skips cleanly when ANOFOX_DATASET_DIR unset)"
        status: pass
    human_judgment: false
  - id: D2
    description: "accuracy.json is ABSENT (deferred): anchor FAILED at MASE=0.8923 vs reference 0.8633, +0.0090 outside upper tolerance 0.8833"
    requirement: IMPR-02
    verification:
      - kind: other
        ref: "test ! -f .planning/baselines/accuracy.json && echo DEFERRED → prints DEFERRED (confirmed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Root-cause finding and +0.0290 residual gap documented as #1 backlog item for 04-03 with root-cause note and two remaining levers (L1/L2)"
    requirement: IMPR-02
    verification: []
    human_judgment: true
    rationale: "Backlog ranking and root-cause attribution require human review of the accuracy narrative"

duration: 15min
completed: 2026-08-11
---

# Phase 04 Plan 02: M3-Monthly AutoETS MASE Investigation Summary

**L3/L4 harness root-cause confirmed — period=12 fix cuts MASE from 1.0452 to 0.8923 (−14.6%) but anchor FAILED (0.0090 outside tolerance); accuracy.json stays deferred, residual +0.0290 gap is the #1 04-03 backlog item.**

## Performance

- **Duration:** ~15 min
- **Tasks completed:** 3 of 3 (Task 1 = full implementation; Task 2 = checkpoint:decision resolved as DEFER; Task 3 = defer branch applied)
- **Commits:** 2 (b6a75cd: harness fix; cd18d36: SUMMARY/STATE docs)
- **accuracy.json:** ABSENT/DEFERRED (correct — anchor FAILED)

## Accomplishments

1. **Root-cause L3/L4 confirmed and fixed (harness-only, b6a75cd):** The M3-monthly accuracy run called `AutoETS::new()` which defaults `seasonal_period.unwrap_or(1)`, forcing `has_seasonal = false` for all monthly series. The one-line fix — `AutoETS::with_period(period)` where `period = period_for_freq(freq)` — brings the full seasonal ETS candidate set back into the search for monthly (12), quarterly (4), and yearly (1) data.

2. **Measured before/after MASE delta:** MASE dropped from **1.0452** to **0.8923** (−0.1529, −14.6%). This closes 84% of the original gap (0.1529 out of 0.1819 = reference 0.8633 vs old MASE 1.0452).

3. **Defer-lock discipline preserved:** The ACCUR-08 anchor FAILED (0.8923 > 0.8833 upper tolerance by +0.0090). Per the plan's must-haves and the emit function's internal guard (`anchor_passed=false` → panic), `accuracy.json` was NOT emitted or committed. It remains absent.

4. **Harness regression-free:** All 5 harness tests pass. The period=12 fix caused no regressions.

## Measured MASE

| | MASE | Source |
|--|------|--------|
| **BEFORE** (period=1 / no seasonal candidates) | 1.0452 | STATE.md baseline |
| **AFTER** (period=12 / seasonal candidates enabled) | 0.8923 | Anchor run 2026-08-11 |
| **Reference** (statsforecast 2.0.3) | 0.8633 | statsforecast_reference.json |
| **Upper tolerance** | 0.8833 | reference + 0.02 |
| **Residual gap** | +0.0290 | 0.8923 − 0.8633 |
| **Gap outside tolerance** | +0.0090 | 0.8923 − 0.8833 |

**Anchor result: FAILED** — 0.8923 is 0.0090 above the ±0.02 tolerance band.

```
MEASURED_MONTHLY_MASE=0.8923  ANCHOR=FAILED
```

## Decision: DEFER

The anchor FAILED. Per the defer-lock discipline, `accuracy.json` stays absent/deferred. The period=12 harness fix is retained as a documented partial improvement — it is real, measured, non-regressing, and useful. The residual +0.0290 MASE gap is handed to plan 04-03 as the **#1 backlog item**.

## Handoff to Plan 04-03 (Ranked Backlog)

**#1 backlog item: Close the residual AutoETS M3-monthly MASE gap (+0.0290; +0.0090 outside tolerance)**

Root-cause note: After the L4 harness fix, the remaining gap of +0.0290 (0.8923 vs reference 0.8633) reflects a genuine small algorithmic difference in seasonal model selection between anofox-forecast and statsforecast. Two untried low-risk src levers remain:

| # | Lever | Location | Expected Gap Closure | Risk |
|---|-------|----------|---------------------|------|
| L1 | Lower seasonal minimum from 3×period to 2×period | `auto_ets.rs:427` | MEDIUM — brings 24–35 obs monthly series into seasonal pool | Low (affects only short series) |
| L2 | Relax or remove F-ratio seasonal gate | `auto_ets.rs:430-471` | MEDIUM — some series suppressed by gate may fit better seasonally | Medium (test impact unknown) |

These were NOT attempted in this plan to avoid overfitting the library to a single benchmark within the time-box. A future dedicated accuracy effort should try L1 first (lower risk), measure the MASE delta, then decide whether L2 is warranted.

**Entry condition for locking accuracy.json:** Re-run `accur08_anchor_m3_monthly_autoets` with `ANOFOX_DATASET_DIR` set; if MASE ≤ 0.8833, set `ANOFOX_WRITE_ACCURACY_BASELINE=1` and re-run the full harness to emit the file. The emit function enforces the gate internally.

## Deviations from Plan

None — plan executed exactly as written. Task 1 (period=12 fix + measurement), Task 2 (checkpoint:decision = DEFER per human decision), Task 3 (defer branch: retain fix, keep accuracy.json absent, document outcome for 04-03).

## Known Stubs

None.

## Threat Surface Scan

No new security-relevant surface introduced. The fix is harness-only (test code in `crates/anofox-bench-harness/tests/accuracy.rs`). No `src/` changes. T-04-03 (accuracy.json tampering) remains active but NOT triggered: `accuracy.json` is NOT written because `anchor_passed=false` and the emit function's guard would panic if forced.

## Self-Check

- [x] Period=12 harness fix retained: `b6a75cd` exists in git log
- [x] `accuracy.json` ABSENT/DEFERRED: `test ! -f .planning/baselines/accuracy.json` → DEFERRED (correct)
- [x] `accuracy.yml` untouched (still workflow_dispatch-only)
- [x] Before/after MASE delta documented: 1.0452 → 0.8923 (−0.1529, −14.6%)
- [x] Anchor FAILED result recorded: 0.8923 vs 0.8833 upper tolerance, +0.0090 outside band
- [x] Decision = DEFER documented with root-cause note
- [x] Residual +0.0290 gap handed to 04-03 as #1 backlog item with L1/L2 levers
- [x] No other harness tests regressed (5/5 pass)

## Self-Check Result: PASSED
