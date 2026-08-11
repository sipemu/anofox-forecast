---
phase: 04-prioritized-improvement-backlog-top-value-fixes
plan: "02"
subsystem: bench-harness / accuracy-investigation
status: checkpoint
tags: [accuracy, autoets, mase, m3-monthly, period-fix]
key-files:
  modified:
    - crates/anofox-bench-harness/tests/accuracy.rs
  created: []
decisions:
  - "L3/L4 root-cause confirmed: AutoETS::new() without period=12 suppressed all seasonal candidates; fix is harness-only (AutoETS::with_period(period))"
  - "MASE improved from 1.0452 to 0.8923 (+13.7% improvement), but anchor FAILED: 0.8923 outside ±0.02 of reference 0.8633 [band: 0.8433–0.8833]"
  - "accuracy.json NOT committed — defer-lock discipline preserved; awaiting checkpoint:decision"
metrics:
  duration_minutes: 8
  completed: "2026-08-11"
  tasks_completed: 1
  tasks_total: 3
  commits: 1
estimate:
  tokens: 55000
actuals:
  tokens: 14000
  tasks: 1
  commits: 1
---

# Phase 04 Plan 02: M3-Monthly AutoETS MASE Investigation Summary

One-liner: L3/L4 harness root-cause confirmed — period=12 fix cuts MASE from 1.0452 to 0.8923 but anchor FAILED (0.0290 outside ±0.02 tolerance); lock/defer decision pending.

## Measured MASE

| | MASE | Source |
|--|------|--------|
| **BEFORE** (period=1 / no seasonal candidates) | 1.0452 | STATE.md baseline |
| **AFTER** (period=12 / seasonal candidates enabled) | 0.8923 | Anchor run 2026-08-11 |
| **Reference** (statsforecast 2.0.3) | 0.8633 | statsforecast_reference.json |
| **Tolerance band** | [0.8433, 0.8833] | ±0.02 of reference |

**Delta:** −0.1529 (−14.6% MASE reduction from the fix)
**Gap to reference:** 0.8923 − 0.8633 = +0.0290 (still 0.0090 outside the upper tolerance of 0.8833)
**Anchor result: FAILED**

```
MEASURED_MONTHLY_MASE=0.8923  ANCHOR=FAILED
```

The period=12 fix is real and large: it closes 84% of the original gap (0.1529 out of 0.1819).
However, the remaining 16% gap keeps the anchor outside the ±0.02 tolerance band.

## Task 1 — L3/L4 Root-Cause Check (COMPLETED)

### Root-Cause Confirmed

The research hypothesis (L3/L4) was correct: `AutoETS::new()` in the per-frequency harness loop
used `seasonal_period.unwrap_or(1)`, which defaults to period=1 and forces `has_seasonal = false`.
All M3-monthly seasonal ETS candidates were excluded from the search.

**Fix site:** `crates/anofox-bench-harness/tests/accuracy.rs` line 228

**Before:**
```rust
let mut autoets = AutoETS::new();
```

**After:**
```rust
let mut autoets = AutoETS::with_period(period);
```

Where `period` is already computed from `period_for_freq(freq)` at line 137:
- `"monthly"` → 12
- `"quarterly"` → 4
- `"yearly"` → 1

### Verification

All 5 harness tests pass without the dataset env var set (anchor skips cleanly):

```
test accur08_anchor_m3_monthly_autoets ... ok   (skips — ANOFOX_DATASET_DIR not set)
test tracer_m3_monthly_autoets_one_series ... ok
test emit_accuracy_baseline_if_write_flag_set ... ok
test per_frequency_stratification ... ok
test msis_coverage_present_monthly ... ok
test result: ok. 5 passed; 0 failed; 0 ignored
```

With `ANOFOX_DATASET_DIR` set:
```
ACCUR-08 anchor FAILED: AutoETS M3-monthly MASE=0.8923 is outside ±0.02 of reference 0.8633.
```

### Commit

`b6a75cd` — `fix(harness): pass seasonal period to AutoETS in M3 monthly accuracy run`

## Checkpoint Pending

**Task 2 (checkpoint:decision):** Lock `accuracy.json` or keep it deferred/UNLOCKED?

See structured checkpoint below. Do NOT commit accuracy.json until the decision is made.

## Remaining Levers (if defer is selected and a future attempt is wanted)

From the research lever table, two additional low-risk levers could be tried in a future
dedicated accuracy effort:

| # | Lever | Location | Expected Gap Closure |
|---|-------|----------|---------------------|
| L1 | Lower seasonal minimum from 3×period to 2×period | `auto_ets.rs:427` | MEDIUM — brings 24–35 obs series into seasonal pool |
| L2 | Relax or remove F-ratio seasonal gate | `auto_ets.rs:430-471` | MEDIUM — some series suppressed by gate may fit better seasonally |

These are NOT applied here per the time-box and STOP criterion: the anchor still fails
after L4, so L1 and L2 would require a dedicated effort with more time.

## Deviations from Plan

None — plan executed exactly as written through Task 1. Checkpoint at Task 2 as expected.

## Known Stubs

None.

## Threat Surface Scan

No new security-relevant surface introduced. The fix is harness-only (test code in
`crates/anofox-bench-harness/tests/accuracy.rs`). No `src/` changes. T-04-03
(accuracy.json tampering) remains active: `accuracy.json` is NOT written because
`anchor_passed=false` and the emit function's internal assert would panic.

## Self-Check

- [x] Modified file exists: `crates/anofox-bench-harness/tests/accuracy.rs`
- [x] Commit exists: `b6a75cd`
- [x] Measured MASE recorded: 0.8923 (ANCHOR=FAILED)
- [x] `/tmp/04-02-measured.txt` written
- [x] `accuracy.json` NOT committed (defer-lock discipline preserved)
- [x] No other harness tests regressed
