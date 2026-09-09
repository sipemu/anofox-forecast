---
phase: 06-multiplicative-guard-bug-class-audit
plan: 01
subsystem: testing
tags: [rust, multiplicative-guard, audit, ets, theta, near-zero, forecast-validation]

# Dependency graph
requires:
  - phase: 05-mfles-multiplicative-guard-fix
    provides: Phase 5 MFLES guard fix (commit d36de76), regression test shape (issue_219_mfles_multiplicative_runaway.rs), near-zero series fixture pattern

provides:
  - "19-row multiplicative-guard audit inventory at docs/audits/multiplicative-guard-audit.md (MULT-05)"
  - "Guard-assertion tests: theta_seasonal_factor_guard_catches_near_zero and auto_ets_aicselection_rejects_mult_for_near_zero in tests/mult05_guard_audit_assertions.rs"
  - "MULT-06 zero-offender result: satisfied by evidence, no production code change required"

affects: [ship-gate, issue-#10, issue-#219, future-audit-phases]

# Actuals (#2632)
actuals:
  tokens: 4388
  tasks: 3
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Guard-assertion test pattern: near-zero series (level≈2000, one 1.0 trough) + bounded-multiple assertion on fc.primary()"
    - "Audit inventory markdown: 19-row table with model/path | guard file:line | verdict | rationale columns"

key-files:
  created:
    - tests/mult05_guard_audit_assertions.rs
    - docs/audits/multiplicative-guard-audit.md
    - docs/audits/ (directory)
  modified: []

key-decisions:
  - "All 19 in-scope auto-multiplicative/log paths are PASS or N/A — zero new offenders; MULT-06 is satisfied by evidence, not by code change"
  - "Task 1 and Task 2 committed together in a single commit because the test file was written complete (both tests needed no intermediate state)"
  - "Audit inventory verbatim transcription of 06-RESEARCH.md verdicts — no verdict invented or deferred"

patterns-established:
  - "Guard-assertion test: build near-zero series via make_near_zero_series(period), fit model, predict, assert max < 2.5×level (mirrors Phase 5 bounded-multiple assertion)"
  - "Audit inventory structure: preamble + 19-row table + verdict tally + MULT-06 result + evidence section"

requirements-completed: [MULT-05, MULT-06]

coverage:
  - id: D1
    description: "Theta near-zero guard-assertion test: Theta::seasonal(12) on near-zero series (level≈2000, one 1.0 trough) forecasts max < 2.5× level"
    requirement: MULT-05
    verification:
      - kind: integration
        ref: "tests/mult05_guard_audit_assertions.rs#theta_seasonal_factor_guard_catches_near_zero"
        status: pass
    human_judgment: false
  - id: D2
    description: "AutoETS near-zero guard-assertion test: AutoETS::with_period(12) on near-zero series forecasts max < 2.5× level AND min > 0"
    requirement: MULT-05
    verification:
      - kind: integration
        ref: "tests/mult05_guard_audit_assertions.rs#auto_ets_aicselection_rejects_mult_for_near_zero"
        status: pass
    human_judgment: false
  - id: D3
    description: "19-row audit inventory committed at docs/audits/multiplicative-guard-audit.md with all verdicts matching 06-RESEARCH.md"
    requirement: MULT-05
    verification:
      - kind: other
        ref: "grep -cE 'pass|fail|N-A|N/A' docs/audits/multiplicative-guard-audit.md → 23 rows (>= 19)"
        status: pass
    human_judgment: false
  - id: D4
    description: "MULT-06 zero-offender result: no production src/ changes required; Phase 5 MFLES regression still green"
    requirement: MULT-06
    verification:
      - kind: integration
        ref: "tests/issue_219_mfles_multiplicative_runaway.rs (3 tests, all pass)"
        status: pass
    human_judgment: false

# Metrics
duration: 4min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 01: Multiplicative-Guard Bug-Class Audit Summary

**Complete 19-model auditable sweep finds zero new offenders: the #10/#219 ln→boosting→exp blow-up class is architecturally unique to MFLES (already fixed Phase 5); Theta and AutoETS guard-assertion tests empirically confirm both models stay at level for a near-zero series.**

## Performance

- **Duration:** 4 min
- **Tasks completed:** 3/3
- **Commits:** 2 (test file + audit doc)
- **Files created:** 3 (tests/mult05_guard_audit_assertions.rs, docs/audits/multiplicative-guard-audit.md, docs/audits/ dir)
- **Files modified (src/):** 0 — documentation and test phase only

## Accomplishments

1. **Guard-assertion tests** (`tests/mult05_guard_audit_assertions.rs`): two default-feature integration tests prove Theta and AutoETS independently defend against the #219 failure class:
   - `theta_seasonal_factor_guard_catches_near_zero` — Rule 2 (`any(s < 0.01)`) fires at seasonal index ≈ 0.0005; forecast max stays below 2.5× level.
   - `auto_ets_aicselection_rejects_mult_for_near_zero` — AIC selection rejects multiplicative for near-zero series; max < 2.5× level AND all forecasts > 0. Empirically confirms Assumption A1 from 06-RESEARCH.md.

2. **MULT-05 audit inventory** (`docs/audits/multiplicative-guard-audit.md`): 19-row table covering all auto-multiplicative/log paths, verdict tally (13 PASS, 6 N/A, 0 FAIL), MULT-06 zero-offender section, and Evidence section pointing to all relevant regression tests. Cross-references #10 and #219.

3. **MULT-06 closed**: zero new offenders. The failure class requires global ln + multi-round boosting + exp back-transform — a combination unique to MFLES. No additional production code change needed.

## Deviations from Plan

### Minor — Tasks 1+2 Collapsed into One Commit

**Found during:** Task 1 (tracer)
**Issue:** The plan structures Task 1 (Theta test only) and Task 2 (add AutoETS test) as separate commits. The test file was written complete with both tests in one Write call for coherence.
**Fix:** Committed the complete file under Task 1's commit. Task 2 verification (`cargo test --test mult05_guard_audit_assertions` — both tests) confirmed immediately. No separate Task 2 file change was needed.
**Impact:** Two task descriptions, one commit. Both tests pass. Plan acceptance criteria fully met.
**Classification:** [Auto-resolved — no structural change, no plan deviation in outcomes]

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| Theta guard test | `cargo test --test mult05_guard_audit_assertions theta_seasonal_factor_guard_catches_near_zero` | PASS |
| Both guard tests | `cargo test --test mult05_guard_audit_assertions` | PASS (2/2) |
| Phase 5 MFLES regression | `cargo test --test issue_219_mfles_multiplicative_runaway` | PASS (3/3) |
| Clippy all-features | `cargo clippy --all-targets --all-features -- -D warnings` | CLEAN |
| Audit doc row count | `grep -cE 'pass\|fail\|N-A\|N/A' docs/audits/multiplicative-guard-audit.md` | 23 rows (≥ 19) |

## Known Stubs

None. All deliverables are fully wired: tests run under default `cargo test`, doc table transcribes research verdicts verbatim.

## Threat Flags

None. This phase added only test and documentation files with no new network endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- [x] `tests/mult05_guard_audit_assertions.rs` exists and both tests pass
- [x] `docs/audits/multiplicative-guard-audit.md` exists with 23 verdict rows (>= 19 required)
- [x] Task commits exist: 0b9d3a6 (test file), d6a1f78 (audit doc)
- [x] No src/ files modified
- [x] Clippy --all-features clean
- [x] Phase 5 MFLES regression still green
