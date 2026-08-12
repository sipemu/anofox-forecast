---
phase: 04-prioritized-improvement-backlog-top-value-fixes
plan: "03"
subsystem: backlog
status: complete
tags: [backlog, ranking, accuracy-gap, coverage, wasm-size, memory, robustness, impr-01]

requires:
  - phase: 04-01
    provides: V-01/V-02/V-03 guards landed; G-01 smoke test; coverage.json ratcheted 91.30%→91.44%, floor 90.3%→90.4%
  - phase: 04-02
    provides: accuracy.json DEFERRED; period=12 harness fix (MASE 1.0452→0.8923, −14.6%); anchor FAILED (0.8923 > 0.8833); residual gap +0.0290 handed to backlog

provides:
  - ".planning/baselines/BACKLOG.md — consolidated ranked improvement backlog across all 8 measurement dimensions (IMPR-01)"
  - "ACC-01 ranked #1 OPEN with evidence (MASE=0.8923 vs ref=0.8633) and L1/L2 levers"
  - "V-04 VAR error-variant divergence documented as known issue at Rank 10"
  - "iai.json/criterion.json rows flagged manual-capture-pending with exact capture commands"
  - "Landed This Phase section: V-01/V-02/V-03/G-01/COV-01/ACC-HARNESS all documented with commits"

affects: []

actuals:
  tokens: 9000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Priority = Value × (Effort × Risk) / 5 ranking formula applied across 8 measurement dimensions"
    - "Separate sections for ranked-open, landed, manual-capture-pending — prevents closed items polluting the priority ordering"

key-files:
  created:
    - .planning/baselines/BACKLOG.md
  modified: []

key-decisions:
  - "ACC-01 (AutoETS MASE gap) ranked #1 open item with evidence-backed score (Value=5, Effort=3, Risk=3, Priority=9.0); two untried levers L1/L2 documented with locations and risk notes"
  - "V-04 (VAR error-variant divergence) placed at Rank 10 with Value=2 — guard IS present, only the error type diverges; not refactored per user constraint"
  - "iai.json and criterion.json rows placed in a separate manual-capture-pending section with zero priority score — no numeric delta can be inferred from placeholder zeros"
  - "Landed items (V-01/V-02/V-03/G-01/COV-01/ACC-HARNESS) in a separate Landed This Phase section, not in the ranked-open table"
  - "MEM-01 (AutoETS peak memory) included as Rank 17 open item — dhat.json real evidence (290,440 bytes); Effort=4 because root-cause investigation needed before optimization"
  - "WSZ-01 (WASM size) included as Rank 16 open item — wasm_size.json real evidence (2,838,958 bytes)"

requirements-completed: [IMPR-01]

duration: 25min
completed: 2026-08-11
---

# Phase 4 Plan 03: Consolidated Ranked Improvement Backlog Summary

**IMPR-01 satisfied: .planning/baselines/BACKLOG.md ranks all 17 open items across 8 measurement dimensions using real captured baseline numbers; ACC-01 (AutoETS MASE gap) is #1 open item; iai/criterion flagged manual-capture-pending; all 04-01/04-02 landed items in a separate Landed section.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 2
- **Files created:** 1 (BACKLOG.md)
- **Commits:** 1 (0b16851)

## Accomplishments

1. **Assembled all backlog inputs** from the eight committed baseline files and the two prior SUMMARYs. Confirmed: accuracy.json ABSENT/DEFERRED (anchor FAILED), iai.json/criterion.json PLACEHOLDER (all zeros), coverage.json/wasm_size.json/dhat.json/statsforecast_reference.json REAL. Verified INPUTS_PRESENT for both required files.

2. **Authored .planning/baselines/BACKLOG.md** with:
   - **Ranked Open Items table** — 17 items scored by Value/Effort/Risk/Priority. Real-evidence items rank ahead of placeholder items.
   - **Manual-Capture-Pending section** — iai.json (3 benchmarks) and criterion.json (14 benchmarks) with exact capture commands; no fabricated numeric deltas.
   - **Landed This Phase section** — V-01/V-02/V-03 NaN guards, G-01 GlobalTheta smoke test, COV-01 coverage ratchet (91.30%→91.44%), ACC-HARNESS period=12 fix (MASE 1.0452→0.8923). All documented with commits and before/after evidence.
   - **Known Inconsistency section** — V-04 VAR error-variant divergence (InvalidParameter vs MissingValues) documented with recommended fix paths; explicitly not refactored this phase.
   - **Accuracy Gap Detail (ACC-01)** — full handoff: L1/L2 lever table, entry conditions for locking accuracy.json, and the exact anchor command.
   - **Dimension Coverage Summary** — all 8 dimensions with real/absent/placeholder status.

3. **Committed with `--no-verify`** per project memory (cargo-fmt hook blocks doc commits): `0b16851`.

## Top-Ranked Open Items

| Rank | ID | Title | Priority Score | Evidence |
|------|----|-------|---------------|----------|
| 1 | ACC-01 | Close residual AutoETS M3-monthly MASE gap | 9.0 | MASE=0.8923 vs ref=0.8633 (+0.0290 gap; +0.0090 outside ±0.02 tolerance) |
| 2 | MEM-01 | AutoETS peak memory reduction | 7.2 | dhat.json: auto_ets=290,440 bytes (outlier vs auto_arima=191,268, auto_theta=133,456) |
| 3 | WSZ-01 | WASM binary size reduction | 5.4 | wasm_size.json: 2,838,958 bytes |
| 4 | G-07 | SmartForecaster dispatch branch coverage | 4.8 | 34.4% file coverage; AID thresholds and routing branches untested |
| 5 | G-09 | Batch partial-failure error aggregation | 3.2 | 42.2% file coverage; Vec<Result<_>> Err arm untested |

## Placeholder Baselines (Manual-Capture-Pending)

| Baseline | Items | Capture Requirement |
|----------|-------|---------------------|
| iai.json (instruction counts) | PERF-IAI-01..03 (3 benchmarks) | Valgrind ≥3.20 machine; `cargo bench --package anofox-bench-harness --bench iai_suite` |
| criterion.json (wall-clock) | PERF-CRT-01..03 (14 benchmark entries) | Quiet local machine, CPU governor = performance; `cargo bench --package anofox-bench-harness --bench criterion_suite` |

No numeric performance delta is claimed or inferred from placeholder zeros.

## Landed Items Summary (from 04-01 and 04-02)

| ID | What Landed | Delta | Commit |
|----|-------------|-------|--------|
| V-01 | GlobalETS NaN/Inf guard | Silent propagation risk closed | 3fc82aa |
| V-02 | GlobalCroston NaN/Inf guard | Silent propagation risk closed | 3fc82aa |
| V-03 | GlobalTheta NaN/Inf guard | Silent propagation risk closed | e2d2e73 |
| G-01 | GlobalTheta smoke test (was 0% covered) | 0% → exercised | e2d2e73 |
| COV-01 | Coverage ratchet | 91.30%→91.44% (+0.14 pp), floor 90.3%→90.4% | 84f9863 |
| ACC-HARNESS | AutoETS period=12 harness fix | MASE 1.0452→0.8923 (−14.6%); anchor still FAILED | b6a75cd |

## Deviations from Plan

None — plan executed exactly as written. Task 1 gathered inputs and confirmed INPUTS_PRESENT; Task 2 authored and committed BACKLOG.md with all required content (ranked-open table, manual-capture-pending section, V-04 known-issue, accuracy MASE numbers, landed section, ranking rationale).

## Known Stubs

None — BACKLOG.md is a complete documentation artifact with no placeholders in the ranked rows; the only placeholder-flagged items are iai/criterion entries which are correctly labeled manual-capture-pending.

## Threat Flags

None. BACKLOG.md is a planning artifact (no network endpoints, auth paths, file access patterns, or schema changes). T-04-05 (placeholder rows for iai/criterion) is mitigated: each row is marked manual-capture-pending with the exact capture command, and no numeric delta is inferred from placeholder zeros. T-04-06 (evidence provenance) is mitigated: every ranked row cites its specific baseline file and number.

## Self-Check: PASSED

- [x] `.planning/baselines/BACKLOG.md` exists: FOUND
- [x] Contains "manual-capture-pending": VERIFIED (grep check BACKLOG_OK)
- [x] Contains V-04/VAR: VERIFIED (grep check BACKLOG_OK)
- [x] Contains MASE numbers (1.0452, 0.8633): VERIFIED (grep check BACKLOG_OK)
- [x] Commit `0b16851` exists in git log: VERIFIED (`git log --oneline -1 -- .planning/baselines/BACKLOG.md`)
- [x] iai.json/criterion.json rows carry no fabricated deltas: VERIFIED (all entries in manual-capture-pending section only; ranked-open table contains no iai/criterion numeric values)
- [x] V-04 documented as known issue, not refactored: VERIFIED (Known Inconsistency section present)
- [x] ACC-01 ranked #1 with MASE evidence: VERIFIED (Rank 1, Priority=9.0, evidence line citing 0.8923 vs 0.8633)
- [x] Landed This Phase section present with commits: VERIFIED (6 rows with commits e2d2e73, 3fc82aa, 84f9863, b6a75cd)

---

*Phase: 04-prioritized-improvement-backlog-top-value-fixes*
*Completed: 2026-08-11*
