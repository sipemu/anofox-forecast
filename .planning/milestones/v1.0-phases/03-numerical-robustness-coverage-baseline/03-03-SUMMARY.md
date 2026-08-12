---
phase: 03-numerical-robustness-coverage-baseline
plan: "03"
subsystem: testing
tags: [coverage, ci, llvm-cov, baseline, ratchet, gap-inventory]

requires:
  - phase: 03-01
    provides: edge-case robustness tests (edge_case_robustness.rs) + P1 raw-vec deferrals (global_ets, global_croston, global_theta, var.rs)
  - phase: 03-02
    provides: property-based tests (proptest suite) + zero counterexamples recorded

provides:
  - scripts/update_coverage.sh — reproducible local coverage capture with provenance
  - .planning/baselines/coverage.json — committed line-coverage baseline (91.30%, floor 90.3%)
  - .github/workflows/ci.yml — coverage: job extended with ratchet floor hard-gate
  - .planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md — Phase 4 gap backlog (20 rows: 5 P1 / 9 P2 / 6 P3)

affects:
  - phase: 04 (coverage gap remediation)
  - CI/merge gate (any PR dropping below 90.3% line coverage will fail the coverage: job)

actuals:
  tokens: 5319
  tasks: 4
  commits: 5

tech-stack:
  added:
    - cargo-llvm-cov --fail-under-lines (CI ratchet gate)
    - jq (reads ratchet_floor_percent from committed JSON in CI)
  patterns:
    - Coverage baseline locked as committed JSON artifact (mirrors Phase 2 accuracy.json pattern)
    - CI ratchet: floor read from committed artifact, enforcement via --fail-under-lines in coverage: job
    - Provenance block in baseline JSON (git_sha, rustc_version, cargo_llvm_cov_version, host_cpu, host_os, active_features)

key-files:
  created:
    - scripts/update_coverage.sh
    - .planning/baselines/coverage.json
    - .planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md
  modified:
    - .github/workflows/ci.yml (coverage: job extended with Enforce coverage floor step)
    - tests/edge_case_robustness.rs (cargo fmt long-line assert fix)

key-decisions:
  - "Lock coverage baseline as-measured: 91.30% line coverage, ratchet_floor_percent 90.3%, scope --package anofox-forecast --all-features (lock-as-measured option from checkpoint:decision)"
  - "CI enforcement added to EXISTING coverage: job as a separate step (not folded into lcov step) to preserve Codecov upload and avoid double-instrumentation complexity"
  - "Coverage scope scoped to --package anofox-forecast only, excluding anofox-bench-harness and js crates that would distort the baseline"

patterns-established:
  - "Coverage baseline pattern: update_coverage.sh → coverage.json (with provenance) → CI reads floor → --fail-under-lines gate; mirrors criterion.json / accuracy.json convention"
  - "Gap inventory pattern: llvm-cov text report → P1/P2/P3 priority rows with file+function+missing-invariant — directly consumable by Phase 4 backlog"

requirements-completed: [COVER-01, COVER-02]

coverage:
  - id: D1
    description: "scripts/update_coverage.sh captures line coverage via cargo-llvm-cov and writes .planning/baselines/coverage.json with provenance block"
    requirement: COVER-01
    verification:
      - kind: automated
        ref: "bash scripts/update_coverage.sh && jq -e '.coverage.ratchet_floor_percent and .coverage.lines_percent and .provenance.git_sha' .planning/baselines/coverage.json"
        status: pass
    human_judgment: false
  - id: D2
    description: ".planning/baselines/coverage.json committed with measured floor (91.30% measured, 90.3% floor) and git-tracked"
    requirement: COVER-01
    verification:
      - kind: automated
        ref: "git ls-files --error-unmatch .planning/baselines/coverage.json"
        status: pass
    human_judgment: false
  - id: D3
    description: "ci.yml coverage: job extended with Enforce coverage floor step using --fail-under-lines $ratchet_floor_percent, scoped to --package anofox-forecast --all-features"
    requirement: COVER-01
    verification:
      - kind: automated
        ref: "grep -q 'Enforce coverage floor' .github/workflows/ci.yml && grep -q 'fail-under-lines' .github/workflows/ci.yml && python3 -c \"import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))\""
        status: pass
    human_judgment: false
  - id: D4
    description: "03-GAP-INVENTORY.md with 20 rows (5 P1 / 9 P2 / 6 P3) structured for Phase 4 backlog consumption"
    requirement: COVER-02
    verification:
      - kind: automated
        ref: "grep -q global_ets .planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md && grep -q 'P1' .planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md && grep -qi 'assertion-free' .planning/phases/03-numerical-robustness-coverage-baseline/03-GAP-INVENTORY.md"
        status: pass
    human_judgment: false

duration: ~15min (continuation agent: Tasks 3-4 + SUMMARY; prior agent: Tasks 1-2)
completed: 2026-08-11
status: complete
---

# Phase 03 Plan 03: Coverage Baseline & CI Ratchet Gate Summary

**Line-coverage baseline locked at 91.30% (floor 90.3%) via committed coverage.json and a cargo-llvm-cov --fail-under-lines hard-gate in the existing ci.yml coverage: job, with a 20-row gap inventory (5 P1 / 9 P2 / 6 P3) seeding the Phase 4 backlog.**

## Performance

- **Duration:** ~15 min total (two-agent execution: Task 1-2 agent paused at checkpoint:decision; continuation agent executed Tasks 3-4 + SUMMARY)
- **Tasks:** 4/4 completed
- **Commits:** 5 (scripts + STATE pause + gap inventory + coverage.json baseline + ci.yml extension with fmt fix)

## What Was Built

1. **`scripts/update_coverage.sh`** — Reproducible local baseline capture script mirroring `update_criterion.sh`/`update_wasm_size.sh` convention. Runs `cargo llvm-cov --package anofox-forecast --all-features --json --summary-only`, extracts `lines.percent`/`lines.covered`/`lines.count` via jq, computes `ratchet_floor_percent = lines_percent - 1.0` (python3 float subtract), writes `.planning/baselines/coverage.json` with a provenance block (`git_sha`, `timestamp_iso`, `rustc_version`, `cargo_llvm_cov_version`, `host_cpu`, `host_os`, `active_features`).

2. **`.planning/baselines/coverage.json`** — Committed line-coverage baseline:
   - `lines_percent`: 91.30339709041328%
   - `lines_total`: 77743 / `lines_covered`: 70982
   - `ratchet_floor_percent`: 90.3
   - Scope: `--package anofox-forecast --all-features`
   - Provenance: rustc 1.97.0, cargo-llvm-cov 0.8.4, git sha 5a02aa3b

3. **`.github/workflows/ci.yml` extended** — Added "Enforce coverage floor" step to the existing `coverage:` job between "Generate coverage" and "Upload to Codecov". The step reads `FLOOR=$(jq '.coverage.ratchet_floor_percent' .planning/baselines/coverage.json)` and runs `cargo llvm-cov --package anofox-forecast --all-features --summary-only --fail-under-lines "$FLOOR"`. Scope matches baseline exactly. No new workflow file created. YAML parses without errors.

4. **`03-GAP-INVENTORY.md`** — 20-row gap inventory structured for Phase 4 backlog consumption:
   - P1 (5): 4 raw-vec NaN guard deferrals from 03-01 (global_ets, global_croston, global_theta, var.rs) + GlobalTheta 0% coverage
   - P2 (9): Coverage gaps in seasonality/traits.rs, models/inspect.rs, changepoint/detector.rs, anomaly/quantile.rs, models/smart.rs, laplace/gpd_tails.rs, batch.rs, postprocess/cqr.rs
   - P3 (6): 5 assertion-free integration tests + 1 ignored utility test

## Decision Checkpoint

**Decision:** `lock-as-measured` (human selected at the blocking checkpoint:decision gate before this continuation agent)

| Field | Value |
|-------|-------|
| Option selected | lock-as-measured |
| Measured coverage | 91.30% (91.30339709041328%) |
| Ratchet floor | 90.3% (measured − 1.0%, rounded to 1 decimal) |
| Scope | `--package anofox-forecast --all-features` |
| Rationale | Matches locked CONTEXT decision and Phase 2 accuracy.json precedent; conservative margin protects against slow erosion immediately |
| Reversibility | One-way door: a lower floor requires a deliberate maintainer diff in coverage.json |
| Gap inventory counts | P1: 5 rows / P2: 9 rows / P3: 6 rows (20 total) |

The baseline is committed as a human-reviewed artifact (T-03-08 mitigation: floor change is always a visible diff; T-03-09 mitigation: identical scope in both baseline capture and CI enforcement).

## Accomplishments

- COVER-01: Line-coverage baseline captured and committed. CI now hard-gates below 90.3% via `--fail-under-lines $ratchet_floor_percent` in the existing `coverage:` job.
- COVER-02: Gap inventory (03-GAP-INVENTORY.md) provides 20 prioritized rows with file + function + missing-invariant columns ready for Phase 4 backlog.
- Phase 3's measurement backbone is complete: criterion.json (performance) + coverage.json (line coverage) + gap inventory → Phase 4 has a machine-readable list of highest-value targets.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] cargo fmt pre-commit hook: long-line assert in tests/edge_case_robustness.rs**
- **Found during:** Task 3 (ci.yml commit)
- **Issue:** `cargo fmt --check` failed on `assert!(result.is_err() || result.is_ok(), "unexpected state after VAR fit on n=2")` in `tests/edge_case_robustness.rs` (created in Task 03-01), which cargo fmt reformats to a multi-line form.
- **Fix:** Ran `cargo fmt` to reformat the assert; staged `tests/edge_case_robustness.rs` together with the ci.yml change.
- **Files modified:** `tests/edge_case_robustness.rs` (line 718-721)
- **Commit:** fe75b19

No architectural deviations. All tasks executed as planned.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `coverage.json` exists on disk | FOUND |
| `ci.yml` exists on disk | FOUND |
| `03-GAP-INVENTORY.md` exists on disk | FOUND |
| `03-03-SUMMARY.md` exists on disk | FOUND |
| commit 69882f7 (coverage.json) exists | FOUND |
| commit fe75b19 (ci.yml extension) exists | FOUND |
| commit b8c0e9c (03-GAP-INVENTORY.md) exists | FOUND |
| commit b2348b6 (update_coverage.sh) exists | FOUND |
| `git ls-files --error-unmatch .planning/baselines/coverage.json` | PASS |
| `jq -e '.coverage.ratchet_floor_percent < .coverage.lines_percent'` | PASS |
| `grep -q 'Enforce coverage floor' ci.yml` | PASS |
| YAML parses | PASS |
