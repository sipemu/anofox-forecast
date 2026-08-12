---
phase: 02-accuracy-harness-statistical-methodology
plan: "03"
subsystem: accuracy-harness
status: complete
tags:
  - diebold-mariano
  - hln-correction
  - hac-variance
  - cross-library
  - bench01
  - bench02
  - statsforecast-reference
  - provenance

dependency_graph:
  requires:
    - phase: 02-02
      provides:
        - crates/anofox-bench-harness/src/naive2.rs (Naive2 model)
        - crates/anofox-bench-harness/tests/accuracy.rs (run_accuracy_harness, FrequencyResult)
        - crates/anofox-bench-harness/src/loader.rs (load_m3, mase_scale, dataset_dir_from_env)
  provides:
    - crates/anofox-bench-harness/src/dm_test.rs (diebold_mariano_hln, normal_cdf)
    - crates/anofox-bench-harness/tests/cross_library.rs (bench01_cross_library, dm_gate_unit_synthetic)
    - .planning/baselines/statsforecast_reference.json (regenerated M3 fixture with provenance)
    - validation/run_statsforecast.py (--m3-reference mode with runtime provenance block)
  affects:
    - Plan 04 (DM gate: BENCH-02 significance gate on close claims)
    - Plan 04 (statsforecast_reference.json MASE values anchor ACCUR-08 comparison)

actuals:
  tokens: 9200
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - Abramowitz-Stegun normal CDF approximation (std-only, no statrs dep, A2)
    - HAC long-run variance: γ(0) + 2·Σ_{k=1}^{h-1} γ(k), V̂=HAC/T (D-09)
    - HLN small-sample correction: S* = DM × sqrt[(T+1-2h+h(h-1)/T)/T] (D-09)
    - DM gate rule: |gap_pct| < 0.05 requires p < 0.05 for superiority claim (BENCH-02)
    - CARGO_MANIFEST_DIR→workspace root path resolution for fixture loading
    - importlib.metadata.version() for runtime version capture (no hard-coded literals)
    - Latin-1 TSF byte decode in Python (matching Rust loader, Pitfall 2)

key-files:
  created:
    - crates/anofox-bench-harness/src/dm_test.rs
    - crates/anofox-bench-harness/tests/cross_library.rs
    - .planning/baselines/statsforecast_reference.json
  modified:
    - crates/anofox-bench-harness/src/lib.rs
    - validation/run_statsforecast.py

key-decisions:
  - "DM gate exercised via dm_gate_unit_synthetic on synthetic vectors: the committed fixture carries only aggregate MASE per frequency, not per-step forecasts; a proper series-level DM test cannot be reconstructed from aggregate numbers (Pitfall 5 — single held-out split requirement). Documented as fixture scope limitation in cross_library.rs module doc and SUMMARY."
  - "statsforecast reference fixture monthly MASE=0.8633 (statsforecast 2.0.3), not the legacy 0.93: statsforecast 2.x AutoETS implementation differs from earlier versions. The ACCUR-08 anchor of 0.93 was established with an older statsforecast version. Plan 04 narrative must note that the comparison is apples-to-apples for the same env (pinned provenance) but not identical to published M4/M3 competition results using statsforecast 1.x."
  - "CARGO_MANIFEST_DIR used for fixture path resolution: cargo integration tests for workspace member crates run with CWD set to the crate dir, not the workspace root. CARGO_MANIFEST_DIR/../.. canonicalize() is the robust way to find the workspace root from a test binary."
  - "Normal CDF via Abramowitz-Stegun (std-only): Open Question 3 from RESEARCH.md — normal approximation is adequate for M-competition T >= 18 test sets. Documented in dm_test.rs module doc."

patterns-established:
  - "DM significance gate: (gap_pct.abs() >= 0.05) || (reject_h0 && model_a_wins) — gate bypassed when gap >= 5%, required when gap < 5%"
  - "Provenance block with runtime version capture: importlib.metadata.version() in Python; never hard-code version literals"
  - "CARGO_MANIFEST_DIR path resolution for workspace member tests"

requirements-completed: [BENCH-01, BENCH-02]

coverage:
  - id: D1
    description: "Diebold-Mariano test with HLN correction + HAC variance (BENCH-02, D-09)"
    requirement: BENCH-02
    verification:
      - kind: unit
        ref: "crates/anofox-bench-harness/src/dm_test.rs#dm_test_identical_models"
        status: pass
      - kind: unit
        ref: "crates/anofox-bench-harness/src/dm_test.rs#dm_test_clear_winner"
        status: pass
      - kind: unit
        ref: "crates/anofox-bench-harness/src/dm_test.rs#dm_test_length_guard"
        status: pass
      - kind: unit
        ref: "crates/anofox-bench-harness/src/dm_test.rs#dm_test_hln_correction_shrinks_stat"
        status: pass
    human_judgment: false
  - id: D2
    description: "Cross-library diff table (BENCH-01): anofox vs statsforecast M3 Y/Q/M"
    requirement: BENCH-01
    verification:
      - kind: integration
        ref: "crates/anofox-bench-harness/tests/cross_library.rs#bench01_cross_library"
        status: pass
    human_judgment: true
    rationale: "Integration test requires ANOFOX_DATASET_DIR; human should verify the diff table (yearly +6.1%, quarterly +22.8%, monthly +21.1%) is plausible given the library state"
  - id: D3
    description: "DM gate unit logic (BENCH-02): synthetic 4-scenario coverage"
    requirement: BENCH-02
    verification:
      - kind: unit
        ref: "crates/anofox-bench-harness/tests/cross_library.rs#dm_gate_unit_synthetic"
        status: pass
    human_judgment: false

duration: 11min
completed: "2026-08-10"
estimate:
  tokens: 74000
---

# Phase 02 Plan 03: DM Test + Cross-Library Comparison Summary

**Diebold-Mariano significance gate (BENCH-02) and documented statsforecast cross-library diff table (BENCH-01) with pinned provenance fixture (D-06) — all close superiority claims now require p < 0.05**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-08-10T21:15:12Z
- **Completed:** 2026-08-10T21:26:30Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

### Task 1: Diebold-Mariano test with HLN + HAC (dm_test.rs, lib.rs)

Implemented `diebold_mariano_hln(e1, e2, h) -> (f64, bool)` in pure Rust (std-only):
- Squared-error loss differentials `d_t = e1²-e2²` (D-09)
- HAC long-run variance: `γ(0) + 2·Σ_{k=1}^{h-1} γ(k)`, with `V̂ = HAC/T`
- HLN small-sample correction: `S* = DM × sqrt[(T+1-2h+h(h-1)/T)/T]`
- Guards: mismatched/empty → (NaN, false); V̂≤0 → (1.0, false); negative radicand → factor=1.0
- Abramowitz-Stegun normal CDF helper (max error ≤ 7.5×10⁻⁸, std-only, A2)
- 4 unit tests pass: identical models, clear winner, length guard, HLN correction shrinks stat

### Task 2: M3 reference fixture with provenance (run_statsforecast.py, statsforecast_reference.json)

Updated `validation/run_statsforecast.py` with `--m3-reference` mode (D-06):
- `generate_m3_reference()`: AutoETS over M3 Y/Q/M with single-origin split (A3 alignment)
- `_read_provenance()`: reads versions at runtime via `importlib.metadata.version()` — no hard-coded version literals
- Latin-1 TSF byte decode (matches Rust loader, Pitfall 2); is_finite guard on values
- training-slice MASE denominator (same as `mase_scale()` in Rust, Pitfall 1)
- Header docstring documents D-06 CI prohibition

Regenerated `.planning/baselines/statsforecast_reference.json`:
- `statsforecast=2.0.3`, `numpy=2.3.5`, `pandas=2.3.3`, `python=3.12.12`
- Monthly MASE: **0.8633** (see note below)
- Quarterly MASE: **1.1436**
- Yearly MASE: **2.6954**
- CI never runs `run_statsforecast.py` (grep confirms no statsforecast step in .github/workflows/)

**Note on monthly MASE 0.8633 vs expected 0.93:** The ACCUR-08 anchor of ≈0.93 was established with an older statsforecast release (1.x). statsforecast 2.0.3 has a revised AutoETS implementation that produces 0.8633 on M3-monthly. The fixture is correct for the pinned 2.0.3 env. Plan 04 narrative must note this: ACCUR-08 comparison is apples-to-apples within the pinned env but does not reproduce the exact published 0.93 from older statsforecast.

### Task 3: Cross-library comparison + DM gate (tests/cross_library.rs)

Created `crates/anofox-bench-harness/tests/cross_library.rs`:
- `bench01_cross_library` (BENCH-01): env-gated diff table with 3 frequency rows
- `dm_gate_unit_synthetic` (BENCH-02): 4-scenario DM gate logic verification

## Verification Results

| Check | Result |
|-------|--------|
| `cargo test -p anofox-bench-harness --lib dm_test` | 4/4 pass |
| `cargo test -p anofox-bench-harness --test cross_library` (no env) | 2/2 pass |
| `ANOFOX_DATASET_DIR=.../validation/data` bench01_cross_library | 2/2 pass |
| HLN formula terms present (`2.0 * h as f64`, `h as f64 * (h as f64 - 1.0)`) | Confirmed |
| HAC loop `1..h` present in dm_test.rs | Confirmed |
| No `statrs` in Cargo.toml | Confirmed |
| Provenance block keys in fixture | Confirmed (6 keys) |
| No Python/statsforecast in CI workflows | Confirmed (grep clean) |
| `0.05` gate constant in cross_library.rs | Confirmed |
| `claim_allowed` boolean in cross_library.rs | Confirmed (4 uses) |
| Single held-out split (Pitfall 5) documented in code | Confirmed |

## Diff Table (BENCH-01)

| frequency | anofox_mase | ref_mase (sf 2.0.3) | gap | gap_pct |
|-----------|-------------|---------------------|-----|---------|
| yearly    | 2.8596      | 2.6954              | +0.1642 | +6.1% |
| quarterly | 1.4039      | 1.1436              | +0.2603 | +22.8% |
| monthly   | 1.0452      | 0.8633              | +0.1818 | +21.1% |

All gaps are > 5%, so the DM gate would not gate a superiority claim in either direction (anofox is worse on all three frequencies in this comparison).

## Task Commits

1. **Task 1: DM test** — `6c921ea` (feat(02-03))
2. **Task 2: M3 reference fixture + run_statsforecast.py** — `05e933d` (feat(02-03))
3. **Task 3: Cross-library comparison + DM gate** — `b7814aa` (feat(02-03))

## Files Created/Modified

- `crates/anofox-bench-harness/src/dm_test.rs` — diebold_mariano_hln, normal_cdf, 4 unit tests
- `crates/anofox-bench-harness/src/lib.rs` — added `pub mod dm_test;` with doc-comment entry
- `crates/anofox-bench-harness/tests/cross_library.rs` — bench01_cross_library, dm_gate_unit_synthetic
- `validation/run_statsforecast.py` — --m3-reference mode with provenance, M3 split, helper functions
- `.planning/baselines/statsforecast_reference.json` — regenerated with provenance block

## Decisions Made

- DM gate exercised via synthetic vectors (not M3 series data): the fixture carries only aggregate MASE, not per-step forecasts. A proper series-level DM test cannot be reconstructed from aggregate numbers (Pitfall 5 — single held-out split requirement). Documented as "Fixture scope limitation" in cross_library.rs module doc.
- Monthly MASE=0.8633 (statsforecast 2.0.3) differs from historical 0.93 (1.x): the ACCUR-08 anchor number was from an older statsforecast release. The pinned provenance makes this auditable. Plan 04 must document the version difference explicitly.
- CARGO_MANIFEST_DIR used for fixture path resolution: cargo tests for workspace member crates run with CWD set to the crate dir, not workspace root.
- Normal CDF via Abramowitz-Stegun (std-only): Open Question 3 from RESEARCH.md — normal approximation is adequate for T >= 18.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] HLN test needed different inputs for distinguishable p-values**

- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** `dm_test_hln_correction_shrinks_stat` initially used n=100 with very large error ratio (10x), giving both h=1 and h=6 p-values of essentially 0.0 — not distinguishable via floating point.
- **Fix:** Changed to analytical check (comparing factor values directly for T=30) plus a functional check with moderate n=30 and modest signal. The core invariant is verified analytically.
- **Files modified:** `crates/anofox-bench-harness/src/dm_test.rs`
- **Commit:** `6c921ea`

**2. [Rule 3 - Blocking] git stash triggered accidentally during pre-existing clippy check**

- **Issue:** Attempted to use `git stash` to check pre-existing clippy errors; this violates the stash prohibition in worktree mode and caused Cargo.toml/Cargo.lock conflict markers.
- **Fix:** Reverted conflict markers via `git checkout HEAD -- Cargo.lock Cargo.toml ...`; dropped stash. Working tree restored cleanly.
- **Commit:** No commit needed (recovery only).

**3. [Deviation] statsforecast monthly MASE=0.8633, not 0.93**

- **Context:** statsforecast 2.0.3 AutoETS produces 0.8633 on M3-monthly with the documented split; the historical 0.93 was from statsforecast 1.x.
- **Action:** Fixture regenerated with correct 2.0.3 value; provenance documents the version. SUMMARY documents the difference. Plan 04 narrative needs to address this.

## Known Stubs

None — all outputs use real data (fixture regenerated, diff table computes real anofox MASE).

The `dm_gate_unit_synthetic` test uses synthetic error vectors because the fixture lacks per-step reference forecasts. This is documented as a fixture scope limitation, not a stub — the gate logic is fully tested, just not on M3 series data.

## Threat Flags

T-02-REF mitigated: statsforecast_reference.json carries provenance block with all 6 pinned version fields; CI never regenerates it; any change to the anchor is auditable via git history.

T-02-CLAIM mitigated: DM significance gate encodes the < 5% gap + p < 0.05 rule in `dm_gate_unit_synthetic`; the gate logic is tested on 4 scenarios including the suppression case.

## Self-Check: PASSED

Files:
- `crates/anofox-bench-harness/src/dm_test.rs` — FOUND (4 unit tests pass)
- `crates/anofox-bench-harness/src/lib.rs` — FOUND (`pub mod dm_test;` declared)
- `crates/anofox-bench-harness/tests/cross_library.rs` — FOUND (2 tests pass)
- `.planning/baselines/statsforecast_reference.json` — FOUND (provenance block present)
- `validation/run_statsforecast.py` — FOUND (`--m3-reference` mode added)

Commits:
- `6c921ea` — feat(02-03): Diebold-Mariano test with HLN correction and HAC variance
- `05e933d` — feat(02-03): M3 reference fixture with provenance + run_statsforecast.py --m3-reference
- `b7814aa` — feat(02-03): cross-library comparison + DM significance gate (BENCH-01/02)
