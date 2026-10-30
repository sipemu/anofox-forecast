---
phase: quick-261030-go2
plan: 01
subsystem: optimization
tags: [basin, nelder-mead, lbfgs, optimizer, benchmark, wasm, github-234]

requires: []
provides:
  - "basin 3.1.0 evaluation backend behind utils::optimization::{nelder_mead, lbfgs_optimize}, gated by feature `basin` + cfg `anofox_basin_optimizer`"
  - "optimizer_backend() doc-hidden fn reporting the active backend"
  - "examples/optimizer_parity_probe.rs repeatable parity/timing harness"
  - "docs/audits/basin-optimizer-parity.md measured parity report for GH #234 step 1"
affects: [GH-234-step-2, GH-234-step-3, utils::optimization, models::arima, models::exponential, models::theta, models::intermittent, models::garch, models::tbats]

actuals:
  tokens: 19120
  tasks: 3
  commits: 3
  plan_head_before: b7234f2e11e90494f959033951ed247611f04c89
  plan_head_after: 2a4ab80

tech-stack:
  added: ["basin 3.1.0 (optional, exact-pinned, default-features=false)"]
  patterns: ["cfg-gated optimizer backend routing (feature + compiler cfg, not feature alone)", "legacy-fallback-on-invalid-config to avoid basin's assert! panics"]

key-files:
  created:
    - src/utils/optimization/basin_backend.rs
    - examples/optimizer_parity_probe.rs
    - docs/audits/basin-optimizer-parity.md
  modified:
    - Cargo.toml
    - Cargo.lock
    - src/utils/optimization.rs

key-decisions:
  - "Routing requires feature `basin` AND compiler cfg `anofox_basin_optimizer`, not the feature alone — avoids silently routing `cargo test --all-features` (what CI runs) through an unvalidated optimizer and avoids downstream Cargo feature-unification surprises"
  - "basin pinned exact (=3.1.0), optional, default-features=false, no basin features enabled — matches the threat register's T-go2-SC mitigation"
  - "Bounded L-BFGS-B uses basin's native box-constraint handling instead of porting legacy's logit/exp reparametrization — the single largest semantic difference between backends, documented in the parity report rather than hidden"
  - "Any basin config outside its admissible coefficient ranges, or a basin solver Err, falls back to the legacy optimizer instead of propagating a panic (threat T-go2-01)"

patterns-established:
  - "basin_backend module convention: Objective/GradientObjective adapters borrow the caller's closure (not own it), so the same closure can still be used for the legacy fallback after a failed basin run"

requirements-completed: [GH-234-STEP-1]

coverage:
  - id: D1
    description: "basin feature + pinned dependency + cfg-gated routing for nelder_mead, with the default/`--all-features`-without-cfg build unchanged"
    requirement: "GH-234-STEP-1"
    verification:
      - kind: unit
        ref: "utils::optimization::basin_backend::tests (15 tests) + utils::optimization::tests::optimizer_backend_is_legacy_by_default / optimizer_backend_is_basin_when_routed"
        status: pass
      - kind: integration
        ref: "RUSTFLAGS=\"--cfg anofox_basin_optimizer\" cargo test --features basin --lib -- utils::optimization models::exponential::ses"
        status: pass
    human_judgment: false
  - id: D2
    description: "basin L-BFGS/L-BFGS-B backend with bound-aware finite-difference gradients, routed through the same gate; all CI-equivalent gates (clippy x3, rustdoc, fmt, wasm32 x3) green"
    requirement: "GH-234-STEP-1"
    verification:
      - kind: unit
        ref: "utils::optimization::basin_backend::lbfgs_tests (8 tests)"
        status: pass
      - kind: integration
        ref: "RUSTFLAGS=\"--cfg anofox_basin_optimizer\" cargo test --features basin --lib -- models::arima::exact_likelihood"
        status: pass
    human_judgment: false
  - id: D3
    description: "Parity probe + full-suite before/after measurement recorded in docs/audits/basin-optimizer-parity.md"
    requirement: "GH-234-STEP-1"
    verification:
      - kind: other
        ref: "cargo test --all-features --no-fail-fast (legacy) vs RUSTFLAGS=\"--cfg anofox_basin_optimizer\" cargo test --all-features --no-fail-fast (basin): 3791 ok / 0 FAILED / 26 ignored, identical on both sides"
        status: pass
      - kind: other
        ref: "examples/optimizer_parity_probe.rs run+compare across 12 model families"
        status: pass
    human_judgment: true
    rationale: "The plan's own <verify> block includes a <human-check>: the maintainer should read docs/audits/basin-optimizer-parity.md and confirm the cfg-gated (not feature-only) routing decision is acceptable for the evaluation period."

duration: 70min
completed: 2026-10-30
status: complete
---

# Phase quick-261030-go2 Plan 01: basin-Backed Optimizer Behind a Feature Flag Summary

**Put basin 3.1.0 behind `utils::optimization`'s existing `nelder_mead`/`lbfgs_optimize` signatures as a cfg-gated evaluation backend, and proved it is a near-exact numerical match for the hand-rolled optimizers across all 12 model families measured — the full `cargo test --all-features` suite (3791 tests) comes back byte-identical in outcome under either backend, with the only test-name difference being the by-design backend-label test itself.**

## Performance

- **Duration:** ~70 min
- **Completed:** 2026-10-30
- **Tasks:** 3/3
- **Files modified/created:** 6 (Cargo.toml, Cargo.lock, src/utils/optimization.rs, src/utils/optimization/basin_backend.rs, examples/optimizer_parity_probe.rs, docs/audits/basin-optimizer-parity.md)

## Accomplishments

- Added `basin` as an optional, exact-pinned (`=3.1.0`), `default-features = false` dependency behind a new `basin` cargo feature; the default dependency tree is unchanged (`cargo tree ... | grep -c '^basin '` = 0).
- Implemented `basin_backend::nelder_mead_basin` and `basin_backend::lbfgs_optimize_basin`, matching the legacy functions' public signatures exactly, reproducing legacy's termination criteria (simplex-diameter test, stagnation counter, finite-difference gradient rule) as closely as basin's API allows, and falling back to the legacy optimizer on invalid configuration or a basin solver error.
- Wired routing behind `feature = "basin"` **and** the compiler cfg `anofox_basin_optimizer` together — not the feature alone — specifically so `cargo test --all-features` (what CI runs) never silently exercises the unvalidated optimizer and so Cargo feature unification in a downstream consumer can't change this crate's model numerics.
- Built `examples/optimizer_parity_probe.rs`, a backend-agnostic measurement harness across SES/Holt/HoltWinters/AutoETS/ARIMA/AutoARIMA/Theta-family/Croston/GARCH/TBATS on deterministic synthetic data, and ran it + the full test suite under both backends to produce `docs/audits/basin-optimizer-parity.md`.
- **Headline result:** zero test-outcome changes anywhere in the 3791-test `--all-features` suite under basin routing; probe forecast deltas are at floating-point noise for 11/12 families and `1.36e-2` absolute (`9.96e-5` relative) for AutoETS's multi-candidate grid search — still far below practical significance. Timing is a wash in aggregate (+12% total suite wall time, driven almost entirely by one batch-heavy stress test; -16% on the bulk 3236-test unit binary).

## Task Commits

1. **Task 1: Tracer — basin feature, pinned dep, cfg-routed Nelder-Mead, end-to-end through SES** - `11ea354` (feat)
2. **Task 2: L-BFGS / L-BFGS-B backend with finite-difference gradients, routing, and all CI/WASM gates** - `24d31c6` (feat)
3. **Task 3: Parity probe and with/without measurement, recorded in docs/audits/basin-optimizer-parity.md** - `2a4ab80` (docs)

**Plan metadata commit:** handled by the orchestrator (not committed by this executor, per constraints).

## Deviations from Plan

### Auto-fixed Issues

None — no Rule 1/2/3 auto-fixes were needed; the implementation followed the plan's `<action>` steps directly.

### Documented Deviations (not auto-fixed; recorded per instructions to report measured numbers faithfully)

**1. Rosenbrock `max_iter: 500` does not converge under either backend**
- **Found during:** Task 2, writing the `<behavior>` test `lbfgs_basin_unbounded_rosenbrock`.
- **Issue:** The plan's `<behavior>` spec asserted convergence within `max_iter: 500` from `(-1.2, 1.0)`. Measured directly: legacy needs ~786 iterations, basin ~675; neither converges at 500. This is not a basin-specific shortfall — legacy exhibits the identical non-convergence at that budget.
- **Fix:** Adjusted the test to `max_iter: 5000` (both backends then converge to `(1,1)` within the stated tolerances) and documented the measured iteration counts in the test comment and in `docs/audits/basin-optimizer-parity.md`.
- **Files:** `src/utils/optimization/basin_backend.rs`
- **Commit:** `24d31c6`

**2. `memory_size: 0` is not a usable "falls back to legacy and still finds the minimum" case**
- **Found during:** Task 2, writing the `<behavior>` test for invalid-config fallback.
- **Issue:** `lbfgs_config_valid` correctly routes `memory_size: 0` to `lbfgs_optimize_legacy` to avoid basin's own `assert!` panic (threat T-go2-01) — but the `lbfgs` crate that legacy itself depends on panics (`assertion failed: buffer_size > 0`) on `memory_size: 0` regardless of basin. This is a pre-existing limitation in legacy, unrelated to this task, and out of scope to fix (scope boundary: file/behavior not introduced by this task's changes).
- **Fix:** Used `backtrack_rho: 1.5` instead for the invalid-config fallback test. That case also doesn't let legacy "still find the minimum" — legacy's own line search fails on iteration 1 with a growing step and breaks immediately (measured: `[0.0]`, not converged). The test was rewritten to assert exact parity with legacy's own (poor) result for that input, rather than a minimum-finding claim that is false for legacy's output too.
- **Files:** `src/utils/optimization/basin_backend.rs`
- **Commit:** `24d31c6`
- **Tracked in ledger:** `.planning/WINDOWS.md` entry #3 (kind: deviation, open) for the `memory_size: 0` legacy panic; entry #4 (kind: lint-warning, open) for the pre-existing `smart.rs` clippy failure below.

**3. Pre-existing `cargo clippy --all-targets -- -D warnings` (without `--all-features`) failure, unrelated to this task**
- **Found during:** Task 2/3 gate verification.
- **Issue:** `src/models/smart.rs` has unused imports and dead code (`AutoETS`, `AidAnalyzer`, `DemandDistribution`/`DemandType` imports; `TREND_R2_TRIGGER`, `trend_r_squared`, `is_autoets_favorable`, `make_ts`) that only trigger under default-feature `--all-targets` clippy, not under `--all-features`. Verified via `git stash` that this failure is **identical before and after** every commit in this plan — it exists at HEAD on `feat/basin-optimizer` independent of any basin work.
- **Fix:** None — out of scope per the scope boundary rule (pre-existing, unrelated file). `cargo clippy --all-targets --all-features -- -D warnings` (the gate that matters for the `basin` feature itself) is green.
- **Files:** `src/models/smart.rs` (not modified by this plan)
- **Tracked in ledger:** `.planning/WINDOWS.md` entry #4.

## Measured Gate Results

- **Default build:** `cargo tree -p anofox-forecast -e normal --prefix none | grep -c '^basin '` = 0. `cargo test --lib` (default features): **2913 passed, 0 failed, 1 ignored** (569.75s) — unaffected, since the basin module is entirely `cfg(feature = "basin")`-gated out.
- **`--features basin` unit tests:** `utils::optimization` module: 39 tests passed (12 legacy NM + 15 basin NM + 8 basin L-BFGS + 2 backend-label tests + 2 cfg-gated).
- **Routed (`RUSTFLAGS="--cfg anofox_basin_optimizer"`) unit tests:** `utils::optimization`, `models::exponential::ses`, `models::arima::exact_likelihood` all pass end-to-end through the basin backend.
- **clippy** (default, `--all-features`, `--all-features` + routed): `--all-features` variants green; default-feature `--all-targets` fails on pre-existing `smart.rs` issues (see Deviation 3).
- **`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`:** green.
- **`cargo fmt --all -- --check`:** green.
- **WASM:** `cargo check --target wasm32-unknown-unknown --no-default-features --features basin` (OK), `cargo build --target wasm32-unknown-unknown --features postprocess,js,basin` (OK), routed wasm32 check (OK) — 3/3 green.
- **`cargo audit`:** 1 vulnerability + 6 warnings, all pre-existing and unrelated to `basin`'s own dependency surface (verified: none of `basin`, `rand 0.10.3`, `rand_core 0.10.1`, `rand_chacha 0.10.0`, `rand_distr 0.6.0`, `ppv-lite86`, `zerocopy`, `web-time` appear in the findings). `cargo-deny` not installed locally.
- **MSRV:** `cargo +1.86.0 check --features basin` fails at dependency resolution with Cargo's own `basin@3.1.0 requires rustc 1.87.0` message. `cargo +1.91.0 check --features basin` succeeds. The project's effective floor is already above 1.87 for unrelated pre-existing reasons (`argmin`'s `is_multiple_of` use fails on 1.86; `trueno`'s AVX-512 backend fails on 1.88 stable) — `basin` does not newly constrain the toolchain in practice.
- **Full suite parity:** `cargo test --all-features --no-fail-fast` — legacy: 3791 ok / 0 FAILED / 26 ignored, 18m 7.97s wall. Basin-routed: 3791 ok / 0 FAILED / 26 ignored, 20m 16.32s wall. Zero test-outcome changes (diff of normalized `test <name> ... <status>` lines is exactly the one by-design backend-label test).
- **Parity probe (release build, 12 model families):** all `ok: true` both sides; max absolute forecast delta `1.36e-2` (AutoETS), everything else `<1e-7` or exactly `0`.

Full detail, including the per-field termination-criteria mapping table for both optimizers and the per-binary/per-family timing tables, is in `docs/audits/basin-optimizer-parity.md`.

## Known Stubs

None.

## Threat Flags

None — all three threats in the plan's threat model (T-go2-SC, T-go2-01, T-go2-02, T-go2-03) were mitigated exactly as specified: exact pin + no basin features (T-go2-SC), invalid-config fallback to legacy instead of panicking (T-go2-01), `max_iter` always set explicitly (T-go2-02), and feature+cfg-gated routing with `optimizer_backend()` observability (T-go2-03). `examples/optimizer_parity_probe.rs` (T-go2-04) remains a local dev tool reading no input files.

## Next Steps (not part of this plan)

- GH #234 step 2: evaluate `basin::Bobyqa` for the `[0,1]`-bounded smoothing parameters (ETS/Holt-Winters alpha/beta/phi) currently optimized via Nelder-Mead.
- GH #234 step 3: if the maintainer accepts this parity evidence, change the routing gate from `feature + cfg` to `feature` alone to flip the default.
- Investigate the `global_models_bench` timing regression (+22% under basin) if step 3 is pursued — likely the per-iteration heap allocation in `run_nelder_mead_executor`'s `stop_when` closure; amortized away everywhere else but visible under batch/global AutoETS's fit volume.

## Self-Check: PASSED

All created files and all three task commit hashes verified present on disk / in `HEAD`'s ancestry:
`src/utils/optimization/basin_backend.rs`, `examples/optimizer_parity_probe.rs`,
`docs/audits/basin-optimizer-parity.md`, this SUMMARY; commits `11ea354`, `24d31c6`, `2a4ab80`.
