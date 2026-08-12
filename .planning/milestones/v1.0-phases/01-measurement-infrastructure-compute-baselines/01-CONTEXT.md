# Phase 1: Measurement Infrastructure & Compute Baselines - Context

**Gathered:** 2026-08-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Stand up the **measurement backbone** for the hardening cycle: a committed baseline
store, the harness/tooling that produces those baselines, the CI workflows that read
them, and the compute/memory/WASM-size baselines themselves — captured only after the
PERF-06 dead-code cleanup so the WASM size number is trustworthy.

**In scope (Requirements MEAS-01..04, PERF-01..06):**
- Committed JSON baseline store at `.planning/baselines/` (criterion, iai-callgrind, dhat, WASM size); CI reads, never writes.
- `bench.yml` and `wasm-size.yml` GitHub Actions workflows.
- criterion fit+predict suite across model families (single + batch), captured locally.
- iai-callgrind instruction-count gates on critical hot paths.
- Native peak-memory (dhat) bounds for major families.
- WASM release binary-size tracking with a CI delta threshold.
- PERF-06 dead-code removal in `crates/anofox-forecast-js/` before the size baseline is locked.

**Out of scope (deferred to later phases / this milestone's boundaries):**
- Accuracy harness, datasets, metrics, cross-library comparison → Phase 2 (`accuracy.yml` is Phase 2, `workflow_dispatch`-only).
- Robustness/edge-case/coverage work → Phase 3.
- Improvement backlog & landing fixes → Phase 4.
- Any new code in library `src/` (MEAS-04) — measurement code lives in `benches/`, `tests/`, `scripts/`, or the new harness crate only.

</domain>

<decisions>
## Implementation Decisions

### Baseline Store Schema (MEAS-01)
- **D-01:** One JSON file **per dimension** under `.planning/baselines/` — `criterion.json`, `iai.json`, `dhat.json`, `wasm_size.json` — each self-contained. Rationale: cleanest for per-dimension `update_*.sh` scripts and per-workflow CI reads; avoids merge conflicts from a shared file. — **Reversibility:** reversible.
- **D-02:** Each baseline record carries a **full provenance fingerprint**: git SHA, ISO timestamp, rustc version, host CPU/OS, active feature flags, plus the metric value(s). Rationale: makes "reproduce on a quiet local machine" (MEAS-02 / success-criterion 3) auditable and explains cross-machine drift. — **Reversibility:** costly — schema is read by every compare path across Phases 1–4; adding required fields later means re-capturing all baselines.
- **D-03:** criterion baselines store **median + MAD** (robust to wall-clock noise) and are **informational only** in CI — CI reports drift but never fails on criterion. iai-callgrind is the hard compute gate. Rationale: aligns with the locked constraint that criterion is captured locally, not on GitHub Actions.

### Harness Location & Runner UX (MEAS-02, MEAS-04)
- **D-04:** Add a **new workspace harness crate** (e.g. `crates/anofox-bench-harness` or an `xtask` crate) that owns the baseline serde structs, read/compare logic, and the dhat + iai bins. `benches/` and `scripts/*.sh` call into it. Rationale: gives the fingerprinted schema a single typed owner (prevents drift across 4 dimensions); MEAS-04 forbids `src/` but explicitly allows a harness crate. — **Reversibility:** costly — a new workspace member touches `Cargo.toml`/lockfile and becomes the import root for all tooling.
- **D-05:** Maintainer entrypoint is **`scripts/update_*.sh` per dimension** (`update_criterion.sh`, `update_iai.sh`, `update_dhat.sh`, `update_wasm_size.sh`), each running the relevant capture and writing its JSON — delegating to the harness crate internally. Rationale: matches the literal MEAS-02 / success-criterion-3 wording ("documented `scripts/update_*.sh`") and stays greppable/discoverable.

### Benchmark Suite (PERF-01, PERF-03)
- **D-06:** Add **one new dedicated baseline bench** (e.g. `benches/baseline_suite.rs`) as the single source of truth for committed criterion baselines. The existing 8 benches stay as-is for ad-hoc/exploratory profiling and are NOT baseline-tracked. Rationale: clean separation of "tracked" vs "exploratory"; avoids churning proven bench code. — **Reversibility:** reversible.
- **D-07:** Tracked matrix = **one representative model per family** — AutoARIMA, AutoETS, AutoTheta, Naive, Croston, AutoEnsemble, Laplace — × {single-series fit+predict, batch-100}. Rationale: covers all 7 families named in PERF-01 (fills the current gaps: intermittent, Laplace, explicit batch) without combinatorial blowup.
- **D-08:** Tracked benches pull from **shared seeded fixtures** in the harness crate — deterministic series at fixed lengths (e.g. n=100, n=1000) with a fixed seed. Rationale: makes baselines reproducible and comparable across benches; the existing per-bench ad-hoc generators produce non-comparable timings.
- **D-09:** Native-parallel and WASM/single-thread (no-Rayon) profiles reported in **separate sections** of the criterion output/baseline (PERF-03).

### CI Gate Strictness (PERF-02, PERF-04, PERF-05)
- **D-10:** **iai-callgrind gate**: run only on **pinned stable Rust** (not beta/nightly); fail CI if instruction count rises **> 1%** vs baseline. Beta/nightly still build+test but do not gate instructions. Rationale: instruction counts drift slightly across rustc versions; ±1% absorbs codegen noise while catching real regressions.
- **D-11:** **WASM size gate** (`wasm-size.yml`): fail if the compiled release `.wasm` grows **> 1% relative** to the committed baseline. Rationale: scale-free, stays meaningful as the binary evolves.
- **D-12:** **dhat peak-memory gate** (PERF-04): **hard assert** peak stays under **baseline × 1.15** (15% headroom) for major model families, run as a native test/bin. Rationale: fails on real memory regressions while tolerating allocator variance.

### Claude's Discretion (sensible defaults, user deferred)
- **iai hot-path selection:** the three named paths — AutoETS fit, AutoARIMA fit, batch-100 — as the initial 3–5 critical hot paths (PERF-02). Room to add 1–2 more if research surfaces an obvious hot path.
- **Dead-code cleanup scope (PERF-06):** remove the named unused `inner()` methods + unused imports in `crates/anofox-forecast-js/`, plus a `cargo-machete` sweep on that crate; confirm removal AND that the npm package still builds before locking the WASM size baseline. Must stay backward-compatible with `@sipemu/anofox-forecast`.
- **`bench.yml` CI scope:** runs the **iai-callgrind instruction gate only**; criterion is local-capture per the locked constraint (CI must not capture wall-clock timings).
- **dhat harness form:** native bin/test in the harness crate using dhat's allocator (NOT `wee_alloc` — banned). Family selection = the "major" families (align with the PERF-01 representative set).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap (authoritative scope)
- `.planning/ROADMAP.md` §"Phase 1" — goal, 5 success criteria, requirement mapping (MEAS-01..04, PERF-01..06).
- `.planning/REQUIREMENTS.md` §MEAS, §PERF — full requirement text; §"Out of Scope" (note the anti-metrics and banned integrations).
- `.planning/STATE.md` §"Key Decisions" + §"Critical Constraints to Honor" — locked sequencing (PERF-06 before PERF-05), local-vs-CI capture rule, `accuracy.yml` deferral, `wee_alloc` ban.

### Codebase maps (current, refreshed 2026-08-09)
- `.planning/codebase/STRUCTURE.md` — where benches/scripts/crates live.
- `.planning/codebase/TESTING.md` — existing criterion/proptest/wasm-bindgen-test surface.
- `.planning/codebase/CONCERNS.md` + `ARCHITECTURE.md` — documented anti-patterns (unwrap/expect, skipped `validate_series_complete()`) — mostly Phase 3, but context for the dead-code sweep.

### Existing code to build on / touch
- `benches/` — 8 existing criterion benches (`arima_benchmark.rs`, `ets_benchmark.rs`, `model_comparison.rs`, `comprehensive_benchmark.rs`, `ensemble_benchmark.rs`, `cv_benchmark.rs`, `prediction_benchmark.rs`, `bootstrap_benchmark.rs`) — kept for ad-hoc use, NOT baseline-tracked.
- `Cargo.toml` §`[[bench]]` (lines ~74–108) — bench registration; `criterion = "0.5"` already a dev-dep.
- `crates/anofox-forecast-js/src/` — PERF-06 cleanup target (`inner()` methods, unused imports).
- `.github/workflows/ci.yml` — existing CI; `bench.yml` / `wasm-size.yml` are NEW additions alongside it.
- `scripts/` — currently Python-only; the new `update_*.sh` scripts land here.

No external ADRs — requirements and constraints are fully captured in `.planning/` docs above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing benches contain working fit/predict harness code and series generators (`make_series`, `make_timestamps`) — reuse as reference for the new tracked suite's fixtures, but formalize into shared **seeded** fixtures in the harness crate (D-08).
- `criterion 0.5` is already a dev-dependency and 8 benches are registered `harness = false` — the new baseline bench follows the same registration pattern.

### Established Patterns
- Bench files use `criterion_group!`/`criterion_main!` with `group.sample_size(...)` — the new suite standardizes sample sizes rather than the current ad-hoc mix.
- Model construction via `AutoX::new()` / `::auto()` constructors and the `Forecaster` trait `fit`/`predict` — the tracked matrix drives all 7 families through this uniform interface.

### Integration Points
- New harness crate registers as a workspace member (root `Cargo.toml` `[workspace] members`), joining `crates/anofox-forecast-js`.
- New `bench.yml` / `wasm-size.yml` sit beside `ci.yml`, `deploy-playground.yml`, `npm.yml`.
- `wasm-size.yml` measures the same `wasm-pack` release build that feeds the npm package — must not regress the published build.

### Coverage gaps in existing benches (drive PERF-01 fills)
- No **intermittent** (Croston/IMAPA/TSB/ADIDA), no **Laplace**, no **explicit batch** coverage today; inconsistent `sample_size`; no committed baselines.

</code_context>

<specifics>
## Specific Ideas

- The 7-family representative set is explicit: **AutoARIMA, AutoETS, AutoTheta, Naive, Croston, AutoEnsemble, Laplace** (D-07).
- iai hot paths anchor on the three named in PERF-02: **AutoETS fit, AutoARIMA fit, batch-100**.
- Gate numbers are concrete and locked: iai **±1%** / WASM size **>1% relative** / dhat **×1.15** hard assert.
- Provenance fingerprint fields are enumerated in D-02 — treat as the required baseline schema.

</specifics>

<deferred>
## Deferred Ideas

- Broader within-family bench coverage (multiple intermittent variants, ETS/HW variants) — considered and rejected for now (D-07 chose representative-per-family); revisit only if Phase 4 backlog points at a specific family.
- Vendored-real-series fixtures for benches — deferred; ties bench timing to Phase-2 dataset/loader work. Synthetic seeded fixtures chosen instead (D-08).
- WASM runtime memory profiling — explicitly a v2 requirement (XROB-02); no in-process dhat equivalent for WASM today.

None outside phase scope surfaced beyond the above.

</deferred>

---

*Phase: 1-Measurement Infrastructure & Compute Baselines*
*Context gathered: 2026-08-09*
