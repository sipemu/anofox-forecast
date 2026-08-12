# Phase 2: Accuracy Harness & Statistical Methodology - Context

**Gathered:** 2026-08-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a **statistically correct accuracy harness** that produces trustworthy
per-frequency forecast-accuracy numbers over the vendored competition datasets,
**validated against a published statsforecast reference before any baseline is
locked**, plus a **documented, reproducible cross-library comparison**.

**In scope (Requirements ACCUR-01..08, BENCH-01..02):**
- Dataset loader for the local `validation/data/` corpus (Monash `.tsf` + CSV/JSON),
  reading M3, M4 sample, Tourism, NN5, gated on `ANOFOX_DATASET_DIR` (ACCUR-01).
- Expanding-window (rolling-origin) CV with per-fold `train_end < test_start`
  temporal-integrity assertions; no in-sample / `fitted_values()` numbers in any
  accuracy table (ACCUR-02).
- MASE (correct seasonal denominator + collapse guard), sMAPE, RMSE, MAE, MSIS,
  empirical interval coverage — no silent NaN/Inf in aggregates (ACCUR-03..05).
- Naive2 accuracy reference (autocorrelation-gated seasonal/non-seasonal) (ACCUR-06).
- Per-frequency stratified reporting (never a single cross-frequency aggregate) (ACCUR-07).
- Committed `accuracy.json` baseline validated against published statsforecast
  AutoETS M3-monthly MASE ≈ 0.93 before lock (ACCUR-08).
- Cross-library comparison vs statsforecast on shared datasets/horizons/preprocessing,
  via the existing `run_statsforecast.py` (BENCH-01).
- Diebold–Mariano significance gate on any "beats reference" claim with a < 5% gap (BENCH-02).
- `accuracy.yml` workflow — **`workflow_dispatch`-only, never gates PR merges** (carried from Phase 1 / MEAS-03).

**Out of scope (deferred to later phases / milestone boundaries):**
- Robustness/edge-case/coverage work → Phase 3 (ROBUST-*, COVER-*).
- Improvement backlog & landing fixes → Phase 4 (IMPR-*).
- Committing accuracy numbers for M4/Tourism/NN5 and all frequencies — the loader
  reads them and the schema supports them, but only M3 (Y/Q/M) numbers are locked
  in this first baseline (see D-05).
- New public forecasting models or auto period-detection integration (PROJECT.md Out of Scope).

</domain>

<decisions>
## Implementation Decisions

### Metrics & CV — reuse the library, validate via the reference
- **D-01:** **Reuse** the library's existing metrics in `src/utils/metrics.rs`
  (MASE via `calculate_mase`, sMAPE, RMSE, MAE, MSIS, coverage) rather than
  reimplementing them in the harness. The bug-catching safety net is the ACCUR-08
  validation against the published statsforecast number — if a reused metric were
  wrong, the M3-monthly ≈ 0.93 anchor would not reproduce. — **Reversibility:** reversible.
- **D-02:** **Reuse** the library's expanding-window `src/utils/cross_validate`
  (`CvFoldGenerator`, `Fold { train_start, train_end, test_start, test_end }`);
  the harness adds an explicit per-fold `train_end < test_start` assertion layer on
  top (ACCUR-02) so temporal integrity is enforced by the harness, not assumed of
  the library. — **Reversibility:** reversible.

### MASE denominator-collapse guard (ACCUR-03) — fix in src/, fallback denominator
- **D-03:** **Fix the collapse guard directly in `src/utils/metrics.rs`** (currently
  `calculate_mase` returns `unwrap_or(NaN)` — a silent NaN on a collapsed seasonal
  denominator, exactly what ACCUR-03 forbids). Framed as a genuine **correctness fix
  to existing shipped code with a regression test + before/after**, not new
  measurement tooling — so it does not violate MEAS-04's intent (which bars new
  *measurement* code from `src/`). — **Reversibility:** costly — touches a public
  metric function used across the library; a behavior change to MASE on
  intermittent/constant series must stay backward-compatible and be covered by a test.
- **D-04:** On denominator collapse (all-constant training window, or fewer than one
  season of data), **substitute a period-1 naive denominator** rather than dropping
  the series. Rationale: matches statsforecast's MASE behavior, which keeps the
  ACCUR-08 validation apples-to-apples and avoids silently shrinking the aggregate's
  series count. — **Reversibility:** reversible.

### Committed-baseline breadth (ACCUR-07, ACCUR-08)
- **D-05:** First committed `accuracy.json` = **Naive2 + AutoETS** over **M3 across
  its frequencies (Yearly / Quarterly / Monthly)**. This gives genuine per-frequency
  stratification (ACCUR-07) on the single *validated* competition dataset, anchors on
  M3-monthly MASE ≈ 0.93 (ACCUR-08), and keeps the first capture small enough to run
  on a maintainer machine. The **loader reads all four required corpora** (M3, M4
  sample, Tourism, NN5 — ACCUR-01) and the per-frequency schema generalizes, so
  additional models/datasets/frequencies plug in without rework — mirrors Phase 1's
  "validate the anchor first, populate breadth later" pattern. — **Reversibility:**
  reversible (adding rows to the schema).

### Cross-library reference (BENCH-01) — committed fixture, pinned regeneration
- **D-06:** The statsforecast reference is a **committed JSON fixture**, regenerated
  **once** on a **pinned Python env** (statsforecast/numpy/pandas versions recorded in
  a provenance block, matching Phase 1's fingerprint discipline), verified to reproduce
  AutoETS M3-monthly MASE ≈ 0.93, then committed. `accuracy.yml` **reads** the fixture —
  no live Python in the gating path. Mirrors the Phase 1 rule (CI reads baselines,
  never captures them) and keeps the reference number auditable. — **Reversibility:**
  reversible.

### Naive2 & Diebold–Mariano (ACCUR-06, BENCH-02)
- **D-07:** **Naive2 lives in the harness** (`crates/anofox-bench-harness`), not the
  public API — it's a measurement reference, not a shipped model (honors MEAS-04, keeps
  the public `Forecaster` API frozen during a hardening phase). It may compose from the
  library's existing `Naive` / `SeasonalNaive` primitives, but the seasonality gate is
  harness-owned. — **Reversibility:** reversible.
- **D-08:** Naive2 seasonality gate = **statsforecast-style 90%-confidence
  autocorrelation test at the seasonal lag** (the canonical M4/statsforecast Naive2
  definition: seasonal-naive when the ACF test passes, else random-walk naive). Keeps
  ACCUR-08 / BENCH-01 apples-to-apples with the reference. — **Reversibility:** reversible.
- **D-09:** Diebold–Mariano test = **squared-error loss + Harvey–Leybourne–Newbold
  (HLN) small-sample correction** with a **horizon-aware (HAC) variance estimator** —
  the standard rigorous form for M-competition-style "beats reference" claims. The
  researcher confirms the exact variance estimator. — **Reversibility:** reversible.

### Claude's Discretion (user deferred "you decide")
- **Dataset scope of the first lock** (D-05): user deferred; chosen M3 Y/Q/M to bound
  maintainer-machine runtime while satisfying ACCUR-07 stratification + ACCUR-08 anchor.
- **Reference capture mechanism** (D-06): user deferred the mechanism (chose fixture +
  pinned regen); user explicitly locked "regenerate + pin, then commit" for provenance.
- **Naive2 location, seasonality gate, DM form** (D-07..09): all deferred to Claude;
  chosen to keep the statsforecast comparison honest and rigorous.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap (authoritative scope)
- `.planning/ROADMAP.md` §"Phase 2: Accuracy Harness & Statistical Methodology" — goal, 5 success criteria, requirement mapping (ACCUR-01..08, BENCH-01..02).
- `.planning/REQUIREMENTS.md` §ACCUR (lines 30–37), §BENCH (lines 52–53); §"Out of Scope".
- `.planning/STATE.md` §"Key Decisions" + §"Critical Constraints to Honor" — `accuracy.yml` `workflow_dispatch`-only; measurement code stays out of `src/` (MEAS-04); no customer/client names in code/tests.
- `.planning/PROJECT.md` §"Key Decisions" — BENCH-01/02 belong in Phase 2 (cross-library depends on a correct harness); standard competition datasets as the accuracy corpus.

### Prior phase context (carried forward)
- `.planning/phases/01-measurement-infrastructure-compute-baselines/01-CONTEXT.md` — baseline JSON schema + D-02 provenance fingerprint (reused for `accuracy.json` and the reference fixture); harness-crate ownership pattern (D-04/D-05 there); local-capture-not-CI rule.

### Existing code to build on / touch
- `src/utils/metrics.rs` — `mae`/`mse`/`rmse`/`smape`/`msis`/`coverage`/`calculate_mase`/`ForecastMetrics::compute`; **the MASE silent-NaN (`unwrap_or(NaN)`) is the D-03 fix target**.
- `src/utils/cross_validation.rs` — `cross_validate`, `CvFoldGenerator`, `Fold` (`train_start/train_end/test_start/test_end`), `CVConfig`, `ConstraintViolation` — reused per D-02.
- `crates/anofox-bench-harness/` — home for the accuracy harness, Naive2 (D-07), the loader, and the DM test; reuses Phase 1's fingerprinted baseline serde structs.
- `validation/data/` — vendored corpus: `m3_monthly.tsf` / `m3_quarterly.tsf` / `m3_yearly.tsf` / `m3_other.tsf`, all `m4_*.tsf`, `tourism_{monthly,quarterly,yearly}.tsf`, `nn5.tsf` / `nn5_weekly.tsf` (Monash `.tsf`), plus `m4_daily_{train,test}.json`, `statsforecast_reference.json`, M5 CSVs.
- `validation/run_statsforecast.py` — reference generator for BENCH-01 (D-06 regenerates + pins it).
- `validation/data/statsforecast_reference.json` — existing reference fixture; **provenance/version unknown → D-06 regenerates before trusting/locking**.
- `tests/` prior art — `full_statsforecast_comparison.rs`, `statsforecast_comparison.rs`, `m4_daily_accuracy_regression.rs`, `nixtla_validation.rs` (patterns for reference comparison + Monash/JSON loading).
- `.planning/baselines/` — where `accuracy.json` (and the reference fixture) land; CI reads, never writes (Phase 1 rule).
- `.github/workflows/` — new `accuracy.yml` sits beside `ci.yml`/`bench.yml`/`wasm-size.yml`; `workflow_dispatch`-only.

No external ADRs — requirements and constraints are fully captured in `.planning/` docs above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Metrics** (`src/utils/metrics.rs`): full point/interval metric set already exists and is reused (D-01). Only gaps to add: the MASE collapse guard (D-03/D-04, in `src/`) and Naive2 (D-07, in harness).
- **CV** (`src/utils/cross_validation.rs`): expanding-window folds with `train_end`/`test_start` fields already present — reused with an added assertion layer (D-02).
- **Reference infra**: `validation/run_statsforecast.py` + `statsforecast_reference.json` exist — regenerated/pinned rather than built from scratch (D-06).
- **Loader prior art**: existing tests already read `.tsf`/JSON competition data (`m4_daily_accuracy_regression.rs`, `full_statsforecast_comparison.rs`) — reference for the ACCUR-01 Monash `.tsf` loader.

### Established Patterns
- Phase 1's per-dimension committed JSON baseline + provenance fingerprint (D-02 there) is the template for `accuracy.json` and the reference fixture.
- Models drive through the uniform `Forecaster` `fit`/`predict` interface (AutoETS `::auto()` etc.) — the harness runs Naive2 + AutoETS through it.
- CI reads baselines, never captures them; capture happens locally / on a pinned env via documented scripts.

### Integration Points
- Accuracy harness code lives in `crates/anofox-bench-harness` (and `tests/` where a native test form is cleaner); **the only `src/` change this phase is the D-03 MASE correctness fix.**
- `accuracy.yml` joins the existing workflow set as `workflow_dispatch`-only.
- The reference fixture + `accuracy.json` land under `.planning/baselines/`.

### Coverage gaps this phase fills
- No Naive2 anywhere (ACCUR-06) — new, in harness (D-07).
- No Diebold–Mariano test anywhere (BENCH-02) — new (D-09).
- MASE silently returns NaN on collapse (ACCUR-03) — fixed in `src/` (D-03/D-04).

</code_context>

<specifics>
## Specific Ideas

- ACCUR-08 anchor is concrete and locked: **statsforecast AutoETS M3-monthly MASE ≈ 0.93** must reproduce before the baseline is committed.
- Naive2 definition anchors on the **statsforecast/M4** form: 90%-confidence ACF test at the seasonal lag → seasonal-naive, else random-walk naive (D-08).
- DM form is concrete: **squared-error loss + HLN small-sample correction + horizon-aware HAC variance** (D-09).
- Reference fixture must carry a **provenance block** (pinned statsforecast/numpy/pandas versions) like Phase 1's fingerprint (D-06).
- MASE collapse fallback = **period-1 naive denominator** (D-04), matching statsforecast.

</specifics>

<deferred>
## Deferred Ideas

- **Accuracy numbers for M4 / Tourism / NN5 and all frequencies** — loader + schema support them now, but only M3 (Y/Q/M) is committed in this first lock (D-05). Expand in a follow-up or when the Phase 4 backlog points at a specific dataset/model.
- **Broader model set in `accuracy.json`** (AutoARIMA/AutoTheta/Croston/Laplace/AutoEnsemble) — considered (Phase 1's 7-family speed set) but deferred; Croston/Laplace need intermittent datasets to be meaningful. First accuracy lock is Naive2 + AutoETS (D-05).
- **Fixing the MASE silent-NaN as a Phase 4 backlog item** — superseded: user chose to fix it in `src/` now (D-03) rather than defer.

None outside phase scope surfaced beyond the above.

</deferred>

---

*Phase: 2-Accuracy Harness & Statistical Methodology*
*Context gathered: 2026-08-10*
