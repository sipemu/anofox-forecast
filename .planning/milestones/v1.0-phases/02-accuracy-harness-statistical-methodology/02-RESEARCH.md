# Phase 02: Accuracy Harness & Statistical Methodology — Research

**Researched:** 2026-08-10
**Domain:** Time-series accuracy metrics, expanding-window CV, Diebold-Mariano test, Monash TSF loader, statsforecast reference validation
**Confidence:** MEDIUM (core metrics verified in source; DM test / Naive2 formulas from cross-checked web sources)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Reuse `src/utils/metrics.rs` (`calculate_mase`, `smape`, `rmse`, `mae`, `msis`, `coverage`) rather than re-implementing in the harness. The ACCUR-08 anchor validates correctness.
- **D-02:** Reuse `src/utils/cross_validation.rs` (`CvFoldGenerator`, `Fold { train_start, train_end, test_start, test_end }`); harness adds explicit per-fold `train_end < test_start` assertion layer.
- **D-03:** Fix the MASE denominator-collapse guard **in `src/utils/metrics.rs`** (currently `unwrap_or(NaN)` at line 528 of `ForecastMetrics::compute` and `None` at line 156 of `calculate_mase`). Framed as a correctness fix with regression test.
- **D-04:** On denominator collapse (constant training window / fewer than one season), substitute a **period-1 naive denominator** (mean absolute first-difference of training data) rather than dropping the series. Matches statsforecast behavior.
- **D-05:** First committed `accuracy.json` = **Naive2 + AutoETS over M3 (Yearly / Quarterly / Monthly)**. Loader reads all four corpora (M3, M4 sample, Tourism, NN5); only M3 Y/Q/M numbers are locked in the first baseline.
- **D-06:** Statsforecast reference = **committed JSON fixture** regenerated once on a **pinned Python env** (statsforecast/numpy/pandas versions in provenance block), then committed. CI reads the fixture — no live Python in the gating path.
- **D-07:** Naive2 lives in `crates/anofox-bench-harness`, not the public API.
- **D-08:** Naive2 seasonality gate = **90%-confidence ACF test at the seasonal lag** (statsforecast/M4 Naive2 canonical form): seasonal-naive when ACF test passes, else random-walk naive.
- **D-09:** Diebold–Mariano = **squared-error loss + HLN small-sample correction + horizon-aware HAC variance**. Researcher confirms exact variance estimator.
- **accuracy.yml** is `workflow_dispatch`-only, never gates PR merges (MEAS-03, MEAS-04).
- All measurement code lives in `benches/`, `tests/`, `scripts/`, harness crate — nothing new enters `src/` except the D-03 correctness fix.
- No customer/client names in code, comments, or test names.

### Claude's Discretion
- Dataset scope of the first lock (D-05): chosen M3 Y/Q/M.
- Reference capture mechanism (D-06): chosen fixture + pinned regen.
- Naive2 location, seasonality gate, DM form (D-07..09): all deferred to Claude.

### Deferred Ideas (OUT OF SCOPE)
- Accuracy numbers for M4 / Tourism / NN5 and all frequencies (schema supports them, not committed this phase).
- Broader model set in `accuracy.json` (AutoARIMA, AutoTheta, Croston, etc.) — deferred.
- Fixing the MASE silent-NaN as a Phase 4 backlog item — superseded by D-03.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ACCUR-01 | Dataset loader reads `validation/data/` (Monash `.tsf` + CSV/JSON) for M3, M4 sample, Tourism, NN5, gated on `ANOFOX_DATASET_DIR` | TSF parser pattern verified in existing examples; loader structure documented in Architecture Patterns |
| ACCUR-02 | Expanding-window CV with per-fold `train_end < test_start` temporal-integrity assertions; no in-sample metrics | `CvFoldGenerator` + `Fold` struct verified in `src/utils/cross_validation.rs:32-53`; assertion wrapper pattern documented |
| ACCUR-03 | MASE computed with correct seasonal denominator + collapse guard; no silent NaN/Inf | Current bug verified at `src/utils/metrics.rs:156` and line 528; fix documented |
| ACCUR-04 | sMAPE, RMSE, MAE implemented/verified with correct denominators | All verified in `src/utils/metrics.rs` — standalone functions at lines 172-212 |
| ACCUR-05 | MSIS and empirical interval-coverage check for prediction intervals | `msis()` at lines 307-341, `coverage()` at lines 348-357; MSIS formula documented |
| ACCUR-06 | Naive2 baseline (autocorrelation-gated seasonal/non-seasonal) available | D-07/D-08 locked; ACF test threshold (90%), implementation pattern documented |
| ACCUR-07 | Per-frequency stratified reporting (never cross-frequency aggregate) | Schema + output structure documented; M3 has built-in frequency field in TSF header |
| ACCUR-08 | Committed baseline validated against statsforecast AutoETS M3 monthly MASE ≈ 0.93 | Reference fixture exists at `validation/data/statsforecast_reference.json`; regeneration pattern in `run_statsforecast.py` |
| BENCH-01 | Documented cross-library comparison vs statsforecast on shared datasets/horizons/preprocessing | `validation/run_statsforecast.py` and `validation/data/statsforecast_reference.json` exist; D-06 pins regen |
| BENCH-02 | "Beats reference" claim with < 5% gap gated by Diebold-Mariano significance test | DM + HLN formula documented; implementation pattern in Pitfalls |
</phase_requirements>

---

## Summary

Phase 02 builds a **statistically correct accuracy harness** on top of existing library infrastructure. The key finding from codebase inspection is that the library already contains all required metric functions (`mase`, `smape`, `rmse`, `mae`, `msis`, `coverage`) and an expanding-window cross-validation engine (`CvFoldGenerator`, `cross_validate`), so this phase is primarily about **wiring and validation**, not building from scratch. The single source-level bug is the MASE denominator-collapse (verified at `src/utils/metrics.rs:156` and line 528).

Two wholly new components are needed: (1) a Monash `.tsf` dataset loader and (2) a Naive2 baseline + Diebold-Mariano test, both living in `crates/anofox-bench-harness`. The anchor for correctness is ACCUR-08: the committed `accuracy.json` must reproduce statsforecast AutoETS M3-monthly MASE ≈ 0.93 before being locked. The existing `skaters_m3_monthly_benchmark.rs` example demonstrates the full TSF-parse → train/test split → MASE pipeline and serves as a verified pattern to build on.

The Diebold-Mariano test with HLN correction and HAC variance is a moderate implementation effort (~50 lines of Rust) but is well-specified. Naive2 composes the library's existing `Naive` / `SeasonalNaive` primitives, gated by a 90%-confidence ACF test at the seasonal lag.

**Primary recommendation:** Build the harness incrementally — loader first, then MASE fix + regression test, then Naive2, then DM test, then the accuracy run + reference validation, then commit the baseline. Never commit `accuracy.json` before the ACCUR-08 anchor passes.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Metric computation (MASE, sMAPE, etc.) | Library (`src/utils/metrics.rs`) | — | Reuse per D-01; bug fix (D-03) touches this tier only |
| Expanding-window CV fold generation | Library (`src/utils/cross_validation.rs`) | Harness assertion wrapper | Reuse per D-02; harness adds assertion, does not re-implement |
| Dataset loader (TSF + JSON) | Harness (`crates/anofox-bench-harness`) | — | Measurement code stays out of `src/` per MEAS-04 |
| Naive2 baseline | Harness (`crates/anofox-bench-harness`) | Composes library `Naive`/`SeasonalNaive` | Not a shipped model (D-07) |
| Diebold-Mariano test | Harness (`crates/anofox-bench-harness`) | — | Measurement/statistical tooling, not library feature |
| Accuracy JSON baseline | `.planning/baselines/accuracy.json` | — | CI reads, never writes (Phase 1 rule) |
| Reference fixture | `.planning/baselines/statsforecast_reference.json` | — | Committed once, pinned regen only |
| accuracy.yml workflow | `.github/workflows/accuracy.yml` | — | `workflow_dispatch`-only; never gates PR |

---

## Standard Stack

### Core (all existing, no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `anofox-forecast` | `0.15.8` (path dep) | Core metrics + CV reuse | All metric/CV logic already exists; phase reuses not re-implements |
| `serde` / `serde_json` | `1.0` (already in harness) | JSON baseline serialization | Used by Phase 1 baseline schema structs |
| `chrono` | `0.4` (already in harness) | Timestamp construction for TSF series | Already a harness dependency |

### Supporting (no new external packages)

No new Cargo dependencies are introduced this phase. All computation uses the existing library. The Diebold-Mariano test, Naive2, and TSF loader are implemented in pure Rust using `std` + the above.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolling MASE in harness | Reuse `calculate_mase` (D-01) | Reuse is correct; any metric bug would surface via ACCUR-08 anchor |
| New `dm_test` crate | Inline HLN+HAC implementation in harness | No published Rust DM crate with proven provenance; inline is ~50 lines and directly verifiable |

**Installation:** No new packages — zero new `cargo add` calls this phase.

---

## Package Legitimacy Audit

> No external packages are introduced this phase. All computation uses the existing harness and library dependencies.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Architecture Patterns

### System Architecture Diagram

```
ANOFOX_DATASET_DIR env var
         │
         ▼
  ┌──────────────────┐
  │  Dataset Loader  │  (harness) reads m3_monthly.tsf etc., extracts
  │  (TSF + JSON)    │  (series_id, frequency, horizon, train, test)
  └────────┬─────────┘
           │  Vec<(SeriesId, Frequency, TrainValues, TestValues)>
           ▼
  ┌──────────────────────────────────────────────────────┐
  │              Accuracy Harness Loop                   │
  │  for each series:                                    │
  │    1. CvFoldGenerator (expanding, library)           │
  │    2. assert fold.train_end < fold.test_start        │  ← temporal gate
  │    3. Naive2.fit(train) + predict(horizon)           │
  │    4. AutoETS.fit(train) + predict(horizon)          │
  │    5. compute MASE / sMAPE / RMSE / MAE per fold     │  ← reuse library metrics
  │    6. MSIS + coverage for interval forecasts         │
  │    7. aggregate per-frequency                        │
  └──────────┬─────────────────────────────────────────-┘
             │  accuracy results: HashMap<Frequency, MetricRow>
             ▼
  ┌──────────────────────────────────┐
  │  ACCUR-08 validation step        │
  │  M3-monthly AutoETS MASE ≈ 0.93  │  ─── compare against ──→  statsforecast_reference.json
  │  (must pass before lock)         │
  └──────────┬───────────────────────┘
             │  if pass
             ▼
  ┌──────────────────────┐
  │  accuracy.json       │  committed to .planning/baselines/
  │  (per-frequency      │
  │   + provenance)      │
  └──────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │  BENCH-02: Diebold-Mariano gate         │
  │  if |MASE gap| < 5%:                   │
  │    DM (HLN + HAC) → reject / fail-gate │
  └─────────────────────────────────────────┘
```

### Recommended Project Structure

```
crates/anofox-bench-harness/
├── src/
│   ├── lib.rs           # re-exports: baseline, fixtures, loader, naive2, dm_test
│   ├── baseline.rs      # ProvenanceFingerprint + all baseline JSON serde structs (Phase 1)
│   ├── fixtures.rs      # deterministic seeded series (Phase 1)
│   ├── loader.rs        # NEW: Monash TSF + JSON dataset loader (ACCUR-01)
│   ├── naive2.rs        # NEW: Naive2 model struct (ACCUR-06, D-07)
│   └── dm_test.rs       # NEW: Diebold-Mariano + HLN + HAC (BENCH-02, D-09)
├── tests/
│   ├── dhat_peak.rs     # Phase 1 (unchanged)
│   └── accuracy.rs      # NEW: accuracy harness integration test (ACCUR-02..08, BENCH-01)
.planning/baselines/
├── accuracy.json        # NEW: Naive2 + AutoETS over M3 Y/Q/M
└── statsforecast_reference.json  # regenerated + pinned (D-06)
.github/workflows/
└── accuracy.yml         # NEW: workflow_dispatch-only
src/utils/
└── metrics.rs           # MODIFIED: D-03 MASE collapse guard fix
```

### Pattern 1: TSF Loader (ACCUR-01)

**What:** Parse Monash `.tsf` files into typed `DatasetSeries` structs, gated on `ANOFOX_DATASET_DIR`.

**When to use:** Any harness code that reads M3/M4/Tourism/NN5 datasets.

**Example (from `examples/skaters_m3_monthly_benchmark.rs:52-87` and `examples/m4_hourly_diagnostic.rs:21-46`):**

```rust
// Source: verified in examples/skaters_m3_monthly_benchmark.rs:52-87
// Pattern is: read bytes as Latin-1, split on @data, parse lines as id:start_ts:v1,v2,...
fn parse_tsf(path: &str) -> Vec<(String, Vec<f64>)> {
    let bytes = std::fs::read(path).expect("read tsf");
    let content: String = bytes.iter().map(|&b| b as char).collect();  // Latin-1 → lossless
    let mut series = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
        if !in_data {
            if line.trim_start().starts_with("@data") { in_data = true; }
            continue;
        }
        let mut parts = line.splitn(3, ':');
        let id = match parts.next() { Some(s) => s.to_string(), None => continue };
        let _start_ts = parts.next();
        let vals_str = match parts.next() { Some(s) => s, None => continue };
        let values: Vec<f64> = vals_str.split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if !values.is_empty() { series.push((id, values)); }
    }
    series
}
```

The harness version must additionally:
- Extract `@frequency` and `@horizon` from the metadata section (before `@data`)
- Return a typed `DatasetSeries { id, frequency, horizon, values }` 
- Be gated by `ANOFOX_DATASET_DIR` env var (skip test if not set)

**TSF metadata tags (from file inspection):**
- `@frequency monthly` / `quarterly` / `yearly` [VERIFIED: validation/data/m3_monthly.tsf:12, m3_quarterly.tsf:12, m3_yearly.tsf:11]
- `@horizon 18` (monthly), `8` (quarterly), `6` (yearly) [VERIFIED: validation/data/m3_monthly.tsf:13, m3_quarterly.tsf:13, m3_yearly.tsf:12]

### Pattern 2: Expanding-Window CV with Temporal Assertion (ACCUR-02)

**What:** Wrap `CvFoldGenerator::generate()` with a per-fold assertion layer.

**When to use:** Every accuracy evaluation fold — replace any use of `fitted_values()` in accuracy context.

**Key types (verified in `src/utils/cross_validation.rs:32-53`):**
```rust
// Source: [VERIFIED: src/utils/cross_validation.rs:32-53]
pub struct Fold {
    pub train_start: usize,  // inclusive
    pub train_end: usize,    // exclusive
    pub test_start: usize,   // inclusive
    pub test_end: usize,     // exclusive
}
```

Assertion wrapper pattern:
```rust
// In harness: assert temporal integrity on each fold
for fold in &folds {
    assert!(
        fold.train_end <= fold.test_start,
        "temporal integrity violation: train_end={} > test_start={}",
        fold.train_end, fold.test_start
    );
    // then fit on train slice, predict horizon, evaluate on test slice
}
```

**Note:** `CvFoldGenerator` by design produces `train_end <= test_start` (gap=0 means they are adjacent), so the assertion is a safety net that should never fire on well-formed inputs — but it must exist per ACCUR-02.

### Pattern 3: MASE Denominator-Collapse Guard Fix (ACCUR-03, D-03/D-04)

**What:** Fix `calculate_mase` to substitute period-1 naive denominator when seasonal denominator collapses.

**Current bug (verified at `src/utils/metrics.rs:156`):**
```rust
// [VERIFIED: src/utils/metrics.rs:156]  — CURRENT BROKEN CODE
if naive_mae == 0.0 {
    return None;  // propagates as NaN via unwrap_or(NaN) at line 528
}
```

**Fix pattern (D-04):**
```rust
// After the seasonal naive_mae calculation at lines 148-154:
let naive_mae = if naive_mae == 0.0 {
    // Denominator collapse guard (D-04): substitute period-1 naive denominator
    // to match statsforecast behavior (keeps series in aggregate, not dropped)
    let p1_mae: f64 = actual.iter().skip(1).zip(actual.iter())
        .map(|(curr, prev)| (curr - prev).abs())
        .sum::<f64>()
        / (n - 1) as f64;
    if p1_mae == 0.0 { return None; }  // truly constant — nothing can scale
    p1_mae
} else {
    naive_mae
};
```

This fix must be accompanied by:
1. A regression test (`#[test] fn mase_constant_series_no_nan()`): constant training window must not return `NaN` in the aggregate — it must return a finite value using the period-1 fallback.
2. Before/after documentation in the commit message and in `.planning/baselines/accuracy.json`'s provenance block.

### Pattern 4: Naive2 (ACCUR-06, D-07/D-08)

**What:** Harness-only model that composes `Naive` + `SeasonalNaive` based on a 90%-confidence ACF test.

**Canonical definition (M4 competition / statsforecast):** [CITED: websearch cross-checked from multiple sources]
- Compute the autocorrelation coefficient of the training series at lag = seasonal_period.
- Test H₀: ACF(seasonal_period) = 0 at 90% confidence (two-sided z-test: |ACF| > 1.645/√n where n = len(training) — this is the approximate critical value for 90% one-tailed, or use ±1.645/√n two-tailed at α=0.1).
- If test rejects H₀ (seasonal signal detected): use `SeasonalNaive` (repeat last season).
- Else: use `Naive` (random walk, last value carried forward).

**Implementation in harness `src/naive2.rs`:**
```rust
pub struct Naive2 {
    seasonal_period: usize,
    inner: Naive2Inner,
}

enum Naive2Inner { Seasonal(SeasonalNaive), Random(Naive) }

impl Naive2 {
    pub fn new(seasonal_period: usize) -> Self {
        Self { seasonal_period, inner: Naive2Inner::Random(Naive::new()) }
    }
    
    pub fn fit(&mut self, train: &[f64]) -> Result<(), ForecastError> {
        let acf_at_lag = acf_at_lag(train, self.seasonal_period);
        let n = train.len() as f64;
        let critical = 1.645 / n.sqrt();  // 90% two-sided (α=0.10)
        if acf_at_lag.abs() > critical {
            let mut m = SeasonalNaive::new(self.seasonal_period);
            m.fit(&make_ts(train))?;
            self.inner = Naive2Inner::Seasonal(m);
        } else {
            let mut m = Naive::new();
            m.fit(&make_ts(train))?;
            self.inner = Naive2Inner::Random(m);
        }
        Ok(())
    }
}
```

ACF at lag k: `acf_k = [Σ_{t=k}^{n-1} (x_t - x̄)(x_{t-k} - x̄)] / [Σ_{t=0}^{n-1} (x_t - x̄)²]`

### Pattern 5: Diebold-Mariano Test with HLN + HAC (BENCH-02, D-09)

**What:** Statistical significance gate for "model A beats model B" claims when gap < 5%.

**Full formula (cross-checked from multiple sources):** [CITED: real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/]

Loss differential: `d_t = e1_t² - e2_t²` (squared error loss, per D-09)

HAC variance of `d_bar`:
```
V_hat = (1/T²) × [γ₀ + 2×Σ_{k=1}^{h-1} γ_k]
```
where `γ_k = (1/T) × Σ_{t=k}^{T-1} (d_t - d_bar)(d_{t-k} - d_bar)`.

Raw DM statistic: `DM = d_bar / sqrt(V_hat)`

HLN small-sample correction (S* statistic):
```
S* = DM × sqrt[(T + 1 - 2h + h(h-1)/T) / T]
```

Compare `S*` against `t(T-1)` distribution (two-sided p-value). Return `(s_star, p_value, reject_h0)`.

Gate rule (BENCH-02): If `|metric_gap| < 0.05` (5%), require `p_value < 0.05` to claim superiority.

### Anti-Patterns to Avoid

- **Using `fitted_values()` for accuracy:** `model.fitted_values()` returns in-sample residuals, not out-of-sample forecasts. Every accuracy table row must come from `model.predict(horizon)` after training on a held-out fold. [VERIFIED: requirement ACCUR-02]
- **Cross-frequency aggregation:** Never average MASE or sMAPE across M3-Yearly and M3-Monthly series in the same number — they have different denominators and horizons. Always stratify by `@frequency` tag from the TSF header. [VERIFIED: requirement ACCUR-07]
- **Silent NaN in aggregates:** Any `NaN` in a per-series metric must be caught and logged, not silently averaged (NaN propagates). Use `f64::is_finite()` checks before aggregation.
- **Committing `accuracy.json` before ACCUR-08 passes:** The M3-monthly AutoETS MASE must be ≈ 0.93 (±0.02 tolerance) before the baseline is committed. Commit the reference fixture first, then validate, then commit `accuracy.json`.
- **Regenerating the reference fixture in CI:** The `accuracy.yml` workflow reads `statsforecast_reference.json` — it never runs Python or updates the fixture. Regeneration is a documented manual step.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Expanding-window fold generation | Custom fold iterator | `CvFoldGenerator` + `Fold` in `src/utils/cross_validation.rs` | Already handles gap/purge/embargo/step correctly |
| MASE / sMAPE / RMSE / MAE / MSIS / coverage | New metric functions | `src/utils/metrics.rs` standalone functions | All exist and are tested; only fix needed is D-03 |
| TimeSeries construction from slice | Custom wrapper | `TimeSeries::univariate(timestamps, values)` | Library entry point |
| JSON serialization with provenance | Custom serializer | `ProvenanceFingerprint` + serde structs in `crates/anofox-bench-harness/src/baseline.rs` | Phase 1 baseline schema already defined |

**Key insight:** The hardest-to-get-right parts (MASE denominator correctness, expanding-window temporal integrity) already exist in the library. This phase's value is in the orchestration: running the harness over real competition data and validating against a published reference.

---

## Common Pitfalls

### Pitfall 1: MASE Denominator Uses Test Window, Not Training Window

**What goes wrong:** MASE naive denominator is computed over the **training window**, not the test window. The existing `calculate_mase` in `src/utils/metrics.rs:148-154` correctly uses `actual` (the test slice) for forecast MAE but `actual` again for the naive denominator — but the `actual` passed to this function in cross-validation is the **test slice**. This means the denominator is the naive MAE over the test slice, not the training slice.

**Root cause:** The current signature `calculate_mase(actual, predicted, seasonal_period)` does not accept the training data separately. The M-competition definition uses the in-sample naive MAE as denominator.

**How to avoid:** For cross-validation harness use, compute the seasonal-naive denominator from the **training slice** before calling `calculate_mase`, or pass the training slice separately. The harness can compute `mase_scale = (training_naive_mae).max(1e-9)` and then `mase = forecast_mae / mase_scale` — matching the pattern in `examples/skaters_m3_monthly_benchmark.rs:313-322` [VERIFIED: skaters_m3_monthly_benchmark.rs:313-322].

**Warning signs:** MASE values above 3 on normal seasonal data, or MASE = 1.0 exactly on all folds (denominator equals test data).

**Impact:** The existing `calculate_mase` signature in `ForecastMetrics::compute` wraps test-slice actual — so it must NOT be called for the competition accuracy harness. The harness must compute the denominator separately from training data.

### Pitfall 2: TSF File Is Latin-1, Not UTF-8

**What goes wrong:** The M3 monthly `.tsf` contains non-UTF-8 bytes (domain category labels with accented characters). `std::fs::read_to_string()` will panic or return `Err` on these files.

**Root cause:** Monash `.tsf` format uses Latin-1 (ISO-8859-1) encoding.

**How to avoid:** Read as bytes then decode losslessly: `bytes.iter().map(|&b| b as char).collect::<String>()`. [VERIFIED: examples/skaters_m3_monthly_benchmark.rs:56-57 and examples/m4_hourly_diagnostic.rs:22-24 — both use this pattern]

**Warning signs:** Panic or `invalid utf-8 sequence` error when loading any `.tsf` file.

### Pitfall 3: NaN Propagation in Aggregate MASE

**What goes wrong:** If any series has a constant training window (seasonal denominator = 0), the current code returns `None` / `NaN` for MASE. If that NaN enters the per-frequency aggregate sum, the whole frequency bucket becomes NaN.

**Root cause:** `calculate_mase` line 156 returns `None`; `ForecastMetrics::compute` line 528 uses `unwrap_or(f64::NAN)`. [VERIFIED: src/utils/metrics.rs:156, 528]

**How to avoid:** D-03 fix (period-1 fallback); plus the harness aggregate must use `filter(|m| m.is_finite())` before summing, and count/log skipped series.

### Pitfall 4: The MSIS Denominator Is Mean Absolute First-Difference, Not Seasonal Difference

**What goes wrong:** The existing `msis()` in `src/utils/metrics.rs:307-341` scales by `mean |y_t - y_{t-1}|` (period-1 first differences), which is correct for the M4 definition when `actual` is the test slice. However, the M4 competition definition scales by the in-sample seasonal first-difference `(1/(n-m)) Σ|y_t - y_{t-m}|`.

**Root cause:** The existing implementation uses period-1 differences, which matches the unscaled "Winkler score" convention but diverges from the M4 MSIS formula when the series has seasonality.

**How to avoid:** For the accuracy harness, the planner must decide whether to use the existing `msis()` as-is (consistent internally) or compute the M4-compatible MSIS with training-set seasonal scaling. The CONTEXT.md does not lock this choice — treat as a planner decision. Recommend: use the existing `msis()` for interval evaluation, document the convention, and do not claim it equals M4-competition MSIS without the seasonal denominator.

**Warning signs:** MSIS values dramatically different from statsforecast for the same prediction intervals.

### Pitfall 5: DM Test Requires Aligned Per-Step Loss Differentials

**What goes wrong:** The DM test operates on a sequence of `T` loss differentials `d_t`, one per test observation. When using expanding-window CV with multiple folds, the loss differentials from different folds (different training sizes) are not iid — naive concatenation inflates `T` artificially.

**Root cause:** DM test assumes a single out-of-sample evaluation period.

**How to avoid:** Run the DM test on the single held-out test split (the last `H` steps after all training data), not on cross-validated fold averages. The cross-validation is for MASE estimation; the DM test gate is applied on the single validation set comparison between anofox and statsforecast reference.

---

## Code Examples

### TSF Metadata Extraction

```rust
// Source: derived from verified pattern in examples/skaters_m3_monthly_benchmark.rs:52-87
// and examples/m4_hourly_diagnostic.rs:21-46
fn parse_tsf_with_meta(path: &str) -> (String, usize, Vec<(String, Vec<f64>)>) {
    let bytes = std::fs::read(path).unwrap();
    let content: String = bytes.iter().map(|&b| b as char).collect();
    let mut frequency = String::new();
    let mut horizon: usize = 0;
    let mut series = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_data {
            if let Some(val) = trimmed.strip_prefix("@frequency") {
                frequency = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("@horizon") {
                horizon = val.trim().parse().unwrap_or(0);
            } else if trimmed.starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        let mut parts = trimmed.splitn(3, ':');
        let id = match parts.next() { Some(s) => s.to_string(), None => continue };
        let _ts = parts.next();
        let vals_str = match parts.next() { Some(s) => s, None => continue };
        let values: Vec<f64> = vals_str.split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        if !values.is_empty() { series.push((id, values)); }
    }
    (frequency, horizon, series)
}
```

### MASE with Training Denominator (Competition-Correct)

```rust
// Source: pattern from examples/skaters_m3_monthly_benchmark.rs:313-322
// [VERIFIED: skaters_m3_monthly_benchmark.rs:313-322]
fn mase_scale(train: &[f64], period: usize) -> f64 {
    if train.len() <= period {
        return 1.0;  // collapse guard: too short for seasonal denominator
    }
    let n = train.len() - period;
    let sum: f64 = (period..train.len())
        .map(|i| (train[i] - train[i - period]).abs())
        .sum();
    (sum / n as f64).max(1e-9)  // .max(1e-9) = D-04 period-1 fallback spirit
}

fn compute_mase(train: &[f64], actual: &[f64], predicted: &[f64], period: usize) -> f64 {
    let denom = mase_scale(train, period);
    let forecast_mae: f64 = actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>() / actual.len() as f64;
    forecast_mae / denom
}
```

### Diebold-Mariano with HLN + HAC

```rust
// Source: [CITED: real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/]
// Implements squared-error loss + HAC variance + HLN small-sample correction per D-09.
pub fn diebold_mariano_hln(
    e1: &[f64],  // forecast errors from model 1 (actual - predicted1)
    e2: &[f64],  // forecast errors from model 2 (actual - predicted2)
    h: usize,    // forecast horizon
) -> (f64, bool) {  // (p_value, reject_h0)
    assert_eq!(e1.len(), e2.len());
    let t = e1.len();
    let d: Vec<f64> = e1.iter().zip(e2.iter()).map(|(e1, e2)| e1*e1 - e2*e2).collect();
    let d_bar = d.iter().sum::<f64>() / t as f64;
    
    // HAC variance: gamma_0 + 2*sum_{k=1}^{h-1} gamma_k
    let gamma = |k: usize| -> f64 {
        (k..t).map(|i| (d[i] - d_bar) * (d[i-k] - d_bar)).sum::<f64>() / t as f64
    };
    let gamma0 = gamma(0);
    let hac_var = if h <= 1 { gamma0 } else {
        gamma0 + 2.0 * (1..h).map(|k| gamma(k)).sum::<f64>()
    };
    let v_hat = hac_var / t as f64;
    let dm = d_bar / v_hat.sqrt();
    
    // HLN correction: S* = DM * sqrt[(T+1-2h+h(h-1)/T)/T]
    let correction = ((t as f64 + 1.0 - 2.0*h as f64 + h as f64*(h as f64-1.0)/t as f64) / t as f64).sqrt();
    let s_star = dm * correction;
    
    // Two-sided t(T-1) p-value (approximate via normal for large T)
    let p_value = 2.0 * normal_cdf(-s_star.abs());  // use statrs::distribution::Normal
    (p_value, p_value < 0.05)
}
```

### Accuracy JSON Schema (extends Phase 1 baseline pattern)

```json
{
  "provenance": {
    "git_sha": "...",
    "timestamp_iso": "...",
    "rustc_version": "...",
    "host_cpu": "...",
    "host_os": "...",
    "active_features": []
  },
  "datasets": {
    "M3": {
      "monthly": {
        "n_series": 1428,
        "horizon": 18,
        "models": {
          "AutoETS": { "mase": 0.93, "smape": 14.2, "rmse": 1234.5, "mae": 876.3 },
          "Naive2":  { "mase": 1.00, "smape": 16.1, "rmse": 1456.7, "mae": 987.6 }
        }
      },
      "quarterly": { ... },
      "yearly": { ... }
    }
  }
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MASE with test-slice denominator | MASE with training-slice denominator (competition-correct) | Phase 02 | Matches published M3/M4 results; enables ACCUR-08 validation |
| `unwrap_or(NaN)` on denominator collapse | Period-1 fallback denominator | Phase 02 (D-03/D-04) | No silent NaN in aggregates |
| No accuracy baseline | Committed `accuracy.json` anchored to statsforecast | Phase 02 | Makes accuracy regressions detectable |

**Deprecated/outdated:**
- Using `fitted_values()` for accuracy: these are in-sample residuals, not out-of-sample forecasts. Forbidden by ACCUR-02.
- Single cross-frequency MASE aggregate: hiding per-frequency differences. Forbidden by ACCUR-07.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Naive2 uses `1.645/sqrt(n)` as the 90%-confidence ACF critical value (z-test) | Pattern 4, Naive2 | ACF test picks wrong branch for some series → ACCUR-08 MASE mismatch vs statsforecast reference |
| A2 | The DM test normal CDF approximation for large T is adequate (vs exact t-table lookup) | Pattern 5, DM formula | Slightly incorrect p-values for very small test sets; use `statrs::distribution::StudentsT` for exact |
| A3 | `statsforecast_reference.json` currently uses the same M3 monthly series split (train/test) as the harness will use | ACCUR-08 validation | If the Python script uses different preprocessing, MASE ≈ 0.93 may not reproduce — D-06 regeneration step must align splits |
| A4 | The MSIS implementation in `src/utils/metrics.rs` uses period-1 first differences for scaling (not seasonal), which differs from M4 competition MSIS | Pitfall 4 | Reported MSIS values are not directly comparable to published M4 numbers |

---

## Open Questions

1. **MASE denominator in the existing `calculate_mase` vs competition convention**
   - What we know: `calculate_mase(actual, predicted, period)` in `metrics.rs` uses `actual` (the test slice) for both forecast MAE and naive MAE denominator [VERIFIED: src/utils/metrics.rs:148-168].
   - What's unclear: The planner needs to decide: (a) fix the signature to accept `train` separately, or (b) always compute MASE manually in the harness using the `mase_scale` helper pattern.
   - Recommendation: The harness should compute MASE manually using `mase_scale(train, period)` (verified pattern from `skaters_m3_monthly_benchmark.rs:313-322`), not via `ForecastMetrics::compute`. The D-01 decision to "reuse library metrics" applies to `mae`, `smape`, `rmse`, `msis`, `coverage` — but MASE in the competition context requires a separate training-slice denominator.

2. **MSIS scaling convention alignment with M4**
   - What we know: Existing `msis()` scales by mean period-1 differences of the test slice.
   - What's unclear: Whether ACCUR-08 includes an MSIS check, and if so, whether the convention must match M4's seasonal-difference denominator.
   - Recommendation: Clarify in the plan whether MSIS is validated against statsforecast in the reference fixture, or reported informally. If no ACCUR-08 MSIS anchor is needed, the current implementation is fine.

3. **`statrs` availability for t-distribution in DM test**
   - What we know: `statrs` is listed as a dependency in `CLAUDE.md` stack, not in `anofox-bench-harness/Cargo.toml`.
   - What's unclear: Whether to add `statrs` to harness dev-dependencies for `StudentsT::cdf`, or use the normal approximation.
   - Recommendation: Use the normal approximation for DM p-values (adequate for M-competition test set sizes which are typically T ≥ 18). Document the approximation.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python + statsforecast | D-06 reference fixture regen | ✓ (see `validation/pyproject.toml`) | statsforecast (pinned in uv.lock) | Use committed fixture if Python unavailable |
| `ANOFOX_DATASET_DIR` env var | ACCUR-01 dataset loader | Manual setup | — | Tests skip when not set |
| M3 `.tsf` files | ACCUR-01, ACCUR-08 | ✓ | `validation/data/m3_{monthly,quarterly,yearly}.tsf` | — |
| `validation/data/statsforecast_reference.json` | D-06 validation | ✓ (exists, provenance unknown) | — | Must regenerate + pin before trusting |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- `statsforecast_reference.json`: exists but provenance not pinned — D-06 must regenerate on pinned env before the ACCUR-08 comparison is trusted.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (native integration tests + harness tests) |
| Config file | None (standard `cargo test`) |
| Quick run command | `cargo test -p anofox-bench-harness` |
| Full suite command | `cargo test --all-features` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ACCUR-01 | Dataset loader reads M3 TSF; skips if `ANOFOX_DATASET_DIR` unset | integration | `ANOFOX_DATASET_DIR=./validation/data cargo test -p anofox-bench-harness tsf_loader` | ❌ Wave 0 |
| ACCUR-02 | Every fold satisfies `train_end <= test_start` | unit | `cargo test -p anofox-bench-harness temporal_integrity` | ❌ Wave 0 |
| ACCUR-03 | MASE on constant training series returns finite value (not NaN) | unit | `cargo test -p anofox-forecast mase_constant_series_no_nan` | ❌ Wave 0 |
| ACCUR-04 | sMAPE, RMSE, MAE match reference values for known inputs | unit | `cargo test -p anofox-forecast --lib metrics` (existing + new) | ✅ (existing) |
| ACCUR-05 | MSIS and coverage compute correctly for known interval inputs | unit | `cargo test -p anofox-forecast --lib msis` (existing) | ✅ (existing) |
| ACCUR-06 | Naive2 selects SeasonalNaive on seasonal series, Naive on flat | unit | `cargo test -p anofox-bench-harness naive2_seasonal_gate` | ❌ Wave 0 |
| ACCUR-07 | Accuracy JSON contains separate entries per frequency | integration | `cargo test -p anofox-bench-harness per_frequency_stratification` | ❌ Wave 0 |
| ACCUR-08 | M3 monthly AutoETS MASE ≈ 0.93 ± 0.02 | integration | `ANOFOX_DATASET_DIR=./validation/data cargo test -p anofox-bench-harness accur08_anchor` | ❌ Wave 0 |
| BENCH-01 | Reference comparison produces diff table | integration | `ANOFOX_DATASET_DIR=./validation/data cargo test -p anofox-bench-harness bench01_cross_library` | ❌ Wave 0 |
| BENCH-02 | DM test returns p > 0.05 when MASE gap < 5% on random data | unit | `cargo test -p anofox-bench-harness dm_test_unit` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p anofox-bench-harness` (harness unit tests)
- **Per wave merge:** `cargo test --all-features` + harness integration tests with dataset env var
- **Phase gate:** All integration tests green; ACCUR-08 anchor passes; `accuracy.json` committed

### Wave 0 Gaps
- [ ] `crates/anofox-bench-harness/src/loader.rs` — covers ACCUR-01
- [ ] `crates/anofox-bench-harness/src/naive2.rs` — covers ACCUR-06
- [ ] `crates/anofox-bench-harness/src/dm_test.rs` — covers BENCH-02
- [ ] `crates/anofox-bench-harness/tests/accuracy.rs` — covers ACCUR-02, ACCUR-07, ACCUR-08, BENCH-01
- [ ] Regression test in `src/utils/metrics.rs` for D-03 fix — covers ACCUR-03

---

## Security Domain

> `security_enforcement: true` in config.json. ASVS Level 1 applies.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Not applicable — internal test tool |
| V3 Session Management | no | Not applicable |
| V4 Access Control | no | Not applicable |
| V5 Input Validation | yes | `ANOFOX_DATASET_DIR` path must be validated before `std::fs::read`; panic on missing file is acceptable in test context |
| V6 Cryptography | no | Not applicable |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `ANOFOX_DATASET_DIR` | Tampering | Use `std::path::Path::canonicalize()` and assert prefix matches expected dir before opening |
| NaN injection via malformed TSF values | Tampering | `filter_map(|t| t.trim().parse::<f64>().ok())` already drops unparseable tokens; add `is_finite()` guard after parse |

---

## Sources

### Primary (HIGH confidence)
- `src/utils/metrics.rs:1-1115` — full metric implementations verified by direct file read
- `src/utils/cross_validation.rs:1-350` — `CvFoldGenerator`, `Fold`, `CVStrategy` structs verified by direct file read
- `crates/anofox-bench-harness/src/baseline.rs:1-81` — `ProvenanceFingerprint` and baseline serde structs verified
- `examples/skaters_m3_monthly_benchmark.rs:52-322` — TSF parser + MASE scale pattern verified
- `validation/data/m3_monthly.tsf:1-17` — TSF header format verified (frequency=monthly, horizon=18)
- `validation/data/m3_quarterly.tsf:1-13` — horizon=8 verified
- `validation/data/m3_yearly.tsf:1-12` — horizon=6 verified

### Secondary (MEDIUM confidence)
- [MetricGate: Diebold-Mariano](https://metricgate.com/docs/diebold-mariano-forecast/) — DM statistic formula, HLN formula, HAC variance definition
- [Real Statistics: DM Test](https://real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/) — S* = DM × sqrt[(T+1-2h+h(h-1)/T)/T], t(T-1) degrees of freedom

### Tertiary (LOW confidence)
- Web search cross-reference: "Naive2 uses 90% confidence ACF test" — mentioned consistently across multiple M4 competition and statsforecast sources; not verified from statsforecast source code directly

---

## Metadata

**Confidence breakdown:**
- Existing code (metrics.rs, cross_validation.rs, examples): HIGH — read directly this session
- DM test formula: MEDIUM — cross-checked from two independent sources
- Naive2 ACF threshold (90%): LOW — consistent across multiple web sources but statsforecast source not directly read
- TSF format: HIGH — verified from actual files in repo and existing working examples
- MSIS convention difference: HIGH — verified by reading existing msis() implementation

**Research date:** 2026-08-10
**Valid until:** 2026-09-10 (metric definitions are stable; statsforecast version pin may need refresh)
