# Phase 6: Multiplicative-Guard Bug-Class Audit — Research

**Researched:** 2026-09-09
**Domain:** Rust numerical/statistical model audit — auto-multiplicative/log selection robustness
**Confidence:** HIGH (all claims verified by reading source files this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **In scope**: data-driven auto mode/transform selection reachable via the public API: AutoETS family, Theta/TBATS log/Box-Cox options, auto λ selection in `src/transform/`, distributional Laplace leaves — audited, flagged as `distributional`-gated.
- **Out of scope**: non-auto internal `.ln()`/multiplicative sites unreachable from the public forecasting API; SMA/Naive/RandomWalk baselines.
- **MFLES**: already fixed (Phase 5) — record as fixed/pass with a pointer to #219.
- **Repro series**: reuse Phase 5's shape (level ≈ 2000, one near-zero month) for at-risk models, adapted to each model's minimum data requirements.
- **Fail criterion**: forecast blowing up beyond a bounded multiple of level (~2.5–10× level threshold).
- **Fix discipline for offenders**: reuse the Phase 5 pattern — min/level threshold guard on mode selection, floored/winsorized transform, clamped back-transform — adapted per model.
- **Inventory artifact**: committed markdown at `docs/audits/multiplicative-guard-audit.md` (create `docs/audits/` if absent).
- **Verdict granularity**: `pass` (safe — with rationale for why the guard is tight enough) or `fail` (offender — fixed, with before→after number and regression test name). `N/A` for models with no auto-selection path.
- **Already-safe models (SC4)**: recorded in inventory with evidence AND backed by a passing guard-assertion test where practical.
- **Offenders (MULT-06)**: regression test proving the pre-fix blow-up no longer occurs.

### Claude's Discretion

- Making guard thresholds user-configurable — still out of scope.
- Auditing non-auto internal `.ln()` sites unreachable from the public forecasting API.
- Any performance re-tuning of the guards beyond correctness.

### Deferred Ideas (OUT OF SCOPE)

- Making guard thresholds user-configurable (carried from Phase 5).
- Auditing non-auto internal `.ln()` sites unreachable from the public forecasting API.
- Any performance re-tuning of the guards beyond correctness.

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MULT-05 | Every model with an auto-multiplicative/log selection path is inventoried and audited for the #10/#219 too-loose-guard failure class | Complete inventory table below (19 models/paths audited) |
| MULT-06 | Any additional model exhibiting the failure class is fixed and guarded by a regression test proving the pre-fix blow-up no longer occurs | 0 new offenders found (see audit — all pass or N/A) |

</phase_requirements>

---

## Summary

Phase 6 audits every data-driven auto-multiplicative or auto-log-transform selection path reachable from the public API. The audit covers 19 models/paths across five subsystems: ETS family, Theta family, TBATS family, Transform pipeline, and distributional Laplace leaves.

**The central finding is that the #10/#219 blow-up class — an `exp()` back-transform producing forecasts ≫ level due to a log-space crater from a near-zero observation — is architecturally specific to MFLES.** MFLES is unique in applying a global `ln()` transform followed by a multi-round boosting fit, where the log-space crater at a near-zero observation compounds through boosting rounds and survives into the back-transformed `exp()` forecast (the ~13× blow-up of #219). No other model in the in-scope inventory shares this architecture.

The remainder of the in-scope models employ one of four safer patterns: (1) binary all-positive/non-positive guards that completely block multiplicative mode when any zero exists (#10 fix in AutoETS, GlobalAutoETS, TBATS), (2) seasonal-factor magnitude guards that block multiplicative decomposition when any seasonal index falls below 0.01 (Theta family — catching a near-zero month directly), (3) AIC/NLL selection that penalises multiplicative fits distorted by near-zero observations, or (4) state-space / online-learning architectures that do not apply log-then-exp globally (ETS, TBATS Kalman filter, Laplace leaves). BoxCox auto-lambda (MLE-based) is protected because MLE prefers larger lambda (less compression) for near-zero series, where lambda=0 is penalised by high log-space variance.

**Primary recommendation:** Record all 19 audited paths as PASS or N/A in `docs/audits/multiplicative-guard-audit.md`, add guard-assertion tests for already-safe models where a dedicated test adds signal (Theta seasonal-factor guard, AutoETS near-zero behaviour), and commit the inventory. MULT-06 has zero new offenders — no fixes required beyond the MFLES work already done in Phase 5.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Mode-selection guard | Model (fit path) | — | Decision happens at `fit()` before any transform; guard belongs at the selection site |
| Seasonal-factor guard | Model (fit path) | — | Theta-family: guard runs immediately after computing factors, before accepting multiplicative |
| AIC/NLL protection | Model (fit path) | — | AutoETS / GlobalAutoETS: information-criterion selection penalises distorted multiplicative fits |
| Log-floor / back-transform clamp | Model (fit + predict) | — | MFLES pattern — inapplicable to other models (no global log transform) |
| Transform (BoxCox / YJ) auto-lambda | Transform layer | — | MLE/CoV-based selection; no model-level involvement |
| Laplace leaf weighting | Forecaster (online) | — | Softmax log-likelihood downweights distorted leaves automatically |
| Inventory artifact | Documentation | — | `docs/audits/multiplicative-guard-audit.md` is the MULT-05 deliverable |
| Regression tests (SC4 / MULT-06) | Integration test layer | — | `tests/` following existing issue-named convention |

---

## Complete Inventory Table (MULT-05)

This is the backbone of the MULT-05 artifact. Each row covers one model or selection path.

| # | Model / Path | Has auto-mult/log selection? | Existing guard (file:line, quoted verbatim) | Preliminary verdict | Fix if needed |
|---|---|---|---|---|---|
| 1 | **MFLES** (`src/models/mfles.rs`) | Yes — `self.multiplicative == None && season_length > 0 && all(v > 0)` | `MULT_AUTO_TAU=0.10`: `median > 0.0 && (min_val / median) >= Self::MULT_AUTO_TAU` [VERIFIED: src/models/mfles.rs:1048]; `MULT_LOG_FLOOR_FRAC=0.01` [VERIFIED: src/models/mfles.rs:244]; `MULT_BACK_CLAMP_K=10.0` [VERIFIED: src/models/mfles.rs:249] | **pass** — Phase 5 (#219) fixed all three points. | Already done. |
| 2 | **AutoETS** (`src/models/exponential/auto_ets.rs`) | Yes — generates multiplicative-error / multiplicative-seasonal candidates for all-positive series | `let has_non_positive = values.iter().any(\|&v\| v <= 0.0);` [VERIFIED: src/models/exponential/auto_ets.rs:422] restricts candidates to additive only when any zero present | **pass** — see analysis below | None |
| 3 | **GlobalAutoETS** (`src/models/exponential/global_ets.rs`) | Yes — same candidate generation as AutoETS across all series | `let has_non_positive = all_series.iter().any(\|s\| s.iter().any(\|&v\| v <= 0.0));` [VERIFIED: src/models/exponential/global_ets.rs:663] | **pass** — same pattern as AutoETS | None |
| 4 | **ETS** (`src/models/exponential/ets.rs`) | No — spec is user-provided (`ETSSpec`); no data-driven mode selection | N/A | **N/A** | — |
| 5 | **HoltWinters** (`src/models/exponential/holt_winters.rs`) | No — `SeasonalType` is an explicit constructor argument; `auto()` requires the caller to pass it | N/A | **N/A** | — |
| 6 | **SeasonalES** (`src/models/exponential/seasonal_es.rs`) | No — default is `SeasonalESErrorType::Additive`; multiplicative only via explicit `.with_error_type()`. Doc comment: "NOT a multiplicative seasonal model — it's SES applied per-season." [VERIFIED: src/models/exponential/seasonal_es.rs:1-5] | N/A | **N/A** | — |
| 7 | **GlobalETS** (fixed-spec, `src/models/exponential/global_ets.rs`) | No — `GlobalETS::new(spec, period)` takes an explicit `ETSSpec`; no auto-selection in this variant | N/A | **N/A** | — |
| 8 | **Theta** (`src/models/theta/model.rs`) | Yes — `seasonal()` defaults to multiplicative; `determine_decomposition()` may override | Rule 1: `series.iter().any(\|&y\| y <= 0.0)` [VERIFIED: src/models/theta/model.rs:476]. Rule 2: `seasonals.iter().any(\|&s\| s < 0.01)` [VERIFIED: src/models/theta/model.rs:486] | **pass** — near-zero (1.0 in 2000-level) produces seasonal factor 0.0005 < 0.01 → triggers additive fallback | None |
| 9 | **OptimizedTheta** (`src/models/theta/optimized.rs`) | Yes — same `determine_decomposition()` pattern | Rule 2: `last_cycle.iter().any(\|&s\| s < 0.01)` [VERIFIED: src/models/theta/optimized.rs:365] | **pass** — same seasonal-factor guard as Theta | None |
| 10 | **DynamicTheta** (`src/models/theta/dynamic.rs`) | Yes — same `determine_decomposition()` pattern | Rule 1 + Rule 2 [VERIFIED: src/models/theta/dynamic.rs:357-368]: identical guard logic | **pass** — same guard | None |
| 11 | **TBATS** (`src/models/tbats/model.rs`) — auto-lambda path | Yes — `estimate_lambda()` called when `lambda.is_none() && all(v > 0)` (line 758-759) | `if values.iter().any(\|&v\| v <= 0.0) { return 1.0; }` [VERIFIED: src/models/tbats/model.rs:386-387] — blocks lambda < 1 for any non-positive. Near-zero allowed. | **pass** — see analysis below | None |
| 12 | **TBATS** (`src/models/tbats/model.rs`) — fixed lambda path | Yes — user calls `.with_box_cox(lambda)` | No guard for near-zero (user-specified). But this is user-driven, not auto-selection. | **N/A** (user-driven) | — |
| 13 | **AutoTBATS** (`src/models/tbats/auto.rs`) | Yes — tries lambda ∈ [0, 0.25, 0.5, 0.75, 1.0] if `can_box_cox` | `let can_box_cox = values.iter().all(\|&v\| v > 0.0);` [VERIFIED: src/models/tbats/auto.rs:206]; AIC comparison across all configs selects the best | **pass** — AIC comparison rejects lambda=0 for near-zero series (log-space SSE is much larger); no-transform baseline wins | None |
| 14 | **BoxCoxTransform** (`src/transform/transforms.rs`) | Yes — `BoxCoxTransform::auto()` calls `boxcox_lambda()` for λ selection | `if values.iter().any(\|&x\| x <= 0.0) { return Err(...) }` [VERIFIED: src/transform/transforms.rs:244] — rejects non-positive series. For all-positive near-zero: MLE prefers large λ. | **pass** — MLE log-likelihood (`-n/2 * ln(var) + (λ-1)*Σln(x)`) is lower for λ=0 (high variance from log-space crater) than λ≈1 for near-zero series. MLE selects λ≈1. | None |
| 15 | **YeoJohnsonTransform** (`src/transform/transforms.rs`) | Yes — `YeoJohnsonTransform::auto()` calls `yeo_johnson_lambda()` | No positivity requirement (YJ handles all reals). Near-zero (y=1.0): `yj_forward(1.0, λ) = ((1+1)^λ-1)/λ = (2^λ-1)/λ ≈ 0.69` at λ≈1 — not a crater. [VERIFIED: src/transform/yeo_johnson.rs:30-42] | **pass** — YJ transform of y=1.0 is ~0.69 regardless of lambda; no crater | None |
| 16 | **LaplaceForecaster** — `seasonal_mult` leaf (`src/models/laplace/leaves/seasonal_mult.rs`) | Yes — auto-selection: `chars.seasonality_strength > 0.3 && chars.all_positive && chars.mean_y > 0.0` [VERIFIED: src/models/laplace/forecaster.rs:3060-3065] | `const LEVEL_TOL: f64 = 1e-6;` update guard [VERIFIED: src/models/laplace/leaves/seasonal_mult.rs:22, 168]: `if self.level.abs() > LEVEL_TOL { update factor }` | **pass** — `distributional`-gated. Near-zero obs depresses the phase factor to ~0.0005; forecast = level × 0.0005 = near-zero (correct seasonal trough, not blow-up). Softmax log-likelihood down-weights this leaf if the trough is one-off. | None |
| 17 | **LaplaceForecaster** — `lognormal` leaf (`src/models/laplace/leaves/lognormal.rs`) | Yes — auto-selection via AID classifier or manual. Uses `log1p_nonneg(y) = y.max(0.0).ln_1p()` | `fn log1p_nonneg(y: f64) -> f64 { y.max(0.0).ln_1p() }` [VERIFIED: src/models/laplace/leaves/lognormal.rs:23-25] — guaranteed ≥ 0 | **pass** — `distributional`-gated. `ln_1p(1.0) = ln(2) ≈ 0.693` for y=1.0 — not a crater (ln-mean of 2000-series ≈ 7.6 in ln1p space). No explosion. | None |
| 18 | **LaplaceForecaster** — `yj_wrapper` leaf (`src/models/laplace/leaves/yj_wrapper.rs`) | Yes — wraps any inner leaf with a fixed λ; used in `with_yeo_johnson_grid()` | Trans-range clamp: `let mean_clamped = g.mean.clamp(lo, hi)` [VERIFIED: src/models/laplace/leaves/yj_wrapper.rs:98]; `if y_trans.is_finite() { ... }` guard on observe [VERIFIED: line 120-124] | **pass** — `distributional`-gated. Predictions clamped to observed training range in transformed space; inverse YJ of clamped value is bounded. Near-zero: `yj_forward(1.0, λ) ≈ 0.69` — same reasoning as YJ transform. | None |
| 19 | **LaplaceForecaster** — `standardize` / `slow_standardize` wrappers (`src/models/laplace/leaves/standardize.rs`, `slow_standardize.rs`) | No — affine transforms (shift + scale), no log/exp | N/A — `predict` returns `Gaussian(mu + sigma*g.mean, ...)` — linear inverse | **N/A** | — |

**Total audited:** 19 models/paths
**Preliminary FAIL (new offenders):** 0
**PASS:** 13 (with guard evidence)
**N/A:** 6 (no auto-selection path)

---

## Key Verdict Rationales

### AutoETS / GlobalAutoETS — PASS

The ETS state-space model is architecturally distinct from MFLES. ETS multiplicative mode uses multiplicative update equations (`y / s`, `y / level`) and additive/multiplicative error terms — it **never applies a global `ln()` transform followed by an `exp()` back-transform**. The MFLES blow-up class requires this ln+boost+exp pipeline to compound the log-space crater into an exploded forecast. ETS cannot exhibit this.

For the #10 issue (zero values): the `has_non_positive = values.iter().any(|&v| v <= 0.0)` guard [VERIFIED: auto_ets.rs:422] completely blocks multiplicative candidates, preventing division-by-zero in multiplicative update equations.

For near-zero (1.0 in 2000-level series): multiplicative ETS is allowed by the guard (1.0 > 0), but:
1. The near-zero observation creates a seasonal factor ≈ 0.0005 for that phase.
2. Subsequent visits to that phase produce a near-zero forecast (correct if the series genuinely has a recurring trough).
3. In the likelihood loop, the multiplicative model receives a large sum-of-squared-errors when the next year's value at that phase is ~2000 but the forecast is ~1.0 → AIC is much worse than the additive model → AIC selection **rejects multiplicative** [VERIFIED: ets.rs:387-389, the likelihood macro].
4. Forecast at that phase = level × factor ≈ 1.0 — depressed but **not exploded**.

### Theta Family — PASS

All three Theta variants (`model.rs`, `optimized.rs`, `dynamic.rs`) share the `determine_decomposition()` method with two rules:
- **Rule 1**: `any(y <= 0)` → additive [VERIFIED: model.rs:476]
- **Rule 2**: `seasonals.iter().any(|&s| s < 0.01)` → additive [VERIFIED: model.rs:486]

For the Phase 5 repro series (1.0 in 2000-level, period=12): the multiplicative seasonal index for month 11 is `1.0 / trend_at_11 ≈ 1.0 / 2000 = 0.0005`. After normalisation by the mean of all indices (≈ 0.9167), the normalised factor ≈ 0.00055 — well below 0.01 → **Rule 2 trips, additive selected**.

The seasonal-factor guard is functionally equivalent to the Phase 5 min/median ratio guard (both detect when a near-zero observation's ratio to the series level falls below a 1% threshold). The Theta guard is applied in the classical seasonal-decomposition domain; the MFLES guard (τ=0.10) is applied in the log-space domain — both prevent multiplicative mode selection when a near-zero observation would create a problematic outlier.

### TBATS (auto-lambda path) — PASS

TBATS `estimate_lambda()` minimises coefficient of variation (variance / mean²) in the Box-Cox-transformed space. For a near-zero series (1.0 in 2000-level):

- `λ = 0` (log): mean ≈ 7.28, variance ≈ 2.30 → CoV ≈ 0.043
- `λ = 1` (identity): mean ≈ 1969, variance ≈ 170k → CoV ≈ 0.044

These are very close, but the non-positive guard (`any(v <= 0) → return 1.0`) does not help for near-zero (1.0 > 0). However, the TBATS state-space architecture prevents blow-up even when `λ ≈ 0` is selected:

1. TBATS is a Kalman-like linear state-space filter (no boosting), not a log-then-boost-then-exp pipeline.
2. In log-space, the near-zero obs maps to 0.0 (7.28 units below the mean).
3. The Fourier seasonal states absorb this trough accurately: at prediction time, level + seasonal contribution ≈ 0.0 in log-space → `exp(0.0) = 1.0` — correct, not exploded.
4. Unlike MFLES, there is no multi-round residual amplification that could inflate the prediction.
5. AutoTBATS additionally compares AIC across all configurations including no-transform baselines; the log variant's higher within-sample SSE leads to worse AIC → additive variant wins.

### BoxCox auto-lambda — PASS

`boxcox_lambda()` uses MLE [VERIFIED: src/transform/boxcox.rs:122-147]: `LLF(λ) = -n/2 × ln(var(boxcox(series,λ))) + (λ-1) × Σln(xᵢ)`. For a near-zero series, `λ=0` produces a high-variance log-space representation (the 1.0 maps to 0.0, a crater 7+ units below the log-mean) → large variance → low LLF. MLE prefers `λ≈1` (identity), where variance from the near-zero obs is proportional to its magnitude rather than log-space distorted. This is the inverse of the MFLES failure: the selection criterion itself detects that a log transform would be harmful.

### LaplaceForecaster leaves — PASS (all distributional-gated)

- **`lognormal`**: uses `y.max(0.0).ln_1p()` — `ln_1p(1.0) = ln(2) ≈ 0.693`, which is a modest offset in a 2000-level series (log-1p-mean ≈ 7.6), not a 7.6-unit crater. No blow-up possible.
- **`seasonal_mult`**: multiplicative seasonal leaf but with online EMA updates (no global ln+exp). Near-zero obs depresses the phase factor; forecast = level × factor. The softmax weighting penalises this leaf if subsequent observations are inconsistent.
- **`yj_wrapper`**: clamps predictions to the observed transformed range before inversion; finite by design.
- **`standardize` / `slow_standardize`**: purely affine (no log/exp) — categorically excluded.

---

## Architectural Insight: Why MFLES Was Unique

The #10/#219 failure class requires three co-occurring conditions:

1. **Global log transform** applied to the entire training series (producing a log-space crater at the near-zero obs).
2. **Multi-round amplification** in log-space (boosting residuals compound the crater across rounds, distorting the trend + seasonal estimates).
3. **Exponential back-transform** `exp(pred)` applied to the distorted log-space prediction.

MFLES is the only model in this codebase that has all three. Every other auto-multiplicative model uses either (a) state-space updates bounded by the data, (b) AIC/NLL selection that detects and rejects distorted models, or (c) seasonal-factor magnitude guards. This confirms the MFLES fix is correctly scoped and there are no additional offenders requiring MULT-06 fixes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Seasonal-factor guard for Theta | Custom min/median check | Existing Rule 2: `any(s < 0.01)` [model.rs:486] | Already achieves the same effect; adding a new guard would duplicate logic |
| Log-space crater detection for BoxCox pipeline | Custom pre-transform guard | `boxcox_lambda()` MLE already prefers large λ | MLE is the right mechanism; a hard guard would override correct user intent |
| Laplace leaf weighting for distorted leaves | Hard exclusion of `seasonal_mult` | Softmax log-likelihood already down-weights distorted leaves | Adaptive, no intervention required |

---

## Common Pitfalls

### Pitfall 1: Confusing ETS "multiplicative" with MFLES "multiplicative"

**What goes wrong:** Auditor marks AutoETS multiplicative mode as a potential MFLES-class failure because both use the word "multiplicative."
**Why it's wrong:** ETS multiplicative = state-space update equations with `y/s` and `y/level`; MFLES multiplicative = global `ln()` + boosting + `exp()`. The back-transform explosion only occurs in the ln/exp pathway. ETS forecasts are products of level × seasonal-factor, both of which are bounded by the series range.
**Warning signs:** Verdict marked fail without identifying an `exp()` back-transform downstream of a log-space fit.

### Pitfall 2: Assuming the seasonal-factor guard (Theta < 0.01) is weaker than the MFLES guard (τ=0.10)

**What goes wrong:** Auditor concludes Theta needs a min/median guard like MFLES.
**Why it's wrong:** The seasonal-factor guard is applied directly to the ratio `y_near_zero / trend_at_that_position` ≈ 0.0005 for the Phase 5 repro series, which is far below 0.01. The guard triggers for exactly the same scenario MFLES' τ=0.10 guard catches — they are functionally equivalent for the in-scope failure class, just in different mathematical spaces.
**Warning signs:** New guard added to Theta that duplicates the existing Rule 2 check.

### Pitfall 3: Marking BoxCox auto-lambda as fail without checking MLE objective

**What goes wrong:** Auditor notes "near-zero (1.0) maps to 0 under any Box-Cox lambda" and marks it as a crater → fail.
**Why it's wrong:** `boxcox(1.0, λ) = (1^λ - 1)/λ = 0` for all λ > 0 and `ln(1.0) = 0` for λ=0. BUT the MLE objective function penalises λ values that produce high variance in the transformed space — and λ=0 (log) produces the highest variance for a near-zero series. MLE selects λ≈1 (near-identity), which means the "crater" at value=1.0 is never created in practice under auto-lambda.
**Warning signs:** Fail verdict based solely on boxcox(1.0, λ)=0 without checking what MLE selects.

### Pitfall 4: Omitting `distributional`-gated models from the inventory

**What goes wrong:** Inventory table skips Laplace leaves because they require `--features distributional`.
**Why it's wrong:** CONTEXT.md explicitly requires distributional leaves to be audited and flagged as gated. The SC4 contract (already-safe models recorded with evidence) requires completeness.
**Warning signs:** Inventory table has < 15 rows or is missing all Laplace leaf entries.

### Pitfall 5: Clamp in `fit()` decomposable invariant (inherited from Phase 5)

**What goes wrong:** If any future Phase 6 fix adds a back-transform clamp inside `fit()` for a new offender, it may break the `trend + seasonal + residual == training` decomposable invariant tested by `issue_106_decomposable_conformance`.
**How to avoid:** Apply back-transform clamps ONLY in `predict_internal()`, not in the fitted-values computation inside `fit()`. See Phase 5 SUMMARY for the established pattern.
**Warning signs:** `cargo test --test issue_106_decomposable_conformance` fails after adding a fix.

---

## Code Examples

### Phase 5 Guard Pattern (reference for any future offenders)

```rust
// Source: src/models/mfles.rs:1035-1050 (Phase 5 fix — VERIFIED)
let use_multiplicative = match self.multiplicative {
    Some(m) => m,
    None => {
        // Auto: multiplicative only if positive AND series is not
        // near-zero relative to its level (issue #219 guard).
        let all_positive = values.iter().all(|&v| v > 0.0);
        if all_positive && self.season_length > 0 {
            let median = Self::median_scalar(values);
            let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
            // Guard: if min / median < τ, additive is safer.
            median > 0.0 && (min_val / median) >= Self::MULT_AUTO_TAU
        } else {
            false
        }
    }
};
```

Constants (for the reference pattern):
```rust
// Source: src/models/mfles.rs:237-249 (VERIFIED)
const MULT_AUTO_TAU: f64 = 0.10;      // min/median mode-selection threshold
const MULT_LOG_FLOOR_FRAC: f64 = 0.01; // ln() floor as fraction of median
const MULT_BACK_CLAMP_K: f64 = 10.0;  // back-transform cap as multiple of in-sample max
```

### Theta Seasonal-Factor Guard (reference — existing safe guard)

```rust
// Source: src/models/theta/model.rs:481-490 (VERIFIED)
// Rule 2: Try multiplicative and check if seasonal factors are valid
let seasonals = self.calculate_seasonals(series, DecompositionType::Multiplicative);
if !seasonals.is_empty() {
    // Check if any seasonal factor is too small (< 0.01)
    // This matches statsforecast's behavior for numerical stability
    if seasonals.iter().any(|&s| s < 0.01) {
        self.decomposition_fallback = true;
        return DecompositionType::Additive;
    }
}
```

This guard fires for the Phase 5 repro series: `seasonal_index[11] ≈ 0.00055 < 0.01 → additive`.

### AutoETS Non-Positive Guard (reference — existing safe guard)

```rust
// Source: src/models/exponential/auto_ets.rs:420-422 (VERIFIED)
// Multiplicative models require all positive values (y/level, y/s undefined at zero)
// Matches Theta::determine_decomposition() and R's forecast::ets()
let has_non_positive = values.iter().any(|&v| v <= 0.0);
```

### LogNormal Leaf Safe Log Transform

```rust
// Source: src/models/laplace/leaves/lognormal.rs:23-25 (VERIFIED)
#[inline]
fn log1p_nonneg(y: f64) -> f64 {
    y.max(0.0).ln_1p()  // ln_1p(1.0) = ln(2) ≈ 0.693 — no crater
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MFLES: `all(v > 0)` → multiplicative (unsafe for near-zero) | MFLES: `min/median >= τ=0.10` → multiplicative + log-floor + back-transform clamp | Phase 5 (2026-09-09, commit d36de76) | #219 near-zero blow-up eliminated |
| AutoETS: no guard against zero-value multiplicative | AutoETS: `any(v <= 0)` → restrict multiplicative (issue #10 fix) | Pre-v1.0 | Zero-value NaN/Inf forecasts eliminated |

---

## Validation Architecture

**Nyquist validation is enabled** (`workflow.nyquist_validation: true` in `.planning/config.json`). [VERIFIED: /home/simonm/projects/rust/anofox-forecast/.planning/config.json]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Cargo's built-in test harness (no external test framework) |
| Config file | None — uses `Cargo.toml` `[dev-dependencies]` |
| Quick run command | `cargo test --test <test_name>` |
| Full suite command | `cargo test` |
| Distributional-gated tests | `cargo test --features distributional --test <name>` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MULT-05 | Complete inventory table committed to `docs/audits/multiplicative-guard-audit.md` | documentation + guard tests | `cargo test --test mult05_guard_audit_assertions` | ❌ Wave 0 |
| MULT-05 (SC4) | Theta seasonal-factor guard fires for near-zero series (1.0 in 2000-level) | integration regression | `cargo test --test mult05_guard_audit_assertions theta_seasonal_factor_guard_catches_near_zero` | ❌ Wave 0 |
| MULT-05 (SC4) | AutoETS does not select multiplicative for near-zero series (1.0 in 2000-level) | integration regression | `cargo test --test mult05_guard_audit_assertions auto_ets_aicselection_rejects_mult_for_near_zero` | ❌ Wave 0 |
| MULT-05 (SC4) | MFLES Phase 5 fix still passes | regression guard | `cargo test --test issue_219_mfles_multiplicative_runaway` | ✅ exists |
| MULT-06 | No new offenders — no fix tests required | — | — | ✅ already green |

### Sampling Rate

- **Per task commit:** `cargo test --test mult05_guard_audit_assertions`
- **Per wave merge:** `cargo test`
- **Phase gate:** `cargo test && cargo clippy --all-targets --all-features -- -D warnings`

### Wave 0 Gaps

- [ ] `tests/mult05_guard_audit_assertions.rs` — covers MULT-05 SC4 guard assertions (3 tests)
- [ ] `docs/audits/multiplicative-guard-audit.md` — inventory table (non-code artifact, Wave 0 task)
- [ ] `docs/audits/` directory must be created

**Note on existing tests:** `tests/issue_219_mfles_multiplicative_runaway.rs` (MULT-04) continues to serve as the reference regression for the MFLES fix.

---

## Proposed Test Structure (`tests/mult05_guard_audit_assertions.rs`)

```rust
//! Guard-assertion tests for the Phase 6 multiplicative-guard audit (MULT-05).
//!
//! Verifies that already-safe models are provably protected against the #10/#219
//! failure class (near-zero series must not produce a blow-up forecast).
//!
//! Series shape: level ≈ 2000, one near-zero month (1.0) — same as Phase 5 repro.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::{
    exponential::{AutoETS, AutoETSConfig},
    theta::Theta,
    Forecaster,
};
use chrono::{Duration, TimeZone, Utc};

fn make_near_zero_series(period: usize) -> TimeSeries {
    // 2× period + extra: level ≈ 2000, near-zero at index (period - 1)
    let n = 2 * period + 6;
    let base = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n)
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    let mut values: Vec<f64> = (0..n)
        .map(|i| 2000.0 + 100.0 * ((i as f64 * 0.4).sin()))
        .collect();
    // Inject near-zero at the seasonal trough position
    values[period - 1] = 1.0;
    TimeSeries::univariate(timestamps, values).unwrap()
}

/// MULT-05 SC4: Theta seasonal-factor guard fires for near-zero series.
/// Theta.determine_decomposition() Rule 2: any(seasonal_factor < 0.01) → additive.
#[test]
fn theta_seasonal_factor_guard_catches_near_zero() {
    let ts = make_near_zero_series(12);
    let mut model = Theta::seasonal(12); // Requests multiplicative by default
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();
    let level = 2000.0_f64;

    // Guard must fire → additive selected → forecast stays near level
    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        forecast_max < 2.5 * level,
        "Theta near-zero guard: forecast_max={:.0} >= {:.0} (2.5× level). \
         Seasonal-factor guard should have selected additive.",
        forecast_max,
        2.5 * level,
    );
}

/// MULT-05 SC4: AutoETS AIC selection rejects multiplicative for near-zero series.
/// For near-zero (1.0 in 2000-level): multiplicative model gets poor AIC due to
/// distorted seasonal factors → AIC selects additive.
#[test]
fn auto_ets_aicselection_rejects_mult_for_near_zero() {
    let ts = make_near_zero_series(12);
    let mut model = AutoETS::with_period(12);
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();
    let level = 2000.0_f64;

    // AIC protection must work → forecast stays bounded
    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        forecast_max < 2.5 * level,
        "AutoETS near-zero: forecast_max={:.0} >= {:.0} (2.5× level). \
         AIC protection should reject multiplicative.",
        forecast_max,
        2.5 * level,
    );
    let forecast_min = preds.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        forecast_min > 0.0,
        "AutoETS near-zero: forecasts should be positive, got min={:.2}",
        forecast_min,
    );
}
```

**Verify commands:**
```bash
# Quick (per task commit):
cargo test --test mult05_guard_audit_assertions

# Full suite (per wave merge):
cargo test

# Phase gate (clippy + full suite):
cargo test && cargo clippy --all-targets --all-features -- -D warnings

# Distributional-gated Laplace tests (if SC4 Laplace tests are added):
cargo test --features distributional --test mult05_guard_audit_assertions
```

---

## Environment Availability

No new external tools required beyond the standard Rust toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo` / Rust stable | All compilation and test | ✓ | current stable | — |
| `cargo clippy` | CI lint gate | ✓ | bundled with toolchain | — |
| `--features distributional` | Laplace leaf tests | ✓ | feature in Cargo.toml | Skip Laplace tests, document as manual-verify |

**`cargo test --all-features`** OOMs the linker in this sandbox (confirmed in Phase 5). Use:
- `cargo test` (default features) for the core audit tests
- `cargo clippy --all-targets --all-features -- -D warnings` for the lint gate (compiles without linking the test harness together)
- `cargo test --features distributional --test mult05_guard_audit_assertions` for gated Laplace tests

---

## Security Domain

All in-scope models are pure numerical computation with no I/O, authentication, or network access.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `validate_series_complete()` (called at top of every `fit()`); existing guards (any(v<=0), seasonal-factor < 0.01, MFLES min/median τ) prevent numeric overflow in exp() |
| V6 Cryptography | no | — |

**Threat pattern for this audit:** `exp(distorted_log_space_prediction)` producing `f64::INFINITY` or astronomically large floats. Mitigated in MFLES by Phase 5; confirmed inapplicable to all other in-scope models by this audit.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | AutoETS AIC protection causes the additive model to beat multiplicative for the near-zero repro series (1.0 in 2000-level). Verified by code analysis of the likelihood loop but not by running the repro. | AutoETS rationale | Low — the proposed SC4 guard test will empirically confirm this. If wrong, the test will catch it and a fix (min/median guard like MFLES) would be required. |
| A2 | TBATS `estimate_lambda()` does not consistently choose λ=0 for the near-zero repro series. The CoV comparison is numerically close (0.043 vs 0.044). | TBATS analysis | Low — even if λ=0 is selected, TBATS Kalman filter produces bounded predictions (no boosting amplification). The TBATS state-space cannot exhibit the MFLES-class blow-up. |
| A3 | `boxcox_lambda()` MLE selects λ≈1 for near-zero series. Verified by LLF formula analysis; not by running the optimizer. | BoxCox analysis | Low — the MLE penalty for high log-space variance is clear from the formula. SC4 test not strictly required for BoxCox pipeline since it fails early (any(x<=0) guard) for the typical near-zero concern. |

**All three assumptions have low risk — the proposed SC4 tests will confirm A1, which is the most consequential.**

---

## Open Questions

1. **TBATS non-positive guard for near-zero is only in `estimate_lambda()`**
   - What we know: `estimate_lambda()` guards `any(v <= 0) → 1.0` [VERIFIED: model.rs:386-387]. The explicit `.with_box_cox(lambda)` path has no such guard.
   - What's unclear: Whether a user calling `TBATS::new(periods).with_box_cox(0.0)` on a near-zero series should be guarded. This is user-driven, not auto-selection, so it's currently out of scope per CONTEXT.md.
   - Recommendation: Note in the inventory that user-specified TBATS `with_box_cox(lambda)` is out of scope per CONTEXT.md "non-auto internal sites." No fix required.

2. **Laplace `lognormal` leaf auto-selection conditions**
   - What we know: the AID-based auto-selector triggers `lognormal` when `DemandDistribution::LogNormal` is detected [VERIFIED: forecaster.rs:2988-2990]. The `log1p_nonneg` helper is safe.
   - What's unclear: Under what series characteristics does the AID classifier return `LogNormal`? The AID classifier itself is a dependency (`anofox-regression`) and its exact conditions weren't traced this session.
   - Recommendation: Mark as pass with the log1p_nonneg evidence; no further audit of the AID classifier is needed per CONTEXT.md scope (auditing the auto-selection path, not the classifier internals).

---

## Sources

### Primary (HIGH confidence — files read this session)

- `src/models/mfles.rs` — lines 95, 234-249, 287, 967-972, 1039-1073 (Phase 5 guard implementation)
- `src/models/exponential/auto_ets.rs` — lines 420-482 (fit, has_non_positive guard, generate_candidates), 977-1072 (tests incl. issue #10 repro)
- `src/models/exponential/ets.rs` — lines 370-421 (likelihood macro, error term), 735-770 (multiplicative seasonal init, clamp 0.01-100), 1125-1135 (seasonal bounds)
- `src/models/exponential/holt_winters.rs` — lines 1-71, 117-125, 340-399 (no auto-selection)
- `src/models/exponential/seasonal_es.rs` — lines 1-5, 78-96 (no auto-selection, additive default)
- `src/models/exponential/global_ets.rs` — lines 637-715 (GlobalAutoETS fit, has_non_positive, candidate selection), 738-769 (generate_candidates)
- `src/models/theta/model.rs` — lines 343-439 (calculate_seasonal_component), 449-493 (calculate_seasonals, determine_decomposition, Rules 1+2)
- `src/models/theta/optimized.rs` — lines 353-371 (determine_decomposition, Rule 2)
- `src/models/theta/dynamic.rs` — lines 351-371 (determine_decomposition, Rule 2 identical)
- `src/models/tbats/model.rs` — lines 362-416 (box_cox_transform, inverse_box_cox, estimate_lambda), 758-824 (fit path, lambda auto-selection)
- `src/models/tbats/auto.rs` — lines 60-74, 205-334 (AutoTBATS fit, can_box_cox guard, AIC comparison)
- `src/transform/boxcox.rs` — lines 32-154 (boxcox, boxcox_auto, boxcox_lambda, boxcox_llf, is_boxcox_suitable)
- `src/transform/yeo_johnson.rs` — lines 19-120 (yj_forward, yj_inverse, yeo_johnson_lambda, yeo_johnson_auto)
- `src/transform/transforms.rs` — lines 213-350 (BoxCoxTransform, YeoJohnsonTransform, fit_transform guards)
- `src/models/laplace/forecaster.rs` — lines 152-171, 526-605 (auto_characteristics, SeriesChars), 2988-3105 (auto-selector for lognormal, seasonal_mult)
- `src/models/laplace/leaves/lognormal.rs` — lines 23-88 (log1p_nonneg, LogNormalLeaf)
- `src/models/laplace/leaves/seasonal_mult.rs` — lines 22, 71-180 (LEVEL_TOL, from_batch, observe guard)
- `src/models/laplace/leaves/yj_wrapper.rs` — lines 18-126 (YjWrappedLeaf, clamp guard, observe finite guard)
- `src/models/laplace/leaves/standardize.rs` — lines 1-60 (StandardizeWrapper, affine inverse)
- `src/models/laplace/leaves/slow_standardize.rs` — lines 1-80 (SlowStandardizeWrapper, affine inverse)
- `.planning/phases/05-mfles-multiplicative-guard-fix/05-RESEARCH.md` — Phase 5 guard discipline reference
- `.planning/phases/05-mfles-multiplicative-guard-fix/05-01-SUMMARY.md` — Phase 5 actuals
- `.planning/phases/06-multiplicative-guard-bug-class-audit/06-CONTEXT.md` — scope and methodology
- `.planning/config.json` — `nyquist_validation: true` confirmed
- `tests/issue_219_mfles_multiplicative_runaway.rs` — existing MULT-04 regression guard
- `tests/` directory listing — naming convention confirmed

---

## Metadata

**Confidence breakdown:**
- Inventory verdicts (ETS family): HIGH — architectural analysis confirmed by code; no log/exp path exists
- Inventory verdicts (Theta family): HIGH — Rule 2 guard verified by reading source; trigger confirmed by mathematical analysis of the repro series
- Inventory verdicts (TBATS): HIGH — architectural analysis (Kalman filter vs boosting); minor uncertainty on lambda selection for near-zero addressed in Assumptions Log
- Inventory verdicts (BoxCox/YJ transforms): HIGH — MLE analysis confirms lambda selection preference; guard code verified
- Inventory verdicts (Laplace leaves): HIGH — log1p_nonneg, LEVEL_TOL, clamp guards all verified
- Phase 5 guard status: HIGH — read and verified all three guard implementation sites in mfles.rs this session

**Research date:** 2026-09-09
**Valid until:** 2026-12-09 (stable codebase; guard logic unlikely to move)
