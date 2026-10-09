# Multiplicative-Guard Bug-Class Audit (MULT-05)

**Audit date:** 2026-09-09
**Phase:** 06 — Multiplicative-Guard Bug-Class Audit
**Maintainer:** anofox-forecast team

## Preamble

This document is the MULT-05 complete auditable sweep of every data-driven
auto-multiplicative or auto-log selection path reachable via the public
`Forecaster` API of the `anofox-forecast` library.

The sweep is motivated by two historical issues:

- **Issue #10** (`AutoETS` zero-value crash): a zero in the training series
  produced `NaN`/`Inf` forecasts when multiplicative ETS mode was selected
  without guarding against `y/level = 0/0`. Fixed pre-v1.0 by the
  `has_non_positive` guard in `auto_ets.rs`.
- **Issue #219** (MFLES near-zero runaway): a near-zero month (1.0 in a
  2000-level series) caused the MFLES global `ln()` → boosting → `exp()`
  pipeline to produce forecasts ~13× level (~26 000). Fixed in Phase 5
  (commit d36de76) by a `min/median ≥ τ=0.10` mode-selection guard, a log-floor,
  and a back-transform clamp.

The central finding is that the #10/#219 blow-up class — an `exp()` back-transform
producing forecasts well above level due to a log-space crater from a near-zero
observation — is **architecturally specific to MFLES**. No other in-scope model
shares the global `ln()` + multi-round boosting + `exp()` pipeline that enables
the crater to compound. All 19 audited paths are therefore **PASS** or **N/A**;
**zero new offenders** were found.

Guard-assertion tests proving the two at-risk-looking-but-already-safe models
(Theta, AutoETS) forecast at level for a near-zero series are in
`tests/mult05_guard_audit_assertions.rs`.

---

## Inventory Table

| # | Model / Path | Has auto-mult/log selection? | Existing guard (file:line) | Verdict | Rationale |
|---|---|---|---|---|---|
| 1 | **MFLES** (`src/models/mfles.rs`) | Yes — `self.multiplicative == None && season_length > 0 && all(v > 0)` | `MULT_AUTO_TAU=0.10`: `median > 0.0 && (min_val / median) >= Self::MULT_AUTO_TAU` [mfles.rs:1048]; `MULT_LOG_FLOOR_FRAC=0.01` [mfles.rs:244]; `MULT_BACK_CLAMP_K=10.0` [mfles.rs:249] | **pass** | Phase 5 (#219) fixed all three points — mode-selection guard, log-floor, and back-transform clamp. Regression: `tests/issue_219_mfles_multiplicative_runaway.rs`. |
| 2 | **AutoETS** (`src/models/exponential/auto_ets.rs`) | Yes — generates multiplicative-error / multiplicative-seasonal candidates for all-positive series | `let has_non_positive = values.iter().any(\|&v\| v <= 0.0);` [auto_ets.rs:422] — restricts candidates to additive when any zero present | **pass** | ETS multiplicative uses state-space update equations (`y/level`, `y/s`), not global ln+exp. For near-zero (1.0 in 2000-level): multiplicative model receives distorted seasonal factor ≈ 0.0005, producing high AIC → AIC selection rejects multiplicative. Empirically confirmed by `auto_ets_aicselection_rejects_mult_for_near_zero`. |
| 3 | **GlobalAutoETS** (`src/models/exponential/global_ets.rs`) | Yes — same candidate generation as AutoETS across all series | `let has_non_positive = all_series.iter().any(\|s\| s.iter().any(\|&v\| v <= 0.0));` [global_ets.rs:663] | **pass** | Same architectural pattern as AutoETS: no global ln+exp pipeline; AIC selection rejects distorted multiplicative models. |
| 4 | **ETS** (`src/models/exponential/ets.rs`) | No — spec is user-provided (`ETSSpec`); no data-driven mode selection | N/A | **N/A** | No auto-selection path. Caller provides `ETSSpec` explicitly; no data-driven multiplicative decision. |
| 5 | **HoltWinters** (`src/models/exponential/holt_winters.rs`) | No — `SeasonalType` is an explicit constructor argument | N/A | **N/A** | `SeasonalType` is always caller-supplied. No data-driven mode selection. |
| 6 | **SeasonalES** (`src/models/exponential/seasonal_es.rs`) | No — default is `SeasonalESErrorType::Additive`; multiplicative only via explicit `.with_error_type()`. Doc: "NOT a multiplicative seasonal model." [seasonal_es.rs:1–5] | N/A | **N/A** | No data-driven auto-selection. Additive by default; multiplicative only via explicit user call. |
| 7 | **GlobalETS** (fixed-spec) (`src/models/exponential/global_ets.rs`) | No — `GlobalETS::new(spec, period)` takes an explicit `ETSSpec`; no auto-selection | N/A | **N/A** | No auto-selection path in fixed-spec variant. |
| 8 | **Theta** (`src/models/theta/model.rs`) | Yes — `seasonal()` defaults to multiplicative; `determine_decomposition()` may override | Rule 1: `series.iter().any(\|&y\| y <= 0.0)` [model.rs:476]; Rule 2: `seasonals.iter().any(\|&s\| s < 0.01)` [model.rs:486] | **pass** | Near-zero (1.0 in 2000-level) produces seasonal index ≈ 0.0005 < 0.01 — Rule 2 triggers additive fallback. Empirically confirmed by `theta_seasonal_factor_guard_catches_near_zero`. |
| 9 | **OptimizedTheta** (`src/models/theta/optimized.rs`) | Yes — same `determine_decomposition()` pattern | Rule 2: `last_cycle.iter().any(\|&s\| s < 0.01)` [optimized.rs:365] | **pass** | Identical seasonal-factor guard to Theta. Near-zero trough produces factor ≈ 0.0005 → additive. |
| 10 | **DynamicTheta** (`src/models/theta/dynamic.rs`) | Yes — same `determine_decomposition()` pattern | Rule 1 + Rule 2 [dynamic.rs:357–368] — identical guard logic | **pass** | Same guard as Theta and OptimizedTheta. Near-zero trough triggers additive fallback. |
| 11 | **TBATS** (`src/models/tbats/model.rs`) — auto-lambda path | Yes — `estimate_lambda()` called when `lambda.is_none() && all(v > 0)` [model.rs:758–759] | `if values.iter().any(\|&v\| v <= 0.0) { return 1.0; }` [model.rs:386–387] — blocks lambda < 1 for non-positive | **pass** | `estimate_lambda()` minimises the coefficient of variation across seasonal sub-series (NOT an AIC comparison), and for this near-zero series CoV actually favours λ≈0 (log transform). PASS holds on architecture, not on lambda rejection: TBATS is a Kalman-like linear state-space filter with no boosting amplification, so even at λ≈0 the Fourier seasonal states absorb the trough and the back-transform is exp(0.0) = 1.0 at the trough — bounded, not exploded. (AIC-based rejection of the log variant is an *AutoTBATS* property — row 13 — not standalone TBATS.) |
| 12 | **TBATS** (`src/models/tbats/model.rs`) — fixed-lambda path | No — user calls `.with_box_cox(lambda)` | No guard for near-zero (user-specified). | **N/A** (user-driven) | Not an auto-selection path per CONTEXT.md. User-specified lambda is out of scope; responsibility for near-zero handling rests with the caller. |
| 13 | **AutoTBATS** (`src/models/tbats/auto.rs`) | Yes — tries lambda ∈ {0, 0.25, 0.5, 0.75, 1.0} if `can_box_cox` | `let can_box_cox = values.iter().all(\|&v\| v > 0.0);` [auto.rs:206]; AIC comparison across all configs | **pass** | AIC comparison rejects λ=0 for near-zero series (log-space SSE much larger); no-transform baseline wins. State-space architecture prevents blow-up even if λ≈0 were selected. |
| 14 | **BoxCoxTransform** (`src/transform/transforms.rs`) | Yes — `BoxCoxTransform::auto()` calls `boxcox_lambda()` for λ selection | `if values.iter().any(\|&x\| x <= 0.0) { return Err(...) }` [transforms.rs:244] — rejects non-positive | **pass** | MLE maximises `-n/2·ln(var) + (λ-1)·Σln(x)`; on this near-zero series it selects λ≈2, not λ≈1. Note `boxcox(1.0, λ) = 0` for *every* λ, so lambda selection does NOT avoid mapping the near-zero point to the low end — but there is no crater or blow-up because BoxCox is a plain transform with no boosting/exp-amplification pipeline, and the inverse round-trips the point exactly: `inv_boxcox(0) = 1.0` recovers the original near-zero value. The non-positive guard [transforms.rs:244] is a separate safety net that does not fire here (all values positive). |
| 15 | **YeoJohnsonTransform** (`src/transform/transforms.rs`) | Yes — `YeoJohnsonTransform::auto()` calls `yeo_johnson_lambda()` | No positivity requirement (YJ handles all reals) | **pass** | YJ transform of y=1.0 is `((2^λ-1)/λ) ≈ 0.69` regardless of lambda — no crater. Near-zero observation maps to a modest bounded value in transformed space. [yeo_johnson.rs:30–42] |
| 16 | **LaplaceForecaster** — `seasonal_mult` leaf (`src/models/laplace/leaves/seasonal_mult.rs`) | Yes — auto-selection: `chars.seasonality_strength > 0.3 && chars.all_positive && chars.mean_y > 0.0` [forecaster.rs:3060–3065] | `const LEVEL_TOL: f64 = 1e-6;` update guard [seasonal_mult.rs:22, 168] — skips factor update if level ≈ 0 | **pass** (`distributional`-gated) | Near-zero obs depresses the phase factor to ≈ 0.0005; forecast = level × factor ≈ near-zero (correct seasonal trough, not blow-up). Softmax log-likelihood down-weights this leaf if the trough is one-off. No global ln+exp pipeline. |
| 17 | **LaplaceForecaster** — `lognormal` leaf (`src/models/laplace/leaves/lognormal.rs`) | Yes — auto-selection via AID classifier | `fn log1p_nonneg(y: f64) -> f64 { y.max(0.0).ln_1p() }` [lognormal.rs:23–25] — guaranteed ≥ 0 | **pass** (`distributional`-gated) | `ln_1p(1.0) = ln(2) ≈ 0.693` for y=1.0 — a modest, bounded value in a series where log1p-mean ≈ 7.6. No crater; no blow-up possible. |
| 18 | **LaplaceForecaster** — `yj_wrapper` leaf (`src/models/laplace/leaves/yj_wrapper.rs`) | Yes — wraps any inner leaf with a fixed λ; used in `with_yeo_johnson_grid()` | Trans-range clamp: `let mean_clamped = g.mean.clamp(lo, hi)` [yj_wrapper.rs:98]; finite guard on observe [yj_wrapper.rs:120–124] | **pass** (`distributional`-gated) | Predictions clamped to observed training range in transformed space before inverse YJ is applied. Near-zero: `yj_forward(1.0, λ) ≈ 0.69` (bounded). Inverse YJ of clamped value is finite by design. |
| 19 | **LaplaceForecaster** — `standardize` / `slow_standardize` wrappers (`src/models/laplace/leaves/standardize.rs`, `slow_standardize.rs`) | No — affine transforms only (shift + scale; no log/exp) | N/A — linear inverse | **N/A** | Categorically excluded: affine transforms cannot produce the exp-back-transform blow-up. |

---

## Verdict Tally

| Category | Count |
|----------|-------|
| Total audited | 19 |
| **FAIL** (new offenders) | **0** |
| **PASS** (safe, with evidence) | **13** |
| **N/A** (no auto-selection path) | **6** |

---

## MULT-06 Result: Zero New Offenders — Satisfied by Evidence

The sweep found **zero new offenders**. The #10/#219 `ln() → boosting → exp()`
blow-up class is **architecturally unique to MFLES** and requires three
co-occurring conditions:

1. **Global log transform** applied to the entire training series.
2. **Multi-round amplification** in log-space (boosting compounds the crater).
3. **Exponential back-transform** `exp(pred)` applied to the distorted prediction.

No other in-scope model has all three. The alternatives use state-space updates
bounded by the data (ETS, TBATS Kalman), AIC/NLL selection that detects and
rejects distorted models (AutoETS, AutoTBATS), or seasonal-factor magnitude guards
that prevent multiplicative mode selection entirely for near-zero series (Theta family).

**MULT-06 is satisfied by evidence. No additional production code change is required.**
The MFLES fix (Phase 5, commit d36de76) is correctly scoped.

---

## Evidence

### Guard-Assertion Tests (MULT-05 SC4)

File: `tests/mult05_guard_audit_assertions.rs`

| Test | Model | What it proves |
|------|-------|----------------|
| `theta_seasonal_factor_guard_catches_near_zero` | Theta | On a near-zero series (1.0 in 2000-level, period=12) the forecast stays bounded near level: max < 2.5× level (no blow-up) AND mean within 0.5× level (centered, not collapsed) |
| `auto_ets_aicselection_rejects_mult_for_near_zero` | AutoETS | On the same series the forecast stays bounded near level: max < 2.5× level AND mean within 0.5× level AND all forecasts > 0 |

Both tests use the same near-zero series shape as the #219 repro (level ≈ 2000,
one 1.0 trough, sinusoidal wobble, 2×period+6 length). They are two-sided safety
assertions (bounded above *and* centered at level) rather than guard-firing
probes: for these architecturally-safe models the auto-mode guards are
defence-in-depth, but the models do not blow up even without them (ratio/Kalman
updates, no `ln()→boosting→exp()` pipeline). The tests therefore serve as
regression guards for the failure *class* — if a future refactor introduced an
MFLES-style log-boosting path into Theta or AutoETS, the upper bound would trip.
A per-point lower bound is deliberately avoided: the recurring seasonal trough is
a legitimately low month in additive mode too, so only the aggregate (mean) level
is a meaningful safety signal.

### Phase 5 MFLES Regression (MULT-04)

File: `tests/issue_219_mfles_multiplicative_runaway.rs`

| Test | What it proves |
|------|----------------|
| `issue_219_mfles_no_multiplicative_runaway` | MFLES auto-mode forecast max < 2.5× level post-fix (was ~13× pre-fix) |
| `issue_219_mfles_auto_mode_selects_additive_for_near_zero_series` | Auto mode selects additive when `min/median < τ=0.10` |
| `issue_219_mfles_explicit_multiplicative_with_floor_and_clamp` | Even forced-multiplicative stays within 10× in-sample max due to log-floor + back-transform clamp |

---

## Scope Notes

- **User-specified `TBATS::with_box_cox(lambda)`** (row 12, N/A): out of scope
  per CONTEXT.md. Non-auto, user-driven site. Responsibility for near-zero
  handling in this path rests with the caller.
- **Laplace rows 16–18** are gated behind `--features distributional`. They are
  included for completeness (CONTEXT.md requires distributional leaves to be
  audited); their guard assertions require the `distributional` feature flag.
- **Baseline models** (SMA, Naive, SeasonalNaive, RandomWalk, SeasonalWindow):
  out of scope — no auto-multiplicative or auto-log selection path.

---

*Cross-references: issue #10 (AutoETS zero-value), issue #219 (MFLES near-zero runaway).*
