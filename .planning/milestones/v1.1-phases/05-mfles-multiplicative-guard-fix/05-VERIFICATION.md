---
phase: 05-mfles-multiplicative-guard-fix
verified: 2026-09-09T06:37:18Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 5: MFLES Multiplicative-Guard Fix Verification Report

**Phase Goal:** Fix the MFLES auto-multiplicative runaway (#219) at all three failure points — auto mode selection, the ln() log transform, and the exp() back-transform — and prove the fix with a committed before/after regression on the #219 repro series (near-zero month, level ≈ 2k; ~13× → ~level).
**Verified:** 2026-09-09T06:37:18Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                   | Status     | Evidence                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | MFLES auto mode selects additive (not multiplicative) when min/median < 0.10 (MULT-01)                 | ✓ VERIFIED | `mfles.rs:1034-1053` — `None` arm checks `median > 0.0 && (min_val/median) >= MULT_AUTO_TAU`; test `issue_219_mfles_auto_mode_selects_additive_for_near_zero_series` passes |
| 2   | In multiplicative mode, values are floored to 0.01×median before ln() (MULT-02)                       | ✓ VERIFIED | `mfles.rs:1065-1070` — `floor = MULT_LOG_FLOOR_FRAC * median`; `y = values.iter().map(\|&v\| v.max(floor).ln())`; test `issue_219_mfles_explicit_multiplicative_with_floor_and_clamp` passes |
| 3   | predict() clamps every multiplicative forecast to [0, 10×in-sample max] (MULT-03)                     | ✓ VERIFIED | `mfles.rs:963-974` — `raw.max(0.0).min(cap)` in `predict_internal()`; fit() inverse at line 1364 uses plain `.exp()` (clamp NOT applied there); `mfles_conforms_to_decomposable_contract` passes (12/12) |
| 4   | The #219 repro series (level ≈ 2k, one near-zero month) forecasts at ~level, not ~13× level (MULT-04) | ✓ VERIFIED | `issue_219_mfles_no_multiplicative_runaway` asserts `forecast_max < 5000`; module header documents before (~26000) / after (<5000) delta; test passes |
| 5   | issue_106_decomposable_conformance still passes — fit() decomposition invariant not broken             | ✓ VERIFIED | `cargo test --test issue_106_decomposable_conformance` → 12 passed; `mfles_conforms_to_decomposable_contract` is in the passing set |
| 6   | Public Forecaster API unchanged; guard parameters are private const; WR-01 serde(default) fix applied  | ✓ VERIFIED | Three consts (`MULT_AUTO_TAU`, `MULT_LOG_FLOOR_FRAC`, `MULT_BACK_CLAMP_K`) are `const f64` in `impl MFLES` (not public); no new builder methods in public API; `#[cfg_attr(feature = "serde", serde(default))]` on `insample_max` field (line 99) — commit `66bd423` |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact                                             | Expected                                     | Status      | Details                                                                  |
| ---------------------------------------------------- | -------------------------------------------- | ----------- | ------------------------------------------------------------------------ |
| `src/models/mfles.rs`                                | Three guards + insample_max field + 3 consts | ✓ VERIFIED  | All present: consts at 237/244/249, field at 100, guard logic at 1034-1073, clamp at 963-974 |
| `tests/issue_219_mfles_multiplicative_runaway.rs`    | 3 regression tests covering MULT-01..04      | ✓ VERIFIED  | File exists, 3 tests, module `//!` header with before/after delta documented |

### Key Link Verification

| From                              | To                                               | Via                                                | Status     | Details                                                                                    |
| --------------------------------- | ------------------------------------------------ | -------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| fit() mode-selection (line 1034)  | additive path for near-zero series               | `None` arm with `min_val/median >= MULT_AUTO_TAU`  | ✓ WIRED    | Code at 1034-1053; `MULT_AUTO_TAU = 0.10`; guard fires silently, no logging                |
| fit() log floor (line 1065)       | `insample_max` struct field (line 100)           | `self.insample_max = Some(max_val)` at line 1069   | ✓ WIRED    | Stored before the constant-series early-return (line 1083); additive path sets `None`      |
| `insample_max` field → predict_internal() clamp | `[0, 10×in-sample max]` bound on exp() | `MULT_BACK_CLAMP_K * m` cap at predict line 972 | ✓ WIRED    | `None` fallback = `f64::INFINITY` (clamp inert for additive/old-serde models)              |
| Clamp NOT applied in fit()        | Decomposable invariant preserved                 | fit() inverse at line 1364 uses plain `.exp()`     | ✓ WIRED    | Confirmed by grep: only one `cap` application at line 972 (predict path); `issue_106` passes |

### Data-Flow Trace (Level 4)

Not applicable — this phase modifies a numeric computation model, not a UI rendering pipeline. No client-visible data variable originates from a query/fetch that could be hollow. The relevant data flow (series values → mode guard → log floor → fit → insample_max → predict clamp) is verified by the behavioral tests.

### Behavioral Spot-Checks

| Behavior                                         | Command                                                              | Result                               | Status  |
| ------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------ | ------- |
| MULT-01/04: auto mode selects additive, no runaway | `cargo test --test issue_219_mfles_multiplicative_runaway`         | 3 passed; 0 failed; finished in 0.00s | ✓ PASS |
| MULT-03: decomposable invariant preserved        | `cargo test --test issue_106_decomposable_conformance`               | 12 passed; 0 failed; finished in 0.13s | ✓ PASS |
| CI hygiene: clippy -D warnings                   | `cargo clippy --all-targets --all-features -- -D warnings`           | Finished dev profile; 0 errors, 0 warnings | ✓ PASS |

### Probe Execution

No phase-declared probes found in PLAN or SUMMARY. `scripts/*/tests/probe-*.sh` not applicable to this phase (pure Rust model fix). Step 7c: SKIPPED (no probe files declared or conventional).

### Requirements Coverage

| Requirement | Source Plan | Description                                                          | Status      | Evidence                                                        |
| ----------- | ----------- | -------------------------------------------------------------------- | ----------- | --------------------------------------------------------------- |
| MULT-01     | 05-01-PLAN  | Auto mode selects additive when min/level below threshold τ          | ✓ SATISFIED | Mode guard at `mfles.rs:1034-1053`; test `issue_219_mfles_auto_mode_selects_additive_for_near_zero_series` passes |
| MULT-02     | 05-01-PLAN  | ln() transform floors/winsorizes near-zero input to 0.01×median     | ✓ SATISFIED | Log floor at `mfles.rs:1065-1070`; test `issue_219_mfles_explicit_multiplicative_with_floor_and_clamp` passes |
| MULT-03     | 05-01-PLAN  | Back-transform clamped to [0, 10×in-sample max]; decomposable preserved | ✓ SATISFIED | Clamp at predict line 963-974; fit unclamped (line 1364); `issue_106` 12/12 pass |
| MULT-04     | 05-01-PLAN  | #219 repro forecasts at ~level; before/after delta committed as guard | ✓ SATISFIED | Test `issue_219_mfles_no_multiplicative_runaway` asserts `< 2.5× level`; module header documents `~26000 → <5000` |

All four Phase 5 requirements satisfied. MULT-05 and MULT-06 are correctly scoped to Phase 6 and not evaluated here.

### Anti-Patterns Found

| File                                             | Line | Pattern           | Severity   | Impact                    |
| ------------------------------------------------ | ---- | ----------------- | ---------- | ------------------------- |
| `src/models/mfles.rs`                            | —    | None found        | —          | —                         |
| `tests/issue_219_mfles_multiplicative_runaway.rs` | —    | None found        | —          | —                         |

No TBD/FIXME/XXX markers in either modified file. No TODO/HACK/PLACEHOLDER patterns. No empty implementations (return null / return [] / return {}) in production paths. The insample_max field is initialized to None in `new()` and written by fit() — not a stub.

### Code Review Finding Dispositions (from 05-REVIEW.md)

| Finding | Severity | Status                                                                                           |
| ------- | -------- | ------------------------------------------------------------------------------------------------ |
| WR-01: `insample_max` missing `serde(default)` | Warning | **Fixed** — commit `66bd423` adds `#[cfg_attr(feature = "serde", serde(default))]` at line 99 |
| WR-02: duplicate median/min scan in auto path   | Warning | **Won't-fix** — author disposition: same pure fn on same immutable slice; hoisting would add a sort to the explicit-additive path. Non-blocking. |
| IN-01: MULT-01/MULT-04 test overlap             | Info    | **Won't-fix** — benign over-test; separate names document distinct guards. No action needed. |
| IN-02: hardcoded `insample_max_approx = 2400.0` | Info   | **Fixed** — commit `66bd423` derives cap from fixture's actual `primary_values().iter().max()`. |

All critical and warning findings either fixed or dispositioned; no new findings introduced.

### Human Verification Required

None. All must-haves are verifiable programmatically. No visual appearance, user flow, or external service checks are required for this phase.

### Gaps Summary

No gaps. All 6 must-haves are verified by direct code inspection and confirmed passing test runs. The phase goal — fix at all three failure points and prove with a committed before/after regression — is fully achieved.

---

_Verified: 2026-09-09T06:37:18Z_
_Verifier: Claude (gsd-verifier)_
