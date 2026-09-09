---
phase: 07-erm-reconciliation-variant-ridge-solve
reviewed: 2026-09-09T00:00:00Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - src/hierarchy/mod.rs
findings:
  critical: 3
  warning: 3
  info: 1
  total: 7
status: issues_found
---

# Phase 7: ERM Reconciliation — Code Review Report

**Reviewed:** 2026-09-09
**Depth:** deep (numerical correctness focus)
**Files Reviewed:** 1 (`src/hierarchy/mod.rs`, diff c415fc2..HEAD)
**Status:** issues_found

## Summary

The ERM reconciliation implementation is algebraically correct end-to-end: the Gram matrix is
`Ŷ Ŷᵀ` (n×n, nodes×nodes SPD), the cross term is `B Ŷᵀ` (m×n), Cholesky is factored once and
solved column-wise, and the final coherent vector is `S·P·ŷ`. The `Eq`-derive drop is correct
and necessary (f64 is not Eq). The four new tests bring the total to 36 and are purely additive.
The reference-formula test hardcodes independently hand-computable constants — it is not circular.

Three blockers are present: the public API accepts negative and NaN `lambda` values that produce
silently wrong output (the parameter contract says λ ≥ 0 but is never enforced), and empty
training history (T=0) silently yields zero forecasts instead of an error. Beyond the blockers,
the reference test's numerical degeneracy (two nodes share identical training history) means an
internal node-ordering bug would be invisible to the current oracle test.

---

## Critical Issues

### CR-01: Negative `lambda` produces silently wrong (amplified) output

**File:** `src/hierarchy/mod.rs:874`
**Issue:** The public API documents `lambda ≥ 0` but never enforces it. When `lambda` is a
small negative value (e.g., -0.3), `G + λI` can remain positive-definite, so Cholesky
succeeds and `erm_reconcile` returns a result without error. The returned `P` is
*amplified* rather than regularized — the exact opposite of the intended shrinkage. A caller
who passes `lambda: -0.3` by accident receives numerically plausible-looking but wrong
reconciled forecasts with no diagnostic.

Demonstration: with two orthogonal nodes (`G = I₂`), `lambda = -0.3` gives `G + λI = 0.7·I`
which is still SPD; Cholesky succeeds, but `P = C · (0.7I)⁻¹ = (1/0.7)C` amplifies instead
of shrinking.

**Fix:** Add a guard at the top of `erm_reconcile` (or in `reconcile` before dispatch):
```rust
if !lambda.is_finite() || lambda < 0.0 {
    return Err(ForecastError::InvalidParameter(format!(
        "ERM: lambda must be finite and non-negative, got {lambda}"
    )));
}
```

---

### CR-02: NaN `lambda` bypasses Cholesky guard and produces silent NaN output

**File:** `src/hierarchy/mod.rs:874`
**Issue:** `gram[i * n + i] += lambda` with `lambda = f64::NAN` makes every diagonal `NaN`.
Inside `cholesky`, the test is `if diag <= 0.0` — IEEE 754 mandates that `NaN <= 0.0` is
`false`, so the guard does not trigger. The Cholesky factor is filled with NaN, and
`cholesky_solve_vec` propagates NaN throughout. The caller receives a `Vec` of NaN values
with no error, silently corrupting downstream forecasts.

This is a data-loss risk: reconciled forecasts stored, forwarded, or used in further
computation are silently NaN without any error indicator.

**Fix:** The guard in CR-01 (`!lambda.is_finite()`) covers this case too. No separate fix
needed beyond the combined check above.

---

### CR-03: T=0 training history produces silent zero forecasts

**File:** `src/hierarchy/mod.rs:853`
**Issue:** When every vector in `base_history` has length 0 (`T=0`), all validation loops
are empty (no iterations), `t_cap = 0`, the Gram sums to the zero matrix, and with
`lambda > 0` the Gram becomes `λI` — which is SPD. Cholesky succeeds, the cross term `C`
is a zero matrix, `P` is a zero matrix, `bottom = 0`, and the reconciled output is all
zeros — returned as `Ok(...)`. The caller has no way to distinguish "ERM computed zero
forecasts" from "ERM received empty training data."

**Fix:** After deriving `t_cap`, add:
```rust
if t_cap == 0 {
    return Err(ForecastError::InvalidParameter(
        "ERM: training history must contain at least one time period (T ≥ 1)".into(),
    ));
}
```

---

## Warnings

### WR-01: Public doc-comment formula is dimensionally inconsistent

**File:** `src/hierarchy/mod.rs:100` (enum arm doc), `src/hierarchy/mod.rs:781` (function doc)
**Issue:** Both doc-comments write the formula as `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹`. Given the
explicitly stated matrix layouts (`Ŷ` is nodes×T and `B` is leaves×T), applying prime
(transpose) as written yields:

- `Ŷ'Ŷ` = (T×n)(n×T) = T×T — wrong; the Gram should be n×n.
- `B'Ŷ` = (T×m)(n×T) — non-conformable.

The inline step comments are correct (`G = Ŷ_stored Ŷ_stored'` at line 866,
`C = B_stored Ŷ_stored'` at line 877) and the *code* is correct. The top-level doc
formula uses the wrong convention. A reader of the public API doc will derive a different
formula than what is implemented, which matters when reasoning about correctness or porting.

**Fix:** Change both doc occurrences to the consistent form:
```
P = B Ŷᵀ (Ŷ Ŷᵀ + λI)⁻¹
```
where `B` is m×T (leaf actuals, rows=leaves) and `Ŷ` is n×T (base forecasts, rows=nodes).
This matches the inline step comments and the code exactly.

---

### WR-02: Reference oracle test is numerically degenerate — ordering bugs are invisible

**File:** `src/hierarchy/mod.rs:2612` (`erm_correctness_reference_formula` test)
**Issue:** The test sets `base_history = {Total: [1,0], A: [1,0], B: [0,1]}`. The
histories for `Total` (node index 0) and `A` (node index 1) are *identical*. As a
consequence:

- Gram rows 0 and 1 are identical (`G[0,:] == G[1,:]`), so swapping Total and A in the
  internal node index order leaves the Gram unchanged.
- `P[leaf][Total]` and `P[leaf][A]` both equal `1/3`, so swapping those columns in `P`
  is invisible.
- `y_hat` entries for Total (10.0) and A (6.0) differ, but `P[0][0] = P[0][1] = 1/3`,
  so any ordering transposition between them produces the same dot product.

Any implementation bug that silently swaps the internal indices of Total and A —
throughout `y_stored`, `gram`, `cross`, `p`, or `y_hat` — would produce the exact same
numerical output. The `erm_coherent_multi_horizon` test uses the same degenerate training
data.

**Fix:** Add an asymmetric reference test with distinct histories for every node and a
larger hierarchy (e.g., Total→{A,B,C}, all three leaves having non-identical histories,
T ≥ n). Example asymmetric parameters that fully break column/row symmetry:

```rust
// Total=[2,1,0], A=[1,0,0], B=[0,1,0], C=[0,0,1], T=3, n=4, m=3, lambda=1.0
// All node histories are distinct => any index transposition produces wrong output
// Hand-compute P via numpy/scipy and hardcode expected bottom values.
```

---

### WR-03: Missing base_history length-mismatch test (partial coverage in `erm_shape_mismatch`)

**File:** `src/hierarchy/mod.rs:2676` (`erm_shape_mismatch` test)
**Issue:** `erm_shape_mismatch` only exercises the case where a *leaf*-history entry has
the wrong length. The symmetric case — a *base*-history entry (node history) having a
different length from `t_base` — is covered by the loop at lines 822–835 but has no test.
The validation loop for base history checks `vec.len() != t_base`, but if the very first
node (`nodes[0]`) has a shorter vector, `t_base` is derived from that entry and all others
get erroneously checked against it. This is not a code bug per se, but the lack of a test
means a refactor could break this path silently.

**Fix:** Add a companion test `erm_shape_mismatch_base_hist` that provides one node with
a different-length base history and asserts `is_err()`.

---

## Info

### IN-01: Indistinguishable error messages for missing base vs. leaf training history

**File:** `src/hierarchy/mod.rs:799–810`
**Issue:** Both `ok_or_else` arms — one for missing `erm_base_history` and one for missing
`erm_leaf_history` — emit the identical message:
`"ERM reconciliation requires training history; call with_erm_training() first"`.
Since `with_erm_training` sets both fields atomically this rarely matters in practice
(both are always `None` or both are `Some`), but if a future refactor allows partial
initialization the messages would be ambiguous.

**Fix:** Differentiate the messages:
```rust
// base_hist:
"ERM: base_history missing; call with_erm_training() first"
// leaf_hist:
"ERM: leaf_history missing; call with_erm_training() first"
```

---

## Checklist Against Review Scope

| Concern | Result |
|---|---|
| Ridge-solve formula `P = B Ŷᵀ(Ŷ Ŷᵀ + λI)⁻¹` | Correct (code); doc notation wrong (WR-01) |
| Gram matrix is n×n (nodes×nodes) SPD | Correct |
| +λI regularization on diagonal | Correct; but λ<0 and λ=NaN not guarded (CR-01, CR-02) |
| Cholesky factor-once-then-solve-columnwise | Correct |
| `reconciled_bottom = P·ŷ`, `all = S·bottom` | Correct |
| In-house `cholesky`/`cholesky_solve_vec` used (no faer) | Confirmed |
| Node ordering consistent across `y_stored`, `gram`, `y_hat` | Internally consistent; hidden by degenerate test (WR-02) |
| Reference test uses hardcoded oracle (not circular) | Confirmed — values are independently derivable |
| Reference test strong enough to catch ordering bugs | No — degenerate (WR-02) |
| `Eq` dropped, `PartialEq + Copy` retained | Correct |
| `with_erm_training` correctness | Correct; T=0 not guarded (CR-03) |
| Missing history → `ForecastError` (not panic/unwrap) | Correct |
| Shape validation (missing entries, unequal lengths) | Present; T=0 gap (CR-03) |
| `erm_requires_training` / `erm_shape_mismatch` tests | Present and correct |
| No `unwrap`/`panic` in production ERM code path | Confirmed |
| Backward-compat (32 pre-existing tests still pass) | Confirmed (32→36, +4 additive) |
| No production logging | Confirmed |
| `serde` derive on `ReconciliationMethod` | Not present; no serde concern |
| Clippy-clean | Not independently verified, but no obvious lint issues |

---

_Reviewed: 2026-09-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
