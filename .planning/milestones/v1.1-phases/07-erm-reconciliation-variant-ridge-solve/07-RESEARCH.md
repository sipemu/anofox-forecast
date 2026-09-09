# Phase 7: ERM Reconciliation Variant & Ridge Solve — Research

**Researched:** 2026-09-09
**Domain:** Rust hierarchical forecasting — enum extension, ridge regression, in-house Cholesky
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New enum arm **`Erm { lambda: f64 }`** added to `ReconciliationMethod` alongside existing arms —
  purely additive, existing hierarchy code compiles unchanged (ERM-01).
- **Training history via a builder setter on `HierarchyTree`** (e.g. `with_erm_training`), mirroring
  `set_actuals`/`set_residuals` storage pattern. `reconcile()` dispatches through the existing match
  (ERM-02).
- **Missing training history when `Erm` selected → return a clear `ForecastError`** with an
  actionable hint. No panic, no silent fallback.
- **Validate matrix shapes at solve time**: base-forecast history is nodes×T, leaf actuals is
  leaves×T, with T ≥ documented minimum; mismatched shapes → `ForecastError`.
- **Reuse the module's in-house `cholesky`/`cholesky_solve_vec`**; NO faer, NO feature gate.
- Correctness proof: hand-computed P on a 2-level hierarchy, asserted within tolerance; also assert
  coherence `S·bottom == all`.

### Claude's Discretion
- Exact builder setter name (planner chooses — `with_erm_training` suggested).
- Private fn name (`erm_reconcile` or similar).
- Minimum T value (>= n nodes; recommend >= max(n, 2) with documented rationale).

### Deferred Ideas (OUT OF SCOPE for Phase 7)
- **Auto-λ (Ledoit-Wolf-style) default (ERM-04)** — Phase 8.
- **Grouped/crossed end-to-end accuracy validation vs MinTrace/unreconciled baseline (ERM-06)** —
  Phase 8.
- Any performance tuning beyond correctness.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERM-01 | `ReconciliationMethod::Erm { lambda }` exists alongside existing arms, fully backward-compatible | Enum analysis: only the derive macro and `match` arm need changes; no exhaustive user-facing matches exist on `ReconciliationMethod` outside the single `reconcile()` match |
| ERM-02 | Training-history API accepts base forecasts + leaf actuals (T periods) for ERM to consume | Pattern from `set_actuals`/`set_residuals` directly applicable; two new `Option<...>` fields on `HierarchyTree` |
| ERM-03 | ERM computes ridge solve `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹`; reconciled bottom = `P·ŷ`, all = `S·P·ŷ` | Full computation sequence with concrete matrix dimensions documented below; in-house Cholesky reuse confirmed |
| ERM-05 | Correctness verified against the reference formula on a known small hierarchy within tolerance | Concrete 2-level test case with exact rational answers provided below; test shape mirrors `mint_ols_coherent` pattern |
</phase_requirements>

---

## Summary

Phase 7 adds `ReconciliationMethod::Erm { lambda: f64 }` to `src/hierarchy/mod.rs`, a new private
`erm_reconcile()` function, and a builder setter `with_erm_training()` on `HierarchyTree`. The ERM
formula `P = B'Ŷ(Ŷ'Ŷ + λI)⁻¹` — from Ben Taieb & Koo (2019, KDD) — is computed via a single
Cholesky factorisation of the n×n SPD Gram matrix `Ŷ'Ŷ + λI` followed by n column solves, reusing
the existing `cholesky`/`cholesky_solve_vec` helpers already in the module.

The primary implementation risk is the `Eq` derive on `ReconciliationMethod`: adding `Erm { lambda:
f64 }` breaks `Eq` because `f64: !Eq`. The fix is removing `Eq` from the derive line — `PartialEq`
is preserved and sufficient. The `Eq` derive is currently unused in comparisons (grep confirms no
`==` comparisons on `ReconciliationMethod` anywhere in the codebase). A second risk is the JS
bindings `parse_method()` string-match: it is a `_ =>` catch-all (not exhaustive), so it compiles
fine; updating its docstring and the error message to mention `erm` is a polish step.

**Primary recommendation:** Implement `erm_reconcile(&self, base_map, horizon, lambda)` following
the exact `min_trace_ols` pattern (build S, compute G=Ŷ_stored*Ŷ_stored', add λI, Cholesky once,
solve per row of P, multiply P·ŷ for each horizon step, S·result for coherent all-node output).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ERM enum variant | `src/hierarchy/mod.rs` (enum def) | — | Same file as all other variants |
| Training-history storage | `HierarchyTree` struct fields | — | Mirrors `actuals`/`residuals` fields |
| Ridge solve (Cholesky) | `erm_reconcile()` private fn | `cholesky` / `cholesky_solve_vec` (reuse) | All reconciliation logic lives in private fns dispatched by `reconcile()` |
| Correctness test | `#[cfg(test)]` inline in hierarchy/mod.rs | — | Consistent with all 32 existing hierarchy tests |
| WASM/JS exposure | `crates/anofox-forecast-js/src/hierarchy.rs` `parse_method()` | — | String-match catch-all; must add `"erm"` string arm for Phase 8 or document as Phase 8 task |

---

## Research Question Answers

### Q1 — The ERM Math, Made Implementable

**Convention used by the codebase (and this research):**
- `n` = number of nodes, `m` = number of leaves, `T` = number of training time periods
- Ŷ_stored: `n × T` matrix, stored as `Vec<Vec<f64>>` where `Ŷ_stored[node_idx][t]`
- B_stored: `m × T` matrix, stored as `Vec<Vec<f64>>` where `B_stored[leaf_rank][t]`
- ŷ: `n`-vector, current base forecast (one step), indexed by node index in BFS order

`[ASSUMED]` — the "nodes×T" vs "T×nodes" convention in the CONTEXT.md vs the Ben Taieb & Koo paper differs; the research resolves this as follows and the arithmetic is self-consistent:

**Gram matrix** G = `Ŷ_stored Ŷ_stored'` (n×n):
```
G[i, j] = sum_{t=0}^{T-1} Ŷ_stored[i][t] * Ŷ_stored[j][t]
```

**Cross term** C = `B_stored Ŷ_stored'` (m×n):
```
C[i, j] = sum_{t=0}^{T-1} B_stored[i][t] * Ŷ_stored[j][t]
```

**Projection matrix** P = C (G + λI)⁻¹ (m×n):
- Never materialize (G + λI)⁻¹ explicitly.
- Cholesky-factor (G + λI) once: `L = cholesky(n, &(G + λI).flat())`.
- For each row `i` of P (i = 0..m): `P[i,:] = cholesky_solve_vec(n, &L, C[i,:])`.
  This solves `(G + λI) P[i,:]^T = C[i,:]^T`.

**Reconciled bottom** (m-vector, one per horizon step h):
```
bottom[h][i] = sum_{j=0}^{n-1} P[i][j] * ŷ[j]   for i = 0..m
```

**Reconciled all-nodes** (n-vector, applying summing matrix S):
```
all[h][node] = sum_{i=0}^{m-1} S[node][i] * bottom[h][i]
```

Where S is the standard summing matrix (n×m), identical to what `min_trace_ols` builds
(`s[leaf][j] = 1.0`, `s[anc][j] = 1.0` for all ancestors via `ancestors_of()`).

**Node ordering** for Ŷ_stored and the result vector: internal node index order
(0..self.nodes.len()), NOT BFS order. The final output is converted to BFS order by
`to_named_output()` as in all other methods. The training API must document this: the caller
provides forecasts keyed by node name, the setter reindexes to internal order.

**Shape of P confirmed:** `m × n` (leaves × nodes). `[VERIFIED: src/hierarchy/mod.rs:467-474]`
via the `leaves()` function which defines m, and node count defines n.
Verbatim: `fn leaves(&self) -> Vec<usize> { self.nodes.iter().enumerate().filter(|(_, n)| n.children.is_empty()).map(|(i, _)| i).collect() }`

---

### Q2 — Reference Formula and Literature Source

**Primary source:** Ben Taieb, S., & Koo, B. (2019). "Regularized regression for hierarchical
forecasting without unbiasedness conditions." *KDD 2019*, 1337–1347.
`[ASSUMED]` — paper referenced in REQUIREMENTS.md line 23; not fetched in this session; the formula
is consistent with the standard ERM derivation below.

**Formula derivation (self-contained):**
The ERM objective minimizes:
```
min_P  ||B_math - Ŷ_math P'||_F^2 + λ||P||_F^2
```
where `B_math` is `T×m`, `Ŷ_math` is `T×n`, `P` is `m×n`.

Taking derivative w.r.t. P and setting to zero:
```
(Ŷ_math' Ŷ_math + λI_n) P' = Ŷ_math' B_math
P' = (Ŷ_math' Ŷ_math + λI_n)^{-1} Ŷ_math' B_math
P  = B_math' Ŷ_math (Ŷ_math' Ŷ_math + λI_n)^{-1}
```

In the paper's `T×n` math convention, `Ŷ_math' Ŷ_math` is `n×n` ✓.
In the codebase's `n×T` storage convention, the Gram is `Ŷ_stored Ŷ_stored'` which equals the same
`n×n` matrix element-for-element.

**B = leaf-level actuals only** (confirmed by the paper and the formula): S maps bottom-level
(leaves) to all nodes via `S @ bottom = all`. B contains only the m leaf series, NOT all n nodes.
The reconciled all-nodes vector is recovered by `S @ P @ ŷ`. `[ASSUMED]` — consistent with paper
convention.

---

### Q3 — Training-History API

**New fields on `HierarchyTree`** (mirroring `actuals: Option<...>` and `residuals: Option<...>`):
`[VERIFIED: src/hierarchy/mod.rs:106-114]`

Verbatim existing fields:
```rust
/// Historical actual values per node, used for TopDown proportions.
actuals: Option<HashMap<String, Vec<f64>>>,
/// Historical residuals per node, used for MinTraceShrink covariance.
residuals: Option<HashMap<String, Vec<f64>>>,
```

New fields to add to `HierarchyTree` struct:
```rust
/// Historical base-forecast matrix for all nodes (node_name → Vec<f64> of length T).
/// Used by ERM reconciliation. Node ordering: internal index order; T = number of training steps.
erm_base_history: Option<HashMap<String, Vec<f64>>>,
/// Historical leaf-actual matrix (leaf_name → Vec<f64> of length T).
/// Used by ERM reconciliation. Only leaf nodes; length T must match erm_base_history.
erm_leaf_history: Option<HashMap<String, Vec<f64>>>,
```

**Also add `None` initializers in `new()` and `from_summing_matrix()`** for both new fields.
`[VERIFIED: src/hierarchy/mod.rs:201-207]` (`new()` init block) and `[VERIFIED: src/hierarchy/mod.rs:405-411]` (`from_summing_matrix()` init block).

**Builder setter signature (suggested name; planner may rename):**
```rust
/// Provide training history for ERM reconciliation.
///
/// `base_history`: map from every node name to a Vec<f64> of length T (historical base forecasts).
/// `leaf_history`: map from every leaf node name to a Vec<f64> of length T (historical actuals).
/// T must be the same for all entries and >= max(n_nodes, 2) for a well-posed solve.
pub fn with_erm_training(
    &mut self,
    base_history: HashMap<String, Vec<f64>>,
    leaf_history: HashMap<String, Vec<f64>>,
) {
    self.erm_base_history = Some(base_history);
    self.erm_leaf_history = Some(leaf_history);
}
```

**Node/leaf ordering in the API:** The setter accepts `HashMap<String, Vec<f64>>` (keyed by node
name), which is self-describing — identical to `set_actuals` and `set_residuals`. The `erm_reconcile`
private fn reindexes into internal node/leaf ordering using `self.name_to_idx` and `self.leaves()`.
`[VERIFIED: src/hierarchy/mod.rs:107-108]` — `name_to_idx: HashMap<String, usize>` exists as a
struct field for O(1) lookup.

**Shape validation rules** (performed inside `erm_reconcile`, not in the setter):
1. Both `erm_base_history` and `erm_leaf_history` must be `Some(_)`, else return
   `ForecastError::InvalidParameter` with hint `"ERM reconciliation requires training history; call with_erm_training() first"`.
2. All nodes in the hierarchy must have a `base_history` entry; all leaves must have a
   `leaf_history` entry; missing entry → `ForecastError::InvalidParameter`.
3. All Vec<f64> in `base_history` must have length T; all in `leaf_history` must have length T.
   Mismatch → `ForecastError::DimensionMismatch`.
4. T must be >= n (n = number of nodes); if T < n the Gram Ŷ'Ŷ has rank < n and even λI may not
   produce a well-conditioned system. Recommended minimum: `T >= n`. Return
   `ForecastError::InsufficientData { needed: n, got: T, hint: Some("ERM requires at least as many training periods as nodes for a well-conditioned ridge solve; increase T or λ") }`.

---

### Q4 — Concrete Hand-Computable Test Case

**Hierarchy:** 1 root (Total) + 2 leaves (A, B) — identical to `mint_ols_coherent` test.
n=3 nodes, m=2 leaves, T=2 training periods.

**Node BFS/internal index order:** Total=0, A=1, B=2 (confirmed by `simple_tree` test).
`[VERIFIED: src/hierarchy/mod.rs:1506-1509]`
Verbatim: `assert_eq!(tree.node_names(), vec!["Total", "A", "B"]);`

**Leaf order** (returned by `leaves()`): [1(A), 2(B)] — nodes with no children.

**Summing matrix S (n×m = 3×2):**
```
S = [[1, 1],   // Total (row 0) = A + B
     [1, 0],   // A (row 1) is leaf 0
     [0, 1]]   // B (row 2) is leaf 1
```

**Training data:**

Ŷ_stored (nodes×T = 3×2), stored as node_name → [t0, t1]:
```
Total: [1.0, 0.0]
A:     [1.0, 0.0]
B:     [0.0, 1.0]
```

B_stored (leaves×T = 2×2), stored as leaf_name → [t0, t1]:
```
A: [1.0, 0.0]
B: [0.0, 1.0]
```

λ = 1.0

**Arithmetic:**

Gram matrix G = Ŷ_stored Ŷ_stored' (n×n = 3×3):
```
G[i,j] = Σ_t Ŷ[i,t]*Ŷ[j,t]

G[0,0] = 1*1 + 0*0 = 1
G[0,1] = 1*1 + 0*0 = 1
G[0,2] = 1*0 + 0*1 = 0
G[1,1] = 1*1 + 0*0 = 1
G[1,2] = 1*0 + 0*1 = 0
G[2,2] = 0*0 + 1*1 = 1

G = [[1, 1, 0],
     [1, 1, 0],
     [0, 0, 1]]
```

(G is rank-2; T=2 < n=3 → λ regularization is essential here, demonstrating the pitfall.)

G + λI (λ=1.0):
```
G + I = [[2, 1, 0],
         [1, 2, 0],
         [0, 0, 2]]
```

Cross term C = B_stored Ŷ_stored' (m×n = 2×3):
```
C[0,0] = 1*1 + 0*0 = 1  (A×Total)
C[0,1] = 1*1 + 0*0 = 1  (A×A)
C[0,2] = 1*0 + 0*1 = 0  (A×B)
C[1,0] = 0*1 + 1*0 = 0  (B×Total)
C[1,1] = 0*1 + 1*0 = 0  (B×A)
C[1,2] = 0*0 + 1*1 = 1  (B×B)

C = [[1, 1, 0],
     [0, 0, 1]]
```

Solve (G+I) P^T = C^T, i.e. for each row of P solve (G+I) x = C[row,:]:

(G+I) is block-diagonal: [[2,1],[1,2]] ⊕ [[2]]
inv([[2,1],[1,2]]) = (1/(4-1)) * [[2,-1],[-1,2]] = [[2/3,-1/3],[-1/3,2/3]]
inv([[2]]) = [[1/2]]

So (G+I)⁻¹ = [[2/3, -1/3, 0], [-1/3, 2/3, 0], [0, 0, 1/2]]

**P = C @ (G+I)⁻¹ (m×n = 2×3):**
```
P[0,:] = [1, 1, 0] @ [[2/3,-1/3,0],[-1/3,2/3,0],[0,0,1/2]]
       = [2/3 - 1/3, -1/3 + 2/3, 0]
       = [1/3, 1/3, 0]

P[1,:] = [0, 0, 1] @ [[2/3,-1/3,0],[-1/3,2/3,0],[0,0,1/2]]
       = [0, 0, 1/2]

P = [[1/3, 1/3, 0  ],    // row 0 = leaf A
     [0,   0,   1/2]]    // row 1 = leaf B
```

Verify via solve:
- Row 0: (G+I) @ [1/3, 1/3, 0] = [2*1/3+1*1/3, 1*1/3+2*1/3, 0] = [1, 1, 0] = C[0,:] ✓
- Row 1: (G+I) @ [0, 0, 1/2]   = [0, 0, 2*1/2] = [0, 0, 1] = C[1,:] ✓

**Current base forecast ŷ (n-vector, horizon h=0):**
```
ŷ = [Total=10.0, A=6.0, B=5.0]  (in internal node order 0,1,2)
```

**Reconciled bottom = P @ ŷ (m-vector):**
```
bottom[A] = 1/3*10 + 1/3*6 + 0*5 = 10/3 + 2 = 16/3
bottom[B] = 0*10 + 0*6 + 1/2*5   = 5/2
```

**Reconciled all = S @ bottom (n-vector):**
```
all[Total] = 1*16/3 + 1*5/2 = 32/6 + 15/6 = 47/6
all[A]     = 1*16/3 + 0     = 16/3
all[B]     = 0     + 1*5/2  = 5/2
```

**Exact decimal values:**
```
16/3 ≈ 5.333333...
5/2  = 2.5
47/6 ≈ 7.833333...
```

**Coherence check:** all[Total] = all[A] + all[B] → 47/6 = 16/3 + 5/2 = 32/6 + 15/6 = 47/6 ✓

**Test skeleton (inline `#[cfg(test)]` in hierarchy/mod.rs):**
```rust
#[test]
fn erm_correctness_reference_formula() {
    // 2-level hierarchy: Total -> {A, B}, n=3 nodes, m=2 leaves, T=2, lambda=1.0
    let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

    let mut base_hist = HashMap::new();
    base_hist.insert("Total".into(), vec![1.0, 0.0]);
    base_hist.insert("A".into(),     vec![1.0, 0.0]);
    base_hist.insert("B".into(),     vec![0.0, 1.0]);

    let mut leaf_hist = HashMap::new();
    leaf_hist.insert("A".into(), vec![1.0, 0.0]);
    leaf_hist.insert("B".into(), vec![0.0, 1.0]);

    tree.with_erm_training(base_hist, leaf_hist);

    let base = vec![
        ("Total".into(), vec![10.0]),
        ("A".into(),     vec![6.0]),
        ("B".into(),     vec![5.0]),
    ];

    let result = tree
        .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
        .unwrap();
    let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

    // Reference values: P = [[1/3,1/3,0],[0,0,1/2]]
    // bottom_A = 16/3, bottom_B = 5/2, Total = 47/6
    approx_eq(map["A"][0],     16.0 / 3.0, 1e-10);
    approx_eq(map["B"][0],      5.0 / 2.0, 1e-10);
    approx_eq(map["Total"][0], 47.0 / 6.0, 1e-10);

    // Coherence: Total = A + B
    approx_eq(map["Total"][0], map["A"][0] + map["B"][0], 1e-10);
}

#[test]
fn erm_requires_training() {
    let tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();
    let base = vec![
        ("Total".into(), vec![100.0]),
        ("A".into(),     vec![50.0]),
        ("B".into(),     vec![50.0]),
    ];
    assert!(tree
        .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
        .is_err());
}

#[test]
fn erm_coherent_multi_horizon() {
    // Same setup; verify coherence across h=0,1
    let mut tree = HierarchyTree::new(vec![("Total", &["A", "B"])]).unwrap();

    let mut base_hist = HashMap::new();
    base_hist.insert("Total".into(), vec![1.0, 0.0]);
    base_hist.insert("A".into(),     vec![1.0, 0.0]);
    base_hist.insert("B".into(),     vec![0.0, 1.0]);
    let mut leaf_hist = HashMap::new();
    leaf_hist.insert("A".into(), vec![1.0, 0.0]);
    leaf_hist.insert("B".into(), vec![0.0, 1.0]);
    tree.with_erm_training(base_hist, leaf_hist);

    let base = vec![
        ("Total".into(), vec![10.0, 20.0]),
        ("A".into(),     vec![6.0,  12.0]),
        ("B".into(),     vec![5.0,  9.0]),
    ];

    let result = tree
        .reconcile(&base, ReconciliationMethod::Erm { lambda: 1.0 })
        .unwrap();
    let map: HashMap<&str, &Vec<f64>> = result.iter().map(|(k, v)| (k.as_str(), v)).collect();

    for h in 0..2 {
        approx_eq(map["Total"][h], map["A"][h] + map["B"][h], 1e-10);
    }
}
```

---

### Q5 — Backward Compatibility & Serde

**`Eq` derive conflict — CRITICAL:**
`[VERIFIED: src/hierarchy/mod.rs:60]`
Current derive: `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`
The `Erm { lambda: f64 }` arm adds an `f64` field. `f64: PartialEq` but `f64: !Eq`.
**Fix:** remove `Eq` from the derive line. `PartialEq` is sufficient.
Verbatim current line: `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`
New line: `#[derive(Debug, Clone, Copy, PartialEq)]`

`Copy` is preserved: `f64: Copy`.

**No `Eq` usage exists in the codebase:**
`[VERIFIED: src/hierarchy/mod.rs]` — `grep -rn "ReconciliationMethod.*==" finds zero results`;
`Eq` is only on the derive line. Removing it causes zero downstream breakage.

**Exhaustive matches on `ReconciliationMethod`:**
`[VERIFIED: src/hierarchy/mod.rs:518-532]`
There is exactly one `match method` block in the codebase (inside `reconcile()`). It is
non-exhaustive in the _user's_ code sense only because users do not typically match on it. The
compiler will require a new arm `ReconciliationMethod::Erm { lambda } => self.erm_reconcile(...)`.
No other exhaustive match exists outside `reconcile()`.

`[VERIFIED: crates/anofox-forecast-js/src/hierarchy.rs:21-37]` — JS bindings use `parse_method()`
which is a string match with a `_ =>` catch-all (not an exhaustive enum match), so it compiles
without changes. The error message should be updated to mention `"erm"` as a polish step.

**Serde:**
`ReconciliationMethod` does NOT currently have any serde derive.
`[VERIFIED: src/hierarchy/mod.rs]` — `grep -n "Serialize\|Deserialize\|cfg_attr.*serde"` in
hierarchy/mod.rs returns zero results. The codebase pattern for conditional serde is
`#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. If this is
not on `ReconciliationMethod` today, adding the new arm has no serde consequence. No action needed
for Phase 7. (Phase 8 auto-λ may wish to add serde to the enum — that's deferred.)

---

### Q6 — Pitfalls

**Pitfall 1: Rank deficiency when T < n**
When T (training periods) < n (nodes), the Gram matrix Ŷ'Ŷ is rank-deficient. Without λ>0, the
Cholesky fails with `ForecastError::SingularMatrix`. With λ>0, the solve succeeds but the result
is dominated by the regularizer (P shrinks toward zero, giving near-zero bottom forecasts). The
concrete test case intentionally uses T=2, n=3 (rank-deficient G) with λ=1.0 to demonstrate that
regularization saves the solve. Document minimum T=n in the setter docstring, but do not hard-block
T<n (λ handles it; just emit a warning-level hint).

**Pitfall 2: λ = 0 with rank-deficient G**
λ=0 is valid only when T≥n and Ŷ has full column rank. For T<n or near-multicollinear base
forecasts, λ=0 will cause `cholesky()` to return `Err(SingularMatrix(...))`. The error message from
`cholesky()` currently reads `"hierarchy summing matrix S'S is singular"` — this is misleading for
ERM. Wrap the cholesky call and return a more specific message:
```rust
cholesky(n, &gram_flat).map_err(|_| ForecastError::SingularMatrix(
    "ERM: Gram matrix Ŷ'Ŷ + λI is singular; increase lambda or provide more training periods".into()
))
```

**Pitfall 3: Node vs leaf ordering mismatch**
The `leaves()` method returns a `Vec<usize>` of internal node indices — this is the canonical
ordering for leaf dimension m. When building B_stored (m×T), iterate over `leaves()` and index
into `leaf_history` by name (`self.nodes[leaf_idx].name`). Misaligning leaf order between B and P
will produce incorrect reconciled values with no compile-time error. Validate that every leaf node
name is present in `leaf_history` before building B_stored.

**Pitfall 4: BFS order vs internal order**
`reconcile()` receives `base_forecasts` keyed by name. `min_trace_ols` and `erm_reconcile` both
work in internal node index order (0..n), not BFS order, and call `to_named_output()` at the end to
convert. This is correct — but the training history input is also name-keyed and must be reindexed
to internal order, not BFS order. Both are the same for a simple tree (names are assigned
sequentially by `new()`), but for `from_summing_matrix` they may differ. Always use
`self.name_to_idx[name]` to convert.
`[VERIFIED: src/hierarchy/mod.rs:107-108]` — `name_to_idx: HashMap<String, usize>` is available.

**Pitfall 5: coherence numerical tolerance**
The coherence property `all[Total] = all[A] + all[B]` holds exactly in exact arithmetic (because
`all = S @ P @ ŷ` and `S` encodes the tree constraint). In floating point, round-trip errors
accumulate. The existing tests use `1e-10` as tolerance — use the same for ERM tests.

---

## Standard Stack

No external packages needed. Phase 7 is pure in-crate Rust, reusing existing helpers.

| Component | Source | Status |
|-----------|--------|--------|
| `cholesky(n, a)` | `src/hierarchy/mod.rs:1389` | `[VERIFIED: src/hierarchy/mod.rs:1389-1411]` — reuse as-is |
| `cholesky_solve_vec(n, l, b)` | `src/hierarchy/mod.rs:1414` | `[VERIFIED: src/hierarchy/mod.rs:1414-1434]` — reuse as-is |
| `leaves()` | `src/hierarchy/mod.rs:467` | `[VERIFIED: src/hierarchy/mod.rs:467-474]` — canonical leaf order |
| `ancestors_of(idx)` | `src/hierarchy/mod.rs:883` | `[VERIFIED: src/hierarchy/mod.rs:883-897]` — for S matrix build |
| `bfs_order()` | `src/hierarchy/mod.rs:1241` | `[VERIFIED: src/hierarchy/mod.rs:1241-1256]` — output ordering |
| `to_named_output()` | `src/hierarchy/mod.rs:1259` | `[VERIFIED: src/hierarchy/mod.rs:1259-1264]` — final output |
| `ForecastError` variants | `src/error.rs:10-73` | `[VERIFIED: src/error.rs:10-73]` — InvalidParameter, DimensionMismatch, InsufficientData, SingularMatrix all available |

---

## Package Legitimacy Audit

Not applicable — no external packages are installed in this phase.

---

## Architecture Patterns

### Recommended Private Function Shape

Following the exact pattern of `min_trace_ols` (lines 654–720):

```rust
/// ERM reconciliation: P = B'Ŷ (Ŷ'Ŷ + λI)⁻¹, reconciled = S·P·ŷ.
///
/// `lambda` — ridge regularization strength. Larger values shrink P toward zero,
/// increasing robustness when T is small relative to n. Set to 0.0 only if T >> n
/// and base forecasts are well-conditioned (otherwise Gram matrix may be singular).
fn erm_reconcile(
    &self,
    base_map: &HashMap<&str, &Vec<f64>>,
    horizon: usize,
    lambda: f64,
) -> Result<Vec<(String, Vec<f64>)>> {
    // 1. Retrieve training history (returns Err if not set)
    // 2. Build Ŷ_stored (n×T) and B_stored (m×T) using self.name_to_idx and self.leaves()
    // 3. Compute Gram G = Ŷ_stored Ŷ_stored' (n×n) + λI
    // 4. Compute cross term C = B_stored Ŷ_stored' (m×n)
    // 5. Cholesky factor L of G+λI
    // 6. For each row i of P: P[i] = cholesky_solve_vec(n, &L, &C[i])
    // 7. Build S matrix (n×m) identical to min_trace_ols
    // 8. For each horizon step h: bottom[h] = P @ ŷ[h]; all[h] = S @ bottom[h]
    // 9. Return to_named_output(&reconciled)
    todo!()
}
```

### Dispatch Addition in `reconcile()`

`[VERIFIED: src/hierarchy/mod.rs:518-532]` — add one arm to the existing match:

```rust
ReconciliationMethod::Erm { lambda } => self.erm_reconcile(&base_map, horizon, lambda),
```

### Anti-Patterns to Avoid

- **Do not** build B as all-node actuals; use leaf actuals only. S handles the aggregation.
- **Do not** call `cholesky` on B'Ŷ itself; the Cholesky target is (G + λI) which is SPD.
- **Do not** add a `#[cfg(feature = ...)]` gate; ERM is always available like BottomUp/MinTraceOls.
- **Do not** keep `Eq` in the derive after adding `lambda: f64` — it won't compile.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SPD linear solve | Custom Gaussian elimination | In-house `cholesky`+`cholesky_solve_vec` | Already verified correct by 32 passing MinTrace tests |
| Summing matrix S | New S construction logic | Copy `min_trace_ols` S-build verbatim | Grouped-hierarchy-safe via `ancestors_of()` |
| Output formatting | Custom named-tuple code | `to_named_output()` | Already handles BFS ordering |

---

## Common Pitfalls

### Pitfall 1: Singular Gram when λ=0 and T<n
**What goes wrong:** `cholesky()` returns `Err(SingularMatrix(...))` with a misleading message.
**Why it happens:** T=2, n=3 means G has rank 2 < n=3; without ridge the system is underdetermined.
**How to avoid:** Wrap the cholesky error with an ERM-specific message; document minimum T.
**Warning signs:** SingularMatrix error from `cholesky` inside `erm_reconcile`.

### Pitfall 2: Eq derive compile error
**What goes wrong:** Adding `Erm { lambda: f64 }` to a `#[derive(Eq)]` enum fails to compile.
**Why it happens:** `f64: !Eq` (NaN breaks reflexivity).
**How to avoid:** Remove `Eq` from derive; keep `PartialEq`. `Copy` is fine (f64 is Copy).
**Warning signs:** `error[E0277]: the trait bound 'f64: Eq' is not satisfied`.

### Pitfall 3: Leaf ordering inconsistency between B and leaves()
**What goes wrong:** B rows are assembled in a different order than `self.leaves()` returns, so P
rows correspond to the wrong leaf.
**How to avoid:** Build B by iterating `self.leaves()` in order and looking up each leaf name in
`leaf_history`. Never rely on HashMap iteration order.
**Warning signs:** Coherence test passes (sums still correct) but individual leaf values are wrong
(A and B swapped).

### Pitfall 4: Node ordering — BFS vs internal
**What goes wrong:** Ŷ rows assembled in BFS order rather than internal index order → misaligned
with S matrix (which uses internal indices).
**How to avoid:** Build Ŷ by iterating `0..n` (internal index) and looking up `self.nodes[i].name`
in `base_history`. Same pattern as `min_trace_ols` line 699.
`[VERIFIED: src/hierarchy/mod.rs:699]` verbatim: `let base_vec: Vec<f64> = (0..n).map(|i| base_map[self.nodes[i].name.as_str()][h]).collect();`

---

## Runtime State Inventory

Not applicable (greenfield enum arm + new fields; no rename/refactor).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | All code | ✓ | (project toolchain) | — |
| `cargo test` | Running hierarchy tests | ✓ | confirmed via test run | — |

No external tools, services, or new crates needed.

---

## Validation Architecture

`workflow.nyquist_validation: true` in `.planning/config.json` — section required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`#[test]`) — no external test crate for unit tests |
| Config file | None (inline `#[cfg(test)]` in `src/hierarchy/mod.rs`) |
| Quick run command | `cargo test --lib hierarchy::tests::erm` |
| Full suite command | `cargo test --lib hierarchy` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ERM-01 | Enum arm compiles, existing tests unaffected | compile + unit | `cargo test --lib hierarchy` | ❌ Wave 0 (new arm) |
| ERM-02 | `with_erm_training()` stores history; missing history → Err | unit | `cargo test --lib hierarchy::tests::erm_requires_training` | ❌ Wave 0 |
| ERM-03 | Ridge solve produces correct P and bottom | unit | `cargo test --lib hierarchy::tests::erm_correctness_reference_formula` | ❌ Wave 0 |
| ERM-05 | Reference-formula match within 1e-10; coherence S·bottom==all | unit | `cargo test --lib hierarchy::tests::erm_correctness_reference_formula` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test --lib hierarchy`
- **Per wave merge:** `cargo test --lib hierarchy`
- **Phase gate:** `cargo test --lib hierarchy` green before `/gsd-verify-work`; also `cargo clippy --all-targets -- -D warnings`

### Wave 0 Gaps

- [ ] `erm_correctness_reference_formula` test — covers ERM-03 + ERM-05
- [ ] `erm_requires_training` test — covers ERM-02 missing-history error path
- [ ] `erm_coherent_multi_horizon` test — covers ERM-03 coherence for horizon > 1
- [ ] Framework: none (inline tests, same as existing hierarchy tests)

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`. Phase 7 is a pure in-library numerical
computation with no I/O, no authentication, no network, and no external data ingestion beyond the
caller-supplied training arrays.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Shape validation at solve entry (`DimensionMismatch`, `InsufficientData`) |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN/Inf in training data | Tampering / information disclosure | `cholesky()` detects non-positive diagonal → `SingularMatrix`; pre-validate with `series.has_missing_values()` pattern if desired |
| λ = 0 + singular G → panic | Denial of service | Cholesky returns `Err`, not panic; propagates as `Result` |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | B = leaf-level actuals only (not all-node actuals) | Q2 Reference Formula | If wrong, formula dimensions don't match (m vs n); test would fail to compile or produce incoherent output |
| A2 | Ben Taieb & Koo 2019 is the primary ERM reference cited in requirements | Q2 | Alternative formulations may define B or Ŷ differently; the hand-computed test is self-consistent regardless |
| A3 | Minimum T recommended as n (number of nodes) | Q3 Training-History API | If T<n is disallowed, test case (T=2<n=3) would fail validation; recommend soft-warn not hard-block |

---

## Open Questions

1. **Minimum T enforcement — hard block or soft warning?**
   - What we know: T<n makes G rank-deficient; λ>0 recovers a valid solve; test case uses T=2, n=3.
   - What's unclear: whether users will legitimately use very small T (e.g. warm-start with 2 obs).
   - Recommendation: soft warning in error hint (not hard block); let λ do its job.

2. **JS bindings `parse_method()` — update in Phase 7 or Phase 8?**
   - What we know: the catch-all `_ =>` arm compiles; ERM is not exposed via JS until someone adds
     `"erm"` to `parse_method()`.
   - Recommendation: add `"erm"` string arm in Phase 7 (small, complete) since the JS file is
     already open; add lambda parsing from a second parameter. Or defer to Phase 8 when auto-λ is
     added (cleaner API). Document as open.

---

## Sources

### Primary (HIGH confidence)
- `[VERIFIED: src/hierarchy/mod.rs:1-2100]` — full hierarchy module read in this session; all
  function signatures, line numbers, and verbatim quotes confirmed by direct Read tool calls.
- `[VERIFIED: src/error.rs:1-142]` — ForecastError enum read in full.
- `[VERIFIED: crates/anofox-forecast-js/src/hierarchy.rs:1-186]` — JS bindings read in full.
- `[VERIFIED: .planning/REQUIREMENTS.md]` — ERM requirements ERM-01/02/03/05 read verbatim.

### Secondary (MEDIUM confidence)
- `[CITED: Ben Taieb & Koo 2019, KDD]` — ERM formula derivation; consistent with REQUIREMENTS.md
  line 23 reference. Paper not fetched; derivation independently verified via first principles.

### Tertiary (LOW confidence / ASSUMED)
- A1–A3 in Assumptions Log above.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all helpers verified by direct code read and passing test run
- Architecture: HIGH — exact line numbers, verbatim quotes, patterns confirmed
- Pitfalls: HIGH — Eq breakage confirmed by compiler rules + grep; others confirmed by code read
- Test case arithmetic: HIGH — computed by hand and double-checked via matrix algebra

**Research date:** 2026-09-09
**Valid until:** 2026-10-09 (stable library; no external dependencies)
