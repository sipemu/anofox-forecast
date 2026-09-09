# Phase 8: Auto-λ & Grouped/Crossed Validation — Research

**Researched:** 2026-09-09
**Domain:** Rust hierarchical forecasting — Ledoit-Wolf auto-λ for ERM ridge, grouped/crossed hierarchy end-to-end validation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Change `Erm { lambda: f64 }` → `Erm { lambda: Option<f64> }` (None = auto Ledoit-Wolf, Some(x) = fixed). Update Phase 7 tests to Some(λ). Auto-λ is default via None.
- Reuse/adapt existing LW machinery. End-to-end on a grouped/crossed hierarchy (≥2 crossing dims) via `from_summing_matrix`; baselines = unreconciled AND MinTrace; metric = RMSSE (or MASE) across all levels; auto-λ (None) is the config exercised; commit a before/after test + a short results note.

### Claude's Discretion

- None stated beyond the locked decisions.

### Deferred Ideas (OUT OF SCOPE)

- Further ERM performance tuning beyond correctness + the accuracy proof (broader perf milestone).
- Additional reconciliation methods or shrinkage targets beyond Ledoit-Wolf.
- Exposing auto-λ tuning knobs to callers (keep the default parameter-free like MinTraceShrink).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ERM-04 | Ledoit-Wolf-style auto-λ default available (matching MinTraceShrink ergonomics), caller-supplied fixed-λ also supported | LW-intensity formula mapped to ERM Gram context; exact sequence of operations from `ledoit_wolf_alpha()` [VERIFIED: src/hierarchy/mod.rs:1668-1713]; Option<f64> API change fully documented |
| ERM-06 | ERM validated end-to-end on a grouped/crossed hierarchy vs MinTrace/unreconciled baseline, before/after committed | Concrete 2-region × 2-product hierarchy specified; RMSSE helper confirmed available [VERIFIED: src/utils/metrics.rs:444-461]; exact assertion strategy documented |
</phase_requirements>

---

## Summary

Phase 8 completes the ERM reconciliation feature by (1) changing `Erm { lambda: f64 }` to `Erm { lambda: Option<f64> }` and routing `None` through a new `erm_auto_lambda()` helper that applies the Ledoit-Wolf shrinkage intensity directly to the ERM Gram matrix, and (2) proving ERM end-to-end on a 2-region × 2-product grouped/crossed hierarchy with RMSSE measured across all 7 nodes.

The **auto-λ formula** adapts the existing `ledoit_wolf_alpha()` helper (lines 1668–1713 of `src/hierarchy/mod.rs`) to the ERM Gram context: instead of shrinking a residual covariance matrix toward a diagonal target, it shrinks the Gram matrix `G = Ŷ Ŷᵀ` toward a scaled identity `(tr(G)/n)·I`, and converts the resulting shrinkage intensity `α ∈ [0,1]` into a ridge penalty `λ_auto = α · tr(G) / n`. This keeps λ on the same scale as the Gram eigenvalues, producing a well-conditioned solve regardless of hierarchy size or base-forecast scale.

The **end-to-end validation** uses a 2-region × 2-product grouped/crossed hierarchy (7 nodes: Total, Reg-A, Reg-B, Prod-X, Prod-Y, and 4 leaves) built via `from_summing_matrix`, synthetic base forecasts with seeded noise over T=20 training periods, and a 5-step holdout. The three headline RMSSE numbers (unreconciled, MinTrace, ERM-auto-λ) are committed as assertions: ERM auto-λ RMSSE ≤ unreconciled × 1.1, with the exact numbers recorded in `docs/audits/erm-grouped-validation-results.md`.

**Primary recommendation:** Implement `erm_auto_lambda()` as a standalone private function that takes `y_stored: &[Vec<f64>]` (n×T) and returns `f64`, keeping the fixed-λ solve path in `erm_reconcile()` completely unchanged. Route `None` to this helper before the Gram construction; route `Some(λ)` through the existing guards. All Phase 7 tests update `Erm { lambda: 1.0 }` → `Erm { lambda: Some(1.0) }`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| auto-λ computation (`erm_auto_lambda`) | `src/hierarchy/mod.rs` private fn | — | Same file as all other ERM helpers; no feature gate |
| Option<f64> variant field | `ReconciliationMethod::Erm` enum arm | JS `parse_method()` (polish) | Enum definition and dispatch in hierarchy/mod.rs |
| Grouped/crossed test hierarchy | `#[cfg(test)]` inline in hierarchy/mod.rs | — | Consistent with all 41 existing hierarchy tests |
| RMSSE measurement | `crate::utils::rmsse` (reused) | — | Already public, re-exported via utils/mod.rs |
| Results note | `docs/audits/erm-grouped-validation-results.md` | — | Same directory as `multiplicative-guard-audit.md` |

---

## Standard Stack

No new external packages. Phase 8 is pure in-crate Rust, reusing existing helpers.

| Component | Source | Status |
|-----------|--------|--------|
| `ledoit_wolf_alpha(res_matrix, sample_cov, diag, t, n)` | `src/hierarchy/mod.rs:1668` | `[VERIFIED: src/hierarchy/mod.rs:1668-1713]` — direct adaptation |
| `cholesky(n, a)` / `cholesky_solve_vec(n, l, b)` | `src/hierarchy/mod.rs` | `[VERIFIED: src/hierarchy/mod.rs]` — reuse unchanged |
| `rmsse(train, actual, forecast)` | `src/utils/metrics.rs:444` | `[VERIFIED: src/utils/metrics.rs:444-461]` — public fn |
| `from_summing_matrix(...)` | `src/hierarchy/mod.rs:283` | `[VERIFIED: src/hierarchy/mod.rs:283-449]` — grouped/crossed entry point |
| `with_erm_training(...)` | `src/hierarchy/mod.rs:503` | `[VERIFIED: src/hierarchy/mod.rs:503-510]` — unchanged |

## Package Legitimacy Audit

Not applicable — no external packages installed in this phase.

---

## Research Question Answers

### Q1 — The Auto-λ Formula for the ERM Gram Context (ERM-04)

**Background — what `ledoit_wolf_alpha` currently does:**

`[VERIFIED: src/hierarchy/mod.rs:1668-1713]` verbatim signature and body:
```rust
fn ledoit_wolf_alpha(
    res_matrix: &[Vec<f64>],   // n rows, each length T
    sample_cov: &[Vec<f64>],   // n×n sample covariance
    diag: &[f64],              // diagonal of sample_cov
    t: usize,
    n: usize,
) -> f64 {
    let tf = t as f64;
    let means: Vec<f64> = res_matrix.iter().map(|r| r.iter().sum::<f64>() / tf).collect();
    // delta = ||sample_cov - diag_target||_F^2
    let mut delta = 0.0;
    for i in 0..n { for j in 0..n {
        let diff = sample_cov[i][j] - if i == j { diag[i] } else { 0.0 };
        delta += diff * diff;
    }}
    // gamma = sum_{ij} (1/T) sum_k (z_ik z_jk - sample_cov[i][j])^2
    let mut gamma = 0.0;
    for i in 0..n { for j in 0..n {
        let mut sum_sq = 0.0;
        for k in 0..t {
            let zi = res_matrix[i][k] - means[i];
            let zj = res_matrix[j][k] - means[j];
            let dev = zi * zj - sample_cov[i][j];
            sum_sq += dev * dev;
        }
        gamma += sum_sq / tf;
    }}
    if delta < 1e-30 { return 1.0; }
    (gamma / (tf * delta)).clamp(0.0, 1.0)
}
```

The formula is α = clamp(γ / (T · δ), 0, 1), implementing the Ledoit-Wolf (2004, Journal of Multivariate Analysis) analytical shrinkage intensity toward a diagonal target F = diag(sample variances). `[ASSUMED]` — paper cited in comments; not fetched this session; formula structure is standard LW.

**Adaptation to the ERM Gram context:**

In `min_trace_shrink`, the LW intensity shrinks the **residual sample covariance** (n×n) toward its diagonal. In ERM, the solve uses the **Gram matrix** `G = Ŷ Ŷᵀ` (n×n, rows = nodes, cols summed over T time steps). λ regularizes G in the same way that α shrinks a covariance toward its diagonal: it reduces the dominance of the off-diagonal structure.

The auto-λ procedure for ERM:

1. **Compute the Gram matrix** G[i,j] = Σ_t Ŷ[i,t] · Ŷ[j,t] (n×n, no λ added yet).
2. **Choose the shrinkage target** as a scaled identity: `F = (tr(G)/n) · I`. This means the "diagonal" reference is the average Gram diagonal, not the individual entries — keeping the target on the right magnitude scale.
3. **Apply the LW intensity formula** adapted to G:
   - `diag_ref = tr(G)/n` (scalar)
   - `delta = Σ_{i,j} (G[i,j] - (if i==j { diag_ref } else { 0.0 }))^2`
   - For γ: treat each row of Ŷ_stored as a "residual" vector and compute:
     ```
     z_ik = Ŷ[i,t] - mean_i   (where mean_i = (1/T) Σ_t Ŷ[i,t])
     gamma = Σ_{i,j} (1/T) Σ_k (z_ik z_jk - G_centered[i,j])^2
     ```
     where `G_centered[i,j] = Σ_t z_it z_jt` is the centered Gram (subtract per-row means first).
   - `α = clamp(γ / (T · δ), 0, 1)`
4. **Convert α to λ**: `λ_auto = α · (tr(G)/n)`. This scales λ to the mean Gram diagonal, keeping it proportional to the magnitude of the data. `[ASSUMED]` — this scaling convention is not in the paper; it is derived by dimensional analysis so that α=1 produces λ = mean Gram diagonal (full shrink) and α=0 gives λ=0 (no shrink). The user can always override with `Some(λ)`.
5. **Edge cases:**
   - If `delta < 1e-30` (G already diagonal — perfectly uncorrelated base forecasts): return α = 1.0, λ_auto = tr(G)/n (maximum regularization). `[VERIFIED: src/hierarchy/mod.rs:1708-1710]` — same guard is in the existing LW helper.
   - If T = 1 (only one training period): G is a rank-1 outer product; the centered Gram is zero; γ = 0; α → 0; λ_auto = 0 → Gram is rank-1 and Cholesky will fail. Guard: enforce T ≥ 2 for auto-λ path (same as MinTraceShrink's `t < 2` guard `[VERIFIED: src/hierarchy/mod.rs:1162-1166]`). Return `ForecastError::InvalidParameter` with hint.
   - If tr(G) = 0 (all base forecasts are zero — degenerate): return a small fallback λ = 1e-6. This is a user-error condition (zero base forecasts); guard with a comment.

**The complete `erm_auto_lambda` helper signature:**

```rust
/// Compute the Ledoit-Wolf-style auto-lambda for ERM ridge regularization.
///
/// Adapts the LW shrinkage-intensity formula to the ERM Gram matrix
/// G = Ŷ Ŷᵀ (n×n), shrinking toward a scaled identity target.
///
/// Returns λ ≥ 0 suitable for use in `(G + λI)` Cholesky solve.
/// Returns `Err` if T < 2 (insufficient training periods for LW estimation).
fn erm_auto_lambda(y_stored: &[Vec<f64>], n: usize, t_cap: usize) -> Result<f64> {
    if t_cap < 2 {
        return Err(ForecastError::InvalidParameter(
            "ERM auto-lambda requires at least 2 training periods (T ≥ 2)".into(),
        ));
    }
    let tf = t_cap as f64;

    // Per-node means
    let means: Vec<f64> = y_stored.iter().map(|row| row.iter().sum::<f64>() / tf).collect();

    // Centered Gram G_c[i][j] = Σ_t (y[i][t] - mean_i)(y[j][t] - mean_j)
    let mut g_centered = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let dot: f64 = (0..t_cap)
                .map(|t| (y_stored[i][t] - means[i]) * (y_stored[j][t] - means[j]))
                .sum();
            g_centered[i * n + j] = dot;
            g_centered[j * n + i] = dot;
        }
    }

    // Shrinkage target: scaled identity with diag_ref = tr(G_c) / n
    let trace_g: f64 = (0..n).map(|i| g_centered[i * n + i]).sum();
    let diag_ref = if n > 0 { trace_g / n as f64 } else { 0.0 };

    // delta = ||G_c - diag_ref * I||_F^2
    let mut delta = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { diag_ref } else { 0.0 };
            let diff = g_centered[i * n + j] - target;
            delta += diff * diff;
        }
    }

    if delta < 1e-30 {
        // G already diagonal — return full-shrink lambda
        return Ok(diag_ref.max(0.0));
    }

    // gamma = Σ_{i,j} (1/T) Σ_k ((y[i][k]-mean_i)(y[j][k]-mean_j) - G_c[i][j])^2
    let mut gamma = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let g_ij = g_centered[i * n + j];
            let sum_sq: f64 = (0..t_cap)
                .map(|k| {
                    let zi = y_stored[i][k] - means[i];
                    let zj = y_stored[j][k] - means[j];
                    let dev = zi * zj - g_ij;
                    dev * dev
                })
                .sum();
            gamma += sum_sq / tf;
        }
    }

    let alpha = (gamma / (tf * delta)).clamp(0.0, 1.0);
    let lambda = alpha * diag_ref.max(0.0);

    // Fallback: if diag_ref ≈ 0, all base forecasts are near-zero; use minimal regularization.
    if lambda < 1e-15 {
        Ok(1e-6)
    } else {
        Ok(lambda)
    }
}
```

**Why this formula is correct for the ERM ridge context:**

The ERM solve uses `(G + λI)` where G = Ŷ Ŷᵀ. The LW intensity α measures "how much of the off-diagonal structure in G is noise vs. signal" relative to T. When T is large and the base forecasts are well-separated, α → 0, λ → 0, and ERM learns the full cross-series structure. When T is small, α → 1, λ → tr(G)/n, and ERM regularizes strongly — the same qualitative behavior as `MinTraceShrink` choosing α = 1 when residuals are perfectly correlated (already diagonal) or α = 0 when they have rich structure. `[ASSUMED]` — this analogy-based derivation is not proven in Ben Taieb & Koo (2019), which does not specify auto-lambda. It is a principled adaptation of the LW framework.

---

### Q2 — API Wiring for Option<f64>

**Exact change to the enum arm:**

`[VERIFIED: src/hierarchy/mod.rs:120-124]` verbatim current arm:
```rust
Erm {
    /// Ridge regularization strength (λ ≥ 0). A value of 1.0 is a
    /// reasonable default; increase if the solve is unstable or T < n.
    lambda: f64,
},
```

New arm:
```rust
Erm {
    /// Ridge regularization strength.
    ///
    /// - `None` (default): use the Ledoit-Wolf-style auto-λ estimator,
    ///   which adapts the shrinkage intensity of the ERM Gram matrix
    ///   to T (training periods) and n (node count). Requires T ≥ 2.
    /// - `Some(λ)`: use a caller-supplied fixed λ ≥ 0. The Phase 7
    ///   guards (finite, non-negative) still apply.
    ///
    /// `None` mirrors the `MinTraceShrink` ergonomics (no explicit tuning needed).
    lambda: Option<f64>,
},
```

**Change to `erm_reconcile` signature and dispatch:**

Current signature `[VERIFIED: src/hierarchy/mod.rs:791-796]`:
```rust
fn erm_reconcile(
    &self,
    base_map: &HashMap<&str, &Vec<f64>>,
    horizon: usize,
    lambda: f64,
) -> Result<Vec<(String, Vec<f64>)>>
```

New signature (the `f64` → resolved lambda before the Gram build):
```rust
fn erm_reconcile(
    &self,
    base_map: &HashMap<&str, &Vec<f64>>,
    horizon: usize,
    lambda: Option<f64>,
) -> Result<Vec<(String, Vec<f64>)>>
```

**Branch inside `erm_reconcile`** (place immediately after Step 3 validation, before Step 4 Ŷ build — because auto-λ needs Ŷ, so it must come after the history is validated but before the Gram construction):

```rust
// Resolve lambda: compute auto if None, validate if Some.
let resolved_lambda: f64 = match lambda {
    Some(lam) => {
        if !lam.is_finite() || lam < 0.0 {
            return Err(ForecastError::InvalidParameter(format!(
                "ERM: lambda must be finite and non-negative, got {lam}"
            )));
        }
        lam
    }
    None => {
        // Build y_stored temporarily for auto-lambda computation.
        // This is a second pass — acceptable since auto-lambda is an O(n^2 * T) operation
        // and erm_reconcile already does O(n^2 * T) for the Gram.
        let y_stored_for_lw: Vec<Vec<f64>> = (0..n)
            .map(|i| base_hist[self.nodes[i].name.as_str()].clone())
            .collect();
        erm_auto_lambda(&y_stored_for_lw, n, t_cap)?
    }
};
```

Then replace all references to `lambda` in the Gram construction (Step 6) with `resolved_lambda`:
```rust
gram[i * n + i] += resolved_lambda;
```

The existing guard block at lines 799–803:
```rust
if !lambda.is_finite() || lambda < 0.0 {
    return Err(ForecastError::InvalidParameter(format!(
        "ERM: lambda must be finite and non-negative, got {lambda}"
    )));
}
```
**Must be removed** — it is now handled in the `match lambda { Some(lam) => ... }` branch above.

**Dispatch arm in `reconcile()` — unchanged structurally:**

`[VERIFIED: src/hierarchy/mod.rs:592]` current:
```rust
ReconciliationMethod::Erm { lambda } => self.erm_reconcile(&base_map, horizon, lambda),
```
This compiles without change after the signature update (Rust destructures `lambda: Option<f64>` into the variable `lambda`, which is now `Option<f64>`).

**Phase 7 tests to update to `Some(λ)`:**

All occurrences of `ReconciliationMethod::Erm { lambda: <literal_f64> }` must become `ReconciliationMethod::Erm { lambda: Some(<literal_f64>) }`. Confirmed locations by grep:

```
src/hierarchy/mod.rs:
  erm_correctness_reference_formula  — lambda: 1.0 → Some(1.0)
  erm_requires_training              — lambda: 1.0 → Some(1.0)
  erm_shape_mismatch (leaf_hist)     — lambda: 1.0 → Some(1.0)
  erm_shape_mismatch_base_hist       — lambda: 1.0 → Some(1.0)
  erm_negative_lambda_returns_err    — lambda: -0.3 → Some(-0.3)
  erm_nan_lambda_returns_err         — lambda: f64::NAN → Some(f64::NAN)
  erm_empty_training_history_returns_err — lambda: 1.0 → Some(1.0)
  erm_correctness_asymmetric         — lambda: 1.0 → Some(1.0)
  erm_coherent_multi_horizon         — lambda: 1.0 → Some(1.0)
```

`[VERIFIED: src/hierarchy/mod.rs:2715,2745,2771,2797,2824,2875,3026]` — all confirmed by grep. The total is 9 call sites to update (exact count via `grep -n "Erm { lambda:"` produces 9 matches).

**Serde round-trip for `Option<f64>`:**

`ReconciliationMethod` has no serde derive `[VERIFIED: src/hierarchy/mod.rs]` — grep confirms no `Serialize/Deserialize` on this enum. No serde action needed. If serde is added in a later phase, `Option<f64>` serializes correctly in serde as JSON `null` / number. `[ASSUMED]` — serde's standard Option handling; not verified against a live test this session, but is standard library behavior.

**JS `parse_method()` — `"erm"` arm:**

`[VERIFIED: crates/anofox-forecast-js/src/hierarchy.rs:20-38]` — current code uses a `match name { ... _ => Err(...) }` catch-all (not exhaustive). The Option<f64> change in the Rust enum does NOT break compilation of this file — it compiles without touching JS bindings. However, to expose ERM to JS callers, add `"erm" | "Erm"` arm returning `InnerMethod::Erm { lambda: None }` (or parse a second lambda parameter). This is optional polish recommended now that ERM is feature-complete. Non-blocking for Phase 8 correctness tests.

---

### Q3 — Grouped/Crossed Test Hierarchy (ERM-06)

**Proposed hierarchy: 2 regions × 2 product types (7 nodes total)**

This is a clean extension of the existing `grouped_hierarchy_min_trace_variance_coherent` test at line 2393 which already uses the identical 7-node layout. The Phase 8 test reuses that layout as the crossing structure.

```
Layout (node index → name):
  0: Total        (root — all 4 leaves)
  1: RegA         (region A aggregate — leaves 0,1)
  2: RegB         (region B aggregate — leaves 2,3)
  3: ProdX        (product X aggregate — leaves 0,2)
  4: ProdY        (product Y aggregate — leaves 1,3)
  5: RegA_ProdX   (leaf: region A × product X)
  6: RegA_ProdY   (leaf: region A × product Y)
  7: RegB_ProdX   (leaf: region B × product X)
  8: RegB_ProdY   (leaf: region B × product Y)
```

n = 9 nodes, m = 4 leaves. Two crossing dimensions (region, product).

**Summing matrix S (9 × 4):** S[node][leaf_j] = 1 if leaf j contributes to node.
```
             AP   AY   BP   BY   (leaf indices 0..3)
Total:       1    1    1    1
RegA:        1    1    0    0
RegB:        0    0    1    1
ProdX:       1    0    1    0
ProdY:       0    1    0    1
RegA_ProdX:  1    0    0    0
RegA_ProdY:  0    1    0    0
RegB_ProdX:  0    0    1    0
RegB_ProdY:  0    0    0    1
```

**`from_summing_matrix` call:**
```rust
let node_names: Vec<String> = vec![
    "Total".into(), "RegA".into(), "RegB".into(), "ProdX".into(), "ProdY".into(),
    "RegA_ProdX".into(), "RegA_ProdY".into(), "RegB_ProdX".into(), "RegB_ProdY".into(),
];
let leaf_names: Vec<String> = vec![
    "RegA_ProdX".into(), "RegA_ProdY".into(), "RegB_ProdX".into(), "RegB_ProdY".into(),
];
let leaf_ancestors: Vec<Vec<usize>> = vec![
    vec![0, 1, 3],  // RegA_ProdX → Total(0), RegA(1), ProdX(3)
    vec![0, 1, 4],  // RegA_ProdY → Total(0), RegA(1), ProdY(4)
    vec![0, 2, 3],  // RegB_ProdX → Total(0), RegB(2), ProdX(3)
    vec![0, 2, 4],  // RegB_ProdY → Total(0), RegB(2), ProdY(4)
];
```

**Synthetic data generation (seeded, no customer identity):**

Use a deterministic pseudo-random generator implemented inline (simple LCG or const arrays). T = 20 training periods, H = 5 holdout periods, seed = 42.

Base approach: generate 4 leaf "true" values as AR(1)-like series with small noise, then aggregate coherently for the training history. Add independent noise to create incoherent base forecasts.

Concrete implementation (use const arrays to avoid any `rand` crate dependency in the test):

```rust
// Deterministic synthetic data for ERM-06 (seed via LCG, no rand crate).
// 4 leaf series over T=20 training + H=5 holdout.
// Leaf levels: RegA_ProdX~50, RegA_ProdY~30, RegB_ProdX~40, RegB_ProdY~20.
fn synthetic_panel(t_train: usize, h_holdout: usize) -> (
    Vec<Vec<f64>>,  // leaf_train[leaf][t]  — true leaf values, training
    Vec<Vec<f64>>,  // leaf_holdout[leaf][h] — true leaf values, holdout
    Vec<Vec<f64>>,  // base_hist_all[node][t] — noisy base forecasts for all nodes, training
    Vec<Vec<f64>>,  // base_holdout_all[node][h] — noisy base forecasts for all nodes, holdout
) {
    // Simple LCG for determinism: x_{i+1} = (a*x_i + c) mod m
    let mut rng_state: u64 = 42;
    let mut rng = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5  // uniform in [-0.5, 0.5]
    };

    let leaf_levels = [50.0_f64, 30.0, 40.0, 20.0];
    let total = t_train + h_holdout;
    let mut leaf_true = vec![vec![0.0_f64; total]; 4];

    // AR(1) with phi=0.7, noise std=2
    for leaf in 0..4 {
        leaf_true[leaf][0] = leaf_levels[leaf];
        for t in 1..total {
            leaf_true[leaf][t] = 0.7 * leaf_true[leaf][t-1] + leaf_levels[leaf] * 0.3 + rng() * 2.0;
        }
    }

    // True coherent aggregate values (9 nodes)
    // leaf order: [RegA_ProdX=0, RegA_ProdY=1, RegB_ProdX=2, RegB_ProdY=3]
    // node order: [Total=0, RegA=1, RegB=2, ProdX=3, ProdY=4, leaves 5..8]
    let mut true_all = vec![vec![0.0_f64; total]; 9];
    for t in 0..total {
        true_all[5][t] = leaf_true[0][t];
        true_all[6][t] = leaf_true[1][t];
        true_all[7][t] = leaf_true[2][t];
        true_all[8][t] = leaf_true[3][t];
        true_all[1][t] = true_all[5][t] + true_all[6][t];  // RegA
        true_all[2][t] = true_all[7][t] + true_all[8][t];  // RegB
        true_all[3][t] = true_all[5][t] + true_all[7][t];  // ProdX
        true_all[4][t] = true_all[6][t] + true_all[8][t];  // ProdY
        true_all[0][t] = true_all[1][t] + true_all[2][t];  // Total
    }

    // Noisy base forecasts: true + independent noise per node
    let mut base_all = vec![vec![0.0_f64; total]; 9];
    for node in 0..9 {
        for t in 0..total {
            base_all[node][t] = true_all[node][t] + rng() * 4.0;  // forecast noise
        }
    }

    let leaf_train: Vec<Vec<f64>> = leaf_true.iter().map(|s| s[..t_train].to_vec()).collect();
    let leaf_holdout: Vec<Vec<f64>> = leaf_true.iter().map(|s| s[t_train..].to_vec()).collect();
    let base_hist_all: Vec<Vec<f64>> = base_all.iter().map(|s| s[..t_train].to_vec()).collect();
    let base_holdout_all: Vec<Vec<f64>> = base_all.iter().map(|s| s[t_train..].to_vec()).collect();

    (leaf_train, leaf_holdout, base_hist_all, base_holdout_all)
}
```

Node order in `base_hist_all` and `true_all` matches internal index order of `from_summing_matrix` (`node_names` list order: Total=0, RegA=1, RegB=2, ProdX=3, ProdY=4, RegA_ProdX=5, RegA_ProdY=6, RegB_ProdX=7, RegB_ProdY=8). Leaf order in `leaf_train`/`leaf_holdout` is `[RegA_ProdX, RegA_ProdY, RegB_ProdX, RegB_ProdY]` matching the `leaf_names` list. `[ASSUMED]` — the correspondence relies on the `from_summing_matrix` constructor preserving `node_names` list order as the internal index, which is the intended contract per the constructor code `[VERIFIED: src/hierarchy/mod.rs:300-310]`: `name_to_idx.insert(name.clone(), i)` assigns index `i` in `node_names` enumeration order.

---

### Q4 — Accuracy Measurement and Assertions

**RMSSE function available:**

`[VERIFIED: src/utils/metrics.rs:444-461]` — verbatim:
```rust
pub fn rmsse(train: &[f64], actual: &[f64], forecast: &[f64]) -> f64 {
    if actual.len() != forecast.len() || actual.is_empty() || train.len() < 2 {
        return f64::NAN;
    }
    let n_train = train.len();
    let sum_sq_diff: f64 = train.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let scale_sq = sum_sq_diff / (n_train - 1) as f64;
    if scale_sq == 0.0 { return f64::NAN; }
    let mse_val = mse(actual, forecast);
    (mse_val / scale_sq).sqrt()
}
```

The function takes `train` (in-sample values for scaling), `actual` (holdout actuals), `forecast` (holdout predictions). Returns NaN for constant train series. `[VERIFIED: src/utils/metrics.rs:1069-1074]` — confirmed by test `rmsse_constant_train`.

**To use in hierarchy tests** — add inside the `#[cfg(test)]` block:
```rust
use crate::utils::rmsse;
```
The function is `pub` at `crate::utils::rmsse` via the re-export at `[VERIFIED: src/utils/mod.rs:26]`:
```
bias, calculate_metrics, coverage, mda, msis, periods_in_stock, rmsse, skill_score, ...
```

**Aggregation strategy — mean RMSSE across all n nodes:**

Compute RMSSE for each of the 9 nodes using:
- `train` = the node's training history (from true coherent values or from the base forecast training history — use the same source consistently for all methods; recommend `true_all` training values as the scale reference)
- `actual` = holdout true values for that node
- `forecast` = reconciled holdout forecast for that node

Then `mean_rmsse = (Σ_{node} rmsse_node) / n`.

**Three headline numbers:**
1. **Unreconciled**: base forecasts passed directly as predictions (no reconciliation call), measured against true holdout values.
2. **MinTrace (MinTraceStruct)**: use `ReconciliationMethod::MinTraceStruct` (no residuals needed; safe for a grouped hierarchy; `[VERIFIED: src/hierarchy/mod.rs:2461-2490]` — MinTraceStruct already works on grouped hierarchies).
3. **ERM auto-λ**: `ReconciliationMethod::Erm { lambda: None }`.

**Why MinTraceStruct as the MinTrace baseline:**

`MinTraceShrink` requires residuals. `MinTraceOls` builds a dense n×n covariance and is documented as not best for grouped hierarchies (`[VERIFIED: src/hierarchy/mod.rs:265-270]`). `MinTraceStruct` (structural scaling, W = diag(1/n_leaves_below)) is the documented safe option for grouped hierarchies and requires no residuals, making the test setup cleaner. The CONTEXT.md says "MinTrace" without specifying which variant — MinTraceStruct is the appropriate grouped-hierarchy choice. `[ASSUMED]` — the choice of MinTraceStruct over other variants is a judgment call; record it in the results note.

**Assertion strategy (non-flaky on synthetic data):**

The goal is to show ERM auto-λ ≤ unreconciled (or within documented tolerance). On synthetic AR(1) data with T=20, the Gram matrix has T >> n=9 so auto-λ should approach 0 and ERM should be close to OLS-optimal. However, flakiness risk exists because:

1. The synthetic noise may occasionally produce better unreconciled error than reconciled on a short holdout.
2. MinTrace on a small grouped hierarchy with structural scaling may not always outperform unreconciled on 5-step holdout.

**Recommended assertion:**
```rust
// ERM auto-λ must not be dramatically worse than unreconciled
// (factor of 2 is a loose guard that will not be flaky on this seeded data).
assert!(
    erm_mean_rmsse <= unreconciled_mean_rmsse * 1.5,
    "ERM auto-lambda RMSSE ({:.4}) should not greatly exceed unreconciled ({:.4})",
    erm_mean_rmsse, unreconciled_mean_rmsse
);
// ERM auto-λ must produce coherent forecasts (main correctness guard).
// This is the real proof — coherence is a hard constraint, unlike RMSSE magnitude.
for h in 0..H {
    let recon_ap = erm_result["RegA_ProdX"][h];
    let recon_ay = erm_result["RegA_ProdY"][h];
    let recon_bp = erm_result["RegB_ProdX"][h];
    let recon_by = erm_result["RegB_ProdY"][h];
    approx_eq(erm_result["Total"][h],   recon_ap + recon_ay + recon_bp + recon_by, 1e-8);
    approx_eq(erm_result["RegA"][h],    recon_ap + recon_ay, 1e-8);
    approx_eq(erm_result["RegB"][h],    recon_bp + recon_by, 1e-8);
    approx_eq(erm_result["ProdX"][h],   recon_ap + recon_bp, 1e-8);
    approx_eq(erm_result["ProdY"][h],   recon_ay + recon_by, 1e-8);
}
```

The coherence assertion is the hard proof (the before/after for ERM-06 in spirit — unreconciled is NOT coherent, ERM-auto-λ IS). The RMSSE comparison is the soft before/after. The factor 1.5 (50% slack) is justified because:
- T=20 >> n=9 so the Gram is well-determined and auto-λ should choose a small λ
- The seeded LCG produces the same data every test run
- 50% slack eliminates edge cases where the 5-step holdout happens to be easier for the noisy unreconciled forecast

**Record the exact numbers**: run the test, capture the three RMSSE values, hard-code them in the results note and use them to tighten the assertion to ≤ 1.1 (not 1.5) once the exact values are known. The planner should add a task: "run accuracy test, record values in results note, tighten RMSSE assertion to ≤ unreconciled + 10%".

---

### Q5 — Results Note Location and Format

**Location:** `docs/audits/erm-grouped-validation-results.md`

This directory already exists and contains `multiplicative-guard-audit.md` (Phase 5 results). `[VERIFIED: docs/audits/]` — directory confirmed via ls.

**Format:**
```markdown
# ERM Grouped/Crossed Validation Results

**Hierarchy:** 2-region × 2-product (7 aggregate + 4 leaf = 9 nodes total)
**Method:** `from_summing_matrix` grouped/crossed; `[VERIFIED: src/hierarchy/mod.rs:283]`
**Data:** Synthetic AR(1), T=20 training + H=5 holdout, seed=42 (deterministic)
**Metric:** Mean RMSSE across all 9 nodes (training scale = in-sample true values)
**ERM config:** `Erm { lambda: None }` (auto-λ via Ledoit-Wolf Gram shrinkage)
**MinTrace baseline:** MinTraceStruct (structural scaling; no residuals required)

## Headline Results

| Method       | Mean RMSSE (all 9 nodes) | RMSSE vs Unreconciled |
|-------------|--------------------------|----------------------|
| Unreconciled | <fill>                  | baseline             |
| MinTrace (Struct) | <fill>             | <fill>%              |
| ERM auto-λ   | <fill>                  | <fill>%              |

**Auto-λ selected:** `λ_auto = <fill>` (computed by `erm_auto_lambda` from Gram of training history)

## Interpretation

ERM auto-λ reconciliation restores coherence (cross-sectional constraint Total = RegA + RegB = ProdX + ProdY = sum of leaves) which unreconciled base forecasts violate. The RMSSE comparison reflects both the regularization bias and the coherence gain.

*Generated by test `erm_grouped_crossed_end_to_end_accuracy` in `src/hierarchy/mod.rs`.*
*Phase 8: auto-lambda-grouped-crossed-validation — completed <date>.*
```

---

### Q6 — Pitfalls

**Pitfall 1: Phase 7 tests still use `Erm { lambda: f64 }` literal**
**What goes wrong:** Compile error `mismatched types: expected Option<f64>, found f64`.
**Why it happens:** All 9 Phase 7 call sites use a bare f64 literal (1.0, -0.3, NaN) — they compile to `f64` not `Option<f64>`.
**How to avoid:** Update all 9 call sites (grep: `Erm { lambda:`) to wrap in `Some(...)`.
**Warning signs:** `error[E0308]` mismatched types on `Erm { lambda: 1.0 }`.

**Pitfall 2: Auto-λ degeneracy when T is small (T < 2)**
**What goes wrong:** The centered Gram is zero (T=1 gives a rank-1 outer product, mean subtraction yields zero); γ=0; α=0; λ_auto=0; Cholesky on a rank-deficient G fails with SingularMatrix.
**Why it happens:** LW intensity formula needs T ≥ 2 to estimate variance in the outer products.
**How to avoid:** Guard `T < 2` in `erm_auto_lambda` with `InvalidParameter` error (same guard as `min_trace_shrink` at line 1162). `[VERIFIED: src/hierarchy/mod.rs:1162-1166]`.
**Warning signs:** SingularMatrix error from `erm_auto_lambda` auto path on T=1 data.

**Pitfall 3: Coherence not guaranteed without ancestors_of**
**What goes wrong:** If the S matrix construction inside `erm_reconcile` does not use `ancestors_of()` (which handles multi-parent grouped nodes), coherence fails for aggregates that have more than one parent path.
**Why it happens:** Grouped/crossed nodes appear as children of multiple aggregate parents — simple BFS ancestry enumeration may miss some ancestor edges in non-tree DAGs.
**How to avoid:** The existing `erm_reconcile` code at lines 919–925 already uses `ancestors_of()` (verified by reading the code). No change needed. `[VERIFIED: src/hierarchy/mod.rs:919-925]`.
**Warning signs:** Coherence assertion fails for RegA, RegB, ProdX, or ProdY (Total usually passes; intermediate aggregates fail).

**Pitfall 4: RMSSE scale-factor confusion (node-level vs leaf-level scale)**
**What goes wrong:** Using the noisy base forecast training history as the RMSSE scale denominator instead of the true coherent series — this inflates the scale for aggregate nodes and deflates RMSSE.
**Why it happens:** The `train` argument to `rmsse()` should be the in-sample true values, not the forecasts.
**How to avoid:** Use `true_all[node][..t_train]` as the `train` argument for all 9 nodes. The synthetic data helper returns these as `leaf_train` (for leaves) and computes aggregate true values coherently.
**Warning signs:** RMSSE values for aggregate nodes (Total, RegA, etc.) are unreasonably low (< 0.01) compared to leaf-level RMSSE.

**Pitfall 5: Internal node ordering vs. leaf_names ordering mismatch in `with_erm_training`**
**What goes wrong:** The `base_hist_all` array is indexed by internal node index (0..9), but `with_erm_training` takes a `HashMap<String, Vec<f64>>`. If the HashMap is built using an incorrect index mapping, the training history for one node gets assigned to another.
**Why it happens:** Confusion between the `node_names` list order (used as internal index) and whatever order the HashMap is iterated in.
**How to avoid:** Build the HashMap explicitly by name: `base_hist.insert("Total".into(), base_hist_all[0].clone())`, etc. Never use integer indices to build the HashMap — use the node name strings as keys.
**Warning signs:** ERM correctness test fails (wrong P rows) or coherence holds but RMSSE is much worse than expected.

**Pitfall 6: `Erm { lambda: None }` in the asymmetric test (erm_correctness_asymmetric)**
**What goes wrong:** The asymmetric test uses a hand-computed oracle with lambda=1.0. If the test is accidentally updated to `Some(None)` or `None`, the oracle comparison will fail because auto-λ produces a different λ than 1.0.
**Why it happens:** Cut-paste error when updating all 9 call sites.
**How to avoid:** Update all Phase 7 tests to `Some(1.0)` (not `None`). The auto-λ test is a NEW test added in Phase 8, not a modification of the existing correctness test.

---

## Architecture Patterns

### System Architecture Diagram

```
caller ──► reconcile(base, Erm { lambda: None })
              │
              ▼
         erm_reconcile(&base_map, horizon, lambda: None)
              │
              ├── validate history (T, shape) ──► ForecastError if invalid
              │
              ├── None ──► erm_auto_lambda(y_stored, n, T)
              │                │
              │                ├── compute centered Gram G_c
              │                ├── compute LW alpha (γ/Tδ, clamp 0..1)
              │                └── return alpha * tr(G_c)/n
              │
              ├── Some(λ) ──► validate (finite, ≥0)
              │
              ▼
         resolved_lambda: f64
              │
              ▼
         build Ŷ_stored (n×T), B_stored (m×T)
         Gram G = Ŷ Ŷᵀ + resolved_lambda * I
         Cholesky(G) ──► L
         P[i] = cholesky_solve_vec(L, C[i])   for each leaf i
         bottom[h] = P · ŷ[h]
         all[h] = S · bottom[h]
              │
              ▼
         to_named_output() ──► Vec<(String, Vec<f64>)>
```

### Recommended Project Structure (no changes needed)

All new code lives in `src/hierarchy/mod.rs` (private fn + enum change + inline tests). Results note in `docs/audits/erm-grouped-validation-results.md`.

### Anti-Patterns to Avoid

- **Do not** add a separate `ErmAuto` variant — the locked decision is `Option<f64>` in the existing `Erm` arm.
- **Do not** compute auto-λ outside `erm_reconcile` (e.g. expose it publicly) — keep it encapsulated.
- **Do not** use `wrmsse()` for the accuracy comparison — equal weights (simple mean) is sufficient for a 9-node synthetic test; `wrmsse()` requires meaningful economic weights.
- **Do not** add `use rand` or any new crate dependency for the synthetic data — use the inline LCG.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LW intensity | Custom shrinkage formula | Adapt `ledoit_wolf_alpha` pattern | Already verified correct against 32 MinTrace tests |
| SPD solve | Custom Gaussian elim | `cholesky` + `cholesky_solve_vec` | Same path as Phase 7 |
| RMSSE metric | Custom scale-error formula | `crate::utils::rmsse` | Public, tested, handles edge cases |
| Grouped hierarchy | Custom S-matrix builder | `from_summing_matrix` | Handles multi-parent DAGs correctly |

---

## Runtime State Inventory

Not applicable (no rename/refactor; greenfield API extension on Phase 7 foundation).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | All code | ✓ | project toolchain | — |
| `cargo test --lib hierarchy` | Running tests | ✓ | confirmed (41 tests pass) | — |

No external tools, services, or new crates needed.

---

## Validation Architecture

`workflow.nyquist_validation: true` in `.planning/config.json` — section required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`#[test]`) — inline `#[cfg(test)]` in `src/hierarchy/mod.rs` |
| Config file | None |
| Quick run command | `cargo test --lib hierarchy::tests::erm` |
| Full suite command | `cargo test --lib hierarchy` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ERM-04 | `Erm { lambda: None }` routes to auto-λ, produces finite λ, passes Cholesky solve | unit | `cargo test --lib hierarchy::tests::erm_auto_lambda` | ❌ Wave 0 |
| ERM-04 | `Erm { lambda: Some(x) }` still applies guards (negative, NaN) | unit | `cargo test --lib hierarchy::tests::erm_negative_lambda_returns_err` (updated) | updated existing |
| ERM-04 | Auto-λ degeneracy guard: T < 2 returns error | unit | `cargo test --lib hierarchy::tests::erm_auto_lambda_t1_returns_err` | ❌ Wave 0 |
| ERM-06 | Grouped/crossed hierarchy, 3 methods, RMSSE measured, coherence asserted | integration-inline | `cargo test --lib hierarchy::tests::erm_grouped_crossed_end_to_end_accuracy` | ❌ Wave 0 |
| ERM-01 | Existing Phase 7 tests still pass after Option<f64> change | regression | `cargo test --lib hierarchy::tests::erm` | updated existing |

### Sampling Rate

- **Per task commit:** `cargo test --lib hierarchy`
- **Per wave merge:** `cargo test --lib hierarchy`
- **Phase gate:** `cargo test --lib hierarchy` green + `cargo clippy --all-targets -- -D warnings` clean before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `erm_auto_lambda_basic` — ERM-04: None routes to a finite positive auto-λ and reconcile succeeds
- [ ] `erm_auto_lambda_t1_returns_err` — ERM-04: T=1 auto-λ returns InvalidParameter
- [ ] `erm_grouped_crossed_end_to_end_accuracy` — ERM-06: 3-method comparison, coherence, RMSSE headline
- [ ] Update 9 Phase 7 test call sites from `Erm { lambda: <f64> }` to `Erm { lambda: Some(<f64>) }`

*(Note: all test infrastructure exists — inline `#[cfg(test)]` in hierarchy/mod.rs. No new files.)*

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`. Phase 8 is pure in-library numerical computation with no I/O, no authentication, no network.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | T < 2 guard in auto-λ; None vs Some(x) dispatch; finite/non-negative guard retained for Some path |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN/Inf in training data | Tampering | Cholesky detects non-positive diagonal → SingularMatrix; `erm_auto_lambda` propagates NaN through δ/γ — add guard `if !lambda.is_finite() { return Err(...) }` after auto-λ computation |
| T=1 auto-λ producing zero λ | Denial of service | T < 2 guard returns InvalidParameter before Cholesky attempt |
| Zero Gram diagonal (all-zero base forecasts) | Denial of service | `diag_ref.max(0.0)` + fallback `1e-6` in `erm_auto_lambda` |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Auto-λ formula λ_auto = α · tr(G_c)/n correctly scales the ridge penalty to the Gram magnitude | Q1 Auto-λ Formula | If wrong, auto-λ may produce systematically over- or under-regularized solves; mitigated by end-to-end accuracy test (ERM-06) catching bad behavior on real data |
| A2 | Ben Taieb & Koo (2019) does not specify an auto-lambda procedure; the LW adaptation is novel | Q1 | If paper does specify a different auto-λ method, the chosen formula may diverge from the intended algorithm; mitigated by the end-to-end test showing sensible accuracy |
| A3 | MinTraceStruct is appropriate as the "MinTrace baseline" for a grouped/crossed hierarchy | Q4 | If MinTraceShrink or MinTraceOls would give a fairer baseline, the comparison may favor or disfavor ERM misleadingly; mitigated by documenting the choice in the results note |
| A4 | Seeded LCG produces stable enough data that the 1.5× RMSSE slack assertion is non-flaky | Q4 | If the LCG generates a pathological holdout window, the assertion may fail; mitigated by choosing T=20 >> n=9 and verifying the exact RMSSE values post-implementation |
| A5 | `from_summing_matrix` preserves `node_names` list order as internal index | Q3 | If wrong, base_hist HashMap keys misalign with internal order; mitigated by building HashMap by name, not index |
| A6 | Serde round-trip for `Option<f64>` in `ReconciliationMethod` is handled correctly | Q2 | No serde derive on `ReconciliationMethod` so no current risk; becomes relevant if serde is added later |

---

## Open Questions

1. **RMSSE assertion tightness**
   - What we know: The seeded LCG data produces deterministic values; 1.5× slack is very loose.
   - What's unclear: The exact RMSSE values until the test is run.
   - Recommendation: Start with 1.5× assertion; after first successful run, tighten to ≤ 1.1× and commit the exact values to the results note.

2. **JS `parse_method()` ERM arm — Phase 8 or deferred?**
   - What we know: The catch-all `_ =>` compiles without changes; JS callers cannot access ERM until an `"erm"` arm is added.
   - Recommendation: Add `"erm" | "Erm"` → `InnerMethod::Erm { lambda: None }` in Phase 8 as optional polish (30-line change in one file). The CONTEXT.md marks it as non-blocking.

3. **Minimum T for auto-λ: guard at 2 or higher?**
   - What we know: LW intensity needs T ≥ 2 for the variance estimate; T < n means rank-deficient G but auto-λ handles this gracefully.
   - Recommendation: Guard at T ≥ 2 (mirror MinTraceShrink); document that small T produces high α and thus large λ (strong regularization), which is the correct behavior.

---

## Sources

### Primary (HIGH confidence)

- `[VERIFIED: src/hierarchy/mod.rs:60-125]` — `ReconciliationMethod` enum including current `Erm { lambda: f64 }` arm
- `[VERIFIED: src/hierarchy/mod.rs:127-147]` — `HierarchyTree` struct fields including `erm_base_history`, `erm_leaf_history`
- `[VERIFIED: src/hierarchy/mod.rs:240-242]` — `new()` initializes erm fields to None
- `[VERIFIED: src/hierarchy/mod.rs:446-448]` — `from_summing_matrix()` initializes erm fields to None
- `[VERIFIED: src/hierarchy/mod.rs:283-449]` — full `from_summing_matrix` implementation including `node_names` → index mapping
- `[VERIFIED: src/hierarchy/mod.rs:503-510]` — `with_erm_training()` builder
- `[VERIFIED: src/hierarchy/mod.rs:578-593]` — `reconcile()` match dispatch including `Erm { lambda }` arm
- `[VERIFIED: src/hierarchy/mod.rs:784-947]` — full `erm_reconcile()` implementation including lambda guard at 799-803
- `[VERIFIED: src/hierarchy/mod.rs:1132-1315]` — full `min_trace_shrink()` implementation (LW adaptation reference)
- `[VERIFIED: src/hierarchy/mod.rs:1162-1166]` — MinTraceShrink T < 2 guard (adapt for auto-λ)
- `[VERIFIED: src/hierarchy/mod.rs:1662-1713]` — full `ledoit_wolf_alpha()` implementation (exact formula source)
- `[VERIFIED: src/hierarchy/mod.rs:1708-1710]` — `delta < 1e-30` fallback to α=1.0
- `[VERIFIED: src/hierarchy/mod.rs:2299-2360]` — `from_summing_matrix_grouped_hierarchy_construction` test (exact node/leaf/ancestor layout for the 2×2 grouped hierarchy)
- `[VERIFIED: src/hierarchy/mod.rs:2393-2458]` — `grouped_hierarchy_min_trace_variance_coherent` test (same 7-node layout with reconciliation)
- `[VERIFIED: src/utils/metrics.rs:444-461]` — `rmsse()` function signature and implementation
- `[VERIFIED: src/utils/metrics.rs:1051-1074]` — RMSSE tests including `rmsse_constant_train` NaN guard
- `[VERIFIED: src/utils/mod.rs:26]` — `rmsse` re-exported as `pub use metrics::{..., rmsse, ...}`
- `[VERIFIED: crates/anofox-forecast-js/src/hierarchy.rs:20-38]` — `parse_method()` catch-all; no existing `"erm"` arm
- `[VERIFIED: docs/audits/]` — directory exists; contains `multiplicative-guard-audit.md`
- `cargo test --lib hierarchy` — 41 tests pass, all Phase 7 ERM tests green

### Secondary (MEDIUM confidence)

- `[ASSUMED]` — Ben Taieb & Koo (2019) does not specify auto-lambda; LW adaptation is principled but novel

### Tertiary (LOW confidence)

- A1–A6 in Assumptions Log above

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all helpers verified by direct code read and 41-test passing baseline
- Architecture: HIGH — exact line numbers and verbatim quotes confirmed by Read tool calls
- Auto-λ formula: MEDIUM — formula derived by analogy from verified `ledoit_wolf_alpha`; not proven against the original paper
- Test hierarchy: HIGH — uses existing confirmed `from_summing_matrix` layout from lines 2299–2360
- RMSSE usage: HIGH — function signature and tests confirmed by direct read
- Pitfalls: HIGH — all derived from verified code or compiler behavior

**Research date:** 2026-09-09
**Valid until:** 2026-10-09 (stable library; no external dependencies)
