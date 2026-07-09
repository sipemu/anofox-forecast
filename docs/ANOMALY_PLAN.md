# Anomaly-detection port — evaluation + implementation plan

**Source**: [microprediction/timemachines heads/mahalanobis.py](https://github.com/microprediction/timemachines/blob/main/src/timemachines/heads/mahalanobis.py) — 611 LOC, MIT.

## What it is

A **streaming Mahalanobis anomaly detector** that wraps any distributional forecaster and emits, per tick:

- `d2` — Mahalanobis distance of the k-vector of standardised surprises
- `p_value` — calibrated tail probability (~ Uniform(0,1) under a well-specified forecaster)
- `run` — consecutive ticks above the guard level (1 = point outlier, growing = changepoint)

**Key insight**: don't detect anomalies on raw `y`. Detect on the k-vector `z_t = [z_1_step, z_2_step, ..., z_k_step]` of standard-normal surprises where `z_m` is the standard-normal quantile of the PIT of `y_t` under the predictive **issued m ticks ago for m steps ahead**. This vector is under calibration ~ N(0, Σ) — exactly the setting classical multivariate outlier detection assumes.

## Architecture

Two layers stack on top of a base streaming forecaster:

```
  observation y
       │
       ▼
  ┌─────────────────────────┐
  │ base forecaster         │   emits k horizon Gaussians / step
  │ (e.g. LaplaceForecaster)│
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ Parade wrapper          │   PIT + z-vector bookkeeping;
  │ (state[z], state[pit])  │   pass-through for forecasts
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ Mahalanobis detector    │   d2, p_value, run
  │ (state[d2], state[p])   │
  └─────────────────────────┘
```

Optional: **zbank** replaces the base forecaster with a bank of forecasters at different `(scale_alpha, stride)` gridpoints, concatenating their z-vectors — detects anomalies at multiple time / memory scales simultaneously.

## Algorithm details worth calling out

Read `docs/ANOMALY_PLAN.md` sections below for full derivations; here's the map:

- **Parade**: cyclic buffer of the last k predictive Gaussians. On observation `y`, compute PIT under each matured horizon predictive → map to standard-normal quantile → emit `z ∈ ℝ^k`.
- **Bulk p-value (Satterthwaite)**: track running mean `m₂` and variance `v₂` of `d²`. Match to `c · χ²_ν` via `c = v/(2m)`, `ν = 2m²/v`. Seed at exact `χ²_k` moments.
- **Tail p-value (POT/GPD)**: excesses over the `pot_level` quantile fit a Generalized Pareto Distribution via probability-weighted moments (PWM). Above the threshold, GPD gives the authoritative p-value.
- **Scatter model**: two options.
  - `factor`: `Σ ≈ Σⱼ λⱼ vⱼvⱼᵀ + D` (leading eigenpairs by power iteration, residual per-horizon variances on diagonal). Inverse via Woodbury.
  - `shrink`: `(1-δ)·S + δ·I` — Ledoit-Wolf-style toward identity.
- **Huberised update**: ticks with `d² > q_guard` (guard-quantile of the null) get weight `q_guard/d²`. Prevents outliers contaminating μ and Σ (the "masking" problem).
- **Changepoint escape**: run of `adapt_after` consecutive guarded ticks resets to full weight — otherwise a structural break looks anomalous forever.
- **Deep-evidence channel** (optional): tracks `-logpdf` of `y` under the 1-step predictive; own POT tail; Bonferroni-combined with the Mahalanobis p-value. Prevents saturation when `z` clamps at ±7σ.

## Fit with our LaplaceForecaster environment

We have:

- ✅ Streaming `.observe(y)` API — the hook the parade needs
- ✅ Multi-horizon `.forecast_dist(k)` returning `Vec<GaussianMixture>` — parade input
- ✅ `Gaussian::cdf(y)` — for PIT
- ✅ Terminal scale-mixture — well-calibrated 1-step density (nlp channel input)
- ✅ `MultiScaleLaplace` (`src/models/laplace/multiscale.rs`) — foundation for zbank

We need:

- ❌ Standard-normal `quantile(p)` — currently only PDF and CDF
- ❌ Chi-square SF `chi2_sf(x, dof)` and quantile `chi2_ppf(p, dof)`
- ❌ GPD fit (PWM) + SF
- ❌ Small dense linalg: Cholesky, forward/back-sub, power iteration for leading eigenpair
- ❌ Parade wrapper
- ❌ Mahalanobis detector state + update
- ❌ (Optional) zbank

Nothing exotic — all pure numerics on `Vec<f64>` / `[f64; k]`. Deterministic port.

## Implementation plan

### Phase 1 — Math primitives (foundational, no forecasting logic)

New module `src/anomaly/` with submodules for the primitives:

```
src/anomaly/
├── mod.rs
├── chi2.rs        // chi2_sf, chi2_ppf via lower incomplete gamma
├── gpd.rs         // gpd_fit_pwm, gpd_sf
├── linalg.rs      // cholesky, mahal2, top_eig, top_factors, solve_sym
└── quantile.rs    // standard_normal_quantile (Beasley-Springer / Wichura)
```

**Rust-idiomatic notes**:
- Flat row-major `Vec<f64>` for k×k matrices with explicit indexing helpers (like the Python source). k is small (≤32 in practice) — no need for `nalgebra`.
- Return `Result<T, AnomalyError>` for degenerate cases (Cholesky on non-PD, GPD fit with < 2 excesses, chi2 with dof ≤ 0).
- Unit tests: chi² and GPD against known reference values (scipy tables), Cholesky roundtrip.

**Effort**: 1-1.5 days. ~350 LOC. No external dependencies.

### Phase 2 — Parade wrapper

New file `src/anomaly/parade.rs`:

```rust
pub struct Parade<F: DistributionalForecaster> {
    base: F,
    k: usize,
    // Ring buffer of the last k predictions: pending[i] is the k-vector
    // Vec<Gaussian> issued i ticks ago. Newest at pending[0].
    pending: VecDeque<Vec<Gaussian>>,
    // Latest PIT + z vectors, updated each tick.
    pit: Vec<Option<f64>>,   // length k
    z: Vec<Option<f64>>,     // length k
}

impl<F> Parade<F> where F: DistributionalForecaster {
    pub fn wrap(base: F, k: usize) -> Self { ... }
    pub fn observe(&mut self, y: f64) -> Result<()> { ... }
    pub fn z(&self) -> &[Option<f64>] { &self.z }
    pub fn pit(&self) -> &[Option<f64>] { &self.pit }
    pub fn forecast_dist(&self, h: usize) -> Result<Vec<GaussianMixture>> {
        self.base.forecast_dist(h)
    }
}
```

**Key details**:
- Clamp PIT away from `{0, 1}` at `ε = 1e-12` so `|z|` bounds at ~7.03σ.
- Winsorize `y` to `mean ± 1e12·(1 + |mean| + std)` before passing to base — magnitude-relative, not sigma-relative (preserves legitimate values after degenerate-variance stretches).
- Skip PIT computation on non-finite CDF output; leave that horizon's `z_m = None`.

**Effort**: half a day. ~150 LOC. Depends on Phase 1's `standard_normal_quantile`.

**Test**: on a known-calibrated stream (fit LaplaceForecaster to Gaussian noise, wrap in Parade, check that after warmup, `z` histogram is standard normal).

### Phase 3 — Mahalanobis detector

New file `src/anomaly/mahalanobis.rs`:

```rust
pub struct MahalanobisConfig {
    pub k: usize,
    pub alpha: f64,          // EWMA rate for μ, Σ, m2, v2
    pub scatter: ScatterMode, // Factor { n_factors: usize, dfloor: f64 } | Shrink { delta: f64 }
    pub guard_p: f64,        // Huberization quantile (default 0.99)
    pub adapt_after: usize,  // Changepoint escape after N guarded ticks
    pub pot_level: f64,      // POT threshold quantile (default 0.98)
    pub min_exc: usize,      // Excesses required before GPD tail (default 30)
}

pub struct MahalanobisDetector<F> {
    cfg: MahalanobisConfig,
    parade: Parade<F>,
    mu: Vec<f64>,       // k
    sigma: Vec<f64>,    // k*k flat
    m2: f64,            // null-bulk mean of d^2
    v2: f64,            // null-bulk variance
    exc: VecDeque<f64>, // POT excesses
    zeta: f64,          // P(d2 > t_pot) EWMA
    run: usize,
    // Deep-evidence (nlp) channel state
    pend1: Option<Gaussian>,
    n_stats: NlpChannel,
    // Output state (updated every tick)
    last: AnomalyOutput,
}

pub struct AnomalyOutput {
    pub d2: Option<f64>,
    pub p_value: Option<f64>,
    pub run: usize,
}

impl<F: DistributionalForecaster> MahalanobisDetector<F> {
    pub fn wrap(base: F, cfg: MahalanobisConfig) -> Self { ... }
    pub fn observe(&mut self, y: f64) -> Result<()> { ... }
    pub fn state(&self) -> &AnomalyOutput { &self.last }
    pub fn forecast_dist(&self, h: usize) -> Result<Vec<GaussianMixture>> {
        self.parade.forecast_dist(h)
    }
}
```

**Order of operations per tick** (mirrors the Python):
1. If NaN/Inf `y`: bail, hold last forecasts, no score. **Score before update — the point must not defend itself.**
2. Compute `nlp = -logpdf_1step(y)` from previous tick's 1-step predictive (for deep-evidence channel).
3. Winsorize `y` (magnitude-relative).
4. Feed `y` to parade (which feeds `base.observe`, updates ring buffer, emits new `z`).
5. If any `z_m == None`: warmup, no score.
6. Score: `d² = (z-μ)ᵀ Σ⁻¹ (z-μ)` via factor model (Woodbury) or shrunk-Σ Cholesky.
7. Bulk p-value: Satterthwaite `c·χ²_ν`.
8. Tail p-value: if `d² > t_pot` and enough excesses, GPD SF.
9. Deep-evidence: if enough excesses, GPD on `nlp` channel; Bonferroni-combine.
10. Huberize update weight: `w = q_guard/d²` if `d² > q_guard`, else `w = 1`; changepoint escape after `adapt_after`.
11. Update `μ`, `Σ`, `m₂`, `v₂`, `ζ`, `exc` all at the Huberized rate.

**Effort**: 2 days. ~500 LOC. Depends on Phases 1 + 2.

**Tests**:
- On stationary Gaussian noise fed through a well-fit LaplaceForecaster, `p_value` should be approximately Uniform(0,1) after ~3k warmup.
- Inject a 10σ spike; assert `p_value < 1e-4`, `run = 1-2`.
- Inject a step-change (level shift); assert `run` grows to `adapt_after`, then p-values normalize.
- Fixed-seed regression test against a Python reference output on the same input stream.

### Phase 4 — zbank (optional multi-scale detector)

New file `src/anomaly/zbank.rs`:

```rust
pub struct ZBank<Fmake: Fn(f64) -> LaplaceForecaster> {
    k: usize,
    make_engine: Fmake,
    sigmas: Vec<f64>,
    strides: Vec<usize>,
    // Per-(sigma, stride) phase copies, keyed by (sigma_idx, stride, phase).
    engines: Vec<(f64, usize, Vec<LaplaceForecaster>)>,
    t: usize,
}
```

Uses the stride-phase trick from `multiscale.rs`: for stride `s`, keep `s` phase-shifted engine copies, advance the one whose phase matches the current tick. Concatenates each engine's `z` into one long z-vector.

**Effort**: half a day. ~200 LOC.

Skip until Phase 3 is validated end-to-end.

### Phase 5 — Docs + integration

- Public API surface: `use anofox_forecast::anomaly::{Parade, MahalanobisDetector, MahalanobisConfig, ScatterMode}`.
- One example: `examples/anomaly_detection.rs` that wraps `.skaters()`, feeds a synthetic stream with an injected anomaly, plots p-values.
- Feature flag: put behind `--features anomaly` since it adds ~1200 LOC and callers not doing detection don't need it.

## Total estimate

| phase | LOC | days |
|---|---:|---:|
| 1 — math primitives | ~350 | 1.5 |
| 2 — parade wrapper | ~150 | 0.5 |
| 3 — Mahalanobis detector | ~500 | 2 |
| 4 — zbank | ~200 | 0.5 |
| 5 — docs + integration | ~100 | 0.5 |
| **total** | **~1300** | **5** |

## Risks & mitigations

- **Numerical**: chi² and GPD are notoriously touchy in the tails. Mitigation: port test values from the Python reference (which itself references A&S 6.5 for gser/gcf). Add fuzz tests with random inputs vs a reference.
- **Ring-buffer semantics**: off-by-one on which prediction is "the one issued m steps ago" is easy to get wrong. Mitigation: unit test the Parade in isolation on a hand-computable stream (e.g. `y_t = t`, deterministic drift forecaster) — one horizon at a time.
- **Deep-evidence dependency on `nlp`**: requires the base to emit calibrated 1-step density. Our `LaplaceForecaster` does but any caller-supplied base must too. Enforce via trait bound.
- **State size**: at k=48 with factor scatter, state is `k + k² + 5·k + 250 excesses ≈ 3 KB`. Fine for streaming.
- **Integration with existing serde work**: since Option A is partially done and the anomaly module will also want serde (checkpoint / restart), design each new struct with `#[cfg_attr(feature = "serde", derive(...))]` from day 1. No `Box<dyn>` fields.

## What we skip

- **The `pot_level=0.98`, `guard_p=0.99` etc. hyperparameter tuning** — port the Python defaults exactly. They're calibrated on a benchmark panel; changing them without a benchmark of our own is guessing.
- **Bank layer (`zbank`)** — Phase 4 is optional; skip on the first pass. Single-engine Mahalanobis already delivers most of the value.
- **Deep-evidence `nlp` channel** — implement in Phase 3 alongside the main detector; not worth splitting into its own phase.

## Recommendation

Do phases 1-3 as a first cut (~4 days). Ship as an opt-in feature. Measure on a synthetic dataset (Gaussian noise + injected outliers) to verify the p-value distribution is uniform under the null. Then decide if zbank is worth the marginal effort based on real use-cases.

Not going to auto-enable this on any existing builder — it's a wrapper API. Users opt in via:

```rust
use anofox_forecast::anomaly::{MahalanobisDetector, MahalanobisConfig};

let base = LaplaceForecaster::new().auto();
let mut det = MahalanobisDetector::wrap(base, MahalanobisConfig::new(k=8));
det.fit(&ts)?;
for &y in &live_stream {
    det.observe(y)?;
    let out = det.state();
    if let Some(p) = out.p_value {
        if p < 0.001 { alert(y, p, out.run); }
    }
}
```
