# Phase 1: Measurement Infrastructure & Compute Baselines - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-09
**Phase:** 1-Measurement Infrastructure & Compute Baselines
**Areas discussed:** Baseline JSON schema, Harness location, Reuse existing benches, Regression gate strictness

---

## Baseline JSON schema

### Granularity
| Option | Description | Selected |
|--------|-------------|----------|
| One file per dimension | criterion/iai/dhat/wasm_size .json, self-contained | ✓ |
| Single baselines.json | One file with a section per dimension | |
| Per-dimension dir + entries | baselines/criterion/*.json, one per case | |

### Provenance
| Option | Description | Selected |
|--------|-------------|----------|
| Full fingerprint | git SHA, timestamp, rustc, host CPU/OS, features + values | ✓ |
| Value + git SHA only | Metric + commit | |
| Value only | Number(s) only | |

### Criterion statistic / CI behavior
| Option | Description | Selected |
|--------|-------------|----------|
| Median+MAD, informational | Robust stat; CI reports drift, never fails on criterion (iai is hard gate) | ✓ |
| Mean+stddev, informational | Familiar but noise-sensitive | |
| Median+MAD, soft CI gate | Fail on large delta — risks flaky CI | |

**User's choice:** One file per dimension · Full fingerprint · Median+MAD informational.
**Notes:** iai-callgrind is the hard compute gate; criterion stays local/informational per locked constraint.

---

## Harness location

### Tooling home
| Option | Description | Selected |
|--------|-------------|----------|
| New harness crate | Workspace member owns schema/compare/dhat+iai bins | ✓ |
| Top-level benches + scripts only | No new crate; ad-hoc JSON | |
| Hybrid: scripts wrap a tiny crate | Split ownership | |

### Runner UX
| Option | Description | Selected |
|--------|-------------|----------|
| scripts/update_*.sh per dimension | One shell script per dimension | ✓ |
| cargo xtask subcommands | Single Rust entrypoint | |
| Both: thin .sh calling xtask | .sh entrypoint delegating to xtask | (implied — .sh delegates to crate) |

**User's choice:** New harness crate + `scripts/update_*.sh` per dimension.
**Notes:** Scripts delegate to the harness crate internally; wording matches MEAS-02 success criterion.

---

## Reuse existing benches

### Approach
| Option | Description | Selected |
|--------|-------------|----------|
| New dedicated baseline bench | Single tracked source of truth; existing 8 stay ad-hoc | ✓ |
| Normalize existing in place | Retrofit all 8 as tracked | |
| Consolidate into one | Merge + delete redundant | |

### Coverage
| Option | Description | Selected |
|--------|-------------|----------|
| Representative per family | 7 families × {single, batch-100} | ✓ |
| Broad within families | Multiple variants per family | |
| Fit-heavy only | Track fit, spot-check predict/batch | |

### Fixtures
| Option | Description | Selected |
|--------|-------------|----------|
| Shared seeded fixtures | Deterministic series, fixed lengths+seed, in harness crate | ✓ |
| Vendored real series | Slice of validation/data/ | |
| Keep per-bench generators | Each bench defines its own | |

**User's choice:** New dedicated baseline bench · representative per family · shared seeded fixtures.
**Notes:** Fills gaps: intermittent (Croston), Laplace, explicit batch — absent from the existing 8.

---

## Regression gate strictness

### iai gate
| Option | Description | Selected |
|--------|-------------|----------|
| ±1% on pinned stable | Gate instructions only on stable; fail >1% | ✓ |
| Exact match on pinned stable | Zero tolerance | |
| ±5% on pinned stable | Looser band | |

### WASM size gate
| Option | Description | Selected |
|--------|-------------|----------|
| Relative >1% | Fail on >1% growth | ✓ |
| Absolute >10 KB | Fixed threshold | |
| Hybrid: >1% or >10 KB | Either bound | |

### dhat gate
| Option | Description | Selected |
|--------|-------------|----------|
| Hard assert w/ headroom | baseline ×1.15, fails CI | ✓ |
| Advisory only | Record, never fail | |
| Hard assert, tight (5%) | baseline ×1.05 | |

**User's choice:** iai ±1% on pinned stable · WASM size >1% relative · dhat hard assert ×1.15.
**Notes:** iai gate runs on pinned stable only; beta/nightly build+test but don't gate instruction counts.

---

## Claude's Discretion

- iai hot-path selection: AutoETS fit, AutoARIMA fit, batch-100 (the three named in PERF-02).
- Dead-code cleanup scope (PERF-06): named `inner()`/unused-imports + `cargo-machete` sweep on `crates/anofox-forecast-js/`; verify npm package still builds before locking size baseline.
- `bench.yml` CI scope: iai-callgrind gate only; criterion captured locally.
- dhat harness form: native bin/test with dhat allocator (NOT `wee_alloc` — banned).

## Deferred Ideas

- Broader within-family bench coverage (multiple intermittent/ETS variants) — revisit only if Phase 4 backlog points there.
- Vendored-real-series fixtures — deferred; ties to Phase-2 dataset work.
- WASM runtime memory profiling — v2 requirement (XROB-02).
