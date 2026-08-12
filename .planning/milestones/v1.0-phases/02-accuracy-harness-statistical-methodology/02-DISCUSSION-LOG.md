# Phase 2: Accuracy Harness & Statistical Methodology - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-10
**Phase:** 2-Accuracy Harness & Statistical Methodology
**Areas discussed:** Metric/CV code strategy, Baseline breadth, Cross-library reference (BENCH-01), Naive2 + DM test design

---

## Area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Metric/CV code strategy | Reuse library metrics/CV vs reimplement independently | ✓ |
| Baseline breadth | Model × dataset matrix committed to accuracy.json | ✓ |
| Cross-library reference (BENCH-01) | Live statsforecast run vs committed reference fixture | ✓ |
| Naive2 + DM test design | Location, seasonality gate, DM loss/form | ✓ |

**User's choice:** All four areas.

---

## Metric/CV code strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Reimplement in harness | Independent metric math + cross-check against src/ | |
| Reuse library metrics | Import src/utils/metrics.rs; ACCUR-08 validation is the safety net | ✓ |
| Hybrid: reuse + assert | Reuse but wrap with independent test-time recomputation | |

| Option | Description | Selected |
|--------|-------------|----------|
| Harness owns folds | Independent expanding-window folds in the harness | |
| Reuse library CV | Call src/utils/cross_validate + add assertions | ✓ |
| You decide | Defer to researcher | |

**User's choice:** Reuse library metrics; reuse library CV.
**Notes:** ACCUR-08 validation against published statsforecast ≈ 0.93 is the bug-catching net for reused metrics.

### MASE NaN guard (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Guard in harness only | Wrap calls; src/ untouched; file library bug for Phase 4 | |
| Fix in src/ now | Add collapse guard directly in src/utils/metrics.rs | ✓ |
| Guard in harness + file backlog | Harness guard + recorded findings note | |

| Option | Description | Selected |
|--------|-------------|----------|
| Drop + count | Exclude collapsed series, report excluded count | |
| Fallback denominator | Period-1 naive denominator when seasonal collapses | ✓ |
| You decide | Match statsforecast behavior | |

**User's choice:** Fix in src/ now; fallback denominator.
**Notes:** Framed as a genuine correctness fix (regression test + before/after), not new measurement tooling — does not violate MEAS-04's intent. Fallback matches statsforecast, keeping ACCUR-08 apples-to-apples.

---

## Baseline breadth

| Option | Description | Selected |
|--------|-------------|----------|
| Naive2 + AutoETS | Smallest trustworthy baseline (ACCUR-08 anchor + ACCUR-06 ref) | ✓ |
| Full stat family | + AutoARIMA, AutoTheta | |
| Family reps (Phase 1 set) | 7-family speed set | |

| Option | Description | Selected |
|--------|-------------|----------|
| M3 monthly first, schema-ready for rest | Lock M3-monthly now; loader/schema generalize | |
| All M3 + M4 + Tourism + NN5 | Full corpus, all frequencies now | |
| You decide | Bound by capture runtime | ✓ |

**User's choice:** Naive2 + AutoETS; dataset scope → you decide.
**Notes (Claude's decision):** Loader reads all four required corpora (ACCUR-01); committed baseline = M3 across Y/Q/M (per-frequency stratification for ACCUR-07, anchored on M3-monthly ≈ 0.93 for ACCUR-08), small enough for a maintainer machine. M4/Tourism/NN5 plug into the same schema as a follow-up.

---

## Cross-library reference (BENCH-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Committed fixture, doc'd regen | Regenerate once on pinned env, commit JSON, CI reads it | (chosen by Claude) |
| Live run in workflow_dispatch | Run statsforecast live each dispatch | |
| You decide | Based on env reproducibility | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Regenerate + pin, then commit | Pin versions, verify ≈ 0.93, commit with provenance | ✓ |
| Trust existing fixture | Reuse statsforecast_reference.json as-is | |
| Inspect first | Researcher inspects before deciding | |

**User's choice:** Capture mechanism → you decide; regenerate + pin, then commit.
**Notes (Claude's decision):** Committed fixture with documented, pinned regeneration; accuracy.yml reads the fixture (no live Python in the gating path), mirroring Phase 1's CI-reads-never-captures rule. Provenance block records statsforecast/numpy/pandas versions.

---

## Naive2 + DM test design

| Option | Description | Selected |
|--------|-------------|----------|
| In the harness | Measurement reference; keeps public API frozen (MEAS-04) | (chosen by Claude) |
| In the library | Public model in src/models/baseline/ | |
| You decide | Based on existing primitives | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| statsforecast-style ACF test | 90%-confidence ACF at seasonal lag | (chosen by Claude) |
| Reuse library seasonality detection | Use existing seasonal-strength code | |
| You decide | Keep comparison honest | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Squared loss + HLN correction | HLN small-sample + horizon-aware HAC variance | (chosen by Claude) |
| Absolute loss, plain DM | Simpler, less robust on short series | |
| You decide | Standard M-competition form | ✓ |

**User's choice:** All three → you decide.
**Notes (Claude's decisions):** Naive2 in the harness (composes from library Naive/SeasonalNaive, harness-owned gate); seasonality gate = statsforecast-style 90% ACF at seasonal lag; DM = squared-error loss + HLN correction + horizon-aware HAC variance. Researcher confirms exact variance estimator.

---

## Claude's Discretion

- Dataset scope of the first committed baseline (chose M3 Y/Q/M to bound runtime).
- Reference capture mechanism (chose committed fixture + pinned regeneration).
- Naive2 location (harness), seasonality gate (statsforecast 90% ACF), DM form (squared + HLN + HAC).

## Deferred Ideas

- Accuracy numbers for M4 / Tourism / NN5 and all frequencies — loader/schema support them; only M3 (Y/Q/M) committed in this first lock.
- Broader model set in accuracy.json (AutoARIMA/AutoTheta/Croston/Laplace/AutoEnsemble) — deferred; intermittent/distributional models need intermittent datasets.
- Fixing MASE silent-NaN as a Phase 4 backlog item — superseded; fixed in src/ now.
