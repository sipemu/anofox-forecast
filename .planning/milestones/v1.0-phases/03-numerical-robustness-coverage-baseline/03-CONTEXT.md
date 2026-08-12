# Phase 3: Numerical Robustness & Coverage Baseline - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Every model family handles malformed and edge-case inputs with a correct `ForecastError`
(never a panic); a code-coverage baseline is committed with a CI floor enforced; and a gap
inventory identifies the highest-priority uncovered paths as structured input for the Phase 4
improvement backlog.

This phase is measurement + hardening of robustness — NOT new forecasting features. It
respects the project's core value: every claimed capability (here, "handles bad input safely"
and "is tested") is measured and proven, not asserted.

</domain>

<decisions>
## Implementation Decisions

### Coverage Floor & CI Enforcement
- **Floor is a ratchet from the measured baseline** (baseline minus a small ~1% margin) so
  coverage can only trend up; the floor is bumped when coverage rises, never silently lowered.
- **Enforce in the EXISTING `ci.yml` coverage job** (which already installs `cargo-llvm-cov`,
  see `.github/workflows/ci.yml:146`). CI **fails** when coverage drops below the floor —
  this is a hard gate, matching ROADMAP SC-4. Do NOT create a new workflow; extend the
  existing job.
- **Metric: line coverage** (cargo-llvm-cov default) — simplest, matches the existing job.
- **Feature set for coverage: a fixed, documented set** (`--all-features` excluding
  wasm/js-only features) so the number is reproducible run-to-run. Document the exact
  invocation next to the baseline.
- Coverage baseline committed to `.planning/baselines/coverage.json` (the established
  baselines location — NOTE the ROADMAP wrote `baselines/coverage.json` loosely; the real
  convention is `.planning/baselines/`, consistent with criterion.json / iai.json / dhat.json /
  statsforecast_reference.json / wasm_size.json). Include a provenance block (tool version,
  feature set, commit) as the other baselines do.

### Edge-Case Robustness Suite
- **One representative model per family** (ETS/AutoETS, ARIMA/AutoARIMA, Theta, TBATS,
  a baseline model, an intermittent model, plus VAR/Laplace where applicable) **plus any
  model flagged fragile** during the gap scan. Not all 30+ models exhaustively — keep the
  suite maintainable.
- **Edge-case input set = the ROADMAP list, as-is:** constant series, n=2, all-zeros /
  intermittent, NaN/Inf-containing, zero-length, extreme-scale inputs.
- **Assertions: assert the exact `ForecastError` variant where the outcome is deterministic**
  (e.g. `EmptyData`, `InsufficientData`, `MissingValues`); otherwise assert `is_err()` +
  no-panic. No test may trigger a panic.
- **Layout: a single new `tests/edge_case_robustness.rs`** integration suite.

### Fix-vs-File Policy & Gap Inventory
- **Missing `validate_series_complete()` (or equivalent) calls:** ~37 of ~45 fit-bearing
  files already call it. For each fit() path missing pre-estimation validation — **fix
  trivial/safe cases inline** in this phase; **file risky/complex cases as P1
  improvement-backlog items** (do not force risky refactors into a robustness phase).
- **Proptest coverage** of the ROADMAP-named fragile areas — changepoint metrics, MSTL
  decomposition, CV boundary conditions — asserting **no-panic and no-NaN** invariants across
  random inputs. Reuse existing proptest patterns (`property_tests.rs`,
  `interval_property_tests.rs`, `laplace_robustness.rs`).
- **Gap inventory: a committed markdown document** listing uncovered paths and assertion-free
  tests with enough detail to serve as backlog input — **file + function + missing invariant**
  per row — **structured so the Phase 4 backlog consumes it directly.**

### Claude's Discretion
- Exact representative-model selection per family, the precise ratchet margin, and the
  gap-inventory file name/location within the phase directory are at Claude's discretion,
  guided by the decisions above and codebase conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `validate_series_complete()` — the established pre-fit validation entry point; already
  called in ~37 files under `src/models/`.
- Existing proptest suites to mirror: `tests/property_tests.rs`,
  `tests/interval_property_tests.rs`, `tests/laplace_robustness.rs`,
  `tests/laplace_component_robustness.rs`.
- `ForecastError` variants (`EmptyData`, `InsufficientData { needed, got, hint }`,
  `MissingValues`, `FitRequired`, ...) in `src/error.rs` — the assertion targets.
- `.planning/baselines/` — established baseline location + provenance-block convention.

### Established Patterns
- Early validation: `validate_series_complete()` at the start of every `fit()`.
- Baselines are JSON with a provenance block; CI jobs live in `.github/workflows/`.
- Project prefers hard CI gates ONLY where the ROADMAP requires them; expensive/accuracy
  checks stay `workflow_dispatch`-only. Coverage IS a required hard gate here (SC-4).

### Integration Points
- `.github/workflows/ci.yml` `coverage:` job (line ~146) — extend to compare against the
  committed floor and fail on regression.
- `.planning/baselines/coverage.json` — new baseline artifact.
- Phase 4 improvement backlog — consumes the gap inventory + any P1-filed missing-validation
  items.

</code_context>

<specifics>
## Specific Ideas

- The coverage floor must be a ratchet (up-only), not a fixed number, to catch slow erosion.
- Do not create a parallel coverage workflow — extend the existing `ci.yml` job.
- Gap inventory is a first-class Phase 4 input; make it machine-readable enough (file +
  function + missing invariant) to seed backlog items without rework.

</specifics>

<deferred>
## Deferred Ideas

- Exhaustive edge-case coverage of ALL 30+ models (deferred — representative-per-family
  chosen instead to keep the suite maintainable).
- Risky/complex missing-validation refactors — filed as P1 backlog items for Phase 4 rather
  than forced into this phase.

</deferred>
