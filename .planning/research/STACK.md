# Stack Research: Measurement & Benchmarking Tooling

**Domain:** Rust time-series forecasting library — performance & validation hardening
**Researched:** 2026-08-09
**Confidence:** MEDIUM (web sources cross-checked; versions verified against crates.io)

---

## Recommended Stack

### (a) CPU / Wall-Clock Benchmarking

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| criterion | 0.8.2 | Wall-clock microbenchmarks (local dev + WASM-size tracking) | Statistical sampling with outlier detection; produces HTML reports and committed baseline JSON; already in this repo's `benches/`; requires rustc 1.86+ |
| iai-callgrind | 0.16.1 | Instruction-count regression detection in GitHub Actions CI | Runs each benchmark once under Valgrind/Callgrind — zero wall-clock noise from CI virtualization; produces comparable numbers across machines; correct choice for automated pass/fail gates |
| critcmp | latest | CLI diff of criterion baseline JSON files | Allows `critcmp before after` diffs across git refs without a dedicated benchmark service |

**Decision: use both, not one.**
Criterion is the right tool for local iteration (fit + predict, single vs batch, native `parallel` feature vs WASM single-threaded builds). iai-callgrind is the right tool for CI regression gates because GitHub Actions runners are virtualized and criterion wall-clock numbers are unreliable across runs.

**Native vs WASM split:**
- Native benchmarks run with `cargo bench` (criterion + iai-callgrind, all feature combinations)
- WASM benchmarks cannot use Valgrind; use criterion only inside `wasm-bindgen-test` or measure via Node.js harness (`performance.now()` wrappers). Document this split explicitly in `benches/README.md`.

**iai-callgrind CI constraint:** Requires `valgrind` installed on the runner (Ubuntu runners have it available via `apt-get install valgrind`). Not available on macOS runners — gate the iai job to `runs-on: ubuntu-latest`.

**What NOT to use:**
- divan (0.1.x) — a newer wall-clock harness; fewer ecosystem integrations and no criterion compatibility; don't migrate existing criterion benches to it
- Original `iai` crate (unmaintained; iai-callgrind is the maintained fork with expanded Callgrind + DHAT integration)

---

### (b) Memory & Allocation Profiling

| Technology | Version | Purpose | Profile |
|------------|---------|---------|---------|
| dhat | 0.3.3 | In-process heap profiling + allocation-count assertions in tests | Native only |
| heaptrack | system package | External peak-memory profiler via LD_PRELOAD; no code changes | Native Linux only |
| Valgrind massif | system package | Heap profiling with call graph; complements dhat for deep diffs | Native Linux only |

**dhat** is the primary in-code tool. Wire it behind a non-default Cargo feature (`dhat-heap`):

```toml
[features]
dhat-heap = ["dep:dhat"]

[dev-dependencies]
dhat = { version = "0.3.3", optional = true }
```

Enable with `#[global_allocator] static ALLOC: dhat::Alloc = dhat::Alloc;` guarded by `#[cfg(feature = "dhat-heap")]`. Use `Profiler::builder().testing().build()` for allocation-count assertions in integration tests. Place allocation tests in separate `tests/` files (not inline `#[cfg(test)]`) to avoid parallelism issues with the global allocator.

**heaptrack** is the right tool for one-shot peak-RSS investigation. Install system-wide (`pacman -S heaptrack` on Manjaro, `apt-get install heaptrack` in CI), then run `heaptrack cargo test --test my_test` and inspect results in `heaptrack_gui`. No crate dependency needed.

**WASM memory profiling:** There is no direct equivalent of dhat for `wasm32-unknown-unknown`. Use the browser DevTools Memory panel or Node.js `process.memoryUsage()` to track WASM heap growth during JS integration testing. Document this as a known gap.

---

### (c) WASM Binary-Size Measurement & Reduction

| Technology | Version | Purpose | Notes |
|------------|---------|---------|-------|
| twiggy | 0.8.0 | Call-graph retained-size profiler for `.wasm` | Run `twiggy top -n 20 pkg/*.wasm` to find top contributors by retained size |
| wasm-opt | Binaryen (bundled with wasm-pack) | Post-compile dead-code elimination + instruction reordering | Already invoked by `wasm-pack build --release`; produces 15-20% additional shrink |
| cargo-bloat | 0.12.1 | Crate-level size accounting for native binaries | Use `cargo bloat --release --crates` to identify dependency bloat before WASM builds |
| wc -c / shell script | n/a | CI binary-size tracking | Record `wc -c pkg/*.wasm` after each build; commit a baseline text file; fail CI if delta exceeds threshold |

**wasm-snip** (0.4.0, last released 2019) is low priority: it replaces function bodies with `unreachable` for aggressive DCE, but is effectively unmaintained and wasm-opt -Oz covers most of the same ground. Do not add it as a CI dependency.

**wee_alloc** was archived August 2025 and has known memory leaks. Do NOT use. The default Rust allocator (dlmalloc) is the safe choice for WASM; `talc` is a viable alternative if future profiling shows the allocator is a meaningful size contributor.

**CI pattern for binary-size tracking:**

```yaml
- name: Build WASM
  run: make build-wasm
- name: Record WASM size
  run: |
    SIZE=$(wc -c < crates/anofox-forecast-js/pkg/anofox_forecast_js_bg.wasm)
    echo "wasm_size=$SIZE" >> $GITHUB_OUTPUT
    echo "WASM binary: $SIZE bytes"
- name: Check size regression
  run: |
    BASELINE=$(cat .planning/baselines/wasm-size.txt)
    CURRENT=${{ steps.record.outputs.wasm_size }}
    if [ "$CURRENT" -gt "$((BASELINE + 10240))" ]; then
      echo "WASM grew by more than 10KB vs baseline ($BASELINE → $CURRENT)"; exit 1
    fi
```

Commit `.planning/baselines/wasm-size.txt` with the initial value and update intentionally after approved improvements.

---

### (d) Code-Coverage Measurement

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| cargo-llvm-cov | 0.8.7 | LLVM source-based line/region/branch coverage | Most accurate Rust coverage tool in 2026; multi-platform; supports HTML, LCOV, JSON, Cobertura output; stable toolchain compatible |

**Setup:**

```toml
# No crate dependency needed — cargo-llvm-cov is a Cargo subcommand
```

```yaml
# GitHub Actions
- uses: taiki-e/install-action@cargo-llvm-cov
- name: Coverage
  run: |
    cargo llvm-cov --all-features --workspace \
      --lcov --output-path lcov.info
    cargo llvm-cov report --html  # local HTML artifact
```

Requires the `llvm-tools-preview` component (`rustup component add llvm-tools-preview`). The `taiki-e/install-action` handles this automatically on GitHub Actions.

**What NOT to use:**
- `cargo-tarpaulin`: Linux-only (no macOS/Windows), ptrace-based instrumentation is less accurate, poor branch coverage; was the standard before llvm-cov existed but should not be chosen for new work. The existing `Makefile` references tarpaulin — leave the reference but make llvm-cov the authoritative CI tool.
- `grcov`: lower accuracy, manual LLVM instrumentation wiring required; llvm-cov subsumes it.

**Coverage gate pattern:** Store the baseline coverage percentage in `.planning/baselines/coverage.txt`. Use `cargo llvm-cov --fail-under-lines <N>` to enforce a minimum after the initial baseline is established.

---

### (e) Forecast-Accuracy Harness

| Component | Technology | Version | Why |
|-----------|-----------|---------|-----|
| Dataset format | CSV (M3/M4) + TSF (Monash) | — | M4 data is publicly available as CSV on GitHub; Monash archive uses `.tsf` (structured text); both parseable without external crates |
| CSV parsing | csv crate | 1.3 | Zero-dep CSV reader; parses M3/M4 files with headers |
| Dataset access | Vendored fixtures in `tests/data/accuracy/` | — | Small subsets only (M3 subset: ~168 series; M4 monthly sample: ~1,000 series); full datasets fetched in a separate offline step |
| Metrics | Hand-rolled in harness | — | MASE, sMAPE, RMSE are straightforward formulas; no external crate needed; implement once in `tests/accuracy/metrics.rs` |
| Baseline storage | Committed JSON in `.planning/baselines/accuracy/` | — | One JSON file per dataset/model pair; CI compares new run against committed values; diffs are reviewable in PRs |
| Feature gate | Non-default `accuracy-harness` Cargo feature | — | Prevents slow dataset tests from running in standard `cargo test`; enable with `cargo test --features accuracy-harness` |

**Metric definitions (implement exactly these):**

```rust
/// Mean Absolute Scaled Error — scale-free; < 1 beats naive forecast
fn mase(actual: &[f64], forecast: &[f64], insample: &[f64], m: usize) -> f64 {
    let mae = actual.iter().zip(forecast).map(|(a, f)| (a - f).abs()).sum::<f64>() / actual.len() as f64;
    let naive_mae = insample[m..].iter().zip(&insample[..insample.len()-m])
        .map(|(a, b)| (a - b).abs()).sum::<f64>() / (insample.len() - m) as f64;
    mae / naive_mae
}

/// symmetric MAPE — used as official M4 metric; range [0, 200]
fn smape(actual: &[f64], forecast: &[f64]) -> f64 {
    actual.iter().zip(forecast)
        .map(|(a, f)| 200.0 * (a - f).abs() / (a.abs() + f.abs()))
        .sum::<f64>() / actual.len() as f64
}

/// Root Mean Squared Error
fn rmse(actual: &[f64], forecast: &[f64]) -> f64 {
    (actual.iter().zip(forecast).map(|(a, f)| (a - f).powi(2)).sum::<f64>()
        / actual.len() as f64).sqrt()
}
```

**Dataset strategy:**

- M4 monthly (100K series): use a stratified 1,000-series sample vendored in `tests/data/accuracy/m4_monthly_sample.csv`. Full dataset runs are a separate optional CI job (`needs: [build]`, triggered by label or schedule).
- M3 (3003 series): small enough to vendor the full set.
- Tourism (1311 series): vendor the full set.
- Source: `https://github.com/Mcompetitions/M4-methods/tree/master/Dataset` for M4; `https://forecastingdata.org/` (Monash) for TSF files — download once and commit the sample subset.

**No reqwest or network downloads in tests.** All dataset files are committed as test fixtures or downloaded by a one-time `make fetch-accuracy-data` Makefile target that stores them in `tests/data/accuracy/` (gitignored for large files, committed for small samples).

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| iai-callgrind 0.16.1 | criterion alone for CI regression | criterion wall-clock is too noisy in GitHub Actions VMs; cannot reliably detect <5% regressions |
| cargo-llvm-cov 0.8.7 | cargo-tarpaulin | tarpaulin is Linux-only, ptrace-based, less accurate; llvm-cov is the clear successor |
| dhat 0.3.3 | valgrind massif only | dhat integrates into Rust's `#[test]` harness for assertion-mode allocation counting; massif requires external execution |
| vendored CSV fixtures | download datasets at test time | network access in tests is fragile for CI; small subsets can be committed; large full datasets are an opt-in offline job |
| hand-rolled MASE/sMAPE/RMSE | augurs / statsforecast crate | no mature Rust crate provides the M-competition metric suite as of 2026; implement directly |
| wasm-opt (via wasm-pack) | wasm-snip | wasm-snip is unmaintained (2019); wasm-opt -Oz achieves similar or better DCE |
| default Rust allocator (WASM) | wee_alloc | wee_alloc archived Aug 2025, has memory leaks; do NOT use |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| wee_alloc | Officially archived Aug 2025; known memory leaks; security risk | Default Rust allocator (dlmalloc for WASM); or `talc` if size is critical |
| cargo-tarpaulin | Linux-only, ptrace-based, less accurate than llvm-cov; superseded | cargo-llvm-cov 0.8.7 |
| wasm-snip 0.4.0 | Unmaintained since 2019; wasm-opt covers the same ground | wasm-opt -Oz (bundled in wasm-pack) |
| divan | Immature ecosystem; no criterion compatibility; don't displace existing benches | criterion (dev) + iai-callgrind (CI) |
| original `iai` crate | Unmaintained; iai-callgrind is the active fork with far more features | iai-callgrind 0.16.1 |
| `grcov` | Manual LLVM instrumentation wiring; cargo-llvm-cov is the ergonomic wrapper | cargo-llvm-cov 0.8.7 |
| reqwest in tests | Network in test harness is fragile; breaks offline/air-gapped CI | Vendored fixtures; Makefile download target |

---

## Native vs WASM Profile Differences

| Dimension | Native (parallel + default features) | WASM (wasm32-unknown-unknown, no parallel) |
|-----------|--------------------------------------|---------------------------------------------|
| Benchmarking | criterion (wall-clock) + iai-callgrind (instruction count) | criterion via wasm-bindgen-test or Node.js `performance.now()` harness |
| Memory profiling | dhat (allocation assertions) + heaptrack (peak RSS) | Browser DevTools Memory panel or Node.js `process.memoryUsage()` |
| Binary-size tracking | cargo-bloat --crates (native ELF) | twiggy + `wc -c` on `.wasm` output |
| Coverage | cargo-llvm-cov --all-features | Not applicable (wasm-bindgen-test does not support llvm coverage) |
| Accuracy harness | `cargo test --features accuracy-harness` | Not applicable (accuracy harness is native-only) |

**Feature flag matrix for CI:**

```yaml
# Standard CI (every PR)
cargo test --all-features          # all features, native
cargo test --no-default-features   # no features
cargo test --features parallel     # parallel only
make test-wasm                     # WASM (no parallel)

# Extended CI (scheduled / on label)
cargo test --features accuracy-harness   # accuracy harness
cargo bench                              # criterion baselines
cargo bench --bench iai_benches          # iai-callgrind (ubuntu only)
cargo llvm-cov --all-features            # coverage report
```

---

## Installation

```bash
# Benchmarking (add to Cargo.toml dev-dependencies)
# criterion is already present; add iai-callgrind:
cargo add --dev iai-callgrind@0.16.1

# CI tool: iai-callgrind runner (must match crate version exactly)
cargo install iai-callgrind-runner --version 0.16.1

# Memory profiling
cargo add --dev dhat@0.3.3 --optional   # guarded by dhat-heap feature

# WASM size analysis
cargo install twiggy@0.8.0
cargo install cargo-bloat@0.12.1

# Coverage
cargo install cargo-llvm-cov@0.8.7
rustup component add llvm-tools-preview

# Benchmark comparison
cargo install critcmp

# Accuracy harness CSV parsing (add to Cargo.toml dev-dependencies)
cargo add --dev csv@1.3

# System packages (CI / local Linux)
# sudo apt-get install valgrind heaptrack   (Ubuntu / GitHub Actions)
# sudo pacman -S valgrind heaptrack         (Manjaro / local dev)
```

---

## Version Compatibility

| Package | Constraint | Notes |
|---------|-----------|-------|
| iai-callgrind 0.16.1 | iai-callgrind-runner must match exactly | Install runner with `--version 0.16.1`; mismatch causes runtime error |
| cargo-llvm-cov 0.8.7 | rustc 1.87+ for installation | Installation requires newer rustc, but coverage can target stable builds |
| criterion 0.8.2 | rustc 1.86+ | Already pinned in Cargo.lock; upgrade from 0.5 at next opportune moment |
| dhat 0.3.3 | stable toolchain | No nightly features required |
| twiggy 0.8.0 | wasm32-unknown-unknown output from wasm-pack | Analyze the `*_bg.wasm` file (not the JS wrapper) |
| wasm-opt | bundled by wasm-pack | wasm-pack release builds run wasm-opt automatically; version is managed by wasm-pack |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|-----------|-------|
| criterion vs iai-callgrind decision | MEDIUM | Web sources cross-checked; iai-callgrind 0.16.1 verified on crates.io July 2025 |
| cargo-llvm-cov recommendation | MEDIUM | GitHub README + crates.io verified; requires rustc 1.87+ noted |
| WASM size tooling (twiggy, wasm-opt) | MEDIUM | Official Rust WASM book + crates.io; wasm-snip unmaintained confirmed |
| dhat usage pattern | MEDIUM | Official docs.rs documentation; dhat 0.3.3 is the current stable |
| wee_alloc deprecation | HIGH | Archived on GitHub Aug 25 2025; multiple sources confirm |
| criterion 0.8.2 version | HIGH | Directly verified via crates.io API |
| accuracy harness design | MEDIUM | Standard M-competition practice; no authoritative Rust-specific source |
| heaptrack/massif for native profiling | MEDIUM | Rust perf book + multiple blog posts; external tool, no version pin needed |

---

## Sources

- [Comparison to Criterion.rs — Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/iai/comparison.html) — criterion vs iai CI tradeoffs
- [iai-callgrind on crates.io](https://crates.io/crates/iai-callgrind) — version 0.16.1 verified
- [iai-callgrind GitHub (clockworklabs fork)](https://github.com/clockworklabs/iai-callgrind) — features, Valgrind dependency
- [Benchmarking and analyzing Rust performance — LambdaClass](https://blog.lambdaclass.com/benchmarking-and-analyzing-rust-performance-with-criterion-and-iai/) — practical tradeoffs
- [cargo-llvm-cov GitHub](https://github.com/taiki-e/cargo-llvm-cov) — setup, GitHub Actions workflow, rustc 1.87+ requirement
- [cargo-llvm-cov on crates.io](https://crates.io/crates/cargo-llvm-cov) — version 0.8.7 verified
- [dhat — docs.rs](https://docs.rs/dhat/latest/dhat/) — API, allocation testing pattern
- [Shrinking .wasm Size — Rust and WebAssembly Book](https://rustwasm.github.io/book/reference/code-size.html) — twiggy, wasm-opt, cargo-bloat
- [twiggy on crates.io](https://crates.io/crates/twiggy) — version 0.8.0 verified
- [wee_alloc — vulert advisory](https://vulert.com/vuln-db/crates.io-wee_alloc-30937) — unmaintained/archived status
- [lol_alloc — lib.rs](https://lib.rs/crates/lol_alloc) — wee_alloc alternative
- [Heap Allocations — The Rust Performance Book](https://nnethercote.github.io/perf-book/heap-allocations.html) — dhat/heaptrack guidance
- [critcmp — GitHub](https://github.com/BurntSushi/critcmp) — criterion baseline comparison
- [Monash Forecasting Repository](https://forecastingdata.org/) — dataset access and TSF format
- [M4-methods Dataset — GitHub](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset) — M4 CSV files

---

*Stack research for: anofox-forecast performance & validation hardening — measurement tooling*
*Researched: 2026-08-09*
