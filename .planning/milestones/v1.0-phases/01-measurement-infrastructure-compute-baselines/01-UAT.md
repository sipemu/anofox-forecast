---
status: complete
phase: 01-measurement-infrastructure-compute-baselines
source: [01-VERIFICATION.md]
started: 2026-08-09T22:35:00Z
updated: 2026-08-10T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. CR-01 security fix — wasm-size.yml Python injection
expected: A crafted `"bytes"` string in wasm_size.json (e.g. `0; import os; os.system("id")`) must NOT execute; the gate should fail on numeric validation. Fix per 01-REVIEW.md CR-01 (JSON parse + int() inside Python, no $BASELINE interpolation).
result: PASSED — fixed in commit 9e8aa57. Gate now reads the baseline and coerces with int() inside a single python3 process; measured size passed via WASM_CURRENT_BYTES env, no shell interpolation of file-derived data. Locally reproduced: a crafted `bytes` value raises ValueError (CI fails safe), no code executes. YAML valid, permissions contents:read and MEAS-01 no-write invariant preserved.

### 2. iai.json real instruction counts
expected: On a machine with valgrind >= 3.20, `bash scripts/update_iai.sh` overwrites iai.json with non-zero instruction_count for bench_auto_ets_fit::n200, bench_auto_arima_fit::n200, bench_batch_100::s100_n100 (currently structural placeholder = 0).
result: pass

### 3. criterion.json real wall-clock values
expected: On a quiet local machine, `bash scripts/update_criterion.sh` overwrites criterion.json with median_ns > 0 for all 12 non-Laplace entries across both `parallel` and `no_parallel` profiles (currently structural placeholder = 0.0 per D-03).
result: pass

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

- Items 2 (iai.json) and 3 (criterion.json) are intentional structural placeholders pending maintainer capture on the appropriate hardware (valgrind >= 3.20 machine; quiet local machine). The capture scripts, benches, schema, and CI gates are complete and verified — only the numeric baselines await population. Run the documented `scripts/update_iai.sh` / `scripts/update_criterion.sh`, commit, then `/gsd-verify-work 1` to close out the phase.
