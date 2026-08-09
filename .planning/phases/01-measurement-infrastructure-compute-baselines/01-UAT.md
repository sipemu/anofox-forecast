---
status: testing
phase: 01-measurement-infrastructure-compute-baselines
source: [01-VERIFICATION.md]
started: 2026-08-09T22:35:00Z
updated: 2026-08-09T22:35:00Z
---

## Current Test

number: 1
name: CR-01 — Python code injection in wasm-size.yml gate
expected: |
  CI fails with a numeric validation error rather than executing injected code.
  Apply the 01-REVIEW.md CR-01 fix: read wasm_size.json and do the comparison
  fully inside one python3 invocation using int(d['bytes']) validation, passing
  $CURRENT as an argument — eliminate the unquoted $BASELINE shell interpolation.
awaiting: user response

## Tests

### 1. CR-01 security fix — wasm-size.yml Python injection
expected: A crafted `"bytes"` string in wasm_size.json (e.g. `0; import os; os.system("id")`) must NOT execute; the gate should fail on numeric validation. Fix per 01-REVIEW.md CR-01 (JSON parse + int() inside Python, no $BASELINE interpolation).
result: [pending]

### 2. iai.json real instruction counts
expected: On a machine with valgrind >= 3.20, `bash scripts/update_iai.sh` overwrites iai.json with non-zero instruction_count for bench_auto_ets_fit::n200, bench_auto_arima_fit::n200, bench_batch_100::s100_n100 (currently structural placeholder = 0).
result: [pending]

### 3. criterion.json real wall-clock values
expected: On a quiet local machine, `bash scripts/update_criterion.sh` overwrites criterion.json with median_ns > 0 for all 12 non-Laplace entries across both `parallel` and `no_parallel` profiles (currently structural placeholder = 0.0 per D-03).
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
