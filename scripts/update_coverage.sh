#!/usr/bin/env bash
# scripts/update_coverage.sh — capture line-coverage baseline into
# .planning/baselines/coverage.json (COVER-01).
#
# Run on a LOCAL MACHINE after Phase 3 tests are merged.
# Do NOT run in CI — this script writes the committed baseline; CI reads it.
#
# Usage: bash scripts/update_coverage.sh
#
# Coverage scope: --package anofox-forecast --all-features
#   Scoped to the main crate so the bench-harness (crates/anofox-bench-harness)
#   and JS bindings (crates/anofox-forecast-js) do not distort the line %.
#   Feature set: --all-features activates all optional Rust features
#   (parallel, postprocess, forecastability, seasonal-detection, serde,
#   distributional, anomaly, js). The 'js' feature adds no native code;
#   it gates getrandom/js behind cfg(target_arch = "wasm32") so it is safe
#   on native x86-64 (verified: ci.yml test: job already uses --all-features).
#
# Produces: .planning/baselines/coverage.json (OVERWRITES — idempotent)
#
# ratchet_floor_percent = lines_percent - 1.0 (up-only ratchet margin).
# CI reads this floor and passes it to --fail-under-lines to block merges
# below it. Bump the floor by re-running this script and committing the result.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Gather provenance fingerprint (matches criterion.json field names exactly)
# ---------------------------------------------------------------------------
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUSTC=$(rustc --version)
COV_VER=$(cargo llvm-cov --version 2>&1 | head -1)
OS=$(uname -sr)
CPU=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")

echo "=== Coverage baseline capture (COVER-01, local-only) ==="
echo "  git_sha:              $GIT_SHA"
echo "  rustc:                $RUSTC"
echo "  cargo-llvm-cov:       $COV_VER"
echo "  os:                   $OS"
echo "  cpu:                  $CPU"
echo "  scope:                --package anofox-forecast --all-features"
echo ""

# ---------------------------------------------------------------------------
# Run cargo-llvm-cov — scoped to main crate, all features, JSON summary only
# ---------------------------------------------------------------------------
COVERAGE_TMP=/tmp/coverage_summary_$$.json

cargo llvm-cov \
  --package anofox-forecast \
  --all-features \
  --json \
  --summary-only \
  --output-path "$COVERAGE_TMP"

echo ""
echo "Coverage measurement complete."

# ---------------------------------------------------------------------------
# Extract line coverage metrics from JSON output
# jq primary path; python3 fallback if jq is absent (Assumption A2)
# ---------------------------------------------------------------------------
if command -v jq >/dev/null 2>&1; then
  LINES_PCT=$(jq '.data[0].totals.lines.percent' "$COVERAGE_TMP")
  LINES_COVERED=$(jq '.data[0].totals.lines.covered' "$COVERAGE_TMP")
  LINES_TOTAL=$(jq '.data[0].totals.lines.count' "$COVERAGE_TMP")
else
  echo "jq not found — falling back to python3 for JSON extraction (Assumption A2 fallback)"
  LINES_PCT=$(python3 -c "import json,sys; d=json.load(open('$COVERAGE_TMP')); print(d['data'][0]['totals']['lines']['percent'])")
  LINES_COVERED=$(python3 -c "import json,sys; d=json.load(open('$COVERAGE_TMP')); print(d['data'][0]['totals']['lines']['covered'])")
  LINES_TOTAL=$(python3 -c "import json,sys; d=json.load(open('$COVERAGE_TMP')); print(d['data'][0]['totals']['lines']['count'])")
fi

# ---------------------------------------------------------------------------
# Compute ratchet floor = lines_percent - 1.0 (up-only margin, one decimal)
# python3 primary (bc fallback per Assumption A5 is python3 anyway)
# ---------------------------------------------------------------------------
FLOOR=$(python3 -c "print(round(float($LINES_PCT) - 1.0, 1))")

echo "  lines_total:          $LINES_TOTAL"
echo "  lines_covered:        $LINES_COVERED"
echo "  lines_percent:        $LINES_PCT%"
echo "  ratchet_floor:        $FLOOR%"

# ---------------------------------------------------------------------------
# Write .planning/baselines/coverage.json via python3 (safe JSON serialisation)
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/.planning/baselines"

GIT_SHA="$GIT_SHA" TIMESTAMP="$TIMESTAMP" RUSTC="$RUSTC" \
COV_VER="$COV_VER" OS="$OS" CPU="$CPU" \
LINES_TOTAL="$LINES_TOTAL" LINES_COVERED="$LINES_COVERED" \
LINES_PCT="$LINES_PCT" FLOOR="$FLOOR" \
python3 - <<'EOF' > "$REPO_ROOT/.planning/baselines/coverage.json"
import json, os

data = {
  "provenance": {
    "git_sha": os.environ["GIT_SHA"],
    "timestamp_iso": os.environ["TIMESTAMP"],
    "rustc_version": os.environ["RUSTC"],
    "cargo_llvm_cov_version": os.environ["COV_VER"],
    "host_cpu": os.environ["CPU"],
    "host_os": os.environ["OS"],
    "active_features": "all (parallel + postprocess + forecastability + seasonal-detection + serde + distributional + anomaly + js)"
  },
  "coverage": {
    "metric": "line",
    "invocation": "cargo llvm-cov --package anofox-forecast --all-features --json --summary-only",
    "lines_total": int(os.environ["LINES_TOTAL"]),
    "lines_covered": int(os.environ["LINES_COVERED"]),
    "lines_percent": float(os.environ["LINES_PCT"]),
    "ratchet_floor_percent": float(os.environ["FLOOR"])
  }
}
print(json.dumps(data, indent=2))
EOF

rm -f "$COVERAGE_TMP"

echo ""
echo "Coverage baseline written to .planning/baselines/coverage.json"
echo "  measured: ${LINES_PCT}%   floor (ratchet): ${FLOOR}%"
echo ""
echo "Next steps:"
echo "  1. Review .planning/baselines/coverage.json"
echo "  2. If the floor looks reasonable, commit it and wire ci.yml (Plan 03-03 Task 3)"
echo "  3. The CI gate reads ratchet_floor_percent and fails below it"
