#!/usr/bin/env bash
# scripts/update_dhat.sh — capture dhat peak-memory baseline into
# .planning/baselines/dhat.json (PERF-04, D-12, MEAS-04).
#
# Runs the dhat_peak integration test in CAPTURE_DHAT=1 mode, which causes
# each family's test block to print "CAPTURE <name> <peak_bytes>" instead of
# asserting. Those lines are parsed by the python3 block below, then written
# as a provenance-stamped JSON baseline (open 'w' — always overwrites).
#
# Usage: bash scripts/update_dhat.sh
#
# Produces: .planning/baselines/dhat.json (OVERWRITES — idempotent)
#
# After running, execute the test to verify the gate passes:
#   cargo test -p anofox-bench-harness --test dhat_peak
#
# FIRST-RUN NOTE (MEAS-04):
#   On first run (no existing dhat.json), the test would normally skip all
#   assertions. This script creates dhat.json so subsequent test runs can
#   assert. Run this script once before relying on the CI gate.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Gather provenance fingerprint
# ---------------------------------------------------------------------------
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUSTC=$(rustc --version)
OS=$(uname -sr)
CPU=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")

echo "=== dhat peak-memory baseline capture (D-12) ==="
echo "  git_sha: $GIT_SHA"
echo "  rustc:   $RUSTC"
echo "  os:      $OS"
echo "  cpu:     $CPU"
echo ""

# ---------------------------------------------------------------------------
# Run the dhat_peak test in capture mode.
# CAPTURE_DHAT=1 causes each test block to print "CAPTURE <name> <bytes>"
# instead of asserting against the (possibly absent) baseline.
# We capture both stdout and stderr to parse the CAPTURE lines.
# ---------------------------------------------------------------------------
echo "Running dhat_peak in capture mode (CAPTURE_DHAT=1)..."

CAPTURE_LOG=$(mktemp /tmp/dhat_capture_XXXXXX.log)
trap 'rm -f "$CAPTURE_LOG"' EXIT

CAPTURE_DHAT=1 cargo test -p anofox-bench-harness --test dhat_peak \
    -- --nocapture 2>&1 | tee "$CAPTURE_LOG"
echo ""

# ---------------------------------------------------------------------------
# Parse CAPTURE lines and write dhat.json
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/.planning/baselines"

python3 - "$CAPTURE_LOG" "$GIT_SHA" "$TIMESTAMP" "$RUSTC" "$CPU" "$OS" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

capture_log, git_sha, timestamp_iso, rustc_version, host_cpu, host_os = sys.argv[1:7]

with open(capture_log) as f:
    output = f.read()

# Parse lines of the form "CAPTURE <name> <peak_bytes>"
capture_re = re.compile(r'^\s*CAPTURE\s+(\S+)\s+(\d+)\s*$')

results = {}
for line in output.splitlines():
    m = capture_re.match(line)
    if m:
        name = m.group(1)
        peak_bytes = int(m.group(2))
        results[name] = peak_bytes

if not results:
    print("ERROR: no CAPTURE lines found in test output.", file=sys.stderr)
    print("  Check that the dhat_peak test ran with CAPTURE_DHAT=1 and --nocapture.", file=sys.stderr)
    sys.exit(1)

benchmarks = [
    {"name": name, "peak_bytes": peak_bytes}
    for name, peak_bytes in sorted(results.items())
]

record = {
    "provenance": {
        "git_sha": git_sha,
        "timestamp_iso": timestamp_iso,
        "rustc_version": rustc_version,
        "host_cpu": host_cpu,
        "host_os": host_os,
        "active_features": [],
    },
    "benchmarks": benchmarks,
}

out_path = Path(".planning/baselines/dhat.json")
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")

print(f"Wrote {out_path}")
print(f"  Entries: {len(benchmarks)}")
for entry in benchmarks:
    print(f"  {entry['name']:30s}  peak={entry['peak_bytes']:>12,} bytes")
PYEOF

echo ""
echo "dhat.json written. Run the gate test to verify:"
echo "  cargo test -p anofox-bench-harness --test dhat_peak"
