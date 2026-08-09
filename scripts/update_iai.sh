#!/usr/bin/env bash
# scripts/update_iai.sh — capture iai-callgrind instruction-count baseline into
# .planning/baselines/iai.json (PERF-02, MEAS-01, MEAS-02).
#
# Run on any machine that has valgrind installed. Instruction counts are
# deterministic under Callgrind on the same architecture regardless of host
# load or CPU frequency — this is the CI-reproducible gate metric (D-03/D-10).
#
# Prerequisites:
#   valgrind >= 3.20.0  (ubuntu-latest = Ubuntu 24.04 ships 3.22)
#   iai-callgrind-runner 0.16.1 --locked  (must match the crate version exactly)
#     cargo install iai-callgrind-runner --version 0.16.1 --locked
#
# Usage: bash scripts/update_iai.sh
#
# Produces: .planning/baselines/iai.json (OVERWRITES — idempotent modulo provenance fields)
set -euo pipefail

# ---------------------------------------------------------------------------
# Guard: valgrind must be installed
# ---------------------------------------------------------------------------
if ! command -v valgrind >/dev/null 2>&1; then
    echo "ERROR: valgrind is not installed or not on PATH." >&2
    echo "       Install with: sudo apt-get install -y valgrind" >&2
    echo "       Requires valgrind >= 3.20.0 (Ubuntu 24.04 ships 3.22)." >&2
    exit 1
fi

VALGRIND_VERSION=$(valgrind --version 2>/dev/null | head -1 || echo "unknown")
echo "valgrind: $VALGRIND_VERSION"

# ---------------------------------------------------------------------------
# Guard: iai-callgrind-runner must be installed at exactly 0.16.1
# ---------------------------------------------------------------------------
if ! command -v iai-callgrind-runner >/dev/null 2>&1; then
    echo "ERROR: iai-callgrind-runner is not installed or not on PATH." >&2
    echo "       Install with: cargo install iai-callgrind-runner --version 0.16.1 --locked" >&2
    echo "       (must match the iai-callgrind crate version exactly — Pitfall 1)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the bench in release mode first to verify it compiles
# ---------------------------------------------------------------------------
echo "Building bench in release mode..."
cargo build -p anofox-bench-harness --benches --release 2>&1

# ---------------------------------------------------------------------------
# Run the iai bench and capture stdout to a temp file
# iai-callgrind output per benchmark:
#
#   bench_auto_ets_fit::n200
#     Instructions:     12345678|N/A
#
# We parse this file with python3 to extract per-bench instruction counts.
# ---------------------------------------------------------------------------
BENCH_LOG=$(mktemp /tmp/iai_bench_XXXXXX.log)
trap 'rm -f "$BENCH_LOG"' EXIT

echo "Running iai_suite bench (this may take several minutes under callgrind)..."
cargo bench -p anofox-bench-harness --bench iai_suite 2>&1 | tee "$BENCH_LOG"
echo ""

# ---------------------------------------------------------------------------
# Gather provenance fingerprint (same fields as update_wasm_size.sh)
# ---------------------------------------------------------------------------
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUSTC=$(rustc --version)
OS=$(uname -sr)
CPU=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")

# ---------------------------------------------------------------------------
# Write baseline JSON (open 'w' — always overwrites, never appends)
# On first run the file is created; on subsequent runs it is replaced.
# ---------------------------------------------------------------------------
mkdir -p .planning/baselines

python3 - "$BENCH_LOG" "$GIT_SHA" "$TIMESTAMP" "$RUSTC" "$CPU" "$OS" <<'PYEOF'
import re, json, sys

bench_log_path, git_sha, timestamp_iso, rustc_version, host_cpu, host_os = sys.argv[1:7]

with open(bench_log_path) as f:
    output = f.read()

# iai-callgrind prints:
#   bench_auto_ets_fit::n200
#     Instructions:     12345678|N/A
# The bench name line has no leading whitespace and contains "::".
# The Instructions line starts with whitespace then "Instructions:".
bench_name_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_:]+$')
instr_re = re.compile(r'^\s+Instructions:\s+(\d+)\|')

results = {}
current_bench = None
for line in output.splitlines():
    stripped = line.strip()
    if bench_name_re.match(stripped) and '::' in stripped:
        current_bench = stripped
    elif current_bench is not None and instr_re.match(line):
        count = int(instr_re.match(line).group(1))
        results[current_bench] = count
        current_bench = None  # consumed

expected = [
    'bench_auto_ets_fit::n200',
    'bench_auto_arima_fit::n200',
    'bench_batch_100::s100_n100',
]
missing = [e for e in expected if e not in results]
if missing:
    print(f"ERROR: could not parse instruction counts for: {missing}", file=sys.stderr)
    print("Check that valgrind ran successfully and the output shows 'Instructions:' lines.", file=sys.stderr)
    sys.exit(1)

record = {
    "provenance": {
        "git_sha": git_sha,
        "timestamp_iso": timestamp_iso,
        "rustc_version": rustc_version,
        "host_cpu": host_cpu,
        "host_os": host_os,
        "active_features": [],
    },
    "benchmarks": [
        {"name": name, "instruction_count": results[name]}
        for name in expected
    ],
}

out_path = ".planning/baselines/iai.json"
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")

print(f"Wrote {out_path} with {len(record['benchmarks'])} entries.")
for entry in record["benchmarks"]:
    print(f"  {entry['name']}: {entry['instruction_count']:,} instructions")
PYEOF
