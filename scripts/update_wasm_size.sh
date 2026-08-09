#!/usr/bin/env bash
# scripts/update_wasm_size.sh — capture WASM size baseline into .planning/baselines/wasm_size.json
#
# Run on a quiet local machine after PERF-06 dead-code removal is complete.
# Captures the WASM size baseline AFTER PERF-06 cleanup.
#
# Usage: bash scripts/update_wasm_size.sh
#
# Produces: .planning/baselines/wasm_size.json (OVERWRITES — idempotent modulo provenance fields)
set -euo pipefail

# ---------------------------------------------------------------------------
# Guard: zero dead-code warnings required before capturing WASM baseline
# (enforces PERF-06-before-PERF-05 sequencing constraint)
# ---------------------------------------------------------------------------
WARNINGS=$(cargo build -p anofox-forecast-js --target wasm32-unknown-unknown 2>&1 | grep -v '^#' | grep -c 'warning:' || true)
if [ "$WARNINGS" -ne 0 ]; then
    echo "ERROR: $WARNINGS compiler warning(s) in anofox-forecast-js on the wasm32 target." >&2
    echo "       Run PERF-06 dead-code cleanup first, then re-run this script." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build release WASM and restore custom files
# ---------------------------------------------------------------------------
wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js --release
git checkout js/package.json js/README.md

# ---------------------------------------------------------------------------
# Measure binary size
# ---------------------------------------------------------------------------
BYTES=$(stat --format=%s js/anofox_forecast_js_bg.wasm)

# ---------------------------------------------------------------------------
# Gather provenance fingerprint
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
python3 - <<PYEOF
import json

record = {
    "provenance": {
        "git_sha": "${GIT_SHA}",
        "timestamp_iso": "${TIMESTAMP}",
        "rustc_version": "${RUSTC}",
        "host_cpu": "${CPU}",
        "host_os": "${OS}",
        "active_features": []
    },
    "filename": "anofox_forecast_js_bg.wasm",
    "bytes": ${BYTES}
}

with open(".planning/baselines/wasm_size.json", "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")

print(f"Wrote .planning/baselines/wasm_size.json: ${BYTES} bytes")
PYEOF
