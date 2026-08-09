#!/usr/bin/env bash
# scripts/update_criterion.sh — capture criterion wall-clock baseline into
# .planning/baselines/criterion.json (PERF-01, PERF-03, D-03, D-06, D-09, MEAS-02).
#
# Run on a QUIET LOCAL MACHINE ONLY — never CI (wall-clock noise, D-03).
# Criterion baselines are informational only; the hard gate uses iai instruction
# counts (bench.yml). Criterion numbers track drift but never fail a PR.
#
# Usage: bash scripts/update_criterion.sh
#
# Produces: .planning/baselines/criterion.json (OVERWRITES — idempotent)
#
# Dual-profile strategy (D-09):
#   The suite runs twice:
#     1. `--features parallel`   → named baseline "parallel_run"
#                                  (tagged "parallel" in criterion.json)
#     2. default (no --features) → named baseline "no_parallel_run"
#                                  (tagged "no_parallel" in criterion.json)
#   Criterion stores each named run under:
#     target/criterion/<bench_id>/<baseline_name>/estimates.json
#   After both runs, python3 parses both directories and merges into one JSON.
#   Bench IDs in baseline_suite.rs are profile-agnostic (no suffix in names).
#
# Metrics captured (D-03):
#   median_ns : median.point_estimate (robust central value, nanoseconds)
#   mad_ns    : median_abs_dev.point_estimate (robust spread, not std)
#
# Field names verified against criterion 0.5 estimates.json structure:
#   { "median": { "point_estimate": <f64>, ... },
#     "median_abs_dev": { "point_estimate": <f64>, ... }, ... }
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Gather provenance fingerprint (same 6 fields as D-02 schema)
# ---------------------------------------------------------------------------
GIT_SHA=$(git rev-parse HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUSTC=$(rustc --version)
OS=$(uname -sr)
CPU=$(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")

echo "=== Criterion baseline capture (D-03, local-only) ==="
echo "  git_sha: $GIT_SHA"
echo "  rustc:   $RUSTC"
echo "  os:      $OS"
echo "  cpu:     $CPU"
echo ""

# ---------------------------------------------------------------------------
# Profile 1: parallel (rayon enabled) — save to named baseline "parallel_run"
# ---------------------------------------------------------------------------
echo "--- Profile 1/2: parallel (--features parallel) ---"
cargo bench --bench baseline_suite --features parallel -- --save-baseline parallel_run
echo ""

# ---------------------------------------------------------------------------
# Profile 2: no_parallel (default, no rayon) — save to named baseline "no_parallel_run"
# ---------------------------------------------------------------------------
echo "--- Profile 2/2: no_parallel (default, no --features) ---"
cargo bench --bench baseline_suite -- --save-baseline no_parallel_run
echo ""

# ---------------------------------------------------------------------------
# Parse both named baseline directories and write criterion.json
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/.planning/baselines"

python3 - "$REPO_ROOT" "$GIT_SHA" "$TIMESTAMP" "$RUSTC" "$CPU" "$OS" <<'PYEOF'
import json
import sys
from pathlib import Path

repo_root, git_sha, timestamp_iso, rustc_version, host_cpu, host_os = sys.argv[1:7]
criterion_dir = Path(repo_root) / "target" / "criterion"

# All bench IDs registered in baseline_suite.rs.
# Laplace benches are only present when built with --features distributional;
# they are included if the estimates.json exists, skipped gracefully if not.
BENCH_NAMES = [
    "auto_arima_fit_predict_n200",
    "auto_arima_batch100_fit_predict_n200",
    "auto_ets_fit_predict_n200_p12",
    "auto_ets_batch100_fit_predict_n200_p12",
    "auto_theta_fit_predict_n200",
    "auto_theta_batch100_fit_predict_n200",
    "naive_fit_predict_n200",
    "naive_batch100_fit_predict_n200",
    "croston_fit_predict_n200",
    "croston_batch100_fit_predict_n200",
    "auto_ensemble_fit_predict_n200",
    "auto_ensemble_batch100_fit_predict_n200",
    "laplace_fit_predict_n200",
    "laplace_batch100_fit_predict_n200",
]

PROFILES = [
    ("parallel_run", "parallel"),
    ("no_parallel_run", "no_parallel"),
]


def parse_baseline(baseline_name, profile_tag):
    """Read estimates.json for each bench under the named criterion baseline."""
    entries = []
    missing = []
    for bench_name in BENCH_NAMES:
        est_path = criterion_dir / bench_name / baseline_name / "estimates.json"
        if not est_path.exists():
            # Laplace is feature-gated; skip silently if absent.
            if "laplace" not in bench_name:
                missing.append(bench_name)
            continue

        with open(est_path) as f:
            est = json.load(f)

        # D-03: use median (not mean) for robust noise resistance.
        median_ns = est["median"]["point_estimate"]
        mad_ns = est["median_abs_dev"]["point_estimate"]

        entries.append({
            "name": bench_name,
            "profile": profile_tag,
            "median_ns": median_ns,
            "mad_ns": mad_ns,
        })

    if missing:
        print(
            f"  WARNING: {len(missing)} expected bench(es) not found in '{baseline_name}':",
            file=sys.stderr,
        )
        for m in missing:
            print(f"    {m}", file=sys.stderr)

    return entries


all_entries = []
for baseline_name, profile_tag in PROFILES:
    entries = parse_baseline(baseline_name, profile_tag)
    print(f"  Profile '{profile_tag}': {len(entries)} entries parsed.")
    all_entries.extend(entries)

record = {
    "provenance": {
        "git_sha": git_sha,
        "timestamp_iso": timestamp_iso,
        "rustc_version": rustc_version,
        "host_cpu": host_cpu,
        "host_os": host_os,
        # Both profiles combined; individual profiles noted under each benchmark entry.
        "active_features": [],
    },
    "benchmarks": all_entries,
}

out_path = Path(repo_root) / ".planning" / "baselines" / "criterion.json"
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
    f.write("\n")

profiles_found = sorted(set(e["profile"] for e in all_entries))
print(f"Wrote {out_path}")
print(f"  Total entries: {len(all_entries)}")
print(f"  Profiles captured: {profiles_found}")
for e in all_entries:
    print(
        f"  {e['profile']:14s}  {e['name']:<52s}  "
        f"median={e['median_ns']:.0f}ns  mad={e['mad_ns']:.0f}ns"
    )
PYEOF
