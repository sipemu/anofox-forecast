#!/usr/bin/env python3
"""Main orchestration script for the validation pipeline.

This script runs all steps of the validation process:
1. Generate synthetic time series data
2. Run Rust forecasts
3. Run statsforecast
4. Compare results and generate reports
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_script(name: str, script: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print(f"{'='*60}\n")

    script_path = SCRIPT_DIR / script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPT_DIR,
    )

    if result.returncode != 0:
        print(f"\n❌ {name} failed with return code {result.returncode}")
        return False

    print(f"\n✓ {name} completed successfully")
    return True


def main():
    """Run the complete validation pipeline."""
    print("=" * 60)
    print("  Forecast Validation Pipeline")
    print("  Comparing anofox-forecast (Rust) vs statsforecast (NIXTLA)")
    print("=" * 60)

    steps = [
        ("Generate synthetic data", "generate_data.py"),
        ("Run Rust forecasts", "run_rust_forecasts.py"),
        ("Run statsforecast", "run_statsforecast.py"),
        ("Compare results", "compare_results.py"),
    ]

    failed_step = None
    for name, script in steps:
        if not run_script(name, script):
            failed_step = name
            break

    print("\n" + "=" * 60)
    if failed_step:
        print(f"  Pipeline failed at: {failed_step}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  Pipeline completed successfully!")
        print("=" * 60)

        # Print output locations
        output_dir = SCRIPT_DIR / "output"
        print(f"\nReports generated in: {output_dir}")
        print("\nFiles:")
        print(f"  - report.md           : Human-readable comparison report")
        print(f"  - point_forecasts.csv : Detailed point forecast comparison")
        print(f"  - confidence_intervals.csv : CI comparison")
        print(f"  - summary_metrics.csv : Summary metrics by model/series")


if __name__ == "__main__":
    main()
