#!/usr/bin/env python3
"""Main orchestration script for the feature validation pipeline.

This script runs all steps of the feature validation process:
1. (Uses existing) synthetic time series data
2. Extract tsfresh features from data
3. Run Rust feature extraction
4. Compare results and generate reports
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


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
        print(f"\n  {name} failed with return code {result.returncode}")
        return False

    print(f"\n  {name} completed successfully")
    return True


def run_cargo(name: str, example: str) -> bool:
    """Run a cargo example and return success status."""
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        ["cargo", "run", "--example", example, "--release"],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"\n  {name} failed with return code {result.returncode}")
        return False

    print(f"\n  {name} completed successfully")
    return True


def main():
    """Run the complete feature validation pipeline."""
    print("=" * 60)
    print("  Feature Validation Pipeline")
    print("  Comparing anofox-forecast features vs tsfresh")
    print("=" * 60)

    # Check if data exists
    data_dir = SCRIPT_DIR / "data"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print("\nData not found. Generating synthetic data first...")
        if not run_script("Generate synthetic data", "generate_data.py"):
            sys.exit(1)

    steps = [
        ("Extract tsfresh features", "extract_tsfresh_features.py", "script"),
        ("Run Rust feature extraction", "feature_export", "cargo"),
        ("Compare results", "compare_features.py", "script"),
    ]

    failed_step = None
    for name, target, step_type in steps:
        if step_type == "script":
            success = run_script(name, target)
        else:
            success = run_cargo(name, target)

        if not success:
            failed_step = name
            break

    print("\n" + "=" * 60)
    if failed_step:
        print(f"  Pipeline failed at: {failed_step}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  Feature Validation Pipeline completed successfully!")
        print("=" * 60)

        output_dir = SCRIPT_DIR / "output"
        print(f"\nReports generated in: {output_dir}")
        print("\nFiles:")
        print("  - feature_report.md      : Human-readable comparison report")
        print("  - feature_comparison.csv : Detailed feature-by-feature comparison")
        print("  - feature_summary.csv    : Summary statistics by feature")

        # Show quick summary if report exists
        report_path = output_dir / "feature_report.md"
        if report_path.exists():
            print("\n" + "-" * 60)
            print("Quick Summary (from report):")
            print("-" * 60)
            content = report_path.read_text()
            # Extract match rate line
            for line in content.split("\n"):
                if "Overall Match Rate" in line:
                    print(f"  {line.strip('#').strip()}")
                    break


if __name__ == "__main__":
    main()
