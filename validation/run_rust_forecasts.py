#!/usr/bin/env python3
"""Run Rust forecasting example via cargo.

This script executes the Rust forecast_export example which reads
the synthetic data and outputs forecast results to CSV files.
"""

import subprocess
import sys
from pathlib import Path

# Project root (one level up from validation/)
PROJECT_ROOT = Path(__file__).parent.parent


def main():
    """Run the Rust forecast_export example."""
    print("=== Running Rust Forecasts ===\n")

    # Check if data exists
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print("Error: No data files found in validation/data/")
        print("Run 'python generate_data.py' first to create the data files.")
        sys.exit(1)

    # Run cargo example
    print("Running: cargo run --example forecast_export --release")
    print()

    try:
        result = subprocess.run(
            ["cargo", "run", "--example", "forecast_export", "--release"],
            cwd=PROJECT_ROOT,
            capture_output=False,  # Show output in real-time
            text=True,
        )

        if result.returncode != 0:
            print(f"\nError: Rust example failed with return code {result.returncode}")
            sys.exit(result.returncode)

    except FileNotFoundError:
        print("Error: 'cargo' not found. Please ensure Rust is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running Rust example: {e}")
        sys.exit(1)

    # Verify results were created
    results_dir = Path(__file__).parent / "results" / "rust"
    point_file = results_dir / "point_forecasts.csv"
    ci_file = results_dir / "confidence_intervals.csv"

    if point_file.exists() and ci_file.exists():
        print("\n✓ Rust forecasts completed successfully")
    else:
        print("\nWarning: Expected output files not found")
        if not point_file.exists():
            print(f"  Missing: {point_file}")
        if not ci_file.exists():
            print(f"  Missing: {ci_file}")


if __name__ == "__main__":
    main()
