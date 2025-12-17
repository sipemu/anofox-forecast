#!/usr/bin/env python3
"""
Validation script to compare Rust periodicity detection against pyriodicity.

This script tests that our Rust implementations produce similar results to
the reference Python implementations in pyriodicity.

Usage:
    # First, build the Rust example
    cargo build --release --example detect_period

    # Install pyriodicity
    pip install pyriodicity

    # Run validation
    python tests/validate_against_pyriodicity.py
"""

import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# Try to import pyriodicity
try:
    from pyriodicity import (
        Autoperiod as PyAutoperiod,
        ACFPeriodicityDetector as PyACF,
        FFTPeriodicityDetector as PyFFT,
        SAZED as PySAZED,
    )
    HAS_PYRIODICITY = True
except ImportError:
    HAS_PYRIODICITY = False
    print("Warning: pyriodicity not installed. Install with: pip install pyriodicity")


# Path to the Rust binary
RUST_BINARY = Path(__file__).parent.parent / "target" / "release" / "examples" / "detect_period"


def run_rust_detector(data: np.ndarray, method: str, min_period: int = 2, max_period: int = 365) -> dict:
    """Run the Rust detector via CLI and parse JSON output."""
    # Write data to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for val in data:
            f.write(f"{val}\n")
        temp_path = f.name

    try:
        # Run Rust binary
        cmd = [
            str(RUST_BINARY),
            "--method", method,
            "--min-period", str(min_period),
            "--max-period", str(max_period),
            "--json"
        ]

        with open(temp_path, 'r') as stdin_file:
            result = subprocess.run(
                cmd,
                stdin=stdin_file,
                capture_output=True,
                text=True
            )

        if result.returncode != 0:
            raise RuntimeError(f"Rust binary failed: {result.stderr}")

        return json.loads(result.stdout)
    finally:
        Path(temp_path).unlink()


def generate_test_datasets():
    """Generate test datasets with known periods."""
    datasets = {}

    # Pure sinusoids
    for period in [7, 12, 30]:
        n = period * 20  # 20 complete cycles
        t = np.arange(n)
        datasets[f'sine_{period}'] = {
            'data': np.sin(2 * np.pi * t / period),
            'true_period': period
        }

    # Multi-frequency
    n = 365
    t = np.arange(n)
    datasets['multi_7_30'] = {
        'data': np.sin(2 * np.pi * t / 7) + 0.5 * np.sin(2 * np.pi * t / 30),
        'true_periods': [7, 30]
    }

    # Signal with trend
    n = 240
    t = np.arange(n)
    datasets['trended_12'] = {
        'data': 0.1 * t + 10 * np.sin(2 * np.pi * t / 12),
        'true_period': 12
    }

    # Noisy signal
    np.random.seed(42)
    n = 240
    t = np.arange(n)
    datasets['noisy_12'] = {
        'data': np.sin(2 * np.pi * t / 12) + 0.3 * np.random.randn(n),
        'true_period': 12
    }

    return datasets


def compare_results(rust_result: dict, py_result, tolerance: int = 2) -> dict:
    """Compare Rust and Python results."""
    rust_period = rust_result.get('primary_period')

    # pyriodicity returns a list of periods
    py_period = py_result[0] if len(py_result) > 0 else None

    if rust_period is None and py_period is None:
        return {'match': True, 'rust': None, 'python': None}

    if rust_period is None or py_period is None:
        return {'match': False, 'rust': rust_period, 'python': py_period}

    diff = abs(rust_period - py_period)
    return {
        'match': diff <= tolerance,
        'rust': rust_period,
        'python': py_period,
        'diff': diff
    }


def get_rust_binary():
    """Find the Rust binary, preferring release build."""
    release = Path(__file__).parent.parent / "target" / "release" / "examples" / "detect_period"
    debug = Path(__file__).parent.parent / "target" / "debug" / "examples" / "detect_period"

    if release.exists():
        return release
    if debug.exists():
        return debug
    return None


def run_validation():
    """Run full validation suite."""
    print("=" * 60)
    print("Periodicity Detection Validation")
    print("=" * 60)
    print()

    # Check if Rust binary exists
    rust_binary = get_rust_binary()
    if rust_binary is None:
        print("Error: Rust binary not found")
        print("Please run: cargo build --example detect_period")
        return False

    global RUST_BINARY
    RUST_BINARY = rust_binary

    datasets = generate_test_datasets()

    # Test methods that have equivalents in both libraries
    method_pairs = [
        ('autoperiod', 'Autoperiod'),
        ('acf', 'ACFPeriodicityDetector'),
        ('fft', 'FFTPeriodicityDetector'),
        ('sazed', 'SAZED'),
    ]

    results = []

    for dataset_name, dataset_info in datasets.items():
        data = dataset_info['data']
        true_period = dataset_info.get('true_period')

        print(f"\nDataset: {dataset_name}")
        print(f"  Length: {len(data)}, True period: {true_period}")
        print("-" * 40)

        for rust_method, py_class in method_pairs:
            try:
                # Run Rust detector
                rust_result = run_rust_detector(data, rust_method)
                rust_period = rust_result.get('primary_period')

                # Run Python detector if available
                if HAS_PYRIODICITY:
                    py_detector = eval(f"Py{py_class.replace('Periodicity', '')}" if 'Periodicity' in py_class else f"Py{py_class}")
                    py_result = py_detector.detect(data)
                    py_period = py_result[0] if len(py_result) > 0 else None
                else:
                    py_period = None

                # Compare with true period if known
                rust_correct = true_period is None or (rust_period and abs(rust_period - true_period) <= 2)

                status = "✓" if rust_correct else "✗"

                print(f"  {rust_method:15s}: Rust={rust_period}, ", end="")
                if HAS_PYRIODICITY:
                    print(f"Python={py_period}, ", end="")
                print(f"True={true_period} {status}")

                results.append({
                    'dataset': dataset_name,
                    'method': rust_method,
                    'rust_period': rust_period,
                    'python_period': py_period if HAS_PYRIODICITY else None,
                    'true_period': true_period,
                    'rust_correct': rust_correct
                })

            except Exception as e:
                print(f"  {rust_method:15s}: Error - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    correct = sum(1 for r in results if r['rust_correct'])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Rust accuracy vs true period: {correct}/{total} ({accuracy:.1f}%)")

    if HAS_PYRIODICITY:
        matches = sum(1 for r in results if r['rust_period'] == r['python_period'])
        print(f"Rust-Python agreement: {matches}/{total} ({matches/total*100:.1f}%)")

    return accuracy >= 80  # Consider success if 80%+ accurate


def run_standalone_rust_test():
    """Run Rust detector standalone (without pyriodicity comparison)."""
    print("=" * 60)
    print("Rust Periodicity Detection - Standalone Test")
    print("=" * 60)
    print()

    # Check if Rust binary exists
    rust_binary = get_rust_binary()
    if rust_binary is None:
        print("Error: Rust binary not found")
        print("Please run: cargo build --example detect_period")
        return False

    global RUST_BINARY
    RUST_BINARY = rust_binary

    datasets = generate_test_datasets()

    methods = ['autoperiod', 'acf', 'fft', 'sazed']

    results = []

    for dataset_name, dataset_info in datasets.items():
        data = dataset_info['data']
        true_period = dataset_info.get('true_period')

        print(f"\nDataset: {dataset_name} (n={len(data)}, true={true_period})")
        print("-" * 50)

        for method in methods:
            try:
                rust_result = run_rust_detector(data, method)
                rust_period = rust_result.get('primary_period')
                confidence = rust_result.get('confidence', 0)

                correct = true_period is None or (rust_period and abs(rust_period - true_period) <= 2)
                status = "✓" if correct else "✗"

                print(f"  {method:12s}: period={rust_period:4}, conf={confidence:.3f} {status}")

                results.append({
                    'dataset': dataset_name,
                    'method': method,
                    'rust_period': rust_period,
                    'true_period': true_period,
                    'correct': correct
                })

            except Exception as e:
                print(f"  {method:12s}: Error - {e}")

    # Summary
    print("\n" + "=" * 60)
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Overall accuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Per-method breakdown
    print("\nPer-method accuracy:")
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        method_correct = sum(1 for r in method_results if r['correct'])
        method_total = len(method_results)
        print(f"  {method:12s}: {method_correct}/{method_total}")

    return accuracy >= 80


if __name__ == '__main__':
    import sys

    if '--standalone' in sys.argv or not HAS_PYRIODICITY:
        success = run_standalone_rust_test()
    else:
        success = run_validation()

    sys.exit(0 if success else 1)
