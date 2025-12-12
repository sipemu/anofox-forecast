#!/usr/bin/env python3
"""Compare Rust and tsfresh feature extraction results.

This script loads feature results from both implementations,
computes comparison metrics, and generates markdown and CSV reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Directories
RESULTS_DIR = Path(__file__).parent / "results"
RUST_DIR = RESULTS_DIR / "rust"
TSFRESH_DIR = RESULTS_DIR / "tsfresh"
OUTPUT_DIR = Path(__file__).parent / "output"

# Tolerance thresholds
RELATIVE_TOLERANCE = 1e-5  # For relative comparison
ABSOLUTE_TOLERANCE = 1e-9  # For values near zero


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature results from both implementations."""
    rust_df = pd.read_csv(RUST_DIR / "features.csv")
    tsfresh_df = pd.read_csv(TSFRESH_DIR / "features.csv")
    return rust_df, tsfresh_df


def normalize_feature_name(name: str) -> str:
    """Normalize feature names for matching.

    Both implementations use 'value__feature__param' format.
    This function normalizes variations to enable matching.
    """
    # Convert to lowercase
    name = name.lower()

    # Remove any extra whitespace
    name = name.strip()

    return name


def compare_features(rust_df: pd.DataFrame, tsfresh_df: pd.DataFrame) -> pd.DataFrame:
    """Compare features between implementations."""
    # Normalize feature names
    rust_df = rust_df.copy()
    tsfresh_df = tsfresh_df.copy()

    rust_df["normalized_name"] = rust_df["feature_name"].apply(normalize_feature_name)
    tsfresh_df["normalized_name"] = tsfresh_df["feature_name"].apply(normalize_feature_name)

    # Merge on series_type and normalized feature name
    merged = rust_df.merge(
        tsfresh_df,
        on=["series_type", "normalized_name"],
        suffixes=("_rust", "_tsfresh"),
    )

    if merged.empty:
        print("Warning: No matching features found between implementations")
        return merged

    # Compute differences
    merged["difference"] = merged["value_rust"] - merged["value_tsfresh"]
    merged["abs_difference"] = merged["difference"].abs()

    # Relative difference (handling near-zero values and NaNs)
    def compute_relative_diff(row):
        rust_val = row["value_rust"]
        tsfresh_val = row["value_tsfresh"]

        # Handle NaN cases
        if pd.isna(rust_val) and pd.isna(tsfresh_val):
            return 0.0  # Both NaN is a match
        if pd.isna(rust_val) or pd.isna(tsfresh_val):
            return np.inf  # One NaN is a mismatch

        # Handle near-zero values
        if abs(tsfresh_val) > ABSOLUTE_TOLERANCE:
            return abs(rust_val - tsfresh_val) / abs(tsfresh_val)
        elif abs(rust_val - tsfresh_val) < ABSOLUTE_TOLERANCE:
            return 0.0
        else:
            return np.inf

    merged["relative_difference"] = merged.apply(compute_relative_diff, axis=1)

    # Match status
    def check_match(row):
        rust_val = row["value_rust"]
        tsfresh_val = row["value_tsfresh"]

        # Both NaN is a match
        if pd.isna(rust_val) and pd.isna(tsfresh_val):
            return True

        # One NaN is not a match
        if pd.isna(rust_val) or pd.isna(tsfresh_val):
            return False

        # Both infinity of same sign is a match
        if np.isinf(rust_val) and np.isinf(tsfresh_val):
            return np.sign(rust_val) == np.sign(tsfresh_val)

        # Check absolute tolerance first (for near-zero values)
        if abs(rust_val - tsfresh_val) < ABSOLUTE_TOLERANCE:
            return True

        # Check relative tolerance
        if abs(tsfresh_val) > ABSOLUTE_TOLERANCE:
            return abs(rust_val - tsfresh_val) / abs(tsfresh_val) < RELATIVE_TOLERANCE

        return False

    merged["matches"] = merged.apply(check_match, axis=1)

    return merged


def compute_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics by feature."""
    if comparison.empty:
        return pd.DataFrame()

    summary = comparison.groupby("normalized_name").agg({
        "matches": ["sum", "count"],
        "abs_difference": ["mean", "max"],
        "relative_difference": ["mean", "max"],
    }).reset_index()

    summary.columns = [
        "feature",
        "matches",
        "total",
        "mean_abs_diff",
        "max_abs_diff",
        "mean_rel_diff",
        "max_rel_diff",
    ]

    summary["match_rate"] = summary["matches"] / summary["total"]

    return summary.sort_values("match_rate")


def generate_markdown_report(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    rust_df: pd.DataFrame,
    tsfresh_df: pd.DataFrame,
) -> str:
    """Generate a markdown report."""
    lines = [
        "# Feature Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "- **Rust implementation**: anofox-forecast features module",
        "- **Python implementation**: tsfresh",
        "",
        f"- **Total Rust features**: {len(rust_df)}",
        f"- **Total tsfresh features**: {len(tsfresh_df)}",
        f"- **Matched comparisons**: {len(comparison)}",
        "",
    ]

    if comparison.empty:
        lines.extend([
            "**Warning**: No matching features found between implementations.",
            "",
            "This may indicate a feature naming mismatch.",
            "",
        ])
        return "\n".join(lines)

    # Overall match rate
    total_matches = comparison["matches"].sum()
    total_comparisons = len(comparison)
    match_rate = total_matches / total_comparisons * 100

    lines.extend([
        f"### Overall Match Rate: {match_rate:.2f}%",
        f"- Matching: {total_matches}/{total_comparisons}",
        "",
    ])

    # Features with perfect matches
    perfect = summary[summary["match_rate"] == 1.0]
    if len(perfect) > 0:
        lines.extend([
            "## Perfectly Matching Features",
            "",
            f"**{len(perfect)} features** have 100% match rate across all series types:",
            "",
        ])
        for _, row in perfect.head(20).iterrows():
            lines.append(f"- `{row['feature']}`")
        if len(perfect) > 20:
            lines.append(f"- ... and {len(perfect) - 20} more")
        lines.append("")

    # Features with issues
    issues = summary[summary["match_rate"] < 1.0]
    if len(issues) > 0:
        lines.extend([
            "## Features with Discrepancies",
            "",
            "| Feature | Match Rate | Mean Abs Diff | Max Abs Diff |",
            "|---------|------------|---------------|--------------|",
        ])
        for _, row in issues.iterrows():
            lines.append(
                f"| `{row['feature'][:50]}` | {row['match_rate']:.2%} | "
                f"{row['mean_abs_diff']:.6f} | {row['max_abs_diff']:.6f} |"
            )
        lines.append("")

    # Detailed discrepancies (top 20)
    discrepancies = comparison[~comparison["matches"]].nlargest(20, "abs_difference")
    if len(discrepancies) > 0:
        lines.extend([
            "## Largest Discrepancies (Top 20)",
            "",
            "| Series | Feature | Rust | tsfresh | Difference |",
            "|--------|---------|------|---------|------------|",
        ])
        for _, row in discrepancies.iterrows():
            feature_short = row["normalized_name"][:40]
            rust_val = f"{row['value_rust']:.6f}" if pd.notna(row["value_rust"]) else "NaN"
            tsfresh_val = f"{row['value_tsfresh']:.6f}" if pd.notna(row["value_tsfresh"]) else "NaN"
            diff_val = f"{row['difference']:.6f}" if pd.notna(row["difference"]) else "N/A"
            lines.append(
                f"| {row['series_type']} | `{feature_short}` | "
                f"{rust_val} | {tsfresh_val} | {diff_val} |"
            )
        lines.append("")

    # Match rate by series type
    by_series = comparison.groupby("series_type").agg({
        "matches": ["sum", "count"]
    }).reset_index()
    by_series.columns = ["series_type", "matches", "total"]
    by_series["match_rate"] = by_series["matches"] / by_series["total"]

    lines.extend([
        "## Match Rate by Series Type",
        "",
        "| Series Type | Match Rate | Matching/Total |",
        "|-------------|------------|----------------|",
    ])
    for _, row in by_series.iterrows():
        lines.append(
            f"| {row['series_type']} | {row['match_rate']:.2%} | "
            f"{int(row['matches'])}/{int(row['total'])} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "Expected causes of discrepancies:",
        "- Different numerical precision in implementations",
        "- Different algorithm implementations (e.g., entropy calculations)",
        "- Different handling of edge cases (empty series, constant values)",
        "- Different default parameter values",
        "- Boolean features may have different representations",
        "",
        f"Tolerance thresholds used:",
        f"- Relative tolerance: {RELATIVE_TOLERANCE}",
        f"- Absolute tolerance: {ABSOLUTE_TOLERANCE}",
        "",
    ])

    return "\n".join(lines)


def main():
    """Compare features and generate reports."""
    print("=== Comparing Feature Results ===\n")

    # Check if results exist
    rust_file = RUST_DIR / "features.csv"
    tsfresh_file = TSFRESH_DIR / "features.csv"

    if not rust_file.exists():
        print(f"Error: {rust_file} not found")
        print("Run 'cargo run --example feature_export --release' first")
        return

    if not tsfresh_file.exists():
        print(f"Error: {tsfresh_file} not found")
        print("Run 'python extract_tsfresh_features.py' first")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    rust_df, tsfresh_df = load_results()
    print(f"  Rust features: {len(rust_df)}")
    print(f"  tsfresh features: {len(tsfresh_df)}")

    # Show unique feature counts
    rust_unique = rust_df["feature_name"].nunique()
    tsfresh_unique = tsfresh_df["feature_name"].nunique()
    print(f"  Rust unique features: {rust_unique}")
    print(f"  tsfresh unique features: {tsfresh_unique}")

    # Compare
    print("\nComparing features...")
    comparison = compare_features(rust_df, tsfresh_df)
    print(f"  Matched comparisons: {len(comparison)}")

    if comparison.empty:
        print("\nWarning: No features matched between implementations.")
        print("This may indicate a naming convention mismatch.")

        # Show sample feature names for debugging
        print("\nSample Rust feature names:")
        for name in rust_df["feature_name"].head(10):
            print(f"  {name}")

        print("\nSample tsfresh feature names:")
        for name in tsfresh_df["feature_name"].head(10):
            print(f"  {name}")
        return

    # Summary
    print("Computing summary...")
    summary = compute_summary(comparison)

    # Generate reports
    print("\nGenerating reports...")

    # Markdown report
    report_md = generate_markdown_report(comparison, summary, rust_df, tsfresh_df)
    report_path = OUTPUT_DIR / "feature_report.md"
    report_path.write_text(report_md)
    print(f"  Markdown report: {report_path}")

    # CSV: Detailed comparison
    comparison_path = OUTPUT_DIR / "feature_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"  Detailed comparison: {comparison_path}")

    # CSV: Summary
    summary_path = OUTPUT_DIR / "feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}")

    # Print summary
    match_rate = comparison["matches"].mean() * 100
    print(f"\n=== Overall Match Rate: {match_rate:.2f}% ===")

    # Highlight perfect and problematic features
    perfect_count = len(summary[summary["match_rate"] == 1.0])
    issue_count = len(summary[summary["match_rate"] < 1.0])
    print(f"  Perfect matches: {perfect_count} features")
    print(f"  With discrepancies: {issue_count} features")

    # Show worst features
    worst = summary[summary["match_rate"] < 0.9].nlargest(5, "max_abs_diff")
    if len(worst) > 0:
        print("\nFeatures with largest discrepancies:")
        for _, row in worst.iterrows():
            print(f"  - {row['feature']}: {row['match_rate']:.2%} match rate")


if __name__ == "__main__":
    main()
