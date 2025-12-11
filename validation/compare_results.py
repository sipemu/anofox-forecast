#!/usr/bin/env python3
"""Compare Rust and statsforecast results and generate reports.

This script loads forecast results from both implementations,
computes comparison metrics, and generates markdown and CSV reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Directories
RESULTS_DIR = Path(__file__).parent / "results"
RUST_DIR = RESULTS_DIR / "rust"
SF_DIR = RESULTS_DIR / "statsforecast"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all result files.

    Returns:
        Tuple of (rust_point, rust_ci, sf_point, sf_ci) DataFrames
    """
    rust_point = pd.read_csv(RUST_DIR / "point_forecasts.csv")
    rust_ci = pd.read_csv(RUST_DIR / "confidence_intervals.csv")
    sf_point = pd.read_csv(SF_DIR / "point_forecasts.csv")
    sf_ci = pd.read_csv(SF_DIR / "confidence_intervals.csv")

    return rust_point, rust_ci, sf_point, sf_ci


def compute_point_comparison(rust_df: pd.DataFrame, sf_df: pd.DataFrame) -> pd.DataFrame:
    """Compare point forecasts between implementations."""
    # Merge on series_type, model, step
    merged = rust_df.merge(
        sf_df,
        on=["series_type", "model", "step"],
        suffixes=("_rust", "_sf"),
    )

    # Compute differences
    merged["difference"] = merged["forecast_rust"] - merged["forecast_sf"]
    merged["abs_difference"] = merged["difference"].abs()
    merged["pct_difference"] = (merged["difference"] / merged["forecast_sf"].abs()) * 100

    return merged


def compute_ci_comparison(rust_df: pd.DataFrame, sf_df: pd.DataFrame) -> pd.DataFrame:
    """Compare confidence intervals between implementations."""
    # Merge on series_type, model, step, level
    merged = rust_df.merge(
        sf_df,
        on=["series_type", "model", "step", "level"],
        suffixes=("_rust", "_sf"),
    )

    # Compute CI widths
    merged["width_rust"] = merged["upper_rust"] - merged["lower_rust"]
    merged["width_sf"] = merged["upper_sf"] - merged["lower_sf"]
    merged["width_difference"] = merged["width_rust"] - merged["width_sf"]

    # Compute bound differences
    merged["lower_diff"] = merged["lower_rust"] - merged["lower_sf"]
    merged["upper_diff"] = merged["upper_rust"] - merged["upper_sf"]

    return merged


def compute_summary_metrics(point_comparison: pd.DataFrame, ci_comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics for each model/series combination."""
    summaries = []

    for (series_type, model), group in point_comparison.groupby(["series_type", "model"]):
        summary = {
            "series_type": series_type,
            "model": model,
            "n_forecasts": len(group),
            "mad": group["abs_difference"].mean(),  # Mean Absolute Difference
            "median_abs_diff": group["abs_difference"].median(),  # Median Absolute Difference
            "max_diff": group["abs_difference"].max(),
            "mean_diff": group["difference"].mean(),
            "std_diff": group["difference"].std(),
        }

        # Correlation
        if len(group) > 1 and group["forecast_rust"].std() > 0 and group["forecast_sf"].std() > 0:
            summary["correlation"] = group["forecast_rust"].corr(group["forecast_sf"])
        else:
            summary["correlation"] = np.nan

        # CI width differences by level
        ci_group = ci_comparison[
            (ci_comparison["series_type"] == series_type) &
            (ci_comparison["model"] == model)
        ]

        for level in [80, 90, 95]:
            level_data = ci_group[ci_group["level"] == level]
            if len(level_data) > 0:
                summary[f"ci_width_diff_{level}"] = level_data["width_difference"].mean()
            else:
                summary[f"ci_width_diff_{level}"] = np.nan

        summaries.append(summary)

    return pd.DataFrame(summaries)


def compute_step_metrics(point_comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics broken down by forecast horizon step."""
    step_summaries = []

    for step, group in point_comparison.groupby("step"):
        summary = {
            "step": step,
            "n_forecasts": len(group),
            "mad": group["abs_difference"].mean(),
            "median_abs_diff": group["abs_difference"].median(),
            "max_diff": group["abs_difference"].max(),
            "mean_diff": group["difference"].mean(),
            "std_diff": group["difference"].std(),
        }
        step_summaries.append(summary)

    return pd.DataFrame(step_summaries)


def generate_markdown_report(
    point_comparison: pd.DataFrame,
    ci_comparison: pd.DataFrame,
    summary: pd.DataFrame,
    step_metrics: pd.DataFrame,
) -> str:
    """Generate a markdown report."""
    lines = [
        "# Forecast Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Rust implementation**: anofox-forecast",
        f"- **Python implementation**: statsforecast (NIXTLA)",
        f"- **Forecast horizon**: 12 steps",
        f"- **Confidence levels**: 80%, 90%, 95%",
        "",
    ]

    # Overall statistics
    models_compared = summary["model"].nunique()
    series_compared = summary["series_type"].nunique()
    total_comparisons = len(summary)

    lines.extend([
        f"- **Models compared**: {models_compared}",
        f"- **Series types**: {series_compared}",
        f"- **Total comparisons**: {total_comparisons}",
        "",
    ])

    # Models with high correlation (good agreement)
    high_corr = summary[summary["correlation"] >= 0.99]
    if len(high_corr) > 0:
        lines.extend([
            f"- **High agreement (corr >= 0.99)**: {len(high_corr)} combinations",
        ])

    # Models with low correlation (potential issues)
    low_corr = summary[summary["correlation"] < 0.95]
    if len(low_corr) > 0:
        lines.extend([
            f"- **Lower agreement (corr < 0.95)**: {len(low_corr)} combinations",
        ])

    lines.extend(["", "---", ""])

    # Results by model
    lines.extend([
        "## Results by Model",
        "",
    ])

    for model in summary["model"].unique():
        model_data = summary[summary["model"] == model]

        lines.extend([
            f"### {model}",
            "",
            "| Series Type | MAD | Median | Max Diff | Correlation | CI Width Diff (95%) |",
            "|-------------|-----|--------|----------|-------------|---------------------|",
        ])

        for _, row in model_data.iterrows():
            corr_str = f"{row['correlation']:.4f}" if not np.isnan(row["correlation"]) else "N/A"
            ci_str = f"{row['ci_width_diff_95']:.4f}" if not np.isnan(row.get("ci_width_diff_95", np.nan)) else "N/A"
            median_str = f"{row['median_abs_diff']:.4f}" if not np.isnan(row.get("median_abs_diff", np.nan)) else "N/A"

            lines.append(
                f"| {row['series_type']} | {row['mad']:.4f} | {median_str} | {row['max_diff']:.4f} | {corr_str} | {ci_str} |"
            )

        lines.extend(["", ""])

    # Confidence Interval Comparison
    lines.extend([
        "---",
        "",
        "## Confidence Interval Comparison",
        "",
        "Mean CI width differences (Rust - statsforecast) by level:",
        "",
        "| Model | Series | 80% | 90% | 95% |",
        "|-------|--------|-----|-----|-----|",
    ])

    for _, row in summary.iterrows():
        ci80 = f"{row.get('ci_width_diff_80', np.nan):.4f}" if not np.isnan(row.get("ci_width_diff_80", np.nan)) else "N/A"
        ci90 = f"{row.get('ci_width_diff_90', np.nan):.4f}" if not np.isnan(row.get("ci_width_diff_90", np.nan)) else "N/A"
        ci95 = f"{row.get('ci_width_diff_95', np.nan):.4f}" if not np.isnan(row.get("ci_width_diff_95", np.nan)) else "N/A"
        lines.append(f"| {row['model']} | {row['series_type']} | {ci80} | {ci90} | {ci95} |")

    lines.extend(["", ""])

    # Detailed point forecast differences
    lines.extend([
        "---",
        "",
        "## Detailed Point Forecast Differences",
        "",
        "Largest absolute differences:",
        "",
    ])

    top_diffs = point_comparison.nlargest(10, "abs_difference")
    if len(top_diffs) > 0:
        lines.extend([
            "| Model | Series | Step | Rust | statsforecast | Difference |",
            "|-------|--------|------|------|---------------|------------|",
        ])
        for _, row in top_diffs.iterrows():
            lines.append(
                f"| {row['model']} | {row['series_type']} | {row['step']} | "
                f"{row['forecast_rust']:.4f} | {row['forecast_sf']:.4f} | {row['difference']:.4f} |"
            )

    # Per-step breakdown
    lines.extend([
        "",
        "---",
        "",
        "## Metrics by Forecast Horizon Step",
        "",
        "Aggregated metrics across all models and series types by forecast step:",
        "",
        "| Step | MAD | Median | Max Diff | Mean Diff | Std |",
        "|------|-----|--------|----------|-----------|-----|",
    ])

    for _, row in step_metrics.iterrows():
        lines.append(
            f"| {int(row['step'])} | {row['mad']:.4f} | {row['median_abs_diff']:.4f} | "
            f"{row['max_diff']:.4f} | {row['mean_diff']:.4f} | {row['std_diff']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **MAD**: Mean Absolute Difference between forecasts",
        "- **Median**: Median Absolute Difference (robust to outliers)",
        "- **Max Diff**: Maximum absolute difference",
        "- **Correlation**: Pearson correlation between forecast values",
        "- **CI Width Diff**: Mean difference in confidence interval width (Rust - statsforecast)",
        "",
        "Differences are expected due to:",
        "- Different optimization algorithms for parameter estimation",
        "- Different numerical precision",
        "- Different default parameter values",
        "- Implementation variations in confidence interval calculation",
        "",
    ])

    return "\n".join(lines)


def main():
    """Compare results and generate reports."""
    print("=== Comparing Forecast Results ===\n")

    # Check if results exist
    if not RUST_DIR.exists() or not SF_DIR.exists():
        print("Error: Results directories not found.")
        print("Run 'python run_rust_forecasts.py' and 'python run_statsforecast.py' first.")
        return

    required_files = [
        RUST_DIR / "point_forecasts.csv",
        RUST_DIR / "confidence_intervals.csv",
        SF_DIR / "point_forecasts.csv",
        SF_DIR / "confidence_intervals.csv",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("Error: Missing result files:")
        for f in missing:
            print(f"  - {f}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    rust_point, rust_ci, sf_point, sf_ci = load_results()

    print(f"  Rust point forecasts: {len(rust_point)} rows")
    print(f"  Rust CI forecasts: {len(rust_ci)} rows")
    print(f"  statsforecast point forecasts: {len(sf_point)} rows")
    print(f"  statsforecast CI forecasts: {len(sf_ci)} rows")

    # Find common models
    rust_models = set(zip(rust_point["series_type"], rust_point["model"]))
    sf_models = set(zip(sf_point["series_type"], sf_point["model"]))
    common_models = rust_models & sf_models

    print(f"\nCommon model/series combinations: {len(common_models)}")
    if rust_models - sf_models:
        print(f"  Only in Rust: {rust_models - sf_models}")
    if sf_models - rust_models:
        print(f"  Only in statsforecast: {sf_models - rust_models}")

    # Compare point forecasts
    print("\nComparing point forecasts...")
    point_comparison = compute_point_comparison(rust_point, sf_point)

    # Compare confidence intervals
    print("Comparing confidence intervals...")
    ci_comparison = compute_ci_comparison(rust_ci, sf_ci)

    # Compute summary metrics
    print("Computing summary metrics...")
    summary = compute_summary_metrics(point_comparison, ci_comparison)

    # Compute per-step metrics
    print("Computing per-step metrics...")
    step_metrics = compute_step_metrics(point_comparison)

    # Generate reports
    print("\nGenerating reports...")

    # Markdown report
    report_md = generate_markdown_report(point_comparison, ci_comparison, summary, step_metrics)
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report_md)
    print(f"  ✓ Markdown report: {report_path}")

    # CSV: Step metrics
    step_path = OUTPUT_DIR / "step_metrics.csv"
    step_metrics.to_csv(step_path, index=False)
    print(f"  ✓ Step metrics CSV: {step_path}")

    # CSV: Point forecasts comparison
    point_path = OUTPUT_DIR / "point_forecasts.csv"
    point_comparison.to_csv(point_path, index=False)
    print(f"  ✓ Point forecasts CSV: {point_path}")

    # CSV: Confidence intervals comparison
    ci_path = OUTPUT_DIR / "confidence_intervals.csv"
    ci_comparison.to_csv(ci_path, index=False)
    print(f"  ✓ Confidence intervals CSV: {ci_path}")

    # CSV: Summary metrics
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  ✓ Summary metrics CSV: {summary_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"\nOverall Mean Absolute Difference: {summary['mad'].mean():.6f}")
    print(f"Overall Mean Correlation: {summary['correlation'].mean():.6f}")

    # Highlight any significant differences
    significant_diffs = summary[summary["mad"] > 1.0]
    if len(significant_diffs) > 0:
        print("\nCombinations with MAD > 1.0:")
        for _, row in significant_diffs.iterrows():
            print(f"  - {row['model']} / {row['series_type']}: MAD = {row['mad']:.4f}")

    low_corr = summary[summary["correlation"] < 0.95]
    if len(low_corr) > 0:
        print("\nCombinations with correlation < 0.95:")
        for _, row in low_corr.iterrows():
            corr = row["correlation"]
            print(f"  - {row['model']} / {row['series_type']}: corr = {corr:.4f}")

    print("\n=== Comparison Complete ===")


if __name__ == "__main__":
    main()
