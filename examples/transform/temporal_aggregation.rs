//! Temporal Aggregation Example
//!
//! This example demonstrates temporal aggregation and resampling
//! operations on time series: aggregate, downsample, upsample,
//! and sliding window aggregation.
//!
//! Run with: cargo run --example temporal_aggregation

use anofox_forecast::core::{AggregationMethod, InterpolationMethod, TimeSeries};
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Temporal Aggregation Example ===\n");

    // Create a sample hourly time series
    let n = 24;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    // Simulate hourly electricity demand (kW)
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let base = 100.0;
            let daily_pattern = 30.0 * ((i as f64 - 6.0) * std::f64::consts::PI / 12.0).sin();
            let noise = ((i * 7 + 3) % 11) as f64 - 5.0;
            (base + daily_pattern + noise).max(50.0)
        })
        .collect();

    let mut ts = TimeSeries::univariate(timestamps, values).unwrap();
    ts.set_frequency(Duration::hours(1));

    println!("Original hourly series: {} observations", ts.len());
    println!(
        "Values: {:?}\n",
        ts.primary_values()
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
    );

    // =========================================================================
    // 1. Aggregate with Sum
    // =========================================================================
    println!("--- Aggregate: Sum (period=4, hourly -> 4-hour totals) ---\n");

    let agg_sum = ts.aggregate(4, AggregationMethod::Sum);

    println!("Result: {} groups from {} observations", agg_sum.len(), ts.len());
    for i in 0..agg_sum.len() {
        let start_hour = i * 4;
        let end_hour = ((i + 1) * 4).min(n);
        println!(
            "  Hours {:>2}-{:>2}: sum = {:.1}",
            start_hour,
            end_hour - 1,
            agg_sum.primary_values()[i]
        );
    }

    // =========================================================================
    // 2. Aggregate with Mean
    // =========================================================================
    println!("\n--- Aggregate: Mean (period=6, hourly -> 6-hour averages) ---\n");

    let agg_mean = ts.aggregate(6, AggregationMethod::Mean);

    println!("Result: {} groups from {} observations", agg_mean.len(), ts.len());
    for i in 0..agg_mean.len() {
        let start_hour = i * 6;
        let end_hour = ((i + 1) * 6).min(n);
        println!(
            "  Hours {:>2}-{:>2}: mean = {:.2}",
            start_hour,
            end_hour - 1,
            agg_mean.primary_values()[i]
        );
    }

    // =========================================================================
    // 3. Other Aggregation Methods
    // =========================================================================
    println!("\n--- Aggregation Methods Comparison (period=8) ---\n");

    let methods = [
        ("Sum", AggregationMethod::Sum),
        ("Mean", AggregationMethod::Mean),
        ("Median", AggregationMethod::Median),
        ("First", AggregationMethod::First),
        ("Last", AggregationMethod::Last),
        ("Min", AggregationMethod::Min),
        ("Max", AggregationMethod::Max),
    ];

    println!(
        "  {:>8}  {:>10}  {:>10}  {:>10}",
        "Method", "Group 1", "Group 2", "Group 3"
    );
    println!("  {:-<44}", "");
    for (name, method) in &methods {
        let agg = ts.aggregate(8, *method);
        let vals: Vec<String> = agg
            .primary_values()
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect();
        println!(
            "  {:>8}  {:>10}  {:>10}  {:>10}",
            name,
            vals.first().unwrap_or(&"-".to_string()),
            vals.get(1).unwrap_or(&"-".to_string()),
            vals.get(2).unwrap_or(&"-".to_string()),
        );
    }

    // =========================================================================
    // 4. Downsample
    // =========================================================================
    println!("\n--- Downsample (factor=3, keep every 3rd point) ---\n");

    let downsampled = ts.downsample(3);

    println!(
        "Original:    {} points",
        ts.len()
    );
    println!(
        "Downsampled: {} points (factor=3)\n",
        downsampled.len()
    );
    println!(
        "  {:>6}  {:>10}  {:>10}",
        "Index", "Original", "Downsampled"
    );
    println!("  {:-<30}", "");
    for (j, &v) in downsampled.primary_values().iter().enumerate() {
        let orig_idx = j * 3;
        println!(
            "  {:>6}  {:>10.1}  {:>10.1}",
            orig_idx,
            ts.primary_values()[orig_idx],
            v
        );
    }

    // =========================================================================
    // 5. Upsample with Linear Interpolation
    // =========================================================================
    println!("\n--- Upsample: Linear Interpolation (factor=3) ---\n");

    // Use a small series for clarity
    let small_ts = TimeSeries::univariate(
        (0..5)
            .map(|i| {
                Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64)
            })
            .collect(),
        vec![10.0, 20.0, 15.0, 25.0, 30.0],
    )
    .unwrap();

    let upsampled_linear = small_ts.upsample(3, InterpolationMethod::Linear);

    println!(
        "Original:  {} points -> Upsampled: {} points\n",
        small_ts.len(),
        upsampled_linear.len()
    );
    println!(
        "  {:>4}  {:>10}  {:>8}",
        "Idx", "Value", "Source"
    );
    println!("  {:-<26}", "");
    for (i, &v) in upsampled_linear.primary_values().iter().enumerate() {
        let source = if i % 3 == 0 { "original" } else { "interp" };
        println!("  {:>4}  {:>10.2}  {:>8}", i, v, source);
    }

    // =========================================================================
    // 6. Upsample with Forward Fill
    // =========================================================================
    println!("\n--- Upsample: Forward Fill (factor=3) ---\n");

    let upsampled_ff = small_ts.upsample(3, InterpolationMethod::ForwardFill);

    println!(
        "  {:>4}  {:>10}  {:>10}",
        "Idx", "Linear", "FwdFill"
    );
    println!("  {:-<34}", "");
    for i in 0..upsampled_linear.len() {
        println!(
            "  {:>4}  {:>10.2}  {:>10.2}",
            i,
            upsampled_linear.primary_values()[i],
            upsampled_ff.primary_values()[i]
        );
    }

    // =========================================================================
    // 7. Sliding Window Aggregation
    // =========================================================================
    println!("\n--- Sliding Window Aggregation ---\n");

    // Rolling sum: window=4, step=1
    let rolling_sum = ts.sliding_window_aggregate(4, 1, AggregationMethod::Sum);
    println!(
        "Rolling sum (window=4, step=1): {} output points from {} input",
        rolling_sum.len(),
        ts.len()
    );
    println!(
        "  First 8: {:?}",
        rolling_sum.primary_values()[..8.min(rolling_sum.len())]
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
    );

    // Rolling mean: window=4, step=1
    let rolling_mean = ts.sliding_window_aggregate(4, 1, AggregationMethod::Mean);
    println!(
        "\nRolling mean (window=4, step=1): {} output points",
        rolling_mean.len()
    );
    println!(
        "  First 8: {:?}",
        rolling_mean.primary_values()[..8.min(rolling_mean.len())]
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // Non-overlapping windows: window=6, step=6
    let non_overlapping = ts.sliding_window_aggregate(6, 6, AggregationMethod::Mean);
    println!(
        "\nNon-overlapping mean (window=6, step=6): {} output points",
        non_overlapping.len()
    );
    for (i, &v) in non_overlapping.primary_values().iter().enumerate() {
        println!("  Window {}: mean = {:.2}", i + 1, v);
    }

    // Custom step: window=6, step=2
    let stepped = ts.sliding_window_aggregate(6, 2, AggregationMethod::Mean);
    println!(
        "\nOverlapping mean (window=6, step=2): {} output points",
        stepped.len()
    );
    println!(
        "  First 6: {:?}",
        stepped.primary_values()[..6.min(stepped.len())]
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // =========================================================================
    // 8. Chained Operations
    // =========================================================================
    println!("\n--- Chaining: Aggregate then Downsample ---\n");

    let chained = ts.aggregate(3, AggregationMethod::Mean);
    let chained = chained.downsample(2);

    println!(
        "Original {} pts -> aggregate(3, Mean) -> downsample(2) -> {} pts",
        ts.len(),
        chained.len()
    );
    println!(
        "  Values: {:?}",
        chained
            .primary_values()
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
    );

    // =========================================================================
    // Summary
    // =========================================================================
    println!(
        "
--- Summary ---

Temporal aggregation operations:

  aggregate(period, method)
    Groups consecutive points and applies Sum, Mean, Median,
    First, Last, Min, or Max.

  downsample(factor)
    Decimation: keeps every factor-th observation.
    No smoothing, just subsampling.

  upsample(factor, method)
    Inserts (factor-1) points between each pair.
    Interpolation: Linear, ForwardFill, BackwardFill, Zero.

  sliding_window_aggregate(window, step, method)
    Rolling window computation with configurable overlap.
    step=1 for fully overlapping, step=window for non-overlapping.
"
    );

    println!("=== Temporal Aggregation Example Complete ===");
}
