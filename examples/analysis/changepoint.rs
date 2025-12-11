//! Changepoint Detection example using PELT.
//!
//! Run with: cargo run --example changepoint

use anofox_forecast::changepoint::{pelt_detect, CostFunction, PeltConfig};

fn main() {
    println!("=== Changepoint Detection Example ===\n");

    println!("PELT (Pruned Exact Linear Time) algorithm detects points");
    println!("where the statistical properties of a series change.\n");

    // 1. Clear Level Shift
    println!("--- Clear Level Shift ---");

    let mut series1: Vec<f64> = vec![10.0; 30];
    series1.extend(vec![50.0; 30]);
    series1.extend(vec![25.0; 30]);

    println!("Data: [10.0 × 30] + [50.0 × 30] + [25.0 × 30]");
    println!("Expected changepoints: 30, 60\n");

    let config1 = PeltConfig::default().penalty(5.0);
    let result1 = pelt_detect(&series1, &config1);

    println!("Detected changepoints: {:?}", result1.changepoints);
    println!("Number of changepoints: {}", result1.n_changepoints);
    println!("Segments: {:?}", result1.segments);
    println!("Segment means: {:?}", result1.segment_means(&series1));

    // 2. Effect of Penalty Parameter
    println!("\n--- Effect of Penalty Parameter ---");

    let mut series2: Vec<f64> = (0..50).map(|_| 10.0).collect();
    series2.extend((0..50).map(|_| 30.0));

    println!("\nData: [10.0 × 50] + [30.0 × 50]");
    println!("True changepoint: 50\n");

    println!("{:<15} {:>15} {:>15}", "Penalty", "Changepoints", "Count");
    println!("{:-<47}", "");

    for penalty in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0] {
        let config = PeltConfig::default().penalty(penalty);
        let result = pelt_detect(&series2, &config);
        println!(
            "{:<15.1} {:>15?} {:>15}",
            penalty, result.changepoints, result.n_changepoints
        );
    }

    // 3. BIC and AIC Penalties
    println!("\n--- Information Criterion Penalties ---");

    let n = series2.len();
    let config_bic = PeltConfig::with_bic_penalty(n);
    let config_aic = PeltConfig::with_aic_penalty();

    let result_bic = pelt_detect(&series2, &config_bic);
    let result_aic = pelt_detect(&series2, &config_aic);

    println!("BIC penalty (log(n) = {:.2}):", (n as f64).ln());
    println!("  Changepoints: {:?}", result_bic.changepoints);

    println!("\nAIC penalty (2.0):");
    println!("  Changepoints: {:?}", result_aic.changepoints);

    // 4. Different Cost Functions
    println!("\n--- Cost Functions ---");

    // Generate data with variance change
    let mut series3: Vec<f64> = (0..50)
        .map(|i| 10.0 + 0.5 * (i as f64 * 0.3).sin())
        .collect();
    series3.extend((0..50).map(|i| 10.0 + 5.0 * (i as f64 * 0.3).sin())); // Higher variance

    println!("Data: Low variance (50) → High variance (50)");

    let config_l2 = PeltConfig::default()
        .cost_function(CostFunction::L2)
        .penalty(10.0);
    let config_normal = PeltConfig::default()
        .cost_function(CostFunction::Normal)
        .penalty(10.0);

    let result_l2 = pelt_detect(&series3, &config_l2);
    let result_normal = pelt_detect(&series3, &config_normal);

    println!("\nL2 cost (mean change detection):");
    println!("  Changepoints: {:?}", result_l2.changepoints);

    println!("\nNormal cost (mean and variance change):");
    println!("  Changepoints: {:?}", result_normal.changepoints);

    // 5. Minimum Segment Length
    println!("\n--- Minimum Segment Length ---");

    let mut series4: Vec<f64> = vec![0.0; 5];
    series4.extend(vec![100.0; 45]);
    series4.extend(vec![0.0; 50]);

    println!("Data: Short segment (5) followed by longer segments");

    let config_min2 = PeltConfig::default().penalty(5.0).min_segment_length(2);
    let config_min10 = PeltConfig::default().penalty(5.0).min_segment_length(10);

    let result_min2 = pelt_detect(&series4, &config_min2);
    let result_min10 = pelt_detect(&series4, &config_min10);

    println!("\nMin segment = 2:");
    println!("  Changepoints: {:?}", result_min2.changepoints);

    println!("\nMin segment = 10:");
    println!("  Changepoints: {:?}", result_min10.changepoints);

    // 6. Gradual Change vs Step Change
    println!("\n--- Gradual vs Step Changes ---");

    // Step change
    let mut step_series: Vec<f64> = vec![10.0; 50];
    step_series.extend(vec![30.0; 50]);

    // Gradual change
    let gradual_series: Vec<f64> = (0..100).map(|i| 10.0 + 20.0 * (i as f64 / 100.0)).collect();

    let config_change = PeltConfig::default().penalty(10.0);

    let result_step = pelt_detect(&step_series, &config_change);
    let result_gradual = pelt_detect(&gradual_series, &config_change);

    println!("Step change (10 → 30 at t=50):");
    println!("  Changepoints: {:?}", result_step.changepoints);

    println!("\nGradual change (10 → 30 over 100 points):");
    println!("  Changepoints: {:?}", result_gradual.changepoints);

    // 7. Multiple Changepoints
    println!("\n--- Multiple Changepoints ---");

    // Generate series with known changepoints
    let cp_locations = vec![20, 45, 70, 85];
    let levels = vec![10.0, 30.0, 15.0, 40.0, 20.0];

    let mut multi_series: Vec<f64> = Vec::new();
    let mut current_level_idx = 0;
    for i in 0..100 {
        if current_level_idx < cp_locations.len() && i == cp_locations[current_level_idx] {
            current_level_idx += 1;
        }
        multi_series.push(levels[current_level_idx] + 0.5 * ((i as f64 * 0.1).sin()));
    }

    println!("True changepoints: {:?}", cp_locations);
    println!("Levels: {:?}", levels);

    let config_multi = PeltConfig::default().penalty(5.0);
    let result_multi = pelt_detect(&multi_series, &config_multi);

    println!("\nDetected changepoints: {:?}", result_multi.changepoints);
    println!(
        "Number detected: {} (true: {})",
        result_multi.n_changepoints,
        cp_locations.len()
    );

    // 8. Segment Analysis
    println!("\n--- Segment Analysis ---");

    println!("\nSegment details:");
    println!(
        "{:<15} {:>10} {:>10} {:>12}",
        "Segment", "Start", "End", "Mean"
    );
    println!("{:-<49}", "");

    for (i, &(start, end)) in result_multi.segments.iter().enumerate() {
        let segment = &multi_series[start..end];
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        println!(
            "{:<15} {:>10} {:>10} {:>12.4}",
            format!("Segment {}", i + 1),
            start,
            end,
            mean
        );
    }

    // 9. No Changepoints Case
    println!("\n--- No Changepoints Case ---");

    let constant_series: Vec<f64> = (0..100)
        .map(|i| 10.0 + 0.1 * (i as f64 * 0.3).sin())
        .collect();

    let config_const = PeltConfig::default().penalty(50.0);
    let result_const = pelt_detect(&constant_series, &config_const);

    println!("Nearly constant series with high penalty:");
    println!("  Changepoints: {:?}", result_const.changepoints);
    println!("  Single segment: {:?}", result_const.segments);

    // 10. Practical Guidance
    println!("\n--- Practical Guidance ---");
    println!(
        "
Penalty Selection:
  - Higher penalty → fewer changepoints (more conservative)
  - Lower penalty → more changepoints (may detect noise)
  - BIC: log(n) - good default, slightly conservative
  - AIC: 2 - may overfit with many changepoints

Cost Functions:
  - L2: Detects mean changes (most common)
  - Normal: Detects mean and variance changes
  - L1: More robust to outliers

Minimum Segment Length:
  - Prevents detection of very short segments
  - Set based on domain knowledge
  - Typically: 2-10% of series length

Common Issues:
  - Too many changepoints: increase penalty
  - Missing true changepoints: decrease penalty
  - Gradual trends: may detect multiple small changes
"
    );

    println!("=== Changepoint Detection Example Complete ===");
}
