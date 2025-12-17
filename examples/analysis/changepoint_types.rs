//! Changepoint Detection Types example.
//!
//! Demonstrates the different cost functions available for detecting
//! various types of changepoints in time series data.
//!
//! Run with: cargo run --example changepoint_types

use anofox_forecast::changepoint::{pelt_detect, CostFunction, PeltConfig};
use std::f64::consts::PI;

fn main() {
    println!("=== Changepoint Detection: Cost Function Types ===\n");

    println!("PELT can detect different types of changes depending on the cost function.");
    println!("This example demonstrates each cost function with synthetic data.\n");

    // 1. L2: Mean Change Detection
    l2_mean_change();

    // 2. L1: Robust to Outliers
    l1_robust_detection();

    // 3. Normal: Variance Change Detection
    normal_variance_change();

    // 4. MeanVariance: Joint Mean+Variance Detection
    mean_variance_joint();

    // 5. LinearTrend: Slope Change Detection
    linear_trend_slope();

    // 6. Cusum: Sustained Shift Detection
    cusum_sustained_shift();

    // 7. Periodicity: Seasonal Pattern Change
    periodicity_seasonal();

    // 8. Poisson: Count Data Rate Change
    poisson_rate_change();

    // 9. Comparative: Same data, different cost functions
    comparative_analysis();

    // 10. Summary guidance
    print_guidance();

    println!("=== Changepoint Detection Types Example Complete ===");
}

/// L2 Cost: Detects changes in mean (most common use case)
fn l2_mean_change() {
    println!("--- 1. L2: Mean Change Detection ---\n");

    println!("L2 cost detects changes in the mean of the series.");
    println!("Best for: General-purpose changepoint detection.\n");

    // Clear level shift: 10 -> 30
    let mut series: Vec<f64> = vec![10.0; 50];
    series.extend(vec![30.0; 50]);

    println!("Data: [10.0 x 50] + [30.0 x 50]");
    println!("True changepoint: 50\n");

    let config = PeltConfig::default()
        .cost_function(CostFunction::L2)
        .penalty(5.0);
    let result = pelt_detect(&series, &config);

    println!("Config: L2 cost, penalty=5.0");
    println!("Detected changepoints: {:?}", result.changepoints);
    println!("Segment means: {:?}", result.segment_means(&series));

    if result.changepoints.contains(&50) {
        println!("Result: Correctly detected mean shift at index 50\n");
    }
}

/// L1 Cost: Robust to outliers (uses median instead of mean)
fn l1_robust_detection() {
    println!("--- 2. L1: Robust to Outliers ---\n");

    println!("L1 cost uses median instead of mean, making it robust to outliers.");
    println!("Best for: Data with occasional extreme values.\n");

    // Level shift with outliers
    let mut series: Vec<f64> = vec![10.0; 50];
    series.extend(vec![30.0; 50]);

    // Add outliers
    series[25] = 500.0; // Extreme outlier in first segment
    series[75] = -200.0; // Extreme outlier in second segment

    println!("Data: [10.0 x 50] + [30.0 x 50] with outliers at 25 (500.0) and 75 (-200.0)");
    println!("True changepoint: 50\n");

    // Compare L1 vs L2
    let config_l1 = PeltConfig::default()
        .cost_function(CostFunction::L1)
        .penalty(100.0);
    let config_l2 = PeltConfig::default()
        .cost_function(CostFunction::L2)
        .penalty(100.0);

    let result_l1 = pelt_detect(&series, &config_l1);
    let result_l2 = pelt_detect(&series, &config_l2);

    println!("L1 changepoints: {:?}", result_l1.changepoints);
    println!("L2 changepoints: {:?}", result_l2.changepoints);
    println!("Note: L1 is more robust to the outliers\n");
}

/// Normal Cost: Detects changes in variance (and mean)
fn normal_variance_change() {
    println!("--- 3. Normal: Variance Change Detection ---\n");

    println!("Normal cost detects changes in variance using log-likelihood.");
    println!("Best for: Detecting volatility changes, quality control.\n");

    // Same mean (10.0), but variance changes
    let mut series: Vec<f64> = (0..50)
        .map(|i| 10.0 + 0.5 * (i as f64 * 0.3).sin()) // Low variance
        .collect();
    series.extend(
        (0..50).map(|i| 10.0 + 5.0 * (i as f64 * 0.3).sin()), // High variance (10x)
    );

    println!("Data: Same mean (~10.0), variance increases 10x at index 50");
    println!("True changepoint: 50\n");

    let config = PeltConfig::default()
        .cost_function(CostFunction::Normal)
        .penalty(10.0);
    let result = pelt_detect(&series, &config);

    println!("Config: Normal cost, penalty=10.0");
    println!("Detected changepoints: {:?}", result.changepoints);

    // Calculate segment variances
    let var1: f64 = series[0..50]
        .iter()
        .map(|x| (x - 10.0).powi(2))
        .sum::<f64>()
        / 50.0;
    let var2: f64 = series[50..100]
        .iter()
        .map(|x| (x - 10.0).powi(2))
        .sum::<f64>()
        / 50.0;
    println!(
        "Segment variances: [{:.4}, {:.4}] (ratio: {:.1}x)\n",
        var1,
        var2,
        var2 / var1
    );
}

/// MeanVariance Cost: Detects joint mean AND variance changes
fn mean_variance_joint() {
    println!("--- 4. MeanVariance: Joint Mean+Variance Detection ---\n");

    println!("MeanVariance cost is more sensitive to simultaneous mean and variance changes.");
    println!("Best for: Detecting regime changes where both statistics shift.\n");

    // Both mean and variance change
    let mut series: Vec<f64> = (0..50)
        .map(|i| 0.0 + 0.5 * (i as f64 * 0.2).sin()) // mean=0, low variance
        .collect();
    series.extend(
        (0..50).map(|i| 10.0 + 3.0 * (i as f64 * 0.2).sin()), // mean=10, higher variance
    );

    println!("Data: mean 0->10, variance low->high at index 50");
    println!("True changepoint: 50\n");

    let config_mv = PeltConfig::default()
        .cost_function(CostFunction::MeanVariance)
        .penalty(15.0);
    let config_normal = PeltConfig::default()
        .cost_function(CostFunction::Normal)
        .penalty(15.0);

    let result_mv = pelt_detect(&series, &config_mv);
    let result_normal = pelt_detect(&series, &config_normal);

    println!("MeanVariance changepoints: {:?}", result_mv.changepoints);
    println!("Normal changepoints: {:?}", result_normal.changepoints);
    println!("MeanVariance is often more sensitive to joint changes\n");
}

/// LinearTrend Cost: Detects changes in slope/trend
fn linear_trend_slope() {
    println!("--- 5. LinearTrend: Slope Change Detection ---\n");

    println!("LinearTrend cost fits linear regression and detects slope changes.");
    println!("Best for: Growth rate changes, trend breaks.\n");

    // First half: slope +1, second half: slope -1
    let mut series: Vec<f64> = (0..50).map(|i| i as f64).collect(); // y = x
    series.extend((0..50).map(|i| 50.0 - i as f64)); // y = 50 - x

    println!("Data: slope +1 (y=x) then slope -1 (y=50-x)");
    println!("True changepoint: 50\n");

    let config = PeltConfig::default()
        .cost_function(CostFunction::LinearTrend)
        .penalty(100.0);
    let result = pelt_detect(&series, &config);

    println!("Config: LinearTrend cost, penalty=100.0");
    println!("Detected changepoints: {:?}", result.changepoints);

    // Compare with L2
    let config_l2 = PeltConfig::default()
        .cost_function(CostFunction::L2)
        .penalty(100.0);
    let result_l2 = pelt_detect(&series, &config_l2);
    println!("L2 changepoints (same data): {:?}", result_l2.changepoints);
    println!("LinearTrend specifically targets slope changes\n");
}

/// Cusum Cost: Detects sustained shifts in mean level
fn cusum_sustained_shift() {
    println!("--- 6. Cusum: Sustained Shift Detection ---\n");

    println!("CUSUM cost detects sustained shifts using cumulative sums.");
    println!("Best for: Process monitoring, quality control, gradual drifts.\n");

    // First half: centered around 0, second half: sustained shift to 5
    let mut series: Vec<f64> = (0..50)
        .map(|i| 0.1 * (i as f64 * 0.5).sin()) // oscillates around 0
        .collect();
    series.extend(
        (0..50).map(|i| 5.0 + 0.1 * (i as f64 * 0.5).sin()), // sustained shift to 5
    );

    println!("Data: oscillates around 0, then sustained shift to 5");
    println!("True changepoint: 50\n");

    let config = PeltConfig::default()
        .cost_function(CostFunction::Cusum)
        .penalty(5.0);
    let result = pelt_detect(&series, &config);

    println!("Config: Cusum cost, penalty=5.0");
    println!("Detected changepoints: {:?}", result.changepoints);
    println!("Segment means: {:?}\n", result.segment_means(&series));
}

/// Periodicity Cost: Detects changes in seasonal patterns
fn periodicity_seasonal() {
    println!("--- 7. Periodicity: Seasonal Pattern Change ---\n");

    println!("Periodicity cost uses FFT to detect changes in seasonal patterns.");
    println!("Best for: Seasonal data, detecting when periodicity changes.\n");

    // First segment: period 8, second segment: period 16
    let mut series: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 8.0).sin()).collect();
    series.extend((0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()));

    println!("Data: sin wave period 8 (64 pts) then period 16 (64 pts)");
    println!("True changepoint: 64\n");

    let config = PeltConfig::default()
        .cost_function(CostFunction::Periodicity)
        .penalty(5.0);
    let result = pelt_detect(&series, &config);

    println!("Config: Periodicity cost, penalty=5.0");
    println!("Detected changepoints: {:?}", result.changepoints);

    // Compare with L2 (which won't see the period change well)
    let config_l2 = PeltConfig::default()
        .cost_function(CostFunction::L2)
        .penalty(5.0);
    let result_l2 = pelt_detect(&series, &config_l2);
    println!("L2 changepoints (same data): {:?}", result_l2.changepoints);
    println!("Periodicity cost specifically targets seasonal pattern changes\n");
}

/// Poisson Cost: For count data with rate changes
fn poisson_rate_change() {
    println!("--- 8. Poisson: Count Data Rate Change ---\n");

    println!("Poisson cost is designed for count data.");
    println!("Best for: Event counts, arrival rates, rare events.\n");

    // Simulated count data: rate 2 then rate 10
    let mut series: Vec<f64> = vec![2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 2.0]; // ~rate 2
    series.extend(vec![2.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0]); // ~rate 2
    series.extend(vec![2.0, 1.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0]); // ~rate 2
    series.extend(vec![10.0, 8.0, 12.0, 9.0, 11.0, 10.0, 8.0, 12.0, 9.0, 11.0]); // ~rate 10
    series.extend(vec![9.0, 11.0, 10.0, 8.0, 12.0, 10.0, 9.0, 11.0, 10.0, 8.0]); // ~rate 10

    println!("Data: count data, rate ~2 (30 pts) then rate ~10 (20 pts)");
    println!("True changepoint: 30\n");

    let config = PeltConfig::default()
        .cost_function(CostFunction::Poisson)
        .penalty(5.0);
    let result = pelt_detect(&series, &config);

    println!("Config: Poisson cost, penalty=5.0");
    println!("Detected changepoints: {:?}", result.changepoints);
    println!(
        "Segment means (rates): {:?}\n",
        result.segment_means(&series)
    );
}

/// Compare different cost functions on the same data
fn comparative_analysis() {
    println!("--- 9. Comparative Analysis ---\n");

    println!("Same data analyzed with different cost functions.\n");

    // Data with both mean shift and variance change
    let mut series: Vec<f64> = (0..50)
        .map(|i| 10.0 + 1.0 * (i as f64 * 0.3).sin())
        .collect();
    series.extend((0..50).map(|i| 30.0 + 5.0 * (i as f64 * 0.3).sin())); // mean and variance change

    println!("Data: mean 10->30, variance 1->5 at index 50\n");

    let cost_fns = [
        ("L1", CostFunction::L1),
        ("L2", CostFunction::L2),
        ("Normal", CostFunction::Normal),
        ("MeanVariance", CostFunction::MeanVariance),
        ("LinearTrend", CostFunction::LinearTrend),
        ("Cusum", CostFunction::Cusum),
    ];

    println!("{:<15} {:>20}", "Cost Function", "Changepoints");
    println!("{:-<37}", "");

    for (name, cost_fn) in cost_fns {
        let config = PeltConfig::default().cost_function(cost_fn).penalty(15.0);
        let result = pelt_detect(&series, &config);
        println!("{:<15} {:>20?}", name, result.changepoints);
    }
    println!();
}

/// Print guidance on when to use each cost function
fn print_guidance() {
    println!("--- 10. Cost Function Selection Guide ---\n");

    println!(
        "
Cost Function Selection Guide:
==============================

L2 (default)
  - Detects: Mean changes
  - Use when: General-purpose, no specific requirements
  - Sensitive to: Outliers

L1
  - Detects: Mean changes (using median)
  - Use when: Data has outliers or heavy tails
  - Robust to: Extreme values

Normal
  - Detects: Mean and/or variance changes
  - Use when: Detecting volatility shifts, heteroscedasticity
  - Based on: Gaussian log-likelihood

MeanVariance
  - Detects: Joint mean AND variance changes
  - Use when: Regime changes affect both statistics
  - More sensitive: To combined shifts than Normal

LinearTrend
  - Detects: Slope/trend changes
  - Use when: Looking for growth rate changes, trend breaks
  - Based on: Linear regression residuals

Cusum
  - Detects: Sustained mean shifts
  - Use when: Process monitoring, quality control
  - Good for: Gradual drifts, control charts

Periodicity
  - Detects: Seasonal pattern changes
  - Use when: Time series has seasonality that may change
  - Based on: FFT periodogram

Poisson
  - Detects: Rate changes in count data
  - Use when: Data are counts (events, arrivals)
  - Assumes: Poisson distribution
"
    );
}
