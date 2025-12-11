//! Scaling and Normalization example.
//!
//! Run with: cargo run --example scaling

use anofox_forecast::transform::{center, normalize, robust_scale, scale_to_range, standardize};

fn main() {
    println!("=== Scaling and Normalization Example ===\n");

    // Generate sample data
    let series: Vec<f64> = vec![
        10.0, 20.0, 30.0, 25.0, 35.0, 40.0, 15.0, 50.0, 45.0, 30.0, 100.0, // outlier
        25.0, 35.0, 30.0, 40.0,
    ];

    println!("Original series:");
    println!("{:?}\n", series);

    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;
    let std_dev = variance.sqrt();

    println!("Statistics:");
    println!("  Mean: {:.4}", mean);
    println!("  Std Dev: {:.4}", std_dev);
    println!(
        "  Min: {:.4}",
        series.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "  Max: {:.4}",
        series.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // 1. Centering (subtract mean)
    println!("\n--- Centering (Subtract Mean) ---");

    let centered = center(&series);

    println!("Formula: x_centered = x - mean");
    println!("\nCentered series:");
    print_rounded(&centered.data, 4);

    let centered_mean = centered.data.iter().sum::<f64>() / centered.data.len() as f64;
    println!("\nCentered mean: {:.6} (should be ≈ 0)", centered_mean);
    println!("Can inverse: subtract mean and add back");

    // 2. Standardization (Z-score)
    println!("\n--- Standardization (Z-score) ---");

    let standardized = standardize(&series);

    println!("Formula: z = (x - mean) / std_dev");
    println!("\nStandardized series:");
    print_rounded(&standardized.data, 4);

    let z_mean = standardized.data.iter().sum::<f64>() / standardized.data.len() as f64;
    let z_var = standardized
        .data
        .iter()
        .map(|z| (z - z_mean).powi(2))
        .sum::<f64>()
        / standardized.data.len() as f64;

    println!("\nStandardized statistics:");
    println!("  Mean: {:.6} (should be ≈ 0)", z_mean);
    println!("  Variance: {:.6} (should be ≈ 1)", z_var);

    // Inverse transform
    let restored = standardized.inverse();
    let max_error: f64 = series
        .iter()
        .zip(restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("\nMax inverse transform error: {:.2e}", max_error);

    // 3. Min-Max Normalization
    println!("\n--- Min-Max Normalization ---");

    let normalized = normalize(&series);

    println!("Formula: x_norm = (x - min) / (max - min)");
    println!("\nNormalized series [0, 1]:");
    print_rounded(&normalized.data, 4);

    let norm_min = normalized
        .data
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let norm_max = normalized
        .data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\nNormalized range:");
    println!("  Min: {:.6} (should be 0)", norm_min);
    println!("  Max: {:.6} (should be 1)", norm_max);

    // 4. Scale to Custom Range
    println!("\n--- Scale to Custom Range ---");

    let scaled_custom = scale_to_range(&series, -1.0, 1.0);

    println!("Scaling to range [-1, 1]:");
    print_rounded(&scaled_custom, 4);

    let custom_min = scaled_custom.iter().cloned().fold(f64::INFINITY, f64::min);
    let custom_max = scaled_custom
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\nScaled range: [{:.6}, {:.6}]", custom_min, custom_max);

    // 5. Robust Scaling
    println!("\n--- Robust Scaling (IQR-based) ---");

    let robust = robust_scale(&series);

    println!("Formula: x_robust = (x - median) / IQR");
    println!("Robust to outliers!\n");

    println!("Robust scaled series:");
    print_rounded(&robust.data, 4);

    // Compare standard vs robust for outlier
    let outlier_idx = 10; // index of 100.0
    println!("\nOutlier handling (value 100.0 at index {}):", outlier_idx);
    println!("  Z-score:      {:.4}", standardized.data[outlier_idx]);
    println!("  Robust scale: {:.4}", robust.data[outlier_idx]);

    // 6. Comparison Table
    println!("\n--- Scaling Method Comparison ---\n");

    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Index", "Original", "Centered", "Z-score", "MinMax", "Robust"
    );
    println!("{:-<68}", "");

    for (i, &value) in series.iter().enumerate() {
        println!(
            "{:>6} {:>10.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            i, value, centered.data[i], standardized.data[i], normalized.data[i], robust.data[i]
        );
    }

    // 7. When to Use Each Method
    println!("\n--- When to Use Each Method ---");
    println!(
        "
Centering:
  - When you only need zero mean
  - Preserves original scale
  - Good for PCA preprocessing

Standardization (Z-score):
  - When features have different units
  - When algorithm assumes normally distributed data
  - Neural networks, SVM, regression
  - Sensitive to outliers

Min-Max Normalization:
  - When you need bounded values [0, 1]
  - When original distribution should be preserved
  - Image processing, neural network inputs
  - Sensitive to outliers

Custom Range Scaling:
  - Tanh activation: scale to [-1, 1]
  - Percentage: scale to [0, 100]
  - Domain-specific requirements

Robust Scaling:
  - When data contains outliers
  - Uses median and IQR (more robust statistics)
  - Outliers have less impact
  - Good for data with long tails
"
    );

    // 8. Effect on Different Distributions
    println!("--- Effect on Different Series Types ---\n");

    // Normal-like
    let normal: Vec<f64> = (0..50)
        .map(|i| {
            let x = (i as f64 - 25.0) / 10.0;
            50.0 + 20.0 * (-x * x / 2.0).exp()
        })
        .collect();

    // Skewed
    let skewed: Vec<f64> = (0..50).map(|i| (1.0 + i as f64 / 10.0).powi(2)).collect();

    // With outliers
    let mut with_outliers = vec![50.0; 50];
    with_outliers[10] = 500.0;
    with_outliers[40] = -100.0;

    let series_types: Vec<(&str, Vec<f64>)> = vec![
        ("Normal-like", normal),
        ("Skewed", skewed),
        ("With Outliers", with_outliers),
    ];

    for (name, data) in series_types {
        let std_scaled = standardize(&data);
        let robust_scaled = robust_scale(&data);

        let std_range = std_scaled
            .data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - std_scaled
                .data
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        let robust_range = robust_scaled
            .data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - robust_scaled
                .data
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

        println!(
            "{:<15}: Z-score range = {:.2}, Robust range = {:.2}",
            name, std_range, robust_range
        );
    }

    println!("\n=== Scaling Example Complete ===");
}

fn print_rounded(values: &[f64], decimals: usize) {
    let factor = 10_f64.powi(decimals as i32);
    let rounded: Vec<String> = values
        .iter()
        .map(|v| format!("{:.prec$}", (v * factor).round() / factor, prec = decimals))
        .collect();
    println!("[{}]", rounded.join(", "));
}
