//! Outlier Detection example.
//!
//! Run with: cargo run --example outlier_detection

use anofox_forecast::detection::{detect_outliers, detect_outliers_auto, OutlierConfig};

fn main() {
    println!("=== Outlier Detection Example ===\n");

    println!("Outlier detection identifies anomalous values in time series");
    println!("that deviate significantly from the expected pattern.\n");

    // Generate sample data with outliers
    let n = 100;
    let mut series: Vec<f64> = (0..n)
        .map(|i| 50.0 + 5.0 * (i as f64 * 0.2).sin() + 0.5 * (i as f64 * 0.1).cos())
        .collect();

    // Add known outliers
    let outlier_indices = vec![15, 42, 67, 89];
    let outlier_values = vec![150.0, -50.0, 200.0, -100.0];

    for (&idx, &val) in outlier_indices.iter().zip(outlier_values.iter()) {
        series[idx] = val;
    }

    println!(
        "Generated: {} observations with {} outliers",
        n,
        outlier_indices.len()
    );
    println!("Outlier positions: {:?}", outlier_indices);
    println!("Outlier values: {:?}\n", outlier_values);

    // Calculate basic statistics
    let mean = series.iter().sum::<f64>() / n as f64;
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    println!("Series Statistics:");
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

    // 1. IQR Method (Default)
    println!("\n--- IQR Method (Interquartile Range) ---");

    let result_iqr = detect_outliers_auto(&series);

    println!("Method: {:?}", result_iqr.method);
    println!("Threshold (multiplier): {:.2}", result_iqr.threshold);
    println!("\nDetected outliers: {:?}", result_iqr.outlier_indices);
    println!("Outlier count: {}", result_iqr.outlier_count());
    println!(
        "Outlier percentage: {:.2}%",
        result_iqr.outlier_percentage()
    );

    // Check detection accuracy
    let correctly_detected: Vec<usize> = outlier_indices
        .iter()
        .filter(|&&idx| result_iqr.is_outlier(idx))
        .cloned()
        .collect();
    println!("\nTrue outliers detected: {:?}", correctly_detected);

    // 2. IQR with Different Multipliers
    println!("\n--- IQR with Different Multipliers ---");
    println!(
        "{:<12} {:>15} {:>20}",
        "Multiplier", "Outlier Count", "Detected Positions"
    );
    println!("{:-<50}", "");

    for mult in [1.0, 1.5, 2.0, 3.0] {
        let config = OutlierConfig::iqr(mult);
        let result = detect_outliers(&series, &config);
        println!(
            "{:<12.1} {:>15} {:>20?}",
            mult,
            result.outlier_count(),
            result.outlier_indices
        );
    }

    // 3. Z-Score Method
    println!("\n--- Z-Score Method ---");

    let result_zscore = detect_outliers(&series, &OutlierConfig::z_score(3.0));

    println!("Method: {:?}", result_zscore.method);
    println!("Threshold: {:.2}", result_zscore.threshold);
    println!("\nDetected outliers: {:?}", result_zscore.outlier_indices);
    println!("Outlier count: {}", result_zscore.outlier_count());

    // Show scores for known outliers
    println!("\nZ-scores for known outlier positions:");
    for &idx in &outlier_indices {
        println!(
            "  Position {}: Z-score = {:.4}",
            idx, result_zscore.scores[idx]
        );
    }

    // 4. Z-Score with Different Thresholds
    println!("\n--- Z-Score with Different Thresholds ---");
    println!(
        "{:<12} {:>15} {:>20}",
        "Threshold", "Outlier Count", "Detected Positions"
    );
    println!("{:-<50}", "");

    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0] {
        let config = OutlierConfig::z_score(threshold);
        let result = detect_outliers(&series, &config);
        println!(
            "{:<12.1} {:>15} {:>20?}",
            threshold,
            result.outlier_count(),
            result.outlier_indices
        );
    }

    // 5. Modified Z-Score (MAD-based)
    println!("\n--- Modified Z-Score (MAD-based) ---");

    let result_modified = detect_outliers(&series, &OutlierConfig::modified_z_score(3.5));

    println!("Method: {:?}", result_modified.method);
    println!("Threshold: {:.2}", result_modified.threshold);
    println!("\nDetected outliers: {:?}", result_modified.outlier_indices);
    println!("Outlier count: {}", result_modified.outlier_count());

    println!("\nModified Z-scores for known outlier positions:");
    for &idx in &outlier_indices {
        println!(
            "  Position {}: Modified Z = {:.4}",
            idx, result_modified.scores[idx]
        );
    }

    // 6. Method Comparison
    println!("\n--- Method Comparison ---");
    println!(
        "{:<20} {:>15} {:>20}",
        "Method", "Outlier Count", "Positions"
    );
    println!("{:-<57}", "");

    let methods: Vec<(&str, OutlierConfig)> = vec![
        ("IQR (1.5)", OutlierConfig::iqr(1.5)),
        ("IQR (3.0)", OutlierConfig::iqr(3.0)),
        ("Z-Score (2.5)", OutlierConfig::z_score(2.5)),
        ("Z-Score (3.0)", OutlierConfig::z_score(3.0)),
        ("Modified Z (3.5)", OutlierConfig::modified_z_score(3.5)),
    ];

    for (name, config) in methods {
        let result = detect_outliers(&series, &config);
        println!(
            "{:<20} {:>15} {:>20?}",
            name,
            result.outlier_count(),
            result.outlier_indices
        );
    }

    // 7. Outlier Scores Distribution
    println!("\n--- Outlier Score Analysis ---");

    let result = detect_outliers(&series, &OutlierConfig::z_score(3.0));

    let non_outlier_scores: Vec<f64> = result
        .scores
        .iter()
        .enumerate()
        .filter(|(i, _)| !result.is_outlier(*i))
        .map(|(_, &s)| s)
        .collect();

    let outlier_scores: Vec<f64> = result
        .outlier_indices
        .iter()
        .map(|&i| result.scores[i])
        .collect();

    if !non_outlier_scores.is_empty() {
        let non_outlier_max = non_outlier_scores.iter().cloned().fold(0.0, f64::max);
        let non_outlier_mean =
            non_outlier_scores.iter().sum::<f64>() / non_outlier_scores.len() as f64;

        println!("Non-outlier scores:");
        println!("  Mean: {:.4}", non_outlier_mean);
        println!("  Max: {:.4}", non_outlier_max);
    }

    if !outlier_scores.is_empty() {
        let outlier_min = outlier_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let outlier_mean = outlier_scores.iter().sum::<f64>() / outlier_scores.len() as f64;

        println!("\nOutlier scores:");
        println!("  Mean: {:.4}", outlier_mean);
        println!("  Min: {:.4}", outlier_min);
    }

    // 8. Handling Edge Cases
    println!("\n--- Edge Cases ---");

    // Empty series
    let empty: Vec<f64> = vec![];
    let result_empty = detect_outliers_auto(&empty);
    println!("Empty series: {} outliers", result_empty.outlier_count());

    // Constant series
    let constant = vec![5.0; 100];
    let result_constant = detect_outliers_auto(&constant);
    println!(
        "Constant series: {} outliers",
        result_constant.outlier_count()
    );

    // Series with single outlier
    let mut single_outlier = vec![10.0; 100];
    single_outlier[50] = 1000.0;
    let result_single = detect_outliers_auto(&single_outlier);
    println!(
        "Single extreme outlier: {} outliers at {:?}",
        result_single.outlier_count(),
        result_single.outlier_indices
    );

    // 9. Practical Guidance
    println!("\n--- Method Selection Guide ---");
    println!(
        "
IQR Method:
  - Non-parametric, works for any distribution
  - Robust to outliers in calculation
  - Use multiplier: 1.5 (mild), 3.0 (extreme)
  - Best for: skewed data, non-normal distributions

Z-Score Method:
  - Assumes normal distribution
  - Sensitive to existing outliers (they affect mean/std)
  - Use threshold: 2.5-3.0 (common), 3.5+ (strict)
  - Best for: normally distributed data

Modified Z-Score (MAD):
  - Robust version of Z-score
  - Uses median instead of mean
  - More resistant to masking effect
  - Best for: when outliers might distort standard statistics

General Tips:
  - Compare multiple methods for confidence
  - Visualize data to verify detected outliers
  - Consider domain knowledge when setting thresholds
  - False positives are often preferable to false negatives
"
    );

    println!("=== Outlier Detection Example Complete ===");
}
