//! Distribution Features example.
//!
//! Run with: cargo run --example distribution

use anofox_forecast::features::{counting, distribution};

fn main() {
    println!("=== Distribution Features Example ===\n");

    // Generate different distributions for comparison
    println!("--- Comparing Different Distributions ---\n");

    // 1. Symmetric distribution (approximately normal)
    let symmetric: Vec<f64> = (0..200)
        .map(|i| {
            let x = (i as f64 - 100.0) / 30.0;
            50.0 + 20.0 * (-x * x / 2.0).exp()
        })
        .collect();

    // 2. Right-skewed distribution
    let right_skewed: Vec<f64> = (0..200)
        .map(|i| {
            let x = i as f64 / 30.0;
            if x > 0.0 {
                10.0 * x * (-x / 2.0).exp()
            } else {
                0.0
            }
        })
        .collect();

    // 3. Left-skewed distribution
    let left_skewed: Vec<f64> = right_skewed.iter().rev().copied().collect();

    // 4. Heavy-tailed distribution
    let heavy_tailed: Vec<f64> = (0..200)
        .map(|i| {
            let x = (i as f64 - 100.0) / 20.0;
            50.0 + 30.0 / (1.0 + x * x) // Cauchy-like
        })
        .collect();

    // 5. Uniform distribution
    let uniform: Vec<f64> = (0..200).map(|i| 20.0 + 60.0 * (i as f64 / 199.0)).collect();

    let distributions: Vec<(&str, &[f64])> = vec![
        ("Symmetric", &symmetric),
        ("Right-skewed", &right_skewed),
        ("Left-skewed", &left_skewed),
        ("Heavy-tailed", &heavy_tailed),
        ("Uniform", &uniform),
    ];

    // Compare skewness and kurtosis
    println!(
        "{:<15} {:>12} {:>12}",
        "Distribution", "Skewness", "Kurtosis"
    );
    println!("{:-<41}", "");

    for (name, data) in &distributions {
        println!(
            "{:<15} {:>12.4} {:>12.4}",
            name,
            distribution::skewness(data),
            distribution::kurtosis(data)
        );
    }

    // 2. Quantiles
    println!("\n--- Quantiles ---");

    let series: Vec<f64> = (0..100)
        .map(|i| 10.0 + 0.5 * i as f64 + 3.0 * (i as f64 * 0.2).sin())
        .collect();

    println!("\nSeries quantiles:");
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        println!("  Q({:.2}): {:.4}", q, distribution::quantile(&series, q));
    }

    // Interquartile range
    let q1 = distribution::quantile(&series, 0.25);
    let q3 = distribution::quantile(&series, 0.75);
    println!("\nIQR (Q3 - Q1): {:.4}", q3 - q1);

    // 3. Variation Coefficient
    println!("\n--- Variation Coefficient ---");
    println!("Measure of relative variability (std_dev / mean)\n");

    for (name, data) in &distributions {
        let cv = distribution::variation_coefficient(data);
        println!("{:<15}: {:.4}", name, cv);
    }

    // 4. Symmetry Analysis
    println!("\n--- Symmetry Analysis ---");

    for (name, data) in &distributions {
        let is_symmetric = distribution::symmetry_looking(data, 0.05);
        println!("{:<15}: symmetric = {}", name, is_symmetric);
    }

    // 5. Large Standard Deviation Check
    println!("\n--- Large Standard Deviation Check ---");
    println!("Checks if std_dev > r × (max - min)\n");

    for r in [0.1, 0.2, 0.25, 0.3] {
        let is_large = distribution::large_standard_deviation(&series, r);
        println!("r = {:.2}: large_std = {}", r, is_large);
    }

    // 6. Variance vs Standard Deviation
    println!("\n--- Variance Larger Than Standard Deviation ---");

    for (name, data) in &distributions {
        let result = distribution::variance_larger_than_standard_deviation(data);
        println!("{:<15}: {}", name, result);
    }

    // 7. Ratio Beyond R Sigma
    println!("\n--- Ratio Beyond R Sigma ---");
    println!("Percentage of values more than r standard deviations from mean\n");

    for r in [1.0, 1.5, 2.0, 2.5, 3.0] {
        let ratio = distribution::ratio_beyond_r_sigma(&series, r);
        println!("r = {:.1}: {:.2}% beyond", r, ratio * 100.0);
    }

    // 8. Counting Features
    println!("\n--- Counting Features ---");

    // Count above/below thresholds
    let threshold = 35.0;
    println!("\nThreshold = {:.1}", threshold);
    println!("Count above: {}", counting::count_above(&series, threshold));
    println!("Count below: {}", counting::count_below(&series, threshold));
    println!("Count above mean: {}", counting::count_above_mean(&series));
    println!("Count below mean: {}", counting::count_below_mean(&series));

    // 9. Range Count
    println!("\n--- Range Count ---");
    println!(
        "Values in range [20, 40]: {}",
        counting::range_count(&series, 20.0, 40.0)
    );
    println!(
        "Values in range [30, 50]: {}",
        counting::range_count(&series, 30.0, 50.0)
    );

    // 10. Crossings
    println!("\n--- Zero/Mean Crossings ---");
    let oscillating: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).sin()).collect();

    println!("Oscillating series:");
    println!(
        "  Crossings at 0: {}",
        counting::number_crossing_m(&oscillating, 0.0)
    );
    println!(
        "  Crossings at 0.5: {}",
        counting::number_crossing_m(&oscillating, 0.5)
    );

    // 11. Strikes
    println!("\n--- Longest Strikes ---");
    println!(
        "Longest consecutive values above mean: {}",
        counting::longest_strike_above_mean(&series)
    );
    println!(
        "Longest consecutive values below mean: {}",
        counting::longest_strike_below_mean(&series)
    );

    // 12. Peaks
    println!("\n--- Peak Detection ---");
    for support in [1, 2, 3, 5] {
        let n_peaks = counting::number_peaks(&series, support);
        println!("Peaks (support={}): {}", support, n_peaks);
    }

    // 13. Location Features
    println!("\n--- Location of Extremes ---");
    println!(
        "First max location (normalized): {:.4}",
        counting::first_location_of_maximum(&series)
    );
    println!(
        "Last max location (normalized):  {:.4}",
        counting::last_location_of_maximum(&series)
    );
    println!(
        "First min location (normalized): {:.4}",
        counting::first_location_of_minimum(&series)
    );
    println!(
        "Last min location (normalized):  {:.4}",
        counting::last_location_of_minimum(&series)
    );

    // 14. Index Mass Quantile
    println!("\n--- Index Mass Quantile ---");
    println!("Index where cumulative sum reaches quantile of total:\n");
    for q in [0.25, 0.5, 0.75] {
        let idx = counting::index_mass_quantile(&series, q);
        println!("  Q({:.2}): {:.4}", q, idx);
    }

    // 15. Duplicates
    println!("\n--- Duplicate Detection ---");
    let with_duplicates = vec![1.0, 2.0, 3.0, 2.0, 4.0, 4.0, 5.0];
    println!("Series with duplicates: {:?}", with_duplicates);
    println!(
        "  Has duplicate: {}",
        counting::has_duplicate(&with_duplicates)
    );
    println!(
        "  Has duplicate max: {}",
        counting::has_duplicate_max(&with_duplicates)
    );
    println!(
        "  Has duplicate min: {}",
        counting::has_duplicate_min(&with_duplicates)
    );

    // 16. Feature Interpretation
    println!("\n--- Feature Interpretation Guide ---");
    println!(
        "
Skewness:
  - = 0: Symmetric distribution
  - > 0: Right-skewed (long right tail)
  - < 0: Left-skewed (long left tail)

Kurtosis:
  - = 0: Normal-like tails (excess kurtosis)
  - > 0: Heavy tails, more outliers
  - < 0: Light tails, fewer outliers

Variation Coefficient:
  - Low (<0.5): Relatively stable
  - High (>1.0): Highly variable relative to mean

Ratio Beyond R Sigma:
  - Higher ratio = more extreme values
  - Compare to normal distribution expectation
    (2 sigma: ~5%, 3 sigma: ~0.3%)
"
    );

    println!("=== Distribution Features Example Complete ===");
}
