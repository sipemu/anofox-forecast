//! Basic Statistical Features example.
//!
//! Run with: cargo run --example basic_features

use anofox_forecast::features::{basic, change};

fn main() {
    println!("=== Basic Statistical Features Example ===\n");

    // Generate sample time series
    let series: Vec<f64> = (0..100)
        .map(|i| {
            10.0 + 0.5 * i as f64 + 3.0 * (i as f64 * 0.2).sin() + 0.5 * (i as f64 * 0.05).cos()
        })
        .collect();

    println!("Generated series: {} observations\n", series.len());

    // 1. Central Tendency
    println!("--- Central Tendency ---");
    println!("Mean:               {:.4}", basic::mean(&series));
    println!("Median:             {:.4}", basic::median(&series));
    println!("Sum:                {:.4}", basic::sum_values(&series));

    // 2. Dispersion
    println!("\n--- Dispersion ---");
    println!("Variance:           {:.4}", basic::variance(&series));
    println!(
        "Standard Deviation: {:.4}",
        basic::standard_deviation(&series)
    );
    println!(
        "Root Mean Square:   {:.4}",
        basic::root_mean_square(&series)
    );

    // 3. Range & Extremes
    println!("\n--- Range & Extremes ---");
    println!("Minimum:            {:.4}", basic::minimum(&series));
    println!("Maximum:            {:.4}", basic::maximum(&series));
    println!(
        "Absolute Maximum:   {:.4}",
        basic::absolute_maximum(&series)
    );
    println!(
        "Range:              {:.4}",
        basic::maximum(&series) - basic::minimum(&series)
    );

    // 4. Energy Features
    println!("\n--- Energy Features ---");
    println!("Absolute Energy:    {:.4}", basic::abs_energy(&series));

    // 5. Change Features
    println!("\n--- Change Features ---");
    println!("Mean Change:        {:.6}", basic::mean_change(&series));
    println!("Mean Abs Change:    {:.4}", basic::mean_abs_change(&series));
    println!(
        "Abs Sum of Changes: {:.4}",
        basic::absolute_sum_of_changes(&series)
    );

    // 6. Derivative Features
    println!("\n--- Derivative Features ---");
    println!(
        "Mean 2nd Derivative: {:.6}",
        basic::mean_second_derivative_central(&series)
    );

    // 7. Length
    println!("\n--- Size ---");
    println!("Length:             {}", basic::length(&series));

    // 8. Mean of N Absolute Max
    println!("\n--- Top Values ---");
    for n in [1, 3, 5, 10] {
        println!(
            "Mean of top {} absolute values: {:.4}",
            n,
            basic::mean_n_absolute_max(&series, n)
        );
    }

    // 9. Change Features (from change module)
    println!("\n--- Advanced Change Features ---");

    // Energy ratio by chunks
    println!("\nEnergy ratio by chunks (10 chunks):");
    for i in 0..10 {
        let ratio = change::energy_ratio_by_chunks(&series, 10, i);
        println!("  Chunk {}: {:.4}", i, ratio);
    }

    // 10. Reoccurrence Analysis
    println!("\n--- Reoccurrence Analysis ---");
    // Round values to check for reoccurrence
    let rounded: Vec<f64> = series.iter().map(|x| (x * 10.0).round() / 10.0).collect();
    println!(
        "Sum of reoccurring values:       {:.4}",
        change::sum_of_reoccurring_values(&rounded)
    );
    println!(
        "Sum of reoccurring data points:  {:.4}",
        change::sum_of_reoccurring_data_points(&rounded)
    );
    println!(
        "% reoccurring values:            {:.4}%",
        100.0 * change::percentage_of_reoccurring_values_to_all_values(&rounded)
    );
    println!(
        "% reoccurring data points:       {:.4}%",
        100.0 * change::percentage_of_reoccurring_datapoints_to_all_datapoints(&rounded)
    );
    println!(
        "Ratio unique values to length:   {:.4}",
        change::ratio_value_number_to_time_series_length(&rounded)
    );

    // 11. Compare Different Series
    println!("\n--- Comparing Different Series Types ---");

    // Constant series
    let constant = vec![5.0; 50];
    // Trending series
    let trending: Vec<f64> = (0..50).map(|i| i as f64).collect();
    // Noisy series
    let noisy: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).sin() * 10.0).collect();
    // Spiky series
    let mut spiky = vec![0.0; 50];
    spiky[10] = 100.0;
    spiky[25] = -50.0;
    spiky[40] = 75.0;

    println!(
        "\n{:<15} {:>12} {:>12} {:>12} {:>12}",
        "Series", "Mean", "Std Dev", "AbsEnergy", "MeanAbsChg"
    );
    println!("{:-<65}", "");

    let series_list: Vec<(&str, &[f64])> = vec![
        ("Constant", &constant),
        ("Trending", &trending),
        ("Noisy", &noisy),
        ("Spiky", &spiky),
    ];

    for (name, s) in series_list {
        println!(
            "{:<15} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            name,
            basic::mean(s),
            basic::standard_deviation(s),
            basic::abs_energy(s),
            basic::mean_abs_change(s)
        );
    }

    // 12. Feature Use Cases
    println!("\n--- Feature Use Cases ---");
    println!(
        "
Mean, Median:
  - Central tendency
  - Median more robust to outliers

Variance, Std Dev:
  - Measure of spread/volatility
  - Useful for risk assessment

Abs Energy:
  - Total signal power
  - Useful for anomaly detection

Mean Abs Change:
  - Measure of series roughness
  - Higher = more volatile

Mean 2nd Derivative:
  - Measure of acceleration
  - Indicates trend changes

Sum of Reoccurring:
  - Pattern repetition
  - Useful for seasonality detection
"
    );

    println!("=== Basic Features Example Complete ===");
}
