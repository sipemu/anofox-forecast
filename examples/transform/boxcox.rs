//! Box-Cox Transformation example.
//!
//! Run with: cargo run --example boxcox

use anofox_forecast::transform::{
    boxcox, boxcox_auto, boxcox_lambda, boxcox_shifted, inv_boxcox, is_boxcox_suitable,
};

fn main() {
    println!("=== Box-Cox Transformation Example ===\n");

    println!("Box-Cox transforms data to approximate normality.");
    println!("Formula: y = (x^lambda - 1) / lambda  (lambda != 0)");
    println!("         y = ln(x)                    (lambda = 0)\n");

    // Generate skewed data (positive values only)
    let skewed_right: Vec<f64> = (1..=50).map(|i| (i as f64).powi(2) / 10.0).collect();

    println!("Original skewed series (first 10):");
    println!("{:?}\n", &skewed_right[..10]);

    // 1. Check Suitability
    println!("--- Checking Box-Cox Suitability ---");

    let is_suitable = is_boxcox_suitable(&skewed_right);
    println!("Is suitable for Box-Cox: {}", is_suitable);

    // Check a series with non-positive values
    let with_negatives = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
    println!(
        "Series with negatives suitable: {}",
        is_boxcox_suitable(&with_negatives)
    );

    // 2. Automatic Lambda Selection
    println!("\n--- Automatic Lambda Selection ---");

    let result = boxcox_auto(&skewed_right);

    println!("Optimal lambda: {:.4}", result.lambda);
    println!("\nTransformed series (first 10):");
    println!(
        "{:?}",
        &result
            .data
            .iter()
            .take(10)
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // 3. Different Lambda Values
    println!("\n--- Effect of Different Lambda Values ---");

    println!("\nCommon lambda values and their effects:");
    println!("  lambda = -1: Reciprocal transform (1/x)");
    println!("  lambda = -0.5: Reciprocal square root (1/sqrt(x))");
    println!("  lambda = 0: Log transform (ln(x))");
    println!("  lambda = 0.5: Square root transform");
    println!("  lambda = 1: No transformation (linear)");
    println!("  lambda = 2: Square transform");

    println!("\nTransformation comparison for value 16.0:");
    let test_value = vec![16.0];
    for lambda in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
        let transformed = boxcox(&test_value, lambda);
        println!("  lambda = {:>5.1}: {:.4}", lambda, transformed[0]);
    }

    // 4. Inverse Transform
    println!("\n--- Inverse Transformation ---");

    let original = vec![1.0, 4.0, 9.0, 16.0, 25.0];
    println!("Original: {:?}", original);

    let transformed = boxcox_auto(&original);
    println!("Lambda: {:.4}", transformed.lambda);
    println!(
        "Transformed: {:?}",
        transformed
            .data
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    let restored = inv_boxcox(&transformed.data, transformed.lambda);
    println!(
        "Restored: {:?}",
        restored
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    let max_error: f64 = original
        .iter()
        .zip(restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("Max inverse error: {:.2e}", max_error);

    // 5. Box-Cox with Shift
    println!("\n--- Box-Cox with Shift (for non-positive values) ---");

    let with_zeros = vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
    println!("Series with zero: {:?}", with_zeros);
    println!(
        "Direct Box-Cox suitable: {}",
        is_boxcox_suitable(&with_zeros)
    );

    let shifted_result = boxcox_shifted(&with_zeros, 0.5); // Using lambda=0.5 (sqrt)
    println!("\nWith automatic shift (lambda=0.5):");
    println!("  Lambda: {:.4}", shifted_result.lambda);
    println!(
        "  Transformed: {:?}",
        shifted_result
            .data
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // 6. Normality Improvement
    println!("\n--- Normality Improvement ---");

    fn calculate_skewness(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = values.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        if m2 < 1e-10 {
            0.0
        } else {
            m3 / m2.powf(1.5)
        }
    }

    println!("Skewness comparison:\n");

    // Create different skewed distributions
    let data_sets: Vec<(&str, Vec<f64>)> = vec![
        (
            "Right-skewed",
            (1..=100).map(|i| (i as f64).powi(2)).collect(),
        ),
        (
            "Exponential",
            (1..=100).map(|i| (i as f64 * 0.1).exp()).collect(),
        ),
        (
            "Log-normal-like",
            (1..=100).map(|i| (1.0 + i as f64 * 0.05).exp()).collect(),
        ),
    ];

    println!(
        "{:<18} {:>12} {:>12} {:>12}",
        "Data Type", "Original", "Transformed", "Lambda"
    );
    println!("{:-<56}", "");

    for (name, data) in data_sets {
        let original_skew = calculate_skewness(&data);
        let transformed = boxcox_auto(&data);
        let transformed_skew = calculate_skewness(&transformed.data);

        println!(
            "{:<18} {:>12.4} {:>12.4} {:>12.4}",
            name, original_skew, transformed_skew, transformed.lambda
        );
    }

    // 7. Lambda Estimation
    println!("\n--- Lambda Estimation Methods ---");

    let test_data: Vec<f64> = (1..=50).map(|i| (i as f64).powf(1.5)).collect();

    let estimated_lambda = boxcox_lambda(&test_data);
    println!("Estimated optimal lambda: {:.4}", estimated_lambda);

    // Compare with different fixed lambdas
    println!("\nVariance of transformed data for different lambdas:");
    for lambda in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
        let transformed = boxcox(&test_data, lambda);
        let mean = transformed.iter().sum::<f64>() / transformed.len() as f64;
        let variance =
            transformed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / transformed.len() as f64;
        let skew = calculate_skewness(&transformed);
        println!(
            "  lambda = {:>5.1}: variance = {:>10.4}, skewness = {:>8.4}",
            lambda, variance, skew
        );
    }

    // 8. Practical Applications
    println!("\n--- Practical Applications ---");
    println!(
        "
When to Use Box-Cox:
  - Right-skewed distributions
  - Non-constant variance (heteroscedasticity)
  - Before regression analysis
  - To stabilize variance in time series

Common Scenarios:
  - Financial data (returns, volumes)
  - Physical measurements (sizes, weights)
  - Count data (with adjustment for zeros)
  - Any positive data with skewness

Important Considerations:
  - Requires strictly positive values
  - Use shift for zero/negative values
  - Inverse transform for interpretable predictions
  - Lambda near 0 suggests log transform
  - Lambda near 1 suggests minimal transformation needed
"
    );

    // 9. Forecasting Workflow
    println!("--- Box-Cox in Forecasting Workflow ---");
    println!(
        "
1. Check if data is suitable (all positive)
2. Apply Box-Cox transformation
3. Fit forecasting model on transformed data
4. Generate forecasts
5. Apply inverse transformation to predictions
6. (Optional) Transform confidence intervals

Note: Confidence intervals should be transformed separately
to maintain proper coverage.
"
    );

    println!("=== Box-Cox Transformation Example Complete ===");
}
