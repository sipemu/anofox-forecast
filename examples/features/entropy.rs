//! Entropy Features example.
//!
//! Run with: cargo run --example entropy

use anofox_forecast::features::entropy;

fn main() {
    println!("=== Entropy Features Example ===\n");

    println!("Entropy measures the complexity, randomness, and");
    println!("predictability of a time series.\n");

    let n = 200;

    // 1. Generate different series types
    // Regular/predictable series
    let regular: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

    // Complex/irregular series
    let complex: Vec<f64> = (0..n)
        .map(|i| {
            (i as f64 * 0.1).sin()
                + 0.5 * (i as f64 * 0.37).cos()
                + 0.3 * (i as f64 * 0.73).sin()
                + 0.2 * (i as f64 * 0.11).cos()
        })
        .collect();

    // Pseudo-random series
    let pseudo_random: Vec<f64> = (0..n)
        .map(|i| ((i as f64 * 137.5).sin() * 10000.0) % 1.0)
        .collect();

    // Step function
    let step: Vec<f64> = (0..n)
        .map(|i| if (i / 20) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    // Constant
    let constant = vec![5.0; n];

    let series_list: Vec<(&str, &[f64])> = vec![
        ("Regular Sine", &regular),
        ("Complex", &complex),
        ("Pseudo-random", &pseudo_random),
        ("Step Function", &step),
        ("Constant", &constant),
    ];

    // 2. Approximate Entropy
    println!("--- Approximate Entropy (ApEn) ---\n");

    println!("Measures unpredictability. Higher = more complex.\n");

    println!(
        "{:<18} {:>12} {:>12}",
        "Series", "ApEn(2, 0.2)", "ApEn(2, 0.5)"
    );
    println!("{:-<44}", "");

    for (name, series) in &series_list {
        let apen_02 = entropy::approximate_entropy(series, 2, 0.2);
        let apen_05 = entropy::approximate_entropy(series, 2, 0.5);
        println!("{:<18} {:>12.4} {:>12.4}", name, apen_02, apen_05);
    }

    // 3. Sample Entropy
    println!("\n--- Sample Entropy (SampEn) ---\n");

    println!("Similar to ApEn but more consistent for short series.\n");

    println!(
        "{:<18} {:>12} {:>12}",
        "Series", "SampEn(2,0.2)", "SampEn(2,0.5)"
    );
    println!("{:-<44}", "");

    for (name, series) in &series_list {
        let sampen_02 = entropy::sample_entropy(series, 2, 0.2);
        let sampen_05 = entropy::sample_entropy(series, 2, 0.5);
        println!("{:<18} {:>12.4} {:>12.4}", name, sampen_02, sampen_05);
    }

    // 4. Permutation Entropy
    println!("\n--- Permutation Entropy ---\n");

    println!("Based on ordinal patterns. Robust to noise.\n");

    println!("{:<18} {:>12} {:>12}", "Series", "PermEn(3)", "PermEn(5)");
    println!("{:-<44}", "");

    for (name, series) in &series_list {
        let permen_3 = entropy::permutation_entropy(series, 3, 1);
        let permen_5 = entropy::permutation_entropy(series, 5, 1);
        println!("{:<18} {:>12.4} {:>12.4}", name, permen_3, permen_5);
    }

    // 5. Effect of Embedding Dimension on Permutation Entropy
    println!("\n--- Effect of Embedding Dimension ---\n");

    println!("Permutation entropy for complex series:\n");
    println!("{:<6} {:>12}", "Dim", "PermEn");
    println!("{:-<20}", "");

    for dim in 3..=7 {
        let pe = entropy::permutation_entropy(&complex, dim, 1);
        println!("{:<6} {:>12.4}", dim, pe);
    }

    // 6. Binned Entropy
    println!("\n--- Binned Entropy ---\n");

    println!("Entropy of the histogram distribution.\n");

    println!(
        "{:<18} {:>10} {:>10} {:>10}",
        "Series", "Bins=5", "Bins=10", "Bins=20"
    );
    println!("{:-<50}", "");

    for (name, series) in &series_list {
        let be_5 = entropy::binned_entropy(series, 5);
        let be_10 = entropy::binned_entropy(series, 10);
        let be_20 = entropy::binned_entropy(series, 20);
        println!(
            "{:<18} {:>10.4} {:>10.4} {:>10.4}",
            name, be_5, be_10, be_20
        );
    }

    // 7. Fourier Entropy
    println!("\n--- Fourier Entropy ---\n");

    println!("Entropy of the spectral power distribution.\n");

    println!("{:<18} {:>15}", "Series", "Fourier Entropy");
    println!("{:-<35}", "");

    for (name, series) in &series_list {
        let fe = entropy::fourier_entropy(series);
        println!("{:<18} {:>15.4}", name, fe);
    }

    // 8. Parameter Sensitivity
    println!("\n--- Parameter Sensitivity (Approximate Entropy) ---\n");

    println!("Effect of tolerance parameter r:\n");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10}",
        "Series", "r=0.1", "r=0.2", "r=0.3", "r=0.5"
    );
    println!("{:-<60}", "");

    for (name, series) in &series_list {
        let apen_01 = entropy::approximate_entropy(series, 2, 0.1);
        let apen_02 = entropy::approximate_entropy(series, 2, 0.2);
        let apen_03 = entropy::approximate_entropy(series, 2, 0.3);
        let apen_05 = entropy::approximate_entropy(series, 2, 0.5);
        println!(
            "{:<18} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name, apen_01, apen_02, apen_03, apen_05
        );
    }

    // 9. Comparing Series Complexity
    println!("\n--- Complexity Ranking ---\n");

    println!("Ranking series by complexity (Sample Entropy):\n");

    let mut rankings: Vec<(&str, f64)> = series_list
        .iter()
        .map(|(name, series)| (*name, entropy::sample_entropy(series, 2, 0.2)))
        .collect();

    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("{:<4} {:<18} {:>12}", "Rank", "Series", "SampEn");
    println!("{:-<36}", "");
    for (i, (name, sampen)) in rankings.iter().enumerate() {
        println!("{:<4} {:<18} {:>12.4}", i + 1, name, sampen);
    }

    // 10. Practical Applications
    println!("\n--- Practical Applications ---\n");
    println!(
        "
Approximate/Sample Entropy:
  - Physiological signal analysis (EEG, ECG)
  - Financial market complexity
  - Manufacturing process monitoring
  - Anomaly detection (sudden complexity changes)

Permutation Entropy:
  - Robust to noise and outliers
  - Good for comparing different time scales
  - Useful for short time series

Binned Entropy:
  - Quick complexity estimate
  - Distribution analysis
  - Comparing value ranges

Fourier Entropy:
  - Spectral complexity
  - Detecting hidden periodicities
  - Signal classification
"
    );

    // 11. Entropy Parameter Guidelines
    println!("--- Parameter Selection Guidelines ---\n");
    println!(
        "
Approximate/Sample Entropy:
  - m (embedding dimension): typically 2
  - r (tolerance): 0.1-0.25 × std_dev of series
  - Larger r → lower entropy (more matches)
  - Larger m → higher entropy (stricter patterns)

Permutation Entropy:
  - dimension: 3-7 (higher = more patterns possible)
  - delay: 1 for consecutive points
  - Maximum entropy = log(dimension!)

Binned Entropy:
  - bins: 5-20 depending on data size
  - More bins → finer resolution
  - Maximum = log(bins)
"
    );

    println!("=== Entropy Features Example Complete ===");
}
