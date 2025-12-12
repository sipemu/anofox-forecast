//! Complexity Features example.
//!
//! Run with: cargo run --example complexity

use anofox_forecast::features::complexity;

fn main() {
    println!("=== Complexity Features Example ===\n");

    println!("Complexity features measure the structural complexity");
    println!("and predictability of time series data.\n");

    let n = 200;

    // Generate different series types
    // Simple sine wave
    let sine: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

    // Complex superposition
    let complex: Vec<f64> = (0..n)
        .map(|i| {
            (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.23).cos() + 0.3 * (i as f64 * 0.47).sin()
        })
        .collect();

    // Pseudo-random
    let random: Vec<f64> = (0..n)
        .map(|i| ((i as f64 * 137.5).sin() * 10000.0) % 1.0)
        .collect();

    // Constant
    let constant = vec![5.0; n];

    // Linear trend
    let linear: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

    // Step function
    let step: Vec<f64> = (0..n).map(|i| ((i / 20) % 3) as f64).collect();

    // Square wave
    let square: Vec<f64> = (0..n)
        .map(|i| if (i / 10) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let series_list: Vec<(&str, &[f64])> = vec![
        ("Sine Wave", &sine),
        ("Complex Wave", &complex),
        ("Pseudo-random", &random),
        ("Constant", &constant),
        ("Linear Trend", &linear),
        ("Step Function", &step),
        ("Square Wave", &square),
    ];

    // 1. C3 Statistic
    println!("--- C3 Statistic ---\n");

    println!("Measures non-linearity based on third-order correlations.");
    println!("C3(lag) = E[x(t) × x(t-lag) × x(t-2*lag)]\n");

    println!(
        "{:<18} {:>10} {:>10} {:>10}",
        "Series", "C3(1)", "C3(2)", "C3(3)"
    );
    println!("{:-<50}", "");

    for (name, series) in &series_list {
        let c3_1 = complexity::c3(series, 1);
        let c3_2 = complexity::c3(series, 2);
        let c3_3 = complexity::c3(series, 3);
        println!("{:<18} {:>10.4} {:>10.4} {:>10.4}", name, c3_1, c3_2, c3_3);
    }

    // 2. CID (Complexity-Invariant Distance Estimate)
    println!("\n--- CID_CE (Complexity Estimate) ---\n");

    println!("Measures the complexity as sum of squared differences.");
    println!("Higher value = more complex/irregular series.\n");

    println!(
        "{:<18} {:>15} {:>15}",
        "Series", "CID_CE(norm)", "CID_CE(raw)"
    );
    println!("{:-<50}", "");

    for (name, series) in &series_list {
        let cid_norm = complexity::cid_ce(series, true);
        let cid_raw = complexity::cid_ce(series, false);
        println!("{:<18} {:>15.4} {:>15.4}", name, cid_norm, cid_raw);
    }

    // 3. Lempel-Ziv Complexity
    println!("\n--- Lempel-Ziv Complexity ---\n");

    println!("Measures algorithmic complexity by counting unique patterns.");
    println!("Higher value = more complex/unpredictable.\n");

    println!("{:<18} {:>15}", "Series", "LZ Complexity");
    println!("{:-<35}", "");

    for (name, series) in &series_list {
        let lz = complexity::lempel_ziv_complexity(series, 10);
        println!("{:<18} {:>15.4}", name, lz);
    }

    // 4. Series Length Effect on Lempel-Ziv
    println!("\n--- Series Length Effect on Lempel-Ziv ---\n");

    println!("LZ complexity increases with series length and pattern diversity:\n");

    // Generate different length series
    let short: Vec<f64> = (0..50).map(|i| ((i * 17 + 13) % 97) as f64).collect();
    let medium: Vec<f64> = (0..100).map(|i| ((i * 17 + 13) % 97) as f64).collect();
    let long: Vec<f64> = (0..200).map(|i| ((i * 17 + 13) % 97) as f64).collect();

    println!("{:<15} {:>10} {:>12}", "Length", "LZ", "Normalized");
    println!("{:-<40}", "");
    println!(
        "{:<15} {:>10.4} {:>12.4}",
        "50 points",
        complexity::lempel_ziv_complexity(&short, 10),
        complexity::lempel_ziv_complexity(&short, 10) / 50.0
    );
    println!(
        "{:<15} {:>10.4} {:>12.4}",
        "100 points",
        complexity::lempel_ziv_complexity(&medium, 10),
        complexity::lempel_ziv_complexity(&medium, 10) / 100.0
    );
    println!(
        "{:<15} {:>10.4} {:>12.4}",
        "200 points",
        complexity::lempel_ziv_complexity(&long, 10),
        complexity::lempel_ziv_complexity(&long, 10) / 200.0
    );

    // 5. Complexity Comparison
    println!("\n--- Complexity Feature Comparison ---\n");

    println!("Different measures capture different aspects of complexity:\n");

    println!(
        "{:<18} {:>12} {:>12} {:>12}",
        "Series", "C3(1)", "CID_CE", "LZ"
    );
    println!("{:-<56}", "");

    for (name, series) in &series_list {
        let c3 = complexity::c3(series, 1);
        let cid = complexity::cid_ce(series, true);
        let lz = complexity::lempel_ziv_complexity(series, 10);
        println!("{:<18} {:>12.4} {:>12.4} {:>12.4}", name, c3, cid, lz);
    }

    // 6. Ranking by Complexity
    println!("\n--- Complexity Ranking (by CID_CE) ---\n");

    let mut rankings: Vec<(&str, f64)> = series_list
        .iter()
        .map(|(name, series)| (*name, complexity::cid_ce(series, true)))
        .collect();

    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("{:<4} {:<18} {:>12}", "Rank", "Series", "CID_CE");
    println!("{:-<36}", "");
    for (i, (name, cid)) in rankings.iter().enumerate() {
        println!("{:<4} {:<18} {:>12.4}", i + 1, name, cid);
    }

    // 7. Understanding C3
    println!("\n--- Understanding C3 Statistic ---\n");

    println!("C3 captures non-linear structure:\n");

    // Symmetric series should have C3 ≈ 0
    let symmetric: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

    // Asymmetric series (sawtooth)
    let sawtooth: Vec<f64> = (0..n).map(|i| (i % 20) as f64 / 20.0).collect();

    println!(
        "Symmetric (sine):  C3(1) = {:.6}",
        complexity::c3(&symmetric, 1)
    );
    println!(
        "Asymmetric (saw):  C3(1) = {:.6}",
        complexity::c3(&sawtooth, 1)
    );
    println!("\nNote: C3 ≈ 0 for symmetric distributions");

    // 8. CID_CE Interpretation
    println!("\n--- CID_CE Interpretation ---\n");

    println!("CID_CE = sqrt(sum((x[i+1] - x[i])^2))");
    println!("\nInterpretation:");
    println!("  - Low:  Smooth, slowly changing series");
    println!("  - High: Rough, rapidly changing series\n");

    // Compare smoothness
    let smooth: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
    let rough: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();

    println!(
        "Slow oscillation: CID_CE = {:.4}",
        complexity::cid_ce(&smooth, true)
    );
    println!(
        "Fast oscillation: CID_CE = {:.4}",
        complexity::cid_ce(&rough, true)
    );

    // 9. Practical Applications
    println!("\n--- Practical Applications ---\n");
    println!(
        "
C3 Statistic:
  - Non-linearity detection
  - Distinguishing linear vs non-linear dynamics
  - Chaos detection

CID_CE (Complexity-Invariant Distance):
  - Time series classification
  - Measuring roughness/smoothness
  - Feature for similarity search

Lempel-Ziv Complexity:
  - Algorithmic complexity estimation
  - Data compression difficulty
  - Pattern richness measurement
  - Randomness testing
"
    );

    // 10. Feature Selection Guide
    println!("--- Feature Selection Guide ---\n");
    println!(
        "
Choose C3 when:
  - Testing for non-linear structure
  - Comparing symmetric vs asymmetric patterns
  - Need lag-specific information

Choose CID_CE when:
  - Measuring overall roughness
  - Classification tasks
  - Comparing series variability

Choose Lempel-Ziv when:
  - Estimating predictability
  - Measuring pattern diversity
  - Detecting repetitive vs random structure

Combine all three for:
  - Comprehensive complexity characterization
  - Robust feature engineering
  - Anomaly detection pipelines
"
    );

    println!("=== Complexity Features Example Complete ===");
}
