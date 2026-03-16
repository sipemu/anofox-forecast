//! AID (Automatic Identification of Demand) classifier example.
//!
//! Demonstrates the AidAnalyzer for demand classification, including
//! summary statistics, per-observation anomaly labels, and the builder API
//! for configuring the classifier.
//!
//! Run with: cargo run --example aid

use anofox_forecast::validation::aid::{
    AidAnalyzer, AidAnomalyLabel, AidInformationCriterion,
};

fn main() {
    println!("=== AID Demand Classification Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Regular demand series
    // -----------------------------------------------------------------------
    println!("--- Regular Demand ---");
    let regular: Vec<f64> = (0..100)
        .map(|i| 50.0 + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin() + 0.5 * i as f64)
        .collect();

    let result = AidAnalyzer::new().analyze(&regular);
    let summary = result.summary();

    println!("Demand type:     {:?}", summary.demand_type);
    println!("Distribution:    {:?}", summary.distribution);
    println!("Mean:            {:.2}", summary.mean);
    println!("Variance:        {:.2}", summary.variance);
    println!("Zero proportion: {:.4}", summary.zero_proportion);
    println!("N observations:  {}", summary.n_observations);
    println!("Fractional:      {}", summary.is_fractional);

    if let Some(shape) = summary.shape {
        println!("Shape:           {:.4}", shape);
    }
    if let Some(scale) = summary.scale {
        println!("Scale:           {:.4}", scale);
    }

    // -----------------------------------------------------------------------
    // 2. Intermittent demand series (many zeros)
    // -----------------------------------------------------------------------
    println!("\n--- Intermittent Demand ---");
    let mut intermittent = vec![0.0; 100];
    // Sprinkle in non-zero demand at irregular intervals
    for &i in &[3, 8, 15, 22, 30, 37, 45, 52, 60, 68, 75, 83, 90, 97] {
        intermittent[i] = 5.0 + (i as f64 * 0.7).sin().abs() * 10.0;
    }

    let result = AidAnalyzer::new().analyze(&intermittent);
    let summary = result.summary();

    println!("Demand type:     {:?}", summary.demand_type);
    println!("Distribution:    {:?}", summary.distribution);
    println!("Mean:            {:.2}", summary.mean);
    println!("Zero proportion: {:.4}", summary.zero_proportion);

    if let Some(zp) = summary.zero_prob {
        println!("Fitted zero P:   {:.4}", zp);
    }

    // -----------------------------------------------------------------------
    // 3. Per-observation anomaly labels
    // -----------------------------------------------------------------------
    println!("\n--- Per-Observation Anomaly Labels ---");

    // Create a series with likely outliers
    let mut outlier_data: Vec<f64> = (0..80)
        .map(|i| 20.0 + 2.0 * (i as f64 * 0.5).sin())
        .collect();
    // Inject high outliers
    outlier_data[10] = 100.0;
    outlier_data[40] = 95.0;
    outlier_data[60] = 110.0;

    let result = AidAnalyzer::new().analyze(&outlier_data);
    let features = result.features();

    println!("Total observations: {}", features.labels.len());

    let counts = features.label_counts();
    let n_anomalies: usize = counts
        .iter()
        .filter(|(k, _)| **k != AidAnomalyLabel::Normal)
        .map(|(_, v)| *v)
        .sum();
    println!("Anomalies found:    {}", n_anomalies);

    // Print label breakdown
    println!("\nLabel breakdown:");
    for (label, count) in &counts {
        println!("  {:<18} {}", format!("{:?}:", label), count);
    }

    // Show flagged observations
    let anomaly_idx = features.anomaly_indices();
    if !anomaly_idx.is_empty() {
        println!("\nFlagged observations:");
        for &i in &anomaly_idx {
            println!(
                "  Index {:>3}: value = {:>7.2}, label = {}",
                i, outlier_data[i], features.labels[i]
            );
        }
    }

    // Convenience checks
    println!("\nHas stockouts:      {}", features.has_stockouts());
    println!("Is new product:     {}", features.is_new_product());
    println!("Is obsolete:        {}", features.is_obsolete_product());

    // -----------------------------------------------------------------------
    // 4. Display formatting
    // -----------------------------------------------------------------------
    println!("\n--- Display Formatting ---");
    let result = AidAnalyzer::new().analyze(&regular);
    println!("{}", result.summary());
    println!("{}", result.features());

    // -----------------------------------------------------------------------
    // 5. Builder API configuration
    // -----------------------------------------------------------------------
    println!("--- Builder API ---");

    // Custom anomaly significance level
    let result_strict = AidAnalyzer::new()
        .anomaly_alpha(0.01)
        .analyze(&outlier_data);
    let result_lenient = AidAnalyzer::new()
        .anomaly_alpha(0.20)
        .analyze(&outlier_data);

    let strict_anomalies: usize = result_strict
        .features()
        .label_counts()
        .iter()
        .filter(|(k, _)| **k != AidAnomalyLabel::Normal)
        .map(|(_, v)| *v)
        .sum();
    let lenient_anomalies: usize = result_lenient
        .features()
        .label_counts()
        .iter()
        .filter(|(k, _)| **k != AidAnomalyLabel::Normal)
        .map(|(_, v)| *v)
        .sum();

    println!("Anomaly alpha = 0.01 (strict):  {} anomalies", strict_anomalies);
    println!("Anomaly alpha = 0.20 (lenient): {} anomalies", lenient_anomalies);

    // Custom intermittent threshold
    let data_25pct_zeros: Vec<f64> = (0..100)
        .map(|i| if i % 4 == 0 { 0.0 } else { 10.0 + i as f64 })
        .collect();

    let default_result = AidAnalyzer::new().analyze(&data_25pct_zeros);
    let strict_result = AidAnalyzer::new()
        .intermittent_threshold(0.10)
        .analyze(&data_25pct_zeros);

    println!(
        "\n25% zeros with default threshold: {:?}",
        default_result.summary().demand_type
    );
    println!(
        "25% zeros with threshold = 0.10:  {:?}",
        strict_result.summary().demand_type
    );

    // Disable anomaly detection
    let no_anomalies = AidAnalyzer::new()
        .detect_anomalies(false)
        .analyze(&outlier_data);
    let all_normal = no_anomalies
        .features()
        .labels
        .iter()
        .all(|l| *l == AidAnomalyLabel::Normal);
    println!(
        "\nWith detect_anomalies(false): all Normal = {}",
        all_normal
    );

    // Change information criterion
    let bic_result = AidAnalyzer::new()
        .ic(AidInformationCriterion::BIC)
        .analyze(&regular);
    println!(
        "BIC-selected distribution: {:?}",
        bic_result.summary().distribution
    );

    // -----------------------------------------------------------------------
    // 6. Information criterion values
    // -----------------------------------------------------------------------
    println!("\n--- Information Criterion Scores ---");
    let result = AidAnalyzer::new().analyze(&regular);
    let summary = result.summary();

    let mut ic_pairs: Vec<_> = summary.ic_values.iter().collect();
    ic_pairs.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    println!("Candidate distributions ranked by IC:");
    for (i, (dist, ic)) in ic_pairs.iter().enumerate() {
        let marker = if **dist == summary.distribution {
            " <-- selected"
        } else {
            ""
        };
        println!("  {}. {:?}: {:.2}{}", i + 1, dist, ic, marker);
    }

    // -----------------------------------------------------------------------
    // 7. Edge cases
    // -----------------------------------------------------------------------
    println!("\n--- Edge Cases ---");

    // All zeros
    let all_zeros = vec![0.0; 30];
    let result = AidAnalyzer::new().analyze(&all_zeros);
    println!(
        "All zeros:    type={:?}, zero_prop={:.2}",
        result.summary().demand_type,
        result.summary().zero_proportion
    );

    // Single observation
    let single = vec![42.0];
    let result = AidAnalyzer::new().analyze(&single);
    println!(
        "Single value: type={:?}, n={}",
        result.summary().demand_type,
        result.summary().n_observations
    );

    println!("\n=== AID Demand Classification Example Complete ===");
}
