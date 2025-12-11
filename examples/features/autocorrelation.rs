//! Autocorrelation Features example.
//!
//! Run with: cargo run --example autocorrelation

use anofox_forecast::features::autocorrelation;

fn main() {
    println!("=== Autocorrelation Features Example ===\n");

    println!("Autocorrelation measures how a series correlates with");
    println!("lagged versions of itself.\n");

    // 1. Generate different series types
    let n = 100;

    // Random-like series
    let random: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.7).sin() * (i as f64 * 0.3).cos())
        .collect();

    // AR(1) process simulation
    let mut ar1 = vec![0.0];
    for i in 1..n {
        ar1.push(0.8 * ar1[i - 1] + 0.5 * ((i as f64 * 0.2).sin()));
    }

    // Seasonal series
    let seasonal: Vec<f64> = (0..n)
        .map(|i| 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
        .collect();

    // Trending series
    let trending: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // 2. ACF at Various Lags
    println!("--- Autocorrelation Function (ACF) ---\n");

    println!("ACF measures correlation between y_t and y_{{t-k}}\n");

    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Series", "Lag 1", "Lag 2", "Lag 5", "Lag 10", "Lag 12"
    );
    println!("{:-<57}", "");

    let series_list: Vec<(&str, &[f64])> = vec![
        ("Random-like", &random),
        ("AR(1) phi=0.8", &ar1),
        ("Seasonal (12)", &seasonal),
        ("Trending", &trending),
    ];

    for (name, series) in &series_list {
        let acf1 = autocorrelation::autocorrelation(series, 1);
        let acf2 = autocorrelation::autocorrelation(series, 2);
        let acf5 = autocorrelation::autocorrelation(series, 5);
        let acf10 = autocorrelation::autocorrelation(series, 10);
        let acf12 = autocorrelation::autocorrelation(series, 12);

        println!(
            "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            name, acf1, acf2, acf5, acf10, acf12
        );
    }

    // 3. Full ACF Plot
    println!("\n--- ACF Plot (Seasonal Series) ---\n");

    println!("Lag    ACF      Bar");
    println!("{:-<40}", "");

    for lag in 1..=24 {
        let acf = autocorrelation::autocorrelation(&seasonal, lag);
        let bar_len = ((acf.abs() * 20.0) as usize).min(20);
        let bar = if acf >= 0.0 {
            format!("{:>20}{}", "", "|".repeat(bar_len))
        } else {
            format!("{}{:>20}", "|".repeat(bar_len), "")
        };
        println!("{:>3}  {:>7.4}  {}", lag, acf, bar);
    }

    // 4. PACF
    println!("\n--- Partial Autocorrelation Function (PACF) ---\n");

    println!("PACF measures direct correlation at lag k,");
    println!("removing effects of intermediate lags.\n");

    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8}",
        "Series", "Lag 1", "Lag 2", "Lag 3", "Lag 4"
    );
    println!("{:-<47}", "");

    for (name, series) in &series_list {
        let pacf1 = autocorrelation::partial_autocorrelation(series, 1);
        let pacf2 = autocorrelation::partial_autocorrelation(series, 2);
        let pacf3 = autocorrelation::partial_autocorrelation(series, 3);
        let pacf4 = autocorrelation::partial_autocorrelation(series, 4);

        println!(
            "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            name, pacf1, pacf2, pacf3, pacf4
        );
    }

    // 5. AR(1) Process Analysis
    println!("\n--- AR(1) Process Analysis ---\n");

    println!("For AR(1): y_t = phi * y_{{t-1}} + e_t");
    println!("ACF(k) = phi^k, PACF only non-zero at lag 1\n");

    println!("Simulated AR(1) with phi ≈ 0.8:");
    println!(
        "  ACF(1) ≈ phi:     {:.4}",
        autocorrelation::autocorrelation(&ar1, 1)
    );
    println!(
        "  ACF(2) ≈ phi²:    {:.4} (expected: {:.4})",
        autocorrelation::autocorrelation(&ar1, 2),
        0.64
    );
    println!(
        "  ACF(5) ≈ phi⁵:    {:.4} (expected: {:.4})",
        autocorrelation::autocorrelation(&ar1, 5),
        0.8_f64.powi(5)
    );
    println!(
        "\n  PACF(1):          {:.4}",
        autocorrelation::partial_autocorrelation(&ar1, 1)
    );
    println!(
        "  PACF(2):          {:.4} (should be ≈ 0)",
        autocorrelation::partial_autocorrelation(&ar1, 2)
    );

    // 6. Aggregated Autocorrelation
    println!("\n--- Aggregated Autocorrelation ---\n");

    println!("Summary statistics over multiple lags:\n");

    // Mean
    let agg_mean = autocorrelation::agg_autocorrelation(&seasonal, 12, "mean");
    // Variance
    let agg_var = autocorrelation::agg_autocorrelation(&seasonal, 12, "var");
    // Median
    let agg_median = autocorrelation::agg_autocorrelation(&seasonal, 12, "median");

    println!("Seasonal series (lags 1-12):");
    println!("  Mean ACF:   {:.4}", agg_mean);
    println!("  Var ACF:    {:.4}", agg_var);
    println!("  Median ACF: {:.4}", agg_median);

    let agg_mean_ar = autocorrelation::agg_autocorrelation(&ar1, 12, "mean");
    let agg_var_ar = autocorrelation::agg_autocorrelation(&ar1, 12, "var");

    println!("\nAR(1) series (lags 1-12):");
    println!("  Mean ACF:   {:.4}", agg_mean_ar);
    println!("  Var ACF:    {:.4}", agg_var_ar);

    // 7. Time Reversal Asymmetry
    println!("\n--- Time Reversal Asymmetry ---\n");

    println!("Measures non-linearity in the series.\n");

    for lag in [1, 2, 3, 5] {
        let tra_random = autocorrelation::time_reversal_asymmetry_statistic(&random, lag);
        let tra_seasonal = autocorrelation::time_reversal_asymmetry_statistic(&seasonal, lag);
        let tra_ar1 = autocorrelation::time_reversal_asymmetry_statistic(&ar1, lag);

        println!(
            "Lag {}: Random={:.4}, Seasonal={:.4}, AR(1)={:.4}",
            lag, tra_random, tra_seasonal, tra_ar1
        );
    }

    // 8. Detecting Seasonality with ACF
    println!("\n--- Detecting Seasonality with ACF ---\n");

    println!("Seasonal period detection: look for ACF peaks\n");

    // Find peaks in ACF
    let max_lag = 30;
    let acf_values: Vec<f64> = (1..=max_lag)
        .map(|lag| autocorrelation::autocorrelation(&seasonal, lag))
        .collect();

    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 1..acf_values.len() - 1 {
        if acf_values[i] > acf_values[i - 1]
            && acf_values[i] > acf_values[i + 1]
            && acf_values[i] > 0.3
        {
            peaks.push((i + 1, acf_values[i])); // +1 because lag starts at 1
        }
    }

    println!("ACF peaks for seasonal series:");
    for (lag, acf) in peaks.iter().take(5) {
        println!("  Lag {}: ACF = {:.4}", lag, acf);
    }
    if let Some((first_peak_lag, _)) = peaks.first() {
        println!("\nDetected period: {} (true period: 12)", first_peak_lag);
    }

    // 9. Model Identification
    println!("\n--- ARIMA Model Identification ---\n");
    println!(
        "
ACF and PACF patterns help identify ARIMA models:

AR(p) Process:
  - ACF: Decays exponentially
  - PACF: Cuts off after lag p

MA(q) Process:
  - ACF: Cuts off after lag q
  - PACF: Decays exponentially

ARMA(p,q) Process:
  - ACF: Decays exponentially
  - PACF: Decays exponentially

Seasonal Patterns:
  - ACF: Peaks at seasonal lags (12, 24, 36...)
  - PACF: Peaks at seasonal lags
"
    );

    // 10. Stationarity Indicator
    println!("--- ACF as Stationarity Indicator ---\n");

    println!("Slowly decaying ACF suggests non-stationarity:\n");

    let mut acf_sum_random = 0.0;
    let mut acf_sum_trending = 0.0;
    let mut acf_sum_ar1 = 0.0;

    for lag in 1..=10 {
        acf_sum_random += autocorrelation::autocorrelation(&random, lag).abs();
        acf_sum_trending += autocorrelation::autocorrelation(&trending, lag).abs();
        acf_sum_ar1 += autocorrelation::autocorrelation(&ar1, lag).abs();
    }

    println!("Sum of |ACF| for lags 1-10:");
    println!("  Random-like: {:.4} (likely stationary)", acf_sum_random);
    println!("  Trending:    {:.4} (non-stationary)", acf_sum_trending);
    println!("  AR(1):       {:.4} (borderline)", acf_sum_ar1);

    println!("\n=== Autocorrelation Features Example Complete ===");
}
