//! Seasonality Detection example.
//!
//! Run with: cargo run --example seasonality

use anofox_forecast::detection::{
    detect_seasonality, detect_seasonality_auto, seasonal_strength, SeasonalityConfig,
};

fn main() {
    println!("=== Seasonality Detection Example ===\n");

    println!("Seasonality detection identifies repeating patterns in time series");
    println!("using autocorrelation analysis.\n");

    // 1. Clear Seasonal Pattern
    println!("--- Clear Seasonal Pattern ---");

    let period = 12;
    let n = 120;

    let seasonal_series: Vec<f64> = (0..n)
        .map(|i| 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
        .collect();

    println!("Generated: {} observations with period {}", n, period);

    let result = detect_seasonality_auto(&seasonal_series);

    println!("\nDetection Results:");
    println!("  Detected: {}", result.detected);
    println!("  Period: {:?}", result.period);
    println!("  Strength: {:.4}", result.strength);
    println!("  Strong seasonality: {}", result.is_strong());
    println!("  Moderate seasonality: {}", result.is_moderate());

    if !result.candidates.is_empty() {
        println!("\nTop candidate periods:");
        for (i, (p, score)) in result.candidates.iter().take(5).enumerate() {
            println!("  {}. Period {}: ACF = {:.4}", i + 1, p, score);
        }
    }

    // 2. No Seasonality
    println!("\n--- No Seasonality (Random Walk) ---");

    let random_series: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.123).sin() * (i as f64 % 7.0))
        .collect();

    let config = SeasonalityConfig::default().with_threshold(0.5);
    let result_random = detect_seasonality(&random_series, &config);

    println!("Threshold: 0.5 (stricter)");
    println!("Detected: {}", result_random.detected);
    println!("Strength: {:.4}", result_random.strength);

    // 3. Weekly Seasonality
    println!("\n--- Weekly Seasonality (Period 7) ---");

    let weekly_series: Vec<f64> = (0..70)
        .map(|i| {
            let day_of_week = i % 7;
            // Higher on weekends (days 5, 6), lower on weekdays
            if day_of_week >= 5 {
                100.0 + 10.0 * (i as f64 * 0.1).sin()
            } else {
                50.0 + 10.0 * (i as f64 * 0.1).sin()
            }
        })
        .collect();

    let config_weekly = SeasonalityConfig::default()
        .with_min_period(2)
        .with_max_period(14)
        .with_threshold(0.3);

    let result_weekly = detect_seasonality(&weekly_series, &config_weekly);

    println!("Config: min_period=2, max_period=14, threshold=0.3");
    println!("Detected: {}", result_weekly.detected);
    println!("Period: {:?}", result_weekly.period);
    println!("Strength: {:.4}", result_weekly.strength);

    // 4. Multiple Seasonal Patterns
    println!("\n--- Multiple Seasonal Patterns ---");

    // Daily and weekly patterns combined
    let multi_series: Vec<f64> = (0..168) // 1 week of hourly data
        .map(|i| {
            let daily = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin();
            let weekly = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 168.0).sin();
            50.0 + daily + weekly
        })
        .collect();

    println!("168 hourly observations (1 week)");
    println!("Embedded patterns: daily (24h), weekly (168h)");

    // Detect with different max periods
    for max_period in [50, 100, 200] {
        let config = SeasonalityConfig::default()
            .with_min_period(2)
            .with_max_period(max_period)
            .with_threshold(0.2);

        let result = detect_seasonality(&multi_series, &config);
        println!(
            "\n  max_period={}: detected period={:?}, strength={:.4}",
            max_period, result.period, result.strength
        );

        if !result.candidates.is_empty() {
            let top3: Vec<_> = result.candidates.iter().take(3).collect();
            println!("    Top candidates: {:?}", top3);
        }
    }

    // 5. Seasonality with Trend
    println!("\n--- Seasonality with Trend ---");

    let trend_seasonal: Vec<f64> = (0..120)
        .map(|i| {
            let trend = 0.5 * i as f64;
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            trend + seasonal
        })
        .collect();

    let result_trend = detect_seasonality_auto(&trend_seasonal);

    println!("Series with linear trend + seasonality (period 12)");
    println!("Detected: {}", result_trend.detected);
    println!("Period: {:?}", result_trend.period);
    println!("Strength: {:.4}", result_trend.strength);

    // 6. Seasonal Strength from Decomposition
    println!("\n--- Seasonal Strength from Decomposition ---");

    // Simulate decomposition components
    let trend: Vec<f64> = (0..100).map(|i| 0.1 * i as f64).collect();
    let seasonal: Vec<f64> = (0..100)
        .map(|i| 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin())
        .collect();
    let remainder: Vec<f64> = (0..100).map(|i| 0.5 * (i as f64 * 0.7).cos()).collect();

    let strength = seasonal_strength(&trend, &seasonal, &remainder);
    println!("Computed seasonal strength: {:.4}", strength);
    println!(
        "Interpretation: {} seasonality",
        if strength >= 0.7 {
            "Strong"
        } else if strength >= 0.4 {
            "Moderate"
        } else {
            "Weak"
        }
    );

    // 7. Different Threshold Effects
    println!("\n--- Effect of Detection Threshold ---");

    let moderate_series: Vec<f64> = (0..100)
        .map(|i| {
            let seasonal = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin();
            let noise = 5.0 * (i as f64 * 0.7).cos();
            50.0 + seasonal + noise
        })
        .collect();

    println!("Moderate seasonality with noise");
    println!(
        "\n{:<12} {:>12} {:>12}",
        "Threshold", "Detected", "Strength"
    );
    println!("{:-<38}", "");

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5] {
        let config = SeasonalityConfig::default().with_threshold(threshold);
        let result = detect_seasonality(&moderate_series, &config);
        println!(
            "{:<12.1} {:>12} {:>12.4}",
            threshold, result.detected, result.strength
        );
    }

    // 8. Edge Cases
    println!("\n--- Edge Cases ---");

    // Constant series
    let constant = vec![10.0; 100];
    let result_constant = detect_seasonality_auto(&constant);
    println!(
        "Constant series: detected={}, strength={:.4}",
        result_constant.detected, result_constant.strength
    );

    // Very short series
    let short = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result_short = detect_seasonality_auto(&short);
    println!(
        "Short series (5 points): detected={}",
        result_short.detected
    );

    // High frequency (period 2)
    let high_freq: Vec<f64> = (0..50)
        .map(|i| if i % 2 == 0 { 10.0 } else { 5.0 })
        .collect();
    let result_high = detect_seasonality_auto(&high_freq);
    println!(
        "High frequency (period 2): detected={}, period={:?}",
        result_high.detected, result_high.period
    );

    // 9. Practical Examples
    println!("\n--- Common Seasonal Periods ---");
    println!(
        "
Period  Use Case
------  --------
2       Alternating patterns
4       Quarterly data
7       Weekly patterns (daily data)
12      Monthly seasonality (monthly data)
24      Hourly data with daily pattern
52      Weekly data with annual seasonality
168     Hourly data with weekly pattern (24×7)
365     Daily data with annual seasonality
"
    );

    // 10. Guidance
    println!("--- Detection Tips ---");
    println!(
        "
1. Data Requirements:
   - Need at least 2× the suspected period
   - More data improves detection accuracy

2. Threshold Selection:
   - 0.3: Detect moderate seasonality (default)
   - 0.5: Only detect strong seasonality
   - 0.1: Detect weak patterns (may have false positives)

3. Period Range:
   - Set max_period to half the series length
   - Set min_period based on sampling frequency

4. Multiple Patterns:
   - Check top candidates, not just the best
   - Harmonics may appear (e.g., period 6 if true period is 12)

5. Preprocessing:
   - Consider removing trend before detection
   - Deseasonalized data should show no pattern
"
    );

    println!("=== Seasonality Detection Example Complete ===");
}
