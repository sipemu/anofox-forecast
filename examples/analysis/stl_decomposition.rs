//! STL Decomposition example.
//!
//! Run with: cargo run --example stl_decomposition

use anofox_forecast::seasonality::{MSTL, STL};

fn main() {
    println!("=== STL Decomposition Example ===\n");

    println!("STL (Seasonal-Trend decomposition using LOESS) breaks down a");
    println!("time series into three additive components:");
    println!("  Y = Trend + Seasonal + Remainder\n");

    // Generate sample data with trend and seasonality
    let period = 12;
    let n = 120; // 10 years of monthly data

    let series: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 0.1 * i as f64; // Linear trend
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin(); // Seasonal
            let noise = 0.5 * ((i as f64 * 0.7).cos()); // Noise
            trend + seasonal + noise
        })
        .collect();

    println!("Generated: {} observations with period={}\n", n, period);

    // 1. Basic STL Decomposition
    println!("--- Basic STL Decomposition ---");

    let stl = STL::new(period);
    let result = stl.decompose(&series).unwrap();

    println!("Decomposition complete!");
    println!("  Trend length:     {}", result.trend.len());
    println!("  Seasonal length:  {}", result.seasonal.len());
    println!("  Remainder length: {}", result.remainder.len());

    // Verify additive decomposition
    let max_reconstruction_error: f64 = series
        .iter()
        .zip(result.trend.iter())
        .zip(result.seasonal.iter())
        .zip(result.remainder.iter())
        .map(|(((y, t), s), r)| (y - (t + s + r)).abs())
        .fold(0.0, f64::max);
    println!(
        "\nMax reconstruction error: {:.2e}",
        max_reconstruction_error
    );

    // 2. Component Statistics
    println!("\n--- Component Statistics ---");

    let trend_mean = result.trend.iter().sum::<f64>() / result.trend.len() as f64;
    let seasonal_mean = result.seasonal.iter().sum::<f64>() / result.seasonal.len() as f64;
    let remainder_mean = result.remainder.iter().sum::<f64>() / result.remainder.len() as f64;

    let trend_std = (result
        .trend
        .iter()
        .map(|x| (x - trend_mean).powi(2))
        .sum::<f64>()
        / result.trend.len() as f64)
        .sqrt();
    let seasonal_std = (result
        .seasonal
        .iter()
        .map(|x| (x - seasonal_mean).powi(2))
        .sum::<f64>()
        / result.seasonal.len() as f64)
        .sqrt();
    let remainder_std = (result
        .remainder
        .iter()
        .map(|x| (x - remainder_mean).powi(2))
        .sum::<f64>()
        / result.remainder.len() as f64)
        .sqrt();

    println!("{:<12} {:>12} {:>12}", "Component", "Mean", "Std Dev");
    println!("{:-<38}", "");
    println!("{:<12} {:>12.4} {:>12.4}", "Trend", trend_mean, trend_std);
    println!(
        "{:<12} {:>12.4} {:>12.4}",
        "Seasonal", seasonal_mean, seasonal_std
    );
    println!(
        "{:<12} {:>12.4} {:>12.4}",
        "Remainder", remainder_mean, remainder_std
    );

    // 3. Trend and Seasonal Strength
    println!("\n--- Trend and Seasonal Strength ---");

    let trend_strength = result.trend_strength();
    let seasonal_strength = result.seasonal_strength();

    println!(
        "Trend strength:    {:.4} ({})",
        trend_strength,
        if trend_strength > 0.5 {
            "Strong"
        } else {
            "Weak"
        }
    );
    println!(
        "Seasonal strength: {:.4} ({})",
        seasonal_strength,
        if seasonal_strength > 0.5 {
            "Strong"
        } else {
            "Weak"
        }
    );

    // 4. Show first season of components
    println!("\n--- First Season of Components ---");
    println!(
        "{:>4} {:>10} {:>10} {:>10} {:>10}",
        "t", "Original", "Trend", "Seasonal", "Remainder"
    );
    println!("{:-<46}", "");
    for (i, &value) in series.iter().enumerate().take(period) {
        println!(
            "{:>4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            i, value, result.trend[i], result.seasonal[i], result.remainder[i]
        );
    }

    // 5. STL with Custom Parameters
    println!("\n--- STL with Custom Parameters ---");

    let stl_custom = STL::new(period)
        .with_seasonal_smoothness(7)
        .with_trend_smoothness(21)
        .with_inner_iterations(3);

    let result_custom = stl_custom.decompose(&series).unwrap();

    println!("Custom parameters:");
    println!("  Seasonal smoothness: 7");
    println!("  Trend smoothness: 21");
    println!("  Inner iterations: 3");
    println!("\nCustom decomposition strength:");
    println!("  Trend strength:    {:.4}", result_custom.trend_strength());
    println!(
        "  Seasonal strength: {:.4}",
        result_custom.seasonal_strength()
    );

    // 6. Robust STL (handles outliers)
    println!("\n--- Robust STL with Outliers ---");

    // Add outliers to the series
    let mut series_with_outliers = series.clone();
    series_with_outliers[30] = 100.0; // Large positive outlier
    series_with_outliers[60] = -80.0; // Large negative outlier
    series_with_outliers[90] = 150.0; // Another outlier

    // Standard STL
    let stl_standard = STL::new(period);
    let result_standard = stl_standard.decompose(&series_with_outliers).unwrap();

    // Robust STL
    let stl_robust = STL::new(period).robust();
    let result_robust = stl_robust.decompose(&series_with_outliers).unwrap();

    println!("Series with 3 outliers:");
    println!(
        "\n{:<15} {:>15} {:>15}",
        "Method", "Trend Strength", "Seasonal Strength"
    );
    println!("{:-<47}", "");
    println!(
        "{:<15} {:>15.4} {:>15.4}",
        "Standard",
        result_standard.trend_strength(),
        result_standard.seasonal_strength()
    );
    println!(
        "{:<15} {:>15.4} {:>15.4}",
        "Robust",
        result_robust.trend_strength(),
        result_robust.seasonal_strength()
    );

    // 7. MSTL (Multiple Seasonal Decomposition)
    println!("\n--- MSTL (Multiple Seasonal Periods) ---");

    // Generate data with multiple seasonal patterns
    let n_multi = 168; // 7 weeks of hourly data
    let series_multi: Vec<f64> = (0..n_multi)
        .map(|i| {
            let trend = 0.01 * i as f64;
            let daily = 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0).sin(); // Daily (24h)
            let weekly = 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 168.0).sin(); // Weekly (168h)
            let noise = 0.2 * ((i as f64 * 0.3).cos());
            50.0 + trend + daily + weekly + noise
        })
        .collect();

    println!("Hourly data with daily (24h) and weekly (168h) seasonality");

    let mstl = MSTL::new(vec![24, 168]);
    if let Some(mstl_result) = mstl.decompose(&series_multi) {
        println!("\nMSTL Results:");
        println!("  Trend length: {}", mstl_result.trend.len());
        println!("  Seasonal periods: {:?}", mstl_result.seasonal_periods);
        println!(
            "  Number of seasonal components: {}",
            mstl_result.seasonal_components.len()
        );

        for (i, (period, seasonal)) in mstl_result
            .seasonal_periods
            .iter()
            .zip(mstl_result.seasonal_components.iter())
            .enumerate()
        {
            let var = seasonal.iter().map(|x| x * x).sum::<f64>() / seasonal.len() as f64;
            println!(
                "  Seasonal {} (period={}): variance = {:.4}",
                i + 1,
                period,
                var
            );
        }

        println!("\nMSTL Strength:");
        println!("  Trend strength:    {:.4}", mstl_result.trend_strength());
        if let Some(strength) = mstl_result.seasonal_strength(0) {
            println!("  Seasonal strength (period 0): {:.4}", strength);
        }
    } else {
        println!("MSTL decomposition failed (insufficient data)");
    }

    // 8. Use Cases
    println!("\n--- STL Use Cases ---");
    println!(
        "
1. Seasonal Adjustment
   - Remove seasonality for trend analysis
   - Compare year-over-year changes

2. Forecasting
   - Forecast trend and seasonal separately
   - Recombine for final forecast

3. Anomaly Detection
   - Large remainder values indicate anomalies
   - Outliers stand out in remainder component

4. Pattern Analysis
   - Identify changing seasonal patterns
   - Detect trend changes

5. Data Visualization
   - Separate components for clearer plots
   - Understand what drives the series
"
    );

    println!("=== STL Decomposition Example Complete ===");
}
