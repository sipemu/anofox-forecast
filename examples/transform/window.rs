//! Window Functions (Rolling/Expanding/EWM) example.
//!
//! Run with: cargo run --example window

use anofox_forecast::transform::{
    ewm_mean, ewm_std, ewm_var, expanding_max, expanding_mean, expanding_min, expanding_sum,
    rolling_max, rolling_mean, rolling_median, rolling_min, rolling_std, rolling_sum, rolling_var,
};

fn main() {
    println!("=== Window Functions Example ===\n");

    // Generate sample data
    let series: Vec<f64> = vec![
        10.0, 12.0, 15.0, 11.0, 13.0, 18.0, 14.0, 16.0, 20.0, 17.0, 19.0, 22.0, 18.0, 21.0, 25.0,
        23.0, 20.0, 24.0, 28.0, 26.0,
    ];

    println!("Original series ({} observations):", series.len());
    println!("{:?}\n", series);

    // 1. Rolling Mean
    println!("--- Rolling Mean ---");

    let rm_3 = rolling_mean(&series, 3, false);
    let rm_5 = rolling_mean(&series, 5, false);
    let rm_3_center = rolling_mean(&series, 3, true);

    println!(
        "\n{:>5} {:>8} {:>12} {:>12} {:>15}",
        "Index", "Value", "RM(3)", "RM(5)", "RM(3) Centered"
    );
    println!("{:-<55}", "");

    for i in 0..series.len() {
        println!(
            "{:>5} {:>8.1} {:>12} {:>12} {:>15}",
            i,
            series[i],
            if rm_3[i].is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.2}", rm_3[i])
            },
            if rm_5[i].is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.2}", rm_5[i])
            },
            if rm_3_center[i].is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.2}", rm_3_center[i])
            },
        );
    }

    // 2. Rolling Statistics Comparison
    println!("\n--- Rolling Statistics (window=5) ---");

    let r_mean = rolling_mean(&series, 5, false);
    let r_std = rolling_std(&series, 5, false);
    let r_min = rolling_min(&series, 5, false);
    let r_max = rolling_max(&series, 5, false);
    let r_sum = rolling_sum(&series, 5, false);
    let _r_var = rolling_var(&series, 5, false);
    let r_med = rolling_median(&series, 5, false);

    println!(
        "\n{:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Index", "Mean", "Std", "Min", "Max", "Sum", "Median"
    );
    println!("{:-<60}", "");

    for i in 4..series.len() {
        // Start at 4 since window=5
        println!(
            "{:>5} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>10.2}",
            i, r_mean[i], r_std[i], r_min[i], r_max[i], r_sum[i], r_med[i],
        );
    }

    // 3. Expanding Windows
    println!("\n--- Expanding Window Statistics ---");

    let e_mean = expanding_mean(&series);
    let e_min = expanding_min(&series);
    let e_max = expanding_max(&series);
    let e_sum = expanding_sum(&series);

    println!(
        "\n{:>5} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Index", "Value", "Cum Mean", "Cum Min", "Cum Max", "Cum Sum"
    );
    println!("{:-<55}", "");

    for i in 0..series.len() {
        println!(
            "{:>5} {:>8.1} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            i, series[i], e_mean[i], e_min[i], e_max[i], e_sum[i],
        );
    }

    // 4. Exponentially Weighted Moving (EWM)
    println!("\n--- Exponentially Weighted Moving Average ---");

    println!("\nEWM gives more weight to recent observations.");
    println!("Formula: EWM_t = alpha * x_t + (1-alpha) * EWM_{{t-1}}");
    println!("span = (2 / alpha) - 1, so alpha = 2 / (span + 1)\n");

    // alpha = 2/(span+1): span=3->0.5, span=5->0.33, span=10->0.18
    let ewm_high = ewm_mean(&series, 0.5); // High alpha - more reactive
    let ewm_med = ewm_mean(&series, 0.33); // Medium alpha
    let ewm_low = ewm_mean(&series, 0.18); // Low alpha - more smoothing

    println!(
        "{:>5} {:>8} {:>12} {:>12} {:>12}",
        "Index", "Value", "α=0.5", "α=0.33", "α=0.18"
    );
    println!("{:-<52}", "");

    for i in 0..series.len() {
        println!(
            "{:>5} {:>8.1} {:>12.2} {:>12.2} {:>12.2}",
            i, series[i], ewm_high[i], ewm_med[i], ewm_low[i],
        );
    }

    // 5. EWM Volatility
    println!("\n--- EWM Standard Deviation (Volatility) ---");

    let ewm_std_5 = ewm_std(&series, 0.33);
    let ewm_var_5 = ewm_var(&series, 0.33);

    println!(
        "\n{:>5} {:>8} {:>12} {:>12}",
        "Index", "Value", "EWM Std", "EWM Var"
    );
    println!("{:-<40}", "");

    for i in 0..10 {
        println!(
            "{:>5} {:>8.1} {:>12.2} {:>12.2}",
            i, series[i], ewm_std_5[i], ewm_var_5[i],
        );
    }

    // 6. Comparison: Rolling vs EWM
    println!("\n--- Rolling Mean vs EWM Mean ---");

    let rolling_5 = rolling_mean(&series, 5, false);
    let ewm_5_vals = ewm_mean(&series, 0.33); // alpha=0.33 ~ span=5

    println!(
        "\n{:>5} {:>8} {:>12} {:>12} {:>12}",
        "Index", "Value", "Rolling(5)", "EWM(α=0.33)", "Difference"
    );
    println!("{:-<52}", "");

    for i in 4..series.len() {
        let diff = (rolling_5[i] - ewm_5_vals[i]).abs();
        println!(
            "{:>5} {:>8.1} {:>12.2} {:>12.2} {:>12.2}",
            i, series[i], rolling_5[i], ewm_5_vals[i], diff,
        );
    }

    // 7. Smoothing Noisy Data
    println!("\n--- Smoothing Noisy Data ---");

    // Generate noisy data
    let noisy: Vec<f64> = (0..30)
        .map(|i| 50.0 + 0.5 * i as f64 + 10.0 * ((i as f64 * 0.7).sin()))
        .collect();

    println!("\nComparing smoothing methods on noisy trend data:");
    println!(
        "{:>5} {:>10} {:>12} {:>12} {:>12}",
        "Index", "Original", "Roll(5)", "EWM(5)", "EWM(10)"
    );
    println!("{:-<55}", "");

    let noisy_roll = rolling_mean(&noisy, 5, false);
    let noisy_ewm_5 = ewm_mean(&noisy, 0.33); // alpha=0.33 ~ span=5
    let noisy_ewm_10 = ewm_mean(&noisy, 0.18); // alpha=0.18 ~ span=10

    for i in (4..noisy.len()).step_by(3) {
        println!(
            "{:>5} {:>10.2} {:>12.2} {:>12.2} {:>12.2}",
            i, noisy[i], noisy_roll[i], noisy_ewm_5[i], noisy_ewm_10[i],
        );
    }

    // 8. Lag and Responsiveness
    println!("\n--- Lag and Responsiveness ---");

    // Create a step change
    let mut step_series: Vec<f64> = vec![10.0; 10];
    step_series.extend(vec![20.0; 10]);

    let step_roll_3 = rolling_mean(&step_series, 3, false);
    let step_roll_5 = rolling_mean(&step_series, 5, false);
    let step_ewm_3 = ewm_mean(&step_series, 0.5); // alpha=0.5 ~ span=3

    println!("\nStep change from 10 to 20 at index 10:");
    println!(
        "{:>5} {:>8} {:>10} {:>10} {:>10}",
        "Index", "Value", "Roll(3)", "Roll(5)", "EWM(3)"
    );
    println!("{:-<46}", "");

    for i in 8..16 {
        println!(
            "{:>5} {:>8.1} {:>10} {:>10} {:>10.2}",
            i,
            step_series[i],
            if step_roll_3[i].is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.2}", step_roll_3[i])
            },
            if step_roll_5[i].is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.2}", step_roll_5[i])
            },
            step_ewm_3[i],
        );
    }

    println!("\nObservation: EWM responds faster to the change than rolling mean.");

    // 9. Use Cases
    println!("\n--- Window Function Use Cases ---");
    println!(
        "
Rolling Mean:
  - Trend detection
  - Smoothing short-term fluctuations
  - Moving average technical indicators

Rolling Std/Var:
  - Volatility measurement
  - Bollinger Bands (mean ± 2*std)
  - Risk monitoring

Rolling Min/Max:
  - Support/resistance levels
  - Range analysis
  - Channel detection

Expanding Mean:
  - Cumulative average
  - Compare current vs historical average
  - Running totals

EWM Mean:
  - Faster response to recent changes
  - MACD indicator (EWM difference)
  - Adaptive smoothing

EWM Std:
  - Recent volatility estimation
  - EWMA volatility models
  - Risk-weighted metrics
"
    );

    // 10. Window Selection Guide
    println!("--- Window Selection Guide ---");
    println!(
        "
Window Size:
  - Small (3-5): Responsive, captures short-term changes
  - Medium (10-20): Balanced smoothing
  - Large (50+): Strong smoothing, identifies long-term trends

Rolling vs EWM:
  - Rolling: Equal weight to all points in window
  - EWM: More weight to recent observations
  - EWM: No missing values at start

Centered vs Non-centered:
  - Non-centered (default): No future data, real-time usable
  - Centered: Better for offline analysis, reduces lag
"
    );

    println!("=== Window Functions Example Complete ===");
}
