//! Period Detection example.
//!
//! Shows how to detect seasonal periods in time series data using
//! spectral analysis with leakage suppression. Results include
//! validation metrics: seasonal strength, ACF, and cycle count.
//!
//! Run with: cargo run --example period_detection

use anofox_forecast::detection::{detect_dominant_period, detect_periods, PeriodDetectionConfig};
use std::f64::consts::PI;

fn main() {
    println!("=== Period Detection Example ===\n");

    // ── 1. Simple single-period signal ───────────────────────────────────
    println!("--- 1. Single period (monthly data, period 12) ---\n");

    let n = 144; // 12 years of monthly data
    let signal: Vec<f64> = (0..n)
        .map(|i| 100.0 + 10.0 * (2.0 * PI * i as f64 / 12.0).sin())
        .collect();

    let dominant = detect_dominant_period(&signal);
    println!("Dominant period: {:?}", dominant);

    let periods = detect_periods(&signal, &PeriodDetectionConfig::default());
    println!("All detected periods:");
    for p in &periods {
        println!(
            "  period={:>3}, power={:.2}, strength={:.3}, acf={:.3}, cycles={}",
            p.period, p.power, p.strength, p.acf, p.n_cycles
        );
    }

    // ── 2. Multiple periods ──────────────────────────────────────────────
    println!("\n--- 2. Multiple periods (12 + 6) ---\n");

    let multi_signal: Vec<f64> = (0..144)
        .map(|i| {
            100.0
            + 10.0 * (2.0 * PI * i as f64 / 12.0).sin()  // annual
            + 3.0 * (2.0 * PI * i as f64 / 6.0).sin() // semi-annual
        })
        .collect();

    let periods = detect_periods(&multi_signal, &PeriodDetectionConfig::default());
    println!("Detected periods:");
    for p in &periods {
        println!(
            "  period={:>3}, power={:.2}, strength={:.3}, acf={:.3}",
            p.period, p.power, p.strength, p.acf
        );
    }

    // ── 3. AirPassengers-style data ──────────────────────────────────────
    println!("\n--- 3. AirPassengers-style (trend + multiplicative seasonality) ---\n");

    let air: Vec<f64> = (0..144)
        .map(|i| {
            let trend = 100.0 + 2.0 * i as f64;
            let seasonal = 1.0 + 0.3 * (2.0 * PI * i as f64 / 12.0).sin();
            trend * seasonal
        })
        .collect();

    let periods = detect_periods(&air, &PeriodDetectionConfig::default());
    println!("Detected periods:");
    for p in &periods {
        println!(
            "  period={:>3}, power={:.2}, strength={:.3}, acf={:.3}, cycles={}",
            p.period, p.power, p.strength, p.acf, p.n_cycles
        );
    }
    println!("\nNote: Period 12 should be detected cleanly, without spectral");
    println!("leakage artifacts (e.g., no spurious period 13).");

    // ── 4. Custom configuration ──────────────────────────────────────────
    println!("\n--- 4. Custom configuration ---\n");

    let config = PeriodDetectionConfig {
        min_period: 3,
        max_period: Some(24),
        max_periods: 3,
        min_power_ratio: 2.0,
        min_strength: 0.1, // filter weak periods
        min_cycles: 3,     // require at least 3 full cycles
        ..Default::default()
    };

    println!(
        "Config: min_period={}, max_period={:?}, min_strength={}, min_cycles={}",
        config.min_period, config.max_period, config.min_strength, config.min_cycles
    );

    let periods = detect_periods(&air, &config);
    println!("Filtered periods:");
    for p in &periods {
        println!(
            "  period={:>3}, strength={:.3}, cycles={}",
            p.period, p.strength, p.n_cycles
        );
    }

    // ── 5. Daily data with weekly pattern ────────────────────────────────
    println!("\n--- 5. Daily data (weekly period = 7) ---\n");

    let daily: Vec<f64> = (0..365)
        .map(|i| {
            50.0 + 8.0 * (2.0 * PI * i as f64 / 7.0).sin() + 3.0 * (2.0 * PI * i as f64 / 7.0).cos()
        })
        .collect();

    let periods = detect_periods(&daily, &PeriodDetectionConfig::default());
    println!("Detected periods:");
    for p in &periods {
        println!(
            "  period={:>3}, power={:.2}, strength={:.3}",
            p.period, p.power, p.strength
        );
    }

    // ── 6. No seasonality ────────────────────────────────────────────────
    println!("\n--- 6. Non-seasonal data ---\n");

    // Irrational-frequency signal — no integer period
    let noise: Vec<f64> = (0..200)
        .map(|i| 50.0 + (i as f64 * 0.1).sin() + (i as f64 * std::f64::consts::E * 0.07).cos())
        .collect();

    let dominant = detect_dominant_period(&noise);
    println!("Dominant period: {:?}", dominant);
    println!("(None = no significant periodicity found)");

    println!("\nTip: Trends can create spurious spectral peaks.");
    println!("Detrend or first-difference before detection if needed.");

    // ── 7. Using detected periods with MSTL ──────────────────────────────
    println!("\n--- 7. Integration with forecasting ---\n");
    println!("Detected periods feed directly into MSTL decomposition:");
    println!();
    println!("  let periods = detect_periods(&values, &config);");
    println!("  let seasonal_periods: Vec<usize> = periods.iter()");
    println!("      .map(|p| p.period)");
    println!("      .collect();");
    println!("  let mut model = MSTLForecaster::new(seasonal_periods);");

    // ── 8. Validation fields ─────────────────────────────────────────────
    println!("\n--- 8. Understanding validation fields ---\n");
    println!("Each detected Period includes:");
    println!("  - period:   the lag (e.g., 12 for annual in monthly data)");
    println!("  - power:    spectral power (higher = stronger signal)");
    println!("  - strength: seasonal differencing strength [0,1]");
    println!("              (1 = perfect seasonal pattern, 0 = no pattern)");
    println!("  - acf:      autocorrelation at the candidate lag");
    println!("              (high acf confirms self-similarity)");
    println!("  - n_cycles: how many full cycles fit in the data");
    println!("              (more cycles = more reliable estimate)");

    println!("\n=== Period Detection Example Complete ===");
}
