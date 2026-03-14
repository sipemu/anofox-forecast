//! Demonstrates the trend and seasonal component system.
//!
//! Generates 4 synthetic series and shows:
//! 1. Individual component usage with recency
//! 2. AutoTrend selection with score tables
//! 3. Pipeline integration with TrendMode::Auto

use anofox_forecast::seasonality::{
    auto_trend::AutoTrend, ExponentialTrend, LogisticTrend, PolynomialTrend, Recency,
    TheilSenTrend, TrendComponent,
};

fn main() {
    println!("=== Trend Component Examples ===\n");

    // ── 1. Linear + noise ────────────────────────────────────────────
    println!("--- 1. Linear + noise ---");
    let linear_data: Vec<f64> = (0..100)
        .map(|i| {
            let t = i as f64;
            2.0 * t + 10.0 + pseudo_noise(i, 42) * 3.0
        })
        .collect();

    let mut poly = PolynomialTrend::new(1).with_recency(Recency::Fraction(0.3));
    poly.fit_trend(&linear_data).unwrap();
    println!(
        "  PolynomialTrend(1): slope={:.4}, R²={:.4}",
        poly.coefficients().unwrap()[1],
        poly.trend_features()
            .iter()
            .find(|(n, _)| *n == "polynomial_r_squared")
            .unwrap()
            .1
    );

    let mut theilsen = TheilSenTrend::new().with_recency(Recency::Fraction(0.3));
    theilsen.fit_trend(&linear_data).unwrap();
    println!(
        "  TheilSenTrend:      slope={:.4}, R²={:.4}",
        theilsen.slope(),
        theilsen
            .trend_features()
            .iter()
            .find(|(n, _)| *n == "theilsen_r_squared")
            .unwrap()
            .1
    );

    let mut auto = AutoTrend::new().with_recency(Recency::Full);
    auto.fit_trend(&linear_data).unwrap();
    let result = auto.selection_result().unwrap();
    println!("\n  AutoTrend selection (AICc):");
    for (name, score) in &result.scores {
        let marker = if *name == result.selected { " <--" } else { "" };
        println!("    {:<20} {:.2}{}", name, score, marker);
    }

    // ── 2. Exponential growth ────────────────────────────────────────
    println!("\n--- 2. Exponential growth ---");
    let exp_data: Vec<f64> = (0..100)
        .map(|i| {
            let t = i as f64;
            2.0 * (0.05 * t).exp() + pseudo_noise(i, 17).abs() * 0.5
        })
        .collect();

    let mut exp_trend = ExponentialTrend::new().with_recency(Recency::Full);
    exp_trend.fit_trend(&exp_data).unwrap();
    println!(
        "  ExponentialTrend: growth_rate={:.4}, R²={:.4}",
        exp_trend.growth_rate(),
        exp_trend
            .trend_features()
            .iter()
            .find(|(n, _)| *n == "exponential_r_squared")
            .unwrap()
            .1
    );

    let mut auto = AutoTrend::new().with_recency(Recency::Full);
    auto.fit_trend(&exp_data).unwrap();
    let result = auto.selection_result().unwrap();
    println!("\n  AutoTrend selection (AICc):");
    for (name, score) in &result.scores {
        let marker = if *name == result.selected { " <--" } else { "" };
        println!("    {:<20} {:.2}{}", name, score, marker);
    }

    // ── 3. Logistic saturation ───────────────────────────────────────
    println!("\n--- 3. Logistic saturation ---");
    let logistic_data: Vec<f64> = (0..100)
        .map(|i| {
            let t = i as f64;
            100.0 / (1.0 + (-0.1 * (t - 50.0)).exp()) + pseudo_noise(i, 7) * 2.0
        })
        .collect();

    let mut logistic = LogisticTrend::new()
        .with_capacity(100.0)
        .with_recency(Recency::Full);
    logistic.fit_trend(&logistic_data).unwrap();
    println!(
        "  LogisticTrend: capacity={:.1}, midpoint={:.1}, steepness={:.4}",
        logistic.capacity(),
        logistic.midpoint(),
        logistic.steepness()
    );

    let forecast = logistic.predict_trend(20);
    println!(
        "  Forecast: last 5 values -> [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        forecast[15], forecast[16], forecast[17], forecast[18], forecast[19]
    );

    // ── 4. Regime change (piecewise with slope reversal) ─────────────
    println!("\n--- 4. Regime change ---");
    let regime_data: Vec<f64> = (0..100)
        .map(|i| {
            let t = i as f64;
            if i < 60 {
                3.0 * t + 10.0 + pseudo_noise(i, 99) * 2.0
            } else {
                // Reverse: slope goes from +3 to -2
                3.0 * 60.0 + 10.0 - 2.0 * (t - 60.0) + pseudo_noise(i, 99) * 2.0
            }
        })
        .collect();

    // Show recency effect: recent trend vs global trend
    let mut recent = PolynomialTrend::new(1).with_recency(Recency::Fraction(0.3));
    recent.fit_trend(&regime_data).unwrap();
    let recent_slope = recent.coefficients().unwrap()[1];

    let mut global = PolynomialTrend::new(1).with_recency(Recency::Full);
    global.fit_trend(&regime_data).unwrap();
    let global_slope = global.coefficients().unwrap()[1];

    println!("  Recent slope (last 30%): {:.4}", recent_slope);
    println!("  Global slope (full):     {:.4}", global_slope);
    println!(
        "  Regime change detected:  {:.4} (abs diff)",
        (recent_slope - global_slope).abs()
    );

    let mut auto = AutoTrend::new().with_recency(Recency::Fraction(0.3));
    auto.fit_trend(&regime_data).unwrap();
    let result = auto.selection_result().unwrap();
    println!("\n  AutoTrend selection (recency=30%, AICc):");
    for (name, score) in &result.scores {
        let marker = if *name == result.selected { " <--" } else { "" };
        println!("    {:<20} {:.2}{}", name, score, marker);
    }

    println!("\n=== Done ===");
}

/// Deterministic pseudo-noise from index and seed.
fn pseudo_noise(index: usize, seed: u64) -> f64 {
    let mut state = seed.wrapping_add(index as u64);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
}
