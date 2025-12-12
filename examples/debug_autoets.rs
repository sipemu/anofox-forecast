use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, ETSSpec, ErrorType, ETSSeasonalType, TrendType, ETS
};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Read noisy_seasonal.csv
    let file = File::open("validation/data/noisy_seasonal.csv").expect("Failed to open file");
    let reader = BufReader::new(file);

    let mut values = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(val) = parts[1].parse::<f64>() {
                values.push(val);
            }
        }
    }

    println!("Loaded {} observations", values.len());
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    println!("Mean: {:.2}, Std: {:.2}", mean, variance.sqrt());

    // Create timestamps
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    println!("\n{}", "=".repeat(80));
    println!("Testing individual ETS models");
    println!("{}", "=".repeat(80));

    // Test different models manually
    let models_to_test = vec![
        ("A,N,N (SES)", ETSSpec::ann(), 1),
        ("A,A,N (Holt)", ETSSpec::aan(), 1),
        ("A,N,A (Seasonal)", ETSSpec::new(ErrorType::Additive, TrendType::None, ETSSeasonalType::Additive), 12),
        ("A,A,A (HW Add)", ETSSpec::aaa(), 12),
    ];

    for (name, spec, period) in models_to_test {
        let mut model = ETS::new(spec, period);
        if model.fit(&ts).is_ok() {
            let aic = model.aic().unwrap_or(f64::MAX);
            let aicc = model.aicc().unwrap_or(f64::MAX);
            let bic = model.bic().unwrap_or(f64::MAX);
            let alpha = model.alpha().unwrap_or(0.0);

            println!("{:15} AIC={:8.2}, AICc={:8.2}, BIC={:8.2}, alpha={:.6}",
                name, aic, aicc, bic, alpha);
        } else {
            println!("{:15} FAILED TO FIT", name);
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("Testing AutoETS with season_length=12");
    println!("{}", "=".repeat(80));

    let config = AutoETSConfig::with_period(12);
    let mut auto_model = AutoETS::with_config(config);
    auto_model.fit(&ts).expect("Failed to fit AutoETS");

    if let Some(spec) = auto_model.selected_spec() {
        println!("Selected: {}", spec.short_name());

        // Get the scores
        let scores = auto_model.model_scores();
        println!("\nTop 5 models by AICc:");
        for (i, (spec, score)) in scores.iter().take(5).enumerate() {
            println!("  {}. {} - AICc={:.4}", i+1, spec.short_name(), score);
        }

        // Get forecasts
        let forecast = auto_model.predict(12).unwrap();
        let forecasts = forecast.primary();
        println!("\nForecasts:");
        for (i, f) in forecasts.iter().enumerate() {
            println!("  Step {}: {:.4}", i+1, f);
        }

        // Check if constant
        let all_same = forecasts.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        println!("\nAll forecasts constant: {}", all_same);
    }
}
