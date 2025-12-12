// Quick test to debug Rust HoltWinters on multiplicative_seasonal data
use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{HoltWinters, SeasonalType};
use anofox_forecast::models::Forecaster;
use chrono::{TimeZone, Utc};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read multiplicative_seasonal data
    let file = File::open("validation/data/multiplicative_seasonal.csv")?;
    let reader = BufReader::new(file);

    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 { continue; } // Skip header

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let ts_str = parts[0].trim();
            let naive = chrono::NaiveDate::parse_from_str(ts_str, "%Y-%m-%d")?
                .and_hms_opt(0, 0, 0).unwrap();
            let ts = Utc.from_utc_datetime(&naive);
            timestamps.push(ts);

            let value: f64 = parts[1].trim().parse()?;
            values.push(value);
        }
    }

    println!("Loaded {} observations", values.len());
    println!();

    // Create time series
    let ts = TimeSeries::univariate(timestamps, values)?;

    // Test with additive seasonality (same as statsforecast validation)
    let mut model = HoltWinters::auto(12, SeasonalType::Additive);
    model.fit(&ts)?;

    println!("=== Rust HoltWinters (Additive, Auto-optimized) ===");
    println!();

    println!("Optimized Parameters:");
    println!("  alpha: {:.10}", model.alpha().unwrap());
    println!("  beta:  {:.10}", model.beta().unwrap());
    println!("  gamma: {:.10}", model.gamma().unwrap());
    println!();

    println!("Final State:");
    println!("  level: {:.10}", model.level().unwrap());
    println!("  trend: {:.10}", model.trend().unwrap());
    println!();

    println!("Seasonal Components:");
    if let Some(seasonals) = model.seasonals() {
        for (i, &s) in seasonals.iter().enumerate() {
            println!("  s[{:2}]: {:12.6}", i, s);
        }
    }
    println!();

    // Generate forecasts
    let forecast = model.predict(12)?;
    let predictions = forecast.primary();

    println!("Forecasts (h=12):");
    for (i, &pred) in predictions.iter().enumerate() {
        println!("  Step {:2}: {:.10}", i+1, pred);
    }
    println!();

    // Manual verification of first forecast
    let l = model.level().unwrap();
    let b = model.trend().unwrap();
    let seasonals = model.seasonals().unwrap();
    let n = ts.len();
    let season_idx = (n + 1 - 1) % 12;
    let s = seasonals[season_idx];

    println!("Manual verification of first forecast:");
    println!("  n = {}", n);
    println!("  h = 1");
    println!("  season_idx = (n + h - 1) % 12 = ({} + 1 - 1) % 12 = {}", n, season_idx);
    println!("  s = seasonals[{}] = {:.10}", season_idx, s);
    println!("  l = {:.10}", l);
    println!("  b = {:.10}", b);
    println!("  forecast = l + h*b + s");
    println!("           = {:.10} + 1*{:.10} + {:.10}", l, b, s);
    let manual = l + b + s;
    println!("           = {:.10}", manual);
    println!("  model.predict()[0] = {:.10}", predictions[0]);
    println!("  Match: {}", (manual - predictions[0]).abs() < 1e-6);

    Ok(())
}
