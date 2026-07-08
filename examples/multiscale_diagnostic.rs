//! Multiscale diagnostic on the fev-27 long-horizon underperformers.
//!
//! Compares `.skaters()`, `MultiScaleLaplace`, and classical baselines
//! on tourism_monthly (H=24, period=12), tourism_quarterly (H=8, period=4),
//! and m4_hourly (H=48, period=24).
//!
//! Run: `cargo run --release --features distributional --example multiscale_diagnostic`

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::laplace::multiscale::MultiScaleLaplace;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::{Forecaster, LaplaceForecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

fn parse_tsf(path: &str) -> Vec<Vec<f64>> {
    let bytes = fs::read(path).unwrap_or_default();
    let content: String = bytes.iter().map(|&b| b as char).collect();
    let mut series = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
        if !in_data {
            if line.trim_start().starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        let toks: Vec<&str> = line.split(':').collect();
        if toks.len() < 2 {
            continue;
        }
        let vals: Vec<f64> = toks[toks.len() - 1]
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        if !vals.is_empty() {
            series.push(vals);
        }
    }
    series
}

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    let s: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    s / pred.len() as f64
}

fn mase_scale(train: &[f64], period: usize) -> f64 {
    let p = if train.len() > period { period } else { 1 };
    if train.len() <= p {
        return 1.0;
    }
    let n = train.len() - p;
    let sum: f64 = (p..train.len())
        .map(|i| (train[i] - train[i - p]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}

fn bench(name: &str, path: &str, horizon: usize, period: usize, sample: usize) {
    let mut series = parse_tsf(path);
    series.retain(|v| v.len() > horizon + period);
    series.truncate(sample);
    let n = series.len();
    eprintln!("\n{name} — {n} series (H={horizon}, period={period})");

    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

    println!("\n=== {name} ({n} series, H={horizon}, period={period}) ===");
    println!("{:<24}{:>14}{:>12}", "config", "mean MASE", "fit (s)");

    // AutoTheta
    {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            let split = values.len() - horizon;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::days(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, period);
            let mut m = if period >= 2 {
                AutoTheta::seasonal(period)
            } else {
                AutoTheta::new()
            };
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let pred = m.predict(horizon).ok().map(|f| f.primary().to_vec());
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            "AutoTheta",
            mean,
            t0.elapsed().as_secs_f64()
        );
    }

    // AutoETS
    {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            let split = values.len() - horizon;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::days(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, period);
            let mut m = if period >= 2 {
                AutoETS::with_period(period)
            } else {
                AutoETS::new()
            };
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let pred = m.predict(horizon).ok().map(|f| f.primary().to_vec());
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            "AutoETS",
            mean,
            t0.elapsed().as_secs_f64()
        );
    }

    // .skaters()
    {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            let split = values.len() - horizon;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::days(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, period);
            let mut m = LaplaceForecaster::new()
                .skaters()
                .auto_with_seasonal_period(period);
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let pred = m.predict(horizon).ok().map(|f| f.primary().to_vec());
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            "skaters",
            mean,
            t0.elapsed().as_secs_f64()
        );
    }

    // MultiScaleLaplace
    {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            let split = values.len() - horizon;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::days(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, period);
            let mut m = MultiScaleLaplace::skaters(horizon);
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let pred = m.predict(horizon).ok().map(|f| f.primary().to_vec());
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            "multi-scale",
            mean,
            t0.elapsed().as_secs_f64()
        );
    }
}

fn main() {
    let sample: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    bench("m4_hourly", "validation/data/m4_hourly.tsf", 48, 24, sample);
    bench(
        "tourism_monthly",
        "validation/data/tourism_monthly.tsf",
        24,
        12,
        sample,
    );
    bench(
        "tourism_quarterly",
        "validation/data/tourism_quarterly.tsf",
        8,
        4,
        sample,
    );
}
