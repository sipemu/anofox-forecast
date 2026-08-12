//! Diagnose why `.skaters()` fails on m4_hourly (H=48, period=24).
//!
//! Runs 3-4 forecaster configs on a 30-series sample of m4_hourly and
//! reports mean MASE + fit time. Isolates the sticky-lattice cost and
//! the seasonal-diff+EMA long-horizon extrapolation cost.
//!
//! Run: `cargo run --release --features distributional --example m4_hourly_diagnostic`

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::laplace::multiscale::MultiScaleLaplace;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::{Forecaster, LaplaceForecaster};
use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 48;
const PERIOD: usize = 24;

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

fn main() {
    let sample: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let mut series = parse_tsf("validation/data/m4_hourly.tsf");
    series.retain(|v| v.len() > HORIZON + 200);
    series.truncate(sample);
    let n = series.len();
    eprintln!("Running on {n} m4_hourly series (H={HORIZON}, period={PERIOD})");

    let base = Utc.with_ymd_and_hms(2015, 1, 1, 0, 0, 0).unwrap();

    #[allow(clippy::type_complexity)]
    let configs: Vec<(&str, Box<dyn Fn() -> LaplaceForecaster>)> = vec![
        (
            "skaters (full)",
            Box::new(|| LaplaceForecaster::new().skaters()),
        ),
        (
            "skaters no_sticky",
            Box::new(|| LaplaceForecaster::new().skaters().no_sticky()),
        ),
        ("auto", Box::new(|| LaplaceForecaster::new().auto())),
    ];

    // Report classical baselines separately for context.
    println!("\n=== m4_hourly ({n} series, H=48) ===");
    println!("{:<24}{:>14}{:>12}", "config", "mean MASE", "fit (s)");

    // Classical: AutoETS + AutoTheta
    for (name, run_ets) in &[("AutoETS", true), ("AutoTheta", false)] {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            if values.len() <= HORIZON + PERIOD {
                continue;
            }
            let split = values.len() - HORIZON;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::hours(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, PERIOD);
            let pred: Option<Vec<f64>> = if *run_ets {
                let mut m = AutoETS::with_period(PERIOD);
                if m.fit(&train_ts).is_err() {
                    continue;
                }
                m.predict(HORIZON).ok().map(|f| f.primary().to_vec())
            } else {
                let mut m = AutoTheta::seasonal(PERIOD);
                if m.fit(&train_ts).is_err() {
                    continue;
                }
                m.predict(HORIZON).ok().map(|f| f.primary().to_vec())
            };
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            name,
            mean,
            t0.elapsed().as_secs_f64()
        );
    }

    // Laplace configs
    for (name, make) in &configs {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            if values.len() <= HORIZON + PERIOD {
                continue;
            }
            let split = values.len() - HORIZON;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::hours(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, PERIOD);
            let mut m = make().auto_with_seasonal_period(PERIOD);
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let pred = m.predict(HORIZON).ok().map(|f| f.primary().to_vec());
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            name,
            mean,
            t0.elapsed().as_secs_f64()
        );
    }

    // Multi-scale wrapper — new after fev-27 follow-up.
    {
        let t0 = Instant::now();
        let mut mases = Vec::new();
        for values in &series {
            if values.len() <= HORIZON + PERIOD {
                continue;
            }
            let split = values.len() - HORIZON;
            let train_v = values[..split].to_vec();
            let test_v = &values[split..];
            let stamps: Vec<_> = (0..train_v.len())
                .map(|i| base + Duration::hours(i as i64))
                .collect();
            let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let scale = mase_scale(&train_v, PERIOD);
            let mut m = MultiScaleLaplace::skaters(HORIZON);
            if m.fit(&train_ts).is_err() {
                continue;
            }
            let pred = m.predict(HORIZON).ok().map(|f| f.primary().to_vec());
            if let Some(p) = pred {
                if p.len() == test_v.len() {
                    mases.push(mae(&p, test_v) / scale);
                }
            }
        }
        let mean = mases.iter().sum::<f64>() / mases.len().max(1) as f64;
        println!(
            "{:<24}{:>14.4}{:>12.1}",
            "multi-scale skaters",
            mean,
            t0.elapsed().as_secs_f64()
        );
    }
}
