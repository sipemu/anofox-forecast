//! Focused check: does the AutoETS/AutoTheta period fix close the gap to
//! fev's Nixtla reference implementations?
//!
//! Tests two configs on a small sample:
//! - `AutoETS::new()` — no period (the old handicapped fev_benchmark)
//! - `AutoETS::with_period(P)` — with fev's canonical period
//!   Same for AutoTheta.
//!
//! Datasets: m3_monthly (period=12, THE fix should matter), m5 (period=1
//! per fev, fix is a no-op there — sanity check).

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;
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

fn run_case(
    label: &str,
    dataset_path: &str,
    horizon: usize,
    period: usize,
    sample: usize,
    with_period_fix: bool,
) {
    let mut kept = parse_tsf(dataset_path);
    kept.retain(|v| v.len() > horizon + 12);
    kept.truncate(sample);
    let n = kept.len();
    if n == 0 {
        println!("{label}: no data");
        return;
    }
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let mut ets_maes = Vec::new();
    let mut theta_maes = Vec::new();
    let mut ets_time = 0.0f64;
    let mut theta_time = 0.0f64;

    for values in &kept {
        let split = values.len() - horizon;
        let train_v = values[..split].to_vec();
        let test_v = &values[split..];
        let scale = mase_scale(&train_v, period);
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base + Duration::days(i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // AutoETS
        let t0 = Instant::now();
        let mut m = if with_period_fix && period >= 2 {
            AutoETS::with_period(period)
        } else {
            AutoETS::new()
        };
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(horizon) {
                if fc.primary().len() == test_v.len() {
                    ets_maes.push(mae(fc.primary(), test_v) / scale);
                }
            }
        }
        ets_time += t0.elapsed().as_secs_f64();

        // AutoTheta
        let t0 = Instant::now();
        let mut m = if with_period_fix && period >= 2 {
            AutoTheta::seasonal(period)
        } else {
            AutoTheta::new()
        };
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(horizon) {
                if fc.primary().len() == test_v.len() {
                    theta_maes.push(mae(fc.primary(), test_v) / scale);
                }
            }
        }
        theta_time += t0.elapsed().as_secs_f64();
    }
    let ets_mean = ets_maes.iter().sum::<f64>() / ets_maes.len().max(1) as f64;
    let theta_mean = theta_maes.iter().sum::<f64>() / theta_maes.len().max(1) as f64;
    println!(
        "{:<40} ETS={:.3} ({:.1}s, n={})  Theta={:.3} ({:.1}s, n={})",
        label,
        ets_mean,
        ets_time,
        ets_maes.len(),
        theta_mean,
        theta_time,
        theta_maes.len()
    );
}

fn main() {
    let sample: usize = std::env::var("SAMPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    println!("Sample per dataset: {sample}");
    println!(
        "\nfev leaderboard reference (Nixtla, all series):\n\
         \x20  m3_monthly:  auto_ets=0.863  auto_theta=0.855\n\
         \x20  m5:          auto_ets=1.101  auto_theta=1.119\n"
    );

    println!("=== m3_monthly (fev period=12) ===");
    run_case(
        "  no fix (AutoETS::new)",
        "validation/data/m3_monthly.tsf",
        18,
        12,
        sample,
        false,
    );
    run_case(
        "  with fix (AutoETS::with_period(12))",
        "validation/data/m3_monthly.tsf",
        18,
        12,
        sample,
        true,
    );

    println!("\n=== m5 (fev period=1, fix is no-op) ===");
    run_case(
        "  no fix (AutoETS::new)",
        "validation/data/m5.tsf",
        28,
        1,
        sample,
        false,
    );
    run_case(
        "  with fix (period=1, no-op)",
        "validation/data/m5.tsf",
        28,
        1,
        sample,
        true,
    );
}
