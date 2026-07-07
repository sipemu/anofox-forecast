//! autogluon/fev-style benchmark across the Monash Time Series Forecasting
//! Archive datasets that make up the Chronos benchmark's classical panel.
//!
//! Reports MASE (fev's canonical point metric) with fev's seasonal-naive
//! scaling per dataset. Runs the anofox-forecast α-25 stack against
//! internal baselines (AutoETS, AutoTheta).
//!
//! Datasets: m3_monthly, m4_hourly/daily/weekly/monthly/quarterly/yearly,
//! tourism_monthly/quarterly, cif_2016. All Monash `.tsf` format.
//!
//! Run: `cargo run --release --features distributional --example fev_benchmark`
//! Configure: `SAMPLE_PER=200` (default: all) to limit series per dataset.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;

#[cfg(feature = "distributional")]
use anofox_forecast::models::{LaplaceForecaster, SmartForecaster};

use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

/// One Monash-formatted dataset with its canonical horizon and seasonal
/// period (from the fev / Chronos benchmark and the Monash archive
/// metadata).
struct Dataset {
    name: &'static str,
    path: &'static str,
    horizon: usize,
    period: usize,
    /// Fev / Monash used this many months/days/weeks per Duration step
    /// for TimeSeries construction — we don't need real dates, just a
    /// consistent monotonic spacing.
    step_seconds: i64,
}

const DATASETS: &[Dataset] = &[
    Dataset {
        name: "m3_monthly",
        path: "validation/data/m3_monthly.tsf",
        horizon: 18,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "m4_hourly",
        path: "validation/data/m4_hourly.tsf",
        horizon: 48,
        period: 24,
        step_seconds: 3600,
    },
    Dataset {
        name: "m4_daily",
        path: "validation/data/m4_daily.tsf",
        horizon: 14,
        period: 7,
        step_seconds: 86400,
    },
    Dataset {
        name: "m4_weekly",
        path: "validation/data/m4_weekly.tsf",
        horizon: 13,
        period: 1,
        step_seconds: 7 * 86400,
    },
    Dataset {
        name: "m4_monthly",
        path: "validation/data/m4_monthly.tsf",
        horizon: 18,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "m4_quarterly",
        path: "validation/data/m4_quarterly.tsf",
        horizon: 8,
        period: 4,
        step_seconds: 90 * 86400,
    },
    Dataset {
        name: "m4_yearly",
        path: "validation/data/m4_yearly.tsf",
        horizon: 6,
        period: 1,
        step_seconds: 365 * 86400,
    },
    Dataset {
        name: "tourism_monthly",
        path: "validation/data/tourism_monthly.tsf",
        horizon: 24,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "tourism_quarterly",
        path: "validation/data/tourism_quarterly.tsf",
        horizon: 8,
        period: 4,
        step_seconds: 90 * 86400,
    },
    Dataset {
        name: "cif_2016",
        path: "validation/data/cif_2016.tsf",
        horizon: 12,
        period: 12,
        step_seconds: 30 * 86400,
    },
];

const MODEL_NAMES: &[&str] = &[
    "AutoETS",
    "AutoTheta",
    "Laplace+auto",
    "Laplace+auto_aid",
    "SmartForecaster",
];
const N_MODELS: usize = 5;

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
        // Robust to variable header shape: split on ':' and take the LAST
        // token as the CSV values.
        let toks: Vec<&str> = line.split(':').collect();
        if toks.len() < 2 {
            continue;
        }
        let vals_str = toks[toks.len() - 1];
        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if !values.is_empty() {
            series.push(values);
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

/// Seasonal-naive scale used by MASE. When `train.len() <= period` fall
/// back to naive-1 (adjacent differences).
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

struct DatasetResult {
    name: &'static str,
    n_series: usize,
    /// One MASE per model, arithmetic mean across series.
    mase_mean: [f64; N_MODELS],
    /// One count per model, series that succeeded.
    n_ok: [usize; N_MODELS],
    /// Total fit time in seconds per model.
    total_s: [f64; N_MODELS],
}

fn run_dataset(ds: &Dataset, sample_per: usize) -> Option<DatasetResult> {
    let mut kept = parse_tsf(ds.path);
    if kept.is_empty() {
        eprintln!("  [{}] no data", ds.name);
        return None;
    }
    kept.retain(|v| v.len() > ds.horizon + 12);
    kept.truncate(sample_per);
    let n_series = kept.len();
    eprintln!(
        "  [{}] {} series (H={}, period={})",
        ds.name, n_series, ds.horizon, ds.period
    );

    let base_date = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let mut mase_sum = [0.0f64; N_MODELS];
    let mut n_ok = [0usize; N_MODELS];
    let mut fit_us_sum = [0u128; N_MODELS];

    for values in &kept {
        let split = values.len() - ds.horizon;
        let train_v = values[..split].to_vec();
        let test_v = &values[split..];
        let scale = mase_scale(&train_v, ds.period);
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base_date + Duration::seconds(ds.step_seconds * i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Model 0: AutoETS
        {
            let t0 = Instant::now();
            let mut m = AutoETS::new();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[0] += mae(p, test_v) / scale;
                        n_ok[0] += 1;
                    }
                }
            }
            fit_us_sum[0] += t0.elapsed().as_micros();
        }
        // Model 1: AutoTheta
        {
            let t0 = Instant::now();
            let mut m = AutoTheta::new();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[1] += mae(p, test_v) / scale;
                        n_ok[1] += 1;
                    }
                }
            }
            fit_us_sum[1] += t0.elapsed().as_micros();
        }
        // Model 2: Laplace + auto (uses dataset period)
        #[cfg(feature = "distributional")]
        {
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new()
                .auto()
                .auto_with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[2] += mae(p, test_v) / scale;
                        n_ok[2] += 1;
                    }
                }
            }
            fit_us_sum[2] += t0.elapsed().as_micros();
        }
        // Model 3: Laplace + auto_aid
        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        {
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new()
                .auto_aid()
                .auto_with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[3] += mae(p, test_v) / scale;
                        n_ok[3] += 1;
                    }
                }
            }
            fit_us_sum[3] += t0.elapsed().as_micros();
        }
        // Model 4: SmartForecaster
        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        {
            let t0 = Instant::now();
            let mut m = SmartForecaster::new().with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[4] += mae(p, test_v) / scale;
                        n_ok[4] += 1;
                    }
                }
            }
            fit_us_sum[4] += t0.elapsed().as_micros();
        }
    }

    let mut mase_mean = [0.0f64; N_MODELS];
    let mut total_s = [0.0f64; N_MODELS];
    for i in 0..N_MODELS {
        mase_mean[i] = if n_ok[i] > 0 {
            mase_sum[i] / n_ok[i] as f64
        } else {
            f64::NAN
        };
        total_s[i] = fit_us_sum[i] as f64 / 1_000_000.0;
    }

    Some(DatasetResult {
        name: ds.name,
        n_series,
        mase_mean,
        n_ok,
        total_s,
    })
}

fn geometric_mean(xs: &[f64]) -> f64 {
    let xs: Vec<f64> = xs
        .iter()
        .filter(|x| x.is_finite() && **x > 0.0)
        .copied()
        .collect();
    if xs.is_empty() {
        return f64::NAN;
    }
    let log_sum: f64 = xs.iter().map(|x| x.ln()).sum();
    (log_sum / xs.len() as f64).exp()
}

fn main() {
    let sample_per: usize = std::env::var("SAMPLE_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    eprintln!("fev-style benchmark — sample={} series/dataset", sample_per);
    let mut results: Vec<DatasetResult> = Vec::new();
    for ds in DATASETS.iter() {
        if let Some(r) = run_dataset(ds, sample_per) {
            results.push(r);
        }
    }

    println!("\n=== fev-style MASE per dataset ===");
    print!("{:<20}{:>8}", "dataset", "n");
    for name in MODEL_NAMES {
        print!("{:>18}", name);
    }
    println!();
    for r in &results {
        print!("{:<20}{:>8}", r.name, r.n_series);
        for i in 0..N_MODELS {
            print!("{:>18.3}", r.mase_mean[i]);
        }
        println!();
    }

    println!("\n=== geometric mean MASE across datasets ===");
    print!("{:<20}", "");
    for name in MODEL_NAMES {
        print!("{:>18}", name);
    }
    println!();
    print!("{:<20}", "geomean MASE");
    for i in 0..N_MODELS {
        let vals: Vec<f64> = results.iter().map(|r| r.mase_mean[i]).collect();
        print!("{:>18.4}", geometric_mean(&vals));
    }
    println!();

    println!("\n=== total fit time per model (s) ===");
    for i in 0..N_MODELS {
        let total: f64 = results.iter().map(|r| r.total_s[i]).sum();
        println!("  {:<20}{:>10.1}s", MODEL_NAMES[i], total);
    }
}
