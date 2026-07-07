//! M3 monthly cross-panel benchmark.
//!
//! 1428 monthly series across 6 domains (demographic, micro, macro,
//! industry, finance, other) from the M3 forecasting competition
//! (Makridakis & Hibon, 2000). 18-month held-out competition horizon.
//! Data via the Monash Time Series Forecasting Archive
//! (`m3_monthly_dataset.tsf`, ~336 KB unpacked to `~1 MB`).
//!
//! Answers the cross-panel question: does the α-21 AID-driven Laplace
//! stack help on monthly economic-adjacent data (typically smoother than
//! M4 daily, longer horizon, period 12)?
//!
//! Run: `cargo run --release --features distributional --example skaters_m3_monthly_benchmark`
//! Configure: `SAMPLE_SIZE=500` (default: all 1428).

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;

#[cfg(feature = "distributional")]
use anofox_forecast::models::{DistributionalForecaster, LaplaceForecaster, SmartForecaster};

use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 18;
const MIN_LEN: usize = 60;

struct SeriesResult {
    mae: f64,
    /// MASE = MAE(forecast) / MAE(seasonal_naive on training), the
    /// canonical fev/autogluon and M-Competition scaled point metric.
    mase: f64,
    coverage90: Option<f64>,
    logpdf_mean: Option<f64>,
    fit_us: u128,
}

struct ModelSummary {
    name: &'static str,
    n_ok: usize,
    mae_median: f64,
    mae_mean: f64,
    mase_mean: f64,
    coverage90_mean: Option<f64>,
    logpdf_mean: Option<f64>,
    total_ms: u128,
}

/// Parse the Monash `.tsf` M3 monthly dataset. Returns `(id, values)` pairs.
fn parse_tsf(path: &str) -> Vec<(String, Vec<f64>)> {
    // Monash .tsf is Latin-1 (non-UTF-8) with CRLF terminators. Read as
    // bytes and decode losslessly.
    let bytes = fs::read(path).expect("read tsf");
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
        // Data line: id:start_ts:v1,v2,v3,...
        let mut parts = line.splitn(3, ':');
        let id = match parts.next() {
            Some(s) => s.to_string(),
            None => continue,
        };
        let _start_ts = parts.next();
        let vals_str = match parts.next() {
            Some(s) => s,
            None => continue,
        };
        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if values.len() > HORIZON + 6 {
            series.push((id, values));
        }
    }
    series
}

fn main() {
    let sample_size: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let path =
        std::env::var("DATA_PATH").unwrap_or_else(|_| "validation/data/m3_monthly.tsf".into());

    eprintln!("Parsing {path}...");
    let mut kept = parse_tsf(&path);
    kept.truncate(sample_size);
    eprintln!("Running on {} series (H={})", kept.len(), HORIZON);

    let base_date = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();

    const N_MODELS: usize = 8;
    let labels: [&str; N_MODELS] = [
        "AutoETS",
        "AutoTheta",
        "Laplace",
        "Laplace+S12",
        "Laplace+AR2+S12",
        "Laplace+auto",
        "Laplace+auto_aid",
        "SmartForecaster",
    ];
    let mut results: Vec<Vec<SeriesResult>> = (0..N_MODELS).map(|_| Vec::new()).collect();

    let global_start = Instant::now();
    for (idx, (id, values)) in kept.iter().enumerate() {
        if idx % 100 == 0 {
            eprintln!(
                "[{}/{}] {} — elapsed {:.1}s",
                idx,
                kept.len(),
                id,
                global_start.elapsed().as_secs_f64()
            );
        }
        if values.len() <= HORIZON + 6 {
            continue;
        }
        let split = values.len() - HORIZON;
        let train_values = values[..split].to_vec();
        let test_values = &values[split..];
        if train_values.len() < MIN_LEN {
            continue;
        }
        let stamps: Vec<_> = (0..train_values.len())
            .map(|i| base_date + Duration::days(30 * i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_values.clone()) {
            Ok(ts) => ts,
            Err(_) => continue,
        };
        let scale = mase_scale(&train_values, 12);

        if let Some(r) = run_point(&mut AutoETS::new(), &train_ts, test_values, scale) {
            results[0].push(r);
        }
        if let Some(r) = run_point(&mut AutoTheta::new(), &train_ts, test_values, scale) {
            results[1].push(r);
        }

        #[cfg(feature = "distributional")]
        {
            let cfgs: [(usize, LaplaceForecaster); 4] = [
                (2, LaplaceForecaster::new()),
                (3, LaplaceForecaster::new().with_seasonal(12)),
                (
                    4,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(12),
                ),
                (
                    5,
                    LaplaceForecaster::new()
                        .auto()
                        .auto_with_seasonal_period(12),
                ),
            ];
            for (slot, model) in cfgs {
                if let Some(r) = run_laplace(model, &train_ts, test_values, scale) {
                    results[slot].push(r);
                }
            }
            #[cfg(feature = "postprocess")]
            if let Some(r) = run_laplace(
                LaplaceForecaster::new()
                    .auto_aid()
                    .auto_with_seasonal_period(12),
                &train_ts,
                test_values,
                scale,
            ) {
                results[6].push(r);
            }
            #[cfg(feature = "postprocess")]
            if let Some(r) = run_smart(&train_ts, test_values, scale) {
                results[7].push(r);
            }
        }
    }

    eprintln!(
        "\nBenchmark done in {:.1}s (wall-clock)",
        global_start.elapsed().as_secs_f64()
    );

    let summaries: Vec<ModelSummary> = labels
        .iter()
        .zip(results.iter())
        .map(|(name, r)| summarize(name, r))
        .collect();
    print_summary(&summaries);
    print_pairwise(&labels, &results);
}

fn run_point<F: Forecaster>(
    model: &mut F,
    train: &TimeSeries,
    test: &[f64],
    mase_denom: f64,
) -> Option<SeriesResult> {
    let t0 = Instant::now();
    if model.fit(train).is_err() {
        return None;
    }
    let fit_us = t0.elapsed().as_micros();
    let fc = model.predict(HORIZON).ok()?;
    let point = fc.primary();
    if point.len() != test.len() {
        return None;
    }
    let m = mae(point, test);
    Some(SeriesResult {
        mae: m,
        mase: m / mase_denom,
        coverage90: None,
        logpdf_mean: None,
        fit_us,
    })
}

#[cfg(feature = "distributional")]
fn run_laplace(
    mut model: LaplaceForecaster,
    train: &TimeSeries,
    test: &[f64],
    mase_denom: f64,
) -> Option<SeriesResult> {
    let t0 = Instant::now();
    if model.fit(train).is_err() {
        return None;
    }
    let fit_us = t0.elapsed().as_micros();
    let mixtures = model.forecast_dist(HORIZON).ok()?;
    if mixtures.len() != test.len() {
        return None;
    }
    let point: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
    let mut covered = 0usize;
    let mut logpdfs = Vec::with_capacity(test.len());
    for (m, &y) in mixtures.iter().zip(test.iter()) {
        if y >= m.quantile(0.05) && y <= m.quantile(0.95) {
            covered += 1;
        }
        let lp = m.logpdf(y);
        if lp.is_finite() {
            logpdfs.push(lp);
        }
    }
    let coverage = covered as f64 / test.len() as f64;
    let logpdf_mean = if logpdfs.is_empty() {
        None
    } else {
        Some(logpdfs.iter().sum::<f64>() / logpdfs.len() as f64)
    };
    let m = mae(&point, test);
    Some(SeriesResult {
        mae: m,
        mase: m / mase_denom,
        coverage90: Some(coverage),
        logpdf_mean,
        fit_us,
    })
}

#[cfg(all(feature = "distributional", feature = "postprocess"))]
fn run_smart(train: &TimeSeries, test: &[f64], mase_denom: f64) -> Option<SeriesResult> {
    let t0 = Instant::now();
    let mut m = SmartForecaster::new().with_seasonal_period(12);
    if m.fit(train).is_err() {
        return None;
    }
    let fit_us = t0.elapsed().as_micros();
    let fc = m.predict(HORIZON).ok()?;
    let point = fc.primary();
    if point.len() != test.len() {
        return None;
    }
    let mae_v = mae(point, test);
    Some(SeriesResult {
        mae: mae_v,
        mase: mae_v / mase_denom,
        coverage90: None,
        logpdf_mean: None,
        fit_us,
    })
}

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    let s: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    s / pred.len() as f64
}

/// Seasonal-naive MASE denominator: mean absolute difference between
/// consecutive same-phase observations in the training set. M3 monthly
/// uses period = 12 per fev / M-Competition convention.
fn mase_scale(train: &[f64], period: usize) -> f64 {
    if train.len() <= period {
        return 1.0;
    }
    let n = train.len() - period;
    let sum: f64 = (period..train.len())
        .map(|i| (train[i] - train[i - period]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut xs: Vec<f64> = xs.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn summarize(name: &'static str, r: &[SeriesResult]) -> ModelSummary {
    let n = r.len();
    let maes: Vec<f64> = r.iter().map(|s| s.mae).collect();
    let mean = if n == 0 {
        f64::NAN
    } else {
        maes.iter().sum::<f64>() / n as f64
    };
    let mae_median = median(&maes);
    let mase_mean = if n == 0 {
        f64::NAN
    } else {
        r.iter().map(|s| s.mase).sum::<f64>() / n as f64
    };
    let coverages: Vec<f64> = r.iter().filter_map(|s| s.coverage90).collect();
    let coverage_mean = if coverages.is_empty() {
        None
    } else {
        Some(coverages.iter().sum::<f64>() / coverages.len() as f64)
    };
    let logpdfs: Vec<f64> = r.iter().filter_map(|s| s.logpdf_mean).collect();
    let logpdf_mean = if logpdfs.is_empty() {
        None
    } else {
        Some(logpdfs.iter().sum::<f64>() / logpdfs.len() as f64)
    };
    let total_ms: u128 = r.iter().map(|s| s.fit_us / 1000).sum();
    ModelSummary {
        name,
        n_ok: n,
        mae_median,
        mae_mean: mean,
        mase_mean,
        coverage90_mean: coverage_mean,
        logpdf_mean,
        total_ms,
    }
}

fn print_summary(summaries: &[ModelSummary]) {
    println!("\n=== M3 monthly cross-panel benchmark ===");
    println!(
        "{:<24}{:>5}  {:>12}  {:>12}  {:>12}  {:>15}  {:>12}  {:>12}",
        "model",
        "n",
        "MAE (median)",
        "MAE (mean)",
        "MASE (mean)",
        "cover@90 (mean)",
        "logpdf (avg)",
        "fit time (s)"
    );
    for s in summaries {
        let cov = s
            .coverage90_mean
            .map(|c| format!("{:.3}", c))
            .unwrap_or_else(|| "-".to_string());
        let lp = s
            .logpdf_mean
            .map(|l| format!("{:.3}", l))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<24}{:>5}  {:>12.4}  {:>12.4}  {:>12.3}  {:>15}  {:>12}  {:>12.2}",
            s.name,
            s.n_ok,
            s.mae_median,
            s.mae_mean,
            s.mase_mean,
            cov,
            lp,
            s.total_ms as f64 / 1000.0
        );
    }
}

fn print_pairwise(labels: &[&str], results: &[Vec<SeriesResult>]) {
    println!("\n=== Pairwise MAE win-rate (row beats column, ties excluded) ===");
    print!("{:<24}", "");
    for l in labels {
        print!("{:>15}", l);
    }
    println!();
    let n = results[0].len();
    for i in 0..labels.len() {
        print!("{:<24}", labels[i]);
        for j in 0..labels.len() {
            if i == j {
                print!("{:>15}", "-");
                continue;
            }
            let mi = &results[i];
            let mj = &results[j];
            let k = mi.len().min(mj.len()).min(n);
            let mut wins = 0usize;
            for t in 0..k {
                if mi[t].mae < mj[t].mae {
                    wins += 1;
                }
            }
            let rate = if k > 0 { wins as f64 / k as f64 } else { 0.0 };
            print!("{:>15.3}", rate);
        }
        println!();
    }
}
