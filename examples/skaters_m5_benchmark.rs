//! Distributional-shell (skaters/laplace-style) accuracy benchmark on M5.
//!
//! Runs `LaplaceForecaster` (v0.12 alpha, `distributional` feature) against
//! the point-forecast stack (`AutoETS`, `AutoTheta`) on the M5 top-1000
//! retail panel. Reports:
//!
//! * per-series MAE on a 28-day held-out window (median + winrates)
//! * empirical 90% interval coverage for the distributional model
//!   (target: 0.90 if calibrated)
//! * wall-clock per model
//!
//! Non-intermittent series only (≥30% non-zero obs) — the shell targets
//! non-price economic-style series, and M5 retail has heavy zero
//! inflation that isn't in-scope for the distributional forecaster.
//!
//! Run: cargo run --release --features distributional --example skaters_m5_benchmark
//! Configure: SAMPLE_SIZE=1000 (default 100) for the full run.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;

#[cfg(feature = "distributional")]
use anofox_forecast::models::{DistributionalForecaster, LaplaceForecaster};

use chrono::{Duration, NaiveDate, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 28;
const MIN_NONZERO_FRAC: f64 = 0.30;
const MIN_LEN: usize = 300;

struct SeriesResult {
    mae: f64,
    coverage90: Option<f64>,
    logpdf_mean: Option<f64>,
    fit_us: u128,
}

struct ModelSummary {
    name: &'static str,
    n_ok: usize,
    mae_median: f64,
    mae_mean: f64,
    coverage90_mean: Option<f64>,
    logpdf_mean: Option<f64>,
    total_ms: u128,
}

fn main() {
    let sample_size: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let path = "validation/data/m5_top1000.csv";
    eprintln!("Loading {path}...");
    let content = fs::read_to_string(path).expect("Failed to read M5 CSV");
    let mut lines = content.lines();
    let header = lines.next().expect("Empty CSV");
    let series_names: Vec<&str> = header.split(',').skip(1).collect();
    let n_series_total = series_names.len();

    // Parse rows.
    let mut timestamps = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(2000); n_series_total];
    for line in lines {
        let mut parts = line.split(',');
        let date_str = parts.next().expect("row missing date");
        let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .expect("bad date")
            .and_hms_opt(0, 0, 0)
            .unwrap();
        timestamps.push(Utc.from_utc_datetime(&date));
        for (i, tok) in parts.enumerate() {
            let v: f64 = tok.parse().unwrap_or(0.0);
            cols[i].push(v);
        }
    }
    let n_obs = timestamps.len();
    eprintln!(
        "Loaded {} series × {} daily observations",
        n_series_total, n_obs
    );

    // Filter: non-intermittent + long enough.
    let mut kept: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, values) in series_names.iter().zip(cols.into_iter()) {
        if values.len() < MIN_LEN {
            continue;
        }
        let nz = values.iter().filter(|&&v| v > 0.0).count() as f64 / values.len() as f64;
        if nz < MIN_NONZERO_FRAC {
            continue;
        }
        kept.push((name.to_string(), values));
    }
    eprintln!(
        "Non-intermittent survivors: {} (≥{:.0}% non-zero, ≥{} obs)",
        kept.len(),
        MIN_NONZERO_FRAC * 100.0,
        MIN_LEN
    );

    // Take the first `sample_size` for the run.
    kept.truncate(sample_size);
    eprintln!("Running benchmark on {} series", kept.len());

    let base_date = timestamps[0];
    let mut ets_results: Vec<SeriesResult> = Vec::new();
    let mut theta_results: Vec<SeriesResult> = Vec::new();
    #[cfg(feature = "distributional")]
    let mut laplace_results: Vec<SeriesResult> = Vec::new();

    let global_start = Instant::now();

    for (idx, (name, values)) in kept.iter().enumerate() {
        if idx % 25 == 0 {
            eprintln!("[{}/{}] {}", idx, kept.len(), name);
        }
        if values.len() <= HORIZON + 20 {
            continue;
        }
        let split = values.len() - HORIZON;
        let train_values = values[..split].to_vec();
        let test_values = &values[split..];

        let stamps: Vec<_> = (0..train_values.len())
            .map(|i| base_date + Duration::days(i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_values.clone()) {
            Ok(ts) => ts,
            Err(_) => continue,
        };

        // AutoETS
        if let Some(r) = run_point(&mut AutoETS::new(), &train_ts, test_values) {
            ets_results.push(r);
        }

        // AutoTheta
        if let Some(r) = run_point(&mut AutoTheta::new(), &train_ts, test_values) {
            theta_results.push(r);
        }

        // LaplaceForecaster (distributional-only branch)
        #[cfg(feature = "distributional")]
        if let Some(r) = run_laplace(&train_ts, test_values) {
            laplace_results.push(r);
        }
    }

    let total_wall = global_start.elapsed();
    eprintln!(
        "\nBenchmark done in {:.1}s (wall-clock)",
        total_wall.as_secs_f64()
    );

    let mut summaries = vec![
        summarize("AutoETS", &ets_results),
        summarize("AutoTheta", &theta_results),
    ];
    #[cfg(feature = "distributional")]
    summaries.push(summarize("LaplaceForecaster", &laplace_results));

    print_summary(&summaries);

    // Win-rate matrix: pairwise, per-series, on the intersection.
    #[cfg(feature = "distributional")]
    print_pairwise(&ets_results, &theta_results, &laplace_results);
}

fn run_point<F: Forecaster>(
    model: &mut F,
    train: &TimeSeries,
    test: &[f64],
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
    let mae = mae(point, test);
    Some(SeriesResult {
        mae,
        coverage90: None,
        logpdf_mean: None,
        fit_us,
    })
}

#[cfg(feature = "distributional")]
fn run_laplace(train: &TimeSeries, test: &[f64]) -> Option<SeriesResult> {
    let t0 = Instant::now();
    let mut model = LaplaceForecaster::new();
    if model.fit(train).is_err() {
        return None;
    }
    let fit_us = t0.elapsed().as_micros();
    let mixtures = model.forecast_dist(HORIZON).ok()?;
    if mixtures.len() != test.len() {
        return None;
    }
    let point: Vec<f64> = mixtures.iter().map(|m| m.mean()).collect();
    let mae = mae(&point, test);

    // 90% coverage: y in [q(0.05), q(0.95)]?
    let mut covered = 0usize;
    let mut logpdfs = Vec::with_capacity(test.len());
    for (m, &y) in mixtures.iter().zip(test.iter()) {
        let lo = m.quantile(0.05);
        let hi = m.quantile(0.95);
        if y >= lo && y <= hi {
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
    Some(SeriesResult {
        mae,
        coverage90: Some(coverage),
        logpdf_mean,
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

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn summarize(name: &'static str, results: &[SeriesResult]) -> ModelSummary {
    let n_ok = results.len();
    let mut maes: Vec<f64> = results.iter().map(|r| r.mae).collect();
    let mae_mean = if n_ok == 0 {
        f64::NAN
    } else {
        maes.iter().sum::<f64>() / n_ok as f64
    };
    let mae_median = median(&mut maes);
    let covs: Vec<f64> = results.iter().filter_map(|r| r.coverage90).collect();
    let coverage90_mean = if covs.is_empty() {
        None
    } else {
        Some(covs.iter().sum::<f64>() / covs.len() as f64)
    };
    let lps: Vec<f64> = results.iter().filter_map(|r| r.logpdf_mean).collect();
    let logpdf_mean = if lps.is_empty() {
        None
    } else {
        Some(lps.iter().sum::<f64>() / lps.len() as f64)
    };
    let total_ms: u128 = results.iter().map(|r| r.fit_us).sum::<u128>() / 1_000;
    ModelSummary {
        name,
        n_ok,
        mae_median,
        mae_mean,
        coverage90_mean,
        logpdf_mean,
        total_ms,
    }
}

fn print_summary(summaries: &[ModelSummary]) {
    println!("\n=== Skaters/Laplace shell vs. anofox point stack — M5 top-1000 ===");
    println!(
        "{:<22}{:>10}{:>14}{:>14}{:>16}{:>14}{:>16}",
        "model",
        "n",
        "MAE (median)",
        "MAE (mean)",
        "cover@90 (mean)",
        "logpdf (avg)",
        "fit time (s)"
    );
    for s in summaries {
        let cov = s
            .coverage90_mean
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "-".to_string());
        let lp = s
            .logpdf_mean
            .map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{:<22}{:>10}{:>14.4}{:>14.4}{:>16}{:>14}{:>16.2}",
            s.name,
            s.n_ok,
            s.mae_median,
            s.mae_mean,
            cov,
            lp,
            s.total_ms as f64 / 1_000.0
        );
    }
}

#[cfg(feature = "distributional")]
fn print_pairwise(ets: &[SeriesResult], theta: &[SeriesResult], laplace: &[SeriesResult]) {
    let n = ets.len().min(theta.len()).min(laplace.len());
    if n == 0 {
        return;
    }
    let winrate = |a: &[SeriesResult], b: &[SeriesResult]| -> f64 {
        let mut wins = 0usize;
        for i in 0..n {
            if a[i].mae < b[i].mae {
                wins += 1;
            }
        }
        wins as f64 / n as f64
    };
    println!(
        "\n=== Pairwise MAE win-rate on {} matched series (row beats column) ===",
        n
    );
    println!(
        "{:<22}{:>14}{:>14}{:>18}",
        "", "AutoETS", "AutoTheta", "LaplaceForecaster"
    );
    println!(
        "{:<22}{:>14}{:>14.3}{:>18.3}",
        "AutoETS",
        "-",
        winrate(ets, theta),
        winrate(ets, laplace)
    );
    println!(
        "{:<22}{:>14.3}{:>14}{:>18.3}",
        "AutoTheta",
        winrate(theta, ets),
        "-",
        winrate(theta, laplace)
    );
    println!(
        "{:<22}{:>14.3}{:>14.3}{:>18}",
        "LaplaceForecaster",
        winrate(laplace, ets),
        winrate(laplace, theta),
        "-"
    );
}
