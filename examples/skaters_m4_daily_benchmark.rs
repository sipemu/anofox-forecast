//! Cross-panel benchmark on M4 daily.
//!
//! Same design as `examples/skaters_m5_benchmark.rs`, retargeted to the
//! M4 daily competition subset. Answers the cross-panel generalization
//! question: does the M5-optimized best config (`Laplace + AR2 + S7 +
//! FD + OU`) generalize to a different panel type?
//!
//! M4 daily is 4,227 non-price economic-adjacent series with a 14-day
//! held-out competition horizon. Weekly seasonality is present but
//! weaker than M5 retail. Series are longer (median ~2,900 obs vs. M5's
//! ~1,900).
//!
//! Run: cargo run --release --features distributional --example skaters_m4_daily_benchmark
//! Configure: SAMPLE_SIZE=500 (default 100) for a larger run.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;

#[cfg(feature = "distributional")]
use anofox_forecast::models::{DistributionalForecaster, LaplaceForecaster, SmartForecaster};

use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const HORIZON: usize = 14;
const MIN_LEN: usize = 200;

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

    let train_json = fs::read_to_string("validation/data/m4_daily_train.json")
        .expect("read m4_daily_train.json");
    let test_json =
        fs::read_to_string("validation/data/m4_daily_test.json").expect("read m4_daily_test.json");
    let train: HashMap<String, Vec<f64>> =
        serde_json::from_str(&train_json).expect("parse train JSON");
    let test: HashMap<String, Vec<f64>> =
        serde_json::from_str(&test_json).expect("parse test JSON");

    eprintln!("Loaded {} training series", train.len());

    // Sort ids for reproducibility, keep only sufficiently long series.
    let mut ids: Vec<&String> = train.keys().collect();
    ids.sort();
    let ids: Vec<&String> = ids
        .into_iter()
        .filter(|id| {
            train.get(*id).is_some_and(|v| v.len() >= MIN_LEN)
                && test.get(*id).is_some_and(|v| v.len() >= HORIZON)
        })
        .take(sample_size)
        .collect();
    eprintln!("Running benchmark on {} series", ids.len());

    let base_date = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();

    // Slot layout — mirror of the M5 benchmark's Laplace variants for
    // easy cross-panel comparison. α-21/α-22 additions: auto_aid, Smart.
    const N_MODELS: usize = 10;
    let labels: [&str; N_MODELS] = [
        "AutoETS",
        "AutoTheta",
        "Laplace",
        "Laplace+AR2",
        "Laplace+S7",
        "Laplace+AR2+S7",
        "Laplace+AR2+S7+FD+OU",
        "Laplace+auto",
        "Laplace+auto_aid",
        "SmartForecaster",
    ];
    let mut results: Vec<Vec<SeriesResult>> = (0..N_MODELS).map(|_| Vec::new()).collect();

    let global_start = Instant::now();

    for (idx, id) in ids.iter().enumerate() {
        if idx % 25 == 0 {
            eprintln!("[{}/{}] {}", idx, ids.len(), id);
        }
        let train_values = train.get(*id).unwrap().clone();
        let test_values = test.get(*id).unwrap();
        if train_values.len() < MIN_LEN {
            continue;
        }

        let stamps: Vec<_> = (0..train_values.len())
            .map(|i| base_date + Duration::days(i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_values.clone()) {
            Ok(ts) => ts,
            Err(_) => continue,
        };

        if let Some(r) = run_point(&mut AutoETS::new(), &train_ts, test_values) {
            results[0].push(r);
        }
        if let Some(r) = run_point(&mut AutoTheta::new(), &train_ts, test_values) {
            results[1].push(r);
        }

        #[cfg(feature = "distributional")]
        {
            let cfgs: [(usize, LaplaceForecaster); 6] = [
                (2, LaplaceForecaster::new()),
                (3, LaplaceForecaster::new().with_ar2_defaults()),
                (4, LaplaceForecaster::new().with_seasonal(7)),
                (
                    5,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7),
                ),
                (
                    6,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7)
                        .with_fractional_diff_defaults()
                        .with_ou_defaults(),
                ),
                (7, LaplaceForecaster::new().auto()),
            ];
            for (slot, model) in cfgs {
                if let Some(r) = run_laplace(model, &train_ts, test_values) {
                    results[slot].push(r);
                }
            }

            #[cfg(feature = "postprocess")]
            if let Some(r) =
                run_laplace(LaplaceForecaster::new().auto_aid(), &train_ts, test_values)
            {
                results[8].push(r);
            }

            #[cfg(feature = "postprocess")]
            if let Some(r) = run_smart(&train_ts, test_values) {
                results[9].push(r);
            }
        }
    }

    let total_wall = global_start.elapsed();
    eprintln!(
        "\nBenchmark done in {:.1}s (wall-clock)",
        total_wall.as_secs_f64()
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
fn run_laplace(
    mut model: LaplaceForecaster,
    train: &TimeSeries,
    test: &[f64],
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
    let mae = mae(&point, test);
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
    Some(SeriesResult {
        mae,
        coverage90: Some(coverage),
        logpdf_mean,
        fit_us,
    })
}

#[cfg(all(feature = "distributional", feature = "postprocess"))]
fn run_smart(train: &TimeSeries, test: &[f64]) -> Option<SeriesResult> {
    let t0 = Instant::now();
    let mut m = SmartForecaster::new();
    if m.fit(train).is_err() {
        return None;
    }
    let fit_us = t0.elapsed().as_micros();
    let fc = m.predict(HORIZON).ok()?;
    let point = fc.primary();
    if point.len() != test.len() {
        return None;
    }
    Some(SeriesResult {
        mae: mae(point, test),
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
    println!("\n=== M4 daily cross-panel benchmark ===");
    println!(
        "{:<22}{:>8}{:>14}{:>14}{:>16}{:>14}{:>16}",
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
            "{:<22}{:>8}{:>14.4}{:>14.4}{:>16}{:>14}{:>16.2}",
            s.name,
            s.n_ok,
            s.mae_median,
            s.mae_mean,
            cov,
            lp,
            s.total_ms as f64 / 1000.0
        );
    }
}

fn print_pairwise(labels: &[&str], results: &[Vec<SeriesResult>]) {
    let n = results.iter().map(|r| r.len()).min().unwrap_or(0);
    if n == 0 {
        return;
    }
    let winrate = |a: &[SeriesResult], b: &[SeriesResult]| -> f64 {
        let wins = (0..n).filter(|&i| a[i].mae < b[i].mae).count();
        wins as f64 / n as f64
    };
    println!(
        "\n=== Pairwise MAE win-rate on {} matched series (row beats column) ===",
        n
    );
    print!("{:<22}", "");
    for name in labels {
        print!("{:>16}", name);
    }
    println!();
    for (i, row_name) in labels.iter().enumerate() {
        print!("{:<22}", row_name);
        for (j, _) in labels.iter().enumerate() {
            if i == j {
                print!("{:>16}", "-");
            } else {
                print!("{:>16.3}", winrate(&results[i], &results[j]));
            }
        }
        println!();
    }
}
