//! Distributional-shell (skaters/laplace-style) accuracy benchmark on M5.
//!
//! Runs four `LaplaceForecaster` configurations (v0.12 alpha,
//! `distributional` feature) — plain 3-leaf, +Holt, +seasonal7,
//! +Holt+seasonal7 — against the point-forecast stack (`AutoETS`,
//! `AutoTheta`) on the M5 top-1000 retail panel. Reports:
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

#[derive(Clone, Copy)]
struct Characteristics {
    /// R² of a linear fit y = a + b·t on the training window. In [0, 1].
    trend_strength: f64,
    /// R² of a phase-mean fit at period 7 on the training window. In [0, 1].
    seasonality_strength: f64,
    /// |Pearson correlation between y_t and y_{t-1}|. In [0, 1].
    acf1: f64,
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

    // Slot layout stable so the pairwise matrix and summary lines match:
    //   0=AutoETS, 1=AutoTheta, 2=Laplace, 3=Laplace+H, 4=Laplace+AR2,
    //   5=Laplace+S7, 6=Laplace+AR2+S7, 7=+Cal, 8=+YJ, 9=+YJ+Cal,
    //   10=Laplace+Pops, 11=Laplace+Pops+AR2+S7,
    //   12=Laplace+FracDiff, 13=Laplace+OU, 14=Laplace+AR2+S7+FracDiff+OU.
    const N_MODELS: usize = 16;
    let labels: [&str; N_MODELS] = [
        "AutoETS",
        "AutoTheta",
        "Laplace",
        "Laplace+H",
        "Laplace+AR2",
        "Laplace+S7",
        "Laplace+AR2+S7",
        "Laplace+AR2+S7+Cal",
        "Laplace+AR2+S7+YJ",
        "Laplace+AR2+S7+YJ+Cal",
        "Laplace+Pops",
        "Laplace+Pops+AR2+S7",
        "Laplace+FD",
        "Laplace+OU",
        "Laplace+AR2+S7+FD+OU",
        "Laplace+AR2+S7,30+FD+OU",
    ];
    let mut results: Vec<Vec<SeriesResult>> = (0..N_MODELS).map(|_| Vec::new()).collect();
    let mut characteristics: Vec<Characteristics> = Vec::new();

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

        // Record series characteristics before any model runs; assumes all
        // model fits below succeed (they do on this dataset). If a future
        // dataset breaks that we'd need per-slot per-series alignment.
        let chars = compute_characteristics(&train_values, 7);

        let n0 = results[0].len();
        if let Some(r) = run_point(&mut AutoETS::new(), &train_ts, test_values) {
            results[0].push(r);
        }
        if let Some(r) = run_point(&mut AutoTheta::new(), &train_ts, test_values) {
            results[1].push(r);
        }

        #[cfg(feature = "distributional")]
        {
            let cfgs: [(usize, LaplaceForecaster); 14] = [
                (2, LaplaceForecaster::new()),
                (3, LaplaceForecaster::new().with_holt_defaults()),
                (4, LaplaceForecaster::new().with_ar2_defaults()),
                (5, LaplaceForecaster::new().with_seasonal(7)),
                (
                    6,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7),
                ),
                (
                    7,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7)
                        .with_calibration(),
                ),
                (
                    8,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7)
                        .with_yeo_johnson_mle(),
                ),
                (
                    9,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7)
                        .with_yeo_johnson_mle()
                        .with_calibration(),
                ),
                (10, LaplaceForecaster::new().with_populations()),
                (
                    11,
                    LaplaceForecaster::new()
                        .with_populations()
                        .with_ar2_defaults()
                        .with_seasonal(7),
                ),
                (12, LaplaceForecaster::new().with_fractional_diff_defaults()),
                (13, LaplaceForecaster::new().with_ou_defaults()),
                (
                    14,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal(7)
                        .with_fractional_diff_defaults()
                        .with_ou_defaults(),
                ),
                (
                    15,
                    LaplaceForecaster::new()
                        .with_ar2_defaults()
                        .with_seasonal_multi(&[7, 30])
                        .with_fractional_diff_defaults()
                        .with_ou_defaults(),
                ),
            ];
            for (slot, model) in cfgs {
                if let Some(r) = run_laplace(model, &train_ts, test_values) {
                    results[slot].push(r);
                }
            }
        }

        // Only record characteristics if every model produced a matching row
        // (keeps `characteristics[i]` aligned with `results[slot][i]`).
        let all_ok = (0..N_MODELS).all(|s| results[s].len() == n0 + 1);
        if all_ok {
            characteristics.push(chars);
        } else {
            for s in 0..N_MODELS {
                results[s].truncate(n0);
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
        .map(|(name, r)| summarize(*name, r))
        .collect();

    print_summary(&summaries);
    print_pairwise(&labels, &results);
    print_sliced_analysis(&labels, &results, &characteristics);
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

fn compute_characteristics(train: &[f64], period: usize) -> Characteristics {
    let n = train.len();
    if n < 2 {
        return Characteristics {
            trend_strength: 0.0,
            seasonality_strength: 0.0,
            acf1: 0.0,
        };
    }
    let mean_y: f64 = train.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = train.iter().map(|y| (y - mean_y).powi(2)).sum();

    // Trend strength: R² of the linear fit y ~ t.
    let t_mean = (n - 1) as f64 / 2.0;
    let (mut sum_ty, mut sum_tt) = (0.0, 0.0);
    for (t, y) in train.iter().enumerate() {
        let dt = t as f64 - t_mean;
        sum_ty += dt * (y - mean_y);
        sum_tt += dt * dt;
    }
    let slope = if sum_tt > 0.0 { sum_ty / sum_tt } else { 0.0 };
    let intercept = mean_y - slope * t_mean;
    let ss_res_trend: f64 = train
        .iter()
        .enumerate()
        .map(|(t, y)| (y - (intercept + slope * t as f64)).powi(2))
        .sum();
    let trend_strength = if ss_tot > 0.0 {
        (1.0 - ss_res_trend / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Seasonality strength: R² of the phase-mean fit at `period`.
    let period = period.max(1);
    let mut phase_sum = vec![0.0f64; period];
    let mut phase_count = vec![0usize; period];
    for (i, &y) in train.iter().enumerate() {
        phase_sum[i % period] += y;
        phase_count[i % period] += 1;
    }
    let phase_mean: Vec<f64> = phase_sum
        .iter()
        .zip(phase_count.iter())
        .map(|(s, &c)| if c > 0 { s / c as f64 } else { mean_y })
        .collect();
    let ss_res_season: f64 = train
        .iter()
        .enumerate()
        .map(|(i, y)| (y - phase_mean[i % period]).powi(2))
        .sum();
    let seasonality_strength = if ss_tot > 0.0 {
        (1.0 - ss_res_season / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // |AR(1) autocorrelation|.
    let mut num = 0.0f64;
    for i in 1..n {
        num += (train[i - 1] - mean_y) * (train[i] - mean_y);
    }
    let acf1 = if ss_tot > 0.0 {
        (num / ss_tot).clamp(-1.0, 1.0).abs()
    } else {
        0.0
    };

    Characteristics {
        trend_strength,
        seasonality_strength,
        acf1,
    }
}

fn print_sliced_analysis(
    labels: &[&str],
    results: &[Vec<SeriesResult>],
    characteristics: &[Characteristics],
) {
    let n = characteristics.len();
    if n < 6 {
        return;
    }
    let n_models = labels.len();
    // Sanity: every model has one row per series.
    for r in results {
        assert_eq!(r.len(), n, "results/characteristics length mismatch");
    }
    let median_mae = |indices: &[usize], slot: usize| -> f64 {
        let mut xs: Vec<f64> = indices.iter().map(|&i| results[slot][i].mae).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = xs.len();
        if m == 0 {
            f64::NAN
        } else if m % 2 == 1 {
            xs[m / 2]
        } else {
            0.5 * (xs[m / 2 - 1] + xs[m / 2])
        }
    };
    let winrate_vs_plain = |indices: &[usize], slot: usize| -> f64 {
        let m = indices.len();
        if m == 0 {
            return f64::NAN;
        }
        let wins = indices
            .iter()
            .filter(|&&i| results[slot][i].mae < results[2][i].mae)
            .count();
        wins as f64 / m as f64
    };

    let dims: [(&str, Box<dyn Fn(&Characteristics) -> f64>); 3] = [
        (
            "trend_strength",
            Box::new(|c: &Characteristics| c.trend_strength),
        ),
        (
            "seasonality_strength",
            Box::new(|c: &Characteristics| c.seasonality_strength),
        ),
        ("acf1", Box::new(|c: &Characteristics| c.acf1)),
    ];

    println!(
        "\n=== Residual slicing — median MAE and Laplace-variant winrate vs. plain Laplace ==="
    );
    for (dim_name, key) in &dims {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            key(&characteristics[i])
                .partial_cmp(&key(&characteristics[j]))
                .unwrap()
        });
        let third = n / 3;
        let buckets: [(&str, &[usize]); 3] = [
            ("low", &order[..third]),
            ("mid", &order[third..2 * third]),
            ("high", &order[2 * third..]),
        ];
        println!("\n[{}] (tercile splits)", dim_name);
        // Header: bucket, n, then median MAE per model.
        print!("{:<8}{:>6}", "bucket", "n");
        for name in labels {
            print!("{:>13}", name);
        }
        // Winrate columns for +H, +S7, +H+S7 vs. plain Laplace (slot 2).
        for name in &labels[3..n_models] {
            print!(
                "{:>14}",
                format!("{} v L", name.trim_start_matches("Laplace+"))
            );
        }
        println!();
        for (bucket_name, indices) in &buckets {
            print!("{:<8}{:>6}", bucket_name, indices.len());
            for slot in 0..n_models {
                print!("{:>13.3}", median_mae(indices, slot));
            }
            for slot in 3..n_models {
                print!("{:>14.3}", winrate_vs_plain(indices, slot));
            }
            println!();
        }
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
    print!("{:<16}", "");
    for name in labels {
        print!("{:>14}", name);
    }
    println!();
    for (i, row_name) in labels.iter().enumerate() {
        print!("{:<16}", row_name);
        for (j, _) in labels.iter().enumerate() {
            if i == j {
                print!("{:>14}", "-");
            } else {
                print!("{:>14.3}", winrate(&results[i], &results[j]));
            }
        }
        println!();
    }
}
