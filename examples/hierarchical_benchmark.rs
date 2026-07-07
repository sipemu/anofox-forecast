//! Evaluate `HierarchicalLaplace` against plain `LaplaceForecaster::auto()`
//! on the short-history fev panels where per-series streaming needs the most
//! help: m1_yearly, m3_yearly, tourism_yearly, m1_quarterly, m3_quarterly.
//!
//! Reports MASE per panel and geomean, plus the per-series MASE gain
//! distribution so we can see whether Hierarchical helps consistently
//! or only on outliers.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::laplace::hierarchical::PriorMode;
use anofox_forecast::models::laplace::HierarchicalLaplace;
use anofox_forecast::models::{Forecaster, LaplaceForecaster};

use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

struct Dataset {
    name: &'static str,
    path: &'static str,
    horizon: usize,
    period: usize,
    /// Median expected length — sets prior_strength.
    prior_strength: f64,
}

const DATASETS: &[Dataset] = &[
    Dataset {
        name: "m1_yearly",
        path: "validation/data/m1_yearly.tsf",
        horizon: 6,
        period: 1,
        prior_strength: 30.0,
    },
    Dataset {
        name: "m3_yearly",
        path: "validation/data/m3_yearly.tsf",
        horizon: 6,
        period: 1,
        prior_strength: 30.0,
    },
    Dataset {
        name: "tourism_yearly",
        path: "validation/data/tourism_yearly.tsf",
        horizon: 4,
        period: 1,
        prior_strength: 20.0,
    },
    Dataset {
        name: "m1_quarterly",
        path: "validation/data/m1_quarterly.tsf",
        horizon: 8,
        period: 4,
        prior_strength: 50.0,
    },
    Dataset {
        name: "m3_quarterly",
        path: "validation/data/m3_quarterly.tsf",
        horizon: 8,
        period: 4,
        prior_strength: 50.0,
    },
    Dataset {
        name: "m1_monthly",
        path: "validation/data/m1_monthly.tsf",
        horizon: 18,
        period: 12,
        prior_strength: 80.0,
    },
    Dataset {
        name: "m3_monthly",
        path: "validation/data/m3_monthly.tsf",
        horizon: 18,
        period: 12,
        prior_strength: 80.0,
    },
    Dataset {
        name: "cif_2016",
        path: "validation/data/cif_2016.tsf",
        horizon: 12,
        period: 12,
        prior_strength: 60.0,
    },
    Dataset {
        name: "fred_md",
        path: "validation/data/fred_md.tsf",
        horizon: 12,
        period: 12,
        prior_strength: 100.0,
    },
    Dataset {
        name: "hospital",
        path: "validation/data/hospital.tsf",
        horizon: 12,
        period: 12,
        prior_strength: 60.0,
    },
];

fn parse_tsf(path: &str) -> Vec<(String, Vec<f64>)> {
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
        let id = toks[0].to_string();
        let vals_str = toks[toks.len() - 1];
        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if !values.is_empty() {
            series.push((id, values));
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

struct PanelResult {
    name: &'static str,
    n_series: usize,
    plain_mase: f64,
    /// MASE for each prior mode: PanelMean, Cluster, Similarity, Decomposition.
    mode_mases: [f64; 4],
}
const MODE_NAMES: [&str; 4] = ["PanelMean", "Cluster", "Similarity", "Decomposition"];

fn run_dataset(ds: &Dataset, sample: usize) -> Option<PanelResult> {
    let mut kept = parse_tsf(ds.path);
    kept.retain(|(_, v)| v.len() > ds.horizon + 12);
    kept.truncate(sample);
    let n_series = kept.len();
    if n_series < 5 {
        eprintln!("  [{}] too few series ({}); skipping", ds.name, n_series);
        return None;
    }
    eprintln!(
        "  [{}] {} series (H={}, prior={:.0})",
        ds.name, n_series, ds.horizon, ds.prior_strength
    );
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();

    // Build TimeSeries per series (training portion only).
    let mut train_ts_map: Vec<(String, TimeSeries)> = Vec::new();
    let mut test_map: Vec<(String, Vec<f64>, f64)> = Vec::new(); // (id, test_values, scale)
    for (id, values) in &kept {
        let split = values.len() - ds.horizon;
        let train_v = values[..split].to_vec();
        let test_v = values[split..].to_vec();
        let scale = mase_scale(&train_v, ds.period);
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base + Duration::days(i as i64 * 30))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v) {
            Ok(t) => t,
            Err(_) => continue,
        };
        train_ts_map.push((id.clone(), train_ts));
        test_map.push((id.clone(), test_v, scale));
    }

    let period = ds.period;
    // Fit plain baseline.
    let mut plain_forecasters: Vec<(String, LaplaceForecaster)> = Vec::new();
    for (id, ts) in &train_ts_map {
        let mut m = LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(period.max(2));
        if m.fit(ts).is_ok() {
            plain_forecasters.push((id.clone(), m));
        }
    }

    // Fit one HierarchicalLaplace per prior mode.
    let modes = [
        PriorMode::PanelMean,
        PriorMode::Cluster {
            k: ((train_ts_map.len() as f64 / 10.0).sqrt() as usize).max(3),
        },
        PriorMode::Similarity,
        PriorMode::Decomposition,
    ];
    let mut hier_per_mode: Vec<HierarchicalLaplace> = Vec::new();
    for mode in modes.iter() {
        let period_local = period;
        let mut hier = HierarchicalLaplace::new(ds.prior_strength, move || {
            LaplaceForecaster::new()
                .auto()
                .auto_with_seasonal_period(period_local.max(2))
        })
        .with_prior_mode(*mode);
        for (id, ts) in &train_ts_map {
            let _ = hier.fit_series(id.clone(), ts);
        }
        let _ = hier.finalize(ds.horizon);
        hier_per_mode.push(hier);
    }

    // MASE.
    let mut plain_mases = Vec::new();
    let mut mode_mases: [Vec<f64>; 4] = Default::default();
    for (id, test_v, scale) in &test_map {
        let plain_fc = plain_forecasters
            .iter()
            .find(|(oid, _)| oid == id)
            .and_then(|(_, m)| m.predict(ds.horizon).ok());
        let Some(p) = plain_fc else { continue };
        if p.primary().len() != test_v.len() {
            continue;
        }
        plain_mases.push(mae(p.primary(), test_v) / *scale);
        for (mi, h) in hier_per_mode.iter().enumerate() {
            if let Ok(fc) = h.predict_series(id, ds.horizon) {
                if fc.primary().len() == test_v.len() {
                    mode_mases[mi].push(mae(fc.primary(), test_v) / *scale);
                }
            }
        }
    }
    if plain_mases.is_empty() {
        return None;
    }
    let plain_mase = plain_mases.iter().sum::<f64>() / plain_mases.len() as f64;
    let mut mode_means = [0.0f64; 4];
    for (mi, v) in mode_mases.iter().enumerate() {
        mode_means[mi] = if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        };
    }
    Some(PanelResult {
        name: ds.name,
        n_series: plain_mases.len(),
        plain_mase,
        mode_mases: mode_means,
    })
}

fn geomean(xs: &[f64]) -> f64 {
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
    let sample: usize = std::env::var("SAMPLE_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    eprintln!("HierarchicalLaplace vs plain LaplaceForecaster::auto()");
    eprintln!("sample={}", sample);
    let mut results = Vec::new();
    for ds in DATASETS {
        if let Some(r) = run_dataset(ds, sample) {
            results.push(r);
        }
    }
    print!("\n{:<20}{:>8}{:>12}", "panel", "n", "plain");
    for name in &MODE_NAMES {
        print!("{:>14}", name);
    }
    println!();
    for r in &results {
        print!("{:<20}{:>8}{:>12.3}", r.name, r.n_series, r.plain_mase,);
        for (mi, mv) in r.mode_mases.iter().enumerate() {
            let delta = 100.0 * (mv - r.plain_mase) / r.plain_mase;
            let sign = if delta >= 0.0 { '+' } else { '-' };
            print!("{:>8.3}{:>1}{:>4.1}%", mv, sign, delta.abs());
            let _ = mi;
        }
        println!();
    }
    let plain_gm = geomean(&results.iter().map(|r| r.plain_mase).collect::<Vec<_>>());
    let mode_gms: [f64; 4] = std::array::from_fn(|mi| {
        geomean(&results.iter().map(|r| r.mode_mases[mi]).collect::<Vec<_>>())
    });
    print!("\n{:<20}{:>8}{:>12.3}", "geomean MASE", "-", plain_gm);
    for (mi, gm) in mode_gms.iter().enumerate() {
        let delta = 100.0 * (gm - plain_gm) / plain_gm;
        let sign = if delta >= 0.0 { '+' } else { '-' };
        print!("{:>8.3}{:>1}{:>4.1}%", gm, sign, delta.abs());
        let _ = mi;
    }
    println!();
}
