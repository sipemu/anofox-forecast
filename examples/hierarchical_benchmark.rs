//! Evaluate `HierarchicalLaplace` against plain `LaplaceForecaster::auto()`
//! on the short-history fev panels where per-series streaming needs the most
//! help: m1_yearly, m3_yearly, tourism_yearly, m1_quarterly, m3_quarterly.
//!
//! Reports MASE per panel and geomean, plus the per-series MASE gain
//! distribution so we can see whether Hierarchical helps consistently
//! or only on outliers.

use anofox_forecast::core::TimeSeries;
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
    hier_mase: f64,
    plain_fit_s: f64,
    hier_fit_s: f64,
    /// Fraction of series where hier improved over plain.
    hier_win_rate: f64,
}

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
    // Fit both models.
    let t0 = Instant::now();
    let mut plain_forecasters: Vec<(String, LaplaceForecaster)> = Vec::new();
    for (id, ts) in &train_ts_map {
        let mut m = LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(period.max(2));
        if m.fit(ts).is_ok() {
            plain_forecasters.push((id.clone(), m));
        }
    }
    let plain_fit_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut hier = HierarchicalLaplace::new(ds.prior_strength, move || {
        LaplaceForecaster::new()
            .auto()
            .auto_with_seasonal_period(period.max(2))
    });
    for (id, ts) in &train_ts_map {
        let _ = hier.fit_series(id.clone(), ts);
    }
    let _ = hier.finalize(ds.horizon);
    let hier_fit_s = t0.elapsed().as_secs_f64();

    // Compute MASE per model.
    let mut plain_mases = Vec::new();
    let mut hier_mases = Vec::new();
    let mut hier_wins = 0usize;
    let mut n_matched = 0usize;
    for (id, test_v, scale) in &test_map {
        let plain_fc = plain_forecasters
            .iter()
            .find(|(oid, _)| oid == id)
            .and_then(|(_, m)| m.predict(ds.horizon).ok());
        let hier_fc = hier.predict_series(id, ds.horizon).ok();
        let (Some(p), Some(h)) = (plain_fc, hier_fc) else {
            continue;
        };
        if p.primary().len() != test_v.len() || h.primary().len() != test_v.len() {
            continue;
        }
        let pm = mae(p.primary(), test_v) / *scale;
        let hm = mae(h.primary(), test_v) / *scale;
        plain_mases.push(pm);
        hier_mases.push(hm);
        if hm < pm {
            hier_wins += 1;
        }
        n_matched += 1;
    }
    if n_matched == 0 {
        return None;
    }
    let plain_mase = plain_mases.iter().sum::<f64>() / n_matched as f64;
    let hier_mase = hier_mases.iter().sum::<f64>() / n_matched as f64;
    Some(PanelResult {
        name: ds.name,
        n_series: n_matched,
        plain_mase,
        hier_mase,
        plain_fit_s,
        hier_fit_s,
        hier_win_rate: hier_wins as f64 / n_matched as f64,
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
    println!(
        "\n{:<20}{:>8}{:>12}{:>12}{:>10}{:>12}{:>12}",
        "panel", "n", "plain MASE", "hier MASE", "Δ", "hier win %", "hier fit(s)"
    );
    for r in &results {
        let delta = 100.0 * (r.hier_mase - r.plain_mase) / r.plain_mase;
        println!(
            "{:<20}{:>8}{:>12.3}{:>12.3}{:>9.1}%{:>11.1}%{:>12.2}",
            r.name,
            r.n_series,
            r.plain_mase,
            r.hier_mase,
            delta,
            r.hier_win_rate * 100.0,
            r.hier_fit_s
        );
    }
    let plain_gm = geomean(&results.iter().map(|r| r.plain_mase).collect::<Vec<_>>());
    let hier_gm = geomean(&results.iter().map(|r| r.hier_mase).collect::<Vec<_>>());
    println!(
        "\n{:<20}{:>8}{:>12.3}{:>12.3}{:>9.1}%",
        "geomean MASE",
        "-",
        plain_gm,
        hier_gm,
        100.0 * (hier_gm - plain_gm) / plain_gm
    );
}
