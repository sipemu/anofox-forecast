//! Full M5 accuracy of the auto/Smart selectors.
//!
//! Runs on the FULL 30,490-series M5 panel (no intermittency filter —
//! real demand data has 40-70% zeros), and compares just the two
//! selectors we ship for zero-config use:
//!
//! * `LaplaceForecaster::new().auto()` — the α-10 per-leaf selector
//!   inside `LaplaceForecaster`. Does NOT know about intermittency.
//! * `SmartForecaster::new()` — the α-16 cross-family router. Detects
//!   intermittent series via `zero_fraction > 0.4` and routes to the
//!   Croston-flavored intermittent leaf + non-negative clamp.
//!
//! Against AutoETS and AutoTheta.
//!
//! Run: cargo run --release --features distributional --example skaters_m5_full_auto
//! Configure: SAMPLE_SIZE=5000 (default 500) to limit runtime.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::{Forecaster, SmartForecaster};

#[cfg(feature = "distributional")]
use anofox_forecast::models::LaplaceForecaster;

use chrono::{Duration, NaiveDate, TimeZone, Utc};
use std::fs;
use std::time::Instant;

const HORIZON: usize = 28;
const MIN_LEN: usize = 60;

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    if pred.is_empty() {
        return f64::NAN;
    }
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

fn main() {
    let sample_size: usize = std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let path = std::env::var("DATA_PATH").unwrap_or_else(|_| "validation/data/m5_full.csv".into());

    eprintln!("Loading {path}...");
    let content = fs::read_to_string(&path).expect("Failed to read CSV");
    let mut lines = content.lines();
    let header = lines.next().expect("Empty CSV");
    let names: Vec<&str> = header.split(',').skip(1).collect();
    let n_total = names.len();

    let mut timestamps = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(2000); n_total];
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
    eprintln!(
        "Loaded {} series × {} observations",
        n_total,
        timestamps.len()
    );

    let mut kept: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, values) in names.iter().zip(cols) {
        if values.len() >= MIN_LEN {
            kept.push((name.to_string(), values));
        }
    }
    kept.truncate(sample_size);
    eprintln!("Running on {} series (no intermittency filter)", kept.len());

    // Per-series MAE arrays.
    let mut ets_maes = Vec::with_capacity(kept.len());
    let mut theta_maes = Vec::with_capacity(kept.len());
    let mut auto_maes = Vec::with_capacity(kept.len());
    let mut auto_aid_maes = Vec::with_capacity(kept.len());
    let mut smart_maes = Vec::with_capacity(kept.len());
    let mut ets_time_us = 0u128;
    let mut theta_time_us = 0u128;
    let mut auto_time_us = 0u128;
    let mut auto_aid_time_us = 0u128;
    let mut smart_time_us = 0u128;

    // Per-series counts of Smart's AID-driven family picks.
    let mut smart_intermittent_poisson = 0usize;
    let mut smart_intermittent_nb = 0usize;
    let mut smart_intermittent_rectnorm = 0usize;
    let mut smart_intermittent_positive = 0usize;
    let mut smart_regular_count = 0usize;
    let mut smart_regular_positive = 0usize;
    let mut smart_regular_normal = 0usize;
    let mut smart_fallback = 0usize;

    let base_date = timestamps[0];
    let start = Instant::now();

    for (idx, (name, values)) in kept.iter().enumerate() {
        if idx % 100 == 0 {
            eprintln!(
                "[{}/{}] {} — elapsed {:.1}s",
                idx,
                kept.len(),
                name,
                start.elapsed().as_secs_f64()
            );
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
        let t0 = Instant::now();
        let mut m = AutoETS::new();
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(HORIZON) {
                ets_maes.push(mae(fc.primary(), test_values));
            }
        }
        ets_time_us += t0.elapsed().as_micros();

        // AutoTheta
        let t0 = Instant::now();
        let mut m = AutoTheta::new();
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(HORIZON) {
                theta_maes.push(mae(fc.primary(), test_values));
            }
        }
        theta_time_us += t0.elapsed().as_micros();

        // Laplace + auto
        #[cfg(feature = "distributional")]
        {
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new().auto();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(HORIZON) {
                    auto_maes.push(mae(fc.primary(), test_values));
                }
            }
            auto_time_us += t0.elapsed().as_micros();
        }

        // Laplace + auto_aid — α-21 AID-driven distribution-family selector
        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        {
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new().auto_aid();
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(HORIZON) {
                    auto_aid_maes.push(mae(fc.primary(), test_values));
                }
            }
            auto_aid_time_us += t0.elapsed().as_micros();
        }

        // SmartForecaster
        let t0 = Instant::now();
        let mut m = SmartForecaster::new();
        if m.fit(&train_ts).is_ok() {
            if let Ok(fc) = m.predict(HORIZON) {
                smart_maes.push(mae(fc.primary(), test_values));
                use anofox_forecast::models::SelectedFamily as F;
                match m.selected_family() {
                    Some(F::IntermittentPoisson) => smart_intermittent_poisson += 1,
                    Some(F::IntermittentNegBinomial) => smart_intermittent_nb += 1,
                    Some(F::IntermittentRectifiedNormal) => smart_intermittent_rectnorm += 1,
                    Some(F::IntermittentPositive) => smart_intermittent_positive += 1,
                    Some(F::RegularCount) => smart_regular_count += 1,
                    Some(F::RegularPositive) => smart_regular_positive += 1,
                    Some(F::RegularNormal) => smart_regular_normal += 1,
                    Some(F::Fallback) => smart_fallback += 1,
                    Some(F::AutoETSStructural) | Some(F::AutoThetaShortHistory) => {}
                    None => {}
                }
            }
        }
        smart_time_us += t0.elapsed().as_micros();
    }

    let total_wall = start.elapsed().as_secs_f64();
    eprintln!("\nDone in {:.1}s", total_wall);

    let summary = |label: &str, mut xs: Vec<f64>, time_us: u128| {
        let n = xs.len();
        let mean: f64 = xs.iter().sum::<f64>() / n.max(1) as f64;
        let med = median(&mut xs);
        let time_s = time_us as f64 / 1_000_000.0;
        println!(
            "{:<28} n={:>6}  MAE(med)={:>7.3}  MAE(mean)={:>8.3}  fit={:>7.1}s",
            label, n, med, mean, time_s
        );
    };

    println!(
        "\n=== Full M5 (or DATA_PATH) — auto/Smart accuracy on {} sampled series ===",
        kept.len()
    );
    summary("AutoETS", ets_maes.clone(), ets_time_us);
    summary("AutoTheta", theta_maes.clone(), theta_time_us);
    #[cfg(feature = "distributional")]
    summary("Laplace+auto", auto_maes.clone(), auto_time_us);
    #[cfg(all(feature = "distributional", feature = "postprocess"))]
    summary("Laplace+auto_aid", auto_aid_maes.clone(), auto_aid_time_us);
    summary("SmartForecaster", smart_maes.clone(), smart_time_us);

    // Winrate: Laplace/Smart vs AutoETS on matched series.
    let n_match = ets_maes
        .len()
        .min(auto_maes.len())
        .min(auto_aid_maes.len().max(auto_maes.len()))
        .min(smart_maes.len());
    if n_match > 0 {
        let auto_wins = (0..n_match).filter(|&i| auto_maes[i] < ets_maes[i]).count();
        let smart_wins = (0..n_match)
            .filter(|&i| smart_maes[i] < ets_maes[i])
            .count();
        println!(
            "\nPairwise winrate vs. AutoETS (on {} matched series):",
            n_match
        );
        println!(
            "  Laplace+auto:    {:.3}",
            auto_wins as f64 / n_match as f64
        );
        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        if auto_aid_maes.len() >= n_match {
            let aid_wins = (0..n_match)
                .filter(|&i| auto_aid_maes[i] < ets_maes[i])
                .count();
            println!("  Laplace+auto_aid:{:.3}", aid_wins as f64 / n_match as f64);
        }
        println!(
            "  SmartForecaster: {:.3}",
            smart_wins as f64 / n_match as f64
        );
    }

    println!(
        "\nSmartForecaster routing (AID-driven):\n\
         \x20 Intermittent+Poisson    = {}\n\
         \x20 Intermittent+NegBinom   = {}\n\
         \x20 Intermittent+RectNormal = {}\n\
         \x20 Intermittent+Positive   = {}\n\
         \x20 Regular+Count           = {}\n\
         \x20 Regular+Positive        = {}\n\
         \x20 Regular+Normal          = {}\n\
         \x20 Fallback                = {}",
        smart_intermittent_poisson,
        smart_intermittent_nb,
        smart_intermittent_rectnorm,
        smart_intermittent_positive,
        smart_regular_count,
        smart_regular_positive,
        smart_regular_normal,
        smart_fallback,
    );
}
