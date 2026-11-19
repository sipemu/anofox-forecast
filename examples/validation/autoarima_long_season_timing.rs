//! Issue #250 timing harness: seasonal `AutoARIMA` with a long period on the
//! last 550 values of M4 Weekly W118. Run with
//! `cargo run --release --example autoarima_long_season_timing -- <period> [flags]`,
//! flags: `approx`, `budget=<seconds>`, `maxmodels=<n>`.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::time::Instant;

const DATA: &str = include_str!("../../tests/data/m4_weekly_w118_last550.csv");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let period: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(30);
    let flags: Vec<&str> = args.iter().skip(2).map(String::as_str).collect();

    let y: Vec<f64> = DATA.lines().map(|l| l.trim().parse().unwrap()).collect();
    let start = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let times = (0..y.len())
        .map(|i| start + Duration::days(7 * i as i64))
        .collect();
    let series = TimeSeries::univariate(times, y).unwrap();

    let mut config = AutoARIMAConfig::default();
    if period > 1 {
        config = config.with_seasonal_period(period);
    }
    for flag in &flags {
        if *flag == "approx" {
            config = config.with_approximation(true);
        } else if let Some(v) = flag.strip_prefix("budget=") {
            config =
                config.with_time_budget(std::time::Duration::from_secs_f64(v.parse().unwrap()));
        } else if let Some(v) = flag.strip_prefix("maxmodels=") {
            config = config.with_max_models(v.parse().unwrap());
        }
    }
    let mut model = AutoARIMA::with_config(config);
    let t = Instant::now();
    model.fit(&series).unwrap();
    println!(
        "m={period} {flags:?}: {:.2} s -> {:?}",
        t.elapsed().as_secs_f64(),
        model.selected_full_order().unwrap()
    );
}
