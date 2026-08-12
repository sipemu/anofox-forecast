//! Full validation: compare Rust forecasts against Python statsforecast reference values.
//! Also benchmarks each model for speed comparison.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig};
use anofox_forecast::models::baseline::{Naive, RandomWalkWithDrift, SeasonalNaive};
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, HoltLinearTrend, SimpleExponentialSmoothing,
};
use anofox_forecast::models::intermittent::Croston;
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::theta::{OptimizedTheta, Theta};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::time::Instant;

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

fn make_series(n: usize, trend: f64, seasonal_amp: f64, period: usize, level: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut y = level + trend * i as f64;
            if seasonal_amp > 0.0 && period > 1 {
                y += seasonal_amp * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            }
            y += ((i * 7 + 3) % 11) as f64 * 0.3 - 1.5;
            y
        })
        .collect()
}

fn load_reference() -> HashMap<String, serde_json::Value> {
    let path = "validation/data/statsforecast_reference.json";
    let content = std::fs::read_to_string(path)
        .expect("Run validation/full_comparison.py first to generate reference values");
    serde_json::from_str(&content).unwrap()
}

fn mad(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / n as f64
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

#[test]
fn full_comparison_and_benchmark() {
    let reference = load_reference();
    let horizon = 12;

    println!("\n========================================================================");
    println!("  Full Validation: Rust anofox-forecast vs Python statsforecast");
    println!("========================================================================\n");

    let series_configs: Vec<(&str, usize, f64, f64, usize, f64)> = vec![
        ("flat", 100, 0.0, 0.0, 1, 50.0),
        ("trend", 100, 0.5, 0.0, 1, 10.0),
        ("seasonal_p12", 120, 0.2, 8.0, 12, 50.0),
        ("seasonal_p7", 140, 0.1, 5.0, 7, 30.0),
    ];

    let mut total_models = 0;
    let mut total_close = 0; // MAD < 1.0
    let mut total_reasonable = 0; // MAD < 5.0
    let mut all_results: Vec<(String, String, f64, f64, f64)> = Vec::new();

    for (name, n, trend, seasonal_amp, period, level) in &series_configs {
        let values = make_series(*n, *trend, *seasonal_amp, *period, *level);
        let timestamps = make_timestamps(*n);
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let ref_series = &reference[*name];
        let ref_models = ref_series["models"].as_object().unwrap();

        println!("--- {} (n={}, period={}) ---", name, n, period);

        // Run each model
        #[allow(clippy::type_complexity)]
        let models: Vec<(&str, Box<dyn FnOnce() -> (Vec<f64>, f64)>)> = {
            let ts = &ts;
            let h = horizon;
            let p = *period;

            #[allow(clippy::type_complexity)]
            let mut m: Vec<(&str, Box<dyn FnOnce() -> (Vec<f64>, f64)>)> = Vec::new();

            // Naive
            if ref_models.contains_key("Naive") {
                let ts_c = ts.clone();
                m.push((
                    "Naive",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = Naive::new();
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // SeasonalNaive
            if ref_models.contains_key("SeasonalNaive") {
                let ts_c = ts.clone();
                m.push((
                    "SeasonalNaive",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = SeasonalNaive::new(p);
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // RandomWalkWithDrift
            if ref_models.contains_key("RandomWalkWithDrift") {
                let ts_c = ts.clone();
                m.push((
                    "RandomWalkWithDrift",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = RandomWalkWithDrift::new();
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // SES
            if ref_models.contains_key("SES") {
                let ts_c = ts.clone();
                m.push((
                    "SES",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = SimpleExponentialSmoothing::with_alpha(0.3, 50.0);
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // Holt
            if ref_models.contains_key("Holt") {
                let ts_c = ts.clone();
                m.push((
                    "Holt",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = HoltLinearTrend::auto();
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // Theta
            if ref_models.contains_key("Theta") {
                let ts_c = ts.clone();
                m.push((
                    "Theta",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = if p > 1 {
                            Theta::seasonal(p)
                        } else {
                            Theta::new()
                        };
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // OptimizedTheta
            if ref_models.contains_key("OptimizedTheta") {
                let ts_c = ts.clone();
                m.push((
                    "OptimizedTheta",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = if p > 1 {
                            OptimizedTheta::seasonal(p)
                        } else {
                            OptimizedTheta::new()
                        };
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // AutoETS
            if ref_models.contains_key("AutoETS") {
                let ts_c = ts.clone();
                m.push((
                    "AutoETS",
                    Box::new(move || {
                        let start = Instant::now();
                        let config = if p > 1 {
                            AutoETSConfig::with_period(p)
                        } else {
                            AutoETSConfig::non_seasonal()
                        };
                        let mut model = AutoETS::with_config(config);
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // AutoARIMA
            if ref_models.contains_key("AutoARIMA") {
                let ts_c = ts.clone();
                m.push((
                    "AutoARIMA",
                    Box::new(move || {
                        let start = Instant::now();
                        let config = AutoARIMAConfig {
                            seasonal_period: p,
                            ..Default::default()
                        };
                        let mut model = AutoARIMA::with_config(config);
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            // MFLES
            if ref_models.contains_key("MFLES") {
                let ts_c = ts.clone();
                m.push((
                    "MFLES",
                    Box::new(move || {
                        let start = Instant::now();
                        let mut model = MFLES::new(vec![p]);
                        model.fit(&ts_c).unwrap();
                        let fc = model.predict(h).unwrap();
                        (
                            fc.primary().to_vec(),
                            start.elapsed().as_secs_f64() * 1000.0,
                        )
                    }),
                ));
            }

            m
        };

        println!(
            "  {:<20} {:>8} {:>8} {:>8}  {:>8}  Match",
            "Model", "MAD", "RMSE", "MaxDiff", "Time(ms)"
        );

        for (model_name, run_fn) in models {
            let (rust_fc, elapsed_ms) = run_fn();

            if let Some(ref_data) = ref_models.get(model_name) {
                if let Some(ref_fc_val) = ref_data.get("forecasts") {
                    let ref_fc: Vec<f64> = ref_fc_val
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect();

                    let mad_val = mad(&rust_fc, &ref_fc);
                    let rmse_val = rmse(&rust_fc, &ref_fc);
                    let max_diff: f64 = rust_fc
                        .iter()
                        .zip(ref_fc.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);

                    let match_level = if mad_val < 0.01 {
                        "EXACT"
                    } else if mad_val < 1.0 {
                        "CLOSE"
                    } else if mad_val < 5.0 {
                        "OK"
                    } else {
                        "DIFF"
                    };

                    total_models += 1;
                    if mad_val < 1.0 {
                        total_close += 1;
                    }
                    if mad_val < 5.0 {
                        total_reasonable += 1;
                    }

                    all_results.push((
                        name.to_string(),
                        model_name.to_string(),
                        mad_val,
                        elapsed_ms,
                        max_diff,
                    ));

                    println!(
                        "  {:<20} {:>8.4} {:>8.4} {:>8.4}  {:>7.3}ms  {}",
                        model_name, mad_val, rmse_val, max_diff, elapsed_ms, match_level
                    );
                }
            }
        }
        println!();
    }

    // Intermittent
    {
        let values = vec![
            0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ];
        let ts = TimeSeries::univariate(make_timestamps(values.len()), values).unwrap();

        println!("--- intermittent (n=60) ---");
        println!(
            "  {:<20} {:>8} {:>8} {:>8}  {:>8}  Match",
            "Model", "MAD", "RMSE", "MaxDiff", "Time(ms)"
        );

        let start = Instant::now();
        let mut model = Croston::new().sba_optimized();
        model.fit(&ts).unwrap();
        let fc = model.predict(horizon).unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let ref_data = &reference["intermittent"]["models"]["CrostonSBA"];
        let ref_fc: Vec<f64> = ref_data["forecasts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let mad_val = mad(fc.primary(), &ref_fc);
        let rmse_val = rmse(fc.primary(), &ref_fc);
        let max_diff: f64 = fc
            .primary()
            .iter()
            .zip(ref_fc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        let match_level = if mad_val < 0.01 {
            "EXACT"
        } else if mad_val < 1.0 {
            "CLOSE"
        } else {
            "DIFF"
        };

        total_models += 1;
        if mad_val < 1.0 {
            total_close += 1;
        }
        if mad_val < 5.0 {
            total_reasonable += 1;
        }

        println!(
            "  {:<20} {:>8.4} {:>8.4} {:>8.4}  {:>7.3}ms  {}",
            "CrostonSBA", mad_val, rmse_val, max_diff, elapsed, match_level
        );
    }

    // Summary
    println!("\n========================================================================");
    println!(
        "  SUMMARY: {}/{} models within MAD<1.0 (CLOSE/EXACT)",
        total_close, total_models
    );
    println!(
        "  SUMMARY: {}/{} models within MAD<5.0 (reasonable)",
        total_reasonable, total_models
    );
    println!("========================================================================");

    // Assert reasonable accuracy for most models
    assert!(
        total_reasonable as f64 / total_models as f64 > 0.7,
        "Less than 70% of models produce reasonable results (MAD<5.0)"
    );
}
