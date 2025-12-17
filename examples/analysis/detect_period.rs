//! Periodicity detection CLI tool for validation against pyriodicity.
//!
//! This example reads time series data and runs various periodicity detectors,
//! outputting results in JSON format for comparison with Python implementations.
//!
//! Usage:
//!   # Read from stdin (one value per line)
//!   echo "1.0\n2.0\n3.0..." | cargo run --example detect_period -- --method autoperiod
//!
//!   # Synthetic test signal
//!   cargo run --example detect_period -- --test --period 12 --length 240
//!
//!   # All methods on synthetic data
//!   cargo run --example detect_period -- --test --period 12 --all-methods

use anofox_forecast::detection::{
    detect_period, detect_period_ensemble, ACFPeriodicityDetector, Autoperiod, CFDAutoperiod,
    FFTPeriodicityDetector, PeriodicityDetector, SAZED,
};
use std::env;
use std::io::{self, BufRead};

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let method = get_arg(&args, "--method").unwrap_or_else(|| "autoperiod".to_string());
    let min_period = get_arg(&args, "--min-period")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let max_period = get_arg(&args, "--max-period")
        .and_then(|s| s.parse().ok())
        .unwrap_or(365);
    let all_methods = args.contains(&"--all-methods".to_string());
    let test_mode = args.contains(&"--test".to_string());
    let json_output = args.contains(&"--json".to_string()) || all_methods;

    // Get data
    let data = if test_mode {
        let period = get_arg(&args, "--period")
            .and_then(|s| s.parse().ok())
            .unwrap_or(12);
        let length = get_arg(&args, "--length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(240);
        let noise = get_arg(&args, "--noise")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        generate_test_signal(length, period, noise)
    } else {
        read_stdin()
    };

    if data.is_empty() {
        eprintln!("Error: No data provided");
        std::process::exit(1);
    }

    if all_methods {
        run_all_methods(&data, min_period, max_period);
    } else if json_output {
        run_single_method_json(&data, &method, min_period, max_period);
    } else {
        run_single_method(&data, &method, min_period, max_period);
    }
}

fn get_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn read_stdin() -> Vec<f64> {
    let stdin = io::stdin();
    stdin
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| line.trim().parse().ok())
        .collect()
}

fn generate_test_signal(length: usize, period: usize, noise_level: f64) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let signal = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            // Deterministic "noise" for reproducibility
            let noise = if noise_level > 0.0 {
                let pseudo_random = ((i * 7 + 3) % 13) as f64 / 13.0 - 0.5;
                pseudo_random * noise_level
            } else {
                0.0
            };
            signal + noise
        })
        .collect()
}

fn run_single_method(data: &[f64], method: &str, min_period: usize, max_period: usize) {
    let result = match method.to_lowercase().as_str() {
        "acf" | "acfperiodicitydetector" => {
            ACFPeriodicityDetector::new(min_period, max_period, 0.3).detect(data)
        }
        "fft" | "fftperiodicitydetector" => {
            FFTPeriodicityDetector::new(min_period, max_period, 3.0).detect(data)
        }
        "autoperiod" => Autoperiod::new(min_period, max_period, 3.0, 0.2).detect(data),
        "cfdautoperiod" | "cfd" => CFDAutoperiod::new(min_period, max_period, 2.0).detect(data),
        "sazed" => SAZED::new(min_period, max_period).detect(data),
        "ensemble" => detect_period_ensemble(data),
        _ => {
            eprintln!(
                "Unknown method: {}. Available: acf, fft, autoperiod, cfdautoperiod, sazed, ensemble",
                method
            );
            std::process::exit(1);
        }
    };

    println!("Method: {}", result.method);
    println!("Data length: {}", data.len());
    println!(
        "Primary period: {}",
        result
            .primary_period
            .map(|p| p.to_string())
            .unwrap_or_else(|| "None".to_string())
    );
    println!("Confidence: {:.4}", result.confidence());
    println!("All detected periods:");
    for p in &result.periods {
        println!(
            "  - Period: {}, Score: {:.4}, Source: {:?}",
            p.period, p.score, p.source
        );
    }
}

fn run_single_method_json(data: &[f64], method: &str, min_period: usize, max_period: usize) {
    let result = match method.to_lowercase().as_str() {
        "acf" | "acfperiodicitydetector" => {
            ACFPeriodicityDetector::new(min_period, max_period, 0.3).detect(data)
        }
        "fft" | "fftperiodicitydetector" => {
            FFTPeriodicityDetector::new(min_period, max_period, 3.0).detect(data)
        }
        "autoperiod" => Autoperiod::new(min_period, max_period, 3.0, 0.2).detect(data),
        "cfdautoperiod" | "cfd" => CFDAutoperiod::new(min_period, max_period, 2.0).detect(data),
        "sazed" => SAZED::new(min_period, max_period).detect(data),
        "ensemble" => detect_period_ensemble(data),
        _ => detect_period(data),
    };

    let periods_json: Vec<String> = result
        .periods
        .iter()
        .map(|p| {
            format!(
                "{{\"period\": {}, \"score\": {:.6}, \"source\": \"{:?}\"}}",
                p.period, p.score, p.source
            )
        })
        .collect();

    println!(
        "{{\"method\": \"{}\", \"primary_period\": {}, \"confidence\": {:.6}, \"periods\": [{}]}}",
        result.method,
        result
            .primary_period
            .map(|p| p.to_string())
            .unwrap_or_else(|| "null".to_string()),
        result.confidence(),
        periods_json.join(", ")
    );
}

fn run_all_methods(data: &[f64], min_period: usize, max_period: usize) {
    let methods: Vec<(&str, Box<dyn PeriodicityDetector>)> = vec![
        (
            "ACFPeriodicityDetector",
            Box::new(ACFPeriodicityDetector::new(min_period, max_period, 0.3)),
        ),
        (
            "FFTPeriodicityDetector",
            Box::new(FFTPeriodicityDetector::new(min_period, max_period, 3.0)),
        ),
        (
            "Autoperiod",
            Box::new(Autoperiod::new(min_period, max_period, 3.0, 0.2)),
        ),
        (
            "CFDAutoperiod",
            Box::new(CFDAutoperiod::new(min_period, max_period, 2.0)),
        ),
        ("SAZED", Box::new(SAZED::new(min_period, max_period))),
    ];

    println!("{{");
    println!("  \"data_length\": {},", data.len());
    println!("  \"min_period\": {},", min_period);
    println!("  \"max_period\": {},", max_period);
    println!("  \"results\": {{");

    for (i, (name, detector)) in methods.iter().enumerate() {
        let result = detector.detect(data);
        let comma = if i < methods.len() - 1 { "," } else { "" };

        let periods_json: Vec<String> = result
            .periods
            .iter()
            .take(5) // Limit to top 5 periods
            .map(|p| format!("{{\"period\": {}, \"score\": {:.6}}}", p.period, p.score))
            .collect();

        println!(
            "    \"{}\": {{\"primary_period\": {}, \"confidence\": {:.6}, \"periods\": [{}]}}{}",
            name,
            result
                .primary_period
                .map(|p| p.to_string())
                .unwrap_or_else(|| "null".to_string()),
            result.confidence(),
            periods_json.join(", "),
            comma
        );
    }

    println!("  }}");
    println!("}}");
}
