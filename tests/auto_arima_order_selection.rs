//! Compare AutoARIMA model order selection: Rust vs Python statsforecast.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn ts(n: usize) -> Vec<chrono::DateTime<Utc>> {
    (0..n)
        .map(|i| Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap() + Duration::days(i as i64))
        .collect()
}

#[test]
fn order_selection_comparison() {
    println!("\n=== AutoARIMA Order Selection: Rust vs Python ===\n");
    println!(
        "{:<25} {:>15} {:>15} {:>8}",
        "Test", "Rust", "Python", "Match?"
    );
    println!("{}", "-".repeat(65));

    // Test 1: Trend + weekly sine (n=100)
    let y1: Vec<f64> = (0..100)
        .map(|i| {
            50.0 + 0.3 * i as f64
                + 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                + ((i * 7 + 3) % 11) as f64 * 0.3
        })
        .collect();
    let ts1 = TimeSeries::univariate(ts(100), y1).unwrap();
    let mut m1 = AutoARIMA::new();
    m1.fit(&ts1).unwrap();
    let (p, d, q) = m1.selected_order().unwrap();
    let rust1 = format!("ARIMA({},{},{})", p, d, q);
    let py1 = "ARIMA(2,1,2)";
    println!(
        "{:<25} {:>15} {:>15} {:>8}",
        "trend+sine n=100",
        rust1,
        py1,
        if rust1 == py1 { "YES" } else { "no" }
    );

    // Test 2: Linear trend (n=50)
    let y2: Vec<f64> = (0..50).map(|i| 10.0 + 0.5 * i as f64).collect();
    let ts2 = TimeSeries::univariate(ts(50), y2).unwrap();
    let mut m2 = AutoARIMA::new();
    m2.fit(&ts2).unwrap();
    let (p, d, q) = m2.selected_order().unwrap();
    let rust2 = format!("ARIMA({},{},{})", p, d, q);
    let py2 = "ARIMA(0,1,0)";
    println!(
        "{:<25} {:>15} {:>15} {:>8}",
        "linear n=50",
        rust2,
        py2,
        if rust2 == py2 { "YES" } else { "no" }
    );

    // Test 3: Random walk (n=200)
    let mut y3 = vec![100.0; 200];
    for i in 1..200 {
        y3[i] = y3[i - 1] + 0.3 + ((i * 7 + 3) % 11) as f64 * 0.4 - 2.2;
    }
    let ts3 = TimeSeries::univariate(ts(200), y3).unwrap();
    let mut m3 = AutoARIMA::new();
    m3.fit(&ts3).unwrap();
    let (p, d, q) = m3.selected_order().unwrap();
    let rust3 = format!("ARIMA({},{},{})", p, d, q);
    let py3 = "ARIMA(2,1,2)";
    println!(
        "{:<25} {:>15} {:>15} {:>8}",
        "random walk n=200",
        rust3,
        py3,
        if rust3 == py3 { "YES" } else { "no" }
    );

    // Test 4: AR(1) process (n=150)
    let mut y4 = vec![50.0; 150];
    for i in 1..150 {
        y4[i] = 30.0 + 0.7 * (y4[i - 1] - 30.0) + ((i * 13 + 7) % 17) as f64 * 0.3 - 2.55;
    }
    let ts4 = TimeSeries::univariate(ts(150), y4).unwrap();
    let mut m4 = AutoARIMA::new();
    m4.fit(&ts4).unwrap();
    let (p, d, q) = m4.selected_order().unwrap();
    let rust4 = format!("ARIMA({},{},{})", p, d, q);
    let py4 = "ARIMA(3,1,2)";
    println!(
        "{:<25} {:>15} {:>15} {:>8}",
        "AR(1) n=150",
        rust4,
        py4,
        if rust4 == py4 { "YES" } else { "no" }
    );

    println!();

    // Count matches
    let matches = [rust1 == py1, rust2 == py2, rust3 == py3, rust4 == py4]
        .iter()
        .filter(|&&m| m)
        .count();
    println!("Matches: {}/4", matches);
}
