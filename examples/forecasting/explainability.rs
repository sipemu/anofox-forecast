//! Forecast explainability example.
//!
//! Demonstrates the Explainable trait and ForecastExplanation, which
//! decompose a forecast into its level, trend, seasonal, and residual
//! components. Shows that the components sum back to the point forecast.
//!
//! Run with: cargo run --example explainability

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::explain::{Explainable, ForecastExplanation};
use anofox_forecast::models::exponential::{ETSSpec, ETS};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn print_explanation(explanation: &ForecastExplanation, forecast_point: &[f64]) {
    let n = explanation.level.len();

    let has_trend = explanation.trend.is_some();
    let has_seasonal = explanation.seasonal.is_some();
    let reconstructed = explanation.sum();

    // Header
    let mut header = format!("{:>4} {:>10}", "h", "Level");
    if has_trend {
        header.push_str(&format!(" {:>10}", "Trend"));
    }
    if has_seasonal {
        header.push_str(&format!(" {:>10}", "Seasonal"));
    }
    header.push_str(&format!(" {:>10} {:>10}", "Recon", "Forecast"));
    println!("{}", header);
    println!("{:-<width$}", "", width = header.len());

    for i in 0..n {
        let mut row = format!("{:>4} {:>10.4}", i + 1, explanation.level[i]);
        if let Some(ref trend) = explanation.trend {
            row.push_str(&format!(" {:>10.4}", trend[i]));
        }
        if let Some(ref seasonal) = explanation.seasonal {
            row.push_str(&format!(" {:>10.4}", seasonal[i]));
        }
        row.push_str(&format!(
            " {:>10.4} {:>10.4}",
            reconstructed[i], forecast_point[i]
        ));
        println!("{}", row);
    }
}

fn main() {
    println!("=== Forecast Explainability Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ETS(A,A,A) - full decomposition with level, trend, and seasonality
    // -----------------------------------------------------------------------
    println!("--- ETS(A,A,A): Level + Trend + Seasonal ---");

    let n = 72;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let base = 50.0;
            let trend = 0.5 * i as f64;
            let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            base + trend + seasonal
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = ETS::new(ETSSpec::aaa(), 12);
    model.fit(&ts).unwrap();

    let horizon = 12;
    let forecast = model.predict(horizon).unwrap();
    let explanation = model.explain(horizon).unwrap();

    print_explanation(&explanation, forecast.primary());

    // Verify components sum to forecast
    let reconstructed = explanation.sum();
    let max_error: f64 = forecast
        .primary()
        .iter()
        .zip(reconstructed.iter())
        .map(|(f, r)| (f - r).abs())
        .fold(0.0_f64, f64::max);

    println!("\nMax reconstruction error: {:.2e}", max_error);
    println!("Components sum to forecast: {}", max_error < 1e-6);

    // Check component presence
    println!("\nComponent presence:");
    println!("  Level:    yes ({} values)", explanation.level.len());
    println!(
        "  Trend:    {}",
        if let Some(t) = &explanation.trend {
            format!("yes ({} values)", t.len())
        } else {
            "no".to_string()
        }
    );
    println!(
        "  Seasonal: {}",
        if let Some(s) = &explanation.seasonal {
            format!("yes ({} values)", s.len())
        } else {
            "no".to_string()
        }
    );
    println!(
        "  Residual: {}",
        if let Some(r) = &explanation.residual {
            format!("yes ({} values)", r.len())
        } else {
            "no".to_string()
        }
    );

    // -----------------------------------------------------------------------
    // 2. ETS(A,A,N) - level + trend, no seasonality
    // -----------------------------------------------------------------------
    println!("\n--- ETS(A,A,N): Level + Trend (no seasonality) ---");

    let n2 = 50;
    let timestamps2: Vec<_> = (0..n2)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values2: Vec<f64> = (0..n2)
        .map(|i| 10.0 + 2.0 * i as f64 + (i as f64 * 0.2).sin())
        .collect();

    let ts2 = TimeSeries::univariate(timestamps2, values2).unwrap();

    let mut model2 = ETS::new(ETSSpec::aan(), 1);
    model2.fit(&ts2).unwrap();

    let horizon2 = 6;
    let forecast2 = model2.predict(horizon2).unwrap();
    let explanation2 = model2.explain(horizon2).unwrap();

    print_explanation(&explanation2, forecast2.primary());

    let reconstructed2 = explanation2.sum();
    let max_error2: f64 = forecast2
        .primary()
        .iter()
        .zip(reconstructed2.iter())
        .map(|(f, r)| (f - r).abs())
        .fold(0.0_f64, f64::max);

    println!("\nMax reconstruction error: {:.2e}", max_error2);
    println!(
        "Seasonal component: {}",
        if explanation2.seasonal.is_some() {
            "present"
        } else {
            "absent"
        }
    );

    // -----------------------------------------------------------------------
    // 3. ETS(A,N,N) - level only (Simple Exponential Smoothing)
    // -----------------------------------------------------------------------
    println!("\n--- ETS(A,N,N): Level Only ---");

    let n3 = 40;
    let timestamps3: Vec<_> = (0..n3)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values3: Vec<f64> = (0..n3)
        .map(|i| 20.0 + (i as f64 * 0.3).sin() * 2.0)
        .collect();

    let ts3 = TimeSeries::univariate(timestamps3, values3).unwrap();

    let mut model3 = ETS::new(ETSSpec::ann(), 1);
    model3.fit(&ts3).unwrap();

    let horizon3 = 5;
    let forecast3 = model3.predict(horizon3).unwrap();
    let explanation3 = model3.explain(horizon3).unwrap();

    print_explanation(&explanation3, forecast3.primary());

    let reconstructed3 = explanation3.sum();
    let max_error3: f64 = forecast3
        .primary()
        .iter()
        .zip(reconstructed3.iter())
        .map(|(f, r)| (f - r).abs())
        .fold(0.0_f64, f64::max);

    println!("\nMax reconstruction error: {:.2e}", max_error3);
    println!(
        "Trend component:    {}",
        if explanation3.trend.is_some() {
            "present"
        } else {
            "absent"
        }
    );
    println!(
        "Seasonal component: {}",
        if explanation3.seasonal.is_some() {
            "present"
        } else {
            "absent"
        }
    );

    // -----------------------------------------------------------------------
    // 4. Validate lengths with has_correct_lengths
    // -----------------------------------------------------------------------
    println!("\n--- Length Validation ---");
    println!(
        "ETS(A,A,A) explanation has correct lengths for h={}: {}",
        horizon,
        explanation.has_correct_lengths(horizon)
    );
    println!(
        "ETS(A,A,N) explanation has correct lengths for h={}: {}",
        horizon2,
        explanation2.has_correct_lengths(horizon2)
    );
    println!(
        "ETS(A,N,N) explanation has correct lengths for h={}: {}",
        horizon3,
        explanation3.has_correct_lengths(horizon3)
    );

    // -----------------------------------------------------------------------
    // 5. Named components
    // -----------------------------------------------------------------------
    println!("\n--- Named Components ---");
    if explanation.named_components.is_empty() {
        println!("ETS(A,A,A): no named components (standard decomposition used)");
    } else {
        for (name, vals) in &explanation.named_components {
            println!("  {}: {} values", name, vals.len());
        }
    }

    println!("\n=== Forecast Explainability Example Complete ===");
}
