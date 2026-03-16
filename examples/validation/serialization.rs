//! Model and Data Serialization Example
//!
//! This example demonstrates JSON and bincode serialization for
//! models, TimeSeries, and Forecast objects.
//!
//! Requires the `serde` feature:
//!   cargo run --example serialization --features serde
//!
//! Run with: cargo run --example serialization --features serde

fn main() {
    #[cfg(not(feature = "serde"))]
    {
        println!("=== Serialization Example ===\n");
        println!("This example requires the 'serde' feature.");
        println!("Run with: cargo run --example serialization --features serde");
    }

    #[cfg(feature = "serde")]
    serialization_demo();
}

#[cfg(feature = "serde")]
fn serialization_demo() {
    use anofox_forecast::core::{Forecast, TimeSeries};
    use anofox_forecast::models::baseline::Naive;
    use anofox_forecast::models::Forecaster;
    use anofox_forecast::utils::persistence::{
        from_bincode, from_json, to_bincode, to_json,
    };
    use chrono::{Duration, TimeZone, Utc};

    println!("=== Model and Data Serialization Example ===\n");

    // Create sample data
    let n = 30;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 10.0 + 0.5 * i as f64;
            let noise = ((i * 7 + 3) % 11) as f64 - 5.0;
            trend + noise
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    println!("Sample data: {} observations", ts.len());
    println!(
        "First 10: {:?}\n",
        &ts.primary_values()[..10]
            .iter()
            .map(|v| format!("{:.1}", v))
            .collect::<Vec<_>>()
    );

    // =========================================================================
    // 1. Model JSON Serialization
    // =========================================================================
    println!("--- Model: to_json() / from_json() ---\n");

    let mut model = Naive::new();
    model.fit(&ts).unwrap();

    let json = to_json(&model).unwrap();

    println!("Serialized Naive model to JSON:");
    println!("  JSON size: {} bytes", json.len());
    println!("  Preview:   {}...", &json[..80.min(json.len())]);

    let restored: Naive = from_json(&json).unwrap();
    let original_pred = model.predict(5).unwrap();
    let restored_pred = restored.predict(5).unwrap();

    println!("\nPrediction comparison (5-step):");
    println!(
        "  {:>4}  {:>10}  {:>10}  {:>6}",
        "Step", "Original", "Restored", "Match"
    );
    println!("  {:-<36}", "");
    for i in 0..5 {
        let a = original_pred.primary()[i];
        let b = restored_pred.primary()[i];
        let ok = (a - b).abs() < 1e-10;
        println!(
            "  {:>4}  {:>10.4}  {:>10.4}  {:>6}",
            i + 1,
            a,
            b,
            if ok { "yes" } else { "no" }
        );
    }

    // =========================================================================
    // 2. Model Bincode Serialization
    // =========================================================================
    println!("\n--- Model: to_bincode() / from_bincode() ---\n");

    let bin = to_bincode(&model).unwrap();
    println!("Serialized Naive model to bincode:");
    println!("  Bincode size: {} bytes", bin.len());
    println!("  JSON size:    {} bytes", json.len());
    println!(
        "  Compression:  {:.1}x smaller",
        json.len() as f64 / bin.len() as f64
    );

    // Note: Models using nan_vec serializer may not round-trip via bincode.
    // This is expected — use JSON for models with NaN-aware fields.
    match from_bincode::<Naive>(&bin) {
        Ok(restored_bin) => {
            let restored_bin_pred = restored_bin.predict(5).unwrap();
            let all_match = (0..5).all(|i| {
                (original_pred.primary()[i] - restored_bin_pred.primary()[i]).abs() < 1e-10
            });
            println!("  Predictions match: {}", all_match);
        }
        Err(e) => {
            println!(
                "  Bincode round-trip failed (expected for nan_vec fields): {}",
                e
            );
            println!("  Use JSON for models with NaN-aware serialization.");
        }
    }

    // =========================================================================
    // 3. TimeSeries Serialization
    // =========================================================================
    println!("\n--- TimeSeries Serialization ---\n");

    // JSON round-trip
    let ts_json = to_json(&ts).unwrap();
    let ts_restored: TimeSeries = from_json(&ts_json).unwrap();

    println!("TimeSeries JSON round-trip:");
    println!("  JSON size:        {} bytes", ts_json.len());
    println!("  Original length:  {}", ts.len());
    println!("  Restored length:  {}", ts_restored.len());
    println!(
        "  Values match:     {}",
        ts.primary_values() == ts_restored.primary_values()
    );
    println!(
        "  Timestamps match: {}",
        ts.timestamps() == ts_restored.timestamps()
    );

    // Bincode round-trip
    let ts_bin = to_bincode(&ts).unwrap();
    let ts_restored_bin: TimeSeries = from_bincode(&ts_bin).unwrap();

    println!("\nTimeSeries bincode round-trip:");
    println!("  Bincode size:     {} bytes", ts_bin.len());
    println!("  JSON size:        {} bytes", ts_json.len());
    println!("  Restored length:  {}", ts_restored_bin.len());
    println!(
        "  Values match:     {}",
        ts.primary_values() == ts_restored_bin.primary_values()
    );

    // =========================================================================
    // 4. Forecast Serialization
    // =========================================================================
    println!("\n--- Forecast Serialization ---\n");

    let forecast = Forecast::from_values_with_intervals(
        vec![10.0, 12.0, 14.0, 16.0, 18.0],
        vec![8.0, 9.5, 11.0, 12.5, 14.0],
        vec![12.0, 14.5, 17.0, 19.5, 22.0],
    );

    // JSON
    let fc_json = to_json(&forecast).unwrap();
    let fc_restored: Forecast = from_json(&fc_json).unwrap();

    println!("Forecast JSON round-trip:");
    println!("  JSON size:  {} bytes", fc_json.len());
    println!("  Horizon:    {}", fc_restored.horizon());
    println!("  Has lower:  {}", fc_restored.has_lower());
    println!("  Has upper:  {}", fc_restored.has_upper());
    println!("  Match:      {}", forecast == fc_restored);

    // Bincode
    let fc_bin = to_bincode(&forecast).unwrap();
    let fc_restored_bin: Forecast = from_bincode(&fc_bin).unwrap();

    println!("\nForecast bincode round-trip:");
    println!("  Bincode size: {} bytes", fc_bin.len());
    println!("  Match:        {}", forecast == fc_restored_bin);

    // Show the round-tripped values
    println!("\n  {:>4}  {:>8}  {:>8}  {:>8}", "Step", "Lower", "Point", "Upper");
    println!("  {:-<36}", "");
    let p = fc_restored.primary();
    let lo = fc_restored.lower_series(0).unwrap();
    let hi = fc_restored.upper_series(0).unwrap();
    for i in 0..fc_restored.horizon() {
        println!(
            "  {:>4}  {:>8.1}  {:>8.1}  {:>8.1}",
            i + 1,
            lo[i],
            p[i],
            hi[i]
        );
    }

    // =========================================================================
    // 5. Size Comparison
    // =========================================================================
    println!("\n--- Format Size Comparison ---\n");

    println!(
        "  {:>18}  {:>10}  {:>10}  {:>8}",
        "Object", "JSON", "Bincode", "Ratio"
    );
    println!("  {:-<50}", "");

    let items: Vec<(&str, usize, usize)> = vec![
        ("Naive model", json.len(), bin.len()),
        ("TimeSeries (30pt)", ts_json.len(), ts_bin.len()),
        ("Forecast (5pt)", fc_json.len(), fc_bin.len()),
    ];
    for (name, json_sz, bin_sz) in &items {
        println!(
            "  {:>18}  {:>8} B  {:>8} B  {:>7.1}x",
            name,
            json_sz,
            bin_sz,
            *json_sz as f64 / *bin_sz as f64
        );
    }

    println!("\n  Bincode is more compact (binary) but not human-readable.");
    println!("  JSON is larger but inspectable and portable.");

    // =========================================================================
    // Summary
    // =========================================================================
    println!(
        "
--- Summary ---

Serialization functions (require 'serde' feature):

  to_json(obj)        -> Result<String>     Human-readable JSON
  from_json(str)      -> Result<T>          Parse from JSON string
  to_bincode(obj)     -> Result<Vec<u8>>    Compact binary format
  from_bincode(bytes) -> Result<T>          Parse from binary

Also available for file I/O:
  save_to_file(obj, path)        JSON to file
  load_from_file(path)           JSON from file
  save_to_bincode(obj, path)     Bincode to file
  load_from_bincode(path)        Bincode from file

Supported types: all Forecaster models, TimeSeries, Forecast.

Note: Models using the nan_vec custom serializer (e.g., ARIMA)
work with JSON but may not round-trip correctly via bincode
for fields containing NaN. Use JSON for model persistence.
"
    );

    println!("=== Serialization Example Complete ===");
}
