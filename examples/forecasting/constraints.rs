//! Forecast constraints example.
//!
//! Demonstrates how to apply constraints to forecast values to enforce
//! domain-specific requirements such as non-negativity, bounded ranges,
//! integer rounding, and custom transformations.
//!
//! Run with: cargo run --example constraints

use anofox_forecast::core::{ConstrainedForecast, Forecast, ForecastConstraint, TimeSeries};
use anofox_forecast::models::exponential::{ETSSpec, ETS};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Forecast Constraints Example ===\n");

    // -----------------------------------------------------------------------
    // Generate sample data with trend and seasonality, then produce a forecast
    // that will contain some negative and fractional values.
    // -----------------------------------------------------------------------
    let n = 60;
    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let base = 5.0;
            let trend = 0.2 * i as f64;
            let seasonal = 8.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            base + trend + seasonal
        })
        .collect();

    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = ETS::new(ETSSpec::aaa(), 12);
    model.fit(&ts).unwrap();

    let forecast = model.predict_with_intervals(12, 0.95).unwrap();

    println!("--- Raw Forecast (may contain negatives) ---");
    let point = forecast.primary();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();

    println!("{:>4} {:>10} {:>10} {:>10}", "h", "Lower", "Point", "Upper");
    println!("{:-<38}", "");
    for i in 0..forecast.horizon() {
        println!(
            "{:>4} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            lower[i],
            point[i],
            upper[i]
        );
    }

    // -----------------------------------------------------------------------
    // 1. NonNegative constraint
    // -----------------------------------------------------------------------
    println!("\n--- ForecastConstraint::NonNegative ---");
    let nn = ConstrainedForecast::apply(&forecast, &[ForecastConstraint::NonNegative]);

    let nn_point = nn.primary();
    let nn_lower = nn.lower_series(0).unwrap();
    println!("{:>4} {:>10} {:>10}", "h", "Point", "Lower");
    println!("{:-<26}", "");
    for i in 0..nn.horizon() {
        println!("{:>4} {:>10.2} {:>10.2}", i + 1, nn_point[i], nn_lower[i]);
    }
    println!(
        "All lower bounds >= 0: {}",
        nn_lower.iter().all(|v| *v >= 0.0)
    );

    // -----------------------------------------------------------------------
    // 2. LowerBound and UpperBound
    // -----------------------------------------------------------------------
    println!("\n--- ForecastConstraint::LowerBound / UpperBound ---");
    let bounded = ConstrainedForecast::apply(
        &forecast,
        &[
            ForecastConstraint::LowerBound(2.0),
            ForecastConstraint::UpperBound(30.0),
        ],
    );

    let b_point = bounded.primary();
    println!("{:>4} {:>10}", "h", "Point");
    println!("{:-<16}", "");
    for (i, p) in b_point.iter().enumerate().take(bounded.horizon()) {
        println!("{:>4} {:>10.2}", i + 1, p);
    }
    println!(
        "All in [2, 30]: {}",
        b_point.iter().all(|v| *v >= 2.0 && *v <= 30.0)
    );

    // -----------------------------------------------------------------------
    // 3. Bounds (combined lower + upper in one variant)
    // -----------------------------------------------------------------------
    println!("\n--- ForecastConstraint::Bounds ---");
    let clamped = ConstrainedForecast::apply(
        &forecast,
        &[ForecastConstraint::Bounds {
            lower: 0.0,
            upper: 25.0,
        }],
    );
    let c_point = clamped.primary();
    println!("{:>4} {:>10}", "h", "Point");
    println!("{:-<16}", "");
    for (i, p) in c_point.iter().enumerate().take(clamped.horizon()) {
        println!("{:>4} {:>10.2}", i + 1, p);
    }

    // -----------------------------------------------------------------------
    // 4. IntegerRound
    // -----------------------------------------------------------------------
    println!("\n--- ForecastConstraint::IntegerRound ---");
    let rounded = ConstrainedForecast::apply(&forecast, &[ForecastConstraint::IntegerRound]);

    let r_point = rounded.primary();
    let r_lower = rounded.lower_series(0).unwrap();
    let r_upper = rounded.upper_series(0).unwrap();
    println!("{:>4} {:>10} {:>10} {:>10}", "h", "Lower", "Point", "Upper");
    println!("{:-<38}", "");
    for i in 0..rounded.horizon() {
        println!(
            "{:>4} {:>10.0} {:>10.0} {:>10.0}",
            i + 1,
            r_lower[i],
            r_point[i],
            r_upper[i]
        );
    }
    println!(
        "Point values are integers: {}",
        r_point.iter().all(|v| (v - v.round()).abs() < 1e-10)
    );
    println!(
        "Lower values are floored:  {}",
        r_lower.iter().all(|v| (v - v.floor()).abs() < 1e-10)
    );
    println!(
        "Upper values are ceiled:   {}",
        r_upper.iter().all(|v| (v - v.ceil()).abs() < 1e-10)
    );

    // -----------------------------------------------------------------------
    // 5. Custom constraint
    // -----------------------------------------------------------------------
    println!("\n--- ForecastConstraint::Custom ---");
    let custom = ConstrainedForecast::apply(
        &forecast,
        &[ForecastConstraint::Custom(Box::new(|v| {
            // Snap to nearest multiple of 5
            (v / 5.0).round() * 5.0
        }))],
    );

    let cu_point = custom.primary();
    println!("{:>4} {:>10} {:>10}", "h", "Original", "Snapped");
    println!("{:-<26}", "");
    for i in 0..custom.horizon() {
        println!("{:>4} {:>10.2} {:>10.0}", i + 1, point[i], cu_point[i]);
    }

    // -----------------------------------------------------------------------
    // 6. Convenience methods on Forecast
    // -----------------------------------------------------------------------
    println!("\n--- Convenience Methods ---");

    // .non_negative()
    let nn2 = forecast.non_negative();
    println!(
        "non_negative() - all >= 0: {}",
        nn2.primary().iter().all(|v| *v >= 0.0)
    );

    // .clamp(lower, upper)
    let clamped2 = forecast.clamp(1.0, 20.0);
    println!(
        "clamp(1, 20)   - all in range: {}",
        clamped2.primary().iter().all(|v| *v >= 1.0 && *v <= 20.0)
    );

    // .round_to_integer()
    let rounded2 = forecast.round_to_integer();
    println!(
        "round_to_integer() - all int: {}",
        rounded2
            .primary()
            .iter()
            .all(|v| (v - v.round()).abs() < 1e-10)
    );

    // -----------------------------------------------------------------------
    // 7. Composing multiple constraints with .constrain()
    // -----------------------------------------------------------------------
    println!("\n--- Composing Constraints ---");
    let composed = forecast.constrain(&[
        ForecastConstraint::NonNegative,
        ForecastConstraint::Bounds {
            lower: 0.0,
            upper: 50.0,
        },
        ForecastConstraint::IntegerRound,
    ]);

    let comp_point = composed.primary();
    let comp_lower = composed.lower_series(0).unwrap();
    let comp_upper = composed.upper_series(0).unwrap();

    println!("{:>4} {:>10} {:>10} {:>10}", "h", "Lower", "Point", "Upper");
    println!("{:-<38}", "");
    for i in 0..composed.horizon() {
        println!(
            "{:>4} {:>10.0} {:>10.0} {:>10.0}",
            i + 1,
            comp_lower[i],
            comp_point[i],
            comp_upper[i]
        );
    }
    println!(
        "Non-negative integers in [0, 50]: {}",
        comp_point
            .iter()
            .all(|v| *v >= 0.0 && *v <= 50.0 && (v - v.round()).abs() < 1e-10)
    );

    // -----------------------------------------------------------------------
    // 8. Constraints on a forecast without intervals
    // -----------------------------------------------------------------------
    println!("\n--- Constraints Without Intervals ---");
    let point_only = Forecast::from_values(vec![-3.5, 1.2, 7.8, -0.4, 12.6]);
    println!("Original:      {:?}", point_only.primary());

    let constrained = point_only.non_negative();
    println!("Non-negative:  {:?}", constrained.primary());
    println!(
        "Has intervals: {}",
        constrained.has_lower() || constrained.has_upper()
    );

    let constrained = point_only.clamp(0.0, 10.0);
    println!("Clamped [0,10]: {:?}", constrained.primary());

    let constrained = point_only.round_to_integer();
    println!("Rounded:       {:?}", constrained.primary());

    println!("\n=== Forecast Constraints Example Complete ===");
}
