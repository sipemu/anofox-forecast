//! Intermittent Demand Forecasting example.
//!
//! Run with: cargo run --example intermittent

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::intermittent::{Croston, ADIDA, TSB};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Intermittent Demand Forecasting Example ===\n");

    println!("Intermittent demand is characterized by:");
    println!("  - Many zero-demand periods");
    println!("  - Sporadic non-zero demands");
    println!("  - Common in spare parts, slow-moving inventory\n");

    // Generate intermittent demand pattern
    let timestamps: Vec<_> = (0..40)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i))
        .collect();

    // Intermittent pattern: demand occurs at irregular intervals
    let values = vec![
        5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
    ];

    let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

    // Analyze the demand pattern
    let non_zero: Vec<f64> = values.iter().filter(|&&v| v > 0.0).copied().collect();
    let demand_count = non_zero.len();
    let avg_demand = non_zero.iter().sum::<f64>() / demand_count as f64;
    let avg_interval = values.len() as f64 / demand_count as f64;

    println!("Demand Pattern Analysis:");
    println!("  Total periods: {}", values.len());
    println!("  Non-zero demands: {}", demand_count);
    println!("  Zero-demand periods: {}", values.len() - demand_count);
    println!("  Average demand size: {:.2}", avg_demand);
    println!(
        "  Average inter-demand interval: {:.2} periods",
        avg_interval
    );
    println!(
        "  Demand ratio: {:.1}%\n",
        100.0 * demand_count as f64 / values.len() as f64
    );

    // 1. Croston's Method (Classic)
    println!("--- Croston's Method (Classic) ---");
    println!("Separately forecasts demand sizes and inter-arrival times\n");

    let mut croston = Croston::new().with_alpha(0.1);
    croston.fit(&ts).unwrap();

    println!("Alpha: {:.2}", croston.alpha());
    println!("Demand level: {:.4}", croston.demand_level().unwrap());
    println!("Interval level: {:.4}", croston.interval_level().unwrap());

    let croston_forecast = croston.predict(10).unwrap();
    println!("\nForecast (flat): {:.4}", croston_forecast.primary()[0]);
    println!(
        "Expected demand rate: {:.4} per period",
        croston.demand_level().unwrap() / croston.interval_level().unwrap()
    );

    // 2. Croston with Optimized Alpha
    println!("\n--- Croston's Method (Optimized) ---");

    let mut croston_opt = Croston::new().optimized();
    croston_opt.fit(&ts).unwrap();

    println!("Optimized alpha: {:.4}", croston_opt.alpha());
    let opt_forecast = croston_opt.predict(1).unwrap();
    println!("Forecast: {:.4}", opt_forecast.primary()[0]);

    // 3. Syntetos-Boylan Approximation (SBA)
    println!("\n--- Syntetos-Boylan Approximation (SBA) ---");
    println!("Applies bias correction: forecast * (1 - alpha/2)\n");

    let mut croston_sba = Croston::new().with_alpha(0.1).sba();
    croston_sba.fit(&ts).unwrap();

    let sba_forecast = croston_sba.predict(1).unwrap();
    println!("SBA Forecast: {:.4}", sba_forecast.primary()[0]);
    println!("Bias correction factor: {:.4}", 1.0 - 0.1 / 2.0);

    // Compare Classic vs SBA
    println!("\nComparison (alpha=0.1):");
    println!("  Classic: {:.4}", croston_forecast.primary()[0]);
    println!("  SBA:     {:.4}", sba_forecast.primary()[0]);
    println!(
        "  Ratio:   {:.4}",
        sba_forecast.primary()[0] / croston_forecast.primary()[0]
    );

    // 4. TSB (Teunter-Syntetos-Babai)
    println!("\n--- TSB Method (Teunter-Syntetos-Babai) ---");
    println!("Updates probability of demand occurrence directly\n");

    let mut tsb = TSB::new();
    tsb.fit(&ts).unwrap();

    let tsb_forecast = tsb.predict(10).unwrap();
    println!("TSB Forecast: {:.4}", tsb_forecast.primary()[0]);
    println!("Demand probability: {:.4}", tsb.probability().unwrap());
    println!("Demand level: {:.4}", tsb.demand_level().unwrap());

    // TSB with custom parameters
    println!("\nTSB with different alpha_d/alpha_p values:");
    for alpha in [0.05, 0.1, 0.2, 0.3] {
        let mut model = TSB::new().with_params(alpha, alpha);
        model.fit(&ts).unwrap();
        let fc = model.predict(1).unwrap();
        println!(
            "  alpha_d={:.2}, alpha_p={:.2}: forecast={:.4}",
            alpha,
            alpha,
            fc.primary()[0]
        );
    }

    // 5. ADIDA (Aggregate-Disaggregate Intermittent Demand Approach)
    println!("\n--- ADIDA Method ---");
    println!("Aggregates data, forecasts, then disaggregates\n");

    let mut adida = ADIDA::new().with_aggregation_level(4); // Aggregation level of 4
    adida.fit(&ts).unwrap();

    let adida_forecast = adida.predict(10).unwrap();
    println!("Aggregation level: {:?}", adida.aggregation_level());
    println!("ADIDA Forecast: {:.4}", adida_forecast.primary()[0]);

    // ADIDA with different aggregation levels
    println!("\nADIDA with different aggregation levels:");
    for agg in [2, 3, 4, 5] {
        let mut model = ADIDA::new().with_aggregation_level(agg);
        model.fit(&ts).unwrap();
        let fc = model.predict(1).unwrap();
        println!("  aggregation={}: forecast={:.4}", agg, fc.primary()[0]);
    }

    // 6. Model Comparison
    println!("\n--- Model Comparison ---");
    println!("{:<25} {:>15}", "Method", "Forecast");
    println!("{:-<42}", "");

    let models: Vec<(&str, f64)> = vec![
        ("Croston Classic", croston_forecast.primary()[0]),
        ("Croston Optimized", opt_forecast.primary()[0]),
        ("Croston SBA", sba_forecast.primary()[0]),
        ("TSB", tsb_forecast.primary()[0]),
        ("ADIDA (agg=4)", adida_forecast.primary()[0]),
    ];

    for (name, fc) in &models {
        println!("{:<25} {:>15.4}", name, fc);
    }

    // 7. Confidence Intervals
    println!("\n--- Confidence Intervals (95%) ---");

    let croston_ci = croston.predict_with_intervals(5, 0.95).unwrap();
    let preds = croston_ci.primary();
    let lower = croston_ci.lower_series(0).unwrap();
    let upper = croston_ci.upper_series(0).unwrap();

    println!("\nCroston with 95% CI:");
    println!(
        "{:>4} {:>12} {:>12} {:>12}",
        "h", "Lower", "Forecast", "Upper"
    );
    println!("{:-<44}", "");
    for i in 0..5 {
        println!(
            "{:>4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            lower[i],
            preds[i],
            upper[i]
        );
    }

    // 8. When to use each method
    println!("\n--- When to Use Each Method ---");
    println!(
        "
Croston Classic:
  - Original method, still widely used
  - Good baseline for intermittent demand

Croston SBA:
  - Corrects upward bias in Classic Croston
  - Generally preferred over Classic

TSB:
  - Better for obsolescence patterns
  - When demand probability changes over time
  - More responsive to demand pattern changes

ADIDA:
  - When data is very sparse
  - Aggregation helps stabilize estimates
  - Good for very slow-moving items
"
    );

    // 9. Demand Classification
    println!("--- Demand Classification ---");
    let cv2 = {
        let mean = avg_demand;
        let variance =
            non_zero.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / demand_count as f64;
        variance / (mean * mean)
    };

    println!("\nDemand Characteristics:");
    println!("  Average demand interval (ADI): {:.2}", avg_interval);
    println!("  Coefficient of variation squared (CV²): {:.4}", cv2);

    println!("\nClassification Matrix:");
    println!("  ADI < 1.32 & CV² < 0.49  -> Smooth demand");
    println!("  ADI >= 1.32 & CV² < 0.49 -> Intermittent");
    println!("  ADI < 1.32 & CV² >= 0.49 -> Erratic");
    println!("  ADI >= 1.32 & CV² >= 0.49 -> Lumpy");

    let classification = match (avg_interval >= 1.32, cv2 >= 0.49) {
        (false, false) => "Smooth",
        (true, false) => "Intermittent",
        (false, true) => "Erratic",
        (true, true) => "Lumpy",
    };
    println!("\nThis demand pattern is: {}", classification);

    println!("\n=== Intermittent Demand Example Complete ===");
}
