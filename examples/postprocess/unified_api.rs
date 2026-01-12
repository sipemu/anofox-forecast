//! Unified PostProcessor API demonstration.
//!
//! This example shows how to use the PostProcessor interface to easily
//! switch between different postprocessing methods with the same API.
//!
//! Run with: cargo run --example postprocess_unified_api

use anofox_forecast::postprocess::{PointForecasts, PostModel, PostProcessor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified PostProcessor API ===\n");

    // Generate training data
    let n = 100;
    let train_forecasts: Vec<f64> = (0..n).map(|i| 50.0 + 0.2 * i as f64).collect();
    let train_actuals: Vec<f64> = train_forecasts
        .iter()
        .enumerate()
        .map(|(i, &f)| f + 1.5 * ((i as f64 * 0.1).sin()) + 0.3)
        .collect();

    let train_f = PointForecasts::from_values(train_forecasts);

    // Test data
    let test_forecasts: Vec<f64> = (100..110).map(|i| 50.0 + 0.2 * i as f64).collect();
    let test_f = PointForecasts::from_values(test_forecasts.clone());

    println!("Training samples: {}", n);
    println!("Test samples: 10\n");

    // Define different processors - same API, different methods
    let processors: Vec<(&str, PostProcessor)> = vec![
        ("Conformal (90%)", PostProcessor::conformal(0.90)),
        (
            "HistoricalSim",
            PostProcessor::historical_sim(vec![0.05, 0.5, 0.95]),
        ),
        ("Normal", PostProcessor::normal(vec![0.05, 0.5, 0.95])),
        ("IDR", PostProcessor::idr(vec![0.05, 0.5, 0.95])),
    ];

    println!("=== Method Comparison ===\n");

    for (name, processor) in &processors {
        println!("--- {} ---", name);

        // Same API for all methods: train -> predict
        let trained = processor.train(&train_f, &train_actuals)?;
        let intervals = processor.predict_intervals(&trained, &test_f)?;

        // Calculate average interval width
        let avg_width: f64 = intervals
            .lower()
            .iter()
            .zip(intervals.upper())
            .map(|(l, u)| u - l)
            .sum::<f64>()
            / intervals.len() as f64;

        println!("  Average interval width: {:.3}", avg_width);

        // Show first prediction
        println!(
            "  First interval: [{:.2}, {:.2}] for forecast {:.2}",
            intervals.lower()[0],
            intervals.upper()[0],
            test_forecasts[0]
        );
        println!();
    }

    // Demonstrate model selection workflow
    println!("=== Model Selection Workflow ===\n");

    // Split data for validation
    let val_split = 80;
    let val_forecasts: Vec<f64> = train_f.values()[val_split..].to_vec();
    let val_actuals: Vec<f64> = train_actuals[val_split..].to_vec();

    let train_subset_f = PointForecasts::from_values(train_f.values()[..val_split].to_vec());
    let train_subset_a = &train_actuals[..val_split];
    let val_f = PointForecasts::from_values(val_forecasts);

    println!("Train subset: {} samples", val_split);
    println!("Validation: {} samples\n", n - val_split);

    let mut best_method = "";
    let mut best_score = f64::MAX;

    for (name, processor) in &processors {
        let trained = processor.train(&train_subset_f, train_subset_a)?;
        let intervals = processor.predict_intervals(&trained, &val_f)?;

        // Calculate coverage and width
        let mut covered = 0;
        let mut total_width = 0.0;

        for i in 0..intervals.len() {
            let lower = intervals.lower()[i];
            let upper = intervals.upper()[i];
            let actual = val_actuals[i];

            if actual >= lower && actual <= upper {
                covered += 1;
            }
            total_width += upper - lower;
        }

        let coverage = covered as f64 / intervals.len() as f64;
        let avg_width = total_width / intervals.len() as f64;

        // Score: penalize undercoverage, reward narrower intervals
        let target_coverage = 0.90;
        let coverage_penalty = if coverage < target_coverage {
            (target_coverage - coverage) * 10.0
        } else {
            0.0
        };
        let score = avg_width + coverage_penalty;

        println!(
            "{:<18} coverage: {:.0}%, width: {:.3}, score: {:.3}",
            name,
            coverage * 100.0,
            avg_width,
            score
        );

        if score < best_score {
            best_score = score;
            best_method = name;
        }
    }

    println!("\nBest method: {} (score: {:.3})", best_method, best_score);

    // Demonstrate point_to_quantiles convenience method
    println!("\n=== Convenience Method: point_to_quantiles ===\n");

    let processor = PostProcessor::historical_sim(vec![0.1, 0.5, 0.9]);

    // One-liner to go from point forecasts to quantiles
    let quantiles = processor.point_to_quantiles(&train_f, &train_actuals, &test_f)?;

    println!("Direct conversion from point forecasts to quantiles:");
    println!(
        "{:<10} {:>10} {:>10} {:>10}",
        "Forecast", "q10", "q50", "q90"
    );
    println!("{:-<45}", "");

    for i in 0..3 {
        let q_row = quantiles.at_time(i).unwrap();
        println!(
            "{:<10.2} {:>10.2} {:>10.2} {:>10.2}",
            test_forecasts[i], q_row[0], q_row[1], q_row[2]
        );
    }
    println!("  ... (showing first 3 of 10)");

    // Show how to create custom models
    println!("\n=== Custom Model Configuration ===\n");

    use anofox_forecast::postprocess::ConformalMethod;

    let custom_processor = PostProcessor::new(PostModel::conformal_with_method(
        0.95,
        ConformalMethod::JackknifePlus,
    ));

    let trained = custom_processor.train(&train_f, &train_actuals)?;
    let intervals = custom_processor.predict_intervals(&trained, &test_f)?;

    println!("Custom: 95% Jackknife+ Conformal");
    println!(
        "  First interval: [{:.2}, {:.2}]",
        intervals.lower()[0],
        intervals.upper()[0]
    );

    println!("\nNotes:");
    println!("- PostProcessor provides uniform API across all methods");
    println!("- Easy to swap methods for comparison");
    println!("- Use point_to_quantiles() for quick one-off predictions");
    println!("- Use train() + predict_*() for reusable models");

    Ok(())
}
