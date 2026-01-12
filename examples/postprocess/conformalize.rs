//! Conformalize: Recalibrating quantile forecasts.
//!
//! This example demonstrates how to use conformalize() to recalibrate
//! miscalibrated quantile forecasts to achieve better coverage.
//!
//! Run with: cargo run --example postprocess_conformalize

#![allow(clippy::needless_range_loop)]
use anofox_forecast::postprocess::{conformalize, QuantileForecasts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Conformalize: Recalibrating Quantile Forecasts ===\n");

    // Simulate miscalibrated quantile forecasts
    // These intervals are too narrow (overconfident)
    let n_calib = 50;
    let n_test = 10;

    // True actuals
    let calib_actuals: Vec<f64> = (0..n_calib)
        .map(|i| 100.0 + 0.5 * i as f64 + 2.0 * ((i as f64 * 0.15).sin()))
        .collect();

    // Miscalibrated forecasts: intervals are too tight
    // q10 and q90 should cover 80% but these are too narrow
    let calib_quantile_values: Vec<Vec<f64>> = calib_actuals
        .iter()
        .map(|&a| {
            vec![
                a - 0.5, // q10: should be wider
                a,       // q50: median
                a + 0.5, // q90: should be wider
            ]
        })
        .collect();

    let calib_forecasts =
        QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], calib_quantile_values)?;

    // Check original coverage on calibration set
    let mut orig_covered = 0;
    for i in 0..n_calib {
        let q_row = calib_forecasts.at_time(i).unwrap();
        if calib_actuals[i] >= q_row[0] && calib_actuals[i] <= q_row[2] {
            orig_covered += 1;
        }
    }
    let orig_coverage = orig_covered as f64 / n_calib as f64;

    println!("Calibration set: {} samples", n_calib);
    println!(
        "Original 80% interval coverage: {:.1}% (target: 80%)",
        orig_coverage * 100.0
    );
    println!("  -> Intervals are TOO NARROW (overconfident)\n");

    // Test set forecasts (also miscalibrated)
    let test_actuals: Vec<f64> = (n_calib..n_calib + n_test)
        .map(|i| 100.0 + 0.5 * i as f64 + 2.0 * ((i as f64 * 0.15).sin()))
        .collect();

    let test_quantile_values: Vec<Vec<f64>> = test_actuals
        .iter()
        .map(|&a| {
            vec![
                a - 0.5, // too narrow
                a,
                a + 0.5, // too narrow
            ]
        })
        .collect();

    let test_forecasts = QuantileForecasts::from_values(vec![0.1, 0.5, 0.9], test_quantile_values)?;

    // Apply conformalize to recalibrate
    println!("Applying conformalize()...\n");
    let result = conformalize(&test_forecasts, &calib_forecasts, &calib_actuals)?;

    let calibrated = result.forecasts();

    // Display adjustments
    println!("Learned adjustments:");
    let adjustments = result.adjustments();
    println!(
        "  q10 adjustment: {:.3} (widens lower bound)",
        adjustments[0]
    );
    println!("  q50 adjustment: {:.3}", adjustments[1]);
    println!(
        "  q90 adjustment: {:.3} (widens upper bound)",
        adjustments[2]
    );

    // Compare before and after
    println!("\n=== Before vs After Conformalization ===");
    println!(
        "{:<6} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Step", "Actual", "Orig q10", "Orig q90", "Cal q10", "Cal q90"
    );
    println!("{:-<60}", "");

    for i in 0..5 {
        let orig_row = test_forecasts.at_time(i).unwrap();
        let cal_row = calibrated.at_time(i).unwrap();
        println!(
            "{:<6} {:>8.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            test_actuals[i],
            orig_row[0],
            orig_row[2],
            cal_row[0],
            cal_row[2]
        );
    }
    println!("  ... (showing first 5 of {})", n_test);

    // Check coverage after calibration
    let mut cal_covered = 0;
    for i in 0..n_test {
        let q_row = calibrated.at_time(i).unwrap();
        if test_actuals[i] >= q_row[0] && test_actuals[i] <= q_row[2] {
            cal_covered += 1;
        }
    }
    let cal_coverage = cal_covered as f64 / n_test as f64;

    // Also check what coverage would have been without calibration
    let mut uncal_covered = 0;
    for i in 0..n_test {
        let q_row = test_forecasts.at_time(i).unwrap();
        if test_actuals[i] >= q_row[0] && test_actuals[i] <= q_row[2] {
            uncal_covered += 1;
        }
    }
    let uncal_coverage = uncal_covered as f64 / n_test as f64;

    // Calculate interval widths
    let orig_width: f64 = (0..n_test)
        .map(|i| {
            let row = test_forecasts.at_time(i).unwrap();
            row[2] - row[0]
        })
        .sum::<f64>()
        / n_test as f64;

    let cal_width: f64 = (0..n_test)
        .map(|i| {
            let row = calibrated.at_time(i).unwrap();
            row[2] - row[0]
        })
        .sum::<f64>()
        / n_test as f64;

    println!("\n=== Coverage Comparison ===");
    println!(
        "Original coverage on test set:    {:.1}%",
        uncal_coverage * 100.0
    );
    println!(
        "Calibrated coverage on test set:  {:.1}%",
        cal_coverage * 100.0
    );
    println!("Target coverage:                  80.0%");

    println!("\n=== Interval Width Comparison ===");
    println!("Original average width:    {:.2}", orig_width);
    println!("Calibrated average width:  {:.2}", cal_width);
    println!(
        "Width increase:            {:.1}%",
        (cal_width / orig_width - 1.0) * 100.0
    );

    println!("\nNotes:");
    println!("- conformalize() widens intervals to achieve target coverage");
    println!("- Uses conformal prediction on calibration residuals");
    println!("- Guaranteed coverage under exchangeability assumption");

    Ok(())
}
