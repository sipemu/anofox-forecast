//! VAR (Vector Autoregression) example.
//!
//! Run with: cargo run --example var

use anofox_forecast::core::{CalendarAnnotations, TimeSeriesBuilder};
use anofox_forecast::models::var::VAR;
use anofox_forecast::models::var_forecaster::VARForecaster;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== VAR (Vector Autoregression) Example ===\n");

    // -----------------------------------------------------------------------
    // Generate a bivariate VAR(1) system:
    //   y1(t) = 0.5 + 0.6 * y1(t-1) + 0.1 * y2(t-1) + noise
    //   y2(t) = 0.3 + 0.05 * y1(t-1) + 0.7 * y2(t-1) + noise
    // -----------------------------------------------------------------------
    let n = 200;
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];
    y1[0] = 1.0;
    y2[0] = -0.5;

    let mut seed: u64 = 42;
    for t in 1..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise1 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise2 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;

        y1[t] = 0.5 + 0.6 * y1[t - 1] + 0.1 * y2[t - 1] + noise1;
        y2[t] = 0.3 + 0.05 * y1[t - 1] + 0.7 * y2[t - 1] + noise2;
    }

    println!("Generated bivariate VAR(1) data: {} observations\n", n);
    println!("True DGP:");
    println!("  y1(t) = 0.50 + 0.60*y1(t-1) + 0.10*y2(t-1) + e1");
    println!("  y2(t) = 0.30 + 0.05*y1(t-1) + 0.70*y2(t-1) + e2\n");

    // -----------------------------------------------------------------------
    // 1. Fit a VAR(1) model
    // -----------------------------------------------------------------------
    println!("--- VAR(1) Fit ---");
    let mut model = VAR::new(1);
    model.fit(&[y1.clone(), y2.clone()]).unwrap();

    let intercepts = model.intercepts().unwrap();
    let coefs = model.coefficients().unwrap();

    println!("Estimated intercepts:");
    println!("  c1 = {:.4}  (true: 0.50)", intercepts[0]);
    println!("  c2 = {:.4}  (true: 0.30)", intercepts[1]);
    println!();
    println!("Estimated coefficient matrix A (lag 1):");
    println!(
        "  A[0][0] = {:.4}  (true: 0.60)   A[0][1] = {:.4}  (true: 0.10)",
        coefs[0][0][0], coefs[0][1][0]
    );
    println!(
        "  A[1][0] = {:.4}  (true: 0.05)   A[1][1] = {:.4}  (true: 0.70)",
        coefs[1][0][0], coefs[1][1][0]
    );

    // Show fitted values and residuals summary
    let residuals = model.residuals().unwrap();
    let rss1: f64 = residuals[0].iter().map(|r| r * r).sum();
    let rss2: f64 = residuals[1].iter().map(|r| r * r).sum();
    let n_eff = residuals[0].len();
    println!();
    println!("Residuals ({} effective observations):", n_eff);
    println!("  y1 RMSE: {:.6}", (rss1 / n_eff as f64).sqrt());
    println!("  y2 RMSE: {:.6}", (rss2 / n_eff as f64).sqrt());

    // -----------------------------------------------------------------------
    // 2. Granger causality test
    // -----------------------------------------------------------------------
    println!("\n--- Granger Causality Test ---");

    let f_y1_causes_y2 = model.granger_causality_test(0, 1).unwrap();
    let f_y2_causes_y1 = model.granger_causality_test(1, 0).unwrap();

    println!("H0: y1 does NOT Granger-cause y2");
    println!("  F-statistic: {:.4}", f_y1_causes_y2);
    println!(
        "  Interpretation: {}",
        if f_y1_causes_y2 > 3.84 {
            "Reject H0 -- y1 Granger-causes y2"
        } else {
            "Fail to reject H0"
        }
    );
    println!();
    println!("H0: y2 does NOT Granger-cause y1");
    println!("  F-statistic: {:.4}", f_y2_causes_y1);
    println!(
        "  Interpretation: {}",
        if f_y2_causes_y1 > 3.84 {
            "Reject H0 -- y2 Granger-causes y1"
        } else {
            "Fail to reject H0"
        }
    );
    println!();
    println!("(True DGP: y1->y2 coupling is 0.05, y2->y1 coupling is 0.10)");

    // -----------------------------------------------------------------------
    // 3. Multi-step forecasting
    // -----------------------------------------------------------------------
    println!("\n--- Multi-Step Forecast (10 steps) ---");
    let horizon = 10;
    let forecasts = model.predict(horizon).unwrap();

    println!("{:>4} {:>12} {:>12}", "h", "y1_hat", "y2_hat");
    println!("{:-<30}", "");
    for h in 0..horizon {
        println!(
            "{:>4} {:>12.4} {:>12.4}",
            h + 1,
            forecasts[0][h],
            forecasts[1][h]
        );
    }

    // -----------------------------------------------------------------------
    // 4. VAR(2) on a higher-order system
    // -----------------------------------------------------------------------
    println!("\n--- VAR(2) on Higher-Order Data ---");

    // Generate a VAR(2) process
    let n2 = 300;
    let mut z1 = vec![0.0; n2];
    let mut z2 = vec![0.0; n2];
    z1[0] = 0.1;
    z1[1] = 0.2;
    z2[0] = -0.1;
    z2[1] = 0.0;

    seed = 99;
    for t in 2..n2 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e1 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let e2 = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;

        z1[t] = 0.4 * z1[t - 1] + 0.1 * z1[t - 2] + 0.05 * z2[t - 1] + e1;
        z2[t] = 0.1 * z1[t - 1] + 0.3 * z2[t - 1] + 0.15 * z2[t - 2] + e2;
    }

    let mut var2 = VAR::new(2);
    var2.fit(&[z1, z2]).unwrap();

    let c2 = var2.coefficients().unwrap();
    println!("Equation for z1:");
    println!(
        "  z1(t-1): {:.4} (true: 0.40)   z1(t-2): {:.4} (true: 0.10)",
        c2[0][0][0], c2[0][0][1]
    );
    println!(
        "  z2(t-1): {:.4} (true: 0.05)   z2(t-2): {:.4} (true: 0.00)",
        c2[0][1][0], c2[0][1][1]
    );
    println!("Equation for z2:");
    println!(
        "  z1(t-1): {:.4} (true: 0.10)   z1(t-2): {:.4} (true: 0.00)",
        c2[1][0][0], c2[1][0][1]
    );
    println!(
        "  z2(t-1): {:.4} (true: 0.30)   z2(t-2): {:.4} (true: 0.15)",
        c2[1][1][0], c2[1][1][1]
    );

    let preds2 = var2.predict(5).unwrap();
    println!("\n5-step forecast:");
    for h in 0..5 {
        println!(
            "  h={}: z1={:.4}, z2={:.4}",
            h + 1,
            preds2[0][h],
            preds2[1][h]
        );
    }

    // -----------------------------------------------------------------------
    // 5. VARForecaster adapter (Forecaster trait)
    // -----------------------------------------------------------------------
    println!("\n--- VARForecaster (Forecaster Trait Adapter) ---");

    let timestamps: Vec<_> = (0..n)
        .map(|i| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64))
        .collect();

    // Build a TimeSeries with y1 as primary and y2 as a regressor
    let cal = CalendarAnnotations::new().with_regressor("y2".to_string(), y2.clone());
    let ts = TimeSeriesBuilder::new()
        .timestamps(timestamps)
        .values(y1.clone())
        .calendar(cal)
        .build()
        .unwrap();

    let mut var_fc = VARForecaster::new(1);
    var_fc.fit(&ts).unwrap();
    println!("Model name: {}", var_fc.name());
    println!("Is fitted: {}", var_fc.is_fitted());

    let fc = var_fc.predict(5).unwrap();
    println!("\nPoint forecasts (y1):");
    for (i, pred) in fc.primary().iter().enumerate() {
        println!("  h={}: {:.4}", i + 1, pred);
    }

    let fc_ci = var_fc.predict_with_intervals(5, 0.95).unwrap();
    let lower = fc_ci.lower_series(0).unwrap();
    let upper = fc_ci.upper_series(0).unwrap();
    let point = fc_ci.primary();

    println!("\nForecast with 95% CI:");
    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "h", "Lower", "Forecast", "Upper"
    );
    println!("{:-<38}", "");
    for i in 0..5 {
        println!(
            "{:>4} {:>10.4} {:>10.4} {:>10.4}",
            i + 1,
            lower[i],
            point[i],
            upper[i]
        );
    }

    // Show fitted values summary
    let fitted = var_fc.fitted_values().unwrap();
    let valid_fitted: Vec<f64> = fitted.iter().copied().filter(|v| v.is_finite()).collect();
    let mean_fitted: f64 = valid_fitted.iter().sum::<f64>() / valid_fitted.len() as f64;
    println!(
        "\nFitted values: {} total ({} valid, mean={:.4})",
        fitted.len(),
        valid_fitted.len(),
        mean_fitted
    );

    println!("\n=== VAR Example Complete ===");
}
