//! Kalman filter example.
//!
//! Run with: cargo run --example kalman

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::kalman::{KalmanFilter, StateSpaceModel};
use anofox_forecast::models::kalman_forecaster::KalmanForecaster;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn main() {
    println!("=== Kalman Filter Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Local level model on constant data with noise
    // -----------------------------------------------------------------------
    println!("--- Local Level Model ---");

    let true_level = 5.0;
    let n = 100;

    // Generate constant level + noise
    let mut seed: u64 = 42;
    let data_ll: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            vec![true_level + noise]
        })
        .collect();

    let model_ll = StateSpaceModel::local_level(1.0, 0.01);
    println!(
        "State-space model: F=[1], H=[1], Q=[0.01], R=[1.0]"
    );
    println!("True level: {:.1}, {} observations with noise\n", true_level, n);

    let mut kf_ll = KalmanFilter::new(model_ll.clone()).unwrap();
    let filtered_ll = kf_ll.filter(&data_ll).unwrap();

    // Show filtered state convergence
    println!("Filtered state convergence:");
    for &t in &[0, 4, 9, 24, 49, 99] {
        let state = &filtered_ll[t];
        let cov = state.covariance[0][0];
        println!(
            "  t={:>3}: state={:>7.4}, P={:.6}, innovation={:>7.4}",
            t, state.state[0], cov, state.innovation[0]
        );
    }

    println!(
        "\nFinal filtered level: {:.4} (true: {:.1})",
        filtered_ll[n - 1].state[0],
        true_level
    );
    println!(
        "Covariance decreased from {:.4} to {:.6}",
        filtered_ll[0].covariance[0][0],
        filtered_ll[n - 1].covariance[0][0]
    );

    // Total log-likelihood
    let kf_ll2 = KalmanFilter::new(model_ll).unwrap();
    let ll = kf_ll2.log_likelihood(&data_ll).unwrap();
    println!("Log-likelihood: {:.4}", ll);

    // -----------------------------------------------------------------------
    // 2. Local linear trend model on trending data
    // -----------------------------------------------------------------------
    println!("\n--- Local Linear Trend Model ---");

    let intercept = 2.0;
    let slope = 0.5;
    let n_trend = 200;

    seed = 123;
    let data_llt: Vec<Vec<f64>> = (0..n_trend)
        .map(|t| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 1.0;
            vec![intercept + slope * t as f64 + noise]
        })
        .collect();

    let model_llt = StateSpaceModel::local_linear_trend(0.5, 0.01, 0.01);
    println!(
        "True DGP: y(t) = {:.1} + {:.1}*t + noise, {} observations",
        intercept, slope, n_trend
    );

    let mut kf_llt = KalmanFilter::new(model_llt).unwrap();
    let filtered_llt = kf_llt.filter(&data_llt).unwrap();

    let last = &filtered_llt[n_trend - 1];
    let expected_level = intercept + slope * (n_trend - 1) as f64;
    println!("\nFiltered state at t={}:", n_trend - 1);
    println!(
        "  Level: {:.4} (expected: {:.1})",
        last.state[0], expected_level
    );
    println!("  Trend: {:.4} (true: {:.1})", last.state[1], slope);

    println!("\nState evolution at selected time steps:");
    println!(
        "{:>4} {:>10} {:>10} {:>12} {:>12}",
        "t", "Level", "Trend", "P(level)", "P(trend)"
    );
    println!("{:-<52}", "");
    for &t in &[0, 9, 49, 99, 149, 199] {
        if t < n_trend {
            let s = &filtered_llt[t];
            println!(
                "{:>4} {:>10.4} {:>10.4} {:>12.6} {:>12.6}",
                t, s.state[0], s.state[1], s.covariance[0][0], s.covariance[1][1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. Forward filtering
    // -----------------------------------------------------------------------
    println!("\n--- Forward Filtering (Innovation Diagnostics) ---");

    // Use the local level filter results to examine innovations
    let innovations: Vec<f64> = filtered_ll.iter().map(|s| s.innovation[0]).collect();
    let mean_inn: f64 = innovations.iter().sum::<f64>() / innovations.len() as f64;
    let var_inn: f64 = innovations
        .iter()
        .map(|v| (v - mean_inn).powi(2))
        .sum::<f64>()
        / innovations.len() as f64;

    println!("Innovation statistics (local level model):");
    println!("  Mean:     {:.6} (should be near 0)", mean_inn);
    println!("  Variance: {:.6}", var_inn);
    println!("  Std dev:  {:.6}", var_inn.sqrt());

    // Show predicted observations vs actual for first 10 steps
    println!("\nPredicted vs actual (first 10 steps):");
    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "t", "Actual", "Predicted", "Innovation"
    );
    println!("{:-<38}", "");
    for t in 0..10 {
        println!(
            "{:>4} {:>10.4} {:>10.4} {:>10.4}",
            t, data_ll[t][0], filtered_ll[t].predicted_obs[0], filtered_ll[t].innovation[0]
        );
    }

    // -----------------------------------------------------------------------
    // 4. Smoothing (Rauch-Tung-Striebel)
    // -----------------------------------------------------------------------
    println!("\n--- RTS Smoothing ---");

    let model_smooth = StateSpaceModel::local_level(1.0, 0.1);
    let mut kf_s = KalmanFilter::new(model_smooth.clone()).unwrap();
    let filtered_s = kf_s.filter(&data_ll).unwrap();

    let kf_s2 = KalmanFilter::new(model_smooth).unwrap();
    let smoothed = kf_s2.smooth(&filtered_s).unwrap();

    println!("Filtered vs smoothed state at selected time steps:");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>12}",
        "t", "Filtered", "P(filt)", "Smoothed", "P(smooth)"
    );
    println!("{:-<56}", "");
    for &t in &[0, 4, 9, 24, 49, 74, 99] {
        if t < n {
            println!(
                "{:>4} {:>12.4} {:>12.6} {:>12.4} {:>12.6}",
                t,
                filtered_s[t].state[0],
                filtered_s[t].covariance[0][0],
                smoothed[t].state[0],
                smoothed[t].covariance[0][0]
            );
        }
    }

    // Count how often smoother has smaller covariance
    let mut smoother_wins = 0;
    for t in 0..n {
        if smoothed[t].covariance[0][0] <= filtered_s[t].covariance[0][0] + 1e-12 {
            smoother_wins += 1;
        }
    }
    println!(
        "\nSmoother covariance <= filtered covariance: {}/{} time steps",
        smoother_wins, n
    );

    // -----------------------------------------------------------------------
    // 5. Multi-step prediction from Kalman filter
    // -----------------------------------------------------------------------
    println!("\n--- Multi-Step Prediction ---");

    let horizon = 10;
    let predictions = kf_ll.predict(horizon).unwrap();

    println!(
        "Predictions from local level model ({} steps ahead):",
        horizon
    );
    println!("{:>4} {:>12}", "h", "y_hat");
    println!("{:-<18}", "");
    for (h, pred) in predictions.iter().enumerate() {
        println!("{:>4} {:>12.4}", h + 1, pred[0]);
    }
    println!(
        "\n(Local level model predicts a constant -- the last filtered state)"
    );

    // Also predict from the trend model
    let preds_trend = kf_llt.predict(horizon).unwrap();
    println!(
        "\nPredictions from local linear trend model ({} steps ahead):",
        horizon
    );
    println!("{:>4} {:>12}", "h", "y_hat");
    println!("{:-<18}", "");
    for (h, pred) in preds_trend.iter().enumerate() {
        println!("{:>4} {:>12.4}", h + 1, pred[0]);
    }

    // -----------------------------------------------------------------------
    // 6. KalmanForecaster adapter (Forecaster trait)
    // -----------------------------------------------------------------------
    println!("\n--- KalmanForecaster (Forecaster Trait Adapter) ---");

    let timestamps: Vec<_> = (0..n)
        .map(|i| {
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64)
        })
        .collect();
    let values: Vec<f64> = data_ll.iter().map(|obs| obs[0]).collect();
    let ts = TimeSeries::univariate(timestamps.clone(), values).unwrap();

    // Local level forecaster
    println!("\nLocal level forecaster:");
    let mut kf_fc_ll = KalmanForecaster::local_level();
    kf_fc_ll.fit(&ts).unwrap();
    println!("  Model name: {}", kf_fc_ll.name());
    println!("  Is fitted: {}", kf_fc_ll.is_fitted());

    let fc_ll = kf_fc_ll.predict(5).unwrap();
    println!("  Point forecasts:");
    for (i, pred) in fc_ll.primary().iter().enumerate() {
        println!("    h={}: {:.4}", i + 1, pred);
    }

    // Local linear trend forecaster
    println!("\nLocal linear trend forecaster:");
    let trend_values: Vec<f64> = data_llt.iter().map(|obs| obs[0]).collect();
    let trend_timestamps: Vec<_> = (0..n_trend)
        .map(|i| {
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::hours(i as i64)
        })
        .collect();
    let ts_trend = TimeSeries::univariate(trend_timestamps, trend_values).unwrap();

    let mut kf_fc_llt = KalmanForecaster::local_linear_trend();
    kf_fc_llt.fit(&ts_trend).unwrap();

    let fc_llt = kf_fc_llt.predict(5).unwrap();
    println!("  Point forecasts:");
    for (i, pred) in fc_llt.primary().iter().enumerate() {
        println!("    h={}: {:.4}", i + 1, pred);
    }

    // Prediction intervals
    let fc_ci = kf_fc_llt.predict_with_intervals(5, 0.95).unwrap();
    let lower = fc_ci.lower_series(0).unwrap();
    let upper = fc_ci.upper_series(0).unwrap();
    let point = fc_ci.primary();

    println!("\n  Forecast with 95% CI (local linear trend):");
    println!(
        "  {:>4} {:>10} {:>10} {:>10}",
        "h", "Lower", "Forecast", "Upper"
    );
    println!("  {:-<38}", "");
    for i in 0..5 {
        println!(
            "  {:>4} {:>10.4} {:>10.4} {:>10.4}",
            i + 1,
            lower[i],
            point[i],
            upper[i]
        );
    }

    // Custom model forecaster
    println!("\nCustom model forecaster:");
    let custom_ssm = StateSpaceModel::local_level(0.5, 0.05);
    let mut kf_fc_custom = KalmanForecaster::with_model(custom_ssm);
    kf_fc_custom.fit(&ts).unwrap();

    let fc_custom = kf_fc_custom.predict(3).unwrap();
    println!("  Custom local level (R=0.5, Q=0.05):");
    for (i, pred) in fc_custom.primary().iter().enumerate() {
        println!("    h={}: {:.4}", i + 1, pred);
    }

    // Fitted values and residuals summary
    let fitted = kf_fc_ll.fitted_values().unwrap();
    let residuals = kf_fc_ll.residuals().unwrap();
    let mean_resid: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let rmse: f64 = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
    println!("\nFitted values and residuals (local level):");
    println!("  Observations: {}", fitted.len());
    println!("  Mean residual: {:.6}", mean_resid);
    println!("  RMSE: {:.6}", rmse);

    println!("\n=== Kalman Filter Example Complete ===");
}
