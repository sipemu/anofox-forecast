//! Model Diagnostics Example
//!
//! This example demonstrates diagnostic tests for validating
//! time series models and checking statistical assumptions.
//!
//! Run with: cargo run --example diagnostics

use anofox_forecast::validation::{
    adf_test, box_pierce, durbin_watson, kpss_test, ljung_box, test_stationarity,
    AutocorrelationType,
};

fn main() {
    println!("=== Model Diagnostics Example ===\n");

    // =========================================================================
    // Residual Analysis with Ljung-Box Test
    // =========================================================================
    println!("--- Ljung-Box Test (Residual Autocorrelation) ---\n");

    // Good residuals: approximately white noise
    let good_residuals: Vec<f64> = (0..100)
        .map(|i| ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
        .collect();

    println!("Testing residuals for autocorrelation...\n");

    let lb_result = ljung_box(&good_residuals, Some(10), 0);
    println!("Ljung-Box Test (white noise residuals):");
    println!("  Q statistic: {:.4}", lb_result.statistic);
    println!("  p-value:     {:.4}", lb_result.p_value);
    println!("  Lags tested: {}", lb_result.lags);
    println!("  Degrees of freedom: {}", lb_result.df);

    if lb_result.is_white_noise(0.05) {
        println!("  Result: PASS - Residuals appear to be white noise (p > 0.05)");
        println!("  Interpretation: No significant autocorrelation detected");
    } else {
        println!("  Result: FAIL - Significant autocorrelation detected (p <= 0.05)");
        println!("  Interpretation: Model may be missing patterns in the data");
    }

    // Bad residuals: autocorrelated
    println!("\nLjung-Box Test (autocorrelated residuals):");
    let mut bad_residuals = vec![0.0; 100];
    bad_residuals[0] = 1.0;
    for i in 1..100 {
        bad_residuals[i] = 0.8 * bad_residuals[i - 1] + 0.2 * ((i * 17) % 23) as f64 / 23.0;
    }

    let lb_bad = ljung_box(&bad_residuals, Some(10), 0);
    println!("  Q statistic: {:.4}", lb_bad.statistic);
    println!("  p-value:     {:.4}", lb_bad.p_value);

    if lb_bad.is_white_noise(0.05) {
        println!("  Result: PASS");
    } else {
        println!("  Result: FAIL - Significant autocorrelation detected");
        println!("  Action: Consider adding AR terms or differencing");
    }

    // =========================================================================
    // Box-Pierce Test (simpler alternative)
    // =========================================================================
    println!("\n--- Box-Pierce Test ---\n");

    let bp_result = box_pierce(&good_residuals, Some(10));
    println!("Box-Pierce test (simpler Ljung-Box variant):");
    println!("  Q statistic: {:.4}", bp_result.statistic);
    println!("  p-value:     {:.4}", bp_result.p_value);
    println!("\nNote: Ljung-Box has small-sample correction, preferred for n < 100");

    // =========================================================================
    // Durbin-Watson Test
    // =========================================================================
    println!("\n--- Durbin-Watson Test (First-Order Autocorrelation) ---\n");

    let dw_good = durbin_watson(&good_residuals);
    println!("Durbin-Watson (white noise):");
    println!("  Statistic: {:.4}", dw_good.statistic);
    println!(
        "  Interpretation: {:?}",
        match dw_good.interpretation {
            AutocorrelationType::None => "No autocorrelation (ideal)",
            AutocorrelationType::PositiveWeak => "Weak positive autocorrelation",
            AutocorrelationType::PositiveStrong => "Strong positive autocorrelation",
            AutocorrelationType::NegativeWeak => "Weak negative autocorrelation",
            AutocorrelationType::NegativeStrong => "Strong negative autocorrelation",
        }
    );

    // Positive autocorrelation example
    let mut pos_autocorr = vec![0.0; 50];
    pos_autocorr[0] = 1.0;
    for i in 1..50 {
        pos_autocorr[i] = 0.9 * pos_autocorr[i - 1];
    }

    let dw_pos = durbin_watson(&pos_autocorr);
    println!("\nDurbin-Watson (positive autocorrelation):");
    println!("  Statistic: {:.4}", dw_pos.statistic);
    println!("  Interpretation: {:?}", dw_pos.interpretation);

    // Negative autocorrelation example
    let neg_autocorr: Vec<f64> = (0..50)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let dw_neg = durbin_watson(&neg_autocorr);
    println!("\nDurbin-Watson (negative autocorrelation):");
    println!("  Statistic: {:.4}", dw_neg.statistic);
    println!("  Interpretation: {:?}", dw_neg.interpretation);

    println!("\nDurbin-Watson interpretation guide:");
    println!("  DW ≈ 0: Strong positive autocorrelation");
    println!("  DW ≈ 2: No autocorrelation (ideal)");
    println!("  DW ≈ 4: Strong negative autocorrelation");

    // =========================================================================
    // ADF Test (Stationarity)
    // =========================================================================
    println!("\n--- ADF Test (Augmented Dickey-Fuller) ---\n");

    // Stationary series (white noise around mean)
    let stationary: Vec<f64> = (0..200)
        .map(|i| 10.0 + ((i * 17 + 13) % 97) as f64 / 50.0 - 1.0)
        .collect();

    let adf_stat = adf_test(&stationary, None);
    println!("ADF Test (stationary series):");
    println!("  Test statistic: {:.4}", adf_stat.statistic);
    println!("  p-value:        {:.4}", adf_stat.p_value);
    println!("  Lags used:      {}", adf_stat.lags);
    println!("  Critical values:");
    println!("    1%:  {:.3}", adf_stat.critical_values.cv_1pct);
    println!("    5%:  {:.3}", adf_stat.critical_values.cv_5pct);
    println!("    10%: {:.3}", adf_stat.critical_values.cv_10pct);
    println!(
        "  Conclusion: {}",
        if adf_stat.is_stationary {
            "STATIONARY (reject unit root)"
        } else {
            "NON-STATIONARY (fail to reject unit root)"
        }
    );

    // Non-stationary series (random walk)
    let mut random_walk = vec![0.0; 200];
    for i in 1..200 {
        random_walk[i] = random_walk[i - 1] + ((i * 17) % 19) as f64 / 10.0 - 0.9;
    }

    let adf_rw = adf_test(&random_walk, None);
    println!("\nADF Test (random walk - non-stationary):");
    println!("  Test statistic: {:.4}", adf_rw.statistic);
    println!("  p-value:        {:.4}", adf_rw.p_value);
    println!(
        "  Conclusion: {}",
        if adf_rw.is_stationary {
            "STATIONARY"
        } else {
            "NON-STATIONARY (as expected for random walk)"
        }
    );

    // Trending series
    let trending: Vec<f64> = (0..200)
        .map(|i| i as f64 * 0.5 + ((i * 13) % 7) as f64 * 0.1)
        .collect();

    let adf_trend = adf_test(&trending, None);
    println!("\nADF Test (trending series):");
    println!("  Test statistic: {:.4}", adf_trend.statistic);
    println!("  p-value:        {:.4}", adf_trend.p_value);
    println!(
        "  Conclusion: {}",
        if adf_trend.is_stationary {
            "STATIONARY"
        } else {
            "NON-STATIONARY (trend present)"
        }
    );

    // =========================================================================
    // KPSS Test (Stationarity)
    // =========================================================================
    println!("\n--- KPSS Test ---\n");

    println!("Note: KPSS null hypothesis is OPPOSITE of ADF");
    println!("  ADF H0:  Series has unit root (non-stationary)");
    println!("  KPSS H0: Series is stationary");
    println!();

    let kpss_stat = kpss_test(&stationary, None);
    println!("KPSS Test (stationary series):");
    println!("  Test statistic: {:.4}", kpss_stat.statistic);
    println!("  p-value:        {:.4}", kpss_stat.p_value);
    println!("  Lags used:      {}", kpss_stat.lags);
    println!("  Critical values:");
    println!("    1%:  {:.3}", kpss_stat.critical_values.cv_1pct);
    println!("    5%:  {:.3}", kpss_stat.critical_values.cv_5pct);
    println!("    10%: {:.3}", kpss_stat.critical_values.cv_10pct);
    println!(
        "  Conclusion: {}",
        if kpss_stat.is_stationary {
            "STATIONARY (fail to reject)"
        } else {
            "NON-STATIONARY (reject stationarity)"
        }
    );

    let kpss_trend = kpss_test(&trending, None);
    println!("\nKPSS Test (trending series):");
    println!("  Test statistic: {:.4}", kpss_trend.statistic);
    println!("  p-value:        {:.4}", kpss_trend.p_value);
    println!(
        "  Conclusion: {}",
        if kpss_trend.is_stationary {
            "STATIONARY"
        } else {
            "NON-STATIONARY (as expected for trending data)"
        }
    );

    // =========================================================================
    // Combined Stationarity Test
    // =========================================================================
    println!("\n--- Combined ADF + KPSS Test ---\n");

    println!("Using both tests together provides more robust conclusions:\n");

    let (adf, kpss, conclusion) = test_stationarity(&stationary);
    println!("Stationary series:");
    println!(
        "  ADF:  statistic={:.4}, stationary={}",
        adf.statistic, adf.is_stationary
    );
    println!(
        "  KPSS: statistic={:.4}, stationary={}",
        kpss.statistic, kpss.is_stationary
    );
    println!("  Combined conclusion: {}", conclusion);

    let (adf, kpss, conclusion) = test_stationarity(&random_walk);
    println!("\nRandom walk:");
    println!(
        "  ADF:  statistic={:.4}, stationary={}",
        adf.statistic, adf.is_stationary
    );
    println!(
        "  KPSS: statistic={:.4}, stationary={}",
        kpss.statistic, kpss.is_stationary
    );
    println!("  Combined conclusion: {}", conclusion);

    let (adf, kpss, conclusion) = test_stationarity(&trending);
    println!("\nTrending series:");
    println!(
        "  ADF:  statistic={:.4}, stationary={}",
        adf.statistic, adf.is_stationary
    );
    println!(
        "  KPSS: statistic={:.4}, stationary={}",
        kpss.statistic, kpss.is_stationary
    );
    println!("  Combined conclusion: {}", conclusion);

    // =========================================================================
    // Interpretation Guide
    // =========================================================================
    println!("\n--- Combined Test Interpretation ---\n");

    println!("ADF rejects + KPSS fails to reject → STATIONARY");
    println!("  Both tests agree the series is stationary");
    println!();

    println!("ADF fails to reject + KPSS rejects → NON-STATIONARY");
    println!("  Both tests agree the series is non-stationary");
    println!();

    println!("Other combinations → INCONCLUSIVE");
    println!("  Tests disagree, may need more data or different tests");

    // =========================================================================
    // Practical Workflow
    // =========================================================================
    println!("\n--- Diagnostic Workflow ---\n");

    println!("Before modeling:");
    println!("  1. Test stationarity with ADF + KPSS");
    println!("  2. If non-stationary, apply differencing or transformations");
    println!("  3. Re-test until stationary");
    println!();

    println!("After model fitting:");
    println!("  1. Extract residuals from fitted model");
    println!("  2. Apply Ljung-Box test for autocorrelation");
    println!("  3. Apply Durbin-Watson for first-order autocorrelation");
    println!("  4. If tests fail, consider:");
    println!("     - Adding AR/MA terms");
    println!("     - Including seasonality");
    println!("     - Using a different model class");
    println!();

    println!("Significance levels:");
    println!("  α = 0.05 is standard");
    println!("  α = 0.01 for stricter testing");
    println!("  α = 0.10 for more lenient testing");
}
