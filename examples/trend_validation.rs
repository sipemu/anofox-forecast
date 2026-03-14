//! Validation of all trend components on synthetic series.
//!
//! For each synthetic series, the true parameters are known. This script fits
//! each component and reports parameter recovery, R², and forecast accuracy.

use anofox_forecast::seasonality::{
    auto_trend::AutoTrend, ExponentialTrend, LogisticTrend, PiecewiseLinearTrend, PolynomialTrend,
    Recency, TheilSenTrend, TrendComponent,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              TREND COMPONENT VALIDATION RESULTS                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    validate_linear();
    validate_quadratic();
    validate_exponential();
    validate_logistic();
    validate_robust();
    validate_regime_change();
    validate_auto_selection();
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().min(predicted.len()) as f64;
    let ss: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    (ss / n).sqrt()
}

fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().min(predicted.len()) as f64;
    let s: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum();
    s / n
}

fn pseudo_noise(index: usize, seed: u64) -> f64 {
    let mut state = seed.wrapping_add(index as u64);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
}

fn get_feature(features: &[(&str, f64)], name: &str) -> f64 {
    features
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| *v)
        .unwrap_or(f64::NAN)
}

// ── 1. Linear data ─────────────────────────────────────────────────────

fn validate_linear() {
    println!("━━━ 1. LINEAR DATA: y = 2t + 10 + noise(σ=2) ━━━");
    println!("    True: slope=2.0, intercept=10.0\n");

    let n = 200;
    let values: Vec<f64> = (0..n)
        .map(|i| 2.0 * i as f64 + 10.0 + pseudo_noise(i, 42) * 2.0)
        .collect();

    // PolynomialTrend(1) — Full
    {
        let mut c = PolynomialTrend::new(1).with_recency(Recency::Full);
        c.fit_trend(&values).unwrap();
        let coefs = c.coefficients().unwrap();
        let r2 = get_feature(&c.trend_features(), "polynomial_r_squared");
        let forecast = c.predict_trend(20);
        let future: Vec<f64> = (n..n + 20).map(|i| 2.0 * i as f64 + 10.0).collect();
        println!(
            "    Poly(1) Full    | intercept={:>8.4}  slope={:>8.4}  R²={:.6}  forecast RMSE={:.4}",
            coefs[0],
            coefs[1],
            r2,
            rmse(&future, &forecast)
        );
    }

    // PolynomialTrend(1) — Recency 30%
    {
        let mut c = PolynomialTrend::new(1).with_recency(Recency::Fraction(0.3));
        c.fit_trend(&values).unwrap();
        let coefs = c.coefficients().unwrap();
        let r2 = get_feature(&c.trend_features(), "polynomial_r_squared");
        let forecast = c.predict_trend(20);
        let future: Vec<f64> = (n..n + 20).map(|i| 2.0 * i as f64 + 10.0).collect();
        println!(
            "    Poly(1) Rec30%  | intercept={:>8.4}  slope={:>8.4}  R²={:.6}  forecast RMSE={:.4}",
            coefs[0],
            coefs[1],
            r2,
            rmse(&future, &forecast)
        );
    }

    // TheilSen — Full
    {
        let mut c = TheilSenTrend::new().with_recency(Recency::Full);
        c.fit_trend(&values).unwrap();
        let r2 = get_feature(&c.trend_features(), "theilsen_r_squared");
        let forecast = c.predict_trend(20);
        let future: Vec<f64> = (n..n + 20).map(|i| 2.0 * i as f64 + 10.0).collect();
        println!(
            "    TheilSen Full   | intercept={:>8.4}  slope={:>8.4}  R²={:.6}  forecast RMSE={:.4}",
            c.intercept(),
            c.slope(),
            r2,
            rmse(&future, &forecast)
        );
    }

    // TheilSen with outliers
    {
        let mut contaminated = values.clone();
        contaminated[50] = 5000.0; // massive outlier
        contaminated[100] = -3000.0;
        contaminated[150] = 8000.0;

        let mut ols = PolynomialTrend::new(1).with_recency(Recency::Full);
        ols.fit_trend(&contaminated).unwrap();
        let ols_slope = ols.coefficients().unwrap()[1];

        let mut ts = TheilSenTrend::new().with_recency(Recency::Full);
        ts.fit_trend(&contaminated).unwrap();

        println!("\n    [Robustness] 3 extreme outliers injected (5000, -3000, 8000):");
        println!(
            "      OLS slope:      {:>8.4}  (true=2.0, error={:.4})",
            ols_slope,
            (ols_slope - 2.0).abs()
        );
        println!(
            "      TheilSen slope: {:>8.4}  (true=2.0, error={:.4})",
            ts.slope(),
            (ts.slope() - 2.0).abs()
        );
    }

    println!();
}

// ── 2. Quadratic data ───────────────────────────────────────────────────

fn validate_quadratic() {
    println!("━━━ 2. QUADRATIC DATA: y = 0.01t² + 0.5t + 10 + noise(σ=1) ━━━");
    println!("    True: a₀=10.0, a₁=0.5, a₂=0.01\n");

    let n = 200;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            0.01 * t * t + 0.5 * t + 10.0 + pseudo_noise(i, 17) * 1.0
        })
        .collect();

    let mut c = PolynomialTrend::new(2).with_recency(Recency::Full);
    c.fit_trend(&values).unwrap();
    let coefs = c.coefficients().unwrap();
    let r2 = get_feature(&c.trend_features(), "polynomial_r_squared");
    let forecast = c.predict_trend(20);
    let future: Vec<f64> = (n..n + 20)
        .map(|i| {
            let t = i as f64;
            0.01 * t * t + 0.5 * t + 10.0
        })
        .collect();

    println!(
        "    Poly(2) Full    | a₀={:>8.4}  a₁={:>8.4}  a₂={:>8.6}  R²={:.6}",
        coefs[0], coefs[1], coefs[2], r2
    );
    println!(
        "    Forecast        | RMSE={:.4}  MAE={:.4}",
        rmse(&future, &forecast),
        mae(&future, &forecast)
    );

    // Compare with linear fit
    let mut lin = PolynomialTrend::new(1).with_recency(Recency::Full);
    lin.fit_trend(&values).unwrap();
    let lin_r2 = get_feature(&lin.trend_features(), "polynomial_r_squared");
    let lin_forecast = lin.predict_trend(20);
    println!(
        "    Poly(1) Full    | R²={:.6}  forecast RMSE={:.4}  (misspecified — worse)",
        lin_r2,
        rmse(&future, &lin_forecast)
    );

    println!();
}

// ── 3. Exponential data ─────────────────────────────────────────────────

fn validate_exponential() {
    println!("━━━ 3. EXPONENTIAL DATA: y = 2·exp(0.05t) + noise(σ=0.5) ━━━");
    println!("    True: a=ln(2)≈0.6931, b=0.05\n");

    let n = 100;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            2.0 * (0.05 * t).exp() + pseudo_noise(i, 7).abs() * 0.5
        })
        .collect();

    let mut c = ExponentialTrend::new().with_recency(Recency::Full);
    c.fit_trend(&values).unwrap();
    let r2 = get_feature(&c.trend_features(), "exponential_r_squared");
    let forecast = c.predict_trend(10);
    let future: Vec<f64> = (n..n + 10).map(|i| 2.0 * (0.05 * i as f64).exp()).collect();

    println!(
        "    ExpTrend Full   | growth_rate={:.6} (true=0.05)  R²={:.6}",
        c.growth_rate(),
        r2
    );
    println!(
        "    Forecast        | RMSE={:.4}  MAE={:.4}",
        rmse(&future, &forecast),
        mae(&future, &forecast)
    );

    // With recency
    let mut c_rec = ExponentialTrend::new().with_recency(Recency::Fraction(0.3));
    c_rec.fit_trend(&values).unwrap();
    let r2_rec = get_feature(&c_rec.trend_features(), "exponential_r_squared");
    let forecast_rec = c_rec.predict_trend(10);
    println!(
        "    ExpTrend Rec30% | growth_rate={:.6} (true=0.05)  R²={:.6}  forecast RMSE={:.4}",
        c_rec.growth_rate(),
        r2_rec,
        rmse(&future, &forecast_rec)
    );

    println!();
}

// ── 4. Logistic data ────────────────────────────────────────────────────

fn validate_logistic() {
    println!("━━━ 4. LOGISTIC DATA: y = 100/(1+exp(-0.1(t-50))) + noise(σ=2) ━━━");
    println!("    True: K=100, midpoint=50, steepness=0.1\n");

    let n = 100;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 / (1.0 + (-0.1 * (t - 50.0)).exp()) + pseudo_noise(i, 99) * 2.0
        })
        .collect();

    // Fixed capacity
    let mut c = LogisticTrend::new()
        .with_capacity(100.0)
        .with_recency(Recency::Full);
    c.fit_trend(&values).unwrap();

    println!(
        "    Fixed K=100     | midpoint={:>7.2} (true=50)  steepness={:.4} (true=0.1)",
        c.midpoint(),
        c.steepness()
    );

    let forecast = c.predict_trend(50);
    println!(
        "    Saturation      | y(100)={:.2}  y(120)={:.2}  y(149)={:.2}  (all ≈ 100.0)",
        forecast[0], forecast[20], forecast[49]
    );

    // Auto capacity
    let mut c_auto = LogisticTrend::new().with_recency(Recency::Full);
    c_auto.fit_trend(&values).unwrap();
    println!(
        "    Auto capacity   | K={:.2} (true=100)  midpoint={:.2}  steepness={:.4}",
        c_auto.capacity(),
        c_auto.midpoint(),
        c_auto.steepness()
    );

    println!();
}

// ── 5. Robustness to outliers ───────────────────────────────────────────

fn validate_robust() {
    println!("━━━ 5. ROBUSTNESS: Linear + 10% outliers ━━━");
    println!("    True: slope=1.0, intercept=5.0\n");

    let n = 200;
    let mut values: Vec<f64> = (0..n).map(|i| 1.0 * i as f64 + 5.0).collect();

    // Inject 10% outliers
    for i in (0..n).step_by(10) {
        values[i] += if i % 20 == 0 { 500.0 } else { -500.0 };
    }

    let mut ols = PolynomialTrend::new(1).with_recency(Recency::Full);
    ols.fit_trend(&values).unwrap();
    let ols_coefs = ols.coefficients().unwrap();

    let mut ts = TheilSenTrend::new().with_recency(Recency::Full);
    ts.fit_trend(&values).unwrap();

    let clean: Vec<f64> = (0..n).map(|i| 1.0 * i as f64 + 5.0).collect();
    let ols_fitted = ols.fitted_trend();
    let ts_fitted = ts.fitted_trend();

    println!(
        "    OLS (Poly1)     | slope={:>8.4} intercept={:>8.4}  in-sample RMSE={:.4}",
        ols_coefs[1],
        ols_coefs[0],
        rmse(&clean, ols_fitted)
    );
    println!(
        "    TheilSen        | slope={:>8.4} intercept={:>8.4}  in-sample RMSE={:.4}",
        ts.slope(),
        ts.intercept(),
        rmse(&clean, ts_fitted)
    );

    let future: Vec<f64> = (n..n + 20).map(|i| 1.0 * i as f64 + 5.0).collect();
    println!(
        "    Forecast RMSE   | OLS={:.4}  TheilSen={:.4}  (TheilSen should be better)",
        rmse(&future, &ols.predict_trend(20)),
        rmse(&future, &ts.predict_trend(20))
    );

    println!();
}

// ── 6. Regime change with recency ───────────────────────────────────────

fn validate_regime_change() {
    println!("━━━ 6. REGIME CHANGE: slope +3 (t<60), slope -2 (t≥60) ━━━");
    println!("    Recent true slope = -2.0\n");

    let n = 100;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            if i < 60 {
                3.0 * t + 10.0
            } else {
                3.0 * 60.0 + 10.0 - 2.0 * (t - 60.0)
            }
        })
        .collect();

    // Full data
    let mut full = PolynomialTrend::new(1).with_recency(Recency::Full);
    full.fit_trend(&values).unwrap();
    let full_slope = full.coefficients().unwrap()[1];

    // Recency 30%
    let mut recent = PolynomialTrend::new(1).with_recency(Recency::Fraction(0.3));
    recent.fit_trend(&values).unwrap();
    let recent_slope = recent.coefficients().unwrap()[1];

    // Recency 20 observations
    let mut win = PolynomialTrend::new(1).with_recency(Recency::Window(20));
    win.fit_trend(&values).unwrap();
    let win_slope = win.coefficients().unwrap()[1];

    // Piecewise
    let mut pw = PiecewiseLinearTrend::new()
        .with_penalty(5.0)
        .with_recency(Recency::Full);
    pw.fit_trend(&values).unwrap();
    let n_seg = pw.segments().map(|s| s.len()).unwrap_or(0);
    let last_slope = pw
        .segments()
        .and_then(|s| s.last())
        .map(|s| s.slope)
        .unwrap_or(f64::NAN);

    println!(
        "    Poly(1) Full    | slope={:>8.4}  (misleading — averages both regimes)",
        full_slope
    );
    println!(
        "    Poly(1) Rec30%  | slope={:>8.4}  (captures recent slope ≈ -2.0)",
        recent_slope
    );
    println!(
        "    Poly(1) Win(20) | slope={:>8.4}  (window=last 20 obs)",
        win_slope
    );
    println!(
        "    Piecewise Full  | {} segments, last slope={:.4}",
        n_seg, last_slope
    );

    // Forecast comparison
    let future: Vec<f64> = (n..n + 20)
        .map(|i| 3.0 * 60.0 + 10.0 - 2.0 * (i as f64 - 60.0))
        .collect();
    println!(
        "\n    Forecast RMSE   | Full={:.4}  Rec30%={:.4}  Win20={:.4}  Piecewise={:.4}",
        rmse(&future, &full.predict_trend(20)),
        rmse(&future, &recent.predict_trend(20)),
        rmse(&future, &win.predict_trend(20)),
        rmse(&future, &pw.predict_trend(20))
    );
    println!("    (Recency-aware methods should have lower forecast RMSE)");

    println!();
}

// ── 7. AutoTrend correct selection ──────────────────────────────────────

fn validate_auto_selection() {
    println!("━━━ 7. AUTOTREND SELECTION ACCURACY ━━━\n");

    let scenarios: Vec<(&str, &str, Vec<f64>)> = vec![
        (
            "Linear",
            "Linear or TheilSen",
            (0..200)
                .map(|i| 2.0 * i as f64 + 10.0 + pseudo_noise(i, 1) * 3.0)
                .collect(),
        ),
        (
            "Quadratic",
            "Quadratic",
            (0..200)
                .map(|i| {
                    let t = i as f64;
                    0.01 * t * t + 0.5 * t + 10.0 + pseudo_noise(i, 2) * 2.0
                })
                .collect(),
        ),
        (
            "Exponential",
            "Exponential",
            (0..150)
                .map(|i| 2.0 * (0.05 * i as f64).exp() + pseudo_noise(i, 3).abs() * 0.5)
                .collect(),
        ),
        (
            "Constant",
            "Linear or TheilSen",
            (0..200).map(|i| 50.0 + pseudo_noise(i, 4) * 1.0).collect(),
        ),
    ];

    for (true_type, expected, values) in &scenarios {
        let mut auto = AutoTrend::new().with_recency(Recency::Full);
        auto.fit_trend(values).unwrap();
        let result = auto.selection_result().unwrap();

        let correct = expected.split(" or ").any(|e| e == result.selected);
        let mark = if correct { "PASS" } else { "MISS" };

        println!(
            "    [{mark}] {true_type:<12} -> selected: {:<16}  (expected: {expected})",
            result.selected
        );
        print!("         scores: ");
        for (i, (name, score)) in result.scores.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}={:.1}", name, score);
        }
        println!();
    }

    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  All validations complete.");
    println!("══════════════════════════════════════════════════════════════════════");
}
