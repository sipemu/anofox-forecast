//! Sensitivity analysis: what happens when trend changes at different points
//! and the recency window doesn't match?

use anofox_forecast::seasonality::{
    AutoRecencyConfig, PolynomialTrend, Recency, TheilSenTrend, TrendComponent,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║          RECENCY SENSITIVITY ANALYSIS                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Series: 200 points, slope +3 then slope -2
    // Changepoint moves: at 60%, 80%, 90%, 95% of the series
    let n = 200;
    let change_points = vec![
        (0.60, "60% (t=120)"),
        (0.80, "80% (t=160)"),
        (0.90, "90% (t=180)"),
        (0.95, "95% (t=190)"),
    ];

    let recency_fractions = vec![0.50, 0.30, 0.20, 0.10, 0.05];

    println!("  Series: slope=+3 before changepoint, slope=-2 after. True recent slope = -2.0");
    println!("  Goal: recover slope=-2.0 for forecasting.\n");

    // Header
    print!("  {:>18}", "Changepoint");
    for &frac in &recency_fractions {
        print!("  Rec {:>3.0}%", frac * 100.0);
    }
    println!("    Full    Window(10)");
    print!("  {:>18}", "");
    for _ in &recency_fractions {
        print!("  {:>8}", "slope");
    }
    println!("  {:>8}  {:>8}", "slope", "slope");
    println!("  {}", "─".repeat(18 + (recency_fractions.len() + 2) * 10));

    for &(cp_frac, label) in &change_points {
        let cp = (n as f64 * cp_frac) as usize;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                if i < cp {
                    3.0 * t + 10.0
                } else {
                    3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
                }
            })
            .collect();

        print!("  {:>18}", label);

        for &frac in &recency_fractions {
            let mut c = PolynomialTrend::new(1).with_recency(Recency::Fraction(frac));
            c.fit_trend(&values).unwrap();
            let slope = c.coefficients().unwrap()[1];
            let color = slope_quality(slope, -2.0, cp_frac, frac);
            print!("  {:>8}", format!("{}{:.2}{}", color.0, slope, color.1));
        }

        // Full
        {
            let mut c = PolynomialTrend::new(1).with_recency(Recency::Full);
            c.fit_trend(&values).unwrap();
            let slope = c.coefficients().unwrap()[1];
            print!("  {:>8}", format!("{:.2}", slope));
        }

        // Window(10)
        {
            let mut c = PolynomialTrend::new(1).with_recency(Recency::Window(10));
            c.fit_trend(&values).unwrap();
            let slope = c.coefficients().unwrap()[1];
            print!("  {:>8}", format!("{:.2}", slope));
        }

        println!();
    }

    println!("\n  True recent slope = -2.00 in all cases.");
    println!("  Values closer to -2.00 = better. Values near +3.00 or mixed = window missed the change.\n");

    // ── Part 2: Forecast RMSE ──────────────────────────────────────────

    println!("━━━ FORECAST RMSE (20-step ahead) ━━━\n");

    print!("  {:>18}", "Changepoint");
    for &frac in &recency_fractions {
        print!("  Rec {:>3.0}%", frac * 100.0);
    }
    println!("    Full    Window(10)");
    println!("  {}", "─".repeat(18 + (recency_fractions.len() + 2) * 10));

    for &(cp_frac, label) in &change_points {
        let cp = (n as f64 * cp_frac) as usize;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                if i < cp {
                    3.0 * t + 10.0
                } else {
                    3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
                }
            })
            .collect();

        // Future continues the -2 slope
        let future: Vec<f64> = (n..n + 20)
            .map(|i| {
                let t = i as f64;
                3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
            })
            .collect();

        print!("  {:>18}", label);

        for &frac in &recency_fractions {
            let mut c = PolynomialTrend::new(1).with_recency(Recency::Fraction(frac));
            c.fit_trend(&values).unwrap();
            let forecast = c.predict_trend(20);
            let rmse = rmse_fn(&future, &forecast);
            print!("  {:>8}", format!("{:.1}", rmse));
        }

        // Full
        {
            let mut c = PolynomialTrend::new(1).with_recency(Recency::Full);
            c.fit_trend(&values).unwrap();
            let forecast = c.predict_trend(20);
            print!("  {:>8}", format!("{:.1}", rmse_fn(&future, &forecast)));
        }

        // Window(10)
        {
            let mut c = PolynomialTrend::new(1).with_recency(Recency::Window(10));
            c.fit_trend(&values).unwrap();
            let forecast = c.predict_trend(20);
            print!("  {:>8}", format!("{:.1}", rmse_fn(&future, &forecast)));
        }

        println!();
    }

    println!();

    // ── Part 3: TheilSen robustness across recency windows ─────────────

    println!("━━━ THEILSEN SLOPE vs POLYNOMIAL SLOPE (regime change at 90%) ━━━\n");
    println!("  Series: 200 pts, slope +3 until t=180, slope -2 after.\n");

    let cp = 180;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            if i < cp {
                3.0 * t + 10.0
            } else {
                3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
            }
        })
        .collect();

    println!("  {:>12}  {:>12}  {:>12}", "Recency", "Poly(1)", "TheilSen");
    println!("  {}", "─".repeat(40));

    for &frac in &[1.0, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05] {
        let rec = if frac >= 1.0 {
            Recency::Full
        } else {
            Recency::Fraction(frac)
        };
        let label = if frac >= 1.0 {
            "Full".to_string()
        } else {
            format!("{:.0}%", frac * 100.0)
        };

        let mut poly = PolynomialTrend::new(1).with_recency(rec.clone());
        poly.fit_trend(&values).unwrap();
        let poly_slope = poly.coefficients().unwrap()[1];

        let mut ts = TheilSenTrend::new().with_recency(rec);
        ts.fit_trend(&values).unwrap();

        println!(
            "  {:>12}  {:>12.4}  {:>12.4}",
            label,
            poly_slope,
            ts.slope()
        );
    }

    println!("\n  At 10% recency (20 obs), both capture -2.0 perfectly since");
    println!("  the window falls entirely within the new regime.\n");

    // ── Part 4: The fundamental tradeoff ───────────────────────────────

    println!("━━━ THE TRADEOFF: narrow window = responsive but noisy ━━━\n");
    println!("  Linear series y=2t+10 with noise σ=5. No regime change.\n");

    let noisy: Vec<f64> = (0..n)
        .map(|i| 2.0 * i as f64 + 10.0 + pseudo_noise(i, 42) * 5.0)
        .collect();

    let future_true: Vec<f64> = (n..n + 20).map(|i| 2.0 * i as f64 + 10.0).collect();

    println!(
        "  {:>12}  {:>12}  {:>12}  {:>12}",
        "Recency", "Slope", "Slope Error", "Forecast RMSE"
    );
    println!("  {}", "─".repeat(52));

    for &frac in &[1.0, 0.50, 0.30, 0.20, 0.10, 0.05] {
        let rec = if frac >= 1.0 {
            Recency::Full
        } else {
            Recency::Fraction(frac)
        };
        let label = if frac >= 1.0 {
            "Full".to_string()
        } else {
            format!("{:.0}%", frac * 100.0)
        };

        let mut c = PolynomialTrend::new(1).with_recency(rec);
        c.fit_trend(&noisy).unwrap();
        let slope = c.coefficients().unwrap()[1];
        let forecast = c.predict_trend(20);

        println!(
            "  {:>12}  {:>12.4}  {:>12.4}  {:>12.4}",
            label,
            slope,
            (slope - 2.0).abs(),
            rmse_fn(&future_true, &forecast)
        );
    }

    println!("\n  Narrower windows have more slope estimation error when there's no");
    println!("  actual regime change — this is the bias-variance tradeoff.\n");

    // ── Part 5: Recency::Auto — changepoint-driven recency ──────────

    println!("━━━ RECENCY::AUTO — CHANGEPOINT-DRIVEN WINDOW ━━━\n");
    println!("  Recency::Auto runs PELT to find the last changepoint and fits from there.\n");

    println!(
        "  {:>18}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Changepoint", "Auto slope", "30% slope", "Auto RMSE", "30% RMSE"
    );
    println!("  {}", "─".repeat(72));

    for &(cp_frac, label) in &change_points {
        let cp = (n as f64 * cp_frac) as usize;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                if i < cp {
                    3.0 * t + 10.0
                } else {
                    3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
                }
            })
            .collect();

        let future: Vec<f64> = (n..n + 20)
            .map(|i| {
                let t = i as f64;
                3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
            })
            .collect();

        // Auto recency
        let mut auto = PolynomialTrend::new(1).with_recency(Recency::auto());
        auto.fit_trend(&values).unwrap();
        let auto_slope = auto.coefficients().unwrap()[1];
        let auto_forecast = auto.predict_trend(20);
        let auto_rmse = rmse_fn(&future, &auto_forecast);

        // Show detected changepoints
        let detected = Recency::auto().detect_changepoints(&values);
        let cp_str = match &detected {
            Some(cps) if !cps.is_empty() => format!("detected at {:?}", cps),
            Some(_) => "none detected".to_string(),
            None => "n/a".to_string(),
        };

        // Fixed 30%
        let mut fixed = PolynomialTrend::new(1).with_recency(Recency::Fraction(0.3));
        fixed.fit_trend(&values).unwrap();
        let fixed_slope = fixed.coefficients().unwrap()[1];
        let fixed_forecast = fixed.predict_trend(20);
        let fixed_rmse = rmse_fn(&future, &fixed_forecast);

        println!(
            "  {:>18}  {:>12.2}  {:>12.2}  {:>12.1}  {:>12.1}",
            label, auto_slope, fixed_slope, auto_rmse, fixed_rmse
        );
        println!("  {:>18}  {}", "", cp_str);
    }

    // Demo with custom auto config
    println!("\n  Custom config: Recency::Auto with lower penalty (more sensitive):");
    let custom_config = AutoRecencyConfig {
        fallback_fraction: 0.3,
        penalty: Some(5.0), // lower = more changepoints detected
        ..AutoRecencyConfig::default()
    };
    let cp = 190; // 95% changepoint
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            if i < cp {
                3.0 * t + 10.0
            } else {
                3.0 * cp as f64 + 10.0 - 2.0 * (t - cp as f64)
            }
        })
        .collect();

    let detected = Recency::Auto(custom_config.clone()).detect_changepoints(&values);
    println!(
        "    penalty=5.0: changepoints={:?}",
        detected.unwrap_or_default()
    );

    let mut c = PolynomialTrend::new(1).with_recency(Recency::Auto(custom_config));
    c.fit_trend(&values).unwrap();
    println!(
        "    slope={:.2} (target=-2.00)\n",
        c.coefficients().unwrap()[1]
    );

    println!("══════════════════════════════════════════════════════════════════════════");
    println!("  CONCLUSION:");
    println!("    - Recency::Auto uses PELT changepoint detection to find the window");
    println!("    - No manual tuning of the fraction needed");
    println!("    - Adapts to wherever the last regime change occurred");
    println!("    - Falls back to 30% if no changepoints are detected");
    println!("    - Penalty controls sensitivity: lower = detects smaller changes");
    println!("══════════════════════════════════════════════════════════════════════════");
}

fn rmse_fn(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    let ss: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (ss / n).sqrt()
}

fn pseudo_noise(index: usize, seed: u64) -> f64 {
    let mut state = seed.wrapping_add(index as u64);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
}

fn slope_quality(
    slope: f64,
    target: f64,
    _cp_frac: f64,
    _rec_frac: f64,
) -> (&'static str, &'static str) {
    let _ = (slope - target).abs();
    ("", "")
}
