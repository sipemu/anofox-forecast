//! Guard-assertion tests for the Phase 6 multiplicative-guard audit (MULT-05).
//!
//! Verifies that already-safe models are provably protected against the #10/#219
//! failure class: a near-zero-relative-to-level series must not produce a blow-up
//! forecast when a model's auto-multiplicative or auto-log path is triggered.
//!
//! Series shape: level ≈ 2000, small sinusoidal wobble, one near-zero month (1.0)
//! at the seasonal trough — same shape as the Phase 5 #219 repro. The series has
//! no customer or client identity; it describes only its numerical structure.
//!
//! Cross-references: issue #10 (AutoETS zero-value), issue #219 (MFLES near-zero runaway).

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::Theta;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

/// Build a monthly series of length `2 * period + 6` with:
/// - base level ≈ 2000 + small sinusoidal wobble (amplitude 100)
/// - one near-zero value (1.0) injected at index `period - 1` (the seasonal trough)
///
/// All other values are strictly positive and in the 1900–2100 range, so the
/// series superficially looks like a good candidate for multiplicative mode.
fn make_near_zero_series(period: usize) -> TimeSeries {
    let n = 2 * period + 6;
    let base = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n)
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    let mut values: Vec<f64> = (0..n)
        .map(|i| 2000.0 + 100.0 * ((i as f64 * 0.4).sin()))
        .collect();
    // Inject the near-zero value at the seasonal trough position.
    // This produces a multiplicative seasonal index ≈ 0.0005 — far below 0.01.
    values[period - 1] = 1.0;
    TimeSeries::univariate(timestamps, values).unwrap()
}

/// MULT-05 SC4: Theta stays bounded near level on a near-zero series.
///
/// `Theta::seasonal(12)` requests multiplicative decomposition by default. On this
/// near-zero series the forecast must stay BOUNDED and CENTERED near level — it must
/// neither blow up (the #10/#219 failure class) nor collapse. Theta is safe here by
/// two reinforcing properties: the seasonal-factor guard (Rule 2:
/// `any(seasonal_factor < 0.01)`, `src/models/theta/model.rs:486`) selects additive
/// because the near-zero month yields a factor ≈ 0.0005, AND Theta has no
/// `ln()→boosting→exp()` pipeline, so even multiplicative decomposition stays bounded.
/// This test is a regression guard for the failure CLASS: if a future refactor
/// introduced an MFLES-style log-boosting path into Theta, the upper bound below
/// would trip.
///
/// Assertions are two-sided: max < 2.5× level (no blow-up) AND the forecast mean
/// stays within 0.5× level of the series level (centered, not collapsed). The mean
/// is used rather than a per-point min because the recurring seasonal trough is a
/// legitimately low month in additive mode too — only the *aggregate* level is a
/// meaningful safety signal here.
#[test]
fn theta_seasonal_factor_guard_catches_near_zero() {
    let ts = make_near_zero_series(12);
    let mut model = Theta::seasonal(12); // requests multiplicative by default
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();
    let level = 2000.0_f64;

    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        forecast_max < 2.5 * level,
        "Theta near-zero safety: forecast_max={:.0} >= {:.0} (2.5× level) — a blow-up \
         of the #10/#219 class. Theta must keep the near-zero series bounded.",
        forecast_max,
        2.5 * level,
    );

    let mean_forecast: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
    assert!(
        (mean_forecast - level).abs() < 0.5 * level,
        "Theta near-zero safety: mean forecast={:.0} is not within 0.5× level of {:.0} \
         — the forecast is not centered at level (blow-up or collapse).",
        mean_forecast,
        level,
    );
}

/// MULT-05 SC4: AutoETS stays bounded near level on a near-zero series.
///
/// `AutoETS::with_period(12)` evaluates additive and multiplicative candidates when
/// all values are positive. Safety here rests on two properties: AIC-based selection
/// penalises the multiplicative candidate (its trough seasonal factor ≈ 0.0005 gives
/// huge squared error at future visits to that phase → worse AIC than additive), AND
/// ETS multiplicative uses state-space RATIO updates (`y/s`, `level×s`), not an
/// `ln()→boosting→exp()` pipeline — so it cannot produce the #10/#219 crater blow-up.
///
/// Assertions are two-sided: max < 2.5× level (no blow-up) AND the forecast mean
/// stays within 0.5× level of the series level (centered, not collapsed) AND all
/// forecasts are positive. The mean (not a per-point min) is the meaningful signal:
/// the recurring seasonal trough is a legitimately low month, so only the aggregate
/// level distinguishes safe behavior from a blow-up/collapse.
///
/// Non-positive guard site: `src/models/exponential/auto_ets.rs:422`.
#[test]
fn auto_ets_aicselection_rejects_mult_for_near_zero() {
    let ts = make_near_zero_series(12);
    let mut model = AutoETS::with_period(12);
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();
    let level = 2000.0_f64;

    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        forecast_max < 2.5 * level,
        "AutoETS near-zero safety: forecast_max={:.0} >= {:.0} (2.5× level) — a blow-up \
         of the #10/#219 class.",
        forecast_max,
        2.5 * level,
    );

    let mean_forecast: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
    assert!(
        (mean_forecast - level).abs() < 0.5 * level,
        "AutoETS near-zero safety: mean forecast={:.0} is not within 0.5× level of {:.0} \
         — the forecast is not centered at level (blow-up or collapse).",
        mean_forecast,
        level,
    );

    let forecast_min = preds.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        forecast_min > 0.0,
        "AutoETS near-zero safety: forecasts should be positive, got min={:.2}.",
        forecast_min,
    );
}
