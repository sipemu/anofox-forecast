//! Regression guard for issue #219: MFLES multiplicative-mode runaway
//! on a series with a near-zero month (level ≈ 2k, min = 1.0).
//!
//! Documented before/after delta:
//! - Pre-fix: auto mode selects multiplicative (all values > 0), `ln(1.0) = 0.0`
//!   opens a log-space crater, forecast blows up to ~26000 (~13× level ≈ 2000).
//! - Post-fix: auto mode recognises `min/median ≈ 0.0005 < τ = 0.10` and
//!   falls back to additive; forecast stays near level (~1× level).
//!
//! Tests:
//! - MULT-04 (`issue_219_mfles_no_multiplicative_runaway`): post-fix forecast
//!   stays below 2.5× level in auto mode.
//! - MULT-01 (`issue_219_mfles_auto_mode_selects_additive_for_near_zero_series`):
//!   behavioral proof that additive was chosen (mean forecast near level).
//! - MULT-02/03 (`issue_219_mfles_explicit_multiplicative_with_floor_and_clamp`):
//!   even when multiplicative is forced, log-floor + back-transform clamp keep
//!   the forecast within 10× in-sample max.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

/// Build the 24-point monthly series that triggers issue #219.
/// Shape: level ≈ 2000, near-zero value (1.0) at index 11 (month 12).
/// All values are strictly positive, so the old code selects multiplicative.
fn make_issue_219_series() -> TimeSeries {
    let values = vec![
        2100.0_f64, 1950.0, 2200.0, 1800.0, 2050.0, 2300.0, 1900.0, 2150.0, 1850.0, 2400.0, 2000.0,
        1.0, // near-zero outlier: min/median ≈ 0.0005, far below τ = 0.10
        2050.0, 1950.0, 2100.0, 1800.0, 2200.0, 1900.0, 2050.0, 2300.0, 1850.0, 2150.0, 2000.0,
        2100.0,
    ];
    let base = Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..values.len())
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

/// MULT-04 regression: in auto mode the forecast must stay below 2.5× level
/// (≤ 5000) and remain positive.
///
/// Pre-fix: `forecast_max ≈ 26000` (~13× level) — this test FAILS.
/// Post-fix: `forecast_max < 5000` (~2.5× level) — this test PASSES.
#[test]
fn issue_219_mfles_no_multiplicative_runaway() {
    let ts = make_issue_219_series();
    let mut model = MFLES::new(vec![12]);
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();

    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let level = 2000.0_f64;

    // Post-fix: must be below 2.5× level (a generous margin that the
    // pre-fix ~13× easily violated). Level is ≈ 2000; threshold is 5000.
    assert!(
        forecast_max < 2.5 * level,
        "issue #219 regression: forecast_max={:.0} >= {:.0} (2.5× level). \
         Before/after target: ~26000 → <5000.",
        forecast_max,
        2.5 * level,
    );

    // Also verify forecasts are positive (not floored to zero).
    let forecast_min = preds.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        forecast_min > 0.0,
        "forecast should remain positive, got min={:.2}",
        forecast_min,
    );
}

/// MULT-01: auto mode selects additive when min/median < τ (0.10).
/// Verified behaviorally: mean forecast is near the series level (~2000)
/// when additive mode is chosen. (`is_multiplicative` is private — we
/// verify behavior, not internals.)
#[test]
fn issue_219_mfles_auto_mode_selects_additive_for_near_zero_series() {
    let ts = make_issue_219_series();
    let mut model = MFLES::new(vec![12]);
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();

    let mean_forecast: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
    assert!(
        (mean_forecast - 2000.0).abs() < 1000.0,
        "mean forecast {:.0} should be near level 2000 when additive mode is selected",
        mean_forecast,
    );
}

/// MULT-02/03: when multiplicative is forced (bypassing MULT-01 guard),
/// the log-floor and back-transform clamp together keep the forecast
/// within 10× the in-sample maximum.
///
/// In-sample max ≈ 2400; cap = 10 × 2400 = 24000.
#[test]
fn issue_219_mfles_explicit_multiplicative_with_floor_and_clamp() {
    let ts = make_issue_219_series();
    // Force multiplicative — guard 1 is bypassed; guards 2 and 3 must protect.
    let mut model = MFLES::builder()
        .seasonal_period(12)
        .multiplicative(true)
        .build();
    model.fit(&ts).unwrap();
    let fc = model.predict(12).unwrap();
    let preds = fc.primary();

    // With floor + clamp the forecast must stay below 10 × in-sample max.
    // Derive the in-sample max from the fixture itself (not a hardcoded
    // constant) so the cap tracks any future change to the series.
    let insample_max = make_issue_219_series()
        .primary_values()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let cap = 10.0 * insample_max;
    let forecast_max = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        forecast_max <= cap,
        "explicit multiplicative with clamp: forecast_max={:.0} exceeds cap={:.0}",
        forecast_max,
        cap,
    );
}
