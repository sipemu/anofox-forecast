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

/// MULT-05 SC4: Theta seasonal-factor guard fires for near-zero series.
///
/// `Theta::seasonal(12)` requests multiplicative decomposition by default.
/// `determine_decomposition()` Rule 2 (`any(seasonal_factor < 0.01)`) triggers
/// because the near-zero month (1.0) produces a seasonal index ≈ 0.0005 — causing
/// an additive fallback. Forecast must therefore stay near level (< 2.5× level).
///
/// Equivalent guard site: `src/models/theta/model.rs:486`.
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
        "Theta near-zero guard: forecast_max={:.0} >= {:.0} (2.5× level). \
         Seasonal-factor guard (Rule 2: any(s < 0.01)) should have selected additive \
         — the near-zero month produces a factor ≈ 0.0005.",
        forecast_max,
        2.5 * level,
    );
}

/// MULT-05 SC4: AutoETS AIC selection rejects multiplicative for near-zero series.
///
/// `AutoETS::with_period(12)` evaluates both additive and multiplicative candidates
/// when all values are positive. For a near-zero series the multiplicative model
/// fits the trough seasonal phase to ≈ 0.0005, but future visits to that phase
/// will have values ~2000 → very high squared error → much worse AIC than additive.
/// AIC selection must therefore prefer additive, keeping forecasts bounded.
///
/// Asserts: (a) max forecast < 2.5× level; (b) all forecasts > 0.
///
/// Equivalent guard sites: `src/models/exponential/auto_ets.rs:422`
/// (non-positive guard) plus AIC-based selection in ETS candidate ranking.
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
        "AutoETS near-zero: forecast_max={:.0} >= {:.0} (2.5× level). \
         AIC protection should reject multiplicative — the near-zero trough \
         produces a seasonal factor ≈ 0.0005, making multiplicative AIC much worse.",
        forecast_max,
        2.5 * level,
    );

    let forecast_min = preds.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        forecast_min > 0.0,
        "AutoETS near-zero: forecasts should be positive, got min={:.2}. \
         A multiplicative blow-up or additive under-forecast may have occurred.",
        forecast_min,
    );
}
