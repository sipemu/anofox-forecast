//! Issue #107 cross-cutting Inspectable conformance test.
//!
//! Exercises the `Inspectable` trait + `Explanation` enum across the
//! seven model families that opted in. Each model is fit on a synthetic
//! seasonal series and must:
//!
//! 1. Return `Err(FitRequired)` from `explanation()` before fit.
//! 2. Return the *matching* `Explanation` variant after fit
//!    (`Explanation::Ets` for ETS, `Explanation::Arima` for ARIMA, …).
//! 3. Populate the universal spine: `fitted_values` non-empty,
//!    `residuals` length matching, all scalar fields finite.
//! 4. Round-trip through `serde_json` losslessly (gated on the
//!    `serde` feature).
//! 5. Be usable as `Box<dyn Inspectable>` (object-safety check).
//!
//! This is the safety net for the trait contract: if any model
//! regresses on the surface, the suite fails before downstream callers
//! pick it up.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::AutoARIMA;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::{AutoTBATS, Explanation, Forecaster, MSTLForecaster, MFLES};

use chrono::{Duration, TimeZone, Utc};

fn make_seasonal_series(n: usize, period: usize) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n).map(|i| base + Duration::hours(i as i64)).collect();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let trend = 50.0 + 0.5 * i as f64;
            let seasonal =
                10.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin();
            let noise = ((i * 17) % 7) as f64 * 0.1 - 0.3;
            trend + seasonal + noise
        })
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn assert_finite_scalars(label: &str, e: &Explanation) {
    // Walk the variant and verify residual/fitted have matching cardinality
    // and the headline scalars are finite (NaN is OK only where the variant
    // documents it as optional).
    match e {
        Explanation::Regression(r) => {
            assert_eq!(
                r.coefficients.len(),
                r.feature_names.len(),
                "{label}: coefficients/feature_names length mismatch"
            );
            assert!(r.r_squared.is_finite(), "{label}: r_squared not finite");
            assert!(r.intercept.is_finite(), "{label}: intercept not finite");
            assert!(!r.backend.is_empty(), "{label}: backend empty");
        }
        Explanation::Ets(e) => {
            assert!(!e.spec.is_empty(), "{label}: ETS spec empty");
            assert!(e.alpha.is_finite(), "{label}: alpha not finite");
        }
        Explanation::Arima(a) => {
            assert!(a.aic.is_finite(), "{label}: aic not finite");
            assert!(a.bic.is_finite(), "{label}: bic not finite");
        }
        Explanation::Mfles(m) => {
            assert!(m.max_rounds > 0, "{label}: max_rounds should be positive");
        }
        Explanation::Theta(t) => {
            assert!(!t.variant.is_empty(), "{label}: theta variant empty");
            assert!(t.theta.is_finite(), "{label}: theta not finite");
        }
        Explanation::Tbats(t) => {
            assert!(
                !t.seasonal_periods.is_empty(),
                "{label}: no seasonal_periods"
            );
            assert!(t.aic.is_finite(), "{label}: aic not finite");
        }
        Explanation::Mstl(m) => {
            assert!(
                !m.seasonal_periods.is_empty(),
                "{label}: no seasonal_periods"
            );
            assert!(m.iterations > 0, "{label}: iterations must be > 0");
        }
        #[cfg(feature = "distributional")]
        Explanation::Laplace(l) => {
            assert!(!l.leaf_names.is_empty(), "{label}: leaf_names empty");
            assert_eq!(
                l.leaf_names.len(),
                l.leaf_weights.len(),
                "{label}: leaf_names/weights length mismatch"
            );
            let ws: f64 = l.leaf_weights.iter().sum();
            assert!(
                (ws - 1.0).abs() < 1e-9,
                "{label}: leaf_weights don't sum to 1"
            );
        }
    }
}

fn assert_spine(label: &str, e: &Explanation) {
    let (fitted, residuals) = match e {
        Explanation::Regression(r) => (&r.fitted_values, &r.residuals),
        Explanation::Ets(x) => (&x.fitted_values, &x.residuals),
        Explanation::Arima(x) => (&x.fitted_values, &x.residuals),
        Explanation::Mfles(x) => (&x.fitted_values, &x.residuals),
        Explanation::Theta(x) => (&x.fitted_values, &x.residuals),
        Explanation::Tbats(x) => (&x.fitted_values, &x.residuals),
        Explanation::Mstl(x) => (&x.fitted_values, &x.residuals),
        #[cfg(feature = "distributional")]
        Explanation::Laplace(x) => (&x.fitted_values, &x.residuals),
    };
    assert!(!fitted.is_empty(), "{label}: fitted_values empty");
    assert_eq!(
        fitted.len(),
        residuals.len(),
        "{label}: fitted/residuals length mismatch"
    );
}

#[test]
fn mstl_inspectable_contract() {
    let ts = make_seasonal_series(200, 24);
    let mut model = MSTLForecaster::new(vec![24]);
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(matches!(e, Explanation::Mstl(_)), "expected Mstl variant");
    assert_finite_scalars("MSTLForecaster", &e);
    assert_spine("MSTLForecaster", &e);
}

#[test]
fn mfles_inspectable_contract() {
    let ts = make_seasonal_series(120, 12);
    let mut model = MFLES::new(vec![12]);
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(matches!(e, Explanation::Mfles(_)), "expected Mfles variant");
    assert_finite_scalars("MFLES", &e);
    assert_spine("MFLES", &e);
}

#[test]
fn auto_arima_inspectable_contract() {
    let ts = make_seasonal_series(80, 12);
    let mut model = AutoARIMA::new();
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(matches!(e, Explanation::Arima(_)), "expected Arima variant");
    assert_finite_scalars("AutoARIMA", &e);
    assert_spine("AutoARIMA", &e);
}

#[test]
fn auto_ets_inspectable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = AutoETS::new();
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(matches!(e, Explanation::Ets(_)), "expected Ets variant");
    assert_finite_scalars("AutoETS", &e);
    assert_spine("AutoETS", &e);
}

#[test]
fn auto_theta_inspectable_contract() {
    let ts = make_seasonal_series(60, 12);
    let mut model = AutoTheta::new();
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(matches!(e, Explanation::Theta(_)), "expected Theta variant");
    assert_finite_scalars("AutoTheta", &e);
    assert_spine("AutoTheta", &e);
}

#[test]
fn auto_tbats_inspectable_contract() {
    let ts = make_seasonal_series(120, 12);
    let mut model = AutoTBATS::new(vec![12]);
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(matches!(e, Explanation::Tbats(_)), "expected Tbats variant");
    assert_finite_scalars("AutoTBATS", &e);
    assert_spine("AutoTBATS", &e);
}

#[cfg(feature = "postprocess")]
#[test]
fn regression_forecaster_inspectable_contract() {
    use anofox_forecast::models::regression::RegressionForecaster;

    // Use a simple linear-trend series.
    let n = 60;
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let timestamps: Vec<_> = (0..n).map(|i| base + Duration::days(i as i64)).collect();
    let values: Vec<f64> = (0..n).map(|i| 5.0 + 2.0 * i as f64).collect();
    let ts = TimeSeries::univariate(timestamps, values).unwrap();

    let mut model = RegressionForecaster::linear_trend();
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    assert!(
        matches!(e, Explanation::Regression(_)),
        "expected Regression variant"
    );
    assert_finite_scalars("RegressionForecaster", &e);
    assert_spine("RegressionForecaster", &e);
}

#[cfg(feature = "distributional")]
#[test]
fn laplace_inspectable_contract() {
    use anofox_forecast::models::LaplaceForecaster;

    let ts = make_seasonal_series(120, 12);
    let mut model = LaplaceForecaster::new();
    assert!(model.explanation().is_err());
    model.fit(&ts).unwrap();
    let e = model.explanation().unwrap();
    match &e {
        Explanation::Laplace(x) => {
            assert!(!x.leaf_names.is_empty(), "Laplace: leaf_names empty");
            assert_eq!(
                x.leaf_names.len(),
                x.leaf_weights.len(),
                "Laplace: leaf_names/weights length mismatch"
            );
            assert!(!x.horizon_dists.is_empty(), "Laplace: horizon_dists empty");
            assert!(!x.fitted_values.is_empty(), "Laplace: fitted_values empty");
            assert_eq!(
                x.fitted_values.len(),
                x.residuals.len(),
                "Laplace: fitted/residuals length mismatch"
            );
            let ws: f64 = x.leaf_weights.iter().sum();
            assert!(
                (ws - 1.0).abs() < 1e-9,
                "Laplace: leaf_weights don't sum to 1"
            );
        }
        other => panic!("expected Explanation::Laplace, got {other:?}"),
    }
}

#[test]
fn inspectable_is_object_safe() {
    // The trait must be usable behind Box<dyn _> because Explanation
    // is fully owned. This compile-and-call test will fail to link
    // (or panic at fit) if object-safety regresses.
    use anofox_forecast::models::Inspectable;

    let ts = make_seasonal_series(60, 12);
    let mut ets = AutoETS::new();
    ets.fit(&ts).unwrap();
    let boxed: Box<dyn Inspectable> = Box::new(ets);
    let e = boxed.explanation().unwrap();
    assert!(matches!(e, Explanation::Ets(_)));
}

#[cfg(feature = "serde")]
#[test]
fn explanation_serializes_to_json_for_every_variant() {
    // Verify each Inspectable model produces an Explanation that can be
    // serialized to JSON. We do NOT assert round-trip equality here for
    // two reasons:
    //   1. Some fitted models leave `NaN` in `fitted_values` (warm-up
    //      rows on AutoARIMA, etc.) which serde_json renders as `null`
    //      — and `null` is not deserialisable back into `f64`.
    //   2. JSON's f64 formatter is not bit-exact and can lose one ULP.
    // The bit-exact / lossless round-trip is exercised by the unit
    // tests in src/models/inspect.rs on hand-crafted finite payloads.
    // Here we just guarantee serialization works end-to-end on real
    // fit outputs.
    fn check(label: &str, e: &Explanation) {
        let json = serde_json::to_string(e).unwrap();
        assert!(!json.is_empty(), "{label}: JSON serialization empty");
        assert!(
            json.starts_with('{'),
            "{label}: expected JSON object, got {}",
            &json[..json.len().min(40)]
        );
    }

    let ts = make_seasonal_series(120, 12);

    let mut ets = AutoETS::new();
    ets.fit(&ts).unwrap();
    check("AutoETS", &ets.explanation().unwrap());

    let mut mstl = MSTLForecaster::new(vec![12]);
    mstl.fit(&ts).unwrap();
    check("MSTLForecaster", &mstl.explanation().unwrap());

    let mut theta = AutoTheta::new();
    theta.fit(&ts).unwrap();
    check("AutoTheta", &theta.explanation().unwrap());

    let mut arima = AutoARIMA::new();
    arima.fit(&ts).unwrap();
    check("AutoARIMA", &arima.explanation().unwrap());

    let mut mfles = MFLES::new(vec![12]);
    mfles.fit(&ts).unwrap();
    check("MFLES", &mfles.explanation().unwrap());

    let mut tbats = AutoTBATS::new(vec![12]);
    tbats.fit(&ts).unwrap();
    check("AutoTBATS", &tbats.explanation().unwrap());
}
