//! ETS / AutoETS in-sample components and initial states
//! (DataZooDE/anofox-forecast-extension#316).
//!
//! `seasonal_component()` used to be unimplemented for `ETS` and `AutoETS`, and
//! `AutoETS::trend_component()` returned the fitted values. Both now split each
//! one-step fitted value into a level/trend part and a seasonal part, so
//! `trend + seasonal + residual == training` holds for every spec. The time-0
//! states are exposed too, so the fit can be replayed from the explanation.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{AutoETS, ETSSpec, ETS};
use anofox_forecast::models::{Explanation, Forecaster, Inspectable};
use chrono::{Duration, TimeZone, Utc};

const PERIOD: usize = 12;

/// Positive monthly series with trend, multiplicative-looking seasonality and
/// deterministic noise (AirPassengers-shaped).
fn series(n: usize) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let timestamps = (0..n)
        .map(|i| base + Duration::days(30 * i as i64))
        .collect();
    let values = (0..n)
        .map(|i| {
            let t = i as f64;
            let season = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * t / PERIOD as f64).sin();
            (120.0 + 2.5 * t) * season + ((i * 7919) % 11) as f64
        })
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn assert_decomposes(name: &str, model: &dyn Forecaster, ts: &TimeSeries, seasonal: bool) {
    let y = ts.primary_values();
    let fitted = model.fitted_values().unwrap();
    let residuals = model.residuals().unwrap();
    let trend = model
        .trend_component()
        .unwrap_or_else(|e| panic!("{name}: trend_component: {e}"));
    assert_eq!(trend.len(), y.len(), "{name}: trend length");

    match model.seasonal_component() {
        Ok(s) => {
            assert!(
                seasonal,
                "{name}: non-seasonal spec exposes a seasonal component"
            );
            assert_eq!(s.len(), y.len(), "{name}: seasonal length");
            assert!(
                s.iter().any(|v| v.abs() > 1e-6),
                "{name}: seasonal component is all zero"
            );
            for t in 0..y.len() {
                let sum = trend[t] + s[t] + residuals[t];
                assert!(
                    (sum - y[t]).abs() < 1e-9 * y[t].abs().max(1.0),
                    "{name}: trend + seasonal + residual != y at t={t}: {sum} vs {}",
                    y[t]
                );
            }
        }
        Err(_) => {
            assert!(!seasonal, "{name}: seasonal spec has no seasonal component");
            // Non-seasonal: the trend part is the whole one-step forecast.
            assert_eq!(trend, fitted, "{name}: non-seasonal trend != fitted");
        }
    }
}

#[test]
fn ets_components_decompose_every_spec() {
    let ts = series(96);
    for notation in [
        "ANN", "AAN", "AAdN", "MNN", "MAN", "ANA", "AAA", "AAdA", "MNA", "MAA", "MNM", "MAM",
        "MAdM", "ANM", "AAM",
    ] {
        let spec = ETSSpec::from_notation(notation).unwrap();
        let mut model = ETS::new(spec, PERIOD);
        model.fit(&ts).unwrap();
        assert_decomposes(notation, &model, &ts, spec.has_seasonal());

        let init = model.initial_states().expect("initial states after fit");
        assert!(init.level.is_finite(), "{notation}: l0");
        assert_eq!(
            init.trend.is_some(),
            spec.has_trend(),
            "{notation}: b0 presence"
        );
        assert_eq!(
            init.seasonals.as_ref().map(Vec::len),
            spec.has_seasonal().then_some(PERIOD),
            "{notation}: s0 length"
        );
    }
}

#[test]
fn ets_initial_states_replay_the_fit() {
    // The exported parameters + time-0 states reproduce the fitted values.
    let ts = series(96);
    for notation in ["AAN", "AAdA", "MAM", "ANA"] {
        let spec = ETSSpec::from_notation(notation).unwrap();
        let mut fitted = ETS::new(spec, PERIOD);
        fitted.fit(&ts).unwrap();
        let init = fitted.initial_states().unwrap().clone();

        let mut replay = ETS::with_params_and_states(
            spec,
            PERIOD,
            fitted.alpha().unwrap(),
            fitted.beta(),
            fitted.gamma(),
            fitted.phi(),
            init.level,
            init.trend.unwrap_or(0.0),
            init.seasonals.clone().unwrap_or_default(),
        );
        replay.fit(&ts).unwrap();

        let a = fitted.fitted_values().unwrap();
        let b = replay.fitted_values().unwrap();
        for t in 0..a.len() {
            assert!(
                (a[t] - b[t]).abs() < 1e-9 * a[t].abs().max(1.0),
                "{notation}: replay differs at t={t}: {} vs {}",
                a[t],
                b[t]
            );
        }
        let ha = fitted.predict(PERIOD).unwrap().primary().to_vec();
        let hb = replay.predict(PERIOD).unwrap().primary().to_vec();
        for h in 0..PERIOD {
            assert!((ha[h] - hb[h]).abs() < 1e-9 * ha[h].abs().max(1.0));
        }
    }
}

#[test]
fn auto_ets_exposes_seasonal_component_and_initial_states() {
    let ts = series(120);
    let mut model = AutoETS::with_period(PERIOD);
    model.fit(&ts).unwrap();
    let spec = model.selected_spec().unwrap();
    assert!(
        spec.has_seasonal(),
        "expected a seasonal selection, got {spec:?}"
    );
    assert_decomposes("AutoETS", &model, &ts, true);

    // AutoETS's trend component is no longer the fitted values.
    assert_ne!(
        model.trend_component().unwrap(),
        model.fitted_values().unwrap()
    );

    let Explanation::Ets(e) = Inspectable::explanation(&model).unwrap() else {
        panic!("AutoETS explanation is not Ets");
    };
    let seasonal = e.seasonal_component.expect("seasonal_component populated");
    assert_eq!(seasonal.len(), e.fitted_values.len());
    assert_eq!(e.trend_component.len(), e.fitted_values.len());
    assert!(e.initial_level.is_some());
    assert_eq!(e.initial_trend.is_some(), spec.has_trend());
    assert_eq!(e.initial_seasonals.map(|s| s.len()), Some(PERIOD));
}

#[test]
fn update_extends_the_components() {
    let ts = series(96);
    let y = ts.primary_values().to_vec();
    let spec = ETSSpec::from_notation("AAA").unwrap();
    let mut model = ETS::new(spec, PERIOD);
    model.fit(&series(84)).unwrap();
    anofox_forecast::models::IncrementalForecaster::update(&mut model, &y[84..]).unwrap();

    let trend = model.trend_component().unwrap();
    let seasonal = model.seasonal_component().unwrap();
    let residuals = model.residuals().unwrap();
    assert_eq!(trend.len(), 96);
    assert_eq!(seasonal.len(), 96);
    for t in 84..96 {
        assert!((trend[t] + seasonal[t] + residuals[t] - y[t]).abs() < 1e-9 * y[t]);
    }
}
