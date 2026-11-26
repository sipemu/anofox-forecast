//! `ETS::with_fixed_params`: fixed smoothing parameters, maximum-likelihood
//! initial states (DataZooDE/anofox-forecast-extension#317).
//!
//! Hyper-model forecasting predicts the smoothing parameters of a classical
//! model from features and lets the model produce the forecast, so the given
//! parameters must be used exactly, with only the initial states estimated.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::{ETSSpec, ETS};
use anofox_forecast::models::{Explanation, Forecaster, Inspectable};
use chrono::{Duration, TimeZone, Utc};

const PERIOD: usize = 12;

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

fn sse(model: &ETS) -> f64 {
    model.residuals().unwrap().iter().map(|r| r * r).sum()
}

#[test]
fn fixed_params_at_the_optimum_recover_the_optimised_fit() {
    // Fixing the smoothing parameters at the jointly optimised values and
    // estimating only the states reaches (about) the same fit, and a better
    // one than `with_params`' heuristic states.
    let ts = series(120);
    for notation in ["ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA", "MAM", "MNM"] {
        let spec = ETSSpec::from_notation(notation).unwrap();
        let mut opt = ETS::new(spec, PERIOD);
        opt.fit(&ts).unwrap();
        let (a, b, g, p) = (opt.alpha().unwrap(), opt.beta(), opt.gamma(), opt.phi());

        let mut fixed = ETS::with_fixed_params(spec, PERIOD, a, b, g, p).unwrap();
        fixed.fit(&ts).unwrap();
        assert!(fixed.has_fixed_params());
        assert_eq!(fixed.alpha(), Some(a), "{notation}: alpha changed");
        assert_eq!(fixed.phi(), p, "{notation}: phi changed");

        let mut heuristic = ETS::with_params(spec, PERIOD, a, b, g, p);
        heuristic.fit(&ts).unwrap();

        let (s_opt, s_fixed, s_heur) = (sse(&opt), sse(&fixed), sse(&heuristic));
        assert!(
            s_fixed <= s_opt * 1.05 + 1e-9,
            "{notation}: fixed-param SSE {s_fixed} vs optimised {s_opt}"
        );
        assert!(
            fixed.log_likelihood().unwrap() >= heuristic.log_likelihood().unwrap() - 1e-6,
            "{notation}: estimated states worse than heuristic ({s_fixed} vs {s_heur})"
        );

        let h_opt = opt.predict(PERIOD).unwrap().primary().to_vec();
        let h_fixed = fixed.predict(PERIOD).unwrap().primary().to_vec();
        for h in 0..PERIOD {
            let rel = (h_opt[h] - h_fixed[h]).abs() / h_opt[h].abs();
            assert!(
                rel < 0.03,
                "{notation}: h={h} forecast {} vs {}",
                h_fixed[h],
                h_opt[h]
            );
        }
    }
}

#[test]
fn given_values_are_used_unclamped() {
    // `with_params` clamps phi to [0.8, 0.98]; fixed params are kept as given.
    let ts = series(96);
    let spec = ETSSpec::from_notation("AAdN").unwrap();
    let mut m = ETS::with_fixed_params(spec, 1, 0.2, Some(0.05), None, Some(0.995)).unwrap();
    m.fit(&ts).unwrap();
    assert_eq!(m.alpha(), Some(0.2));
    assert_eq!(m.beta(), Some(0.05));
    assert_eq!(m.phi(), Some(0.995));
    assert!(m
        .predict(6)
        .unwrap()
        .primary()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn parameters_must_match_the_spec() {
    let aan = ETSSpec::from_notation("AAN").unwrap();
    let ann = ETSSpec::from_notation("ANN").unwrap();
    let aada = ETSSpec::from_notation("AAdA").unwrap();
    let cases = [
        (
            ETS::with_fixed_params(aan, 1, 0.3, None, None, None),
            "requires a fixed beta",
        ),
        (
            ETS::with_fixed_params(ann, 1, 0.3, Some(0.1), None, None),
            "does not use beta",
        ),
        (
            ETS::with_fixed_params(ann, 1, 0.3, None, Some(0.1), None),
            "does not use gamma",
        ),
        (
            ETS::with_fixed_params(ann, 1, 1.5, None, None, None),
            "alpha = 1.5",
        ),
        (
            ETS::with_fixed_params(ann, 1, 0.0, None, None, None),
            "alpha = 0",
        ),
        (
            ETS::with_fixed_params(aada, 12, 0.3, Some(0.1), Some(0.1), None),
            "requires a fixed phi",
        ),
        (
            ETS::with_fixed_params(aada, 12, 0.3, Some(0.1), Some(0.1), Some(0.0)),
            "phi = 0",
        ),
        (
            ETS::with_fixed_params(aada, 1, 0.3, Some(0.1), Some(0.1), Some(0.9)),
            "seasonal_period >= 2",
        ),
    ];
    for (result, expected) in cases {
        let msg = result.err().expect("expected an error").to_string();
        assert!(msg.contains(expected), "expected '{expected}', got: {msg}");
    }
}

#[test]
fn fixed_params_count_only_the_states() {
    // AIC counts estimated parameters only: fixing alpha/beta/gamma removes
    // three, so at the same likelihood the AIC is 6 lower.
    let ts = series(120);
    let spec = ETSSpec::from_notation("AAA").unwrap();
    let mut m = ETS::with_fixed_params(spec, PERIOD, 0.3, Some(0.05), Some(0.1), None).unwrap();
    m.fit(&ts).unwrap();
    let ll = m.log_likelihood().unwrap();
    // states: l0, b0, 11 free seasonals, + sigma^2
    let k = 1.0 + 1.0 + (PERIOD - 1) as f64 + 1.0;
    assert!((m.aic().unwrap() - (-2.0 * ll + 2.0 * k)).abs() < 1e-6);
}

#[test]
fn fixed_spec_ets_is_inspectable() {
    let ts = series(120);
    let spec = ETSSpec::from_notation("AAA").unwrap();
    let mut m = ETS::with_fixed_params(spec, PERIOD, 0.3, Some(0.05), Some(0.1), None).unwrap();
    m.fit(&ts).unwrap();
    let Explanation::Ets(e) = Inspectable::explanation(&m).unwrap() else {
        panic!("not an Ets explanation");
    };
    assert_eq!(e.alpha, 0.3);
    assert_eq!(e.beta, Some(0.05));
    assert_eq!(e.gamma, Some(0.1));
    assert_eq!(e.seasonal_period, PERIOD);
    assert_eq!(e.fitted_values.len(), 120);
    assert_eq!(e.seasonal_component.map(|s| s.len()), Some(120));
    assert_eq!(e.initial_seasonals.map(|s| s.len()), Some(PERIOD));
    // Also reachable through the Forecaster trait.
    assert!(Forecaster::explanation(&m).is_ok());

    // An optimised fixed-spec ETS is inspectable too.
    let mut opt = ETS::new(spec, PERIOD);
    opt.fit(&ts).unwrap();
    assert!(matches!(
        Inspectable::explanation(&opt).unwrap(),
        Explanation::Ets(_)
    ));
}
