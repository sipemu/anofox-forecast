//! Issue #233: incremental model updates and compact serializable state.
//!
//! Contracts checked here:
//! - **Exactness:** fit on `n` + `update` with `k` equals fit on `n + k` with
//!   the same fixed parameters (and initial states) -- final states, point
//!   forecasts, prediction intervals, fitted values and residuals.
//! - **Tracking:** auto-fit on `n` + `update` with `k` stays close to a full
//!   auto refit on `n + k` (parameters are not re-estimated by `update`).
//! - **Round-trip:** `state()` -> (serde) -> `from_state()` reproduces
//!   forecasts and subsequent updates; the state size does not grow with `n`.

use anofox_forecast::core::{CalendarAnnotations, TimeSeries};
use anofox_forecast::error::ForecastError;
use anofox_forecast::models::exponential::{
    AutoETS, ETSSpec, HoltLinearTrend, HoltWinters, SeasonalType, SimpleExponentialSmoothing, ETS,
};
use anofox_forecast::models::{Forecaster, IncrementalForecaster, ModelState};
use chrono::{Duration, TimeZone, Utc};

const PERIOD: usize = 12;
/// Relative tolerance for states, point forecasts, fitted values and
/// residuals: the same f64 recursion runs in `fit()` and `update()`.
const EXACT_TOL: f64 = 1e-9;
/// Relative tolerance for prediction intervals of SES / Holt / HoltWinters.
/// Their `fit()` sums squared residuals with `crate::simd::sum_of_squares`,
/// which accumulates in **f32** (~1e-7 relative), while `update()` extends
/// the stored sum in f64; the reference itself is only f32-accurate.
const INTERVAL_TOL: f64 = 1e-6;

fn make_ts(values: &[f64]) -> TimeSeries {
    let base = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let timestamps = (0..values.len())
        .map(|i| base + Duration::days(i as i64))
        .collect();
    TimeSeries::univariate(timestamps, values.to_vec()).unwrap()
}

/// Positive monthly-style series: level 100, trend 0.3/step, period-12
/// seasonality of amplitude 10, deterministic pseudo-random noise.
fn series(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..n)
        .map(|t| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state % 10_000) as f64 / 10_000.0 - 0.5;
            let season = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / PERIOD as f64).sin();
            100.0 + 0.3 * t as f64 + season + 4.0 * noise
        })
        .collect()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            (x - y).abs() <= tol * scale,
            "{what}[{i}]: {x} vs {y} (tol {tol})"
        );
    }
}

/// Upper-bound half-widths of the first forecast series.
fn half_widths(fc: &anofox_forecast::core::Forecast) -> Vec<f64> {
    fc.upper_series(0)
        .unwrap()
        .iter()
        .zip(fc.primary())
        .map(|(u, p)| u - p)
        .collect()
}

/// Fit `make()` on the first `n` points and update with the rest, versus
/// fit `make()` on everything; compare everything observable.
fn check_update_matches_refit<M, F>(make: F, n: usize, k: usize, label: &str)
where
    M: IncrementalForecaster,
    F: Fn() -> M,
{
    let values = series(n + k);
    let mut full = make();
    full.fit(&make_ts(&values)).unwrap();

    let mut inc = make();
    inc.fit(&make_ts(&values[..n])).unwrap();
    // Split the update in two calls to cover state carried across updates.
    inc.update(&values[n..n + k / 2]).unwrap();
    inc.update(&values[n + k / 2..]).unwrap();

    let h = 2 * PERIOD;
    assert_close(
        inc.predict(h).unwrap().primary(),
        full.predict(h).unwrap().primary(),
        EXACT_TOL,
        &format!("{label} point forecast"),
    );
    let fi = inc.predict_with_intervals(h, 0.95).unwrap();
    let ff = full.predict_with_intervals(h, 0.95).unwrap();
    assert_close(
        &half_widths(&fi),
        &half_widths(&ff),
        INTERVAL_TOL,
        &format!("{label} interval half-width"),
    );
    assert_close(
        inc.fitted_values().unwrap(),
        full.fitted_values().unwrap(),
        EXACT_TOL,
        &format!("{label} fitted"),
    );
    assert_close(
        inc.residuals().unwrap(),
        full.residuals().unwrap(),
        EXACT_TOL,
        &format!("{label} residuals"),
    );
    assert_eq!(inc.training_values().ok(), full.training_values().ok());

    // The exported states agree as well (same n, same parameters).
    let (si, sf) = (inc.state().unwrap(), full.state().unwrap());
    assert_eq!(si.n_obs(), n + k, "{label} n_obs");
    assert_eq!(si.n_obs(), sf.n_obs());
}

#[test]
fn ses_update_matches_fixed_param_refit() {
    check_update_matches_refit(|| SimpleExponentialSmoothing::new(0.3), 60, 24, "SES");
}

#[test]
fn ses_auto_update_matches_refit_at_fitted_alpha() {
    // `auto()` estimates alpha on the first n points; a refit on n + k at
    // that alpha (same interval df convention) is the exact reference.
    let values = series(84);
    let mut inc = SimpleExponentialSmoothing::auto();
    inc.fit(&make_ts(&values[..60])).unwrap();
    inc.update(&values[60..]).unwrap();
    let mut full = SimpleExponentialSmoothing::new(inc.alpha().unwrap());
    full.fit(&make_ts(&values)).unwrap();
    assert_close(
        &half_widths(&inc.predict_with_intervals(6, 0.9).unwrap()),
        &half_widths(&full.predict_with_intervals(6, 0.9).unwrap()),
        INTERVAL_TOL,
        "SES auto upper",
    );
}

#[test]
fn holt_update_matches_fixed_param_refit() {
    check_update_matches_refit(|| HoltLinearTrend::new(0.4, 0.1), 60, 24, "Holt");
    check_update_matches_refit(
        || HoltLinearTrend::damped(0.4, 0.1, 0.9),
        60,
        24,
        "Holt damped",
    );
}

#[test]
fn holt_winters_update_matches_fixed_param_refit() {
    for st in [SeasonalType::Additive, SeasonalType::Multiplicative] {
        check_update_matches_refit(
            || HoltWinters::new(0.3, 0.05, 0.2, PERIOD, st),
            48,
            25,
            &format!("HoltWinters {st:?}"),
        );
    }
}

#[test]
fn ets_update_matches_fixed_param_refit_all_specs() {
    let specs = [
        "ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA", "ANM", "AAM", "AAdM", "MNN", "MAN", "MAdN",
        "MNA", "MAA", "MAdA", "MNM", "MAM", "MAdM",
    ];
    for notation in specs {
        let spec = ETSSpec::from_notation(notation).unwrap();
        let period = if spec.has_seasonal() { PERIOD } else { 1 };
        let seasonals: Vec<f64> = if !spec.has_seasonal() {
            Vec::new()
        } else if notation.ends_with('M') {
            (0..PERIOD)
                .map(|i| 1.0 + 0.1 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
                .collect()
        } else {
            (0..PERIOD)
                .map(|i| 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin())
                .collect()
        };
        let make = || {
            ETS::with_params_and_states(
                spec,
                period,
                0.3,
                spec.has_trend().then_some(0.05),
                spec.has_seasonal().then_some(0.1),
                spec.is_damped().then_some(0.9),
                100.0,
                if spec.has_trend() { 0.3 } else { 0.0 },
                seasonals.clone(),
            )
        };
        check_update_matches_refit(make, 48, 25, notation);

        // Information criteria are extended consistently too.
        let values = series(73);
        let mut full = make();
        full.fit(&make_ts(&values)).unwrap();
        let mut inc = make();
        inc.fit(&make_ts(&values[..48])).unwrap();
        inc.update(&values[48..]).unwrap();
        assert_close(
            &[inc.aic().unwrap(), inc.bic().unwrap()],
            &[full.aic().unwrap(), full.bic().unwrap()],
            EXACT_TOL,
            &format!("{notation} aic/bic"),
        );
    }
}

/// Auto-fit + update vs a full auto refit: parameters are frozen at the
/// n-point estimate, so this measures tracking, not identity. Prints the
/// measured deviation; the bound is 2% of the forecast level.
#[test]
fn auto_models_update_tracks_full_refit() {
    fn rel_dev<M: IncrementalForecaster>(mut inc: M, mut full: M, n: usize, k: usize) -> f64 {
        let values = series(n + k);
        inc.fit(&make_ts(&values[..n])).unwrap();
        inc.update(&values[n..]).unwrap();
        full.fit(&make_ts(&values)).unwrap();
        let (a, b) = (inc.predict(PERIOD).unwrap(), full.predict(PERIOD).unwrap());
        a.primary()
            .iter()
            .zip(b.primary())
            .map(|(x, y)| (x - y).abs() / y.abs())
            .fold(0.0, f64::max)
    }
    let (n, k) = (120, 24);
    let cases = [
        (
            "SES",
            rel_dev(
                SimpleExponentialSmoothing::auto(),
                SimpleExponentialSmoothing::auto(),
                n,
                k,
            ),
        ),
        (
            "Holt",
            rel_dev(HoltLinearTrend::auto(), HoltLinearTrend::auto(), n, k),
        ),
        (
            "HoltWinters",
            rel_dev(
                HoltWinters::auto(PERIOD, SeasonalType::Additive),
                HoltWinters::auto(PERIOD, SeasonalType::Additive),
                n,
                k,
            ),
        ),
        (
            "ETS(A,A,A)",
            rel_dev(
                ETS::new(ETSSpec::aaa(), PERIOD),
                ETS::new(ETSSpec::aaa(), PERIOD),
                n,
                k,
            ),
        ),
    ];
    for (name, dev) in cases {
        eprintln!("tracking {name}: max relative forecast deviation {dev:.4e}");
        assert!(dev < 0.02, "{name}: update drifted {dev:.4} from refit");
    }
}

#[test]
fn auto_ets_update_and_state_delegate_to_selected_model() {
    let values = series(96);
    let mut auto = AutoETS::new();
    auto.fit(&make_ts(&values[..72])).unwrap();
    let state = auto.state().unwrap();
    assert!(matches!(state, ModelState::ETS(_)));

    auto.update(&values[72..]).unwrap();
    let mut restored = AutoETS::from_state(state).unwrap();
    restored.update(&values[72..]).unwrap();
    assert_close(
        restored.predict(PERIOD).unwrap().primary(),
        auto.predict(PERIOD).unwrap().primary(),
        0.0,
        "AutoETS restored+update",
    );
    assert_eq!(auto.training_values().unwrap().len(), 96);
}

fn round_trip_case<M: IncrementalForecaster>(mut model: M, n: usize) {
    let values = series(n + 10);
    model.fit(&make_ts(&values[..n])).unwrap();
    let state = model.state().unwrap();
    let mut restored = M::from_state(state.clone()).unwrap();
    assert_eq!(restored.state().unwrap(), state, "state round-trip");
    assert!(restored.fitted_values().is_none());

    let h = PERIOD;
    assert_close(
        restored.predict(h).unwrap().primary(),
        model.predict(h).unwrap().primary(),
        0.0,
        "restored point forecast",
    );
    model.update(&values[n..]).unwrap();
    restored.update(&values[n..]).unwrap();
    let (a, b) = (
        restored.predict_with_intervals(h, 0.8).unwrap(),
        model.predict_with_intervals(h, 0.8).unwrap(),
    );
    assert_close(a.primary(), b.primary(), EXACT_TOL, "restored+update point");
    assert_close(
        a.upper_series(0).unwrap(),
        b.upper_series(0).unwrap(),
        EXACT_TOL,
        "restored+update upper",
    );

    // Type-erased restore gives the same forecasts.
    let boxed = restored.state().unwrap().restore().unwrap();
    assert_eq!(boxed.name(), model.name());
    assert_close(
        boxed.predict(h).unwrap().primary(),
        b.primary(),
        EXACT_TOL,
        "boxed restore",
    );
}

#[test]
fn state_round_trip_all_models() {
    round_trip_case(SimpleExponentialSmoothing::auto(), 60);
    round_trip_case(HoltLinearTrend::auto_damped(), 60);
    round_trip_case(HoltWinters::auto(PERIOD, SeasonalType::Multiplicative), 60);
    round_trip_case(ETS::new(ETSSpec::mam(), PERIOD), 60);
    round_trip_case(ETS::new(ETSSpec::aadn(), 1), 60);
}

#[test]
fn update_errors() {
    let values = series(40);

    let mut unfitted = SimpleExponentialSmoothing::auto();
    assert!(matches!(
        unfitted.update(&[1.0]),
        Err(ForecastError::FitRequired { .. })
    ));
    assert!(matches!(
        ETS::new(ETSSpec::ann(), 1).state(),
        Err(ForecastError::FitRequired { .. })
    ));

    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&make_ts(&values)).unwrap();
    let before = ses.state().unwrap();
    assert!(matches!(
        ses.update(&[1.0, f64::NAN]),
        Err(ForecastError::MissingValues)
    ));
    assert_eq!(
        ses.state().unwrap(),
        before,
        "failed update must not mutate"
    );
    ses.update(&[]).unwrap();
    assert_eq!(ses.state().unwrap(), before, "empty update is a no-op");

    assert!(matches!(
        HoltLinearTrend::from_state(before),
        Err(ForecastError::InvalidParameter(_))
    ));

    let mut hw = HoltWinters::auto(PERIOD, SeasonalType::Additive);
    hw.fit(&make_ts(&values)).unwrap();
    let ModelState::HoltWinters(mut st) = hw.state().unwrap() else {
        panic!("expected a HoltWinters state");
    };
    st.seasonals.pop();
    assert!(matches!(
        HoltWinters::from_state(ModelState::HoltWinters(st)),
        Err(ForecastError::InvalidParameter(_))
    ));
}

#[test]
fn update_rejects_models_fit_with_regressors() {
    let values = series(40);
    let mut ts = make_ts(&values);
    ts.set_calendar(
        CalendarAnnotations::new()
            .with_regressor("x".to_string(), (0..40).map(|i| (i % 7) as f64).collect()),
    );

    let mut ets = ETS::new(ETSSpec::ann(), 1);
    ets.fit(&ts).unwrap();
    assert!(matches!(
        ets.update(&[1.0]),
        Err(ForecastError::InvalidParameter(_))
    ));
    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&ts).unwrap();
    assert!(matches!(
        ses.update(&[1.0]),
        Err(ForecastError::InvalidParameter(_))
    ));
}

#[cfg(feature = "serde")]
#[test]
fn state_serializes_and_size_is_independent_of_history() {
    use anofox_forecast::utils::persistence::{from_bincode, from_json, to_bincode, to_json};

    let mut sizes = Vec::new();
    for n in [48, 480, 4800] {
        let values = series(n);
        let mut model = ETS::new(ETSSpec::aaa(), PERIOD);
        model.fit(&make_ts(&values)).unwrap();
        let state = model.state().unwrap();

        let bytes = to_bincode(&state).unwrap();
        let back: ModelState = from_bincode(&bytes).unwrap();
        assert_eq!(back, state);
        // serde_json's default float parser may round the last ulp, so
        // JSON is checked on the forecasts rather than bit-for-bit.
        let json = to_json(&state).unwrap();
        let back: ModelState = from_json(&json).unwrap();
        assert_close(
            ETS::from_state(back)
                .unwrap()
                .predict(PERIOD)
                .unwrap()
                .primary(),
            model.predict(PERIOD).unwrap().primary(),
            1e-12,
            "json round-trip",
        );

        let full_model = to_bincode(&model).unwrap().len();
        eprintln!(
            "n={n}: state {} B (bincode), {} B (json); full model {} B (bincode)",
            bytes.len(),
            json.len(),
            full_model
        );
        sizes.push(bytes.len());
    }
    assert!(
        sizes.windows(2).all(|w| w[0] == w[1]),
        "state size grew: {sizes:?}"
    );
}

#[cfg(feature = "serde")]
#[test]
fn models_serialized_without_error_sums_still_load_and_refuse_update() {
    use anofox_forecast::utils::persistence::{from_json, to_json};

    let mut ses = SimpleExponentialSmoothing::auto();
    ses.fit(&make_ts(&series(30))).unwrap();
    // Drop the new field to mimic a model serialized by an older version.
    let mut value: serde_json::Value = serde_json::from_str(&to_json(&ses).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("sse");
    let old: SimpleExponentialSmoothing = from_json(&value.to_string()).unwrap();
    assert_eq!(
        old.predict(3).unwrap().primary(),
        ses.predict(3).unwrap().primary()
    );
    let mut old = old;
    assert!(matches!(
        old.update(&[1.0]),
        Err(ForecastError::FitRequired { .. })
    ));
}
