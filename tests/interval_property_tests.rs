//! Comprehensive prediction interval property tests.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::baseline::{
    HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive, SeasonalWindowAverage,
    SimpleMovingAverage, WindowAverage,
};
use anofox_forecast::models::intermittent::{Croston, ADIDA, IMAPA, TSB};
use anofox_forecast::models::Forecaster;
use chrono::{Duration, TimeZone, Utc};

fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    (0..n).map(|i| base + Duration::hours(i as i64)).collect()
}

fn make_trending_series(n: usize) -> TimeSeries {
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| (i as f64) * 2.0 + 0.5 * (i as f64).sin())
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn make_seasonal_series(n: usize, period: usize) -> TimeSeries {
    let timestamps = make_timestamps(n);
    let values: Vec<f64> = (0..n)
        .map(|i| ((i % period) as f64) * 10.0 + (i as f64) * 0.1)
        .collect();
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn make_intermittent_series() -> TimeSeries {
    let timestamps = make_timestamps(20);
    let values = vec![
        5.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 3.0,
        0.0, 0.0,
    ];
    TimeSeries::univariate(timestamps, values).unwrap()
}

fn assert_intervals_valid(model: &dyn Forecaster, horizon: usize, level: f64, model_name: &str) {
    let forecast = model
        .predict_with_intervals(horizon, level)
        .unwrap_or_else(|e| panic!("{}: predict_with_intervals failed: {}", model_name, e));
    assert!(
        forecast.has_lower(),
        "{}: should have lower bounds at level {}",
        model_name,
        level
    );
    assert!(
        forecast.has_upper(),
        "{}: should have upper bounds at level {}",
        model_name,
        level
    );
    let preds = forecast.primary();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    for i in 0..horizon {
        assert!(
            lower[i] <= preds[i] + 1e-10,
            "{}: lower[{}] ({}) should be <= pred[{}] ({})",
            model_name,
            i,
            lower[i],
            i,
            preds[i]
        );
        assert!(
            preds[i] <= upper[i] + 1e-10,
            "{}: pred[{}] ({}) should be <= upper[{}] ({})",
            model_name,
            i,
            preds[i],
            i,
            upper[i]
        );
    }
}

fn assert_wider_at_higher_confidence(model: &dyn Forecaster, horizon: usize, model_name: &str) {
    let f80 = model.predict_with_intervals(horizon, 0.80).unwrap();
    let f95 = model.predict_with_intervals(horizon, 0.95).unwrap();
    let l80 = f80.lower_series(0).unwrap();
    let u80 = f80.upper_series(0).unwrap();
    let l95 = f95.lower_series(0).unwrap();
    let u95 = f95.upper_series(0).unwrap();
    for i in 0..horizon {
        let w80 = u80[i] - l80[i];
        let w95 = u95[i] - l95[i];
        assert!(
            w95 >= w80 - 1e-10,
            "{}: 95% width ({}) should be >= 80% width ({}) at h={}",
            model_name,
            w95,
            w80,
            i
        );
    }
}

#[test]
fn naive_intervals_nonempty() {
    let ts = make_trending_series(20);
    let mut model = Naive::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "Naive");
}

#[test]
fn naive_intervals_wider_at_higher_confidence() {
    let ts = make_trending_series(20);
    let mut model = Naive::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "Naive");
}

#[test]
fn naive_intervals_widen_with_horizon() {
    let ts = make_trending_series(20);
    let mut model = Naive::new();
    model.fit(&ts).unwrap();
    let forecast = model.predict_with_intervals(5, 0.95).unwrap();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    for i in 1..5 {
        let w_prev = upper[i - 1] - lower[i - 1];
        let w_curr = upper[i] - lower[i];
        assert!(
            w_curr > w_prev - 1e-10,
            "Naive: interval should widen with horizon"
        );
    }
}

#[test]
fn seasonal_naive_intervals_nonempty() {
    let ts = make_seasonal_series(16, 4);
    let mut model = SeasonalNaive::new(4);
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 8, 0.95, "SeasonalNaive");
}

#[test]
fn seasonal_naive_intervals_wider_at_higher_confidence() {
    let ts = make_seasonal_series(16, 4);
    let mut model = SeasonalNaive::new(4);
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 8, "SeasonalNaive");
}

#[test]
fn seasonal_naive_intervals_widen_across_seasons() {
    let ts = make_seasonal_series(16, 4);
    let mut model = SeasonalNaive::new(4);
    model.fit(&ts).unwrap();
    let forecast = model.predict_with_intervals(8, 0.95).unwrap();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    let w_first = upper[0] - lower[0];
    let w_second = upper[4] - lower[4];
    assert!(
        w_second > w_first,
        "SeasonalNaive: second season should be wider"
    );
}

#[test]
fn rw_drift_intervals_nonempty() {
    let ts = make_trending_series(20);
    let mut model = RandomWalkWithDrift::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "RandomWalkWithDrift");
}

#[test]
fn rw_drift_intervals_wider_at_higher_confidence() {
    let ts = make_trending_series(20);
    let mut model = RandomWalkWithDrift::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "RandomWalkWithDrift");
}

#[test]
fn rw_drift_intervals_widen_with_horizon() {
    let ts = make_trending_series(20);
    let mut model = RandomWalkWithDrift::new();
    model.fit(&ts).unwrap();
    let forecast = model.predict_with_intervals(5, 0.95).unwrap();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    for i in 1..5 {
        let w_prev = upper[i - 1] - lower[i - 1];
        let w_curr = upper[i] - lower[i];
        assert!(
            w_curr > w_prev - 1e-10,
            "RandomWalkWithDrift: interval should widen"
        );
    }
}

#[test]
fn sma_intervals_nonempty() {
    let ts = make_trending_series(20);
    let mut model = SimpleMovingAverage::new(5);
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "SMA");
}

#[test]
fn sma_intervals_wider_at_higher_confidence() {
    let ts = make_trending_series(20);
    let mut model = SimpleMovingAverage::new(5);
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "SMA");
}

#[test]
fn sma_intervals_constant_width() {
    let ts = make_trending_series(20);
    let mut model = SimpleMovingAverage::new(5);
    model.fit(&ts).unwrap();
    let forecast = model.predict_with_intervals(5, 0.95).unwrap();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    let w0 = upper[0] - lower[0];
    for i in 1..5 {
        let wi = upper[i] - lower[i];
        assert!(
            (wi - w0).abs() < 1e-10,
            "SMA: interval width should be constant"
        );
    }
}

#[test]
fn historic_average_intervals_nonempty() {
    let ts = make_trending_series(20);
    let mut model = HistoricAverage::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "HistoricAverage");
}

#[test]
fn historic_average_intervals_wider_at_higher_confidence() {
    let ts = make_trending_series(20);
    let mut model = HistoricAverage::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "HistoricAverage");
}

#[test]
fn window_average_intervals_nonempty() {
    let ts = make_trending_series(20);
    let mut model = WindowAverage::new(5);
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "WindowAverage");
}

#[test]
fn window_average_intervals_wider_at_higher_confidence() {
    let ts = make_trending_series(20);
    let mut model = WindowAverage::new(5);
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "WindowAverage");
}

#[test]
fn seasonal_window_avg_intervals_nonempty() {
    let ts = make_seasonal_series(16, 4);
    let mut model = SeasonalWindowAverage::new(4, 2);
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 8, 0.95, "SeasonalWindowAverage");
}

#[test]
fn seasonal_window_avg_intervals_wider_at_higher_confidence() {
    let ts = make_seasonal_series(16, 4);
    let mut model = SeasonalWindowAverage::new(4, 2);
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 8, "SeasonalWindowAverage");
}

#[test]
fn croston_intervals_nonempty() {
    let ts = make_intermittent_series();
    let mut model = Croston::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "Croston");
}

#[test]
fn croston_intervals_wider_at_higher_confidence() {
    let ts = make_intermittent_series();
    let mut model = Croston::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "Croston");
}

#[test]
fn croston_intervals_constant_width() {
    let ts = make_intermittent_series();
    let mut model = Croston::new();
    model.fit(&ts).unwrap();
    let forecast = model.predict_with_intervals(5, 0.95).unwrap();
    let lower = forecast.lower_series(0).unwrap();
    let upper = forecast.upper_series(0).unwrap();
    let w0 = upper[0] - lower[0];
    for i in 1..5 {
        let wi = upper[i] - lower[i];
        assert!(
            (wi - w0).abs() < 1e-10,
            "Croston: interval width should be constant"
        );
    }
}

#[test]
fn adida_intervals_nonempty() {
    let ts = make_intermittent_series();
    let mut model = ADIDA::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "ADIDA");
}

#[test]
fn adida_intervals_wider_at_higher_confidence() {
    let ts = make_intermittent_series();
    let mut model = ADIDA::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "ADIDA");
}

#[test]
fn tsb_intervals_nonempty() {
    let ts = make_intermittent_series();
    let mut model = TSB::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "TSB");
}

#[test]
fn tsb_intervals_wider_at_higher_confidence() {
    let ts = make_intermittent_series();
    let mut model = TSB::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "TSB");
}

#[test]
fn imapa_intervals_nonempty() {
    let ts = make_intermittent_series();
    let mut model = IMAPA::new();
    model.fit(&ts).unwrap();
    assert_intervals_valid(&model, 5, 0.95, "IMAPA");
}

#[test]
fn imapa_intervals_wider_at_higher_confidence() {
    let ts = make_intermittent_series();
    let mut model = IMAPA::new();
    model.fit(&ts).unwrap();
    assert_wider_at_higher_confidence(&model, 5, "IMAPA");
}

#[test]
fn all_models_handle_zero_horizon_intervals() {
    let ts = make_trending_series(20);
    let seasonal_ts = make_seasonal_series(16, 4);
    let intermittent_ts = make_intermittent_series();

    let mut naive = Naive::new();
    naive.fit(&ts).unwrap();
    assert!(naive.predict_with_intervals(0, 0.95).is_ok());

    let mut rw = RandomWalkWithDrift::new();
    rw.fit(&ts).unwrap();
    assert!(rw.predict_with_intervals(0, 0.95).is_ok());

    let mut sma = SimpleMovingAverage::new(5);
    sma.fit(&ts).unwrap();
    assert!(sma.predict_with_intervals(0, 0.95).is_ok());

    let mut ha = HistoricAverage::new();
    ha.fit(&ts).unwrap();
    assert!(ha.predict_with_intervals(0, 0.95).is_ok());

    let mut wa = WindowAverage::new(5);
    wa.fit(&ts).unwrap();
    assert!(wa.predict_with_intervals(0, 0.95).is_ok());

    let mut sn = SeasonalNaive::new(4);
    sn.fit(&seasonal_ts).unwrap();
    assert!(sn.predict_with_intervals(0, 0.95).is_ok());

    let mut swa = SeasonalWindowAverage::new(4, 2);
    swa.fit(&seasonal_ts).unwrap();
    assert!(swa.predict_with_intervals(0, 0.95).is_ok());

    let mut croston = Croston::new();
    croston.fit(&intermittent_ts).unwrap();
    assert!(croston.predict_with_intervals(0, 0.95).is_ok());

    let mut adida = ADIDA::new();
    adida.fit(&intermittent_ts).unwrap();
    assert!(adida.predict_with_intervals(0, 0.95).is_ok());

    let mut tsb = TSB::new();
    tsb.fit(&intermittent_ts).unwrap();
    assert!(tsb.predict_with_intervals(0, 0.95).is_ok());

    let mut imapa = IMAPA::new();
    imapa.fit(&intermittent_ts).unwrap();
    assert!(imapa.predict_with_intervals(0, 0.95).is_ok());
}
