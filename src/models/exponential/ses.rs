//! Simple Exponential Smoothing (SES) forecasting model.
//!
//! SES is suitable for forecasting data with no clear trend or seasonality.

use crate::core::{Forecast, TimeSeries};
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;
use crate::utils::optimization::{nelder_mead, NelderMeadConfig};
use crate::utils::stats::quantile_normal;

/// Simple Exponential Smoothing forecaster.
///
/// The model equation is:
/// `level_t = α × y_t + (1-α) × level_{t-1}`
///
/// where α (alpha) is the smoothing parameter (0 < α < 1).
///
/// # Example
/// ```
/// use anofox_forecast::models::exponential::SimpleExponentialSmoothing;
/// use anofox_forecast::models::Forecaster;
/// use anofox_forecast::core::TimeSeries;
/// use chrono::{TimeZone, Utc, Duration};
///
/// let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
/// let timestamps: Vec<_> = (0..10).map(|i| base + Duration::hours(i)).collect();
/// let values = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
/// let ts = TimeSeries::univariate(timestamps, values).unwrap();
///
/// let mut model = SimpleExponentialSmoothing::new(0.3);
/// model.fit(&ts).unwrap();
///
/// let forecast = model.predict(3).unwrap();
/// assert_eq!(forecast.horizon(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleExponentialSmoothing {
    /// Smoothing parameter (0 < alpha < 1).
    alpha: Option<f64>,
    /// Whether to optimize alpha automatically.
    optimize: bool,
    /// Current level state.
    level: Option<f64>,
    /// Fitted values.
    fitted: Option<Vec<f64>>,
    /// Residuals.
    residuals: Option<Vec<f64>>,
    /// Residual variance for prediction intervals.
    residual_variance: Option<f64>,
    /// Original series length.
    n: usize,
}

impl SimpleExponentialSmoothing {
    /// Create a new SES model with a fixed smoothing parameter.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing parameter (0 < alpha < 1)
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: Some(alpha.clamp(0.0001, 0.9999)),
            optimize: false,
            level: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Create a new SES model with automatic alpha optimization.
    ///
    /// Alpha will be chosen to minimize the sum of squared errors.
    pub fn auto() -> Self {
        Self {
            alpha: None,
            optimize: true,
            level: None,
            fitted: None,
            residuals: None,
            residual_variance: None,
            n: 0,
        }
    }

    /// Get the smoothing parameter.
    pub fn alpha(&self) -> Option<f64> {
        self.alpha
    }

    /// Get the current level.
    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// Calculate SSE for a given alpha value.
    fn calculate_sse(values: &[f64], alpha: f64) -> f64 {
        if values.is_empty() {
            return f64::MAX;
        }

        let mut level = values[0];
        let mut sse = 0.0;

        for &y in &values[1..] {
            let error = y - level;
            sse += error * error;
            level = alpha * y + (1.0 - alpha) * level;
        }

        sse
    }

    /// Optimize alpha using Nelder-Mead.
    fn optimize_alpha(values: &[f64]) -> f64 {
        let config = NelderMeadConfig {
            max_iter: 500,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = nelder_mead(
            |params| Self::calculate_sse(values, params[0]),
            &[0.5],
            Some(&[(0.0001, 0.9999)]),
            config,
        );

        result.optimal_point[0].clamp(0.0001, 0.9999)
    }
}

impl Default for SimpleExponentialSmoothing {
    fn default() -> Self {
        Self::auto()
    }
}

impl Forecaster for SimpleExponentialSmoothing {
    fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        let values = series.primary_values();
        if values.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        self.n = values.len();

        // Optimize alpha if needed
        if self.optimize {
            self.alpha = Some(Self::optimize_alpha(values));
        }

        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;

        // Initialize level with first observation
        let mut level = values[0];
        let mut fitted = Vec::with_capacity(self.n);
        let mut residuals = Vec::with_capacity(self.n);

        // First fitted value is the initial level
        fitted.push(level);
        residuals.push(0.0); // No residual for first observation

        // Compute fitted values and residuals
        for &y in &values[1..] {
            let forecast = level;
            fitted.push(forecast);
            residuals.push(y - forecast);
            level = alpha * y + (1.0 - alpha) * level;
        }

        self.level = Some(level);
        self.fitted = Some(fitted);

        // Calculate residual variance (excluding first observation)
        let valid_residuals: Vec<f64> = residuals[1..].to_vec();
        if !valid_residuals.is_empty() {
            let variance =
                crate::simd::sum_of_squares(&valid_residuals) / valid_residuals.len() as f64;
            self.residual_variance = Some(variance);
        }

        self.residuals = Some(residuals);

        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<Forecast> {
        let level = self.level.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        // SES produces flat forecasts at the final level
        let predictions = vec![level; horizon];
        Ok(Forecast::from_values(predictions))
    }

    fn predict_with_intervals(&self, horizon: usize, level: f64) -> Result<Forecast> {
        let current_level = self.level.ok_or(ForecastError::FitRequired)?;
        let variance = self.residual_variance.unwrap_or(0.0);
        let alpha = self.alpha.ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(Forecast::new());
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let predictions = vec![current_level; horizon];
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            // Variance increases with forecast horizon
            // Var(e_{n+h}) = sigma^2 * (1 + sum_{j=1}^{h-1} (1-alpha)^{2j})
            // = sigma^2 * (1 + (1-alpha)^2 * (1 - (1-alpha)^{2(h-1)}) / (1 - (1-alpha)^2))
            let factor = if h == 1 {
                1.0
            } else {
                let beta = 1.0 - alpha;
                let beta2 = beta * beta;
                if (1.0 - beta2).abs() < 1e-10 {
                    h as f64
                } else {
                    1.0 + beta2 * (1.0 - beta2.powi((h - 1) as i32)) / (1.0 - beta2)
                }
            };
            let se = (variance * factor).sqrt();
            lower.push(current_level - z * se);
            upper.push(current_level + z * se);
        }

        Ok(Forecast::from_values_with_intervals(
            predictions,
            lower,
            upper,
        ))
    }

    fn fitted_values(&self) -> Option<&[f64]> {
        self.fitted.as_deref()
    }

    fn fitted_values_with_intervals(&self, level: f64) -> Option<Forecast> {
        let fitted = self.fitted.as_ref()?;
        let variance = self.residual_variance?;

        if variance <= 0.0 {
            return Some(Forecast::from_values(fitted.clone()));
        }

        let z = quantile_normal((1.0 + level) / 2.0);
        let sigma = variance.sqrt();

        let lower: Vec<f64> = fitted.iter().map(|&f| f - z * sigma).collect();
        let upper: Vec<f64> = fitted.iter().map(|&f| f + z * sigma).collect();

        Some(Forecast::from_values_with_intervals(
            fitted.clone(),
            lower,
            upper,
        ))
    }

    fn residuals(&self) -> Option<&[f64]> {
        self.residuals.as_deref()
    }

    fn name(&self) -> &str {
        "SimpleExponentialSmoothing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::{Duration, TimeZone, Utc};

    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        (0..n).map(|i| base + Duration::hours(i as i64)).collect()
    }

    #[test]
    fn ses_with_fixed_alpha() {
        let timestamps = make_timestamps(10);
        let values = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        assert_relative_eq!(model.alpha().unwrap(), 0.3, epsilon = 1e-10);
        assert!(model.level().is_some());

        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.horizon(), 3);

        // All forecasts should be equal (flat)
        let preds = forecast.primary();
        assert_relative_eq!(preds[0], preds[1], epsilon = 1e-10);
        assert_relative_eq!(preds[1], preds[2], epsilon = 1e-10);
    }

    #[test]
    fn ses_auto_optimization() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + 0.5 * (i as f64).sin()).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::auto();
        model.fit(&ts).unwrap();

        let alpha = model.alpha().unwrap();
        assert!(alpha > 0.0 && alpha < 1.0);

        let forecast = model.predict(5).unwrap();
        assert_eq!(forecast.horizon(), 5);
    }

    #[test]
    fn ses_constant_series() {
        let timestamps = make_timestamps(10);
        let values = vec![5.0; 10];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.5);
        model.fit(&ts).unwrap();

        let forecast = model.predict(3).unwrap();
        let preds = forecast.primary();

        // For constant series, forecast should equal the constant
        for pred in preds {
            assert_relative_eq!(*pred, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn ses_known_calculation() {
        // Manually verify SES calculation
        let timestamps = make_timestamps(4);
        let values = vec![10.0, 12.0, 14.0, 13.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let alpha = 0.5;
        let mut model = SimpleExponentialSmoothing::new(alpha);
        model.fit(&ts).unwrap();

        // Level calculation:
        // l_0 = 10
        // l_1 = 0.5*12 + 0.5*10 = 11
        // l_2 = 0.5*14 + 0.5*11 = 12.5
        // l_3 = 0.5*13 + 0.5*12.5 = 12.75
        assert_relative_eq!(model.level().unwrap(), 12.75, epsilon = 1e-10);

        // Fitted values should be the previous levels
        let fitted = model.fitted_values().unwrap();
        assert_relative_eq!(fitted[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(fitted[1], 10.0, epsilon = 1e-10);
        assert_relative_eq!(fitted[2], 11.0, epsilon = 1e-10);
        assert_relative_eq!(fitted[3], 12.5, epsilon = 1e-10);
    }

    #[test]
    fn ses_residuals_are_correct() {
        let timestamps = make_timestamps(5);
        let values = vec![10.0, 12.0, 11.0, 13.0, 14.0];
        let ts = TimeSeries::univariate(timestamps, values.clone()).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values().unwrap();
        let residuals = model.residuals().unwrap();

        // Check residuals = actual - fitted
        for i in 1..5 {
            assert_relative_eq!(residuals[i], values[i] - fitted[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn ses_confidence_intervals() {
        let timestamps = make_timestamps(20);
        let values: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let forecast = model.predict_with_intervals(5, 0.95).unwrap();
        assert!(forecast.has_lower());
        assert!(forecast.has_upper());

        let lower = forecast.lower_series(0).unwrap();
        let upper = forecast.upper_series(0).unwrap();
        let preds = forecast.primary();

        // Intervals should contain the point forecast
        for i in 0..5 {
            assert!(lower[i] < preds[i]);
            assert!(upper[i] > preds[i]);
        }

        // Intervals should widen with horizon
        let width_1 = upper[0] - lower[0];
        let width_5 = upper[4] - lower[4];
        assert!(width_5 >= width_1);
    }

    #[test]
    fn ses_alpha_clamped_to_valid_range() {
        // Alpha should be clamped to (0, 1)
        let model_low = SimpleExponentialSmoothing::new(-0.5);
        assert!(model_low.alpha().unwrap() > 0.0);

        let model_high = SimpleExponentialSmoothing::new(1.5);
        assert!(model_high.alpha().unwrap() < 1.0);
    }

    #[test]
    fn ses_empty_data_returns_error() {
        let timestamps: Vec<chrono::DateTime<Utc>> = vec![];
        let values: Vec<f64> = vec![];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        assert!(matches!(model.fit(&ts), Err(ForecastError::EmptyData)));
    }

    #[test]
    fn ses_requires_fit_before_predict() {
        let model = SimpleExponentialSmoothing::new(0.3);
        assert!(matches!(model.predict(5), Err(ForecastError::FitRequired)));
    }

    #[test]
    fn ses_zero_horizon_returns_empty() {
        let timestamps = make_timestamps(5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let forecast = model.predict(0).unwrap();
        assert_eq!(forecast.horizon(), 0);
    }

    #[test]
    fn ses_name_is_correct() {
        let model = SimpleExponentialSmoothing::new(0.3);
        assert_eq!(model.name(), "SimpleExponentialSmoothing");
    }

    #[test]
    fn ses_default_is_auto() {
        let model = SimpleExponentialSmoothing::default();
        assert!(model.optimize);
        assert!(model.alpha.is_none());
    }

    #[test]
    fn ses_high_alpha_responds_quickly() {
        let timestamps = make_timestamps(10);
        // Step change from 10 to 20
        let values = vec![10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0];
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model_low = SimpleExponentialSmoothing::new(0.1);
        let mut model_high = SimpleExponentialSmoothing::new(0.9);

        model_low.fit(&ts).unwrap();
        model_high.fit(&ts).unwrap();

        // High alpha should be closer to 20
        assert!(model_high.level().unwrap() > model_low.level().unwrap());
    }

    /// Validation test comparing SES output with statsforecast (NIXTLA).
    ///
    /// Data: Synthetic stationary series (100 observations, seed=42, mean=50, std=5)
    /// Generated by: validation/generate_data.py
    /// Reference: statsforecast.models.SimpleExponentialSmoothing(alpha=0.1)
    ///
    /// This test ensures our SES implementation matches the industry-standard
    /// statsforecast library when using the same fixed alpha parameter.
    #[test]
    fn ses_matches_statsforecast_stationary() {
        // Stationary series: 100 observations, seed=42, mean=50, std=5
        // Source: validation/data/stationary.csv (full precision)
        let values = vec![
            51.52358539877216,
            44.80007946879752,
            53.752255979032284,
            54.70282358195607,
            40.24482405673082,
            43.48910246568841,
            50.639202015836425,
            48.41878703828209,
            49.91599421247856,
            45.7347803621321,
            54.396989874314144,
            53.88895967714474,
            50.33015348780608,
            55.63620603484016,
            52.33754671126023,
            45.70353768558381,
            51.843753920412496,
            45.20558699585501,
            54.39225150653636,
            49.75037044506873,
            49.075688182273694,
            46.5953522779803,
            56.11270669337015,
            49.22735258965599,
            47.85836088918447,
            48.23933224755885,
            52.661545927766745,
            51.82722032182039,
            52.06366305797994,
            52.15410501503941,
            60.70823800435231,
            47.967924918076925,
            47.43878635464231,
            45.93113635876061,
            53.07989711287748,
            55.64486146360446,
            49.430262711725625,
            45.79921761518736,
            45.877593921543806,
            53.252963939123504,
            53.71627085601721,
            52.71577134152597,
            46.67245146355653,
            51.1608066153336,
            50.58342904570364,
            51.09344298364506,
            54.35714388974095,
            51.11797774387341,
            53.39456781535947,
            50.33789534744446,
            51.44559699344992,
            53.1564411291927,
            42.71422090072167,
            48.401643918213495,
            47.64813672853602,
            46.80561075878329,
            48.62428874386658,
            57.47470655617198,
            45.670844421533786,
            54.84139177295741,
            41.58565114192098,
            48.32557485007113,
            50.81376532552503,
            52.93111165679639,
            53.55613289896427,
            53.96673617599963,
            48.25637463875781,
            47.68824103667716,
            54.28987940628577,
            49.04347837559192,
            43.62156838331039,
            44.3335639299826,
            45.402738569991946,
            52.48580372026882,
            50.712128680352826,
            53.45242677033884,
            47.86373676831733,
            50.79269845538357,
            53.12795196983669,
            48.45326730139881,
            52.28387618778706,
            46.69037029466674,
            48.18473076717464,
            48.09131053000836,
            44.0208017720548,
            52.43486240392791,
            47.65298829898638,
            50.06247059363844,
            52.403733294529545,
            52.23265588014972,
            53.32692554486393,
            49.50757257745288,
            47.883508439779234,
            49.601408945468,
            41.56332783020985,
            42.76443763788456,
            43.38650193822799,
            45.01376586199259,
            51.99887113361719,
            45.472604723199694,
        ];
        let timestamps = make_timestamps(values.len());
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        // Use fixed alpha=0.1 to match statsforecast validation
        let mut model = SimpleExponentialSmoothing::new(0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Expected values from statsforecast.models.SimpleExponentialSmoothing(alpha=0.1)
        // SES produces flat forecasts (all values equal to final level)
        let expected = 47.77994447774433;

        // All 12 forecasts should be the same (flat forecast)
        for &pred in preds.iter() {
            assert_relative_eq!(pred, expected, epsilon = 1e-6);
        }
    }

    /// Validation test comparing SES output with statsforecast on trend data.
    ///
    /// Data: Synthetic trend series (100 observations, seed=42, intercept=10, slope=0.5)
    /// Generated by: validation/generate_data.py
    /// Reference: statsforecast.models.SimpleExponentialSmoothing(alpha=0.1)
    #[test]
    fn ses_matches_statsforecast_trend() {
        // Trend series: 100 observations, seed=42, intercept=10, slope=0.5
        // Source: validation/data/trend.csv (full precision)
        let values = vec![
            8.865512337881832,
            14.397684893358196,
            9.931208086815722,
            13.71254670540126,
            9.199146959970369,
            11.88368732639711,
            10.149933835268255,
            12.482900772298311,
            16.520924412372185,
            9.318038730422954,
            16.303270930637574,
            16.213206806996833,
            14.217550132909617,
            12.161826436834636,
            17.21638852314161,
            15.911521872808592,
            18.698028634064112,
            18.56555643657033,
            23.805336673962746,
            18.781933117580927,
            16.929507522134404,
            21.037826904868947,
            21.65999005191529,
            25.577562725721307,
            24.505333737743737,
            23.57061317744853,
            27.389908673658685,
            19.933710837031448,
            22.08074540175076,
            21.720272175783425,
            23.83057059053269,
            21.369941557331074,
            27.90545284044321,
            25.83333190870368,
            22.587581116492025,
            24.453262756377377,
            28.940541542350587,
            31.014379703683144,
            34.99019267507536,
            38.24158739802199,
            31.24322829982799,
            27.531385639904407,
            24.603861157806072,
            32.303134387031506,
            29.56117671406902,
            31.253928219460946,
            31.16370960282057,
            33.077627340750844,
            37.19794069236293,
            34.97114570233603,
            34.52409548888394,
            32.39303874152257,
            30.97595116588693,
            35.04107627278001,
            36.83865234754504,
            42.803789740739646,
            38.39082356441866,
            41.44821853306917,
            37.50211320438255,
            35.94516870074892,
            37.10464971330288,
            38.32432180639274,
            47.38540919730549,
            39.03583996232684,
            44.51546761120903,
            39.79121846573892,
            45.79471903862273,
            44.65485289831759,
            43.53008630702573,
            44.3777124215937,
            43.03563691371183,
            46.838216604446245,
            44.63504955897766,
            42.823182708698255,
            43.16618727704114,
            48.01776375316636,
            52.73727376923131,
            48.97997484072032,
            48.64408502167035,
            50.35747841880763,
            53.91800522512048,
            51.158147504091566,
            49.76721830749879,
            54.818866130179664,
            53.28626931538484,
            57.10726797598797,
            53.54970331166572,
            49.8265929048385,
            49.895522402263005,
            59.45278379669375,
            60.17099716234989,
            54.9614423601522,
            54.85043803659204,
            60.8843328767266,
            53.67886295386953,
            54.81581894313252,
            59.929980384067136,
            57.31618463142123,
            58.98463439983978,
            59.00967130442646,
        ];
        let timestamps = make_timestamps(values.len());
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.1);
        model.fit(&ts).unwrap();

        let forecast = model.predict(12).unwrap();
        let preds = forecast.primary();

        // Expected value from statsforecast
        let expected = 55.16535189466602;

        for &pred in preds.iter() {
            assert_relative_eq!(pred, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn ses_fitted_values_with_intervals() {
        let n = 20;
        let timestamps = make_timestamps(n);
        let values: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let ts = TimeSeries::univariate(timestamps, values).unwrap();

        let mut model = SimpleExponentialSmoothing::new(0.3);
        model.fit(&ts).unwrap();

        let fitted = model.fitted_values_with_intervals(0.95).unwrap();
        assert!(fitted.has_lower());
        assert!(fitted.has_upper());
        assert_eq!(fitted.horizon(), n);

        let lower = fitted.lower_series(0).unwrap();
        let upper = fitted.upper_series(0).unwrap();
        let primary = fitted.primary();

        // Intervals should contain the point forecast
        for i in 0..n {
            assert!(lower[i] <= primary[i]);
            assert!(upper[i] >= primary[i]);
        }

        // Intervals should have constant width (in-sample variance is constant)
        let width_first = upper[1] - lower[1]; // Skip first which may be NaN
        let width_last = upper[n - 1] - lower[n - 1];
        assert_relative_eq!(width_first, width_last, epsilon = 1e-10);
    }
}
