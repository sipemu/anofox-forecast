//! Complete model diagnostics pipeline.
use crate::error::ForecastError;
use crate::features::autocorrelation::{autocorrelation, partial_autocorrelation};
use crate::models::Forecaster;
use crate::validation::residual_tests::{jarque_bera, ljung_box};
/// Complete diagnostic report for a fitted model.
#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    pub ljung_box_pvalue: f64,
    pub ljung_box_lags: usize,
    pub residual_acf: Vec<f64>,
    pub residual_pacf: Vec<f64>,
    pub jarque_bera_statistic: f64,
    pub jarque_bera_pvalue: f64,
    pub heteroscedasticity_pvalue: f64,
    pub residual_mean: f64,
    pub residual_std: f64,
    pub passes_all: bool,
}
impl ModelDiagnostics {
    pub fn from_residuals(residuals: &[f64], significance: f64) -> Self {
        let n = residuals.len();
        let residual_mean = if n > 0 {
            residuals.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let residual_std = if n > 1 {
            (residuals
                .iter()
                .map(|r| (r - residual_mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        let max_lag = 10.min(n / 5).max(1);
        let lb = ljung_box(residuals, Some(max_lag), 0);
        let acf_max = max_lag.min(n.saturating_sub(1));
        let residual_acf: Vec<f64> = (1..=acf_max)
            .map(|l| autocorrelation(residuals, l))
            .collect();
        let residual_pacf: Vec<f64> = (1..=acf_max)
            .map(|l| partial_autocorrelation(residuals, l))
            .collect();
        let jb = jarque_bera(residuals);
        let hp = breusch_pagan(residuals);
        let passes_all =
            lb.p_value > significance && jb.p_value > significance && hp > significance;
        ModelDiagnostics {
            ljung_box_pvalue: lb.p_value,
            ljung_box_lags: lb.lags,
            residual_acf,
            residual_pacf,
            jarque_bera_statistic: jb.statistic,
            jarque_bera_pvalue: jb.p_value,
            heteroscedasticity_pvalue: hp,
            residual_mean,
            residual_std,
            passes_all,
        }
    }
    pub fn from_forecaster(
        model: &dyn Forecaster,
        significance: f64,
    ) -> Result<Self, ForecastError> {
        let raw = model
            .residuals()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let r: Vec<f64> = raw
            .iter()
            .copied()
            .filter(|v: &f64| v.is_finite())
            .collect();
        Ok(Self::from_residuals(&r, significance))
    }
    pub fn summary(&self) -> String {
        format!("Model Diagnostics\n=================\nResidual mean:     {:.6}\nResidual std:      {:.6}\nLjung-Box p-value: {:.4} (lags={}){}\nJarque-Bera stat:  {:.4}, p-value: {:.4}{}\nBreusch-Pagan p:   {:.4}{}\nOverall:           {}",
            self.residual_mean, self.residual_std,
            self.ljung_box_pvalue, self.ljung_box_lags,
            if self.ljung_box_pvalue <= 0.05 { " [FAIL: autocorrelation detected]" } else { " [PASS]" },
            self.jarque_bera_statistic, self.jarque_bera_pvalue,
            if self.jarque_bera_pvalue <= 0.05 { " [FAIL: non-normal residuals]" } else { " [PASS]" },
            self.heteroscedasticity_pvalue,
            if self.heteroscedasticity_pvalue <= 0.05 { " [FAIL: heteroscedasticity detected]" } else { " [PASS]" },
            if self.passes_all { "PASS - residuals appear well-behaved" } else { "FAIL - residuals show issues" })
    }
}
fn breusch_pagan(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n < 4 {
        return 1.0;
    }
    let sq: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let tm = (n - 1) as f64 / 2.0;
    let sm: f64 = sq.iter().sum::<f64>() / n as f64;
    let (mut st, mut sp, mut sr) = (0.0, 0.0, 0.0);
    for (i, &s) in sq.iter().enumerate() {
        let tc = i as f64 - tm;
        let sc = s - sm;
        st += tc * tc;
        sp += tc * sc;
        sr += sc * sc;
    }
    if st < 1e-30 || sr < 1e-30 {
        return 1.0;
    }
    let b = sp / st;
    let ssreg = b * b * st;
    let ssres = sr - ssreg;
    if ssres <= 0.0 {
        return 0.0;
    }
    let f = ssreg / (ssres / (n - 2) as f64);
    if f < 0.0 || f.is_nan() || f.is_infinite() {
        return 1.0;
    }
    (t_distribution_sf(f.sqrt(), (n - 2) as f64) * 2.0).clamp(0.0, 1.0)
}
fn t_distribution_sf(t: f64, df: f64) -> f64 {
    if df <= 0.0 || t.is_nan() {
        return f64::NAN;
    }
    if df > 100.0 {
        return 0.5 * erfc_a(t * (1.0 - 1.0 / (4.0 * df)) / std::f64::consts::SQRT_2);
    }
    0.5 * rib(df / (df + t * t), df / 2.0, 0.5)
}
fn rib(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - rib(1.0 - x, b, a);
    }
    let pf = (a * x.ln() + b * (1.0 - x).ln() - lnb(a, b) - a.ln()).exp();
    let mut c = 1.0_f64;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(1e-30);
    let mut h = d;
    for m in 1..200 {
        let mf = m as f64;
        let ne = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 / (1.0 + ne * d).max(1e-30);
        c = (1.0 + ne / c).max(1e-30);
        h *= d * c;
        let no = -((a + mf) * (a + b + mf) * x) / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 / (1.0 + no * d).max(1e-30);
        c = (1.0 + no / c).max(1e-30);
        let dl = d * c;
        h *= dl;
        if (dl - 1.0).abs() < 1e-14 {
            break;
        }
    }
    pf * h
}
fn lnb(a: f64, b: f64) -> f64 {
    lng(a) + lng(b) - lng(a + b)
}
fn lng(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let cs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut t = x + 5.5;
    t -= (x + 0.5) * t.ln();
    let mut s = 1.000000000190015;
    for (j, &c) in cs.iter().enumerate() {
        s += c / (x + 1.0 + j as f64);
    }
    -t + (2.5066282746310005 * s / x).ln()
}
fn erfc_a(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 {
        tau
    } else {
        2.0 - tau
    }
}
impl std::fmt::Display for ModelDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::models::baseline::Naive;
    use crate::models::Forecaster;
    use chrono::{TimeZone, Utc};
    fn mts(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }
    fn wn(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = (s >> 33) as f64 / (1u64 << 31) as f64;
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (s >> 33) as f64 / (1u64 << 31) as f64;
                let u1 = u1.clamp(1e-10, 1.0 - 1e-10);
                let u2 = u2.clamp(1e-10, 1.0 - 1e-10);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect()
    }
    #[test]
    fn white_noise_passes_all() {
        let r = wn(1000, 12345);
        let d = ModelDiagnostics::from_residuals(&r, 0.01);
        assert!(d.ljung_box_pvalue > 0.01);
        assert!(d.jarque_bera_pvalue > 0.01);
        assert!(d.heteroscedasticity_pvalue > 0.01);
        assert!(d.passes_all);
        assert!(d.residual_mean.abs() < 0.5);
        assert!(d.residual_std > 0.0);
        assert!(!d.residual_acf.is_empty());
        assert!(!d.residual_pacf.is_empty());
        for (i, &v) in d.residual_acf.iter().enumerate() {
            assert!(v.abs() < 0.2, "ACF lag {}={}", i + 1, v);
        }
    }
    #[test]
    fn autocorrelated_fails_ljung_box() {
        let noise = wn(300, 123);
        let mut r = vec![0.0; 300];
        r[0] = noise[0];
        for i in 1..300 {
            r[i] = 0.9 * r[i - 1] + 0.1 * noise[i];
        }
        let d = ModelDiagnostics::from_residuals(&r, 0.05);
        assert!(d.ljung_box_pvalue < 0.05);
        assert!(!d.passes_all);
    }
    #[test]
    fn non_normal_fails_jarque_bera() {
        let mut r = vec![0.1; 400];
        for i in 0..40 {
            r[i * 10] = 20.0;
        }
        let d = ModelDiagnostics::from_residuals(&r, 0.05);
        assert!(d.jarque_bera_pvalue < 0.05);
        assert!(!d.passes_all);
    }
    #[test]
    fn heteroscedastic_fails_breusch_pagan() {
        let n = wn(500, 999);
        let r: Vec<f64> = n
            .iter()
            .enumerate()
            .map(|(i, &e)| e * (1.0 + i as f64 * 0.05))
            .collect();
        let d = ModelDiagnostics::from_residuals(&r, 0.05);
        assert!(d.heteroscedasticity_pvalue < 0.05);
        assert!(!d.passes_all);
    }
    #[test]
    fn from_forecaster_naive() {
        let ts = TimeSeries::univariate(
            mts(100),
            (1..=100)
                .map(|i| 10.0 + 0.5 * (i as f64).sin() + 0.3 * (i as f64 * 0.7).cos())
                .collect(),
        )
        .unwrap();
        let mut m = Naive::new();
        m.fit(&ts).unwrap();
        let d = ModelDiagnostics::from_forecaster(&m, 0.05).unwrap();
        assert!(!d.ljung_box_pvalue.is_nan());
        assert!(!d.jarque_bera_pvalue.is_nan());
        assert!(!d.heteroscedasticity_pvalue.is_nan());
        assert!(!d.residual_mean.is_nan());
        assert!(!d.residual_std.is_nan());
        assert!(!d.residual_acf.is_empty());
        assert!(!d.residual_pacf.is_empty());
    }
    #[test]
    fn from_forecaster_not_fitted() {
        let m = Naive::new();
        let r = ModelDiagnostics::from_forecaster(&m, 0.05);
        assert!(r.is_err());
        assert!(matches!(r.unwrap_err(), ForecastError::FitRequired { .. }));
    }
    #[test]
    fn summary_contains_key_info() {
        let d = ModelDiagnostics::from_residuals(&wn(200, 77), 0.05);
        let s = d.summary();
        assert!(s.contains("Model Diagnostics"));
        assert!(s.contains("Ljung-Box"));
        assert!(s.contains("Jarque-Bera"));
        assert!(s.contains("Breusch-Pagan"));
        assert!(s.contains("Residual mean"));
        assert!(s.contains("Residual std"));
        assert!(s.contains("Overall:"));
    }
    #[test]
    fn display_matches_summary() {
        let d = ModelDiagnostics::from_residuals(&wn(200, 55), 0.05);
        assert_eq!(format!("{}", d), d.summary());
    }
    #[test]
    fn empty_residuals() {
        let d = ModelDiagnostics::from_residuals(&[], 0.05);
        assert!(d.residual_mean.abs() < 1e-10);
        assert!(d.residual_std.abs() < 1e-10);
    }
    #[test]
    fn constant_residuals() {
        let d = ModelDiagnostics::from_residuals(&vec![3.0; 100], 0.05);
        assert!((d.residual_mean - 3.0).abs() < 1e-10);
        assert!(d.residual_std < 1e-10);
    }
    #[test]
    fn very_short_residuals() {
        let d = ModelDiagnostics::from_residuals(&[1.0, -1.0, 0.5], 0.05);
        assert!(!d.residual_mean.is_nan());
    }
    #[test]
    fn breusch_pagan_constant_variance() {
        assert!(breusch_pagan(&wn(300, 42)) > 0.05);
    }
    #[test]
    fn breusch_pagan_increasing_variance() {
        let n = wn(400, 88);
        let r: Vec<f64> = n
            .iter()
            .enumerate()
            .map(|(i, &e)| e * (1.0 + i as f64 * 0.04))
            .collect();
        assert!(breusch_pagan(&r) < 0.05);
    }
    #[test]
    fn breusch_pagan_too_short() {
        assert!((breusch_pagan(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn breusch_pagan_empty() {
        assert!((breusch_pagan(&[]) - 1.0).abs() < 1e-10);
    }
}
