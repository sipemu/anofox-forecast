//! Forecast explanation and decomposition.
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct ForecastExplanation {
    pub level: Vec<f64>,
    pub trend: Option<Vec<f64>>,
    pub seasonal: Option<Vec<f64>>,
    pub residual: Option<Vec<f64>>,
    pub named_components: Vec<(String, Vec<f64>)>,
}

impl ForecastExplanation {
    pub fn sum(&self) -> Vec<f64> {
        let n = self.level.len();
        let mut total = self.level.clone();
        if let Some(ref trend) = self.trend {
            for i in 0..n.min(trend.len()) {
                total[i] += trend[i];
            }
        }
        if let Some(ref seasonal) = self.seasonal {
            for i in 0..n.min(seasonal.len()) {
                total[i] += seasonal[i];
            }
        }
        for (_, ref vals) in &self.named_components {
            for i in 0..n.min(vals.len()) {
                total[i] += vals[i];
            }
        }
        total
    }
    pub fn has_correct_lengths(&self, expected: usize) -> bool {
        if self.level.len() != expected {
            return false;
        }
        if let Some(ref t) = self.trend {
            if t.len() != expected {
                return false;
            }
        }
        if let Some(ref s) = self.seasonal {
            if s.len() != expected {
                return false;
            }
        }
        if let Some(ref r) = self.residual {
            if r.len() != expected {
                return false;
            }
        }
        for (_, ref v) in &self.named_components {
            if v.len() != expected {
                return false;
            }
        }
        true
    }
}

pub trait Explainable {
    fn explain(&self, horizon: usize) -> Result<ForecastExplanation>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TimeSeries;
    use crate::models::exponential::{ETSSpec, ETS};
    use crate::models::mstl_forecaster::MSTLForecaster;
    use crate::models::theta::Theta;
    use crate::models::Forecaster;
    use chrono::{TimeZone, Utc};
    fn make_timestamps(n: usize) -> Vec<chrono::DateTime<Utc>> {
        (0..n)
            .map(|i| {
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64)
            })
            .collect()
    }
    #[test]
    fn ets_explanation_components_sum_to_forecast() {
        let n = 50;
        let ts = TimeSeries::univariate(
            make_timestamps(n),
            (0..n)
                .map(|i| {
                    10.0 + 0.5 * i as f64
                        + 3.0 * (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()
                })
                .collect(),
        )
        .unwrap();
        let mut model = ETS::new(ETSSpec::aaa(), 12);
        model.fit(&ts).unwrap();
        let horizon = 12;
        let forecast = model.predict(horizon).unwrap();
        let explanation = model.explain(horizon).unwrap();
        let reconstructed = explanation.sum();
        for (i, (&pred, &recon)) in forecast
            .primary()
            .iter()
            .zip(reconstructed.iter())
            .enumerate()
        {
            assert!(
                (pred - recon).abs() < 1e-6,
                "ETS mismatch at h={}: pred={}, recon={}",
                i,
                pred,
                recon
            );
        }
    }
    #[test]
    fn ets_explanation_correct_lengths() {
        let ts = TimeSeries::univariate(
            make_timestamps(30),
            (0..30).map(|i| 10.0 + i as f64).collect(),
        )
        .unwrap();
        let mut model = ETS::new(ETSSpec::aan(), 1);
        model.fit(&ts).unwrap();
        let explanation = model.explain(5).unwrap();
        assert!(explanation.has_correct_lengths(5));
        assert!(explanation.trend.is_some());
        assert!(explanation.seasonal.is_none());
    }
    #[test]
    fn ets_ann_explanation() {
        let ts = TimeSeries::univariate(
            make_timestamps(30),
            (0..30).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect(),
        )
        .unwrap();
        let mut model = ETS::new(ETSSpec::ann(), 1);
        model.fit(&ts).unwrap();
        let explanation = model.explain(5).unwrap();
        assert_eq!(explanation.level.len(), 5);
        assert!(explanation.trend.is_none());
        assert!(explanation.seasonal.is_none());
    }
    #[test]
    fn theta_explanation_components_sum() {
        let ts = TimeSeries::univariate(
            make_timestamps(50),
            (0..50).map(|i| 10.0 + 0.3 * i as f64).collect(),
        )
        .unwrap();
        let mut model = Theta::new();
        model.fit(&ts).unwrap();
        let horizon = 10;
        let forecast = model.predict(horizon).unwrap();
        let explanation = model.explain(horizon).unwrap();
        let reconstructed = explanation.sum();
        for (i, (&pred, &recon)) in forecast
            .primary()
            .iter()
            .zip(reconstructed.iter())
            .enumerate()
        {
            assert!(
                (pred - recon).abs() < 1e-6,
                "Theta mismatch at h={}: pred={}, recon={}",
                i,
                pred,
                recon
            );
        }
    }
    #[test]
    fn mstl_explanation_components_sum() {
        let ts = TimeSeries::univariate(
            make_timestamps(100),
            (0..100)
                .map(|i| {
                    50.0 + 0.1 * i as f64
                        + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin()
                })
                .collect(),
        )
        .unwrap();
        let mut model = MSTLForecaster::new(vec![7]);
        model.fit(&ts).unwrap();
        let horizon = 14;
        let forecast = model.predict(horizon).unwrap();
        let explanation = model.explain(horizon).unwrap();
        let reconstructed = explanation.sum();
        for (i, (&pred, &recon)) in forecast
            .primary()
            .iter()
            .zip(reconstructed.iter())
            .enumerate()
        {
            assert!(
                (pred - recon).abs() < 1e-6,
                "MSTL mismatch at h={}: pred={}, recon={}",
                i,
                pred,
                recon
            );
        }
    }
    #[test]
    fn mstl_explanation_has_named_seasonal_components() {
        let ts = TimeSeries::univariate(
            make_timestamps(100),
            (0..100)
                .map(|i| 50.0 + 5.0 * (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
                .collect(),
        )
        .unwrap();
        let mut model = MSTLForecaster::new(vec![7]);
        model.fit(&ts).unwrap();
        let explanation = model.explain(7).unwrap();
        assert!(explanation.has_correct_lengths(7));
        assert!(!explanation.named_components.is_empty());
    }
}
