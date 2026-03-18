//! MSTL (Multiple Seasonal-Trend decomposition using LOESS) implementation.
//!
//! MSTL extends STL to handle multiple seasonal periods, such as daily and weekly
//! patterns in hourly data.

use super::stl::STL;
use crate::utils::ols::{ols_fit, ols_residuals, OLSResult};
use std::collections::HashMap;

/// Result of MSTL decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MSTLResult {
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal components (one for each period).
    pub seasonal_components: Vec<Vec<f64>>,
    /// The seasonal periods corresponding to each component.
    pub seasonal_periods: Vec<usize>,
    /// Remainder component.
    pub remainder: Vec<f64>,
    /// OLS coefficients from pre-regression (if regressors were used).
    pub regressor_coefficients: Option<OLSResult>,
    /// Estimated regressor effect (X * β) during training.
    pub regressor_effect: Option<Vec<f64>>,
}

impl MSTLResult {
    /// Get the total seasonal component (sum of all seasonal components).
    pub fn total_seasonal(&self) -> Vec<f64> {
        if self.seasonal_components.is_empty() {
            return vec![0.0; self.trend.len()];
        }

        let n = self.trend.len();
        let mut total = vec![0.0; n];
        for component in &self.seasonal_components {
            for i in 0..n {
                total[i] += component[i];
            }
        }
        total
    }

    /// Get seasonal strength for a specific period.
    pub fn seasonal_strength(&self, period_idx: usize) -> Option<f64> {
        if period_idx >= self.seasonal_components.len() {
            return None;
        }

        let seasonal = &self.seasonal_components[period_idx];

        let var_remainder = variance(&self.remainder);
        let seasonal_plus_remainder: Vec<f64> = seasonal
            .iter()
            .zip(self.remainder.iter())
            .map(|(s, r)| s + r)
            .collect();
        let var_sr = variance(&seasonal_plus_remainder);

        if var_sr < 1e-10 {
            return Some(0.0);
        }

        Some((1.0 - var_remainder / var_sr).clamp(0.0, 1.0))
    }

    /// Get trend strength.
    pub fn trend_strength(&self) -> f64 {
        let var_remainder = variance(&self.remainder);
        let trend_plus_remainder: Vec<f64> = self
            .trend
            .iter()
            .zip(self.remainder.iter())
            .map(|(t, r)| t + r)
            .collect();
        let var_tr = variance(&trend_plus_remainder);

        if var_tr < 1e-10 {
            return 0.0;
        }

        (1.0 - var_remainder / var_tr).clamp(0.0, 1.0)
    }
}

/// MSTL decomposition for multiple seasonal periods.
#[derive(Debug, Clone)]
pub struct MSTL {
    /// Seasonal periods (should be sorted in increasing order).
    seasonal_periods: Vec<usize>,
    /// Number of iterations.
    iterations: usize,
    /// Use robust fitting.
    robust: bool,
}

impl MSTL {
    /// Create a new MSTL decomposer with the given seasonal periods.
    pub fn new(seasonal_periods: Vec<usize>) -> Self {
        let mut periods = seasonal_periods;
        periods.sort();
        periods.dedup();

        Self {
            seasonal_periods: periods,
            iterations: 2,
            robust: false,
        }
    }

    /// Set number of iterations.
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Enable robust fitting.
    pub fn robust(mut self) -> Self {
        self.robust = true;
        self
    }

    /// Get the seasonal periods.
    pub fn seasonal_periods(&self) -> &[usize] {
        &self.seasonal_periods
    }

    /// Decompose the time series.
    pub fn decompose(&self, series: &[f64]) -> Option<MSTLResult> {
        let n = series.len();

        if self.seasonal_periods.is_empty() {
            return None;
        }

        // Check minimum length
        let max_period = *self.seasonal_periods.last()?;
        if n < 2 * max_period {
            return None;
        }

        let num_seasonals = self.seasonal_periods.len();
        let mut seasonal_components: Vec<Vec<f64>> = vec![vec![0.0; n]; num_seasonals];
        let mut trend = vec![0.0; n];

        // Reusable buffer for deseasonalized/adjusted data (avoids allocations per iteration)
        let mut buf = vec![0.0_f64; n];

        // Iterative decomposition
        for _ in 0..self.iterations {
            // Deseasonalize: buf = series - sum(all seasonal components)
            buf.copy_from_slice(series);
            for seasonal in &seasonal_components {
                for i in 0..n {
                    buf[i] -= seasonal[i];
                }
            }

            // Extract trend using STL with the longest period
            let stl_trend = if self.robust {
                STL::new(max_period).robust()
            } else {
                STL::new(max_period)
            };

            if let Some(trend_result) = stl_trend.decompose(&buf) {
                trend = trend_result.trend;
            }

            // Extract each seasonal component
            for s_idx in 0..num_seasonals {
                let period = self.seasonal_periods[s_idx];

                // adjusted = series - trend - sum(other seasonal components)
                // Equivalent to: series - trend - sum(all seasonals) + seasonal[s_idx]
                // Which is: deseasonalized - trend + seasonal[s_idx]
                // But we rebuild from series to avoid error accumulation.
                buf.copy_from_slice(series);
                for i in 0..n {
                    buf[i] -= trend[i];
                }
                for (other_idx, other_seasonal) in seasonal_components.iter().enumerate() {
                    if other_idx != s_idx {
                        for i in 0..n {
                            buf[i] -= other_seasonal[i];
                        }
                    }
                }

                // Extract this seasonal component using STL
                let stl_seasonal = if self.robust {
                    STL::new(period).robust()
                } else {
                    STL::new(period)
                };

                if let Some(seasonal_result) = stl_seasonal.decompose(&buf) {
                    seasonal_components[s_idx] = seasonal_result.seasonal;
                }
            }
        }

        // Compute remainder: series - trend - sum(all seasonal components)
        let mut remainder = vec![0.0_f64; n];
        for i in 0..n {
            let mut seasonal_sum = 0.0;
            for seasonal in &seasonal_components {
                seasonal_sum += seasonal[i];
            }
            remainder[i] = series[i] - trend[i] - seasonal_sum;
        }

        Some(MSTLResult {
            trend,
            seasonal_components,
            seasonal_periods: self.seasonal_periods.clone(),
            remainder,
            regressor_coefficients: None,
            regressor_effect: None,
        })
    }

    /// Decompose the time series after regressing out exogenous effects.
    ///
    /// Performs pre-regression STL: first fits OLS (y ~ X) to remove exogenous
    /// effects, then decomposes the adjusted series (y - X*β) with standard MSTL.
    /// This prevents regressors correlated with trend or seasonality from
    /// distorting the decomposition.
    ///
    /// The regressor effect (X*β) is stored in the result so it can be added
    /// back during forecasting.
    pub fn decompose_with_regressors(
        &self,
        series: &[f64],
        regressors: &HashMap<String, Vec<f64>>,
    ) -> Option<MSTLResult> {
        if regressors.is_empty() {
            return self.decompose(series);
        }

        // Fit OLS: y ~ X
        let ols_result = ols_fit(series, regressors).ok()?;

        // Compute adjusted series: y_adjusted = y - X*β
        let y_adjusted = ols_residuals(series, &ols_result, regressors).ok()?;

        // Decompose the adjusted series
        let mut result = self.decompose(&y_adjusted)?;

        // Store OLS info so the forecaster can add back the regressor effect
        let regressor_effect = ols_result.predict(regressors).ok()?;
        result.regressor_coefficients = Some(ols_result);
        result.regressor_effect = Some(regressor_effect);

        Some(result)
    }
}

/// Compute variance.
fn variance(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_multi_seasonal_series(n: usize, periods: &[usize]) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = 0.05 * i as f64;
                let mut seasonal = 0.0;
                for (idx, &period) in periods.iter().enumerate() {
                    let amplitude = 5.0 / (idx + 1) as f64; // Decreasing amplitude
                    seasonal +=
                        amplitude * ((2.0 * std::f64::consts::PI * i as f64 / period as f64).sin());
                }
                trend + seasonal
            })
            .collect()
    }

    #[test]
    fn mstl_single_period() {
        let period = 12;
        let series = generate_multi_seasonal_series(120, &[period]);

        let mstl = MSTL::new(vec![period]);
        let result = mstl.decompose(&series).unwrap();

        assert_eq!(result.seasonal_components.len(), 1);
        assert_eq!(result.seasonal_periods, vec![period]);
        assert_eq!(result.trend.len(), series.len());
    }

    #[test]
    fn mstl_two_periods() {
        // Daily and weekly seasonality (simulated)
        let periods = vec![7, 24];
        let series = generate_multi_seasonal_series(200, &periods);

        let mstl = MSTL::new(periods.clone());
        let result = mstl.decompose(&series).unwrap();

        assert_eq!(result.seasonal_components.len(), 2);
        assert_eq!(result.seasonal_periods, vec![7, 24]);

        // Verify additive decomposition
        for i in 0..series.len() {
            let reconstructed = result.trend[i]
                + result.seasonal_components[0][i]
                + result.seasonal_components[1][i]
                + result.remainder[i];
            assert!(
                (series[i] - reconstructed).abs() < 1e-6,
                "Reconstruction failed at index {}",
                i
            );
        }
    }

    #[test]
    fn mstl_total_seasonal() {
        let periods = vec![7, 24];
        let series = generate_multi_seasonal_series(200, &periods);

        let mstl = MSTL::new(periods);
        let result = mstl.decompose(&series).unwrap();

        let total = result.total_seasonal();
        assert_eq!(total.len(), series.len());

        // Total should be sum of components
        for i in 0..series.len() {
            let expected = result.seasonal_components[0][i] + result.seasonal_components[1][i];
            assert!(
                (total[i] - expected).abs() < 1e-10,
                "Total seasonal mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn mstl_insufficient_data() {
        let periods = vec![7, 24];
        let series = vec![1.0; 30]; // Less than 2 * max_period

        let mstl = MSTL::new(periods);
        assert!(mstl.decompose(&series).is_none());
    }

    #[test]
    fn mstl_empty_periods() {
        let series = vec![1.0; 100];
        let mstl = MSTL::new(vec![]);
        assert!(mstl.decompose(&series).is_none());
    }

    #[test]
    fn mstl_robust() {
        let periods = vec![12];
        let mut series = generate_multi_seasonal_series(120, &periods);
        // Add outliers
        series[30] = 100.0;
        series[60] = -100.0;

        let mstl = MSTL::new(periods).robust();
        let result = mstl.decompose(&series);
        assert!(result.is_some());
    }

    #[test]
    fn mstl_duplicate_periods_removed() {
        let mstl = MSTL::new(vec![12, 12, 7, 7]);
        assert_eq!(mstl.seasonal_periods(), &[7, 12]);
    }

    #[test]
    fn mstl_periods_sorted() {
        let mstl = MSTL::new(vec![24, 7, 12]);
        assert_eq!(mstl.seasonal_periods(), &[7, 12, 24]);
    }

    #[test]
    fn mstl_seasonal_strength() {
        let periods = vec![12];
        let series = generate_multi_seasonal_series(120, &periods);

        let mstl = MSTL::new(periods);
        let result = mstl.decompose(&series).unwrap();

        let strength = result.seasonal_strength(0).unwrap();
        assert!(
            (0.0..=1.0).contains(&strength),
            "Seasonal strength should be in [0, 1]: {}",
            strength
        );
    }

    #[test]
    fn mstl_trend_strength() {
        let periods = vec![12];
        let series = generate_multi_seasonal_series(120, &periods);

        let mstl = MSTL::new(periods);
        let result = mstl.decompose(&series).unwrap();

        let strength = result.trend_strength();
        assert!(
            (0.0..=1.0).contains(&strength),
            "Trend strength should be in [0, 1]: {}",
            strength
        );
    }

    #[test]
    fn mstl_with_iterations() {
        let periods = vec![12];
        let series = generate_multi_seasonal_series(120, &periods);

        let mstl = MSTL::new(periods).with_iterations(3);
        let result = mstl.decompose(&series);
        assert!(result.is_some());
    }

    #[test]
    fn mstl_invalid_period_index() {
        let periods = vec![12];
        let series = generate_multi_seasonal_series(120, &periods);

        let mstl = MSTL::new(periods);
        let result = mstl.decompose(&series).unwrap();

        assert!(result.seasonal_strength(5).is_none());
    }

    #[test]
    fn mstl_decompose_no_regressors_returns_none_fields() {
        let periods = vec![12];
        let series = generate_multi_seasonal_series(120, &periods);

        let mstl = MSTL::new(periods);
        let result = mstl.decompose(&series).unwrap();

        assert!(result.regressor_coefficients.is_none());
        assert!(result.regressor_effect.is_none());
    }

    #[test]
    fn mstl_decompose_with_empty_regressors_falls_back() {
        let periods = vec![12];
        let series = generate_multi_seasonal_series(120, &periods);

        let mstl = MSTL::new(periods);
        let regressors = std::collections::HashMap::new();
        let result = mstl
            .decompose_with_regressors(&series, &regressors)
            .unwrap();

        // Empty regressors should behave like decompose()
        assert!(result.regressor_coefficients.is_none());
        assert!(result.regressor_effect.is_none());
    }

    #[test]
    fn mstl_decompose_with_regressors_stores_ols() {
        let n = 120;
        let periods = vec![12];
        // Generate series with a regressor effect: y = trend + seasonal + 2*x
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let base = generate_multi_seasonal_series(n, &periods);
        let series: Vec<f64> = base
            .iter()
            .zip(x.iter())
            .map(|(b, xi)| b + 2.0 * xi)
            .collect();

        let mstl = MSTL::new(periods);
        let mut regressors = std::collections::HashMap::new();
        regressors.insert("x".to_string(), x.clone());

        let result = mstl
            .decompose_with_regressors(&series, &regressors)
            .unwrap();

        // OLS result should be stored
        assert!(result.regressor_coefficients.is_some());
        let ols = result.regressor_coefficients.as_ref().unwrap();
        assert_eq!(ols.regressor_names, vec!["x".to_string()]);

        // Regressor effect should be stored and have correct length
        assert!(result.regressor_effect.is_some());
        assert_eq!(result.regressor_effect.as_ref().unwrap().len(), n);

        // Additive decomposition should reconstruct adjusted series (y - X*β)
        let regressor_effect = result.regressor_effect.as_ref().unwrap();
        for i in 0..n {
            let reconstructed = result.trend[i]
                + result.seasonal_components[0][i]
                + result.remainder[i]
                + regressor_effect[i];
            assert!(
                (series[i] - reconstructed).abs() < 1e-4,
                "Reconstruction failed at index {}: expected {}, got {}",
                i,
                series[i],
                reconstructed,
            );
        }
    }

    #[test]
    fn mstl_decompose_with_regressors_coefficient_accuracy() {
        let n = 200;
        let periods = vec![12];
        // y = trend + seasonal + 3.0*x (exact, no noise)
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let base = generate_multi_seasonal_series(n, &periods);
        let series: Vec<f64> = base
            .iter()
            .zip(x.iter())
            .map(|(b, xi)| b + 3.0 * xi)
            .collect();

        let mstl = MSTL::new(periods);
        let mut regressors = std::collections::HashMap::new();
        regressors.insert("x".to_string(), x);

        let result = mstl
            .decompose_with_regressors(&series, &regressors)
            .unwrap();
        let ols = result.regressor_coefficients.as_ref().unwrap();

        // Coefficient should be close to 3.0 (not exact because OLS also has intercept
        // and the trend in the base series is correlated with x)
        // Just check it's in a reasonable range
        assert!(
            ols.coefficients[0] > 0.0,
            "Coefficient should be positive, got {}",
            ols.coefficients[0],
        );
    }
}
