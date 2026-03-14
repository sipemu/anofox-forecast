use std::fmt;

/// Metrics for a single forecast horizon step.
#[derive(Debug, Clone)]
pub struct HorizonStep {
    /// 1-indexed horizon step.
    pub horizon: usize,
    pub mae: f64,
    pub rmse: f64,
    /// None if zeros in actuals at this horizon.
    pub mape: Option<f64>,
    /// mean(forecast - actual).
    pub bias: f64,
    /// Number of CV folds contributing.
    pub n_samples: usize,
}

/// Per-horizon breakdown of forecast accuracy from cross-validation.
#[derive(Debug, Clone)]
pub struct HorizonAnalysis {
    pub steps: Vec<HorizonStep>,
    pub model_name: String,
}

impl HorizonAnalysis {
    /// Build a per-horizon analysis from cross-validation fold results.
    ///
    /// `fold_actuals` and `fold_forecasts` are parallel vectors of per-fold
    /// actual/forecast slices. Each inner slice has length equal to the
    /// forecast horizon. All inner slices must have the same length.
    ///
    /// Example: 5 CV folds with horizon=3 → `fold_actuals` has 5 elements,
    /// each of length 3.
    pub fn from_folds(
        model_name: impl Into<String>,
        fold_actuals: &[&[f64]],
        fold_forecasts: &[&[f64]],
    ) -> Self {
        let model_name = model_name.into();

        let horizon = match fold_actuals.first() {
            Some(first) => first.len(),
            None => {
                return Self {
                    steps: Vec::new(),
                    model_name,
                };
            }
        };

        let n_folds = fold_actuals.len();
        let mut steps = Vec::with_capacity(horizon);

        for h in 0..horizon {
            let mut sum_abs_err = 0.0;
            let mut sum_sq_err = 0.0;
            let mut sum_bias = 0.0;
            let mut sum_ape = 0.0;
            let mut has_zero_actual = false;

            for i in 0..n_folds {
                let actual = fold_actuals[i][h];
                let forecast = fold_forecasts[i][h];
                let err = forecast - actual;

                sum_abs_err += err.abs();
                sum_sq_err += err * err;
                sum_bias += err;

                if actual == 0.0 {
                    has_zero_actual = true;
                } else {
                    sum_ape += (err.abs()) / actual.abs();
                }
            }

            let n = n_folds as f64;
            let mae = sum_abs_err / n;
            let rmse = (sum_sq_err / n).sqrt();
            let bias = sum_bias / n;
            let mape = if has_zero_actual {
                None
            } else {
                Some(sum_ape / n * 100.0)
            };

            steps.push(HorizonStep {
                horizon: h + 1,
                mae,
                rmse,
                mape,
                bias,
                n_samples: n_folds,
            });
        }

        Self { steps, model_name }
    }

    /// Number of horizon steps.
    pub fn horizon_length(&self) -> usize {
        self.steps.len()
    }

    /// Step with highest RMSE (hardest to forecast).
    pub fn hardest_horizon(&self) -> Option<&HorizonStep> {
        self.steps.iter().max_by(|a, b| {
            a.rmse
                .partial_cmp(&b.rmse)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Step with lowest RMSE (easiest to forecast).
    pub fn easiest_horizon(&self) -> Option<&HorizonStep> {
        self.steps.iter().min_by(|a, b| {
            a.rmse
                .partial_cmp(&b.rmse)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// If >= 2 steps, `(last_rmse - first_rmse) / first_rmse`.
    /// Shows how fast error grows with horizon.
    /// Returns `None` if < 2 steps or `first_rmse == 0`.
    pub fn error_growth_rate(&self) -> Option<f64> {
        if self.steps.len() < 2 {
            return None;
        }
        let first_rmse = self.steps.first().unwrap().rmse;
        let last_rmse = self.steps.last().unwrap().rmse;
        if first_rmse == 0.0 {
            return None;
        }
        Some((last_rmse - first_rmse) / first_rmse)
    }

    /// Table of horizon steps.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Horizon Analysis for '{}'\n", self.model_name));
        out.push_str(&format!(
            "{:>7}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}\n",
            "Horizon", "MAE", "RMSE", "MAPE(%)", "Bias", "Samples"
        ));
        out.push_str(&format!("{}\n", "-".repeat(62)));
        for step in &self.steps {
            let mape_str = match step.mape {
                Some(v) => format!("{:.4}", v),
                None => "N/A".to_string(),
            };
            out.push_str(&format!(
                "{:>7}  {:>10.4}  {:>10.4}  {:>10}  {:>10.4}  {:>9}\n",
                step.horizon, step.mae, step.rmse, mape_str, step.bias, step.n_samples
            ));
        }
        if let Some(rate) = self.error_growth_rate() {
            out.push_str(&format!("Error growth rate: {:.2}%\n", rate * 100.0));
        }
        out
    }
}

impl fmt::Display for HorizonAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl fmt::Display for HorizonStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mape_str = match self.mape {
            Some(v) => format!("{:.4}%", v),
            None => "N/A".to_string(),
        };
        write!(
            f,
            "h={}: MAE={:.4}, RMSE={:.4}, MAPE={}, Bias={:.4}, n={}",
            self.horizon, self.mae, self.rmse, mape_str, self.bias, self.n_samples
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_folds() {
        let analysis = HorizonAnalysis::from_folds("test", &[], &[]);
        assert!(analysis.steps.is_empty());
        assert_eq!(analysis.horizon_length(), 0);
        assert!(analysis.hardest_horizon().is_none());
        assert!(analysis.easiest_horizon().is_none());
        assert!(analysis.error_growth_rate().is_none());
    }

    #[test]
    fn single_fold_single_horizon() {
        let actuals: &[f64] = &[10.0];
        let forecasts: &[f64] = &[12.0];
        let analysis = HorizonAnalysis::from_folds("single", &[actuals], &[forecasts]);

        assert_eq!(analysis.horizon_length(), 1);
        let step = &analysis.steps[0];
        assert_eq!(step.horizon, 1);
        assert!((step.mae - 2.0).abs() < 1e-10);
        assert!((step.rmse - 2.0).abs() < 1e-10);
        assert!((step.bias - 2.0).abs() < 1e-10);
        assert!(step.mape.is_some());
        assert!((step.mape.unwrap() - 20.0).abs() < 1e-10);
        assert_eq!(step.n_samples, 1);
    }

    #[test]
    fn multiple_folds_multiple_horizons() {
        // 3 folds, horizon=4
        let a1: &[f64] = &[10.0, 20.0, 30.0, 40.0];
        let f1: &[f64] = &[11.0, 22.0, 33.0, 44.0];

        let a2: &[f64] = &[10.0, 20.0, 30.0, 40.0];
        let f2: &[f64] = &[9.0, 18.0, 27.0, 36.0];

        let a3: &[f64] = &[10.0, 20.0, 30.0, 40.0];
        let f3: &[f64] = &[10.0, 20.0, 30.0, 40.0];

        let analysis = HorizonAnalysis::from_folds("multi", &[a1, a2, a3], &[f1, f2, f3]);

        assert_eq!(analysis.horizon_length(), 4);
        assert_eq!(analysis.steps[0].n_samples, 3);

        // h=1: errors are +1, -1, 0 → MAE = 2/3, bias = 0
        let s0 = &analysis.steps[0];
        assert!((s0.mae - 2.0 / 3.0).abs() < 1e-10);
        assert!((s0.bias - 0.0).abs() < 1e-10);
    }

    #[test]
    fn perfect_forecast() {
        let actuals: &[f64] = &[5.0, 10.0, 15.0];
        let forecasts: &[f64] = &[5.0, 10.0, 15.0];
        let analysis = HorizonAnalysis::from_folds("perfect", &[actuals], &[forecasts]);

        for step in &analysis.steps {
            assert!((step.mae - 0.0).abs() < 1e-10);
            assert!((step.rmse - 0.0).abs() < 1e-10);
            assert!((step.bias - 0.0).abs() < 1e-10);
            assert!(step.mape.is_some());
            assert!((step.mape.unwrap() - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn increasing_error() {
        // Simulate error growing with horizon: h1 err=1, h2 err=2, h3 err=3
        let actuals: &[f64] = &[10.0, 10.0, 10.0];
        let forecasts: &[f64] = &[11.0, 12.0, 13.0];
        let analysis = HorizonAnalysis::from_folds("growing", &[actuals], &[forecasts]);

        assert!(analysis.steps[0].rmse < analysis.steps[1].rmse);
        assert!(analysis.steps[1].rmse < analysis.steps[2].rmse);
    }

    #[test]
    fn hardest_and_easiest() {
        let a1: &[f64] = &[10.0, 10.0, 10.0];
        let f1: &[f64] = &[11.0, 15.0, 10.5]; // errors: 1, 5, 0.5

        let analysis = HorizonAnalysis::from_folds("extremes", &[a1], &[f1]);

        let hardest = analysis.hardest_horizon().unwrap();
        assert_eq!(hardest.horizon, 2); // error of 5

        let easiest = analysis.easiest_horizon().unwrap();
        assert_eq!(easiest.horizon, 3); // error of 0.5
    }

    #[test]
    fn error_growth_rate_positive() {
        let actuals: &[f64] = &[10.0, 10.0];
        let forecasts: &[f64] = &[11.0, 13.0]; // RMSE: 1.0, 3.0

        let analysis = HorizonAnalysis::from_folds("growth", &[actuals], &[forecasts]);
        let rate = analysis.error_growth_rate().unwrap();
        // (3.0 - 1.0) / 1.0 = 2.0
        assert!((rate - 2.0).abs() < 1e-10);
    }

    #[test]
    fn error_growth_rate_zero() {
        // Constant error across horizons
        let actuals: &[f64] = &[10.0, 20.0, 30.0];
        let forecasts: &[f64] = &[12.0, 22.0, 32.0]; // error=2 at every step

        let analysis = HorizonAnalysis::from_folds("constant", &[actuals], &[forecasts]);
        let rate = analysis.error_growth_rate().unwrap();
        assert!((rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn mape_with_zero_actual() {
        let actuals: &[f64] = &[0.0, 10.0];
        let forecasts: &[f64] = &[5.0, 12.0];
        let analysis = HorizonAnalysis::from_folds("zero_act", &[actuals], &[forecasts]);

        // h=1 has zero actual → MAPE should be None
        assert!(analysis.steps[0].mape.is_none());
        // h=2 has non-zero actual → MAPE should be Some
        assert!(analysis.steps[1].mape.is_some());
    }

    #[test]
    fn bias_overforecast() {
        // All forecasts above actuals → positive bias
        let a1: &[f64] = &[10.0];
        let f1: &[f64] = &[15.0];
        let a2: &[f64] = &[20.0];
        let f2: &[f64] = &[25.0];

        let analysis = HorizonAnalysis::from_folds("over", &[a1, a2], &[f1, f2]);
        assert!(analysis.steps[0].bias > 0.0);
        assert!((analysis.steps[0].bias - 5.0).abs() < 1e-10);
    }

    #[test]
    fn bias_underforecast() {
        // All forecasts below actuals → negative bias
        let a1: &[f64] = &[10.0];
        let f1: &[f64] = &[7.0];
        let a2: &[f64] = &[20.0];
        let f2: &[f64] = &[17.0];

        let analysis = HorizonAnalysis::from_folds("under", &[a1, a2], &[f1, f2]);
        assert!(analysis.steps[0].bias < 0.0);
        assert!((analysis.steps[0].bias - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn display_contains_table() {
        let actuals: &[f64] = &[10.0, 20.0];
        let forecasts: &[f64] = &[12.0, 22.0];
        let analysis = HorizonAnalysis::from_folds("display_test", &[actuals], &[forecasts]);

        let text = analysis.to_string();
        assert!(text.contains("display_test"));
        assert!(text.contains("Horizon"));
        assert!(text.contains("1"));
        assert!(text.contains("2"));
        assert!(text.contains("MAE"));
        assert!(text.contains("RMSE"));

        // Also test HorizonStep Display
        let step_text = format!("{}", analysis.steps[0]);
        assert!(step_text.contains("h=1"));
        assert!(step_text.contains("MAE="));
        assert!(step_text.contains("RMSE="));
    }
}
