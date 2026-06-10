//! Residual-Ridge shim for `predict_with_exog` on models without native
//! exogenous-variable support.
//!
//! Used to give every auto-tuned forecaster the same exog-aware predict
//! surface as `AutoARIMA::predict_with_exog`. Downstream code can call
//! `model.predict_with_exog(h, &future_regs)` on any model uniformly.
//!
//! The recipe (per issue #106):
//!
//! 1. Base forecast: `model.predict(horizon)`.
//! 2. Fit Ridge on `residual_component()` vs the training-window
//!    regressor matrix.
//! 3. Project Ridge onto `future_regressors`.
//! 4. Add the projection to the base forecast.
//!
//! The shim is a free function (rather than a default trait impl) so the
//! per-model builder can override the Ridge λ independently and so the
//! shim's behaviour stays explicit in each model's `predict_with_exog`.

use std::collections::HashMap;

use crate::core::Forecast;
use crate::error::{ForecastError, Result};
use crate::models::Forecaster;

/// Default Ridge λ for the residual shim. Small enough to recover known
/// linear coefficients in the round-trip test, large enough to keep
/// near-singular `X'X` stable.
pub const DEFAULT_RIDGE_LAMBDA: f64 = 0.1;

/// Compute an exog-adjusted forecast using a residual-Ridge shim.
///
/// # Arguments
/// - `model`: a fitted forecaster exposing `predict`, `residual_component`,
///   and `training_regressors`.
/// - `horizon`: forecast horizon.
/// - `future_regressors`: name → length-`horizon` vector of future regressor
///   values. Empty map ⇒ this fast-paths to `model.predict(horizon)`.
/// - `ridge_lambda`: Ridge regularisation strength; use
///   [`DEFAULT_RIDGE_LAMBDA`] if you don't have a strong reason to
///   override.
///
/// # Errors
/// - Model has no training regressors retained.
/// - Future regressor map references a name not seen at fit time.
/// - Future regressor vectors have length ≠ `horizon`.
/// - All residuals are NaN (e.g. model wasn't fit, or wasn't fit on
///   non-degenerate data).
/// - Ridge solve fails on a near-singular system even with λ added.
pub fn residual_ridge_shim<F: Forecaster + ?Sized>(
    model: &F,
    horizon: usize,
    future_regressors: &HashMap<String, Vec<f64>>,
    ridge_lambda: f64,
) -> Result<Forecast> {
    // Fast path: no regressors requested → return base forecast unchanged.
    let base = model.predict(horizon)?;
    if future_regressors.is_empty() {
        return Ok(base);
    }

    // Order regressor names deterministically so the column ordering is
    // stable across fit/predict and across runs.
    let mut names: Vec<&String> = future_regressors.keys().collect();
    names.sort();

    // Length-validate the future regressors.
    for &name in &names {
        let v = &future_regressors[name];
        if v.len() != horizon {
            return Err(ForecastError::DimensionMismatch {
                expected: horizon,
                got: v.len(),
            });
        }
    }

    // Pull residuals and training regressors from the model.
    let residuals_vec = model.residual_component()?;
    let train_regs = model.training_regressors().ok_or_else(|| {
        ForecastError::InvalidParameter(format!(
            "{} does not retain training regressors — residual-Ridge shim unavailable",
            model.name()
        ))
    })?;

    // Validate every requested regressor was seen during training.
    for &name in &names {
        if !train_regs.contains_key(name) {
            return Err(ForecastError::InvalidParameter(format!(
                "residual_ridge_shim: future regressor '{}' was not present at fit time",
                name
            )));
        }
    }

    // Build the design matrix and target vector, skipping NaN residuals
    // (warmup rows). Each training regressor is right-aligned with the
    // residual vector: we take its last `residuals.len()` values.
    let residual_len = residuals_vec.len();
    let p = names.len();
    let mut x_rows: Vec<f64> = Vec::with_capacity(residual_len * p);
    let mut y_rows: Vec<f64> = Vec::with_capacity(residual_len);
    for (i, &r) in residuals_vec.iter().enumerate() {
        if !r.is_finite() {
            continue;
        }
        // Row of regressor values aligned with residual i.
        let mut row = Vec::with_capacity(p);
        let mut row_finite = true;
        for &name in &names {
            let col = &train_regs[name];
            if col.len() < residual_len {
                return Err(ForecastError::DimensionMismatch {
                    expected: residual_len,
                    got: col.len(),
                });
            }
            let offset = col.len() - residual_len + i;
            let v = col[offset];
            if !v.is_finite() {
                row_finite = false;
                break;
            }
            row.push(v);
        }
        if !row_finite {
            continue;
        }
        x_rows.extend(row);
        y_rows.push(r);
    }
    let n = y_rows.len();
    if n == 0 {
        return Err(ForecastError::InvalidParameter(
            "residual_ridge_shim: no finite residuals to fit Ridge on".into(),
        ));
    }
    if n < p {
        return Err(ForecastError::InsufficientData {
            needed: p,
            got: n,
            hint: Some(format!(
                "residual_ridge_shim: fewer finite residuals ({}) than regressors ({}); reduce regressor count or increase training window",
                n, p
            )),
        });
    }

    // Solve Ridge: (X'X + λI) β = X'y via Cholesky.
    let beta = solve_ridge(&x_rows, &y_rows, n, p, ridge_lambda)?;

    // Apply the adjustment to the base forecast.
    let mut adjustment = vec![0.0_f64; horizon];
    for (j, &name) in names.iter().enumerate() {
        let fut = &future_regressors[name];
        for (h, &v) in fut.iter().enumerate() {
            adjustment[h] += beta[j] * v;
        }
    }

    apply_adjustment(base, &adjustment)
}

/// Add a per-horizon adjustment to a forecast's primary series.
fn apply_adjustment(mut base: Forecast, adjustment: &[f64]) -> Result<Forecast> {
    let primary = base.primary_mut();
    if primary.len() != adjustment.len() {
        return Err(ForecastError::DimensionMismatch {
            expected: primary.len(),
            got: adjustment.len(),
        });
    }
    for (p, a) in primary.iter_mut().zip(adjustment) {
        *p += a;
    }
    Ok(base)
}

/// Solve `(X'X + λI) β = X'y` via in-place Cholesky decomposition.
fn solve_ridge(x: &[f64], y: &[f64], n: usize, p: usize, lambda: f64) -> Result<Vec<f64>> {
    // Build X'X (symmetric, p × p) with λ added to the diagonal.
    let mut xtx = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += x[k * p + i] * x[k * p + j];
            }
            xtx[i * p + j] = s;
        }
        xtx[i * p + i] += lambda;
    }

    // X'y (length p).
    let mut xty = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0_f64;
        for k in 0..n {
            s += x[k * p + j] * y[k];
        }
        xty[j] = s;
    }

    cholesky_solve(&xtx, &xty, p).ok_or_else(|| {
        ForecastError::SingularMatrix(format!(
            "residual_ridge_shim: X'X + {}·I is not positive-definite",
            lambda
        ))
    })
}

/// Solve `A x = b` where `A` is symmetric positive-definite via Cholesky.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Compute Cholesky factor `L`: A = L · L^T.
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                if l[j * n + j] == 0.0 {
                    return None;
                }
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    // Forward solve L y = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }
    // Backward solve L^T x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::TimeSeries;

    /// Mock decomposable forecaster for unit-testing the shim in isolation.
    /// Stores a few synthetic residuals + training regressors and returns
    /// a constant base forecast.
    #[derive(Debug)]
    struct MockDecomposable {
        residuals: Vec<f64>,
        train_regs: Option<HashMap<String, Vec<f64>>>,
        base_forecast: Forecast,
    }

    impl Forecaster for MockDecomposable {
        fn fit(&mut self, _series: &TimeSeries) -> Result<()> {
            Ok(())
        }
        fn predict(&self, _horizon: usize) -> Result<Forecast> {
            Ok(self.base_forecast.clone())
        }
        fn fitted_values(&self) -> Option<&[f64]> {
            None
        }
        fn residuals(&self) -> Option<&[f64]> {
            Some(&self.residuals)
        }
        fn training_regressors(&self) -> Option<&HashMap<String, Vec<f64>>> {
            self.train_regs.as_ref()
        }
        fn name(&self) -> &str {
            "MockDecomposable"
        }
    }

    #[test]
    fn shim_recovers_known_coefficient() {
        // residuals[i] = 5 * x[i] + small_noise → Ridge should recover β ≈ 5.
        let n = 60;
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin()).collect();
        let residuals: Vec<f64> = (0..n)
            .map(|i| 5.0 * x[i] + ((i % 7) as f64 - 3.0) * 0.001)
            .collect();
        let mut train_regs = HashMap::new();
        train_regs.insert("x".to_string(), x.clone());

        let horizon = 5;
        let model = MockDecomposable {
            residuals,
            train_regs: Some(train_regs),
            base_forecast: Forecast::from_values(vec![10.0; horizon]),
        };

        // Future x at the next 5 sin-curve positions.
        let future_x: Vec<f64> = (n..n + horizon).map(|i| ((i as f64) * 0.1).sin()).collect();
        let mut future_regs = HashMap::new();
        future_regs.insert("x".to_string(), future_x.clone());

        let adjusted =
            residual_ridge_shim(&model, horizon, &future_regs, DEFAULT_RIDGE_LAMBDA).unwrap();

        // adjusted = base + β·future_x, with β ≈ 5.
        for h in 0..horizon {
            let expected = 10.0 + 5.0 * future_x[h];
            let got = adjusted.primary()[h];
            assert!(
                (got - expected).abs() < 0.5,
                "h={}: expected ≈ {}, got {}",
                h,
                expected,
                got
            );
        }
    }

    #[test]
    fn shim_empty_future_regressors_is_base_forecast() {
        let n = 30;
        let residuals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let horizon = 3;
        let model = MockDecomposable {
            residuals,
            train_regs: Some(HashMap::new()),
            base_forecast: Forecast::from_values(vec![1.0, 2.0, 3.0]),
        };
        let adjusted =
            residual_ridge_shim(&model, horizon, &HashMap::new(), DEFAULT_RIDGE_LAMBDA).unwrap();
        assert_eq!(adjusted.primary(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn shim_rejects_horizon_mismatch_on_future_regs() {
        let n = 20;
        let residuals = vec![0.0; n];
        let mut train_regs = HashMap::new();
        train_regs.insert("x".to_string(), vec![1.0; n]);
        let horizon = 3;
        let model = MockDecomposable {
            residuals,
            train_regs: Some(train_regs),
            base_forecast: Forecast::from_values(vec![1.0; horizon]),
        };
        let mut future_regs = HashMap::new();
        future_regs.insert("x".to_string(), vec![1.0; 5]); // wrong length
        let err =
            residual_ridge_shim(&model, horizon, &future_regs, DEFAULT_RIDGE_LAMBDA).unwrap_err();
        assert!(matches!(err, ForecastError::DimensionMismatch { .. }));
    }

    #[test]
    fn shim_rejects_unknown_regressor_name() {
        let n = 20;
        let residuals = vec![0.0; n];
        let mut train_regs = HashMap::new();
        train_regs.insert("x".to_string(), vec![1.0; n]);
        let horizon = 3;
        let model = MockDecomposable {
            residuals,
            train_regs: Some(train_regs),
            base_forecast: Forecast::from_values(vec![1.0; horizon]),
        };
        let mut future_regs = HashMap::new();
        future_regs.insert("z".to_string(), vec![1.0; horizon]); // never trained on z
        let err =
            residual_ridge_shim(&model, horizon, &future_regs, DEFAULT_RIDGE_LAMBDA).unwrap_err();
        assert!(matches!(err, ForecastError::InvalidParameter(_)));
    }

    #[test]
    fn shim_errors_when_model_has_no_training_regressors() {
        let horizon = 3;
        let model = MockDecomposable {
            residuals: vec![0.0; 20],
            train_regs: None, // training_regressors() will return None
            base_forecast: Forecast::from_values(vec![1.0; horizon]),
        };
        let mut future_regs = HashMap::new();
        future_regs.insert("anything".to_string(), vec![1.0; horizon]);
        let err =
            residual_ridge_shim(&model, horizon, &future_regs, DEFAULT_RIDGE_LAMBDA).unwrap_err();
        assert!(matches!(err, ForecastError::InvalidParameter(_)));
    }
}
