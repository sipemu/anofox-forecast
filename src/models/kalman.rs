//! Kalman filter framework for linear Gaussian state-space models.
//!
//! State equation:   x(t) = F * x(t-1) + w(t),  w ~ N(0, Q)
//! Observation equation: y(t) = H * x(t) + v(t), v ~ N(0, R)
//!
//! Supports filtering (forward pass), Rauch-Tung-Striebel smoothing,
//! multi-step prediction, and log-likelihood computation.

use crate::error::{ForecastError, Result};

// ---------------------------------------------------------------------------
// Small dense matrix helpers (column-major Vec<Vec<f64>> where outer = rows)
// ---------------------------------------------------------------------------

/// Create an n x n identity matrix.
fn mat_eye(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Create an n x m zero matrix.
fn mat_zeros(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; cols]; rows]
}

/// Number of rows.
fn mat_rows(m: &[Vec<f64>]) -> usize {
    m.len()
}

/// Number of columns (assumes non-empty and rectangular).
fn mat_cols(m: &[Vec<f64>]) -> usize {
    if m.is_empty() {
        0
    } else {
        m[0].len()
    }
}

/// Matrix addition: C = A + B.
fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = mat_rows(a);
    let c = mat_cols(a);
    let mut out = mat_zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

/// Matrix subtraction: C = A - B.
fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = mat_rows(a);
    let c = mat_cols(a);
    let mut out = mat_zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

/// Matrix multiply: C = A * B.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let ar = mat_rows(a);
    let ac = mat_cols(a);
    let bc = mat_cols(b);
    let mut out = mat_zeros(ar, bc);
    for i in 0..ar {
        for k in 0..ac {
            let a_ik = a[i][k];
            for j in 0..bc {
                out[i][j] += a_ik * b[k][j];
            }
        }
    }
    out
}

/// Matrix-vector multiply: y = A * x.
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let r = mat_rows(a);
    let c = mat_cols(a);
    let mut out = vec![0.0; r];
    for i in 0..r {
        let mut s = 0.0;
        for j in 0..c {
            s += a[i][j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Transpose.
fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = mat_rows(a);
    let c = mat_cols(a);
    let mut out = mat_zeros(c, r);
    for i in 0..r {
        for j in 0..c {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns lower-triangular L such that A = L * L^T.
fn cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = mat_rows(a);
    let mut l = mat_zeros(n, n);
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j][k] * l[j][k];
        }
        let diag = a[j][j] - sum;
        if diag < 0.0 {
            return Err(ForecastError::SingularMatrix(
                "matrix is not positive-definite in Cholesky decomposition".into(),
            ));
        }
        l[j][j] = diag.sqrt();
        if l[j][j] == 0.0 {
            return Err(ForecastError::SingularMatrix(
                "zero diagonal in Cholesky decomposition".into(),
            ));
        }
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            l[i][j] = (a[i][j] - s) / l[j][j];
        }
    }
    Ok(l)
}

/// Solve L * x = b via forward substitution (L lower triangular).
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i][j] * x[j];
        }
        x[i] = (b[i] - s) / l[i][i];
    }
    x
}

/// Solve L^T * x = b via back substitution (L lower triangular).
fn back_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += l[j][i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (b[i] - s) / l[i][i];
    }
    x
}

/// Inverse of a symmetric positive-definite matrix via Cholesky.
fn mat_inv_spd(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = mat_rows(a);
    let l = cholesky(a)?;
    let mut inv = mat_zeros(n, n);
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        let y = forward_solve(&l, &e);
        let x = back_solve(&l, &y);
        for row in 0..n {
            inv[row][col] = x[row];
        }
    }
    Ok(inv)
}

/// Log-determinant of a symmetric positive-definite matrix via Cholesky.
fn mat_log_det_spd(a: &[Vec<f64>]) -> Result<f64> {
    let l = cholesky(a)?;
    let mut ld = 0.0;
    for i in 0..mat_rows(&l) {
        ld += l[i][i].ln();
    }
    Ok(2.0 * ld)
}

/// Quadratic form x^T * A^{-1} * x for SPD A.
fn quad_form_inv(a: &[Vec<f64>], x: &[f64]) -> Result<f64> {
    let l = cholesky(a)?;
    let y = forward_solve(&l, x);
    Ok(y.iter().map(|v| v * v).sum())
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Specification of a linear Gaussian state-space model.
///
/// State equation:   x(t) = F * x(t-1) + w(t),  w ~ N(0, Q)
/// Observation equation: y(t) = H * x(t) + v(t), v ~ N(0, R)
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// F: state transition matrix (n_state x n_state).
    pub transition: Vec<Vec<f64>>,
    /// H: observation matrix (n_obs x n_state).
    pub observation: Vec<Vec<f64>>,
    /// Q: process noise covariance (n_state x n_state).
    pub process_noise: Vec<Vec<f64>>,
    /// R: observation noise covariance (n_obs x n_obs).
    pub observation_noise: Vec<Vec<f64>>,
}

impl StateSpaceModel {
    /// Number of state dimensions.
    pub fn n_state(&self) -> usize {
        mat_rows(&self.transition)
    }

    /// Number of observation dimensions.
    pub fn n_obs(&self) -> usize {
        mat_rows(&self.observation)
    }

    /// Validate matrix dimensions for consistency.
    pub fn validate(&self) -> Result<()> {
        let ns = self.n_state();
        let no = self.n_obs();

        if ns == 0 {
            return Err(ForecastError::InvalidParameter(
                "state dimension must be at least 1".into(),
            ));
        }
        if no == 0 {
            return Err(ForecastError::InvalidParameter(
                "observation dimension must be at least 1".into(),
            ));
        }

        // F: ns x ns
        if mat_rows(&self.transition) != ns || mat_cols(&self.transition) != ns {
            return Err(ForecastError::InvalidParameter(format!(
                "transition matrix must be {}x{}, got {}x{}",
                ns,
                ns,
                mat_rows(&self.transition),
                mat_cols(&self.transition)
            )));
        }
        // H: no x ns
        if mat_rows(&self.observation) != no || mat_cols(&self.observation) != ns {
            return Err(ForecastError::InvalidParameter(format!(
                "observation matrix must be {}x{}, got {}x{}",
                no,
                ns,
                mat_rows(&self.observation),
                mat_cols(&self.observation)
            )));
        }
        // Q: ns x ns
        if mat_rows(&self.process_noise) != ns || mat_cols(&self.process_noise) != ns {
            return Err(ForecastError::InvalidParameter(format!(
                "process noise matrix must be {}x{}, got {}x{}",
                ns,
                ns,
                mat_rows(&self.process_noise),
                mat_cols(&self.process_noise)
            )));
        }
        // R: no x no
        if mat_rows(&self.observation_noise) != no || mat_cols(&self.observation_noise) != no {
            return Err(ForecastError::InvalidParameter(format!(
                "observation noise matrix must be {}x{}, got {}x{}",
                no,
                no,
                mat_rows(&self.observation_noise),
                mat_cols(&self.observation_noise)
            )));
        }

        Ok(())
    }

    /// Create a local level (random walk plus noise) model.
    ///
    /// State: level(t) = level(t-1) + w(t), w ~ N(0, level_var)
    /// Observation: y(t) = level(t) + v(t), v ~ N(0, obs_var)
    pub fn local_level(obs_var: f64, level_var: f64) -> Self {
        Self {
            transition: vec![vec![1.0]],
            observation: vec![vec![1.0]],
            process_noise: vec![vec![level_var]],
            observation_noise: vec![vec![obs_var]],
        }
    }

    /// Create a local linear trend model.
    ///
    /// State: [level(t), trend(t)]
    ///   level(t) = level(t-1) + trend(t-1) + w1(t)
    ///   trend(t) = trend(t-1) + w2(t)
    /// Observation: y(t) = level(t) + v(t)
    pub fn local_linear_trend(obs_var: f64, level_var: f64, trend_var: f64) -> Self {
        Self {
            transition: vec![vec![1.0, 1.0], vec![0.0, 1.0]],
            observation: vec![vec![1.0, 0.0]],
            process_noise: vec![vec![level_var, 0.0], vec![0.0, trend_var]],
            observation_noise: vec![vec![obs_var]],
        }
    }
}

/// Result of a single Kalman filter time step.
#[derive(Debug, Clone)]
pub struct KalmanState {
    /// Filtered (or smoothed) state estimate.
    pub state: Vec<f64>,
    /// Error covariance of the state estimate.
    pub covariance: Vec<Vec<f64>>,
    /// Predicted observation (H * x_predicted).
    pub predicted_obs: Vec<f64>,
    /// Innovation (y - predicted_obs).
    pub innovation: Vec<f64>,
    /// Contribution to log-likelihood from this step.
    pub log_likelihood: f64,
}

/// Kalman filter with filtering, smoothing, and prediction.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    model: StateSpaceModel,
    state: Option<Vec<f64>>,
    covariance: Option<Vec<Vec<f64>>>,
}

impl KalmanFilter {
    /// Create a new Kalman filter from a state-space model specification.
    ///
    /// The model dimensions are validated on construction.
    pub fn new(model: StateSpaceModel) -> Result<Self> {
        model.validate()?;
        Ok(Self {
            model,
            state: None,
            covariance: None,
        })
    }

    /// Set initial state estimate and error covariance.
    pub fn set_initial_state(&mut self, state: Vec<f64>, covariance: Vec<Vec<f64>>) {
        self.state = Some(state);
        self.covariance = Some(covariance);
    }

    /// Run the forward Kalman filter on a sequence of observations.
    ///
    /// Each element of `observations` is a vector of length `n_obs`.
    /// Returns one `KalmanState` per time step.
    pub fn filter(&mut self, observations: &[Vec<f64>]) -> Result<Vec<KalmanState>> {
        if observations.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        let ns = self.model.n_state();
        let no = self.model.n_obs();

        // Validate observation dimensions.
        for (t, obs) in observations.iter().enumerate() {
            if obs.len() != no {
                return Err(ForecastError::DimensionMismatch {
                    expected: no,
                    got: obs.len(),
                });
            }
            // Check for NaN/Inf.
            for v in obs {
                if !v.is_finite() {
                    return Err(ForecastError::InvalidParameter(format!(
                        "non-finite observation at time step {}",
                        t
                    )));
                }
            }
        }

        let f = &self.model.transition;
        let h = &self.model.observation;
        let q = &self.model.process_noise;
        let r = &self.model.observation_noise;

        // Initial state: use diffuse prior if not set.
        let mut x = self.state.clone().unwrap_or_else(|| vec![0.0; ns]);
        let mut p = self.covariance.clone().unwrap_or_else(|| {
            let mut m = mat_eye(ns);
            for i in 0..ns {
                m[i][i] = 1e6;
            }
            m
        });

        let mut results = Vec::with_capacity(observations.len());

        let ht = mat_transpose(h);
        let ft = mat_transpose(f);

        for obs in observations {
            // --- Predict ---
            let x_pred = mat_vec(f, &x);
            let p_pred = mat_add(&mat_mul(&mat_mul(f, &p), &ft), q);

            // --- Update ---
            // Innovation: y - H * x_pred
            let y_pred = mat_vec(h, &x_pred);
            let innovation: Vec<f64> = obs.iter().zip(y_pred.iter()).map(|(a, b)| a - b).collect();

            // Innovation covariance: S = H * P_pred * H^T + R
            let s = mat_add(&mat_mul(&mat_mul(h, &p_pred), &ht), r);

            // Check if S is (near) singular — degenerate case with zero noise.
            let s_max_diag = (0..no).map(|i| s[i][i].abs()).fold(0.0_f64, f64::max);
            let degenerate = s_max_diag < 1e-30;

            let ll;
            if degenerate {
                // S is effectively zero: perfect prediction, no update needed.
                x = x_pred;
                p = p_pred;
                ll = 0.0;
            } else {
                // Kalman gain: K = P_pred * H^T * S^{-1}
                let s_inv = mat_inv_spd(&s)?;
                let k = mat_mul(&mat_mul(&p_pred, &ht), &s_inv);

                // Updated state: x = x_pred + K * innovation
                let k_inn = mat_vec(&k, &innovation);
                x = x_pred
                    .iter()
                    .zip(k_inn.iter())
                    .map(|(a, b)| a + b)
                    .collect();

                // Updated covariance: P = (I - K * H) * P_pred
                let kh = mat_mul(&k, h);
                let i_kh = mat_sub(&mat_eye(ns), &kh);
                p = mat_mul(&i_kh, &p_pred);

                // Symmetrize P for numerical stability.
                let pt = mat_transpose(&p);
                for i in 0..ns {
                    for j in 0..ns {
                        p[i][j] = 0.5 * (p[i][j] + pt[i][j]);
                    }
                }

                // Log-likelihood contribution:
                // -0.5 * (n_obs * ln(2*pi) + ln|S| + innovation^T * S^{-1} * innovation)
                let log_det = mat_log_det_spd(&s)?;
                let quad = quad_form_inv(&s, &innovation)?;
                ll = -0.5 * (no as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad);
            }

            results.push(KalmanState {
                state: x.clone(),
                covariance: p.clone(),
                predicted_obs: y_pred,
                innovation,
                log_likelihood: ll,
            });
        }

        // Store final state for subsequent predictions.
        self.state = Some(x);
        self.covariance = Some(p);

        Ok(results)
    }

    /// Rauch-Tung-Striebel smoother (backward pass).
    ///
    /// Takes the output of `filter()` and returns smoothed state estimates.
    pub fn smooth(&self, filtered: &[KalmanState]) -> Result<Vec<KalmanState>> {
        if filtered.is_empty() {
            return Err(ForecastError::EmptyData);
        }

        let n = filtered.len();
        let f = &self.model.transition;
        let q = &self.model.process_noise;
        let ft = mat_transpose(f);

        let mut smoothed = filtered.to_vec();

        // Backward pass: t = n-2 .. 0
        for t in (0..n.saturating_sub(1)).rev() {
            // Predicted state and covariance at t+1 from filtered state at t.
            let x_pred = mat_vec(f, &filtered[t].state);
            let p_pred = mat_add(&mat_mul(&mat_mul(f, &filtered[t].covariance), &ft), q);

            // Smoother gain: G = P_t * F^T * P_pred^{-1}
            let p_pred_inv = mat_inv_spd(&p_pred)?;
            let g = mat_mul(&mat_mul(&filtered[t].covariance, &ft), &p_pred_inv);

            // Smoothed state: x_s(t) = x_f(t) + G * (x_s(t+1) - x_pred(t+1))
            let diff: Vec<f64> = smoothed[t + 1]
                .state
                .iter()
                .zip(x_pred.iter())
                .map(|(a, b)| a - b)
                .collect();
            let correction = mat_vec(&g, &diff);
            smoothed[t].state = filtered[t]
                .state
                .iter()
                .zip(correction.iter())
                .map(|(a, b)| a + b)
                .collect();

            // Smoothed covariance:
            // P_s(t) = P_f(t) + G * (P_s(t+1) - P_pred(t+1)) * G^T
            let gt = mat_transpose(&g);
            let p_diff = mat_sub(&smoothed[t + 1].covariance, &p_pred);
            smoothed[t].covariance = mat_add(
                &filtered[t].covariance,
                &mat_mul(&mat_mul(&g, &p_diff), &gt),
            );

            // Symmetrize for numerical stability.
            let ns = mat_rows(&smoothed[t].covariance);
            let sym = mat_transpose(&smoothed[t].covariance);
            for i in 0..ns {
                for j in 0..ns {
                    smoothed[t].covariance[i][j] = 0.5 * (smoothed[t].covariance[i][j] + sym[i][j]);
                }
            }
        }

        Ok(smoothed)
    }

    /// Multi-step ahead prediction.
    ///
    /// Produces `horizon` predicted observations starting from the current state.
    /// The filter must have been run (or initial state set) before calling this.
    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        let x = self.state.as_ref().ok_or(ForecastError::FitRequired)?;
        let p = self.covariance.as_ref().ok_or(ForecastError::FitRequired)?;

        if horizon == 0 {
            return Ok(vec![]);
        }

        let f = &self.model.transition;
        let h = &self.model.observation;
        let q = &self.model.process_noise;
        let ft = mat_transpose(f);

        let mut x_cur = x.clone();
        let mut p_cur = p.clone();
        let mut predictions = Vec::with_capacity(horizon);

        for _ in 0..horizon {
            x_cur = mat_vec(f, &x_cur);
            p_cur = mat_add(&mat_mul(&mat_mul(f, &p_cur), &ft), q);
            let y_pred = mat_vec(h, &x_cur);
            predictions.push(y_pred);
        }

        Ok(predictions)
    }

    /// Compute the total log-likelihood of observations under the model.
    pub fn log_likelihood(&self, observations: &[Vec<f64>]) -> Result<f64> {
        let mut kf = self.clone();
        let filtered = kf.filter(observations)?;
        Ok(filtered.iter().map(|s| s.log_likelihood).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate constant data with additive noise.
    fn constant_with_noise(n: usize, level: f64, noise_std: f64) -> Vec<Vec<f64>> {
        // Deterministic pseudo-noise via simple LCG.
        let mut seed: u64 = 42;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (seed >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
                let noise = (u - 0.5) * 2.0 * noise_std;
                vec![level + noise]
            })
            .collect()
    }

    /// Helper: generate linear data y = intercept + slope * t + noise.
    fn linear_with_noise(n: usize, intercept: f64, slope: f64, noise_std: f64) -> Vec<Vec<f64>> {
        let mut seed: u64 = 123;
        (0..n)
            .map(|t| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (seed >> 33) as f64 / (1u64 << 31) as f64;
                let noise = (u - 0.5) * 2.0 * noise_std;
                vec![intercept + slope * t as f64 + noise]
            })
            .collect()
    }

    #[test]
    fn local_level_on_constant_data() {
        let model = StateSpaceModel::local_level(1.0, 0.01);
        let mut kf = KalmanFilter::new(model).unwrap();
        let data = constant_with_noise(100, 5.0, 0.5);
        let filtered = kf.filter(&data).unwrap();

        assert_eq!(filtered.len(), 100);

        // The filtered state should converge near 5.0.
        let last = &filtered[99];
        assert!(
            (last.state[0] - 5.0).abs() < 1.0,
            "filtered state {} not near 5.0",
            last.state[0]
        );

        // Covariance should decrease over time.
        assert!(
            filtered[99].covariance[0][0] < filtered[0].covariance[0][0],
            "covariance should decrease"
        );
    }

    #[test]
    fn local_linear_trend_on_linear_data() {
        let model = StateSpaceModel::local_linear_trend(0.5, 0.01, 0.01);
        let mut kf = KalmanFilter::new(model).unwrap();
        let data = linear_with_noise(200, 2.0, 0.5, 0.3);
        let filtered = kf.filter(&data).unwrap();

        assert_eq!(filtered.len(), 200);

        // State has 2 components: [level, trend].
        assert_eq!(filtered[199].state.len(), 2);

        // After 200 steps, the level should be near 2.0 + 0.5 * 199 = 101.5.
        let expected_level = 2.0 + 0.5 * 199.0;
        assert!(
            (filtered[199].state[0] - expected_level).abs() < 5.0,
            "level {} not near expected {}",
            filtered[199].state[0],
            expected_level
        );

        // Trend should be near 0.5.
        assert!(
            (filtered[199].state[1] - 0.5).abs() < 0.3,
            "trend {} not near 0.5",
            filtered[199].state[1]
        );
    }

    #[test]
    fn smoother_improves_over_filter() {
        let model = StateSpaceModel::local_level(1.0, 0.1);
        let mut kf = KalmanFilter::new(model.clone()).unwrap();
        let level = 10.0;
        let data = constant_with_noise(50, level, 1.0);
        let filtered = kf.filter(&data).unwrap();

        let kf2 = KalmanFilter::new(model).unwrap();
        let smoothed = kf2.smooth(&filtered).unwrap();

        assert_eq!(smoothed.len(), 50);

        // Smoothed covariance should be <= filtered covariance at each step.
        // Check the first half where the difference is most notable.
        let mut smoother_better_count = 0;
        for t in 0..25 {
            if smoothed[t].covariance[0][0] <= filtered[t].covariance[0][0] + 1e-12 {
                smoother_better_count += 1;
            }
        }
        assert!(
            smoother_better_count >= 20,
            "smoother should have smaller covariance in most early steps, got {}/25",
            smoother_better_count
        );
    }

    #[test]
    fn log_likelihood_computation() {
        let model = StateSpaceModel::local_level(1.0, 0.01);
        let kf = KalmanFilter::new(model).unwrap();
        let data = constant_with_noise(50, 5.0, 0.5);

        let ll = kf.log_likelihood(&data).unwrap();

        // Log-likelihood should be finite and negative.
        assert!(ll.is_finite(), "log-likelihood should be finite");
        assert!(ll < 0.0, "log-likelihood should be negative for noisy data");

        // Verify it equals sum of per-step log-likelihoods.
        let mut kf2 = KalmanFilter::new(StateSpaceModel::local_level(1.0, 0.01)).unwrap();
        let filtered = kf2.filter(&data).unwrap();
        let ll_sum: f64 = filtered.iter().map(|s| s.log_likelihood).sum();
        assert!(
            (ll - ll_sum).abs() < 1e-10,
            "total log-likelihood should equal sum of per-step values"
        );
    }

    #[test]
    fn prediction_correct_dimensions() {
        let model = StateSpaceModel::local_linear_trend(1.0, 0.1, 0.01);
        let mut kf = KalmanFilter::new(model).unwrap();
        let data = linear_with_noise(50, 0.0, 1.0, 0.5);
        kf.filter(&data).unwrap();

        let preds = kf.predict(10).unwrap();
        assert_eq!(preds.len(), 10);
        for pred in &preds {
            assert_eq!(
                pred.len(),
                1,
                "each prediction should have n_obs=1 dimensions"
            );
        }

        // Predictions should be roughly increasing (trend model).
        for i in 1..preds.len() {
            assert!(
                preds[i][0] > preds[i - 1][0] - 1.0,
                "predictions should be approximately non-decreasing"
            );
        }
    }

    #[test]
    fn zero_noise_model() {
        // With zero observation noise, the filter should track perfectly.
        let model = StateSpaceModel::local_level(0.0, 0.0);
        let mut kf = KalmanFilter::new(model).unwrap();
        // Set a specific initial state to avoid singular S with zero noise and diffuse prior.
        kf.set_initial_state(vec![5.0], vec![vec![0.0]]);

        let data = vec![vec![5.0]; 10];
        let filtered = kf.filter(&data).unwrap();

        // State should be exactly 5.0 throughout.
        for (t, s) in filtered.iter().enumerate() {
            assert!(
                (s.state[0] - 5.0).abs() < 1e-12,
                "state at t={} should be 5.0, got {}",
                t,
                s.state[0]
            );
        }
    }

    #[test]
    fn single_observation() {
        let model = StateSpaceModel::local_level(1.0, 0.5);
        let mut kf = KalmanFilter::new(model).unwrap();
        let data = vec![vec![3.0]];
        let filtered = kf.filter(&data).unwrap();

        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].state[0].is_finite());
        assert!(filtered[0].log_likelihood.is_finite());
    }

    #[test]
    fn empty_observations_returns_error() {
        let model = StateSpaceModel::local_level(1.0, 0.5);
        let mut kf = KalmanFilter::new(model).unwrap();
        let result = kf.filter(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let model = StateSpaceModel::local_level(1.0, 0.5);
        let mut kf = KalmanFilter::new(model).unwrap();
        // Observation should be 1-d, but we provide 2-d.
        let data = vec![vec![1.0, 2.0]];
        let result = kf.filter(&data);
        assert!(result.is_err());
    }

    #[test]
    fn set_initial_state_affects_filter() {
        let model = StateSpaceModel::local_level(0.1, 0.01);
        let data = constant_with_noise(20, 10.0, 0.1);

        // With default diffuse prior (state = 0, large P).
        let mut kf1 = KalmanFilter::new(model.clone()).unwrap();
        let r1 = kf1.filter(&data).unwrap();

        // With informed prior close to truth.
        let mut kf2 = KalmanFilter::new(model).unwrap();
        kf2.set_initial_state(vec![10.0], vec![vec![0.01]]);
        let r2 = kf2.filter(&data).unwrap();

        // The informed prior should produce a state closer to 10.0 at t=0.
        assert!(
            (r2[0].state[0] - 10.0).abs() < (r1[0].state[0] - 10.0).abs(),
            "informed prior should give better initial estimate"
        );
    }

    #[test]
    fn predict_without_filter_returns_error() {
        let model = StateSpaceModel::local_level(1.0, 0.5);
        let kf = KalmanFilter::new(model).unwrap();
        let result = kf.predict(5);
        assert!(matches!(result.unwrap_err(), ForecastError::FitRequired));
    }

    #[test]
    fn model_validation_rejects_bad_dimensions() {
        let model = StateSpaceModel {
            transition: vec![vec![1.0]],
            observation: vec![vec![1.0, 0.0]], // 1x2, but state dim is 1
            process_noise: vec![vec![1.0]],
            observation_noise: vec![vec![1.0]],
        };
        let result = KalmanFilter::new(model);
        assert!(result.is_err());
    }

    #[test]
    fn smoothing_empty_returns_error() {
        let model = StateSpaceModel::local_level(1.0, 0.5);
        let kf = KalmanFilter::new(model).unwrap();
        let result = kf.smooth(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn predict_zero_horizon() {
        let model = StateSpaceModel::local_level(1.0, 0.5);
        let mut kf = KalmanFilter::new(model).unwrap();
        kf.set_initial_state(vec![0.0], vec![vec![1.0]]);
        let preds = kf.predict(0).unwrap();
        assert!(preds.is_empty());
    }
}
