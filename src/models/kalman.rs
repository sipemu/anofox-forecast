//! Kalman filter framework for linear Gaussian state-space models.
//!
//! State equation:   x(t) = F * x(t-1) + w(t),  w ~ N(0, Q)
//! Observation equation: y(t) = H * x(t) + v(t), v ~ N(0, R)
//!
//! Supports filtering (forward pass), Rauch-Tung-Striebel smoothing,
//! multi-step prediction, and log-likelihood computation.

use crate::error::{ForecastError, Result};

// ---------------------------------------------------------------------------
// DenseMatrix: flat row-major layout for cache-friendly matrix operations
// ---------------------------------------------------------------------------

/// Dense matrix stored in row-major flat `Vec<f64>` for cache locality.
#[derive(Debug, Clone)]
struct DenseMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl DenseMatrix {
    /// Create a new matrix from existing data (row-major order).
    #[allow(dead_code)]
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    /// Create a zero matrix of given dimensions.
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create an n x n identity matrix.
    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Get element at (row, col).
    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Set element at (row, col).
    #[inline(always)]
    #[allow(dead_code)]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Get a row as a slice.
    #[inline]
    fn row(&self, r: usize) -> &[f64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get column values as a new Vec.
    #[allow(dead_code)]
    fn col(&self, c: usize) -> Vec<f64> {
        (0..self.rows)
            .map(|r| self.data[r * self.cols + c])
            .collect()
    }

    /// Convert from `Vec<Vec<f64>>` (row-major nested).
    fn from_nested(nested: &[Vec<f64>]) -> Self {
        let rows = nested.len();
        if rows == 0 {
            return Self::zeros(0, 0);
        }
        let cols = nested[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for row in nested {
            data.extend_from_slice(row);
        }
        Self { data, rows, cols }
    }

    /// Convert to `Vec<Vec<f64>>` for public API compatibility.
    fn to_nested(&self) -> Vec<Vec<f64>> {
        (0..self.rows).map(|r| self.row(r).to_vec()).collect()
    }

    // --- In-place operations ---

    /// self += other
    fn add_inplace(&mut self, other: &DenseMatrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// self -= other
    #[allow(dead_code)]
    fn sub_inplace(&mut self, other: &DenseMatrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= *b;
        }
    }

    /// self *= scalar
    #[allow(dead_code)]
    fn scale_inplace(&mut self, scalar: f64) {
        for v in self.data.iter_mut() {
            *v *= scalar;
        }
    }

    // --- Allocating operations ---

    /// Matrix multiply into pre-allocated buffer: out = self * other
    fn mul_into(&self, other: &DenseMatrix, out: &mut DenseMatrix) {
        debug_assert_eq!(self.cols, other.rows);
        debug_assert_eq!(out.rows, self.rows);
        debug_assert_eq!(out.cols, other.cols);
        // Zero the output
        for v in out.data.iter_mut() {
            *v = 0.0;
        }
        let ar = self.rows;
        let ac = self.cols;
        let bc = other.cols;
        for i in 0..ar {
            let out_row = i * bc;
            let a_row = i * ac;
            for k in 0..ac {
                let a_ik = self.data[a_row + k];
                let b_row = k * bc;
                for j in 0..bc {
                    out.data[out_row + j] += a_ik * other.data[b_row + j];
                }
            }
        }
    }

    /// Matrix-vector multiply: y = self * x
    fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self.cols, x.len());
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let row_start = i * self.cols;
            let mut s = 0.0;
            for j in 0..self.cols {
                s += self.data[row_start + j] * x[j];
            }
            out[i] = s;
        }
        out
    }

    /// Copy contents from another matrix of the same dimensions.
    fn copy_from(&mut self, other: &DenseMatrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);
        self.data.copy_from_slice(&other.data);
    }

    /// Symmetrize in-place: self = (self + self^T) / 2
    fn symmetrize_inplace(&mut self) {
        debug_assert_eq!(self.rows, self.cols);
        let n = self.rows;
        for i in 0..n {
            for j in (i + 1)..n {
                let ij = i * n + j;
                let ji = j * n + i;
                let avg = 0.5 * (self.data[ij] + self.data[ji]);
                self.data[ij] = avg;
                self.data[ji] = avg;
            }
        }
    }

    /// Set self = A - B (reuses allocation).
    fn set_sub(&mut self, a: &DenseMatrix, b: &DenseMatrix) {
        debug_assert_eq!(a.rows, b.rows);
        debug_assert_eq!(a.cols, b.cols);
        debug_assert_eq!(self.rows, a.rows);
        debug_assert_eq!(self.cols, a.cols);
        for i in 0..self.data.len() {
            self.data[i] = a.data[i] - b.data[i];
        }
    }

    /// Fill with zeros.
    fn zero_fill(&mut self) {
        for v in self.data.iter_mut() {
            *v = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition and solvers using DenseMatrix
// ---------------------------------------------------------------------------

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Writes lower-triangular L into `l` such that A = L * L^T.
fn dm_cholesky(a: &DenseMatrix, l: &mut DenseMatrix) -> Result<()> {
    debug_assert_eq!(a.rows, a.cols);
    let n = a.rows;
    debug_assert_eq!(l.rows, n);
    debug_assert_eq!(l.cols, n);
    l.zero_fill();
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            let ljk = l.data[j * n + k];
            sum += ljk * ljk;
        }
        let diag = a.data[j * n + j] - sum;
        if diag < 0.0 {
            return Err(ForecastError::SingularMatrix(
                "matrix is not positive-definite in Cholesky decomposition".into(),
            ));
        }
        let ljj = diag.sqrt();
        if ljj == 0.0 {
            return Err(ForecastError::SingularMatrix(
                "zero diagonal in Cholesky decomposition".into(),
            ));
        }
        l.data[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l.data[i * n + k] * l.data[j * n + k];
            }
            l.data[i * n + j] = (a.data[i * n + j] - s) / ljj;
        }
    }
    Ok(())
}

/// Solve L * x = b via forward substitution (L lower triangular).
fn dm_forward_solve(l: &DenseMatrix, b: &[f64], x: &mut [f64]) {
    let n = b.len();
    for i in 0..n {
        let mut s = 0.0;
        let row = i * l.cols;
        for j in 0..i {
            s += l.data[row + j] * x[j];
        }
        x[i] = (b[i] - s) / l.data[row + i];
    }
}

/// Solve L^T * x = b via back substitution (L lower triangular).
fn dm_back_solve(l: &DenseMatrix, b: &[f64], x: &mut [f64]) {
    let n = b.len();
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += l.data[j * l.cols + i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (b[i] - s) / l.data[i * l.cols + i];
    }
}

/// Inverse of a symmetric positive-definite matrix via Cholesky.
/// Uses pre-allocated scratch buffers.
fn dm_inv_spd(
    a: &DenseMatrix,
    l: &mut DenseMatrix,
    inv: &mut DenseMatrix,
    y_buf: &mut [f64],
    x_buf: &mut [f64],
) -> Result<()> {
    let n = a.rows;
    dm_cholesky(a, l)?;
    inv.zero_fill();
    for col in 0..n {
        // Set up unit vector in y_buf
        for v in y_buf[..n].iter_mut() {
            *v = 0.0;
        }
        y_buf[col] = 1.0;
        dm_forward_solve(l, &y_buf[..n], &mut x_buf[..n]);
        // Now x_buf holds the forward-solve result; solve L^T * result = x_buf
        // Reuse y_buf for the final result
        dm_back_solve(l, &x_buf[..n], &mut y_buf[..n]);
        for row in 0..n {
            inv.data[row * n + col] = y_buf[row];
        }
    }
    Ok(())
}

/// Log-determinant of a symmetric positive-definite matrix via Cholesky.
fn dm_log_det_spd(a: &DenseMatrix, l: &mut DenseMatrix) -> Result<f64> {
    dm_cholesky(a, l)?;
    let n = a.rows;
    let mut ld = 0.0;
    for i in 0..n {
        ld += l.data[i * n + i].ln();
    }
    Ok(2.0 * ld)
}

/// Quadratic form x^T * A^{-1} * x for SPD A.
fn dm_quad_form_inv(
    a: &DenseMatrix,
    x: &[f64],
    l: &mut DenseMatrix,
    y_buf: &mut [f64],
) -> Result<f64> {
    let n = a.rows;
    dm_cholesky(a, l)?;
    dm_forward_solve(l, x, &mut y_buf[..n]);
    Ok(y_buf[..n].iter().map(|v| v * v).sum())
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Internal version of StateSpaceModel using DenseMatrix for performance.
#[derive(Debug, Clone)]
struct InternalSSM {
    transition: DenseMatrix,        // F: ns x ns
    observation: DenseMatrix,       // H: no x ns
    process_noise: DenseMatrix,     // Q: ns x ns
    observation_noise: DenseMatrix, // R: no x no
    transition_t: DenseMatrix,      // F^T: ns x ns (pre-computed)
    observation_t: DenseMatrix,     // H^T: ns x no (pre-computed)
}

impl InternalSSM {
    fn from_model(model: &StateSpaceModel) -> Self {
        let f = DenseMatrix::from_nested(&model.transition);
        let h = DenseMatrix::from_nested(&model.observation);
        let q = DenseMatrix::from_nested(&model.process_noise);
        let r = DenseMatrix::from_nested(&model.observation_noise);
        // Pre-compute transposes (done once, reused every step).
        let ft = {
            let ns = f.rows;
            let mut t = DenseMatrix::zeros(ns, ns);
            for i in 0..ns {
                for j in 0..ns {
                    t.data[j * ns + i] = f.data[i * ns + j];
                }
            }
            t
        };
        let ht = {
            let no = h.rows;
            let ns = h.cols;
            let mut t = DenseMatrix::zeros(ns, no);
            for i in 0..no {
                for j in 0..ns {
                    t.data[j * no + i] = h.data[i * ns + j];
                }
            }
            t
        };
        Self {
            transition: f,
            observation: h,
            process_noise: q,
            observation_noise: r,
            transition_t: ft,
            observation_t: ht,
        }
    }

    fn n_state(&self) -> usize {
        self.transition.rows
    }

    fn n_obs(&self) -> usize {
        self.observation.rows
    }
}

/// Pre-allocated scratch buffers for the Kalman filter loop.
struct FilterScratch {
    // Predict step temporaries
    fp: DenseMatrix,     // F * P, ns x ns
    p_pred: DenseMatrix, // F * P * F^T + Q, ns x ns

    // Update step temporaries
    hp: DenseMatrix,    // H * P_pred, no x ns
    s: DenseMatrix,     // H * P_pred * H^T + R, no x no
    pht: DenseMatrix,   // P_pred * H^T, ns x no
    s_inv: DenseMatrix, // S^{-1}, no x no
    k: DenseMatrix,     // Kalman gain, ns x no
    kh: DenseMatrix,    // K * H, ns x ns
    i_kh: DenseMatrix,  // I - K*H, ns x ns

    // Cholesky scratch
    l_obs: DenseMatrix, // Cholesky factor for observation space, no x no

    // Vector scratch
    y_buf: Vec<f64>,
    x_buf: Vec<f64>,
}

impl FilterScratch {
    fn new(ns: usize, no: usize) -> Self {
        let buf_len = ns.max(no);
        Self {
            fp: DenseMatrix::zeros(ns, ns),
            p_pred: DenseMatrix::zeros(ns, ns),
            hp: DenseMatrix::zeros(no, ns),
            s: DenseMatrix::zeros(no, no),
            pht: DenseMatrix::zeros(ns, no),
            s_inv: DenseMatrix::zeros(no, no),
            k: DenseMatrix::zeros(ns, no),
            kh: DenseMatrix::zeros(ns, ns),
            i_kh: DenseMatrix::zeros(ns, ns),
            l_obs: DenseMatrix::zeros(no, no),
            y_buf: vec![0.0; buf_len],
            x_buf: vec![0.0; buf_len],
        }
    }
}

/// Pre-allocated scratch buffers for the RTS smoother loop.
struct SmoothScratch {
    fp: DenseMatrix,         // F * P, ns x ns
    p_pred: DenseMatrix,     // F * P * F^T + Q, ns x ns
    p_pred_inv: DenseMatrix, // P_pred^{-1}, ns x ns
    pft: DenseMatrix,        // P * F^T, ns x ns
    g: DenseMatrix,          // smoother gain, ns x ns
    gt: DenseMatrix,         // G^T, ns x ns
    p_diff: DenseMatrix,     // P_s(t+1) - P_pred, ns x ns
    gp: DenseMatrix,         // G * p_diff, ns x ns
    gpgt: DenseMatrix,       // G * p_diff * G^T, ns x ns

    // Cholesky scratch
    l: DenseMatrix,

    // Vector scratch
    y_buf: Vec<f64>,
    x_buf: Vec<f64>,
    diff: Vec<f64>,
}

impl SmoothScratch {
    fn new(ns: usize) -> Self {
        Self {
            fp: DenseMatrix::zeros(ns, ns),
            p_pred: DenseMatrix::zeros(ns, ns),
            p_pred_inv: DenseMatrix::zeros(ns, ns),
            pft: DenseMatrix::zeros(ns, ns),
            g: DenseMatrix::zeros(ns, ns),
            gt: DenseMatrix::zeros(ns, ns),
            p_diff: DenseMatrix::zeros(ns, ns),
            gp: DenseMatrix::zeros(ns, ns),
            gpgt: DenseMatrix::zeros(ns, ns),
            l: DenseMatrix::zeros(ns, ns),
            y_buf: vec![0.0; ns],
            x_buf: vec![0.0; ns],
            diff: vec![0.0; ns],
        }
    }
}

/// Helper: row count of nested Vec matrix.
fn mat_rows(m: &[Vec<f64>]) -> usize {
    m.len()
}

/// Helper: column count of nested Vec matrix.
fn mat_cols(m: &[Vec<f64>]) -> usize {
    if m.is_empty() {
        0
    } else {
        m[0].len()
    }
}

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
    internal: InternalSSM,
    state: Option<Vec<f64>>,
    covariance: Option<DenseMatrix>,
}

impl KalmanFilter {
    /// Create a new Kalman filter from a state-space model specification.
    ///
    /// The model dimensions are validated on construction.
    pub fn new(model: StateSpaceModel) -> Result<Self> {
        model.validate()?;
        let internal = InternalSSM::from_model(&model);
        Ok(Self {
            internal,
            state: None,
            covariance: None,
        })
    }

    /// Set initial state estimate and error covariance.
    pub fn set_initial_state(&mut self, state: Vec<f64>, covariance: Vec<Vec<f64>>) {
        self.state = Some(state);
        self.covariance = Some(DenseMatrix::from_nested(&covariance));
    }

    /// Run the forward Kalman filter on a sequence of observations.
    ///
    /// Each element of `observations` is a vector of length `n_obs`.
    /// Returns one `KalmanState` per time step.
    pub fn filter(&mut self, observations: &[Vec<f64>]) -> Result<Vec<KalmanState>> {
        if observations.is_empty() {
            return Err(ForecastError::EmptyData);
        }
        let ns = self.internal.n_state();
        let no = self.internal.n_obs();

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

        let ssm = &self.internal;

        // Initial state: use diffuse prior if not set.
        let mut x = self.state.clone().unwrap_or_else(|| vec![0.0; ns]);
        let mut p = self.covariance.clone().unwrap_or_else(|| {
            let mut m = DenseMatrix::identity(ns);
            for i in 0..ns {
                m.data[i * ns + i] = 1e6;
            }
            m
        });

        let mut results = Vec::with_capacity(observations.len());

        // Pre-allocate all scratch buffers (reused every iteration).
        let mut scratch = FilterScratch::new(ns, no);

        for obs in observations {
            // --- Predict ---
            let x_pred = ssm.transition.mul_vec(&x);

            // p_pred = F * P * F^T + Q
            ssm.transition.mul_into(&p, &mut scratch.fp);
            scratch.fp.mul_into(&ssm.transition_t, &mut scratch.p_pred);
            scratch.p_pred.add_inplace(&ssm.process_noise);

            // --- Update ---
            // Innovation: y - H * x_pred
            let y_pred = ssm.observation.mul_vec(&x_pred);
            let innovation: Vec<f64> = obs.iter().zip(y_pred.iter()).map(|(a, b)| a - b).collect();

            // Innovation covariance: S = H * P_pred * H^T + R
            ssm.observation.mul_into(&scratch.p_pred, &mut scratch.hp);
            scratch.hp.mul_into(&ssm.observation_t, &mut scratch.s);
            scratch.s.add_inplace(&ssm.observation_noise);

            // Check if S is (near) singular -- degenerate case with zero noise.
            let s_max_diag = (0..no)
                .map(|i| scratch.s.get(i, i).abs())
                .fold(0.0_f64, f64::max);
            let degenerate = s_max_diag < 1e-30;

            let ll;
            if degenerate {
                // S is effectively zero: perfect prediction, no update needed.
                x = x_pred;
                p.copy_from(&scratch.p_pred);
                ll = 0.0;
            } else {
                // Kalman gain: K = P_pred * H^T * S^{-1}
                dm_inv_spd(
                    &scratch.s,
                    &mut scratch.l_obs,
                    &mut scratch.s_inv,
                    &mut scratch.y_buf,
                    &mut scratch.x_buf,
                )?;
                scratch
                    .p_pred
                    .mul_into(&ssm.observation_t, &mut scratch.pht);
                scratch.pht.mul_into(&scratch.s_inv, &mut scratch.k);

                // Updated state: x = x_pred + K * innovation
                let k_inn = scratch.k.mul_vec(&innovation);
                x.clear();
                x.extend(x_pred.iter().zip(k_inn.iter()).map(|(a, b)| a + b));

                // Updated covariance: P = (I - K * H) * P_pred
                scratch.k.mul_into(&ssm.observation, &mut scratch.kh);
                // i_kh = I - kh
                for i in 0..ns {
                    for j in 0..ns {
                        let idx = i * ns + j;
                        scratch.i_kh.data[idx] =
                            if i == j { 1.0 } else { 0.0 } - scratch.kh.data[idx];
                    }
                }
                scratch.i_kh.mul_into(&scratch.p_pred, &mut p);

                // Symmetrize P for numerical stability.
                p.symmetrize_inplace();

                // Log-likelihood contribution:
                // -0.5 * (n_obs * ln(2*pi) + ln|S| + innovation^T * S^{-1} * innovation)
                let log_det = dm_log_det_spd(&scratch.s, &mut scratch.l_obs)?;
                let quad = dm_quad_form_inv(
                    &scratch.s,
                    &innovation,
                    &mut scratch.l_obs,
                    &mut scratch.y_buf,
                )?;
                ll = -0.5 * (no as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad);
            }

            results.push(KalmanState {
                state: x.clone(),
                covariance: p.to_nested(),
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
        let ssm = &self.internal;
        let ns = ssm.n_state();

        let mut smoothed = filtered.to_vec();

        // Pre-allocate scratch buffers (reused every iteration).
        let mut scratch = SmoothScratch::new(ns);

        // Backward pass: t = n-2 .. 0
        for t in (0..n.saturating_sub(1)).rev() {
            let p_filt = DenseMatrix::from_nested(&filtered[t].covariance);

            // Predicted state and covariance at t+1 from filtered state at t.
            let x_pred = ssm.transition.mul_vec(&filtered[t].state);

            // p_pred = F * P_filt * F^T + Q
            ssm.transition.mul_into(&p_filt, &mut scratch.fp);
            scratch.fp.mul_into(&ssm.transition_t, &mut scratch.p_pred);
            scratch.p_pred.add_inplace(&ssm.process_noise);

            // Smoother gain: G = P_filt * F^T * P_pred^{-1}
            dm_inv_spd(
                &scratch.p_pred,
                &mut scratch.l,
                &mut scratch.p_pred_inv,
                &mut scratch.y_buf,
                &mut scratch.x_buf,
            )?;
            p_filt.mul_into(&ssm.transition_t, &mut scratch.pft);
            scratch.pft.mul_into(&scratch.p_pred_inv, &mut scratch.g);

            // Smoothed state: x_s(t) = x_f(t) + G * (x_s(t+1) - x_pred(t+1))
            for i in 0..ns {
                scratch.diff[i] = smoothed[t + 1].state[i] - x_pred[i];
            }
            let correction = scratch.g.mul_vec(&scratch.diff);
            for i in 0..ns {
                smoothed[t].state[i] = filtered[t].state[i] + correction[i];
            }

            // Smoothed covariance:
            // P_s(t) = P_f(t) + G * (P_s(t+1) - P_pred(t+1)) * G^T
            let p_smooth_next = DenseMatrix::from_nested(&smoothed[t + 1].covariance);
            scratch.p_diff.set_sub(&p_smooth_next, &scratch.p_pred);

            // gt = G^T
            for i in 0..ns {
                for j in 0..ns {
                    scratch.gt.data[j * ns + i] = scratch.g.data[i * ns + j];
                }
            }

            scratch.g.mul_into(&scratch.p_diff, &mut scratch.gp);
            scratch.gp.mul_into(&scratch.gt, &mut scratch.gpgt);

            // P_s(t) = P_f(t) + gpgt
            let mut p_smoothed = p_filt;
            p_smoothed.add_inplace(&scratch.gpgt);
            p_smoothed.symmetrize_inplace();

            smoothed[t].covariance = p_smoothed.to_nested();
        }

        Ok(smoothed)
    }

    /// Multi-step ahead prediction.
    ///
    /// Produces `horizon` predicted observations starting from the current state.
    /// The filter must have been run (or initial state set) before calling this.
    pub fn predict(&self, horizon: usize) -> Result<Vec<Vec<f64>>> {
        let x = self
            .state
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;
        let p = self
            .covariance
            .as_ref()
            .ok_or(ForecastError::FitRequired { model: None })?;

        if horizon == 0 {
            return Ok(vec![]);
        }

        let ssm = &self.internal;
        let ns = ssm.n_state();

        let mut x_cur = x.clone();
        let mut p_cur = p.clone();
        let mut predictions = Vec::with_capacity(horizon);

        // Pre-allocate scratch for predict loop.
        let mut fp = DenseMatrix::zeros(ns, ns);
        let mut p_next = DenseMatrix::zeros(ns, ns);

        for _ in 0..horizon {
            x_cur = ssm.transition.mul_vec(&x_cur);
            ssm.transition.mul_into(&p_cur, &mut fp);
            fp.mul_into(&ssm.transition_t, &mut p_next);
            p_next.add_inplace(&ssm.process_noise);
            p_cur.copy_from(&p_next);
            let y_pred = ssm.observation.mul_vec(&x_cur);
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
        assert!(matches!(
            result.unwrap_err(),
            ForecastError::FitRequired { .. }
        ));
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
