//! Optimization utilities for parameter estimation.

/// Result of Nelder-Mead optimization.
#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    /// The optimal point found.
    pub optimal_point: Vec<f64>,
    /// The objective function value at the optimal point.
    pub optimal_value: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Configuration for Nelder-Mead optimization.
#[derive(Debug, Clone, Copy)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Stagnation window: terminate if best value hasn't improved by more than
    /// `tolerance` in this many iterations. 0 = disabled.
    pub stagnation_window: usize,
    /// Reflection coefficient (default: 1.0).
    pub alpha: f64,
    /// Expansion coefficient (default: 2.0).
    pub gamma: f64,
    /// Contraction coefficient (default: 0.5).
    pub rho: f64,
    /// Shrinkage coefficient (default: 0.5).
    pub sigma: f64,
    /// Initial simplex step size (default: 0.05).
    pub initial_step: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance: 1e-8,
            stagnation_window: 0,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            initial_step: 0.05,
        }
    }
}

/// Contiguous simplex buffer: `(n+1)` vertices of dimension `n` stored flat.
/// Eliminates `n+2` heap allocations and pointer-chasing of `Vec<Vec<f64>>`.
struct Simplex {
    data: Vec<f64>,
    dim: usize,
}

impl Simplex {
    fn new(dim: usize) -> Self {
        Self {
            data: vec![0.0; (dim + 1) * dim],
            dim,
        }
    }

    #[inline]
    fn vertex(&self, i: usize) -> &[f64] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    fn vertex_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    fn n_vertices(&self) -> usize {
        self.dim + 1
    }
}

/// Sanitize objective value: replace NaN/Inf with MAX to prevent silent propagation.
#[inline]
fn sanitize_objective(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::MAX
    }
}

/// Perform Nelder-Mead simplex optimization.
///
/// # Arguments
/// * `objective` - The objective function to minimize
/// * `initial` - Initial guess for the optimal point
/// * `bounds` - Optional bounds for each dimension as (min, max) pairs
/// * `config` - Configuration parameters
///
/// # Returns
/// `NelderMeadResult` containing the optimal point and convergence information.
///
/// # Example
/// ```
/// use anofox_forecast::utils::optimization::{nelder_mead, NelderMeadConfig};
///
/// // Minimize (x-2)^2 + (y-3)^2
/// let result = nelder_mead(
///     |x| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
///     &[0.0, 0.0],
///     None,
///     NelderMeadConfig::default(),
/// );
///
/// assert!(result.converged);
/// assert!((result.optimal_point[0] - 2.0).abs() < 0.01);
/// assert!((result.optimal_point[1] - 3.0).abs() < 0.01);
/// ```
pub fn nelder_mead<F>(
    objective: F,
    initial: &[f64],
    bounds: Option<&[(f64, f64)]>,
    config: NelderMeadConfig,
) -> NelderMeadResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = initial.len();
    if n == 0 {
        return NelderMeadResult {
            optimal_point: vec![],
            optimal_value: f64::NAN,
            iterations: 0,
            converged: false,
        };
    }

    // Initialize simplex with n+1 vertices in contiguous buffer
    let mut simplex = Simplex::new(n);
    {
        let v0 = simplex.vertex_mut(0);
        v0.copy_from_slice(initial);
        apply_bounds_in_place(v0, bounds);
    }

    for i in 0..n {
        let vi = simplex.vertex_mut(i + 1);
        vi.copy_from_slice(initial);
        let step = if initial[i].abs() > 1e-10 {
            config.initial_step * initial[i].abs()
        } else {
            config.initial_step
        };
        vi[i] += step;
        apply_bounds_in_place(vi, bounds);
    }

    // Evaluate objective at all vertices
    let mut values: Vec<f64> = (0..simplex.n_vertices())
        .map(|i| sanitize_objective(objective(simplex.vertex(i))))
        .collect();

    // Pre-allocate scratch buffers
    let mut indices: Vec<usize> = (0..=n).collect();
    let mut centroid = vec![0.0; n];
    let mut reflected = vec![0.0; n];
    let mut expanded = vec![0.0; n];
    let mut contracted = vec![0.0; n];
    let mut temp = vec![0.0; n];

    let mut iterations = 0;
    let mut converged = false;
    let mut stagnation_counter = 0usize;
    let mut stagnation_best = f64::MAX;

    while iterations < config.max_iter {
        iterations += 1;

        // Sort vertices by objective value
        sort_simplex_indices(&mut indices, &values);
        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check convergence: value range and simplex diameter
        if check_convergence(
            &simplex,
            &values,
            best_idx,
            worst_idx,
            config.tolerance,
            &mut centroid,
        ) {
            converged = true;
            break;
        }

        // Stagnation check: terminate if best value hasn't improved significantly
        if config.stagnation_window > 0 {
            let current_best = values[best_idx];
            if stagnation_best - current_best > config.tolerance {
                stagnation_best = current_best;
                stagnation_counter = 0;
            } else {
                stagnation_counter += 1;
                if stagnation_counter >= config.stagnation_window {
                    converged = true;
                    break;
                }
            }
        }

        // Try reflection/expansion; if not accepted, try contraction with the reflected value
        let reflected_value = match try_reflection_expansion(
            &objective,
            &config,
            bounds,
            &mut simplex,
            &mut values,
            worst_idx,
            best_idx,
            second_worst_idx,
            &centroid,
            &mut reflected,
            &mut expanded,
        ) {
            None => continue, // reflection or expansion accepted
            Some(rv) => rv,   // pass reflected_value to contraction
        };

        if try_contraction(
            &objective,
            &config,
            bounds,
            &mut simplex,
            &mut values,
            worst_idx,
            &centroid,
            &reflected,
            reflected_value,
            &mut contracted,
        ) {
            continue;
        }

        // Shrink all vertices towards the best
        shrink_simplex(
            &objective,
            &config,
            bounds,
            &mut simplex,
            &mut values,
            best_idx,
            &mut temp,
        );
    }

    // Find best vertex
    let best_idx = values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    NelderMeadResult {
        optimal_point: simplex.vertex(best_idx).to_vec(),
        optimal_value: values[best_idx],
        iterations,
        converged,
    }
}

/// Sort simplex indices by objective value in ascending order.
#[inline]
fn sort_simplex_indices(indices: &mut [usize], values: &[f64]) {
    for (i, idx) in indices.iter_mut().enumerate() {
        *idx = i;
    }
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Check convergence by value range and simplex diameter (squared distance).
/// Returns true if converged. Writes centroid as a side effect (needed by caller).
#[inline]
fn check_convergence(
    simplex: &Simplex,
    values: &[f64],
    best_idx: usize,
    worst_idx: usize,
    tolerance: f64,
    centroid: &mut [f64],
) -> bool {
    let range = values[worst_idx] - values[best_idx];
    if range < tolerance {
        return true;
    }

    compute_centroid_into(simplex, worst_idx, centroid);
    let tol_sq = tolerance * tolerance;
    // any() short-circuits: stops at first vertex exceeding tolerance
    !(0..simplex.n_vertices()).any(|i| distance_sq(simplex.vertex(i), centroid) >= tol_sq)
}

/// Compute reflection and try expansion. Returns `Some(reflected_value)` if neither
/// reflection nor expansion was accepted (caller should try contraction), or `None`
/// if the worst vertex was successfully replaced.
#[inline]
fn try_reflection_expansion<F: Fn(&[f64]) -> f64>(
    objective: &F,
    config: &NelderMeadConfig,
    bounds: Option<&[(f64, f64)]>,
    simplex: &mut Simplex,
    values: &mut [f64],
    worst_idx: usize,
    best_idx: usize,
    second_worst_idx: usize,
    centroid: &[f64],
    reflected: &mut [f64],
    expanded: &mut [f64],
) -> Option<f64> {
    reflect_into(simplex.vertex(worst_idx), centroid, config.alpha, reflected);
    apply_bounds_in_place(reflected, bounds);
    let reflected_value = sanitize_objective(objective(reflected));

    if reflected_value < values[second_worst_idx] && reflected_value >= values[best_idx] {
        simplex.vertex_mut(worst_idx).copy_from_slice(reflected);
        values[worst_idx] = reflected_value;
        return None; // accepted
    }

    if reflected_value < values[best_idx] {
        expand_into(centroid, reflected, config.gamma, expanded);
        apply_bounds_in_place(expanded, bounds);
        let expanded_value = sanitize_objective(objective(expanded));

        if expanded_value < reflected_value {
            simplex.vertex_mut(worst_idx).copy_from_slice(expanded);
            values[worst_idx] = expanded_value;
        } else {
            simplex.vertex_mut(worst_idx).copy_from_slice(reflected);
            values[worst_idx] = reflected_value;
        }
        return None; // accepted
    }

    Some(reflected_value) // not accepted, pass value to contraction
}

/// Try contraction step. Returns true if the worst vertex was replaced.
#[inline]
fn try_contraction<F: Fn(&[f64]) -> f64>(
    objective: &F,
    config: &NelderMeadConfig,
    bounds: Option<&[(f64, f64)]>,
    simplex: &mut Simplex,
    values: &mut [f64],
    worst_idx: usize,
    centroid: &[f64],
    reflected: &[f64],
    reflected_value: f64,
    contracted: &mut [f64],
) -> bool {
    if reflected_value < values[worst_idx] {
        // Outside contraction
        contract_into(centroid, reflected, config.rho, contracted);
        apply_bounds_in_place(contracted, bounds);
        let contracted_value = sanitize_objective(objective(contracted));

        if contracted_value <= reflected_value {
            simplex.vertex_mut(worst_idx).copy_from_slice(contracted);
            values[worst_idx] = contracted_value;
            return true;
        }
    } else {
        // Inside contraction
        contract_into(centroid, simplex.vertex(worst_idx), config.rho, contracted);
        apply_bounds_in_place(contracted, bounds);
        let contracted_value = sanitize_objective(objective(contracted));

        if contracted_value < values[worst_idx] {
            simplex.vertex_mut(worst_idx).copy_from_slice(contracted);
            values[worst_idx] = contracted_value;
            return true;
        }
    }

    false
}

/// Shrink all vertices towards the best vertex.
#[inline]
fn shrink_simplex<F: Fn(&[f64]) -> f64>(
    objective: &F,
    config: &NelderMeadConfig,
    bounds: Option<&[(f64, f64)]>,
    simplex: &mut Simplex,
    values: &mut [f64],
    best_idx: usize,
    temp: &mut [f64],
) {
    let n = temp.len();
    temp.copy_from_slice(simplex.vertex(best_idx));
    for i in 0..=n {
        if i != best_idx {
            let vi = simplex.vertex_mut(i);
            for j in 0..n {
                vi[j] = temp[j] + config.sigma * (vi[j] - temp[j]);
            }
            apply_bounds_in_place(vi, bounds);
            values[i] = sanitize_objective(objective(simplex.vertex(i)));
        }
    }
}

/// Compute centroid of simplex excluding one vertex, writing into `out`.
fn compute_centroid_into(simplex: &Simplex, exclude_idx: usize, out: &mut [f64]) {
    let count = simplex.n_vertices() - 1;
    for o in out.iter_mut() {
        *o = 0.0;
    }

    for i in 0..simplex.n_vertices() {
        if i != exclude_idx {
            let vertex = simplex.vertex(i);
            for (o, &v) in out.iter_mut().zip(vertex.iter()) {
                *o += v;
            }
        }
    }

    let inv = 1.0 / count as f64;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Reflect a point through the centroid, writing into `out`.
fn reflect_into(point: &[f64], centroid: &[f64], alpha: f64, out: &mut [f64]) {
    for ((o, c), p) in out.iter_mut().zip(centroid.iter()).zip(point.iter()) {
        *o = c + alpha * (c - p);
    }
}

/// Expand from centroid towards reflected point, writing into `out`.
fn expand_into(centroid: &[f64], reflected: &[f64], gamma: f64, out: &mut [f64]) {
    for ((o, c), r) in out.iter_mut().zip(centroid.iter()).zip(reflected.iter()) {
        *o = c + gamma * (r - c);
    }
}

/// Contract between centroid and a point, writing into `out`.
fn contract_into(centroid: &[f64], point: &[f64], rho: f64, out: &mut [f64]) {
    for ((o, c), p) in out.iter_mut().zip(centroid.iter()).zip(point.iter()) {
        *o = c + rho * (p - c);
    }
}

/// Apply bounds to a point in place.
fn apply_bounds_in_place(point: &mut [f64], bounds: Option<&[(f64, f64)]>) {
    if let Some(b) = bounds {
        for (i, x) in point.iter_mut().enumerate() {
            if i < b.len() {
                *x = x.clamp(b[i].0, b[i].1);
            }
        }
    }
}

/// Squared distance between two points (avoids sqrt; uses `d*d` instead of `powi(2)`).
#[inline]
fn distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn nelder_mead_quadratic_2d() {
        // Minimize (x-2)^2 + (y-3)^2
        let result = nelder_mead(
            |x| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_point[1], 3.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_value, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn nelder_mead_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        // Minimum at (1, 1)
        let config = NelderMeadConfig {
            max_iter: 5000,
            tolerance: 1e-10,
            ..Default::default()
        };

        let result = nelder_mead(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
            &[0.0, 0.0],
            None,
            config,
        );

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.optimal_point[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn nelder_mead_1d() {
        // Minimize (x-5)^2
        let result = nelder_mead(
            |x| (x[0] - 5.0).powi(2),
            &[0.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 5.0, epsilon = 0.1);
    }

    #[test]
    fn nelder_mead_with_bounds() {
        // Minimize (x-5)^2 with x in [0, 3]
        // Optimal should be at boundary x=3
        let result = nelder_mead(
            |x| (x[0] - 5.0).powi(2),
            &[1.0],
            Some(&[(0.0, 3.0)]),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(result.optimal_point[0], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_with_bounds_2d() {
        // Minimize (x-2)^2 + (y-3)^2 with x in [0,1], y in [0,1]
        // Optimal should be at (1, 1)
        let result = nelder_mead(
            |x| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.5, 0.5],
            Some(&[(0.0, 1.0), (0.0, 1.0)]),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_point[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_exponential_smoothing_alpha() {
        // Simulate finding optimal alpha for exponential smoothing
        // Given data with known optimal alpha around 0.3
        let data = [10.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 16.0];

        let sse = |params: &[f64]| {
            let alpha = params[0];
            let mut level = data[0];
            let mut error_sum = 0.0;

            for &y in &data[1..] {
                let forecast = level;
                let error = y - forecast;
                error_sum += error * error;
                level = alpha * y + (1.0 - alpha) * level;
            }

            error_sum
        };

        let result = nelder_mead(
            sse,
            &[0.5],
            Some(&[(0.01, 0.99)]),
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert!(result.optimal_point[0] > 0.01 && result.optimal_point[0] < 0.99);
    }

    #[test]
    fn nelder_mead_empty_initial() {
        let result = nelder_mead(|_| 0.0, &[], None, NelderMeadConfig::default());

        assert!(!result.converged);
        assert!(result.optimal_value.is_nan());
    }

    #[test]
    fn nelder_mead_already_optimal() {
        // Start at the optimal point
        let result = nelder_mead(
            |x| (x[0] - 2.0).powi(2),
            &[2.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_3d() {
        // Minimize x^2 + y^2 + z^2
        let result = nelder_mead(
            |x| x[0].powi(2) + x[1].powi(2) + x[2].powi(2),
            &[1.0, 2.0, 3.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_point[1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_point[2], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_config_custom() {
        let config = NelderMeadConfig {
            max_iter: 100,
            tolerance: 1e-4,
            stagnation_window: 0,
            alpha: 1.5,
            gamma: 2.5,
            rho: 0.4,
            sigma: 0.4,
            initial_step: 0.1,
        };

        let result = nelder_mead(|x| (x[0] - 1.0).powi(2), &[0.0], None, config);

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 0.01);
    }

    #[test]
    fn nelder_mead_nan_objective_handled() {
        // Objective that returns NaN for negative x, valid for positive x
        let result = nelder_mead(
            |x| {
                if x[0] < 0.0 {
                    f64::NAN
                } else {
                    (x[0] - 3.0).powi(2)
                }
            },
            &[1.0],
            None,
            NelderMeadConfig::default(),
        );

        // Should still converge to valid minimum despite NaN regions
        assert!(result.converged);
        assert!(result.optimal_value.is_finite());
        assert_relative_eq!(result.optimal_point[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn nelder_mead_inf_objective_handled() {
        // Objective that returns Inf for some regions
        let result = nelder_mead(
            |x| {
                if x[0] < -1.0 {
                    f64::INFINITY
                } else {
                    (x[0] - 2.0).powi(2)
                }
            },
            &[1.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert!(result.optimal_value.is_finite());
        assert_relative_eq!(result.optimal_point[0], 2.0, epsilon = 0.1);
    }
}

// =========================================================================
// L-BFGS optimizer with finite-difference gradients
// =========================================================================

/// Configuration for L-BFGS optimization.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance: f64,
    /// L-BFGS memory size (number of stored vectors).
    pub memory_size: usize,
    /// Step size for finite-difference gradient approximation.
    pub fd_step: f64,
    /// Backtracking line search: sufficient decrease parameter (Armijo).
    pub armijo_c: f64,
    /// Backtracking line search: step shrink factor.
    pub backtrack_rho: f64,
    /// Maximum line search steps.
    pub max_linesearch: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            memory_size: 7,
            fd_step: 1e-7,
            armijo_c: 1e-4,
            backtrack_rho: 0.5,
            max_linesearch: 20,
        }
    }
}

/// Run L-BFGS optimization with finite-difference gradients and optional box constraints.
///
/// Uses the `lbfgs` crate for Hessian approximation, with backtracking line search
/// and tanh-based parameter transformation for box constraints.
///
/// Much faster than Nelder-Mead for smooth objectives (ARIMA CSS): converges in
/// ~30 iterations vs NM's 200-1000.
pub fn lbfgs_optimize<F>(
    objective: F,
    initial: &[f64],
    bounds: Option<&[(f64, f64)]>,
    config: LbfgsConfig,
) -> NelderMeadResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = initial.len();
    if n == 0 {
        return NelderMeadResult {
            optimal_point: vec![],
            optimal_value: f64::NAN,
            iterations: 0,
            converged: false,
        };
    }

    let has_bounds = bounds.is_some();

    // Transform initial point to unconstrained space if bounds exist
    let mut x = if has_bounds {
        to_unconstrained(initial, bounds.unwrap())
    } else {
        initial.to_vec()
    };

    let eval = |x_unc: &[f64]| -> f64 {
        if has_bounds {
            let x_con = to_constrained(x_unc, bounds.unwrap());
            sanitize_objective(objective(&x_con))
        } else {
            sanitize_objective(objective(x_unc))
        }
    };

    let mut lbfgs_state = lbfgs::Lbfgs::<f64>::new(n, config.memory_size).with_sy_epsilon(1e-10);

    let mut fx = eval(&x);
    let mut grad = finite_difference_gradient(&eval, &x, config.fd_step);

    // Initial Hessian update
    lbfgs_state.update_hessian(&grad, &x);

    let mut iterations = 0;
    let mut converged = false;

    let mut x_new = vec![0.0; n];
    let mut best_x = x.clone();
    let mut best_fx = fx;

    while iterations < config.max_iter {
        iterations += 1;

        // Compute search direction: d = -H^{-1} * g
        let mut direction = grad.clone();
        lbfgs_state.apply_hessian(&mut direction);
        for d in direction.iter_mut() {
            *d = -*d;
        }

        // Backtracking line search (Armijo condition)
        let directional_deriv: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

        if directional_deriv >= 0.0 {
            // Not a descent direction — reset to steepest descent
            direction.copy_from_slice(&grad);
            for d in direction.iter_mut() {
                *d = -*d;
            }
        }

        let mut step = 1.0;
        let mut ls_ok = false;

        for _ in 0..config.max_linesearch {
            for i in 0..n {
                x_new[i] = x[i] + step * direction[i];
            }
            let fx_new = eval(&x_new);

            if fx_new <= fx + config.armijo_c * step * directional_deriv {
                fx = fx_new;
                x.copy_from_slice(&x_new);
                ls_ok = true;
                break;
            }
            step *= config.backtrack_rho;
        }

        if !ls_ok {
            // Line search failed — accept last point
            break;
        }

        if fx < best_fx {
            best_x.copy_from_slice(&x);
            best_fx = fx;
        }

        // Compute new gradient
        let new_grad = finite_difference_gradient(&eval, &x, config.fd_step);

        // Check convergence
        let grad_norm: f64 = new_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tolerance {
            converged = true;
            grad = new_grad;
            break;
        }

        // Update L-BFGS Hessian approximation
        lbfgs_state.update_hessian(&new_grad, &x);
        grad = new_grad;
    }

    // Transform back to constrained space
    let optimal_point = if has_bounds {
        to_constrained(&best_x, bounds.unwrap())
    } else {
        best_x
    };

    NelderMeadResult {
        optimal_point,
        optimal_value: best_fx,
        iterations,
        converged,
    }
}

/// Compute finite-difference gradient.
fn finite_difference_gradient(f: &dyn Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();

    for i in 0..n {
        let hi = h * (1.0 + x[i].abs()); // relative step
        x_plus[i] = x[i] + hi;
        let f_plus = f(&x_plus);
        x_plus[i] = x[i] - hi;
        let f_minus = f(&x_plus);
        x_plus[i] = x[i]; // restore

        grad[i] = (f_plus - f_minus) / (2.0 * hi);
        if !grad[i].is_finite() {
            grad[i] = 0.0;
        }
    }

    grad
}

/// Transform from constrained to unconstrained space using atanh.
fn to_unconstrained(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    x.iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lo, hi))| {
            if lo.is_infinite() && hi.is_infinite() {
                xi
            } else if lo.is_infinite() {
                // Upper bound only: x = hi - exp(u)
                (hi - xi).max(1e-10).ln()
            } else if hi.is_infinite() {
                // Lower bound only: x = lo + exp(u)
                (xi - lo).max(1e-10).ln()
            } else {
                // Both bounds: x = lo + (hi-lo) * sigmoid(u)
                let range = hi - lo;
                let normalized = ((xi - lo) / range).clamp(0.001, 0.999);
                (normalized / (1.0 - normalized)).ln() // logit
            }
        })
        .collect()
}

/// Transform from unconstrained to constrained space using sigmoid/exp.
fn to_constrained(u: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    u.iter()
        .zip(bounds.iter())
        .map(|(&ui, &(lo, hi))| {
            if lo.is_infinite() && hi.is_infinite() {
                ui
            } else if lo.is_infinite() {
                hi - ui.exp()
            } else if hi.is_infinite() {
                lo + ui.exp()
            } else {
                let range = hi - lo;
                let sigmoid = 1.0 / (1.0 + (-ui).exp());
                lo + range * sigmoid
            }
        })
        .collect()
}
