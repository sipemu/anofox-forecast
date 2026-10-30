//! Evaluation backend for `utils::optimization` backed by the `basin` crate
//! (GH #234, step 1).
//!
//! This module exists purely to measure whether `basin` 3.1.0 can replace the
//! hand-rolled Nelder-Mead / L-BFGS optimizers in this crate, with the same
//! public signatures. It is compiled only with the `basin` cargo feature, and
//! the public `nelder_mead` / `lbfgs_optimize` functions route to it only when
//! the additional compiler cfg `anofox_basin_optimizer` is also set (see
//! `super`'s module-level routing). Every basin solver call falls back to the
//! legacy implementation on invalid configuration or a solver error, so a
//! basin build can never panic where the legacy path would simply run.

use super::{NelderMeadConfig, NelderMeadResult};

/// Borrowing adapter from this crate's `Fn(&[f64]) -> f64` objective closures
/// to basin's [`basin::CostFunction`] trait. Keeps a reference (rather than
/// taking ownership) so the same closure can be reused for the legacy
/// fallback if the basin solver run fails.
struct Objective<'a, F: Fn(&[f64]) -> f64> {
    f: &'a F,
}

impl<F: Fn(&[f64]) -> f64> basin::CostFunction for Objective<'_, F> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(super::sanitize_objective((self.f)(x)))
    }
}

/// Whether the four Nelder-Mead coefficients and the iteration/tolerance
/// settings are inside basin's admissible ranges. basin's constructors
/// `assert!` on violations (threat T-go2-01); checking first lets the caller
/// fall back to the legacy optimizer instead of panicking.
fn nelder_mead_config_valid(config: &NelderMeadConfig) -> bool {
    config.alpha > 0.0
        && config.gamma > 1.0
        && config.rho > 0.0
        && config.rho < 1.0
        && config.sigma > 0.0
        && config.sigma < 1.0
        && config.max_iter > 0
        && config.tolerance.is_finite()
        && config.tolerance >= 0.0
}

/// Normalize `bounds` to exactly `n` entries. Legacy tolerates bounds shorter
/// than the point (ignoring the missing coordinates); basin requires an exact
/// length, so missing entries are padded with `(-inf, inf)`, which basin
/// accepts as "no bound" markers.
fn normalize_bounds(bounds: &[(f64, f64)], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lower = vec![f64::NEG_INFINITY; n];
    let mut upper = vec![f64::INFINITY; n];
    for i in 0..n.min(bounds.len()) {
        lower[i] = bounds[i].0;
        upper[i] = bounds[i].1;
    }
    (lower, upper)
}

/// Build the initial simplex exactly as the legacy Nelder-Mead does: vertex 0
/// is `initial` clamped, and vertex `i+1` is `initial` with coordinate `i`
/// nudged by a relative (or absolute, near zero) step, then clamped.
fn build_initial_simplex(
    initial: &[f64],
    bounds: Option<&[(f64, f64)]>,
    config: &NelderMeadConfig,
) -> Vec<Vec<f64>> {
    let n = initial.len();
    let mut vertices = Vec::with_capacity(n + 1);

    let mut v0 = initial.to_vec();
    super::apply_bounds_in_place(&mut v0, bounds);
    vertices.push(v0);

    for i in 0..n {
        let mut vi = initial.to_vec();
        let step = if initial[i].abs() > 1e-10 {
            config.initial_step * initial[i].abs()
        } else {
            config.initial_step
        };
        vi[i] += step;
        super::apply_bounds_in_place(&mut vi, bounds);
        vertices.push(vi);
    }

    vertices
}

/// Drive a basin Nelder-Mead `Executor` to completion, reproducing the
/// legacy simplex-diameter convergence test (the second clause of legacy's
/// `check_convergence`) through `stop_when`, and the legacy stagnation
/// counter through `no_objective_improvement`. Generic over the problem `P`
/// and solver `So` so the bounded (`Constrained` + `Projected`) and unbounded
/// code paths share this logic instead of duplicating it.
fn run_nelder_mead_executor<P, So>(
    problem: P,
    solver: So,
    vertices: Vec<Vec<f64>>,
    config: &NelderMeadConfig,
) -> Result<basin::OptimizationResult<basin::SimplexProgress<Vec<f64>, f64>>, So::Error>
where
    So: basin::Solver<P, basin::SimplexProgress<Vec<f64>, f64>>,
{
    let tolerance = config.tolerance;
    let tolerance_sq = tolerance * tolerance;

    let mut executor = basin::Executor::new(
        problem,
        solver,
        basin::SimplexProgress::from_simplex(vertices),
    )
    .max_iter(config.max_iter as u64)
    .stop_when(move |state: &basin::SimplexProgress<Vec<f64>, f64>| {
        let (vertices, _costs) = state.evaluated_vertices()?;
        if vertices.len() < 2 {
            return None;
        }
        // basin keeps the simplex sorted best-first, so the worst vertex is
        // always last; mirror legacy's `compute_centroid_into`, which
        // excludes the worst vertex from the centroid.
        let worst = vertices.len() - 1;
        let dim = vertices[0].len();
        let mut centroid = vec![0.0f64; dim];
        for (i, v) in vertices.iter().enumerate() {
            if i != worst {
                for (c, x) in centroid.iter_mut().zip(v.iter()) {
                    *c += x;
                }
            }
        }
        let count = (vertices.len() - 1) as f64;
        for c in centroid.iter_mut() {
            *c /= count;
        }
        let within_tolerance = vertices
            .iter()
            .all(|v| super::distance_sq(v, &centroid) < tolerance_sq);
        within_tolerance.then(|| basin::ApplicationStop::new("simplex_size"))
    });

    if config.stagnation_window > 0 {
        executor = executor.no_objective_improvement(config.stagnation_window as u64, tolerance);
    }

    executor.run()
}

/// Translate a completed basin run into this crate's `NelderMeadResult`.
///
/// `converged` is true when basin reports a numerical convergence predicate
/// (the cost-tolerance test configured in the caller), the application stop
/// from [`run_nelder_mead_executor`]'s `stop_when` (the simplex-diameter
/// test), or the stagnation stop. `iterations` adds one when converged,
/// because legacy increments its loop counter before the check that stops
/// it, while basin's `iter()` counts completed iterations at the boundary
/// where termination was detected (before that increment).
fn nelder_mead_result_from_basin(
    result: &basin::OptimizationResult<basin::SimplexProgress<Vec<f64>, f64>>,
    config: &NelderMeadConfig,
) -> NelderMeadResult {
    let converged = result.report.termination.is_converged()
        || matches!(
            result.report.termination,
            basin::Termination::Application(_)
        )
        || result.report.code() == basin::TerminationCode::NoImprovement;

    let iterations = (result.iter() as usize + usize::from(converged)).min(config.max_iter);

    NelderMeadResult {
        optimal_point: result.best_param().clone(),
        optimal_value: result.best_cost(),
        iterations,
        converged,
    }
}

/// Basin-backed Nelder-Mead, matching [`super::nelder_mead`]'s signature.
///
/// Routes bounded problems through basin's `Projected` (clamping) variant via
/// a `Constrained` box-bounds problem, and unbounded problems through plain
/// `NelderMead`. Falls back to `nelder_mead_legacy` when the point is
/// empty, the configuration is outside basin's admissible coefficient ranges
/// (threat T-go2-01), or the basin solver run itself returns `Err` (e.g. an
/// invalid box under basin's stricter bounds validation).
pub fn nelder_mead_basin<F>(
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
        return super::nelder_mead_legacy(objective, initial, bounds, config);
    }

    if !nelder_mead_config_valid(&config) {
        return super::nelder_mead_legacy(objective, initial, bounds, config);
    }

    let vertices = build_initial_simplex(initial, bounds, &config);
    let obj = Objective { f: &objective };

    let run_result = match bounds {
        Some(b) => {
            let (lower, upper) = normalize_bounds(b, n);
            let constraints = basin::ConstraintSet::new()
                .with_bounds(basin::BoxBounds::new(Some(lower), Some(upper)));
            let problem = basin::Constrained::new(obj, constraints);
            let solver = basin::NelderMead::with_params(
                config.alpha,
                config.gamma,
                config.rho,
                config.sigma,
            )
            .with_absolute_simplex_size_tolerance(None)
            .with_relative_simplex_size_tolerance(None)
            .with_relative_simplex_cost_tolerance(None)
            .with_absolute_simplex_cost_tolerance(config.tolerance)
            .projected();
            run_nelder_mead_executor(problem, solver, vertices, &config).ok()
        }
        None => {
            let solver = basin::NelderMead::with_params(
                config.alpha,
                config.gamma,
                config.rho,
                config.sigma,
            )
            .with_absolute_simplex_size_tolerance(None)
            .with_relative_simplex_size_tolerance(None)
            .with_relative_simplex_cost_tolerance(None)
            .with_absolute_simplex_cost_tolerance(config.tolerance);
            run_nelder_mead_executor(obj, solver, vertices, &config).ok()
        }
    };

    match run_result {
        Some(result) => nelder_mead_result_from_basin(&result, &config),
        // basin's `run()` returned `Err` (e.g. an invalid box under its
        // stricter bounds validation); preserve legacy behavior rather than
        // propagate a panic or a silently wrong result.
        None => super::nelder_mead_legacy(objective, initial, bounds, config),
    }
}

// =========================================================================
// L-BFGS / L-BFGS-B backend with bound-aware finite-difference gradients
// =========================================================================

use super::LbfgsConfig;

/// Objective + bound-aware finite-difference gradient adapter for
/// [`lbfgs_optimize_basin`]. Unlike [`Objective`], this always carries a
/// `lower`/`upper` pair (filled with `(-inf, inf)` for the unbounded path),
/// so the central-difference probe can detect when it would step outside a
/// finite box and switch to a one-sided difference without ever leaving it.
struct GradientObjective<'a, F: Fn(&[f64]) -> f64> {
    f: &'a F,
    lower: &'a [f64],
    upper: &'a [f64],
    fd_step: f64,
}

impl<F: Fn(&[f64]) -> f64> basin::CostFunction for GradientObjective<'_, F> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(super::sanitize_objective((self.f)(x)))
    }
}

impl<F: Fn(&[f64]) -> f64> basin::Gradient for GradientObjective<'_, F> {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        let n = x.len();
        let mut grad = vec![0.0; n];
        let mut probe = x.clone();

        for i in 0..n {
            let h = self.fd_step * (1.0 + x[i].abs());
            // `cost()` sanitizes NaN/Inf to f64::MAX for basin's own calls,
            // but the Gradient impl below reads `self.f` directly, so every
            // probe here re-sanitizes independently (same as legacy's
            // `finite_difference_gradient`).
            let can_step_up = x[i] + h <= self.upper[i];
            let can_step_down = x[i] - h >= self.lower[i];

            let g = if can_step_up && can_step_down {
                probe[i] = x[i] + h;
                let f_plus = super::sanitize_objective((self.f)(&probe));
                probe[i] = x[i] - h;
                let f_minus = super::sanitize_objective((self.f)(&probe));
                probe[i] = x[i];
                (f_plus - f_minus) / (2.0 * h)
            } else if can_step_down {
                // Upper bound would be exceeded: backward difference.
                let f_x = super::sanitize_objective((self.f)(x));
                probe[i] = x[i] - h;
                let f_minus = super::sanitize_objective((self.f)(&probe));
                probe[i] = x[i];
                (f_x - f_minus) / h
            } else if can_step_up {
                // Lower bound would be exceeded: forward difference.
                probe[i] = x[i] + h;
                let f_plus = super::sanitize_objective((self.f)(&probe));
                probe[i] = x[i];
                let f_x = super::sanitize_objective((self.f)(x));
                (f_plus - f_x) / h
            } else {
                // Box narrower than the step on both sides; no safe probe.
                0.0
            };

            grad[i] = if g.is_finite() { g } else { 0.0 };
        }

        Ok(grad)
    }
}

/// Whether the L-BFGS configuration is inside basin's admissible ranges.
/// basin's constructors `assert!` on violations (threat T-go2-01); checking
/// first lets the caller fall back to the legacy optimizer.
fn lbfgs_config_valid(config: &LbfgsConfig) -> bool {
    config.memory_size >= 1
        && config.backtrack_rho > 0.0
        && config.backtrack_rho < 1.0
        && config.armijo_c > 0.0
        && config.armijo_c < 1.0
        && config.max_linesearch >= 1
        && config.max_iter > 0
        && config.tolerance.is_finite()
        && config.tolerance >= 0.0
        && config.fd_step.is_finite()
        && config.fd_step > 0.0
}

/// Basin-backed L-BFGS / L-BFGS-B, matching [`super::lbfgs_optimize`]'s
/// signature.
///
/// Bounded problems route through basin's native L-BFGS-B (box constraints
/// handled directly by the Cauchy-point/subspace-minimization machinery)
/// instead of legacy's logit/exp reparametrization — the largest semantic
/// difference between the two backends; see `docs/audits/basin-optimizer-parity.md`.
/// Unbounded problems route through basin's unconstrained L-BFGS (two-loop
/// recursion). Both share `GradientObjective`'s bound-aware finite-difference
/// gradient. Falls back to `lbfgs_optimize_legacy` when the point is
/// empty, the configuration is outside basin's admissible ranges (threat
/// T-go2-01), or the basin solver run itself returns `Err`.
pub fn lbfgs_optimize_basin<F>(
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
        return super::lbfgs_optimize_legacy(objective, initial, bounds, config);
    }

    if !lbfgs_config_valid(&config) {
        return super::lbfgs_optimize_legacy(objective, initial, bounds, config);
    }

    let (lower, upper) = match bounds {
        Some(b) => normalize_bounds(b, n),
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };

    let run_result = match bounds {
        Some(_) => {
            let obj = GradientObjective {
                f: &objective,
                lower: &lower,
                upper: &upper,
                fd_step: config.fd_step,
            };
            let constraints = basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
                Some(lower.clone()),
                Some(upper.clone()),
            ));
            let problem = basin::Constrained::new(obj, constraints);
            let initial_point: Vec<f64> = initial
                .iter()
                .zip(lower.iter().zip(upper.iter()))
                .map(|(&x, (&lo, &hi))| x.clamp(lo, hi))
                .collect();
            let line_search = basin::Backtracking::new()
                .rho(config.backtrack_rho)
                .with_sufficient_decrease_coefficient(config.armijo_c)
                .max_iter(config.max_linesearch as u32);
            let solver = basin::Lbfgs::with_line_search(line_search)
                .with_m_capacity(config.memory_size)
                .with_absolute_projected_gradient_tolerance(config.tolerance);
            basin::Executor::new(problem, solver, basin::FirstOrderState::new(initial_point))
                .max_iter(config.max_iter as u64)
                .run()
                .ok()
        }
        None => {
            let obj = GradientObjective {
                f: &objective,
                lower: &lower,
                upper: &upper,
                fd_step: config.fd_step,
            };
            let line_search = basin::Backtracking::new()
                .rho(config.backtrack_rho)
                .with_sufficient_decrease_coefficient(config.armijo_c)
                .max_iter(config.max_linesearch as u32);
            let solver = basin::Lbfgs::with_line_search(line_search)
                .with_m_capacity(config.memory_size)
                .unbounded()
                .with_absolute_gradient_tolerance(config.tolerance);
            basin::Executor::new(obj, solver, basin::FirstOrderState::new(initial.to_vec()))
                .max_iter(config.max_iter as u64)
                .run()
                .ok()
        }
    };

    match run_result {
        Some(result) => NelderMeadResult {
            optimal_point: result.best_param().clone(),
            optimal_value: result.best_cost(),
            iterations: result.iter() as usize,
            converged: result.report.termination.is_converged(),
        },
        // basin's `run()` returned `Err`; preserve legacy behavior rather
        // than propagate a panic or a silently wrong result.
        None => super::lbfgs_optimize_legacy(objective, initial, bounds, config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn nelder_mead_basin_quadratic_2d() {
        let result = nelder_mead_basin(
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
    fn nelder_mead_basin_rosenbrock() {
        let config = NelderMeadConfig {
            max_iter: 5000,
            tolerance: 1e-10,
            ..Default::default()
        };

        let result = nelder_mead_basin(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
            &[0.0, 0.0],
            None,
            config,
        );

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.optimal_point[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn nelder_mead_basin_1d() {
        let result = nelder_mead_basin(
            |x| (x[0] - 5.0).powi(2),
            &[0.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 5.0, epsilon = 0.1);
    }

    #[test]
    fn nelder_mead_basin_with_bounds() {
        let result = nelder_mead_basin(
            |x| (x[0] - 5.0).powi(2),
            &[1.0],
            Some(&[(0.0, 3.0)]),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(result.optimal_point[0], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_basin_with_bounds_2d() {
        let result = nelder_mead_basin(
            |x| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.5, 0.5],
            Some(&[(0.0, 1.0), (0.0, 1.0)]),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_point[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_basin_exponential_smoothing_alpha() {
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

        let result = nelder_mead_basin(
            sse,
            &[0.5],
            Some(&[(0.01, 0.99)]),
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert!(result.optimal_point[0] > 0.01 && result.optimal_point[0] < 0.99);
    }

    #[test]
    fn nelder_mead_basin_empty_initial() {
        let result = nelder_mead_basin(|_| 0.0, &[], None, NelderMeadConfig::default());

        assert!(!result.converged);
        assert!(result.optimal_value.is_nan());
    }

    #[test]
    fn nelder_mead_basin_already_optimal() {
        let result = nelder_mead_basin(
            |x| (x[0] - 2.0).powi(2),
            &[2.0],
            None,
            NelderMeadConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_basin_3d() {
        let result = nelder_mead_basin(
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
    fn nelder_mead_basin_config_custom() {
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

        let result = nelder_mead_basin(|x| (x[0] - 1.0).powi(2), &[0.0], None, config);

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 0.01);
    }

    #[test]
    fn nelder_mead_basin_nan_objective_handled() {
        let result = nelder_mead_basin(
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

        assert!(result.converged);
        assert!(result.optimal_value.is_finite());
        assert_relative_eq!(result.optimal_point[0], 3.0, epsilon = 0.1);
    }

    #[test]
    fn nelder_mead_basin_inf_objective_handled() {
        let result = nelder_mead_basin(
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

    #[test]
    fn nelder_mead_basin_partial_bounds_no_panic() {
        // 2-D point, one bound pair: coordinate 0 is constrained, coordinate
        // 1 is padded with (-inf, inf) by `normalize_bounds`.
        let result = nelder_mead_basin(
            |x| (x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2),
            &[1.0, 1.0],
            Some(&[(0.0, 3.0)]),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(result.optimal_point[0], 3.0, epsilon = 1e-3);
        assert_relative_eq!(result.optimal_point[1], 5.0, epsilon = 1e-2);
    }

    #[test]
    fn nelder_mead_basin_stagnation_stops_early() {
        let config = NelderMeadConfig {
            max_iter: 10000,
            tolerance: 1e-3,
            stagnation_window: 20,
            ..Default::default()
        };

        let result = nelder_mead_basin(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
            &[-1.2, 1.0],
            None,
            config,
        );

        assert!(result.converged);
        assert!(result.iterations < config.max_iter);
    }

    #[test]
    fn nelder_mead_basin_matches_legacy_quadratic() {
        let basin_result = nelder_mead_basin(
            |x| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            None,
            NelderMeadConfig::default(),
        );
        let legacy_result = super::super::nelder_mead_legacy(
            |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            None,
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(
            basin_result.optimal_value,
            legacy_result.optimal_value,
            epsilon = 1e-6
        );
        assert_eq!(basin_result.converged, legacy_result.converged);
    }

    #[test]
    fn nelder_mead_basin_matches_legacy_bounded_2d() {
        let bounds = [(0.0, 1.0), (0.0, 1.0)];
        let basin_result = nelder_mead_basin(
            |x| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.5, 0.5],
            Some(&bounds),
            NelderMeadConfig::default(),
        );
        let legacy_result = super::super::nelder_mead_legacy(
            |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.5, 0.5],
            Some(&bounds),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(
            basin_result.optimal_value,
            legacy_result.optimal_value,
            epsilon = 1e-6
        );
        assert_eq!(basin_result.converged, legacy_result.converged);
    }

    #[test]
    fn nelder_mead_basin_matches_legacy_smoothing_alpha() {
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

        let basin_result = nelder_mead_basin(
            sse,
            &[0.5],
            Some(&[(0.01, 0.99)]),
            NelderMeadConfig::default(),
        );
        let legacy_result = super::super::nelder_mead_legacy(
            sse,
            &[0.5],
            Some(&[(0.01, 0.99)]),
            NelderMeadConfig::default(),
        );

        assert_relative_eq!(
            basin_result.optimal_value,
            legacy_result.optimal_value,
            epsilon = 1e-6
        );
        assert_eq!(basin_result.converged, legacy_result.converged);
    }

    #[test]
    fn nelder_mead_basin_invalid_config_falls_back_to_legacy() {
        // gamma must be > 1 (expansion coefficient); 0.5 is out of range and
        // would panic inside basin's `NelderMead::with_params`.
        let config = NelderMeadConfig {
            gamma: 0.5,
            ..Default::default()
        };

        let result = nelder_mead_basin(|x| (x[0] - 4.0).powi(2), &[0.0], None, config);

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 4.0, epsilon = 1e-3);
    }
}

#[cfg(test)]
mod lbfgs_tests {
    use super::*;
    use crate::utils::optimization::LbfgsConfig;
    use approx::assert_relative_eq;

    #[test]
    fn lbfgs_basin_unbounded_quadratic() {
        let result = lbfgs_optimize_basin(
            |x| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            None,
            LbfgsConfig::default(),
        );

        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 5.0, epsilon = 1e-4);
        assert_relative_eq!(result.optimal_point[1], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn lbfgs_basin_unbounded_rosenbrock() {
        // Deviation from the plan's `<behavior>` spec: `max_iter: 500` does
        // not converge Rosenbrock from (-1.2, 1.0) under either backend with
        // finite-difference gradients (legacy needs ~786 iterations, basin
        // ~675, both measured directly); 500 was an incorrect assumption in
        // the plan, not a basin-specific shortfall. Using a budget that
        // actually lets both backends reach the minimum, documented in
        // docs/audits/basin-optimizer-parity.md.
        let config = LbfgsConfig {
            max_iter: 5000,
            ..Default::default()
        };
        let result = lbfgs_optimize_basin(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
            &[-1.2, 1.0],
            None,
            config,
        );

        assert_relative_eq!(result.optimal_point[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.optimal_point[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn lbfgs_basin_bounded_corner() {
        // ARIMA's AR-coefficient bound shape: a tight box on one coordinate,
        // unbounded on the other. The unconstrained minimum (5, 3) lies
        // outside the box on x, so the solver should settle at the x
        // boundary and the unconstrained optimum on y.
        let result = lbfgs_optimize_basin(
            |x| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            Some(&[(-0.995, 0.995), (f64::NEG_INFINITY, f64::INFINITY)]),
            LbfgsConfig::default(),
        );

        assert_relative_eq!(result.optimal_point[0], 0.995, epsilon = 1e-6);
        assert_relative_eq!(result.optimal_point[1], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn lbfgs_basin_nan_objective_handled() {
        let result = lbfgs_optimize_basin(
            |x| {
                if x[0] < 0.0 {
                    f64::NAN
                } else {
                    (x[0] - 2.0).powi(2)
                }
            },
            &[1.0],
            None,
            LbfgsConfig::default(),
        );

        assert!(result.optimal_value.is_finite());
    }

    #[test]
    fn lbfgs_basin_empty_initial() {
        let result = lbfgs_optimize_basin(|_| 0.0, &[], None, LbfgsConfig::default());

        assert!(!result.converged);
        assert!(result.optimal_value.is_nan());
    }

    #[test]
    fn lbfgs_basin_invalid_config_falls_back_to_legacy() {
        // `backtrack_rho` outside (0, 1) is invalid for basin's `Backtracking`
        // line search (threat T-go2-01). Deviation from the plan's
        // `<behavior>` spec, found during this task: legacy does not
        // actually recover a usable minimum for this input either — an
        // ever-growing step (rho > 1) never satisfies the Armijo condition,
        // so legacy's own line search fails on iteration 1 and breaks
        // immediately (measured directly: `[0.0]`, not converged). The
        // fallback therefore cannot "still find the quadratic minimum"; what
        // it must do, and does, is reproduce legacy's own (poor) result
        // exactly rather than basin's, without panicking.
        //
        // `memory_size: 0` is NOT exercised here for the same reason, one
        // level further: `lbfgs_config_valid` correctly routes it to
        // `lbfgs_optimize_legacy` to avoid basin's own panic, but the
        // `lbfgs` crate legacy itself depends on panics
        // (`assertion failed: buffer_size > 0`) on `memory_size: 0`
        // regardless of basin — a pre-existing legacy limitation, not
        // introduced by this backend. Both are recorded in
        // docs/audits/basin-optimizer-parity.md.
        let config = LbfgsConfig {
            backtrack_rho: 1.5,
            ..Default::default()
        };
        let basin_result = lbfgs_optimize_basin(|x| (x[0] - 5.0).powi(2), &[0.0], None, config);
        let legacy_result = super::super::lbfgs_optimize_legacy(
            |x: &[f64]| (x[0] - 5.0).powi(2),
            &[0.0],
            None,
            config,
        );

        assert!(basin_result.optimal_value.is_finite());
        assert_eq!(basin_result.optimal_point, legacy_result.optimal_point);
        assert_eq!(basin_result.optimal_value, legacy_result.optimal_value);
        assert_eq!(basin_result.converged, legacy_result.converged);
    }

    #[test]
    fn lbfgs_basin_matches_legacy_unbounded_quadratic() {
        let basin_result = lbfgs_optimize_basin(
            |x| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            None,
            LbfgsConfig::default(),
        );
        let legacy_result = super::super::lbfgs_optimize_legacy(
            |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            None,
            LbfgsConfig::default(),
        );

        assert_relative_eq!(
            basin_result.optimal_value,
            legacy_result.optimal_value,
            epsilon = 1e-6
        );
    }

    #[test]
    fn lbfgs_basin_matches_legacy_bounded_corner() {
        let bounds = [(-0.995, 0.995), (f64::NEG_INFINITY, f64::INFINITY)];
        let basin_result = lbfgs_optimize_basin(
            |x| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            Some(&bounds),
            LbfgsConfig::default(),
        );
        let legacy_result = super::super::lbfgs_optimize_legacy(
            |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] - 3.0).powi(2),
            &[0.0, 0.0],
            Some(&bounds),
            LbfgsConfig::default(),
        );

        assert_relative_eq!(
            basin_result.optimal_value,
            legacy_result.optimal_value,
            epsilon = 1e-6
        );
    }
}

// =========================================================================
// BOBYQA for box-bounded smoothing parameters (GH #234, step 2)
// =========================================================================

/// Objective values at or above this are treated as "inadmissible" markers
/// (the ETS / Holt-Winters objectives return `f64::MAX` outside their
/// admissible region). BOBYQA interpolates a quadratic model through the
/// sampled values, so a raw `f64::MAX` would overflow the model; such values
/// are replaced by a finite penalty instead (see [`ScaledObjective`]).
const INADMISSIBLE: f64 = 1e100;

/// BOBYQA adapter: works in the unit cube `u in [0, 1]^n` (mapped to the
/// caller's box by `x = lower + u * (upper - lower)`), so one trust-region
/// radius fits every coordinate even when the box mixes `[0, 1]` smoothing
/// parameters with data-scaled initial-state bounds, and replaces
/// inadmissible objective values by `penalty`.
struct ScaledObjective<'a, F: Fn(&[f64]) -> f64> {
    f: &'a F,
    lower: Vec<f64>,
    width: Vec<f64>,
    penalty: f64,
}

impl<F: Fn(&[f64]) -> f64> ScaledObjective<'_, F> {
    fn to_x(&self, u: &[f64]) -> Vec<f64> {
        u.iter()
            .zip(self.lower.iter().zip(&self.width))
            .map(|(u, (lo, w))| lo + u.clamp(0.0, 1.0) * w)
            .collect()
    }
}

impl<F: Fn(&[f64]) -> f64> basin::CostFunction for ScaledObjective<'_, F> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, u: &Vec<f64>) -> Result<f64, Self::Error> {
        let v = (self.f)(&self.to_x(u));
        Ok(if v.is_finite() && v < INADMISSIBLE {
            v
        } else {
            self.penalty
        })
    }
}

/// Basin BOBYQA (Powell 2009) for a fully box-bounded problem, returning a
/// [`NelderMeadResult`] so callers can swap it in for [`super::nelder_mead`].
///
/// Requirements (otherwise the call falls back to [`super::nelder_mead`] with
/// the same arguments): `bounds` present, one finite `(lower, upper)` pair
/// per coordinate with `upper > lower`, a non-empty start point, and a
/// finite, admissible objective value at the (clamped) start point.
///
/// Termination mapping from [`NelderMeadConfig`]: the trust region shrinks
/// from `rho_beg = 0.1` to `rho_end = 1e-6` in unit-cube coordinates (i.e.
/// relative to each coordinate's box width); the evaluation budget is
/// `max_iter * (n + 1)` objective calls, the worst-case budget of the
/// Nelder-Mead iterations it replaces (a shrink step costs `n + 1` calls).
/// `optimal_value` is the raw objective at the returned point (never the
/// internal penalty).
pub fn bobyqa_basin<F>(
    objective: F,
    initial: &[f64],
    bounds: Option<&[(f64, f64)]>,
    config: NelderMeadConfig,
) -> NelderMeadResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = initial.len();
    let usable = bounds.filter(|b| {
        n > 0
            && b.len() == n
            && b.iter()
                .all(|&(lo, hi)| lo.is_finite() && hi.is_finite() && hi > lo)
    });
    let Some(b) = usable else {
        return super::nelder_mead(objective, initial, bounds, config);
    };
    if config.max_iter == 0 {
        return super::nelder_mead(objective, initial, bounds, config);
    }

    let lower: Vec<f64> = b.iter().map(|&(lo, _)| lo).collect();
    let width: Vec<f64> = b.iter().map(|&(lo, hi)| hi - lo).collect();
    let u0: Vec<f64> = initial
        .iter()
        .zip(lower.iter().zip(&width))
        .map(|(x, (lo, w))| ((x - lo) / w).clamp(0.0, 1.0))
        .collect();

    let mut scaled = ScaledObjective {
        f: &objective,
        lower,
        width,
        penalty: 0.0,
    };
    let f0 = objective(&scaled.to_x(&u0));
    if !f0.is_finite() || f0 >= INADMISSIBLE {
        return super::nelder_mead(objective, initial, bounds, config);
    }
    scaled.penalty = f0 + 1e3 * (1.0 + f0.abs());

    let constraints = basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
        Some(vec![0.0; n]),
        Some(vec![1.0; n]),
    ));
    let problem = basin::Constrained::new(scaled, constraints);
    let solver = basin::Bobyqa::new()
        .with_initial_radius(0.1)
        .with_final_radius(1e-6);
    let max_evals = (config.max_iter as u64).saturating_mul(n as u64 + 1);
    let run = basin::Executor::new(problem, solver, basin::PointState::new(u0))
        .max_cost_evals(max_evals)
        .run();

    match run {
        Ok(result) => {
            let u = result.best_param();
            let x: Vec<f64> = u
                .iter()
                .zip(b)
                .map(|(u, &(lo, hi))| lo + u.clamp(0.0, 1.0) * (hi - lo))
                .collect();
            let value = objective(&x);
            NelderMeadResult {
                optimal_point: x,
                optimal_value: super::sanitize_objective(value),
                iterations: result.iter() as usize,
                converged: result.report.termination.is_converged(),
            }
        }
        Err(_) => super::nelder_mead(objective, initial, bounds, config),
    }
}

#[cfg(test)]
mod bobyqa_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bobyqa_basin_bounded_quadratic() {
        let result = bobyqa_basin(
            |x| (x[0] - 0.3).powi(2) + (x[1] - 0.7).powi(2),
            &[0.5, 0.5],
            Some(&[(0.0, 1.0), (0.0, 1.0)]),
            NelderMeadConfig::default(),
        );
        assert!(result.converged);
        assert_relative_eq!(result.optimal_point[0], 0.3, epsilon = 1e-5);
        assert_relative_eq!(result.optimal_point[1], 0.7, epsilon = 1e-5);
    }

    #[test]
    fn bobyqa_basin_active_bound_and_mixed_scales() {
        // Optimum outside the box in x[0] (active lower bound) and a second
        // coordinate on a data-sized scale, as in ETS's joint state bounds.
        let result = bobyqa_basin(
            |x| (x[0] + 1.0).powi(2) + ((x[1] - 250.0) / 100.0).powi(2),
            &[0.5, 0.0],
            Some(&[(0.0, 1.0), (-500.0, 500.0)]),
            NelderMeadConfig::default(),
        );
        assert_relative_eq!(result.optimal_point[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(result.optimal_point[1], 250.0, epsilon = 1e-2);
        assert_relative_eq!(result.optimal_value, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn bobyqa_basin_inadmissible_region_reports_raw_value() {
        // f64::MAX marks an inadmissible region (x[0] > 0.8), as the ETS
        // objectives do; the minimum is inside the admissible part.
        let result = bobyqa_basin(
            |x| {
                if x[0] > 0.8 {
                    f64::MAX
                } else {
                    (x[0] - 0.75).powi(2)
                }
            },
            &[0.2],
            Some(&[(0.0, 1.0)]),
            NelderMeadConfig::default(),
        );
        assert!(result.optimal_point[0] <= 0.8);
        assert_relative_eq!(result.optimal_point[0], 0.75, epsilon = 1e-4);
        assert!(result.optimal_value < 1e-6);
    }

    #[test]
    fn bobyqa_basin_falls_back_without_finite_box() {
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        for bounds in [
            None,
            Some(&[(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)][..]),
        ] {
            let a = bobyqa_basin(f, &[0.0, 0.0], bounds, NelderMeadConfig::default());
            let b = crate::utils::optimization::nelder_mead(
                f,
                &[0.0, 0.0],
                bounds,
                NelderMeadConfig::default(),
            );
            assert_eq!(a.optimal_point, b.optimal_point);
        }
        let empty = bobyqa_basin(f, &[], Some(&[]), NelderMeadConfig::default());
        assert!(empty.optimal_point.is_empty());
    }
}
