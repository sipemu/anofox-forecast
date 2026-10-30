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
/// `NelderMead`. Falls back to [`super::nelder_mead_legacy`] when the point is
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
