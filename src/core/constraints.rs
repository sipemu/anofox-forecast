//! Forecast constraints for enforcing bounds, non-negativity, and rounding.
//!
//! Constraints can be applied to forecasts to ensure predictions satisfy
//! domain-specific requirements such as non-negativity, bounded ranges,
//! or integer values.

use super::Forecast;

/// A constraint that can be applied to forecast values.
pub enum ForecastConstraint {
    /// Clamp all values to be >= 0.
    NonNegative,
    /// Clamp all values to be >= the given lower bound.
    LowerBound(f64),
    /// Clamp all values to be <= the given upper bound.
    UpperBound(f64),
    /// Clamp all values to be within [lower, upper].
    Bounds { lower: f64, upper: f64 },
    /// Round point forecasts to the nearest integer;
    /// floor lower intervals, ceil upper intervals.
    IntegerRound,
    /// Apply an arbitrary transformation function to each value.
    Custom(Box<dyn Fn(f64) -> f64 + Send + Sync>),
}

/// Applies constraints to [`Forecast`] values.
pub struct ConstrainedForecast;

impl ConstrainedForecast {
    /// Apply a sequence of constraints to a forecast, returning a new forecast
    /// with adjusted point predictions and intervals.
    ///
    /// Constraints are applied in the order they appear in the slice.
    pub fn apply(forecast: &Forecast, constraints: &[ForecastConstraint]) -> Forecast {
        let mut point: Vec<Vec<f64>> = forecast.point().to_vec();
        let mut lower: Option<Vec<Vec<f64>>> = forecast.lower().map(|l| l.to_vec());
        let mut upper: Option<Vec<Vec<f64>>> = forecast.upper().map(|u| u.to_vec());

        for constraint in constraints {
            match constraint {
                ForecastConstraint::NonNegative => {
                    apply_to_all(&mut point, |v| v.max(0.0));
                    apply_to_optional(&mut lower, |v| v.max(0.0));
                    apply_to_optional(&mut upper, |v| v.max(0.0));
                }
                ForecastConstraint::LowerBound(lb) => {
                    let lb = *lb;
                    apply_to_all(&mut point, |v| v.max(lb));
                    apply_to_optional(&mut lower, |v| v.max(lb));
                    apply_to_optional(&mut upper, |v| v.max(lb));
                }
                ForecastConstraint::UpperBound(ub) => {
                    let ub = *ub;
                    apply_to_all(&mut point, |v| v.min(ub));
                    apply_to_optional(&mut lower, |v| v.min(ub));
                    apply_to_optional(&mut upper, |v| v.min(ub));
                }
                ForecastConstraint::Bounds {
                    lower: lb,
                    upper: ub,
                } => {
                    let lb = *lb;
                    let ub = *ub;
                    apply_to_all(&mut point, |v| v.clamp(lb, ub));
                    apply_to_optional(&mut lower, |v| v.clamp(lb, ub));
                    apply_to_optional(&mut upper, |v| v.clamp(lb, ub));
                }
                ForecastConstraint::IntegerRound => {
                    apply_to_all(&mut point, |v| v.round());
                    apply_to_optional(&mut lower, |v| v.floor());
                    apply_to_optional(&mut upper, |v| v.ceil());
                }
                ForecastConstraint::Custom(f) => {
                    apply_to_all(&mut point, f);
                    apply_to_optional(&mut lower, f);
                    apply_to_optional(&mut upper, f);
                }
            }
        }

        rebuild_forecast(point, lower, upper)
    }
}

/// Apply a function to every value across all dimensions.
fn apply_to_all<F: Fn(f64) -> f64>(data: &mut [Vec<f64>], f: F) {
    for series in data.iter_mut() {
        for val in series.iter_mut() {
            *val = f(*val);
        }
    }
}

/// Apply a function to every value in an optional interval matrix.
fn apply_to_optional<F: Fn(f64) -> f64>(data: &mut Option<Vec<Vec<f64>>>, f: F) {
    if let Some(ref mut vecs) = data {
        apply_to_all(vecs, f);
    }
}

/// Rebuild a `Forecast` from raw point/lower/upper data.
fn rebuild_forecast(
    point: Vec<Vec<f64>>,
    lower: Option<Vec<Vec<f64>>>,
    upper: Option<Vec<Vec<f64>>>,
) -> Forecast {
    let dims = point.len();
    let mut forecast = Forecast::with_dimensions(dims);
    for (i, series) in point.into_iter().enumerate() {
        forecast.series_mut(i).extend(series);
    }
    if let Some(lower_vecs) = lower {
        for (i, series) in lower_vecs.into_iter().enumerate() {
            forecast.lower_series_mut(i).extend(series);
        }
    }
    if let Some(upper_vecs) = upper {
        for (i, series) in upper_vecs.into_iter().enumerate() {
            forecast.upper_series_mut(i).extend(series);
        }
    }
    forecast
}

// ---------------------------------------------------------------------------
// Convenience methods on Forecast
// ---------------------------------------------------------------------------

impl Forecast {
    /// Apply a set of constraints to this forecast, returning a new constrained forecast.
    pub fn constrain(&self, constraints: &[ForecastConstraint]) -> Forecast {
        ConstrainedForecast::apply(self, constraints)
    }

    /// Return a new forecast with all values clamped to be non-negative.
    pub fn non_negative(&self) -> Forecast {
        ConstrainedForecast::apply(self, &[ForecastConstraint::NonNegative])
    }

    /// Return a new forecast with all values clamped to `[lower, upper]`.
    pub fn clamp(&self, lower: f64, upper: f64) -> Forecast {
        ConstrainedForecast::apply(self, &[ForecastConstraint::Bounds { lower, upper }])
    }

    /// Return a new forecast with point values rounded to the nearest integer,
    /// lower intervals floored, and upper intervals ceiled.
    pub fn round_to_integer(&self) -> Forecast {
        ConstrainedForecast::apply(self, &[ForecastConstraint::IntegerRound])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_negative_clamps_negative_forecasts_to_zero() {
        let forecast = Forecast::from_values_with_intervals(
            vec![-3.0, 2.0, -1.0],
            vec![-5.0, 0.5, -2.0],
            vec![-1.0, 3.0, 0.5],
        );
        let result = forecast.non_negative();
        assert_eq!(result.primary(), &[0.0, 2.0, 0.0]);
        assert_eq!(result.lower_series(0).unwrap(), &[0.0, 0.5, 0.0]);
        assert_eq!(result.upper_series(0).unwrap(), &[0.0, 3.0, 0.5]);
    }

    #[test]
    fn bounds_enforcement_clamps_point_lower_and_upper() {
        let forecast = Forecast::from_values_with_intervals(
            vec![1.0, 5.0, 10.0, 15.0],
            vec![-1.0, 3.0, 8.0, 13.0],
            vec![3.0, 7.0, 12.0, 17.0],
        );
        let result = forecast.clamp(2.0, 12.0);
        assert_eq!(result.primary(), &[2.0, 5.0, 10.0, 12.0]);
        assert_eq!(result.lower_series(0).unwrap(), &[2.0, 3.0, 8.0, 12.0]);
        assert_eq!(result.upper_series(0).unwrap(), &[3.0, 7.0, 12.0, 12.0]);
    }

    #[test]
    fn integer_rounding_rounds_point_and_floor_ceil_intervals() {
        let forecast = Forecast::from_values_with_intervals(
            vec![2.3, 4.7, 6.5],
            vec![1.2, 3.8, 5.1],
            vec![3.9, 5.1, 7.8],
        );
        let result = forecast.round_to_integer();
        assert_eq!(result.primary(), &[2.0, 5.0, 7.0]);
        assert_eq!(result.lower_series(0).unwrap(), &[1.0, 3.0, 5.0]);
        assert_eq!(result.upper_series(0).unwrap(), &[4.0, 6.0, 8.0]);
    }

    #[test]
    fn multiple_constraints_compose_in_order() {
        let forecast = Forecast::from_values_with_intervals(
            vec![-2.7, 3.3, -0.1],
            vec![-4.2, 1.8, -1.5],
            vec![-1.1, 4.9, 0.6],
        );
        let result = forecast.constrain(&[
            ForecastConstraint::NonNegative,
            ForecastConstraint::IntegerRound,
        ]);
        assert_eq!(result.primary(), &[0.0, 3.0, 0.0]);
        assert_eq!(result.lower_series(0).unwrap(), &[0.0, 1.0, 0.0]);
        assert_eq!(result.upper_series(0).unwrap(), &[0.0, 5.0, 1.0]);
    }

    #[test]
    fn custom_constraint_function_works() {
        let forecast = Forecast::from_values(vec![1.0, 2.0, 3.0, 4.0]);
        let result = forecast.constrain(&[ForecastConstraint::Custom(Box::new(|v| v * 2.0))]);
        assert_eq!(result.primary(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn identity_when_no_clamping_needed() {
        let forecast = Forecast::from_values_with_intervals(
            vec![5.0, 10.0, 15.0],
            vec![3.0, 8.0, 13.0],
            vec![7.0, 12.0, 17.0],
        );
        let result = forecast.non_negative();
        assert_eq!(result.primary(), forecast.primary());
        assert_eq!(
            result.lower_series(0).unwrap(),
            forecast.lower_series(0).unwrap()
        );
        assert_eq!(
            result.upper_series(0).unwrap(),
            forecast.upper_series(0).unwrap()
        );
    }

    #[test]
    fn lower_bound_constraint() {
        let forecast = Forecast::from_values(vec![-5.0, 0.0, 5.0, 10.0]);
        let result = forecast.constrain(&[ForecastConstraint::LowerBound(2.0)]);
        assert_eq!(result.primary(), &[2.0, 2.0, 5.0, 10.0]);
    }

    #[test]
    fn upper_bound_constraint() {
        let forecast = Forecast::from_values(vec![1.0, 5.0, 10.0, 20.0]);
        let result = forecast.constrain(&[ForecastConstraint::UpperBound(8.0)]);
        assert_eq!(result.primary(), &[1.0, 5.0, 8.0, 8.0]);
    }

    #[test]
    fn constraints_on_multivariate_forecast() {
        let mut forecast = Forecast::with_dimensions(2);
        forecast.series_mut(0).extend([-1.0, 2.0, 5.0]);
        forecast.series_mut(1).extend([3.0, -4.0, 7.0]);
        let result = forecast.non_negative();
        assert_eq!(result.series(0).unwrap(), &[0.0, 2.0, 5.0]);
        assert_eq!(result.series(1).unwrap(), &[3.0, 0.0, 7.0]);
    }

    #[test]
    fn constraints_on_forecast_without_intervals() {
        let forecast = Forecast::from_values(vec![-1.0, 3.0, -2.0]);
        let result = forecast.non_negative();
        assert_eq!(result.primary(), &[0.0, 3.0, 0.0]);
        assert!(!result.has_lower());
        assert!(!result.has_upper());
    }

    #[test]
    fn empty_constraints_returns_identical_forecast() {
        let forecast = Forecast::from_values(vec![1.0, 2.0, 3.0]);
        let result = forecast.constrain(&[]);
        assert_eq!(result.primary(), &[1.0, 2.0, 3.0]);
    }
}
