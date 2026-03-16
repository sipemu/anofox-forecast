# Forecast Constraints

**Run:** `cargo run --example constraints`

## What this example demonstrates

Shows how to enforce domain-specific requirements on forecast values -- such as non-negativity, bounded ranges, integer rounding, and custom transformations -- after a model has produced its raw output. Constraints are applied to both point forecasts and prediction intervals, and can be composed together in a single call.

## Sections

1. **Raw forecast** -- Fits an ETS(A,A,A) model and produces a 12-step forecast with 95% intervals that may contain negative values.
2. **NonNegative constraint** -- Clamps all values to zero or above.
3. **LowerBound / UpperBound** -- Restricts forecasts to the range [2, 30].
4. **Bounds** -- Combined lower + upper bound in a single variant, clamping to [0, 25].
5. **IntegerRound** -- Rounds point forecasts to the nearest integer, floors lower bounds, and ceils upper bounds.
6. **Custom constraint** -- Applies an arbitrary closure (snap to nearest multiple of 5).
7. **Convenience methods** -- Demonstrates `forecast.non_negative()`, `.clamp()`, and `.round_to_integer()` shorthand.
8. **Composing constraints** -- Chains NonNegative + Bounds + IntegerRound via `forecast.constrain()`.
9. **Constraints without intervals** -- Applies constraints to a point-only `Forecast` with no intervals.

## Key types

- `ForecastConstraint` -- enum with `NonNegative`, `LowerBound`, `UpperBound`, `Bounds`, `IntegerRound`, `Custom`
- `ConstrainedForecast::apply()` -- applies a slice of constraints to a `Forecast`
- `Forecast` convenience methods -- `.non_negative()`, `.clamp()`, `.round_to_integer()`, `.constrain()`
- `ETS` / `ETSSpec` -- exponential smoothing model used to generate the raw forecast
