# Temporal Aggregation

**Run:** `cargo run --example temporal_aggregation`

## What this example demonstrates

Walks through all temporal resampling operations on a `TimeSeries`: fixed-period aggregation (sum, mean, median, first, last, min, max), decimation-style downsampling, upsampling with interpolation, and sliding-window aggregation with configurable overlap. Uses a 24-hour synthetic electricity demand series.

## Sections

1. **Aggregate with Sum** -- Groups hourly data into 4-hour totals using `aggregate(4, Sum)`.
2. **Aggregate with Mean** -- Groups into 6-hour averages using `aggregate(6, Mean)`.
3. **Aggregation methods comparison** -- Applies all 7 aggregation methods (Sum, Mean, Median, First, Last, Min, Max) with period=8 and compares results side by side.
4. **Downsample** -- Decimates by factor 3 using `downsample(3)`, keeping every 3rd observation.
5. **Upsample with Linear Interpolation** -- Inserts 2 interpolated points between each original using `upsample(3, Linear)`.
6. **Upsample with Forward Fill** -- Repeats each value using `upsample(3, ForwardFill)` and compares with linear interpolation.
7. **Sliding window aggregation** -- Demonstrates rolling sum and mean (window=4, step=1), non-overlapping windows (window=6, step=6), and overlapping with custom step (window=6, step=2).
8. **Chained operations** -- Combines `aggregate(3, Mean)` followed by `downsample(2)` to show composability.

## Key types

- `TimeSeries` -- core series type with methods `aggregate`, `downsample`, `upsample`, `sliding_window_aggregate`
- `AggregationMethod` -- enum: `Sum`, `Mean`, `Median`, `First`, `Last`, `Min`, `Max`
- `InterpolationMethod` -- enum: `Linear`, `ForwardFill`, `BackwardFill`, `Zero`
