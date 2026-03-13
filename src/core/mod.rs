//! Core data structures for time series forecasting.

pub mod constraints;
mod forecast;
mod time_series;

pub use constraints::{ConstrainedForecast, ForecastConstraint};
pub use forecast::Forecast;
pub use time_series::{
    AggregationMethod, CalendarAnnotations, Frequency, InterpolationMethod, MissingValuePolicy,
    TimeSeries, TimeSeriesBuilder, ValueLayout,
};
