# Basic Statistical Features

**Run:** `cargo run --example basic_features`

## What this example demonstrates

Computes a wide range of basic statistical features on a synthetic time series, including central tendency, dispersion, energy, change rates, and derivative measures. It also compares how these features behave across constant, trending, noisy, and spiky series.

## Sections

1. **Central tendency** -- computes mean, median, and sum of a 100-point synthetic series.
2. **Dispersion** -- calculates variance, standard deviation, and root mean square.
3. **Range and extremes** -- shows minimum, maximum, absolute maximum, and range.
4. **Energy features** -- computes absolute energy (sum of squared values).
5. **Change features** -- mean change, mean absolute change, and absolute sum of changes.
6. **Derivative features** -- mean second derivative (central) as an acceleration measure.
7. **Length** -- reports series length.
8. **Top values** -- mean of the N largest absolute values for several N.
9. **Energy ratio by chunks** -- distributes energy across 10 equal chunks using the `change` module.
10. **Reoccurrence analysis** -- sums and percentages of reoccurring values and data points on a rounded series.
11. **Series comparison** -- tabulates mean, std dev, absolute energy, and mean absolute change for constant, trending, noisy, and spiky series.
12. **Feature use cases** -- prints guidance on when to use each feature category.

## Key types

- `anofox_forecast::features::basic` -- `mean`, `median`, `sum_values`, `variance`, `standard_deviation`, `root_mean_square`, `minimum`, `maximum`, `absolute_maximum`, `abs_energy`, `mean_change`, `mean_abs_change`, `absolute_sum_of_changes`, `mean_second_derivative_central`, `length`, `mean_n_absolute_max`
- `anofox_forecast::features::change` -- `energy_ratio_by_chunks`, `sum_of_reoccurring_values`, `sum_of_reoccurring_data_points`, `percentage_of_reoccurring_values_to_all_values`, `percentage_of_reoccurring_datapoints_to_all_datapoints`, `ratio_value_number_to_time_series_length`
