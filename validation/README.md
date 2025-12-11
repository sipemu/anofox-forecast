# Forecast Validation Suite

This validation suite compares forecasts between the Rust `anofox-forecast` crate and the Python `statsforecast` (NIXTLA) package.

## Overview

The validation generates synthetic time series with different characteristics and runs equivalent forecasting models from both implementations, comparing:

- Point forecasts
- Confidence intervals (80%, 90%, 95%)

## Requirements

- **Rust**: For running anofox-forecast
- **Python 3.10+**: For running statsforecast
- **uv**: Python package manager (recommended)

## Quick Start

```bash
cd validation

# Install Python dependencies
uv sync

# Run the complete validation pipeline
uv run python run_all.py
```

## Step-by-Step Execution

You can also run each step individually:

```bash
# 1. Generate synthetic time series data
uv run python generate_data.py

# 2. Run Rust forecasts (via cargo)
uv run python run_rust_forecasts.py

# 3. Run statsforecast models
uv run python run_statsforecast.py

# 4. Compare results and generate reports
uv run python compare_results.py
```

## Synthetic Time Series

Four types of time series are generated (100 observations each):

| Type | Description |
|------|-------------|
| `stationary` | White noise around a mean (μ=50, σ=5) |
| `trend` | Linear trend with noise (intercept=10, slope=0.5) |
| `seasonal` | Seasonal pattern (period=12, amplitude=10) |
| `trend_seasonal` | Combined trend and seasonality |

## Models Compared

| Category | Model | Rust | statsforecast |
|----------|-------|------|---------------|
| Baseline | Naive | `Naive` | `Naive` |
| Baseline | Seasonal Naive | `SeasonalNaive` | `SeasonalNaive` |
| Baseline | Random Walk w/ Drift | `RandomWalkWithDrift` | `RandomWalkWithDrift` |
| Smoothing | Simple Exp. Smoothing | `SimpleExponentialSmoothing` | `SimpleExponentialSmoothing` |
| Smoothing | Holt's Linear | `HoltLinearTrend` | `Holt` |
| Smoothing | Holt-Winters | `HoltWinters` | `HoltWinters` |
| ARIMA | ARIMA(1,1,1) | `ARIMA::new(1,1,1)` | `ARIMA(order=(1,1,1))` |
| ARIMA | Auto ARIMA | `AutoARIMA` | `AutoARIMA` |
| ETS | Auto ETS | `AutoETS` | `AutoETS` |
| Theta | Theta | `Theta` | `Theta` |
| Intermittent | Croston | `Croston` | `CrostonClassic` |
| Intermittent | Croston SBA | `Croston::sba()` | `CrostonSBA` |
| Intermittent | TSB | `TSB` | `TSB` |

## Output

Results are saved to `output/`:

| File | Description |
|------|-------------|
| `report.md` | Human-readable markdown report |
| `point_forecasts.csv` | Detailed point forecast comparison |
| `confidence_intervals.csv` | CI bounds comparison |
| `summary_metrics.csv` | Summary metrics by model/series |

### Metrics Computed

- **MAD**: Mean Absolute Difference between forecasts
- **Max Diff**: Maximum absolute difference
- **Correlation**: Pearson correlation between forecasts
- **CI Width Diff**: Mean difference in confidence interval width

## Directory Structure

```
validation/
├── pyproject.toml          # Python dependencies
├── README.md               # This file
├── generate_data.py        # Synthetic data generation
├── run_statsforecast.py    # statsforecast runner
├── run_rust_forecasts.py   # Rust example runner
├── compare_results.py      # Comparison and reporting
├── run_all.py              # Main orchestration script
├── data/                   # Generated synthetic data
├── results/
│   ├── rust/               # Rust forecast results
│   └── statsforecast/      # statsforecast results
└── output/                 # Final comparison reports
```

## Notes

- Differences between implementations are expected and documented
- This validation does NOT fix any differences - it only reports them
- Random seed is fixed (42) for reproducibility
- Some models may fail on certain series types (e.g., intermittent models on continuous data)

## Interpreting Results

### Good Agreement
- Correlation close to 1.0
- Small MAD relative to forecast magnitude
- Similar CI widths

### Expected Differences
- Parameter optimization may converge to different values
- Different numerical precision
- Different default parameters
- Confidence interval calculation methods may vary
