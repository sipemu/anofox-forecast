# Forecast Validation Framework

Comparison of **anofox-forecast** (Rust) vs **NIXTLA statsforecast** (Python)

## Overview

This validation framework provides a systematic comparison between the Rust forecasting library and NIXTLA's widely-used statsforecast Python package. The goal is to ensure implementation correctness and identify any differences in forecasting behavior.

## Quick Start

```bash
# Run the complete validation pipeline
cd validation
uv run python run_all.py
```

This will:
1. Generate synthetic time series data
2. Run Rust forecasts via cargo
3. Run statsforecast models
4. Compare results and generate reports

## Output Files

After running validation, reports are generated in `validation/output/`:

| File | Description |
|------|-------------|
| `report.md` | Human-readable comparison report |
| `point_forecasts.csv` | Detailed point forecast comparison |
| `confidence_intervals.csv` | Confidence interval comparison |
| `summary_metrics.csv` | Summary metrics by model/series |

---

## Test Series Types

### Standard Series (11 types)

| Series | Description | Period | Purpose |
|--------|-------------|--------|---------|
| `stationary` | White noise around mean=50 | - | Baseline, no pattern |
| `trend` | Linear trend (slope=0.5) with noise | - | Trend detection |
| `seasonal` | Sinusoidal seasonal pattern | 12 | Seasonal detection (additive) |
| `trend_seasonal` | Combined trend + seasonal | 12 | Multiple components |
| `seasonal_negative` | Seasonal with negative values | 12 | Multiplicative fallback test |
| `multiplicative_seasonal` | True multiplicative seasonality | 12 | Multiplicative handling |
| `intermittent` | Sparse demand (~30% non-zero) | - | Intermittent demand models |
| `high_frequency` | Daily + weekly seasonality | 24, 168 | MSTL, multiple seasonalities |
| `structural_break` | Level shift at midpoint | - | Robustness, changepoints |
| `long_memory` | ARFIMA-like slow decay | - | Long memory processes |
| `noisy_seasonal` | High noise-to-signal seasonal | 12 | Model selection robustness |

### Series Parameters

- **Default observations**: 100 (500 for high_frequency)
- **Random seed**: 42 (reproducible)
- **Seasonal period**: 12 (monthly data)

---

## Current Validation Results (29 Models)

### Status Summary

| Status | Models | Percentage |
|--------|--------|------------|
| EXCELLENT (MAD < 0.01) | 13 | 44.8% |
| VERY GOOD (MAD < 0.1) | 1 | 3.4% |
| GOOD (MAD < 1.0) | 8 | 27.6% |
| ACCEPTABLE (MAD < 2.0) | 7 | 24.1% |
| NEEDS WORK (MAD ≥ 2.0) | 0 | 0.0% |

### Perfect Match (MAD ≈ 0) - 11 Models

| Model | Rust | statsforecast | MAD |
|-------|------|---------------|-----|
| Naive | `Naive` | `Naive` | 0.0000 |
| SeasonalNaive | `SeasonalNaive` | `SeasonalNaive` | 0.0000 |
| RandomWalkWithDrift | `RandomWalkWithDrift` | `RandomWalkWithDrift` | 0.0000 |
| SES | `SES` | `SimpleExponentialSmoothing` | 0.0000 |
| Croston | `Croston` | `CrostonClassic` | 0.0000 |
| CrostonSBA | `Croston::sba()` | `CrostonSBA` | 0.0000 |
| TSB | `TSB` | `TSB` | 0.0000 |
| SeasonalWindowAverage | `SeasonalWindowAverage` | `SeasonalWindowAverage` | 0.0000 |
| HistoricAverage | `HistoricAverage` | `HistoricAverage` | 0.0000 |
| WindowAverage | `WindowAverage` | `WindowAverage` | 0.0000 |
| SeasonalES | `SeasonalES` | `SeasonalExponentialSmoothing` | 0.0000 |

### Excellent Agreement (MAD < 0.01) - 2 Models

| Model | Rust | statsforecast | MAD |
|-------|------|---------------|-----|
| ADIDA | `ADIDA` | `ADIDA` | 0.0004 |
| IMAPA | `IMAPA` | `IMAPA` | 0.0004 |

### Very Good Agreement (MAD < 0.1) - 1 Model

| Model | Rust | statsforecast | MAD |
|-------|------|---------------|-----|
| MFLES | `MFLES` | `MFLES` | 0.0296 |

### Good Agreement (MAD < 1.0) - 8 Models

| Model | Rust | statsforecast | MAD | Notes |
|-------|------|---------------|-----|-------|
| Holt | `Holt` | `Holt` | 0.1658 | Minor optimization differences |
| GARCH | `GARCH` | `GARCH` | 0.4311 | Different optimizer convergence |
| OptimizedTheta | `OptimizedTheta` | `OptimizedTheta` | 0.4744 | |
| AutoTheta | `AutoTheta` | `AutoTheta` | 0.5202 | Model selection differences |
| AutoETS | `AutoETS` | `AutoETS` | 0.5384 | Model selection differences |
| Theta | `Theta` | `Theta` | 0.7894 | |
| MSTLForecaster | `MSTLForecaster` | `MSTL` | 0.8173 | Decomposition differences |
| DynamicTheta | `DynamicTheta` | `DynamicTheta` | 0.9442 | |

### Acceptable Agreement (MAD < 2.0) - 7 Models

| Model | Rust | statsforecast | MAD | Notes |
|-------|------|---------------|-----|-------|
| SARIMA | `SARIMA` | `ARIMA` | 1.0743 | Different optimization |
| ARIMA | `ARIMA` | `ARIMA` | 1.1438 | Parameter estimation |
| DynamicOptimizedTheta | `DynamicOptimizedTheta` | `DynamicOptimizedTheta` | 1.1494 | |
| HoltWinters | `HoltWinters` | `HoltWinters` | 1.3949 | Seasonal init differences |
| AutoARIMA | `AutoARIMA` | `AutoARIMA` | 1.6782 | Model selection algorithms |
| AutoTBATS | `AutoTBATS` | `AutoTBATS` | 1.8830 | Complex seasonality handling |
| TBATS | `TBATS` | `TBATS` | 1.9439 | Trigonometric terms |

---

## Model Implementation Status

### Fully Implemented (29 models)

**Basic Methods:**
- Naive, SeasonalNaive, RandomWalkWithDrift
- HistoricAverage, WindowAverage, SeasonalWindowAverage

**Exponential Smoothing:**
- SES, Holt, HoltWinters
- ETS (30 variants), AutoETS
- SeasonalES

**ARIMA Family:**
- ARIMA, SARIMA, AutoARIMA

**Theta Methods:**
- Theta, OptimizedTheta, DynamicTheta
- DynamicOptimizedTheta, AutoTheta

**Intermittent Demand:**
- Croston, CrostonSBA, TSB, ADIDA, IMAPA

**Advanced:**
- MFLES (gradient boosted decomposition)
- MSTLForecaster (multiple seasonal decomposition)
- TBATS, AutoTBATS (trigonometric seasonality)
- GARCH (volatility modeling)

### Not Yet Implemented

| Model | Priority | Notes |
|-------|----------|-------|
| AutoMFLES | Medium | Auto-tuned MFLES |
| CrostonOptimized | Low | Optimized Croston |
| ThetaPegels | Low | Pegels variation |

---

## Running Validation

### Prerequisites

```bash
# Install Python dependencies
cd validation
uv sync  # or pip install -r requirements.txt
```

### Individual Steps

```bash
# Generate data only
uv run python generate_data.py

# Run Rust forecasts
cargo run --example forecast_export --release

# Run statsforecast
uv run python run_statsforecast.py

# Compare results
uv run python compare_results.py
```

### Full Pipeline

```bash
uv run python run_all.py
```

---

## Metrics Used

| Metric | Description |
|--------|-------------|
| **MAD** | Mean Absolute Difference between forecasts |
| **Correlation** | Pearson correlation coefficient |
| **Max Diff** | Maximum absolute difference |
| **CI Width Diff** | Difference in confidence interval width |

### Interpretation

- **MAD = 0**: Perfect match
- **MAD < 0.01**: Excellent (essentially identical)
- **MAD < 0.1**: Very good (minor floating point differences)
- **MAD < 1.0**: Good (minor optimization differences)
- **MAD < 2.0**: Acceptable (algorithm differences)
- **MAD ≥ 2.0**: Needs investigation

---

## File Structure

```
validation/
├── README.md              # This documentation
├── generate_data.py       # Synthetic data generation
├── run_statsforecast.py   # statsforecast model runner
├── run_rust_forecasts.py  # Rust model runner (via cargo)
├── compare_results.py     # Result comparison and reporting
├── run_all.py             # Complete pipeline orchestration
├── pyproject.toml         # Python dependencies
├── data/                  # Generated CSV data files
├── results/
│   ├── rust/              # Rust forecast results
│   └── statsforecast/     # statsforecast results
└── output/                # Comparison reports
```
