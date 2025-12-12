# C++ Theta Model Implementations from anofox-forecast

This directory contains the complete C++ implementation of Theta forecasting models from the DataZooDE/anofox-forecast repository.

## Repository Information

- **Source**: https://github.com/DataZooDE/anofox-forecast
- **Library**: anofox-time (C++ time series forecasting library)
- **Language**: C++ (89.6% of codebase)
- **License**: As per repository
- **Last Updated**: December 11, 2025

## Files Downloaded

### Header Files (.hpp)
1. `theta.hpp` - Base Theta class (Standard Theta Method - STM)
2. `optimized_theta.hpp` - Optimized Theta Method (OTM) 
3. `dynamic_theta.hpp` - Dynamic Theta Method (DSTM)
4. `auto_theta.hpp` - Automatic Theta model selection
5. `dynamic_optimized_theta.hpp` - Dynamic Optimized Theta Method (DOTM)
6. `theta_pegels.hpp` - Pegels state-space formulation
7. `theta_utils.hpp` - Utility functions for seasonality

### Implementation Files (.cpp)
1. `theta.cpp` - STM implementation
2. `optimized_theta.cpp` - OTM implementation
3. `dynamic_theta.cpp` - DSTM implementation
4. `auto_theta.cpp` - AutoTheta implementation
5. `dynamic_optimized_theta.cpp` - DOTM implementation
6. `theta_pegels.cpp` - Core Pegels state-space logic
7. `theta_utils.cpp` - Deseasonalization utilities

## Model Variants

### 1. Standard Theta Method (STM)
- **File**: `theta.hpp`, `theta.cpp`
- **Class**: `Theta`
- **Description**: Classic Theta method with fixed theta=2.0
- **Features**:
  - Pegels state-space formulation
  - Seasonal decomposition/reseasonalization
  - Fixed parameters (alpha can be set externally)

### 2. Optimized Theta Method (OTM)
- **File**: `optimized_theta.hpp`, `optimized_theta.cpp`
- **Class**: `OptimizedTheta`
- **Description**: Optimizes alpha, theta, and initial level parameters
- **Optimization**: Uses L-BFGS or Nelder-Mead
- **Features**:
  - Automatic parameter tuning
  - 3-step-ahead MSE criterion
  - Delegates to Theta class after optimization

### 3. Dynamic Theta Method (DSTM)
- **File**: `dynamic_theta.hpp`, `dynamic_theta.cpp`
- **Class**: `DynamicTheta`
- **Description**: Dynamic updates of An and Bn parameters
- **Features**:
  - Updates trend parameters at each time step
  - Pegels state-space with dynamic mode
  - Seasonal handling

### 4. Dynamic Optimized Theta Method (DOTM)
- **File**: `dynamic_optimized_theta.hpp`, `dynamic_optimized_theta.cpp`
- **Class**: `DynamicOptimizedTheta`
- **Description**: Combines dynamic updates with parameter optimization
- **Features**:
  - M4 competition winner component
  - Optimizes all parameters with dynamic An/Bn
  - Best performance for many series

### 5. AutoTheta
- **File**: `auto_theta.hpp`, `auto_theta.cpp`
- **Class**: `AutoTheta`
- **Description**: Automatic model selection among all variants
- **Features**:
  - Automatic seasonality detection using ACF
  - Additive/multiplicative decomposition
  - MSE-based model selection
  - Comprehensive diagnostics

## Core Implementation Details

### Pegels State-Space Formulation
- **File**: `theta_pegels.hpp`, `theta_pegels.cpp`
- **State Vector**: [level, meany, An, Bn, mu]
  - `level`: Exponentially smoothed level
  - `meany`: Running mean
  - `An`: Intercept parameter
  - `Bn`: Slope parameter
  - `mu`: Forecast component

### Key Functions

#### `init_state()`
Initializes state vector:
- **Static models (STM, OTM)**: Compute An/Bn from full data
- **Dynamic models (DSTM, DOTM)**: Initialize with first value

#### `update()`
Updates state at each time step:
- Level update via exponential smoothing
- Running mean update
- An/Bn updates (static vs dynamic)
- Mu computation: `level + (1 - 1/theta) * (An * (1-alpha)^i + Bn * ...)`

#### `forecast()`
Generates h-step-ahead forecasts by iteratively updating state with synthetic observations

#### `calc()`
Computes MSE and residuals over training data
- Returns scaled MSE for optimization
- Tracks multi-step-ahead errors (AMSE)

#### `optimize()`
Parameter optimization using:
1. **L-BFGS** (default): Gradient-based, faster
   - 50 max iterations
   - Numerical gradients via ThetaGradients
   - Bounds: alpha [0.01, 0.99], theta [1.0, 10.0]
2. **Nelder-Mead**: Derivative-free fallback
   - 1000 max iterations
   - Simplex-based search

### Seasonal Handling

#### `theta_utils::deseasonalize()`
- Centered moving average for trend extraction
- Seasonal indices normalized to mean 1.0
- Divides original data by indices

#### `theta_utils::reseasonalize()`
- Multiplies forecasts by seasonal indices
- Cycles through pattern using modulo arithmetic

### AutoTheta Seasonality Detection

Uses ACF (Autocorrelation Function) test:
1. Compute ACF up to seasonal lag
2. Test if ACF at lag m is significant
3. Standard error via Bartlett's formula
4. 95% confidence threshold (z = 1.645)

## Model Selection Strategy

AutoTheta evaluates models in priority order:
1. **DOTM** (default, M4 winner)
2. OTM
3. DSTM
4. STM

Selection based on:
- MSE on training data
- 3-step-ahead forecast accuracy
- Convergence success

## Key Differences from R/Python Implementations

1. **Optimization**:
   - Primary optimizer is L-BFGS (gradient-based)
   - Nelder-Mead available as fallback
   - Bounded optimization with strict parameter ranges

2. **State-Space**:
   - Full Pegels formulation with 5-element state vector
   - Explicit tracking of An, Bn, meany components
   - Workspace pre-allocation for efficiency

3. **Performance**:
   - Pre-allocated workspaces to minimize allocations
   - Reusable gradient computation buffers
   - Reduced max iterations (50 for L-BFGS)

## Usage Pattern

```cpp
// Standard Theta
auto theta = Theta(seasonal_period=12);
theta.fit(timeseries);
auto forecast = theta.predict(horizon=10);

// Optimized Theta
auto otm = OptimizedTheta(seasonal_period=12, OptimizerType::LBFGS);
otm.fit(timeseries);
auto forecast = otm.predict(horizon=10);

// AutoTheta (automatic selection)
auto auto_theta = AutoTheta(seasonal_period=12);
auto_theta.fit(timeseries);
auto forecast = auto_theta.predict(horizon=10);
auto diagnostics = auto_theta.getDiagnostics();
```

## Integration with DuckDB

These models are part of the anofox-forecast DuckDB extension, providing SQL-accessible forecasting:

```sql
SELECT forecast_theta(value, 12, 10) FROM timeseries;
```

## Reference Implementation

This C++ implementation is based on and aligned with:
- Nixtla's statsforecast (Python)
- Pegels state-space formulation
- M3/M4 competition methodologies

## Validation Notes

For Rust forecast crate validation:
1. Compare state vector evolution (level, An, Bn, mu)
2. Verify optimization convergence and parameter values
3. Check seasonal decomposition indices
4. Validate forecast outputs against reference data
