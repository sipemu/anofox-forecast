# GARCH Implementation from anofox-forecast Repository

This document contains the complete C++ GARCH implementation extracted from the DataZooDE/anofox-forecast repository.

## Repository Information

- **Repository**: DataZooDE/anofox-forecast
- **Description**: Statistical timeseries forecasting in DuckDB
- **Language**: C++
- **Stars**: 19
- **License**: Not specified in extraction

## File Locations

- **Header**: `anofox-time/include/anofox-time/models/garch.hpp`
- **Implementation**: `anofox-time/src/models/garch.cpp`

## Header File (garch.hpp)

```cpp
#pragma once

#include "anofox-time/models/iforecaster.hpp"
#include "anofox-time/utils/logging.hpp"
#include <optional>
#include <vector>

namespace anofoxtime::models {

class GARCH {
public:
    GARCH(int p, int q, double omega, std::vector<double> alpha, std::vector<double> beta);

    void fit(const std::vector<double>& data);
    double forecastVariance(int horizon) const;

    const std::vector<double>& residuals() const { return residuals_; }
    const std::vector<double>& conditionalVariance() const { return sigma2_; }

private:
    void validateParameters() const;

    int p_;
    int q_;
    double omega_;
    std::vector<double> alpha_;
    std::vector<double> beta_;
    double mean_ = 0.0;
    std::vector<double> residuals_;
    std::vector<double> sigma2_;
};

} // namespace anofoxtime::models
```

## Implementation File (garch.cpp)

```cpp
#include "anofox-time/models/garch.hpp"
#include <numeric>
#include <stdexcept>

namespace anofoxtime::models {

GARCH::GARCH(int p, int q, double omega, std::vector<double> alpha, std::vector<double> beta)
    : p_(p), q_(q), omega_(omega), alpha_(std::move(alpha)), beta_(std::move(beta)) {
    if (p_ <= 0 || q_ <= 0) {
        throw std::invalid_argument("GARCH requires positive p and q orders.");
    }
    if (static_cast<int>(alpha_.size()) != p_ || static_cast<int>(beta_.size()) != q_) {
        throw std::invalid_argument("Alpha/Beta size must match p/q respectively.");
    }
    validateParameters();
}

void GARCH::validateParameters() const {
    if (omega_ <= 0.0) {
        throw std::invalid_argument("Omega must be positive.");
    }

    for (double a : alpha_) {
        if (a < 0.0) {
            throw std::invalid_argument("Alpha coefficients must be non-negative.");
        }
    }

    for (double b : beta_) {
        if (b < 0.0) {
            throw std::invalid_argument("Beta coefficients must be non-negative.");
        }
    }

    double sum = std::accumulate(alpha_.begin(), alpha_.end(), 0.0) +
                 std::accumulate(beta_.begin(), beta_.end(), 0.0);
    if (sum >= 1.0) {
        throw std::invalid_argument("Sum of alpha and beta must be < 1 for stationarity.");
    }
}

void GARCH::fit(const std::vector<double>& data) {
    if (data.size() < static_cast<std::size_t>(std::max(p_, q_))) {
        throw std::invalid_argument("Insufficient data for GARCH fitting.");
    }

    mean_ = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    residuals_.resize(data.size());
    sigma2_.resize(data.size());

    double init = 0.0;
    for (double x : data) {
        init += (x - mean_) * (x - mean_);
    }
    init /= data.size();

    std::fill(sigma2_.begin(), sigma2_.begin() + std::max(p_, q_), init);

    for (std::size_t t = 0; t < data.size(); ++t) {
        residuals_[t] = data[t] - mean_;
        if (t < static_cast<std::size_t>(std::max(p_, q_))) {
            continue;
        }

        double var = omega_;
        for (int i = 0; i < p_; ++i) {
            var += alpha_[i] * residuals_[t - i - 1] * residuals_[t - i - 1];
        }
        for (int j = 0; j < q_; ++j) {
            var += beta_[j] * sigma2_[t - j - 1];
        }

        sigma2_[t] = var;
    }

    ANOFOX_INFO("GARCH({},{}) model fitted. Omega={}, mean={}", p_, q_, omega_, mean_);
}

double GARCH::forecastVariance(int horizon) const {
    if (sigma2_.empty()) {
        throw std::runtime_error("GARCH model must be fitted before forecasting.");
    }
    if (horizon <= 0) {
        throw std::invalid_argument("Horizon must be positive.");
    }

    double last_var = sigma2_.back();
    double arch_sum = std::accumulate(alpha_.begin(), alpha_.end(), 0.0);
    double garch_sum = std::accumulate(beta_.begin(), beta_.end(), 0.0);

    double unconditional = omega_ / (1.0 - arch_sum - garch_sum);

    double variance = last_var;
    for (int h = 0; h < horizon; ++h) {
        variance = omega_ + (arch_sum + garch_sum) * variance;
    }

    return variance + unconditional;
}

} // namespace anofoxtime::models
```

## Implementation Details

### Model Specification

This is a GARCH(p,q) model where:
- **p**: Order of the ARCH component (number of lagged squared residuals)
- **q**: Order of the GARCH component (number of lagged conditional variances)

The conditional variance equation is:
```
σ²(t) = ω + Σ(αᵢ * ε²(t-i)) + Σ(βⱼ * σ²(t-j))
```

Where:
- `ω` (omega): Constant term
- `αᵢ` (alpha): ARCH coefficients for squared residuals
- `βⱼ` (beta): GARCH coefficients for lagged variances
- `ε(t)`: Residual at time t
- `σ²(t)`: Conditional variance at time t

### Key Features

1. **Parameter Validation**:
   - Omega must be positive
   - Alpha and Beta coefficients must be non-negative
   - Sum of all coefficients must be < 1 for stationarity

2. **Fitting Process**:
   - Calculates mean of the data
   - Computes residuals as deviations from mean
   - Initializes variance with sample variance
   - Recursively computes conditional variance using GARCH equation

3. **Variance Forecasting**:
   - Multi-step ahead variance forecasting
   - Converges to unconditional variance over long horizons
   - Uses iterative formula: `σ²(t+h) = ω + (Σα + Σβ) * σ²(t+h-1)`

4. **State Management**:
   - Stores residuals and conditional variances
   - Provides accessor methods for diagnostic purposes
   - Includes logging via ANOFOX_INFO macro

### Dependencies

- `anofox-time/models/iforecaster.hpp`: Interface for forecasting models
- `anofox-time/utils/logging.hpp`: Logging utilities
- Standard C++ libraries: `<numeric>`, `<stdexcept>`, `<optional>`, `<vector>`

### Limitations

- This implementation assumes pre-specified parameters (no automatic parameter estimation via MLE)
- No goodness-of-fit statistics or model diagnostics
- Forecast variance calculation may need adjustment (the unconditional variance addition appears non-standard)
- No support for exogenous variables or asymmetric effects (like EGARCH or GJR-GARCH)

## Related Files in anofox-time Library

The anofox-time library contains 35+ forecasting models including:
- ARIMA and Auto-ARIMA
- ETS (Error, Trend, Seasonal) and Auto-ETS
- Theta models
- TBATS
- Holt-Winters
- Seasonal decomposition (MSTL, MFLES)
- Intermittent demand models (Croston, ADIDA, IMAPA, TSB)

## Repository Structure

```
anofox-forecast/
├── anofox-time/              # Core forecasting library (submodule)
│   ├── include/anofox-time/
│   │   ├── models/          # 39 forecasting model headers
│   │   ├── core/            # TimeSeries, Forecast data structures
│   │   ├── transform/       # Data transformers
│   │   ├── detectors/       # Anomaly detection
│   │   └── ...
│   ├── src/
│   │   ├── models/          # 39 implementation files
│   │   └── ...
│   ├── tests/               # 417+ unit tests
│   └── examples/            # 10 demonstration programs
└── src/                     # DuckDB extension integration
```

## Sources

- [DataZoo GmbH GitHub Organization](https://github.com/datazoode)
- [anofox-forecast Repository](https://github.com/DataZooDE/anofox-forecast)
- Direct file URLs:
  - Header: https://raw.githubusercontent.com/DataZooDE/anofox-forecast/main/anofox-time/include/anofox-time/models/garch.hpp
  - Implementation: https://raw.githubusercontent.com/DataZooDE/anofox-forecast/main/anofox-time/src/models/garch.cpp
