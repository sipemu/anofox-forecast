# @sipemu/anofox-forecast

WebAssembly bindings for [anofox-forecast](https://crates.io/crates/anofox-forecast), a comprehensive time series forecasting library with 40+ models, automatic model selection, probabilistic postprocessing, and more.

## Installation

```bash
npm install @sipemu/anofox-forecast
```

## Usage

### Basic Example

```javascript
import { TimeSeries, NaiveForecaster, ThetaForecaster } from '@sipemu/anofox-forecast';

// Create a time series from values
const data = [10, 12, 15, 14, 18, 20, 22, 25, 24, 28];
const ts = new TimeSeries(new Float64Array(data));

// Create and fit a forecaster
const model = new NaiveForecaster();
model.fit(ts);

// Generate predictions
const forecast = model.predict(5);
console.log('Predictions:', forecast.values);
```

### With Timestamps

```javascript
import { TimeSeries } from '@sipemu/anofox-forecast';

const values = [10, 12, 15, 14, 18];
const timestamps = [
  Date.parse('2024-01-01'),
  Date.parse('2024-01-02'),
  Date.parse('2024-01-03'),
  Date.parse('2024-01-04'),
  Date.parse('2024-01-05'),
];

const ts = TimeSeries.withTimestamps(
  new Float64Array(values),
  new Float64Array(timestamps)
);
```

## Available Forecasters

### Baseline Models

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `NaiveForecaster` | Last observation repeated | - |
| `MeanForecaster` | Historical mean | - |
| `SeasonalNaiveForecaster` | Same season from previous cycle | `period` |
| `RandomWalkDriftForecaster` | Random walk with trend | - |
| `SMAForecaster` | Simple Moving Average | `window` |
| `WindowAverageForecaster` | Rolling window average | `window_size` |
| `SeasonalWindowAverageForecaster` | Seasonal window average | `period`, `window` |

### Exponential Smoothing Models

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `SESForecaster` | Simple Exponential Smoothing | `alpha` |
| `HoltForecaster` | Holt Linear Trend (Double ES) | `alpha`, `beta` |
| `HoltWintersForecaster` | Triple Exponential Smoothing | `alpha`, `beta`, `gamma`, `period` |
| `SeasonalESForecaster` | Seasonal Exponential Smoothing | `period` |
| `ETSForecaster` | ETS state-space model | `error`, `trend`, `seasonal`, `period` |
| `AutoETSForecaster` | Automatic ETS selection | - |

### Theta Models

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `ThetaForecaster` | Standard Theta method | - |
| `OptimizedThetaForecaster` | Optimized Theta | - |
| `DynamicThetaForecaster` | Dynamic coefficient updates | `alpha` |
| `AutoThetaForecaster` | Automatic Theta selection | - |

### ARIMA Models

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `ARIMAForecaster` | ARIMA | `p`, `d`, `q` |
| `SARIMAForecaster` | Seasonal ARIMA | `p`, `d`, `q`, `P`, `D`, `Q`, `period` |
| `AutoARIMAForecaster` | Automatic ARIMA selection | - |

### Intermittent Demand Models

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `CrostonForecaster` | Croston's method | - |
| `TSBForecaster` | Teunter-Syntetos-Babai | - |
| `ADIDAForecaster` | Aggregate-Disaggregate approach | - |
| `IMAPAForecaster` | Multiple Aggregation Prediction | - |

### Advanced Models

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `TBATSForecaster` | TBATS (complex seasonality) | `seasonal_periods[]` |
| `AutoTBATSForecaster` | Automatic TBATS | `seasonal_periods[]` |
| `MFLESForecaster` | Multiple Frequency LOESS | `seasonal_periods[]` |
| `MSTLForecasterWrapper` | MSTL decomposition | `seasonal_periods[]` |
| `GARCHForecaster` | GARCH volatility model | `p`, `q` |

### Auto Selection

| Forecaster | Description | Parameters |
|------------|-------------|------------|
| `AutoForecaster` | Best of ARIMA, ETS, Theta | - |
| `AutoEnsembleForecaster` | Ensemble of top-K models | - |

## Prediction Intervals

Most models support prediction intervals:

```javascript
import { NaiveForecaster } from '@sipemu/anofox-forecast';

const model = new NaiveForecaster();
model.fit(ts);

// Get forecast with 95% prediction intervals
const forecast = model.predictWithIntervals(5, 0.95);

console.log('Point predictions:', forecast.values);
console.log('Lower bound:', forecast.lower);
console.log('Upper bound:', forecast.upper);
```

## API Reference

### TimeSeries

- `new TimeSeries(values: Float64Array)` - Create from values
- `TimeSeries.withTimestamps(values, timestamps)` - Create with timestamps (ms since epoch)
- `ts.length` - Number of observations
- `ts.values` - Get values as array
- `ts.isEmpty()` - Check if empty
- `ts.hasMissingValues()` - Check for NaN values
- `ts.slice(start, end)` - Get a slice of the series

### Forecast

- `forecast.horizon` - Number of predictions
- `forecast.values` - Point predictions
- `forecast.lower` - Lower prediction interval (if available)
- `forecast.upper` - Upper prediction interval (if available)
- `forecast.hasLower()` - Check if lower interval exists
- `forecast.hasUpper()` - Check if upper interval exists

### ETS Model Specification

For `ETSForecaster`, use string codes:
- **Error**: `"A"` (additive) or `"M"` (multiplicative)
- **Trend**: `"N"` (none), `"A"` (additive), or `"Ad"` (additive damped)
- **Seasonal**: `"N"` (none), `"A"` (additive), or `"M"` (multiplicative)

```javascript
// ETS(A,A,M) - Additive error, Additive trend, Multiplicative seasonal
const ets = new ETSForecaster("A", "A", "M", 12);
```

#### Standard ETS Notation

You can also use standard ETS notation (following [FPP3 taxonomy](https://otexts.com/fpp3/taxonomy.html)):

```javascript
// Create from notation string
const model = ETSForecaster.fromNotation("AAA", 12);  // Holt-Winters additive
const model = ETSForecaster.fromNotation("MAM", 12);  // Multiplicative Holt-Winters
const model = ETSForecaster.fromNotation("AAdM", 12); // Damped trend, multiplicative seasonal
```

Valid notation format: `ErrorTrendSeasonal`
- First letter: Error (A or M)
- Second letter(s): Trend (N, A, or Ad)
- Third letter: Seasonal (N, A, or M)

#### Model Validation

Some ETS combinations are unstable and will throw an error:
- `MAA` - Multiplicative error + Additive trend + Additive seasonal
- `MAdA` - Multiplicative error + Damped trend + Additive seasonal

You can check validity before creating:

```javascript
// Check if a specification is valid
ETSForecaster.isValidSpec("A", "A", "A");  // true
ETSForecaster.isValidSpec("M", "A", "M");  // true
ETSForecaster.isValidSpec("M", "A", "A");  // false (unstable)
```

## Probabilistic Postprocessing

Generate calibrated prediction intervals using conformal prediction, historical simulation, or normal approximation:

```javascript
import { JsConformalPredictor, JsPointForecasts, JsPostProcessor } from '@sipemu/anofox-forecast';

// Conformal prediction intervals (distribution-free)
const predictor = new JsConformalPredictor(0.9); // 90% coverage
predictor.calibrate(forecasts, actuals);
const intervals = predictor.predictIntervals(newForecasts);
console.log('Lower:', intervals.lower);
console.log('Upper:', intervals.upper);

// Unified PostProcessor API
const processor = JsPostProcessor.conformal(0.95);
const trained = processor.train(forecasts, actuals);
const pi = processor.predictIntervals(trained, newForecasts);
```

### Available Methods
- `JsConformalPredictor` — distribution-free intervals (split, cross-val, jackknife+)
- `JsNormalPredictor` — Gaussian error assumption baseline
- `JsHistoricalSimulator` — non-parametric empirical error distribution
- `JsPostProcessor` — unified API wrapping all methods
- `JsBacktestConfig` / `JsBacktestResult` — rolling/expanding window backtesting

## Calendar Annotations

Add holidays and named regressors for models that support exogenous variables:

```javascript
import { CalendarAnnotations } from '@sipemu/anofox-forecast';

const calendar = new CalendarAnnotations();
calendar.addHoliday(Date.parse('2024-12-25'));
calendar.addRegressor('temperature', new Float64Array([20, 22, 25, 23]));

ts.setCalendar(calendar);
model.fit(ts); // ARIMA, MFLES, etc. will automatically use the regressors
```

## Browser Usage

```html
<script type="module">
  import init, { TimeSeries, ThetaForecaster } from './anofox_forecast_js.js';

  async function main() {
    await init();

    const data = new Float64Array([10, 12, 15, 14, 18, 20, 22, 25]);
    const ts = new TimeSeries(data);

    const model = new ThetaForecaster();
    model.fit(ts);

    const forecast = model.predict(5);
    console.log('Forecast:', forecast.values);
  }

  main();
</script>
```

## Node.js Usage

```javascript
import { TimeSeries, AutoARIMAForecaster } from '@sipemu/anofox-forecast';

// Load your data
const data = [/* your time series data */];
const ts = new TimeSeries(new Float64Array(data));

// Forecast with AutoARIMA
const model = new AutoARIMAForecaster();
model.fit(ts);
const forecast = model.predict(10);
```

## Limitations

- The `parallel` feature from the Rust crate is not available in WASM
- Postprocessing features (conformal prediction, IDR) are not yet exposed
- Cross-validation utilities are not yet exposed

## License

MIT
