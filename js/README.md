# @anofox/forecast

WebAssembly bindings for [anofox-forecast](https://crates.io/crates/anofox-forecast), a time series forecasting library.

## Installation

```bash
npm install @anofox/forecast
```

## Usage

### Basic Example

```javascript
import { TimeSeries, NaiveForecaster, ThetaForecaster } from '@anofox/forecast';

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
import { TimeSeries } from '@anofox/forecast';

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

### Available Forecasters

| Forecaster | Description |
|------------|-------------|
| `NaiveForecaster` | Uses the last observation as forecast |
| `MeanForecaster` | Uses the historical mean as forecast |
| `SeasonalNaiveForecaster` | Uses observations from the same season |
| `RandomWalkDriftForecaster` | Random walk with drift |
| `WindowAverageForecaster` | Rolling window average |
| `SESForecaster` | Simple Exponential Smoothing |
| `HoltForecaster` | Double Exponential Smoothing (Holt's method) |
| `HoltWintersForecaster` | Triple Exponential Smoothing |
| `DampedHoltWintersForecaster` | Damped trend variant |
| `ThetaForecaster` | Theta method |
| `OptimizedThetaForecaster` | Optimized Theta method |

### Prediction Intervals

Some models support prediction intervals:

```javascript
import { NaiveForecaster } from '@anofox/forecast';

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

## Browser Usage

```html
<script type="module">
  import init, { TimeSeries, ThetaForecaster } from './anofox_forecast_js.js';

  async function main() {
    await init();

    const data = new Float64Array([10, 12, 15, 14, 18, 20, 22, 25]);
    const ts = new TimeSeries(data);

    const model = new ThetaForecaster(2.0);
    model.fit(ts);

    const forecast = model.predict(5);
    console.log('Forecast:', forecast.values);
  }

  main();
</script>
```

## Node.js Usage

```javascript
import { readFile } from 'fs/promises';
import { TimeSeries, OptimizedThetaForecaster } from '@anofox/forecast';

// Load your data
const data = [/* your time series data */];
const ts = new TimeSeries(new Float64Array(data));

// Forecast
const model = new OptimizedThetaForecaster();
model.fit(ts);
const forecast = model.predict(10);
```

## Limitations

- The `parallel` feature from the Rust crate is not available in WASM
- Some advanced features like postprocessing (conformal prediction) are not yet exposed

## License

MIT
