# @sipemu/anofox-forecast

WebAssembly bindings for the [anofox-forecast](https://crates.io/crates/anofox-forecast) time series forecasting library.

40+ forecasting models, automatic model selection, probabilistic postprocessing, and more — running entirely in the browser via WebAssembly.

## Installation

```bash
npm install @sipemu/anofox-forecast
```

## Quick Start

```javascript
import init, { TimeSeries, AutoForecaster, NaiveForecaster } from '@sipemu/anofox-forecast';

await init();

const ts = new TimeSeries(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
const model = new AutoForecaster();
model.fit(ts);
const forecast = model.predict(5);
console.log(forecast.values); // Float64Array with 5 predictions
```

## Available Models

### Auto Selection
- `AutoForecaster` — selects best model across ARIMA, ETS, and Theta families
- `AutoEnsembleForecaster` — automatic ensemble of top-K models
- `AutoARIMAForecaster` — automatic ARIMA order selection
- `AutoETSForecaster` — automatic ETS model selection
- `AutoThetaForecaster` — automatic Theta variant selection
- `AutoTBATSForecaster` — automatic TBATS configuration

### Exponential Smoothing
- `SESForecaster` — Simple Exponential Smoothing
- `HoltForecaster` — Holt's Linear Trend
- `HoltWintersForecaster` — Holt-Winters (additive/multiplicative seasonality)
- `SeasonalESForecaster` — Seasonal Exponential Smoothing
- `ETSForecaster` — Error-Trend-Seasonal state-space framework

### ARIMA
- `ARIMAForecaster` — ARIMA(p,d,q) model

### Theta
- `ThetaForecaster` — Standard Theta
- `OptimizedThetaForecaster` — Optimized Theta
- `DynamicThetaForecaster` — Dynamic Theta

### Baseline
- `NaiveForecaster`, `MeanForecaster`, `SeasonalNaiveForecaster`
- `RandomWalkWithDriftForecaster`, `SMAForecaster`, `WindowAverageForecaster`
- `SeasonalWindowAverageForecaster`

### Intermittent Demand
- `CrostonForecaster`, `ADIDAForecaster`, `TSBForecaster`, `IMAPAForecaster`

### Complex Seasonality & Volatility
- `TBATSForecaster`, `MFLESForecaster`, `MSTLForecaster`
- `GARCHForecaster` — volatility modeling

### Ensemble
- `EnsembleForecaster` — combine multiple models

## Probabilistic Postprocessing

Generate calibrated prediction intervals using conformal prediction, historical simulation, or normal approximation.

```javascript
import { JsConformalPredictor, JsPointForecasts, JsPostProcessor } from '@sipemu/anofox-forecast';

// Conformal prediction intervals
const predictor = new JsConformalPredictor(0.9); // 90% coverage
predictor.calibrate(forecasts, actuals);
const intervals = predictor.predictIntervals(newForecasts);
console.log(intervals.lower);  // Float64Array
console.log(intervals.upper);  // Float64Array

// Unified PostProcessor API
const processor = JsPostProcessor.conformal(0.95);
const trained = processor.train(forecasts, actuals);
const pi = processor.predictIntervals(trained, newForecasts);

// Backtesting
const config = new JsBacktestConfig(50, 5, 7); // window=50, step=5, horizon=7
const result = backtestPostProcessor(processor, forecasts, actuals, config);
console.log(`Coverage: ${result.coverage}`);
```

### Available Postprocessing Methods
- `JsConformalPredictor` — distribution-free prediction intervals (split, cross-val, jackknife+)
- `JsNormalPredictor` — Gaussian error assumption baseline
- `JsHistoricalSimulator` — non-parametric empirical error distribution
- `JsPostProcessor` — unified API wrapping all methods
- `JsBacktestConfig` / `JsBacktestResult` — rolling/expanding window backtesting

## Common API Pattern

All forecasters share the same interface:

```javascript
const model = new SomeForecaster(/* config */);
model.fit(timeSeries);

// Point forecasts
const forecast = model.predict(horizon);
console.log(forecast.values);

// With confidence intervals
const forecastCI = model.predictWithIntervals(horizon, 0.95);
console.log(forecastCI.lower);
console.log(forecastCI.upper);
```

## ETS Notation

Create ETS models using standard notation (A=Additive, M=Multiplicative, N=None):

```javascript
const ets = ETSForecaster.fromNotation('AAA', 12); // ETS(A,A,A) with period 12
ets.fit(ts);
const forecast = ets.predict(12);
```

## Calendar Annotations

Add holidays and named regressors for models that support exogenous variables:

```javascript
const calendar = new CalendarAnnotations();
calendar.addHoliday(Date.parse('2024-12-25'));
calendar.addRegressor('temperature', new Float64Array([20, 22, 25, ...]));

ts.setCalendar(calendar);
model.fit(ts); // ARIMA, MFLES, etc. will use the regressors
```

## TypeScript Support

Full TypeScript definitions are included. Import types directly:

```typescript
import { TimeSeries, Forecast, AutoForecaster } from '@sipemu/anofox-forecast';

const ts: TimeSeries = new TimeSeries(new Float64Array([1, 2, 3]));
const forecast: Forecast = model.predict(5);
```

## Links

- [Rust crate](https://crates.io/crates/anofox-forecast)
- [API documentation](https://docs.rs/anofox-forecast)
- [GitHub repository](https://github.com/sipemu/anofox-forecast)
- [DuckDB extension](https://github.com/DataZooDE/anofox-forecast)

## License

MIT
