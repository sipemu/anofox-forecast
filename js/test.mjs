/**
 * Integration tests for @sipemu/anofox-forecast npm package.
 *
 * Run with: node --test test.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Dynamic import since WASM needs to be initialized
const pkg = await import('./anofox_forecast_js.js');

// Load WASM binary manually for Node.js (web target uses fetch which doesn't work in Node)
const wasmPath = join(__dirname, 'anofox_forecast_js_bg.wasm');
const wasmBytes = await readFile(wasmPath);
await pkg.default(wasmBytes);

const {
  TimeSeries,
  NaiveForecaster,
  MeanForecaster,
  SeasonalNaiveForecaster,
  SMAForecaster,
  SESForecaster,
  HoltForecaster,
  HoltWintersForecaster,
  ETSForecaster,
  AutoETSForecaster,
  ThetaForecaster,
  AutoThetaForecaster,
  ARIMAForecaster,
  AutoARIMAForecaster,
  CrostonForecaster,
} = pkg;

// Helper to create test data
function createTestSeries(n = 30) {
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    values[i] = 10 + i + Math.sin(i * 0.5) * 2;
  }
  return new TimeSeries(values);
}

function createSeasonalSeries(n = 48, period = 12) {
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const trend = 10 + 0.5 * i;
    const seasonal = 5 * Math.sin(2 * Math.PI * i / period);
    values[i] = trend + seasonal;
  }
  return new TimeSeries(values);
}

// TimeSeries tests
test('TimeSeries creation', () => {
  const values = new Float64Array([1, 2, 3, 4, 5]);
  const ts = new TimeSeries(values);

  assert.strictEqual(ts.length, 5);
  assert.strictEqual(ts.isEmpty(), false);
  assert.deepStrictEqual(Array.from(ts.values), [1, 2, 3, 4, 5]);
});

test('TimeSeries with timestamps', () => {
  const values = new Float64Array([1, 2, 3]);
  const timestamps = new Float64Array([1000, 2000, 3000]);

  const ts = TimeSeries.withTimestamps(values, timestamps);
  assert.strictEqual(ts.length, 3);
});

test('TimeSeries slice', () => {
  const values = new Float64Array([1, 2, 3, 4, 5]);
  const ts = new TimeSeries(values);

  const sliced = ts.slice(1, 4);
  assert.strictEqual(sliced.length, 3);
  assert.deepStrictEqual(Array.from(sliced.values), [2, 3, 4]);
});

// Baseline forecasters
test('NaiveForecaster', () => {
  const ts = createTestSeries();
  const model = new NaiveForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.horizon, 5);
  assert.strictEqual(forecast.values.length, 5);
});

test('MeanForecaster', () => {
  const ts = createTestSeries();
  const model = new MeanForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
  // All forecasts should be the same (the mean)
  const firstVal = forecast.values[0];
  for (const val of forecast.values) {
    assert.strictEqual(val, firstVal);
  }
});

test('SeasonalNaiveForecaster', () => {
  const ts = createSeasonalSeries(48, 12);
  const model = new SeasonalNaiveForecaster(12);

  model.fit(ts);
  const forecast = model.predict(12);

  assert.strictEqual(forecast.values.length, 12);
});

test('SMAForecaster', () => {
  const ts = createTestSeries();
  const model = new SMAForecaster(5);

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

// Exponential smoothing
test('SESForecaster', () => {
  const ts = createTestSeries();
  const model = new SESForecaster(0.3);

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

test('HoltForecaster', () => {
  const ts = createTestSeries();
  const model = new HoltForecaster(0.3, 0.1);

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

test('HoltWintersForecaster', () => {
  const ts = createSeasonalSeries(48, 12);
  const model = new HoltWintersForecaster(0.3, 0.1, 0.1, 12);

  model.fit(ts);
  const forecast = model.predict(12);

  assert.strictEqual(forecast.values.length, 12);
});

test('HoltWintersForecaster.auto', () => {
  const ts = createSeasonalSeries(48, 12);
  const model = HoltWintersForecaster.auto(12, "additive");

  model.fit(ts);
  const forecast = model.predict(12);

  assert.strictEqual(forecast.values.length, 12);
});

// ETS
test('ETSForecaster with components', () => {
  const ts = createSeasonalSeries(48, 12);
  const model = new ETSForecaster("A", "A", "A", 12);

  model.fit(ts);
  const forecast = model.predict(12);

  assert.strictEqual(forecast.values.length, 12);
});

test('ETSForecaster.fromNotation', () => {
  const ts = createSeasonalSeries(48, 12);
  const model = ETSForecaster.fromNotation("AAA", 12);

  model.fit(ts);
  const forecast = model.predict(12);

  assert.strictEqual(forecast.values.length, 12);
});

test('ETSForecaster.isValidSpec', () => {
  // Valid combinations
  assert.strictEqual(ETSForecaster.isValidSpec("A", "A", "A"), true);
  assert.strictEqual(ETSForecaster.isValidSpec("M", "A", "M"), true);

  // Invalid combinations (unstable)
  assert.strictEqual(ETSForecaster.isValidSpec("M", "A", "A"), false);
  assert.strictEqual(ETSForecaster.isValidSpec("M", "Ad", "A"), false);
});

test('ETSForecaster rejects unstable MAA', () => {
  assert.throws(() => {
    new ETSForecaster("M", "A", "A", 12);
  });
});

test('AutoETSForecaster', () => {
  const ts = createTestSeries();
  const model = new AutoETSForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

// Theta
test('ThetaForecaster', () => {
  const ts = createTestSeries();
  const model = new ThetaForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

test('AutoThetaForecaster', () => {
  const ts = createTestSeries();
  const model = new AutoThetaForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

// ARIMA
test('ARIMAForecaster', () => {
  const ts = createTestSeries();
  const model = new ARIMAForecaster(1, 1, 1);

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

test('AutoARIMAForecaster', () => {
  const ts = createTestSeries();
  const model = new AutoARIMAForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

// Intermittent demand
test('CrostonForecaster', () => {
  const values = new Float64Array([0, 0, 5, 0, 0, 0, 3, 0, 0, 7, 0, 0, 4, 0, 0]);
  const ts = new TimeSeries(values);
  const model = new CrostonForecaster();

  model.fit(ts);
  const forecast = model.predict(5);

  assert.strictEqual(forecast.values.length, 5);
});

// Prediction intervals
test('Prediction intervals', () => {
  const ts = createTestSeries();
  const model = new NaiveForecaster();

  model.fit(ts);
  const forecast = model.predictWithIntervals(5, 0.95);

  assert.strictEqual(forecast.values.length, 5);
  assert.strictEqual(forecast.hasLower(), true);
  assert.strictEqual(forecast.hasUpper(), true);

  const lower = forecast.lower;
  const upper = forecast.upper;

  assert.strictEqual(lower.length, 5);
  assert.strictEqual(upper.length, 5);

  // Lower should be less than point forecast, upper should be greater
  for (let i = 0; i < 5; i++) {
    assert.ok(lower[i] < forecast.values[i]);
    assert.ok(upper[i] > forecast.values[i]);
  }
});

// Error handling
test('Predict without fit throws', () => {
  const model = new NaiveForecaster();

  assert.throws(() => {
    model.predict(5);
  });
});

console.log('All tests passed!');
