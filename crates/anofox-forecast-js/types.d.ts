/**
 * TypeScript type definitions for anofox-forecast-js
 *
 * WebAssembly bindings for the anofox-forecast time series forecasting library.
 *
 * @packageDocumentation
 */

// =============================================================================
// Core Types
// =============================================================================

/**
 * A univariate time series with values indexed by position.
 *
 * Timestamps are optional and can be provided as milliseconds since Unix epoch.
 */
export class TimeSeries {
  /**
   * Create a new time series from an array of values.
   *
   * Timestamps are automatically generated as sequential integers.
   *
   * @param values - Array of numeric values
   * @throws Error if the time series cannot be constructed
   */
  constructor(values: Float64Array);

  /**
   * Create a time series with explicit timestamps.
   *
   * @param values - Array of numeric values
   * @param timestamps_ms - Array of timestamps as milliseconds since Unix epoch
   * @returns A new TimeSeries instance
   * @throws Error if values and timestamps lengths do not match
   */
  static withTimestamps(values: Float64Array, timestamps_ms: Float64Array): TimeSeries;

  /** The number of observations in the series. */
  readonly length: number;

  /** All values in the series as a Float64Array. */
  readonly values: Float64Array;

  /**
   * Check if the series is empty.
   *
   * @returns true if the series contains no observations
   */
  isEmpty(): boolean;

  /**
   * Get a slice of the time series.
   *
   * @param start - Start index (inclusive)
   * @param end - End index (exclusive)
   * @returns A new TimeSeries containing the specified range
   * @throws Error if the range is invalid
   */
  slice(start: number, end: number): TimeSeries;

  /**
   * Check if the series contains missing values (NaN).
   *
   * @returns true if any value is NaN
   */
  hasMissingValues(): boolean;

  /**
   * Attach calendar annotations (holidays, regressors) to this time series.
   *
   * Models that support exogenous variables will automatically use the
   * calendar annotations during fitting.
   *
   * @param calendar - CalendarAnnotations instance
   */
  setCalendar(calendar: CalendarAnnotations): void;

  /**
   * Check if calendar annotations are attached.
   *
   * @returns true if annotations have been set
   */
  hasCalendar(): boolean;

  /**
   * Remove calendar annotations from this time series.
   */
  clearCalendar(): void;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Forecast result containing point predictions and optional prediction intervals.
 */
export class Forecast {
  /** The forecast horizon (number of predictions). */
  readonly horizon: number;

  /** Point predictions as a Float64Array. */
  readonly values: Float64Array;

  /** Lower prediction interval bounds, or undefined if not available. */
  readonly lower: Float64Array | undefined;

  /** Upper prediction interval bounds, or undefined if not available. */
  readonly upper: Float64Array | undefined;

  /**
   * Check if lower prediction interval bounds are available.
   *
   * @returns true if lower bounds exist
   */
  hasLower(): boolean;

  /**
   * Check if upper prediction interval bounds are available.
   *
   * @returns true if upper bounds exist
   */
  hasUpper(): boolean;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Calendar Annotations
// =============================================================================

/**
 * Calendar annotations for holidays and external regressors.
 *
 * Use this to define holidays, events, promotions, or any named
 * numeric regressors that can influence a forecast. Attach to a
 * TimeSeries via `setCalendar()` before fitting a model that
 * supports exogenous variables.
 *
 * @example
 * ```typescript
 * const cal = new CalendarAnnotations();
 * cal.addHoliday(new Date("2024-12-25").getTime());
 * cal.addHoliday(new Date("2025-01-01").getTime());
 * cal.addRegressor("promo", new Float64Array([0, 0, 1, 1, 0, 0, 0]));
 *
 * const ts = TimeSeries.withTimestamps(values, timestamps);
 * ts.setCalendar(cal);
 * ```
 */
export class CalendarAnnotations {
  /** Create an empty CalendarAnnotations instance. */
  constructor();

  /**
   * Add a single holiday date.
   *
   * @param timestamp_ms - Holiday date as milliseconds since Unix epoch
   */
  addHoliday(timestamp_ms: number): void;

  /**
   * Set multiple holiday dates at once (replaces any existing holidays).
   *
   * @param timestamps_ms - Array of holiday dates as milliseconds since Unix epoch
   */
  setHolidays(timestamps_ms: Float64Array): void;

  /**
   * Add a named regressor with values aligned to the time series.
   *
   * The values array must have the same length as the time series this
   * will be attached to.
   *
   * @param name - Name of the regressor (e.g., "promo", "temperature")
   * @param values - Array of numeric values, one per time step
   */
  addRegressor(name: string, values: Float64Array): void;

  /**
   * Get the number of holidays.
   *
   * @returns Number of registered holidays
   */
  holidayCount(): number;

  /**
   * Get the number of named regressors.
   *
   * @returns Number of registered regressors
   */
  regressorCount(): number;

  /**
   * Get holiday dates as milliseconds since Unix epoch.
   *
   * @returns Array of holiday timestamps
   */
  getHolidays(): Float64Array;

  /**
   * Get the names of all registered regressors.
   *
   * @returns Array of regressor names
   * @throws Error if serialization fails
   */
  regressorNames(): string[];

  /**
   * Get values for a named regressor.
   *
   * @param name - Regressor name
   * @returns Array of values, or undefined if the regressor does not exist
   */
  getRegressor(name: string): Float64Array | undefined;

  /**
   * Check whether any regressors have been added.
   *
   * @returns true if at least one regressor exists
   */
  hasRegressors(): boolean;

  /**
   * Check if a specific date is a holiday.
   *
   * @param timestamp_ms - Date to check as milliseconds since Unix epoch
   * @returns true if the date matches any registered holiday
   */
  isHoliday(timestamp_ms: number): boolean;

  /**
   * Check if a specific date is a business day (not weekend, not holiday).
   *
   * @param timestamp_ms - Date to check as milliseconds since Unix epoch
   * @returns true if the date is a weekday and not a holiday
   */
  isBusinessDay(timestamp_ms: number): boolean;

  /**
   * Serialize the calendar annotations to a JSON string.
   *
   * @returns JSON string representation
   * @throws Error if serialization fails
   */
  toJSON(): string;

  /**
   * Deserialize calendar annotations from a JSON string.
   *
   * @param json - JSON string produced by `toJSON()`
   * @returns A new CalendarAnnotations instance
   * @throws Error if the JSON is invalid
   */
  static fromJSON(json: string): CalendarAnnotations;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Standalone Functions
// =============================================================================

/**
 * Get the library version string.
 *
 * @returns The version of the anofox-forecast library (e.g. "0.4.5")
 */
export function version(): string;

// =============================================================================
// Feature Extraction Functions
// =============================================================================

/**
 * Compute the autocorrelation of a time series at a specific lag.
 *
 * @param values - Array of numeric values
 * @param lag - Lag value
 * @returns Autocorrelation coefficient at the given lag
 */
export function autocorrelation(values: Float64Array, lag: number): number;

/**
 * Compute the partial autocorrelation of a time series at a specific lag.
 *
 * Uses the Durbin-Levinson algorithm.
 *
 * @param values - Array of numeric values
 * @param lag - Lag value (must be >= 1)
 * @returns Partial autocorrelation coefficient at the given lag
 */
export function partialAutocorrelation(values: Float64Array, lag: number): number;

/**
 * Compute the arithmetic mean of a time series.
 *
 * @param values - Array of numeric values
 * @returns Arithmetic mean, or NaN if empty
 */
export function mean(values: Float64Array): number;

/**
 * Compute the population variance of a time series.
 *
 * Uses population formula (n denominator) matching tsfresh.
 *
 * @param values - Array of numeric values
 * @returns Population variance, or NaN if empty
 */
export function variance(values: Float64Array): number;

/**
 * Compute the skewness (third standardized moment) of a time series.
 *
 * Measures the asymmetry of the distribution.
 *
 * @param values - Array of numeric values
 * @returns Skewness, or NaN if fewer than 3 values
 */
export function skewness(values: Float64Array): number;

/**
 * Compute the excess kurtosis (fourth standardized moment) of a time series.
 *
 * Measures the "tailedness" of the distribution. A normal distribution has
 * excess kurtosis of 0.
 *
 * @param values - Array of numeric values
 * @returns Excess kurtosis, or NaN if fewer than 4 values
 */
export function kurtosis(values: Float64Array): number;

/**
 * Compute the approximate entropy of a time series.
 *
 * Measures the complexity/regularity of a time series, including self-matches.
 *
 * @param values - Array of numeric values
 * @param m - Embedding dimension (typically 2)
 * @param r - Tolerance (typically 0.2 * standard deviation)
 * @returns Approximate entropy, or NaN if insufficient data
 */
export function approximateEntropy(values: Float64Array, m: number, r: number): number;

/**
 * Compute the sample entropy of a time series.
 *
 * Measures the complexity/regularity of a time series. Lower values indicate
 * more regularity. Unlike approximate entropy, excludes self-matches.
 *
 * @param values - Array of numeric values
 * @param m - Embedding dimension (typically 2)
 * @param r - Tolerance (typically 0.2 * standard deviation)
 * @returns Sample entropy, or NaN if insufficient data
 */
export function sampleEntropy(values: Float64Array, m: number, r: number): number;

// =============================================================================
// Changepoint Detection
// =============================================================================

/** Cost function type for changepoint detection. */
export type CostFunction =
  | "l1"
  | "L1"
  | "l2"
  | "L2"
  | "normal"
  | "Normal"
  | "poisson"
  | "Poisson"
  | "linearTrend"
  | "LinearTrend"
  | "meanVariance"
  | "MeanVariance"
  | "cusum"
  | "Cusum"
  | "CUSUM";

/** Result of PELT changepoint detection. */
export interface PeltResult {
  /** Detected changepoint indices. */
  changepoints: number[];
  /** Segment boundaries as [start, end] pairs. */
  segments: [number, number][];
  /** Total cost (excluding penalty). */
  cost: number;
  /** Number of changepoints detected. */
  nChangepoints: number;
  /** Mean value for each segment. */
  segmentMeans: number[];
}

/**
 * Detect changepoints in a time series using the PELT algorithm.
 *
 * @param values - Array of numeric values
 * @param penalty - Penalty for each changepoint (higher = fewer changepoints)
 * @param costFunction - Cost function (default: "l2")
 * @param minSegmentLength - Minimum segment length (default: 2)
 * @returns Object with changepoint detection results
 * @throws Error if the cost function name is invalid
 */
export function detectChangepoints(
  values: Float64Array,
  penalty: number,
  costFunction?: CostFunction,
  minSegmentLength?: number,
): PeltResult;

/**
 * Detect changepoints using BIC penalty (penalty = log(n)).
 *
 * BIC penalty automatically adapts to series length, providing a good
 * default for most use cases.
 *
 * @param values - Array of numeric values
 * @param costFunction - Cost function (default: "l2")
 * @returns Object with changepoint detection results
 * @throws Error if the cost function name is invalid
 */
export function detectChangepointsBic(
  values: Float64Array,
  costFunction?: CostFunction,
): PeltResult;

// =============================================================================
// Decomposition
// =============================================================================

/** Result of STL decomposition. */
export interface StlResult {
  /** Trend component. */
  trend: number[];
  /** Seasonal component. */
  seasonal: number[];
  /** Remainder component. */
  remainder: number[];
  /** Seasonal strength (0 to 1). */
  seasonalStrength: number;
  /** Trend strength (0 to 1). */
  trendStrength: number;
}

/** Result of MSTL decomposition. */
export interface MstlResult {
  /** Trend component. */
  trend: number[];
  /** Seasonal components (one array per period). */
  seasonals: number[][];
  /** The seasonal periods corresponding to each component. */
  seasonalPeriods: number[];
  /** Remainder component. */
  remainder: number[];
  /** Trend strength (0 to 1). */
  trendStrength: number;
}

/**
 * Perform STL (Seasonal-Trend decomposition using LOESS).
 *
 * Decomposes a time series into trend, seasonal, and remainder components.
 *
 * @param values - Array of numeric values
 * @param period - Seasonal period (e.g., 12 for monthly data)
 * @param robust - Enable robust fitting to reduce outlier influence (default: false)
 * @returns Object with decomposition components and strength measures
 * @throws Error if data length is less than 2 * period
 */
export function stlDecompose(
  values: Float64Array,
  period: number,
  robust?: boolean,
): StlResult;

/**
 * Perform MSTL (Multiple Seasonal-Trend decomposition using LOESS).
 *
 * Decomposes a time series with multiple seasonal periods into trend,
 * multiple seasonal components, and remainder.
 *
 * @param values - Array of numeric values
 * @param periods - Array of seasonal periods (e.g., [7, 365] for daily data)
 * @returns Object with decomposition components
 * @throws Error if data length is less than 2 * max(periods)
 */
export function mstlDecompose(
  values: Float64Array,
  periods: Uint32Array | number[],
): MstlResult;

// =============================================================================
// Validation / Statistical Tests
// =============================================================================

/** Result of the Ljung-Box test. */
export interface LjungBoxResult {
  /** Test statistic Q. */
  statistic: number;
  /** P-value (approximate). */
  pValue: number;
  /** Number of lags tested. */
  lags: number;
  /** Degrees of freedom. */
  df: number;
  /** Whether residuals appear to be white noise at 5% significance. */
  isWhiteNoise: boolean;
}

/** Result of the Durbin-Watson test. */
export interface DurbinWatsonResult {
  /** Test statistic (0 to 4). */
  statistic: number;
  /** Interpretation of the autocorrelation type. */
  interpretation:
    | "positiveStrong"
    | "positiveWeak"
    | "none"
    | "negativeWeak"
    | "negativeStrong";
}

/** Result of the Jarque-Bera test. */
export interface JarqueBeraResult {
  /** Test statistic. */
  statistic: number;
  /** P-value (chi-squared with df=2). */
  pValue: number;
  /** Whether residuals appear normally distributed at 5% significance. */
  isNormal: boolean;
  /** Skewness of the residuals. */
  skewness: number;
  /** Excess kurtosis of the residuals. */
  excessKurtosis: number;
}

/** Comprehensive residual diagnostics result. */
export interface DiagnoseResidualsResult {
  /** Ljung-Box test results. */
  ljungBox: LjungBoxResult;
  /** Durbin-Watson test results. */
  durbinWatson: DurbinWatsonResult;
  /** Jarque-Bera test results. */
  jarqueBera: JarqueBeraResult;
  /** Residual mean. */
  mean: number;
  /** Residual variance. */
  variance: number;
  /** Number of residuals. */
  n: number;
  /** Overall assessment: true if residuals pass all tests at 5% significance. */
  isAdequate: boolean;
}

/** Critical values at common significance levels. */
export interface CriticalValues {
  /** Critical value at 1% significance. */
  cv1pct: number;
  /** Critical value at 5% significance. */
  cv5pct: number;
  /** Critical value at 10% significance. */
  cv10pct: number;
}

/** Result of a stationarity test (ADF or KPSS). */
export interface StationarityResult {
  /** Test statistic. */
  statistic: number;
  /** P-value (approximate). */
  pValue: number;
  /** Number of lags used. */
  lags: number;
  /** Whether the series appears stationary. */
  isStationary: boolean;
  /** Critical values at 1%, 5%, and 10% significance levels. */
  criticalValues: CriticalValues;
}

/**
 * Perform the Ljung-Box test for autocorrelation in residuals.
 *
 * Tests the null hypothesis that residuals are independently distributed
 * (white noise). A low p-value suggests significant autocorrelation remains.
 *
 * @param residuals - Array of model residuals
 * @param lags - Number of lags to include (default: min(10, n/5))
 * @param fittedParams - Number of fitted model parameters for df adjustment (default: 0)
 * @returns Object with test statistic, p-value, and white noise assessment
 * @throws Error if computation fails
 */
export function ljungBox(
  residuals: Float64Array,
  lags?: number,
  fittedParams?: number,
): LjungBoxResult;

/**
 * Perform the Durbin-Watson test for first-order autocorrelation.
 *
 * The statistic ranges from 0 to 4:
 * - Near 0: Strong positive autocorrelation
 * - Near 2: No autocorrelation
 * - Near 4: Strong negative autocorrelation
 *
 * @param residuals - Array of model residuals
 * @returns Object with test statistic and interpretation
 * @throws Error if computation fails
 */
export function durbinWatson(residuals: Float64Array): DurbinWatsonResult;

/**
 * Perform the Jarque-Bera test for normality of residuals.
 *
 * Tests the null hypothesis that residuals are normally distributed by
 * examining skewness and kurtosis.
 *
 * @param residuals - Array of model residuals
 * @returns Object with test statistic, p-value, and normality assessment
 * @throws Error if computation fails
 */
export function jarqueBera(residuals: Float64Array): JarqueBeraResult;

/**
 * Run comprehensive residual diagnostics.
 *
 * Combines Ljung-Box (autocorrelation), Durbin-Watson (first-order autocorrelation),
 * and Jarque-Bera (normality) tests into a single diagnostic report.
 *
 * @param residuals - Array of model residuals
 * @param fittedParams - Number of fitted model parameters (default: 0)
 * @returns Comprehensive diagnostics object
 * @throws Error if computation fails
 */
export function diagnoseResiduals(
  residuals: Float64Array,
  fittedParams?: number,
): DiagnoseResidualsResult;

/**
 * Perform the Augmented Dickey-Fuller (ADF) test for unit root.
 *
 * Tests the null hypothesis that the series has a unit root (non-stationary).
 * Rejection (low p-value) implies stationarity.
 *
 * @param values - Array of numeric values
 * @param maxLags - Maximum lags to include (default: (n-1)^(1/3))
 * @returns Object with test statistic, p-value, and stationarity assessment
 * @throws Error if computation fails
 */
export function adfTest(values: Float64Array, maxLags?: number): StationarityResult;

/**
 * Perform the KPSS test for stationarity.
 *
 * Tests the null hypothesis that the series is (level) stationary.
 * Rejection (high statistic) implies non-stationarity.
 *
 * Note: ADF and KPSS test opposite null hypotheses:
 * - ADF: H0 = non-stationary, reject => stationary
 * - KPSS: H0 = stationary, reject => non-stationary
 *
 * @param values - Array of numeric values
 * @param lags - Number of lags for HAC variance (default: 4*(n/100)^0.25)
 * @returns Object with test statistic, p-value, and stationarity assessment
 * @throws Error if computation fails
 */
export function kpssTest(values: Float64Array, lags?: number): StationarityResult;

// =============================================================================
// Cross-Validation
// =============================================================================

/** Supported model types for cross-validation and bootstrap forecasting. */
export type ModelType =
  | "naive"
  | "ses"
  | "simpleexponentialsmoothing"
  | "holt"
  | "holtlineartrend"
  | "autoarima"
  | "autoets"
  | "autotheta"
  | `sma${number}`;

/** Result of cross-validation. */
export interface CrossValidationResult {
  /** Mean RMSE across folds. */
  rmse: number;
  /** Mean MAE across folds. */
  mae: number;
  /** Mean MAPE across folds (NaN if unavailable). */
  mape: number;
  /** Mean SMAPE across folds. */
  smape: number;
  /** Number of folds evaluated. */
  folds: number;
  /** Standard deviation of MAE across folds. */
  maeStd: number;
  /** Standard deviation of RMSE across folds. */
  rmseStd: number;
}

/**
 * Perform time series cross-validation with expanding window.
 *
 * Evaluates a forecasting model using multiple train/test splits where
 * the training window grows with each fold.
 *
 * @param values - Array of numeric values
 * @param timestamps - Optional array of timestamps as milliseconds since epoch
 * @param modelType - Model type string (e.g., "naive", "ses", "autoarima", "sma5")
 * @param horizon - Forecast horizon for each fold
 * @param initialWindow - Initial training window size (default: max(10, length/3))
 * @returns Object with aggregated cross-validation metrics
 * @throws Error if the model type is unknown
 */
export function crossValidate(
  values: Float64Array,
  timestamps: Float64Array | undefined | null,
  modelType: ModelType | string,
  horizon: number,
  initialWindow?: number,
): CrossValidationResult;

// =============================================================================
// Bootstrap Forecast
// =============================================================================

/** Result of bootstrap forecast. */
export interface BootstrapResult {
  /** Point forecast values. */
  point: number[];
  /** Lower prediction interval bounds. */
  lower: number[];
  /** Upper prediction interval bounds. */
  upper: number[];
  /** Confidence level used. */
  level: number;
  /** Number of bootstrap samples used. */
  nSamples: number;
}

/**
 * Generate a bootstrap forecast with empirical prediction intervals.
 *
 * Uses residual bootstrap: resamples fitted residuals, generates synthetic
 * series, re-fits the model, and collects forecast distributions to
 * compute confidence intervals.
 *
 * @param values - Array of numeric values
 * @param timestamps - Optional array of timestamps as milliseconds since epoch
 * @param modelType - Model type string (e.g., "naive", "ses", "autoarima", "sma5")
 * @param horizon - Number of steps to forecast
 * @param level - Confidence level (e.g., 0.95 for 95% intervals)
 * @param nSamples - Number of bootstrap samples (default: 200)
 * @returns Object with point forecast and prediction interval bounds
 * @throws Error if the model type is unknown
 */
export function bootstrapForecast(
  values: Float64Array,
  timestamps: Float64Array | undefined | null,
  modelType: ModelType | string,
  horizon: number,
  level: number,
  nSamples?: number,
): BootstrapResult;

// =============================================================================
// Baseline Forecasters
// =============================================================================

/**
 * Naive forecaster -- uses the last observation as the forecast for all future periods.
 */
export class NaiveForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Mean (Historic Average) forecaster -- uses the historical mean as the forecast.
 */
export class MeanForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Seasonal Naive forecaster -- repeats observations from the same season.
 */
export class SeasonalNaiveForecaster {
  /**
   * @param period - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
   */
  constructor(period: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Random Walk with Drift forecaster.
 */
export class RandomWalkDriftForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Simple Moving Average forecaster.
 */
export class SMAForecaster {
  /**
   * @param window - Window size for the moving average
   */
  constructor(window: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Window Average forecaster -- uses the mean of the last N observations.
 */
export class WindowAverageForecaster {
  /**
   * @param windowSize - Size of the rolling window
   */
  constructor(windowSize: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Seasonal Window Average forecaster.
 */
export class SeasonalWindowAverageForecaster {
  /**
   * @param period - Seasonal period
   * @param window - Number of seasonal cycles to average
   */
  constructor(period: number, window: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Exponential Smoothing Forecasters
// =============================================================================

/**
 * Simple Exponential Smoothing (SES) forecaster.
 */
export class SESForecaster {
  /**
   * @param alpha - Smoothing parameter (0 < alpha <= 1)
   */
  constructor(alpha: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Holt Linear Trend (Double Exponential Smoothing) forecaster.
 */
export class HoltForecaster {
  /**
   * @param alpha - Level smoothing parameter (0 < alpha <= 1)
   * @param beta - Trend smoothing parameter (0 < beta <= 1)
   */
  constructor(alpha: number, beta: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Holt-Winters (Triple Exponential Smoothing) forecaster.
 *
 * The constructor creates a model with additive seasonality. Use the static
 * `multiplicative` method for multiplicative seasonality, or `auto` for
 * automatic parameter optimization.
 */
export class HoltWintersForecaster {
  /**
   * Create with additive seasonality.
   *
   * @param alpha - Level smoothing parameter
   * @param beta - Trend smoothing parameter
   * @param gamma - Seasonal smoothing parameter
   * @param period - Seasonal period
   */
  constructor(alpha: number, beta: number, gamma: number, period: number);

  /**
   * Create with multiplicative seasonality.
   *
   * @param alpha - Level smoothing parameter
   * @param beta - Trend smoothing parameter
   * @param gamma - Seasonal smoothing parameter
   * @param period - Seasonal period
   * @returns A new HoltWintersForecaster with multiplicative seasonality
   */
  static multiplicative(
    alpha: number,
    beta: number,
    gamma: number,
    period: number,
  ): HoltWintersForecaster;

  /**
   * Create with automatic parameter optimization.
   *
   * @param period - Seasonal period
   * @param seasonalType - Seasonal type: "additive" (or "a") or "multiplicative" (or "m")
   * @returns A new HoltWintersForecaster with optimized parameters
   * @throws Error if seasonalType is invalid
   */
  static auto(period: number, seasonalType: string): HoltWintersForecaster;

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Seasonal Exponential Smoothing forecaster.
 */
export class SeasonalESForecaster {
  /**
   * @param period - Seasonal period
   */
  constructor(period: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * ETS (Error-Trend-Seasonal) state-space model.
 *
 * Follows the ETS taxonomy from Forecasting: Principles and Practice (FPP3).
 *
 * Use string codes: "A" = Additive, "M" = Multiplicative, "N" = None.
 * Some combinations are invalid/unstable per FPP3 (MAA, MAdA).
 */
export class ETSForecaster {
  /**
   * Create an ETS model with specified components.
   *
   * @param error - Error type: "A" (additive) or "M" (multiplicative)
   * @param trend - Trend type: "N" (none), "A" (additive), or "Ad" (additive damped)
   * @param seasonal - Seasonal type: "N" (none), "A" (additive), or "M" (multiplicative)
   * @param period - Seasonal period (ignored if seasonal is "N")
   * @throws Error if the combination is unstable (MAA or MAdA)
   */
  constructor(error: string, trend: string, seasonal: string, period: number);

  /**
   * Create an ETS model from standard notation.
   *
   * Format: ErrorTrendSeasonal (e.g., "ANN", "AAA", "MAM", "AAdM")
   *
   * @param notation - ETS notation string
   * @param period - Seasonal period (required if notation has seasonal component)
   * @returns A new ETSForecaster
   * @throws Error for invalid notation or unstable combinations
   */
  static fromNotation(notation: string, period: number): ETSForecaster;

  /**
   * Check if an ETS specification is valid/stable.
   *
   * @param error - Error type: "A" or "M"
   * @param trend - Trend type: "N", "A", or "Ad"
   * @param seasonal - Seasonal type: "N", "A", or "M"
   * @returns true if the combination is stable and usable
   */
  static isValidSpec(error: string, trend: string, seasonal: string): boolean;

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * AutoETS -- automatic ETS model selection.
 *
 * Follows the ETS taxonomy from FPP3. Searches over valid ETS specifications
 * and selects the best model based on information criteria.
 */
export class AutoETSForecaster {
  /** Create AutoETS with default configuration. */
  constructor();

  /**
   * Create AutoETS with a specific seasonal period.
   *
   * @param period - Seasonal period
   * @returns A new AutoETSForecaster configured with the given period
   */
  static withPeriod(period: number): AutoETSForecaster;

  /**
   * Create AutoETS restricted to additive models only.
   *
   * Excludes multiplicative error and multiplicative seasonality.
   *
   * @returns A new AutoETSForecaster restricted to additive models
   */
  static additiveOnly(): AutoETSForecaster;

  /**
   * Create AutoETS with custom configuration.
   *
   * @param period - Optional seasonal period (null/undefined for auto-detection)
   * @param allowMultiplicativeError - Allow multiplicative error models
   * @param allowMultiplicativeSeasonal - Allow multiplicative seasonality
   * @param allowDamped - Allow damped trend models
   * @returns A new AutoETSForecaster with the specified configuration
   */
  static withConfig(
    period: number | undefined | null,
    allowMultiplicativeError: boolean,
    allowMultiplicativeSeasonal: boolean,
    allowDamped: boolean,
  ): AutoETSForecaster;

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Theta Forecasters
// =============================================================================

/**
 * Theta forecaster -- the standard Theta method.
 */
export class ThetaForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Optimized Theta forecaster -- automatically optimizes parameters.
 */
export class OptimizedThetaForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Dynamic Theta forecaster -- updates coefficients dynamically.
 */
export class DynamicThetaForecaster {
  /**
   * @param alpha - Smoothing parameter for the forecast
   */
  constructor(alpha: number);

  /**
   * Create an optimized Dynamic Theta model.
   *
   * @returns A new DynamicThetaForecaster with optimized parameters
   */
  static optimized(): DynamicThetaForecaster;

  /**
   * Create a seasonal Dynamic Theta model.
   *
   * @param period - Seasonal period
   * @returns A new DynamicThetaForecaster configured for seasonal data
   */
  static seasonal(period: number): DynamicThetaForecaster;

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * AutoTheta -- automatic Theta model selection.
 */
export class AutoThetaForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// ARIMA Forecasters
// =============================================================================

/**
 * ARIMA (Autoregressive Integrated Moving Average) forecaster.
 */
export class ARIMAForecaster {
  /**
   * @param p - AR order (autoregressive)
   * @param d - Differencing order
   * @param q - MA order (moving average)
   */
  constructor(p: number, d: number, q: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * SARIMA (Seasonal ARIMA) forecaster.
 */
export class SARIMAForecaster {
  /**
   * @param p - AR order
   * @param d - Differencing order
   * @param q - MA order
   * @param seasonalP - Seasonal AR order
   * @param seasonalD - Seasonal differencing order
   * @param seasonalQ - Seasonal MA order
   * @param period - Seasonal period
   */
  constructor(
    p: number,
    d: number,
    q: number,
    seasonalP: number,
    seasonalD: number,
    seasonalQ: number,
    period: number,
  );

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * AutoARIMA -- automatic ARIMA order selection.
 */
export class AutoARIMAForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Intermittent Demand Forecasters
// =============================================================================

/**
 * Croston's method for intermittent demand forecasting.
 *
 * Suitable for data with many zero values (sparse demand).
 */
export class CrostonForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * TSB (Teunter-Syntetos-Babai) method for intermittent demand.
 *
 * An improvement over Croston's method with separate smoothing for
 * demand probability and demand size.
 */
export class TSBForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * ADIDA (Aggregate-Disaggregate Intermittent Demand Approach).
 *
 * Aggregates data to remove zero periods, forecasts at the aggregated level,
 * and disaggregates back to the original frequency.
 */
export class ADIDAForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * IMAPA (Intermittent Multiple Aggregation Prediction Algorithm).
 *
 * Uses multiple temporal aggregation levels to produce robust forecasts
 * for intermittent demand data.
 */
export class IMAPAForecaster {
  constructor();

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Advanced Forecasters
// =============================================================================

/**
 * TBATS (Trigonometric seasonality, Box-Cox, ARMA errors, Trend, Seasonal) forecaster.
 *
 * Designed for complex seasonal patterns, including multiple seasonal periods,
 * non-integer periods, and high-frequency data.
 */
export class TBATSForecaster {
  /**
   * @param seasonalPeriods - Array of seasonal periods (e.g., [7, 365] for daily data)
   */
  constructor(seasonalPeriods: Uint32Array | number[]);

  /**
   * Create a TBATSForecaster with specified seasonal periods.
   *
   * @param periods - Array of seasonal periods
   * @returns A new TBATSForecaster
   */
  static withSeasonalPeriods(periods: Uint32Array | number[]): TBATSForecaster;

  /**
   * Enable Box-Cox transformation.
   *
   * @param lambda - Box-Cox parameter (0 = log, 1 = identity)
   */
  setBoxCox(lambda: number): void;

  /**
   * Enable damped trend.
   *
   * @param phi - Damping parameter (typically 0.8-0.99)
   */
  setDampedTrend(phi: number): void;

  /**
   * Set ARMA error orders.
   *
   * @param p - AR order
   * @param q - MA order
   */
  setArma(p: number, q: number): void;

  /**
   * Set Fourier K (number of harmonics) for each seasonal period.
   *
   * @param k - Array of K values (one per seasonal period)
   */
  setFourierK(k: Uint32Array | number[]): void;

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * AutoTBATS -- automatic TBATS model selection.
 */
export class AutoTBATSForecaster {
  /**
   * @param seasonalPeriods - Array of seasonal periods
   */
  constructor(seasonalPeriods: Uint32Array | number[]);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * MFLES (Multiple Frequency Locally Estimated Scatterplot Smoothing) forecaster.
 */
export class MFLESForecaster {
  /**
   * @param seasonalPeriods - Array of seasonal periods
   */
  constructor(seasonalPeriods: Uint32Array | number[]);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * MSTL Forecaster -- Multiple Seasonal-Trend decomposition using LOESS
 * combined with a trend forecaster.
 */
export class MSTLForecasterWrapper {
  /**
   * @param seasonalPeriods - Array of seasonal periods
   */
  constructor(seasonalPeriods: Uint32Array | number[]);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * GARCH (Generalized Autoregressive Conditional Heteroskedasticity) forecaster.
 *
 * Models time-varying volatility in time series data.
 */
export class GARCHForecaster {
  /**
   * @param p - GARCH order (lagged variance terms)
   * @param q - ARCH order (lagged squared residuals)
   */
  constructor(p: number, q: number);

  /**
   * Fit the model to a time series.
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Ensemble Forecasters
// =============================================================================

/** Supported model name strings for the EnsembleForecaster constructor. */
export type EnsembleModelName =
  | "naive"
  | "mean"
  | "historicaverage"
  | "rwdrift"
  | "randomwalkwithdrift"
  | "ses"
  | "simpleexponentialsmoothing"
  | "holt"
  | "holtlineartrend"
  | "autoarima"
  | "autoets"
  | "autotheta"
  | `sma${number}`
  | `wa${number}`;

/**
 * Ensemble forecaster that combines multiple models.
 *
 * Supports mean, median, weighted MSE, and custom-weight combination.
 * Models are specified by name strings.
 */
export class EnsembleForecaster {
  /**
   * Create an ensemble from an array of model name strings.
   *
   * Supported names: "naive", "mean", "rwdrift", "ses", "holt",
   * "autoarima", "autoets", "autotheta", "sma5", "wa10", etc.
   *
   * @param modelNames - Array of model name strings
   * @throws Error if any model name is unknown
   */
  constructor(modelNames: (EnsembleModelName | string)[]);

  /**
   * Set custom combination weights.
   *
   * Weights are normalized to sum to 1. Length must match number of models.
   *
   * @param weights - Array of combination weights
   */
  setWeights(weights: Float64Array | number[]): void;

  /** Set the combination method to median. */
  setMedian(): void;

  /** Set the combination method to weighted MSE. */
  setWeightedMse(): void;

  /**
   * Fit all models in the ensemble.
   *
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails for any model
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values using the combined ensemble.
   *
   * @param horizon - Number of steps to forecast
   * @returns Forecast with combined point predictions
   * @throws Error if the ensemble has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   *
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with combined predictions and intervals
   * @throws Error if the ensemble has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** Get the number of models in the ensemble. */
  modelCount(): number;

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// Auto Model Selection
// =============================================================================

/** Score entry for model comparison. */
export interface ModelScore {
  /** Model name. */
  name: string;
  /** Model score (lower is better). */
  score: number;
}

/**
 * Automatic model selection across ARIMA, ETS, and Theta families.
 *
 * Fits all enabled auto models and selects the best one based on
 * in-sample MSE or cross-validation error.
 */
export class AutoForecaster {
  /**
   * Create a new AutoForecaster with default configuration.
   *
   * By default, includes AutoARIMA, AutoETS, and AutoTheta candidates
   * and uses in-sample MSE for model selection.
   */
  constructor();

  /**
   * Create a seasonal AutoForecaster.
   *
   * @param period - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
   * @returns A new AutoForecaster configured for seasonal data
   */
  static seasonal(period: number): AutoForecaster;

  /**
   * Create an AutoForecaster with custom configuration.
   *
   * @param seasonalPeriod - Seasonal period (0 or undefined for non-seasonal)
   * @param includeArima - Include AutoARIMA candidate (default: true)
   * @param includeEts - Include AutoETS candidate (default: true)
   * @param includeTheta - Include AutoTheta candidate (default: true)
   * @param useCrossValidation - Use cross-validation instead of in-sample MSE (default: false)
   * @returns A new AutoForecaster with the specified configuration
   */
  static withConfig(
    seasonalPeriod?: number,
    includeArima?: boolean,
    includeEts?: boolean,
    includeTheta?: boolean,
    useCrossValidation?: boolean,
  ): AutoForecaster;

  /**
   * Fit the model to a time series.
   *
   * Fits all candidate models and selects the best one.
   *
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   *
   * @param horizon - Number of steps to forecast
   * @returns Forecast with point predictions
   * @throws Error if the model has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   *
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with point predictions and intervals
   * @throws Error if the model has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /**
   * Get the name of the selected model.
   *
   * @returns Name of the best model, or undefined if not yet fitted
   */
  selectedModelName(): string | undefined;

  /**
   * Get all candidate scores as an array of { name, score } objects,
   * sorted by score (ascending).
   *
   * @returns Array of model scores
   * @throws Error if serialization fails
   */
  allScores(): ModelScore[];

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Automatic ensemble that selects top-K models across families.
 *
 * Fits AutoARIMA, AutoETS, and AutoTheta, ranks them by in-sample MSE,
 * and combines the top-K into a weighted ensemble forecast.
 */
export class AutoEnsembleForecaster {
  /**
   * Create a new AutoEnsembleForecaster with default configuration.
   *
   * By default, uses top-3 models with weighted MSE combination.
   */
  constructor();

  /**
   * Create a seasonal AutoEnsembleForecaster.
   *
   * @param period - Seasonal period
   * @returns A new AutoEnsembleForecaster configured for seasonal data
   */
  static seasonal(period: number): AutoEnsembleForecaster;

  /**
   * Create with custom configuration.
   *
   * @param topK - Number of top models to include in ensemble (default: 3)
   * @param seasonalPeriod - Seasonal period (0 or undefined for non-seasonal)
   * @returns A new AutoEnsembleForecaster with the specified configuration
   */
  static withConfig(
    topK?: number,
    seasonalPeriod?: number,
  ): AutoEnsembleForecaster;

  /**
   * Fit the ensemble to a time series.
   *
   * Fits all candidate models, selects the top-K, and builds the ensemble.
   *
   * @param series - TimeSeries to fit
   * @throws Error if fitting fails
   */
  fit(series: TimeSeries): void;

  /**
   * Predict future values.
   *
   * @param horizon - Number of steps to forecast
   * @returns Forecast with combined point predictions
   * @throws Error if the ensemble has not been fitted
   */
  predict(horizon: number): Forecast;

  /**
   * Predict with prediction intervals.
   *
   * @param horizon - Number of steps to forecast
   * @param level - Confidence level (e.g., 0.95 for 95% intervals)
   * @returns Forecast with combined predictions and intervals
   * @throws Error if the ensemble has not been fitted
   */
  predictWithIntervals(horizon: number, level: number): Forecast;

  /** Get the number of models in the ensemble. */
  modelCount(): number;

  /**
   * Get all candidate scores as an array of { name, score } objects,
   * sorted by score (ascending).
   *
   * @returns Array of model scores
   * @throws Error if serialization fails
   */
  allScores(): ModelScore[];

  /** The model name. */
  readonly name: string;

  /** Release WASM memory associated with this object. */
  free(): void;
}

// =============================================================================
// PostProcessing — Prediction Intervals & Calibration
// =============================================================================

/**
 * Point forecasts used as input for postprocessing methods.
 */
export class JsPointForecasts {
  /**
   * Create point forecasts from an array of values.
   * @param values - Array of forecast values
   */
  constructor(values: Float64Array | number[]);

  /** The number of forecast points. */
  readonly length: number;

  /** The forecast values. */
  readonly values: Float64Array;

  /** Check if empty. */
  isEmpty(): boolean;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Prediction intervals with lower/upper bounds and a coverage level.
 */
export class JsPredictionIntervals {
  /** Lower bounds of intervals. */
  readonly lower: Float64Array;

  /** Upper bounds of intervals. */
  readonly upper: Float64Array;

  /** The coverage level (e.g., 0.90). */
  readonly coverage: number;

  /** The number of intervals. */
  readonly length: number;

  /** Get the interval widths. */
  widths(): Float64Array;

  /** Get the interval midpoints. */
  midpoints(): Float64Array;

  /**
   * Compute empirical coverage given actual values.
   * @param actuals - Actual observed values
   * @returns Fraction of actuals within the intervals, or undefined if lengths mismatch
   */
  empiricalCoverage(actuals: Float64Array | number[]): number | undefined;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Conformal prediction result storing calibration data.
 */
export class JsConformalResult {
  /** The quantile value (interval half-width). */
  quantileValue(): number;

  /** The coverage level. */
  readonly coverage: number;

  /** The nonconformity scores. */
  scores(): Float64Array;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Conformal predictor for distribution-free prediction intervals.
 *
 * Provides coverage-guaranteed intervals without distributional assumptions.
 */
export class JsConformalPredictor {
  /**
   * Create a split conformal predictor.
   * @param coverage - Target coverage level in (0, 1), e.g. 0.90
   */
  constructor(coverage: number);

  /**
   * Create with the cross-validation method.
   * @param coverage - Target coverage level in (0, 1)
   * @param nFolds - Number of cross-validation folds
   */
  static crossVal(coverage: number, nFolds: number): JsConformalPredictor;

  /**
   * Create with the Jackknife+ method.
   * @param coverage - Target coverage level in (0, 1)
   */
  static jackknifePlus(coverage: number): JsConformalPredictor;

  /**
   * Calibrate the predictor on historical forecasts and actuals.
   * @param forecasts - Historical point forecast values
   * @param actuals - Corresponding actual observed values
   * @throws Error if lengths mismatch or insufficient data
   */
  calibrate(
    forecasts: Float64Array | number[],
    actuals: Float64Array | number[],
  ): JsConformalResult;

  /**
   * Generate prediction intervals for new point forecasts.
   * @param result - Fitted result from calibrate()
   * @param pointForecasts - New point forecast values
   */
  predictIntervals(
    result: JsConformalResult,
    pointForecasts: Float64Array | number[],
  ): JsPredictionIntervals;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Normal prediction result storing Gaussian fit parameters.
 */
export class JsNormalResult {
  /** Mean error (bias). */
  readonly mean: number;

  /** Standard deviation of errors. */
  stdDev(): number;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Normal predictor assuming Gaussian forecast errors.
 *
 * Generates quantile-based prediction intervals.
 */
export class JsNormalPredictor {
  /**
   * Create a new normal predictor.
   * @param quantiles - Sorted quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9]
   */
  constructor(quantiles: Float64Array | number[]);

  /**
   * Fit the predictor on historical forecasts and actuals.
   * @param forecasts - Historical point forecast values
   * @param actuals - Corresponding actual observed values
   * @throws Error if lengths mismatch or insufficient data
   */
  fit(
    forecasts: Float64Array | number[],
    actuals: Float64Array | number[],
  ): JsNormalResult;

  /**
   * Generate prediction intervals for new point forecasts.
   * @param result - Fitted result from fit()
   * @param pointForecasts - New point forecast values
   * @throws Error if fewer than 2 quantiles were specified
   */
  predictIntervals(
    result: JsNormalResult,
    pointForecasts: Float64Array | number[],
  ): JsPredictionIntervals;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Historical simulation result storing the empirical error distribution.
 */
export class JsHistoricalSimResult {
  /** The sorted errors. */
  errors(): Float64Array;

  /** The quantile values. */
  quantileValues(): Float64Array;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Historical simulator for non-parametric prediction intervals.
 *
 * Uses the empirical error distribution to generate intervals.
 */
export class JsHistoricalSimulator {
  /**
   * Create a new historical simulator.
   * @param quantiles - Sorted quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9]
   */
  constructor(quantiles: Float64Array | number[]);

  /**
   * Create a simulator with a rolling window.
   * @param quantiles - Sorted quantile levels in (0, 1)
   * @param window - Number of recent observations to use
   */
  static withWindow(
    quantiles: Float64Array | number[],
    window: number,
  ): JsHistoricalSimulator;

  /**
   * Fit the simulator on historical forecasts and actuals.
   * @param forecasts - Historical point forecast values
   * @param actuals - Corresponding actual observed values
   * @throws Error if lengths mismatch or data is empty
   */
  simulate(
    forecasts: Float64Array | number[],
    actuals: Float64Array | number[],
  ): JsHistoricalSimResult;

  /**
   * Generate prediction intervals for new point forecasts.
   * @param result - Fitted result from simulate()
   * @param pointForecasts - New point forecast values
   * @throws Error if fewer than 2 quantiles were specified
   */
  predictIntervals(
    result: JsHistoricalSimResult,
    pointForecasts: Float64Array | number[],
  ): JsPredictionIntervals;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Opaque handle to a trained postprocessing model.
 *
 * Returned by JsPostProcessor.train() and consumed by predictIntervals().
 */
export class JsTrainedModel {
  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Unified postprocessor wrapping conformal, normal, and historical simulation
 * methods behind a single API.
 */
export class JsPostProcessor {
  /**
   * Create a conformal prediction postprocessor.
   * @param coverage - Target coverage level in (0, 1), e.g. 0.90
   */
  static conformal(coverage: number): JsPostProcessor;

  /**
   * Create a normal prediction postprocessor.
   * @param quantiles - Sorted quantile levels in (0, 1)
   */
  static normal(quantiles: Float64Array | number[]): JsPostProcessor;

  /**
   * Create a historical simulation postprocessor.
   * @param quantiles - Sorted quantile levels in (0, 1)
   */
  static historicalSim(quantiles: Float64Array | number[]): JsPostProcessor;

  /**
   * Train the postprocessor on historical data.
   * @param forecasts - JsPointForecasts with historical predictions
   * @param actuals - Corresponding actual observed values
   * @throws Error if training fails
   */
  train(
    forecasts: JsPointForecasts,
    actuals: Float64Array | number[],
  ): JsTrainedModel;

  /**
   * Generate prediction intervals from a trained model.
   * @param trained - A JsTrainedModel from train()
   * @param forecasts - New point forecasts
   * @throws Error if prediction fails
   */
  predictIntervals(
    trained: JsTrainedModel,
    forecasts: JsPointForecasts,
  ): JsPredictionIntervals;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Configuration for backtesting a postprocessor.
 */
export class JsBacktestConfig {
  /**
   * Create a new backtest configuration with default settings.
   *
   * Defaults: initialWindow=50, step=1, horizon=1, expanding=true.
   */
  constructor();

  /** Set the initial training window size. Returns self for chaining. */
  initialWindow(size: number): JsBacktestConfig;

  /** Set the step size between folds. Returns self for chaining. */
  step(step: number): JsBacktestConfig;

  /** Set the forecast horizon. Returns self for chaining. */
  horizon(horizon: number): JsBacktestConfig;

  /** Set whether to use expanding (true) or rolling (false) window. Returns self for chaining. */
  expanding(expanding: boolean): JsBacktestConfig;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Results from backtesting a postprocessor.
 */
export class JsBacktestResult {
  /** Get the number of backtest folds. */
  numFolds(): number;

  /** Overall coverage across all folds. */
  readonly coverage: number;

  /** Get average interval width across all folds. */
  intervalWidths(): number;

  /**
   * Get calibration error (absolute deviation from target coverage).
   * @param targetCoverage - The target coverage to compare against
   */
  calibrationError(targetCoverage: number): number;

  /** Release WASM memory associated with this object. */
  free(): void;
}

/**
 * Run a backtest on a postprocessor.
 *
 * @param processor - The postprocessor to evaluate
 * @param forecasts - All historical point forecasts
 * @param actuals - All historical actual values
 * @param config - Backtest configuration
 * @throws Error if backtesting fails (e.g., insufficient data)
 */
export function backtestPostProcessor(
  processor: JsPostProcessor,
  forecasts: JsPointForecasts,
  actuals: Float64Array | number[],
  config: JsBacktestConfig,
): JsBacktestResult;

// =============================================================================
// WASM Initialization
// =============================================================================

/**
 * Initialize the WASM module.
 *
 * When using the default wasm-pack output, this is typically done by importing
 * and calling the default export or the `init` function from the generated JS glue.
 *
 * @example
 * ```typescript
 * import init, { TimeSeries, AutoForecaster } from 'anofox-forecast-js';
 *
 * await init();
 *
 * const ts = new TimeSeries(new Float64Array([1, 2, 3, 4, 5]));
 * const model = new AutoForecaster();
 * model.fit(ts);
 * const forecast = model.predict(3);
 * console.log(forecast.values);
 * ```
 */
export default function init(
  input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module,
): Promise<void>;
