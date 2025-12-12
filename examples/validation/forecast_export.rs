//! Validation example: Export forecasts to CSV for comparison with statsforecast.
//!
//! This example reads synthetic time series data from CSV files,
//! runs various forecasting models, and exports the results to CSV
//! for comparison with the Python statsforecast package.
//!
//! Run with: cargo run --example forecast_export

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::arima::{AutoARIMA, AutoARIMAConfig, ARIMA, SARIMA};
use anofox_forecast::models::baseline::SeasonalWindowAverage;
use anofox_forecast::models::baseline::{
    HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive, WindowAverage,
};
use anofox_forecast::models::exponential::{
    AutoETS, AutoETSConfig, ETSSpec, HoltLinearTrend, HoltWinters, SeasonalES, SeasonalType,
    SimpleExponentialSmoothing, ETS,
};
use anofox_forecast::models::garch::GARCH;
use anofox_forecast::models::intermittent::{Croston, ADIDA, IMAPA, TSB};
use anofox_forecast::models::mfles::MFLES;
use anofox_forecast::models::mstl_forecaster::{MSTLForecaster, TrendForecastMethod};
use anofox_forecast::models::tbats::{AutoTBATS, TBATS};
use anofox_forecast::models::theta::{
    AutoTheta, DynamicOptimizedTheta, DynamicTheta, OptimizedTheta, Theta,
};
use anofox_forecast::models::Forecaster;
use chrono::{NaiveDateTime, TimeZone, Utc};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

// Configuration
const HORIZON: usize = 12;
const SEASONAL_PERIOD: usize = 12;
const CONFIDENCE_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];

// Data directories (relative to project root)
const DATA_DIR: &str = "validation/data";
const RESULTS_DIR: &str = "validation/results/rust";

/// Series types to process
const SERIES_TYPES: [&str; 11] = [
    "stationary",
    "trend",
    "seasonal",
    "trend_seasonal",
    "seasonal_negative",       // Has negative values - tests fallback to additive
    "multiplicative_seasonal", // True multiplicative seasonality
    "intermittent",            // Sparse demand data with zeros
    "high_frequency",          // Multiple seasonalities (MSTL test)
    "structural_break",        // Level shift (robustness test)
    "long_memory",             // ARFIMA-like slow decay
    "noisy_seasonal",          // High noise-to-signal ratio
];

/// Read a CSV file and return timestamps and values
fn read_csv(
    path: &Path,
) -> Result<(Vec<chrono::DateTime<Utc>>, Vec<f64>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            // Skip header
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            // Parse timestamp (format: 2020-01-01 00:00:00 or 2020-01-01)
            let ts_str = parts[0].trim();
            let naive =
                NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S").or_else(|_| {
                    chrono::NaiveDate::parse_from_str(ts_str, "%Y-%m-%d")
                        .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
                })?;
            let ts = Utc.from_utc_datetime(&naive);
            timestamps.push(ts);

            // Parse value
            let value: f64 = parts[1].trim().parse()?;
            values.push(value);
        }
    }

    Ok((timestamps, values))
}

/// Forecast result structure
struct ForecastResult {
    model_name: String,
    series_type: String,
    point_forecasts: Vec<f64>,
    intervals: HashMap<String, (Vec<f64>, Vec<f64>)>, // level -> (lower, upper)
}

/// Run a single model and return results
fn run_model<F: Forecaster>(
    model: &mut F,
    ts: &TimeSeries,
    model_name: &str,
    series_type: &str,
    has_native_intervals: bool,
) -> Option<ForecastResult> {
    // Fit the model
    if model.fit(ts).is_err() {
        eprintln!("  Warning: {} failed to fit on {}", model_name, series_type);
        return None;
    }

    // Get point forecasts
    let point_forecast = match model.predict(HORIZON) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "  Warning: {} failed to predict on {}: {}",
                model_name, series_type, e
            );
            return None;
        }
    };
    let point_forecasts = point_forecast.primary().to_vec();

    // Get confidence intervals for each level (only if model has native support)
    let mut intervals = HashMap::new();
    if has_native_intervals {
        for &level in &CONFIDENCE_LEVELS {
            match model.predict_with_intervals(HORIZON, level) {
                Ok(forecast_ci) => {
                    if let (Ok(lower), Ok(upper)) =
                        (forecast_ci.lower_series(0), forecast_ci.upper_series(0))
                    {
                        let level_key = format!("{:.0}", level * 100.0);
                        intervals.insert(level_key, (lower.to_vec(), upper.to_vec()));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: {} CI level {} failed on {}: {}",
                        model_name, level, series_type, e
                    );
                }
            }
        }
    }

    Some(ForecastResult {
        model_name: model_name.to_string(),
        series_type: series_type.to_string(),
        point_forecasts,
        intervals,
    })
}

/// Write point forecasts to CSV
fn write_point_forecasts(results: &[ForecastResult], path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Header
    writeln!(file, "series_type,model,step,forecast")?;

    for result in results {
        for (i, &forecast) in result.point_forecasts.iter().enumerate() {
            writeln!(
                file,
                "{},{},{},{}",
                result.series_type,
                result.model_name,
                i + 1,
                forecast
            )?;
        }
    }

    Ok(())
}

/// Write confidence intervals to CSV
fn write_confidence_intervals(results: &[ForecastResult], path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Header
    writeln!(file, "series_type,model,step,level,lower,upper")?;

    for result in results {
        for (level, (lower, upper)) in &result.intervals {
            for i in 0..lower.len() {
                writeln!(
                    file,
                    "{},{},{},{},{},{}",
                    result.series_type,
                    result.model_name,
                    i + 1,
                    level,
                    lower[i],
                    upper[i]
                )?;
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rust Forecast Validation Export ===\n");

    // Create results directory
    fs::create_dir_all(RESULTS_DIR)?;

    let mut all_results: Vec<ForecastResult> = Vec::new();

    // Process each series type
    for series_type in SERIES_TYPES {
        println!("Processing {} series...", series_type);

        let csv_path = Path::new(DATA_DIR).join(format!("{}.csv", series_type));
        if !csv_path.exists() {
            eprintln!("  Error: Data file not found: {:?}", csv_path);
            eprintln!("  Run 'python generate_data.py' first to create the data files.");
            continue;
        }

        // Read data
        let (timestamps, values) = read_csv(&csv_path)?;
        let ts = TimeSeries::univariate(timestamps, values)?;

        println!("  Loaded {} observations", ts.len());

        // Run each model
        // Models with native CI support: Naive, SeasonalNaive, RandomWalkWithDrift, Holt, HoltWinters, ARIMA, AutoARIMA, AutoETS, Theta
        // Models without native CI support: SES, Croston, CrostonSBA, TSB

        // 1. Naive - has native intervals
        {
            let mut model = Naive::new();
            if let Some(result) = run_model(&mut model, &ts, "Naive", series_type, true) {
                all_results.push(result);
                println!("  ✓ Naive");
            }
        }

        // 2. Seasonal Naive - has native intervals
        {
            let mut model = SeasonalNaive::new(SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "SeasonalNaive", series_type, true) {
                all_results.push(result);
                println!("  ✓ SeasonalNaive");
            }
        }

        // 3. Random Walk with Drift - has native intervals
        {
            let mut model = RandomWalkWithDrift::new();
            if let Some(result) =
                run_model(&mut model, &ts, "RandomWalkWithDrift", series_type, true)
            {
                all_results.push(result);
                println!("  ✓ RandomWalkWithDrift");
            }
        }

        // 4. Simple Exponential Smoothing - NO native intervals
        // Use fixed alpha=0.1 to match statsforecast validation
        {
            let mut model = SimpleExponentialSmoothing::new(0.1);
            if let Some(result) = run_model(&mut model, &ts, "SES", series_type, false) {
                all_results.push(result);
                println!("  ✓ SES (point only)");
            }
        }

        // 5. Holt's Linear Trend - has native intervals
        // statsforecast's Holt is actually ETS(A,A,N), so we use that for comparison
        {
            let mut model = ETS::new(ETSSpec::aan(), SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "Holt", series_type, true) {
                all_results.push(result);
                println!("  ✓ Holt");
            }
        }

        // 6. Holt-Winters - has native intervals
        {
            let mut model = HoltWinters::auto(SEASONAL_PERIOD, SeasonalType::Additive);
            if let Some(result) = run_model(&mut model, &ts, "HoltWinters", series_type, true) {
                all_results.push(result);
                println!("  ✓ HoltWinters");
            }
        }

        // 7. ARIMA(1,1,1) - has native intervals
        {
            let mut model = ARIMA::new(1, 1, 1);
            if let Some(result) = run_model(&mut model, &ts, "ARIMA_1_1_1", series_type, true) {
                all_results.push(result);
                println!("  ✓ ARIMA(1,1,1)");
            }
        }

        // 8. AutoARIMA with SARIMA support - has native intervals
        {
            let config = AutoARIMAConfig::default()
                .with_seasonal_period(SEASONAL_PERIOD)
                .with_seasonal_orders(1, 1, 1);
            let mut model = AutoARIMA::with_config(config);
            if let Some(result) = run_model(&mut model, &ts, "AutoARIMA", series_type, true) {
                all_results.push(result);
                println!("  ✓ AutoARIMA (with SARIMA)");
            }
        }

        // 8b. SARIMA(1,1,1)(1,1,1)[12] - explicit seasonal ARIMA
        {
            let mut model = SARIMA::new(1, 1, 1, 1, 1, 1, SEASONAL_PERIOD);
            if let Some(result) =
                run_model(&mut model, &ts, "SARIMA_1_1_1_1_1_1_12", series_type, true)
            {
                all_results.push(result);
                println!("  ✓ SARIMA(1,1,1)(1,1,1)[12]");
            }
        }

        // 9. AutoETS - has native intervals
        {
            let config = AutoETSConfig::with_period(SEASONAL_PERIOD);
            let mut model = AutoETS::with_config(config);
            if let Some(result) = run_model(&mut model, &ts, "AutoETS", series_type, true) {
                all_results.push(result);
                println!("  ✓ AutoETS");
            }
        }

        // 10. Theta - has native intervals
        // Use seasonal variant for seasonal data to match statsforecast behavior
        {
            let mut model = Theta::seasonal(SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "Theta", series_type, true) {
                all_results.push(result);
                println!("  ✓ Theta");
            }
        }

        // 11. Croston (classic) - NO native intervals
        {
            let mut model = Croston::new();
            if let Some(result) = run_model(&mut model, &ts, "Croston", series_type, false) {
                all_results.push(result);
                println!("  ✓ Croston (point only)");
            }
        }

        // 12. Croston SBA - NO native intervals
        {
            let mut model = Croston::new().sba();
            if let Some(result) = run_model(&mut model, &ts, "CrostonSBA", series_type, false) {
                all_results.push(result);
                println!("  ✓ CrostonSBA (point only)");
            }
        }

        // 13. TSB - NO native intervals
        {
            let mut model = TSB::new();
            if let Some(result) = run_model(&mut model, &ts, "TSB", series_type, false) {
                all_results.push(result);
                println!("  ✓ TSB (point only)");
            }
        }

        // 14. ADIDA - NO native intervals
        {
            let mut model = ADIDA::new();
            if let Some(result) = run_model(&mut model, &ts, "ADIDA", series_type, false) {
                all_results.push(result);
                println!("  ✓ ADIDA (point only)");
            }
        }

        // 15. SeasonalWindowAverage - NO native intervals in statsforecast
        {
            let mut model = SeasonalWindowAverage::new(SEASONAL_PERIOD, 2);
            if let Some(result) =
                run_model(&mut model, &ts, "SeasonalWindowAverage", series_type, false)
            {
                all_results.push(result);
                println!("  ✓ SeasonalWindowAverage (point only)");
            }
        }

        // 16. IMAPA - Intermittent Multiple Aggregation Prediction Algorithm
        {
            let mut model = IMAPA::new();
            if let Some(result) = run_model(&mut model, &ts, "IMAPA", series_type, false) {
                all_results.push(result);
                println!("  ✓ IMAPA (point only)");
            }
        }

        // 17. OptimizedTheta - Theta with optimized parameters (seasonal)
        {
            let mut model = OptimizedTheta::seasonal(SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "OptimizedTheta", series_type, true) {
                all_results.push(result);
                println!("  ✓ OptimizedTheta");
            }
        }

        // 18. DynamicTheta - Dynamic coefficient updates (with seasonal decomposition)
        {
            let mut model = DynamicTheta::seasonal(SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "DynamicTheta", series_type, true) {
                all_results.push(result);
                println!("  ✓ DynamicTheta");
            }
        }

        // 18b. DynamicOptimizedTheta - Dynamic with optimized parameters (with seasonal)
        {
            let mut model = DynamicOptimizedTheta::seasonal_optimized(SEASONAL_PERIOD);
            if let Some(result) =
                run_model(&mut model, &ts, "DynamicOptimizedTheta", series_type, true)
            {
                all_results.push(result);
                println!("  ✓ DynamicOptimizedTheta");
            }
        }

        // 19. AutoTheta - Automatic Theta model selection (seasonal)
        {
            let mut model = AutoTheta::seasonal(SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "AutoTheta", series_type, true) {
                all_results.push(result);
                println!("  ✓ AutoTheta");
            }
        }

        // 20. MSTLForecaster - MSTL decomposition based forecaster
        {
            let mut model = MSTLForecaster::new(vec![SEASONAL_PERIOD]);
            if let Some(result) = run_model(&mut model, &ts, "MSTLForecaster", series_type, true) {
                all_results.push(result);
                println!("  ✓ MSTLForecaster");
            }
        }

        // 21. MFLES - Gradient boosted decomposition
        {
            let mut model = MFLES::new(vec![SEASONAL_PERIOD]);
            if let Some(result) = run_model(&mut model, &ts, "MFLES", series_type, true) {
                all_results.push(result);
                // Debug output for noisy_seasonal
                if series_type == "noisy_seasonal" {
                    let (trend, penalty, seasonality, is_mult) = model.debug_state();
                    eprintln!("  DEBUG MFLES noisy_seasonal:");
                    eprintln!("    is_multiplicative: {}", is_mult);
                    eprintln!("    trend: {:?}", trend);
                    eprintln!("    penalty: {:?}", penalty);
                    if let Some(s) = seasonality {
                        eprintln!("    seasonality (first 5): {:?}", &s[..5.min(s.len())]);
                    }
                }
                println!("  ✓ MFLES");
            }
        }

        // 22. SeasonalES - Multiplicative seasonal exponential smoothing
        {
            let mut model = SeasonalES::new(SEASONAL_PERIOD);
            if let Some(result) = run_model(&mut model, &ts, "SeasonalES", series_type, true) {
                all_results.push(result);
                println!("  ✓ SeasonalES");
            }
        }

        // 23. TBATS - Complex seasonality model
        {
            let mut model = TBATS::new(vec![SEASONAL_PERIOD]);
            if let Some(result) = run_model(&mut model, &ts, "TBATS", series_type, true) {
                all_results.push(result);
                println!("  ✓ TBATS");
            }
        }

        // 24. AutoTBATS - Automatic TBATS selection
        {
            let mut model = AutoTBATS::new(vec![SEASONAL_PERIOD]);
            if let Some(result) = run_model(&mut model, &ts, "AutoTBATS", series_type, true) {
                all_results.push(result);
                println!("  ✓ AutoTBATS");
            }
        }

        // 25. GARCH - Volatility modeling (for stationary/financial series)
        {
            let mut model = GARCH::new(1, 1);
            if let Some(result) = run_model(&mut model, &ts, "GARCH", series_type, true) {
                all_results.push(result);
                println!("  ✓ GARCH");
            }
        }

        // 26. HistoricAverage - Mean of all historical values
        {
            let mut model = HistoricAverage::new();
            if let Some(result) = run_model(&mut model, &ts, "HistoricAverage", series_type, false)
            {
                all_results.push(result);
                println!("  ✓ HistoricAverage (point only)");
            }
        }

        // 27. WindowAverage - Mean of last N observations
        {
            let mut model = WindowAverage::new(12); // Same as statsforecast default window_size=12
            if let Some(result) = run_model(&mut model, &ts, "WindowAverage", series_type, false) {
                all_results.push(result);
                println!("  ✓ WindowAverage (point only)");
            }
        }

        println!();
    }

    // Write results
    println!("Writing results...");

    let point_path = Path::new(RESULTS_DIR).join("point_forecasts.csv");
    write_point_forecasts(&all_results, &point_path)?;
    println!("  ✓ Point forecasts: {:?}", point_path);

    let ci_path = Path::new(RESULTS_DIR).join("confidence_intervals.csv");
    write_confidence_intervals(&all_results, &ci_path)?;
    println!("  ✓ Confidence intervals: {:?}", ci_path);

    println!("\n=== Export Complete ===");
    println!(
        "Total forecasts exported: {} model/series combinations",
        all_results.len()
    );

    Ok(())
}
