//! Validation example: Export feature calculations to CSV for comparison with tsfresh.
//!
//! This example reads synthetic time series data from CSV files,
//! extracts features using the Rust features module, and exports
//! the results to CSV for comparison with the Python tsfresh package.
//!
//! Run with: cargo run --example feature_export --release

use anofox_forecast::features::{
    autocorrelation, basic, change, complexity, counting, distribution, entropy, trend,
};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

// Configuration
const DATA_DIR: &str = "validation/data";
const RESULTS_DIR: &str = "validation/results/rust";

const SERIES_TYPES: [&str; 11] = [
    "stationary",
    "trend",
    "seasonal",
    "trend_seasonal",
    "seasonal_negative",
    "multiplicative_seasonal",
    "intermittent",
    "high_frequency",
    "structural_break",
    "long_memory",
    "noisy_seasonal",
];

/// Feature result structure
struct FeatureResult {
    series_type: String,
    feature_name: String,
    value: f64,
}

/// Read a CSV file and return values only
fn read_csv(path: &Path) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue;
        } // Skip header

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let value: f64 = parts[1].trim().parse()?;
            values.push(value);
        }
    }
    Ok(values)
}

/// Extract all features for a series
fn extract_features(series: &[f64], series_type: &str) -> Vec<FeatureResult> {
    let mut results = Vec::new();

    // Helper to add result
    let mut add = |name: &str, value: f64| {
        results.push(FeatureResult {
            series_type: series_type.to_string(),
            feature_name: name.to_string(),
            value,
        });
    };

    // ================= Basic features (basic.rs) =================
    add("value__mean", basic::mean(series));
    add("value__variance", basic::variance(series));
    add(
        "value__standard_deviation",
        basic::standard_deviation(series),
    );
    add("value__median", basic::median(series));
    add("value__minimum", basic::minimum(series));
    add("value__maximum", basic::maximum(series));
    add("value__length", basic::length(series));
    add("value__abs_energy", basic::abs_energy(series));
    add("value__absolute_maximum", basic::absolute_maximum(series));
    add(
        "value__absolute_sum_of_changes",
        basic::absolute_sum_of_changes(series),
    );
    add("value__mean_abs_change", basic::mean_abs_change(series));
    add("value__mean_change", basic::mean_change(series));
    add(
        "value__mean_second_derivative_central",
        basic::mean_second_derivative_central(series),
    );
    add("value__root_mean_square", basic::root_mean_square(series));
    add("value__sum_values", basic::sum_values(series));

    // mean_n_absolute_max with parameters
    for n in [1, 3, 5, 7] {
        add(
            &format!("value__mean_n_absolute_max__number_of_maxima_{}", n),
            basic::mean_n_absolute_max(series, n),
        );
    }

    // ================= Distribution features (distribution.rs) =================
    add("value__skewness", distribution::skewness(series));
    add("value__kurtosis", distribution::kurtosis(series));

    for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
        add(
            &format!("value__quantile__q_{}", q),
            distribution::quantile(series, q),
        );
    }

    add(
        "value__large_standard_deviation__r_0.25",
        if distribution::large_standard_deviation(series, 0.25) {
            1.0
        } else {
            0.0
        },
    );
    add(
        "value__variance_larger_than_standard_deviation",
        if distribution::variance_larger_than_standard_deviation(series) {
            1.0
        } else {
            0.0
        },
    );
    add(
        "value__variation_coefficient",
        distribution::variation_coefficient(series),
    );
    add(
        "value__symmetry_looking__r_0.05",
        if distribution::symmetry_looking(series, 0.05) {
            1.0
        } else {
            0.0
        },
    );

    for r in [1.0, 1.5, 2.0, 2.5, 3.0] {
        add(
            &format!("value__ratio_beyond_r_sigma__r_{}", r),
            distribution::ratio_beyond_r_sigma(series, r),
        );
    }

    // ================= Autocorrelation features (autocorrelation.rs) =================
    for lag in [1, 2, 3, 5, 10] {
        add(
            &format!("value__autocorrelation__lag_{}", lag),
            autocorrelation::autocorrelation(series, lag),
        );
    }

    for lag in [1, 2, 3, 5] {
        add(
            &format!("value__partial_autocorrelation__lag_{}", lag),
            autocorrelation::partial_autocorrelation(series, lag),
        );
    }

    add(
        "value__agg_autocorrelation__f_agg_\"mean\"__maxlag_10",
        autocorrelation::agg_autocorrelation(series, 10, "mean"),
    );
    add(
        "value__agg_autocorrelation__f_agg_\"var\"__maxlag_10",
        autocorrelation::agg_autocorrelation(series, 10, "var"),
    );
    add(
        "value__agg_autocorrelation__f_agg_\"std\"__maxlag_10",
        autocorrelation::agg_autocorrelation(series, 10, "std"),
    );
    add(
        "value__agg_autocorrelation__f_agg_\"median\"__maxlag_10",
        autocorrelation::agg_autocorrelation(series, 10, "median"),
    );

    for lag in [1, 2, 3] {
        add(
            &format!("value__time_reversal_asymmetry_statistic__lag_{}", lag),
            autocorrelation::time_reversal_asymmetry_statistic(series, lag),
        );
    }

    // ================= Counting features (counting.rs) =================
    add(
        "value__count_above_mean",
        counting::count_above_mean(series) as f64,
    );
    add(
        "value__count_below_mean",
        counting::count_below_mean(series) as f64,
    );

    for n in [1, 3, 5] {
        add(
            &format!("value__number_peaks__n_{}", n),
            counting::number_peaks(series, n) as f64,
        );
    }

    add(
        "value__number_crossing_m__m_0",
        counting::number_crossing_m(series, 0.0) as f64,
    );
    add(
        "value__longest_strike_above_mean",
        counting::longest_strike_above_mean(series) as f64,
    );
    add(
        "value__longest_strike_below_mean",
        counting::longest_strike_below_mean(series) as f64,
    );
    add(
        "value__first_location_of_maximum",
        counting::first_location_of_maximum(series),
    );
    add(
        "value__first_location_of_minimum",
        counting::first_location_of_minimum(series),
    );
    add(
        "value__last_location_of_maximum",
        counting::last_location_of_maximum(series),
    );
    add(
        "value__last_location_of_minimum",
        counting::last_location_of_minimum(series),
    );
    add(
        "value__has_duplicate",
        if counting::has_duplicate(series) {
            1.0
        } else {
            0.0
        },
    );
    add(
        "value__has_duplicate_max",
        if counting::has_duplicate_max(series) {
            1.0
        } else {
            0.0
        },
    );
    add(
        "value__has_duplicate_min",
        if counting::has_duplicate_min(series) {
            1.0
        } else {
            0.0
        },
    );

    for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
        add(
            &format!("value__index_mass_quantile__q_{}", q),
            counting::index_mass_quantile(series, q),
        );
    }

    add(
        "value__value_count__value_0",
        counting::value_count(series, 0.0) as f64,
    );
    add(
        "value__range_count__max_1__min_-1",
        counting::range_count(series, -1.0, 1.0) as f64,
    );

    // ================= Entropy features (entropy.rs) =================
    let std = basic::standard_deviation(series);
    let r = if std > 1e-10 { 0.2 * std } else { 0.2 };

    add(
        "value__sample_entropy",
        entropy::sample_entropy(series, 2, r),
    );
    add(
        "value__approximate_entropy__m_2__r_0.2",
        entropy::approximate_entropy(series, 2, r),
    );
    add(
        "value__permutation_entropy__dimension_3__tau_1",
        entropy::permutation_entropy(series, 3, 1),
    );
    add(
        "value__binned_entropy__max_bins_10",
        entropy::binned_entropy(series, 10),
    );

    // ================= Complexity features (complexity.rs) =================
    add(
        "value__cid_ce__normalize_True",
        complexity::cid_ce(series, true),
    );
    add(
        "value__cid_ce__normalize_False",
        complexity::cid_ce(series, false),
    );

    for lag in [1, 2, 3] {
        add(
            &format!("value__c3__lag_{}", lag),
            complexity::c3(series, lag),
        );
    }

    add(
        "value__lempel_ziv_complexity__bins_10",
        complexity::lempel_ziv_complexity(series, 10),
    );

    // ================= Trend features (trend.rs) =================
    let lt = trend::linear_trend(series);
    add("value__linear_trend__attr_\"slope\"", lt.slope);
    add("value__linear_trend__attr_\"intercept\"", lt.intercept);
    add("value__linear_trend__attr_\"rvalue\"", lt.r_squared.sqrt());
    add("value__linear_trend__attr_\"stderr\"", lt.stderr);
    add("value__linear_trend__attr_\"pvalue\"", lt.p_value);

    // agg_linear_trend with different configurations
    add(
        "value__agg_linear_trend__attr_\"slope\"__chunk_len_10__f_agg_\"mean\"",
        trend::agg_linear_trend(series, 10, "mean", "slope"),
    );
    add(
        "value__agg_linear_trend__attr_\"slope\"__chunk_len_10__f_agg_\"var\"",
        trend::agg_linear_trend(series, 10, "var", "slope"),
    );
    add(
        "value__agg_linear_trend__attr_\"intercept\"__chunk_len_10__f_agg_\"mean\"",
        trend::agg_linear_trend(series, 10, "mean", "intercept"),
    );
    add(
        "value__agg_linear_trend__attr_\"rvalue\"__chunk_len_10__f_agg_\"mean\"",
        trend::agg_linear_trend(series, 10, "mean", "rvalue"),
    );

    // AR coefficients (OLS-based, k=10 means AR(10) model)
    // coeff 0 is intercept, coeffs 1-3 are AR parameters
    for coeff in [0, 1, 2, 3] {
        add(
            &format!("value__ar_coefficient__coeff_{}__k_10", coeff),
            trend::ar_coefficient(series, 10, coeff),
        );
    }

    add(
        "value__augmented_dickey_fuller__attr_\"teststat\"",
        trend::augmented_dickey_fuller(series),
    );

    // ================= Change features (change.rs) =================
    add(
        "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_1.0__ql_0.0",
        change::change_quantiles(series, 0.0, 1.0, true, "mean"),
    );
    add(
        "value__change_quantiles__f_agg_\"mean\"__isabs_False__qh_1.0__ql_0.0",
        change::change_quantiles(series, 0.0, 1.0, false, "mean"),
    );
    add(
        "value__change_quantiles__f_agg_\"var\"__isabs_True__qh_0.75__ql_0.25",
        change::change_quantiles(series, 0.25, 0.75, true, "var"),
    );

    for i in 0..5 {
        add(
            &format!(
                "value__energy_ratio_by_chunks__num_segments_10__segment_focus_{}",
                i
            ),
            change::energy_ratio_by_chunks(series, 10, i),
        );
    }

    add(
        "value__percentage_of_reoccurring_datapoints_to_all_datapoints",
        change::percentage_of_reoccurring_datapoints_to_all_datapoints(series),
    );
    add(
        "value__percentage_of_reoccurring_values_to_all_values",
        change::percentage_of_reoccurring_values_to_all_values(series),
    );
    add(
        "value__ratio_value_number_to_time_series_length",
        change::ratio_value_number_to_time_series_length(series),
    );
    add(
        "value__sum_of_reoccurring_data_points",
        change::sum_of_reoccurring_data_points(series),
    );
    add(
        "value__sum_of_reoccurring_values",
        change::sum_of_reoccurring_values(series),
    );

    results
}

/// Write features to CSV
fn write_features(results: &[FeatureResult], path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "series_type,feature_name,value")?;

    for result in results {
        writeln!(
            file,
            "{},{},{}",
            result.series_type, result.feature_name, result.value
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rust Feature Validation Export ===\n");

    // Create results directory
    fs::create_dir_all(RESULTS_DIR)?;

    let mut all_results: Vec<FeatureResult> = Vec::new();

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
        let values = read_csv(&csv_path)?;
        println!("  Loaded {} observations", values.len());

        // Extract features
        let features = extract_features(&values, series_type);
        println!("  Extracted {} features", features.len());

        all_results.extend(features);
    }

    // Write results
    let output_path = Path::new(RESULTS_DIR).join("features.csv");
    write_features(&all_results, &output_path)?;

    println!("\n=== Export Complete ===");
    println!("Output: {:?}", output_path);
    println!(
        "Total features exported: {} ({} series x {} features/series)",
        all_results.len(),
        SERIES_TYPES.len(),
        all_results.len() / SERIES_TYPES.len().max(1)
    );

    Ok(())
}
