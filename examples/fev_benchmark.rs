//! autogluon/fev-style benchmark across the Monash Time Series Forecasting
//! Archive datasets that make up the Chronos benchmark's classical panel.
//!
//! Reports MASE (fev's canonical point metric) with fev's seasonal-naive
//! scaling per dataset. Runs the anofox-forecast α-25 stack against
//! internal baselines (AutoETS, AutoTheta).
//!
//! Datasets: m3_monthly, m4_hourly/daily/weekly/monthly/quarterly/yearly,
//! tourism_monthly/quarterly, cif_2016. All Monash `.tsf` format.
//!
//! Run: `cargo run --release --features distributional --example fev_benchmark`
//! Configure: `SAMPLE_PER=200` (default: all) to limit series per dataset.

use anofox_forecast::core::TimeSeries;
use anofox_forecast::models::exponential::AutoETS;
use anofox_forecast::models::theta::AutoTheta;
use anofox_forecast::models::Forecaster;

#[cfg(feature = "distributional")]
use anofox_forecast::models::{LaplaceForecaster, SmartForecaster};

use chrono::{Duration, TimeZone, Utc};
use std::fs;
use std::time::Instant;

/// One Monash-formatted dataset with its canonical horizon and seasonal
/// period (from the fev / Chronos benchmark and the Monash archive
/// metadata).
struct Dataset {
    name: &'static str,
    path: &'static str,
    horizon: usize,
    period: usize,
    /// Fev / Monash used this many months/days/weeks per Duration step
    /// for TimeSeries construction — we don't need real dates, just a
    /// consistent monotonic spacing.
    step_seconds: i64,
}

const DATASETS: &[Dataset] = &[
    // fev's Chronos benchmark tasks we can match with Monash data.
    // (name, path, horizon, period, step_seconds) — horizons match
    // fev's task definitions.
    Dataset {
        name: "m3_monthly",
        path: "validation/data/m3_monthly.tsf",
        horizon: 18,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "m3_quarterly",
        path: "validation/data/m3_quarterly.tsf",
        horizon: 8,
        period: 4,
        step_seconds: 90 * 86400,
    },
    Dataset {
        name: "m3_yearly",
        path: "validation/data/m3_yearly.tsf",
        horizon: 6,
        period: 1,
        step_seconds: 365 * 86400,
    },
    Dataset {
        name: "m1_monthly",
        path: "validation/data/m1_monthly.tsf",
        horizon: 18,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "m1_quarterly",
        path: "validation/data/m1_quarterly.tsf",
        horizon: 8,
        period: 4,
        step_seconds: 90 * 86400,
    },
    Dataset {
        name: "m1_yearly",
        path: "validation/data/m1_yearly.tsf",
        horizon: 6,
        period: 1,
        step_seconds: 365 * 86400,
    },
    Dataset {
        name: "m4_hourly",
        path: "validation/data/m4_hourly.tsf",
        horizon: 48,
        period: 24,
        step_seconds: 3600,
    },
    Dataset {
        name: "m4_daily",
        path: "validation/data/m4_daily.tsf",
        horizon: 14,
        period: 7,
        step_seconds: 86400,
    },
    Dataset {
        name: "m4_weekly",
        path: "validation/data/m4_weekly.tsf",
        horizon: 13,
        period: 1,
        step_seconds: 7 * 86400,
    },
    Dataset {
        name: "m4_monthly",
        path: "validation/data/m4_monthly.tsf",
        horizon: 18,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "m4_quarterly",
        path: "validation/data/m4_quarterly.tsf",
        horizon: 8,
        period: 4,
        step_seconds: 90 * 86400,
    },
    Dataset {
        name: "m4_yearly",
        path: "validation/data/m4_yearly.tsf",
        horizon: 6,
        period: 1,
        step_seconds: 365 * 86400,
    },
    Dataset {
        name: "tourism_monthly",
        path: "validation/data/tourism_monthly.tsf",
        horizon: 24,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "tourism_quarterly",
        path: "validation/data/tourism_quarterly.tsf",
        horizon: 8,
        period: 4,
        step_seconds: 90 * 86400,
    },
    Dataset {
        name: "tourism_yearly",
        path: "validation/data/tourism_yearly.tsf",
        horizon: 4,
        period: 1,
        step_seconds: 365 * 86400,
    },
    Dataset {
        name: "cif_2016",
        path: "validation/data/cif_2016.tsf",
        horizon: 12,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "nn5_weekly",
        path: "validation/data/nn5_weekly.tsf",
        horizon: 8,
        period: 1,
        step_seconds: 7 * 86400,
    },
    Dataset {
        name: "covid_deaths",
        path: "validation/data/covid_deaths.tsf",
        horizon: 30,
        period: 1,
        step_seconds: 86400,
    },
    Dataset {
        name: "fred_md",
        path: "validation/data/fred_md.tsf",
        horizon: 12,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "hospital",
        path: "validation/data/hospital.tsf",
        horizon: 12,
        period: 12,
        step_seconds: 30 * 86400,
    },
    Dataset {
        name: "australian_electricity",
        path: "validation/data/australian_electricity.tsf",
        horizon: 48,
        period: 48,
        step_seconds: 1800,
    },
    // α-27 addition: the 9 fev tasks hosted on HuggingFace Datasets
    // (chronos_datasets / chronos_datasets_extra), converted to Monash
    // TSF via /tmp/convert_hf.py.
    Dataset {
        name: "m5",
        path: "validation/data/m5.tsf",
        horizon: 28,
        period: 1,
        step_seconds: 86400,
    },
    Dataset {
        name: "nn5",
        path: "validation/data/nn5.tsf",
        horizon: 56,
        period: 1,
        step_seconds: 86400,
    },
    Dataset {
        name: "exchange_rate",
        path: "validation/data/exchange_rate.tsf",
        horizon: 30,
        period: 5,
        step_seconds: 86400,
    },
    Dataset {
        name: "dominick",
        path: "validation/data/dominick.tsf",
        horizon: 8,
        period: 1,
        step_seconds: 7 * 86400,
    },
    Dataset {
        name: "ercot",
        path: "validation/data/ercot.tsf",
        horizon: 24,
        period: 24,
        step_seconds: 3600,
    },
    Dataset {
        name: "car_parts",
        path: "validation/data/car_parts.tsf",
        horizon: 12,
        period: 12,
        step_seconds: 30 * 86400,
    },
    // traffic, ETTh, ETTm dropped — long histories × period 24-96 make the
    // classical AutoETS grid search (19 ETS variants × Kalman filter over
    // 10k+ timesteps) prohibitively slow. Included in a future targeted
    // benchmark that uses a simpler AutoETS spec for those.
];

const MODEL_NAMES: &[&str] = &[
    "AutoETS",
    "AutoTheta",
    "Laplace+auto",
    "Laplace+auto_aid",
    "SmartForecaster",
    "Laplace+skaters",
];
const N_MODELS: usize = 6;

/// Weighted Quantile Loss — fev's canonical probabilistic metric. For a
/// predicted quantile `q_hat` at level `q`, the loss is
/// `2 * max(q * (y - q_hat), (q - 1) * (y - q_hat))`. WQL is the sum
/// of these losses across quantiles and horizons divided by the sum of
/// `|y|`. Lower is better.
///
/// This function computes WQL for a single series given the point
/// prediction `mean` and standard deviation `std` per horizon. We use a
/// Gaussian assumption on the mean+std to derive quantiles for models
/// that only expose a point forecast, which is the fev fallback
/// (implemented as `PropheticQuantile` in their Python codebase).
const WQL_QUANTILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

fn wql_from_quantile_matrix(matrix: &[Vec<f64>], truth: &[f64]) -> f64 {
    // matrix[q][h] = predicted quantile q at horizon h.
    let mut num = 0.0f64;
    let mut denom = 0.0f64;
    for (qi, &q) in WQL_QUANTILES.iter().enumerate() {
        for h in 0..truth.len() {
            let y = truth[h];
            let qhat = matrix[qi][h];
            let d = y - qhat;
            let loss = 2.0 * ((q * d).max((q - 1.0) * d));
            num += loss;
            denom += y.abs();
        }
    }
    if denom < 1e-9 {
        f64::NAN
    } else {
        num / denom
    }
}

/// For point-only models, use a Gaussian approximation around the
/// point forecast — the fev PropheticQuantile fallback. `sigma_scale`
/// is a rough spread parameter (we use the seasonal-naive scale as a
/// conservative default).
fn gaussian_quantile(mean: f64, sigma: f64, q: f64) -> f64 {
    // Inverse standard-normal CDF via Beasley-Springer-Moro approximation.
    let z = inv_normal_cdf(q);
    mean + sigma * z
}

/// Beasley-Springer-Moro approximation of Φ⁻¹(q). Accurate to ~1e-9
/// over q ∈ [1e-8, 1 - 1e-8].
fn inv_normal_cdf(q: f64) -> f64 {
    // Coefficients for the rational approximation.
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if q < p_low {
        let z = (-2.0 * q.ln()).sqrt();
        (((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5])
            / ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1.0)
    } else if q <= p_high {
        let z = q - 0.5;
        let r = z * z;
        z * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let z = (-2.0 * (1.0 - q).ln()).sqrt();
        -(((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5])
            / ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1.0)
    }
}

fn parse_tsf(path: &str) -> Vec<Vec<f64>> {
    let bytes = fs::read(path).unwrap_or_default();
    let content: String = bytes.iter().map(|&b| b as char).collect();
    let mut series = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
        if !in_data {
            if line.trim_start().starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        // Robust to variable header shape: split on ':' and take the LAST
        // token as the CSV values.
        let toks: Vec<&str> = line.split(':').collect();
        if toks.len() < 2 {
            continue;
        }
        let vals_str = toks[toks.len() - 1];
        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| tok.trim().parse::<f64>().ok())
            .collect();
        if !values.is_empty() {
            series.push(values);
        }
    }
    series
}

fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    let s: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    s / pred.len() as f64
}

/// Seasonal-naive scale used by MASE. When `train.len() <= period` fall
/// back to naive-1 (adjacent differences).
fn mase_scale(train: &[f64], period: usize) -> f64 {
    let p = if train.len() > period { period } else { 1 };
    if train.len() <= p {
        return 1.0;
    }
    let n = train.len() - p;
    let sum: f64 = (p..train.len())
        .map(|i| (train[i] - train[i - p]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}

struct DatasetResult {
    name: &'static str,
    n_series: usize,
    /// One MASE per model, arithmetic mean across series.
    mase_mean: [f64; N_MODELS],
    /// One WQL per model, arithmetic mean across series.
    wql_mean: [f64; N_MODELS],
    /// One count per model, series that succeeded.
    n_ok: [usize; N_MODELS],
    /// Total fit time in seconds per model.
    total_s: [f64; N_MODELS],
}

fn run_dataset(ds: &Dataset, sample_per: usize, enabled: &[bool]) -> Option<DatasetResult> {
    let mut kept = parse_tsf(ds.path);
    if kept.is_empty() {
        eprintln!("  [{}] no data", ds.name);
        return None;
    }
    kept.retain(|v| v.len() > ds.horizon + 12);
    kept.truncate(sample_per);
    let n_series = kept.len();
    eprintln!(
        "  [{}] {} series (H={}, period={})",
        ds.name, n_series, ds.horizon, ds.period
    );

    let base_date = Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
    let mut mase_sum = [0.0f64; N_MODELS];
    let mut wql_sum = [0.0f64; N_MODELS];
    let mut n_ok = [0usize; N_MODELS];
    let mut fit_us_sum = [0u128; N_MODELS];

    for values in &kept {
        let split = values.len() - ds.horizon;
        let train_v = values[..split].to_vec();
        let test_v = &values[split..];
        let scale = mase_scale(&train_v, ds.period);
        let stamps: Vec<_> = (0..train_v.len())
            .map(|i| base_date + Duration::seconds(ds.step_seconds * i as i64))
            .collect();
        let train_ts = match TimeSeries::univariate(stamps, train_v.clone()) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Model 0: AutoETS — point-only; Gaussian PropheticQuantile fallback for WQL.
        // α-27 fix: pass the dataset's canonical seasonal period so ETS
        // enables its seasonal state-space variants. Nixtla's AutoETS
        // does this automatically from series metadata; ours needs it
        // explicitly (was previously calling AutoETS::new() → no period
        // → non-seasonal, huge handicap on monthly/hourly panels).
        if enabled[0] {
            let t0 = Instant::now();
            let mut m = if ds.period >= 2 {
                AutoETS::with_period(ds.period)
            } else {
                AutoETS::new()
            };
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[0] += mae(p, test_v) / scale;
                        // Fev PropheticQuantile fallback: Gaussian(mean, scale).
                        let matrix: Vec<Vec<f64>> = WQL_QUANTILES
                            .iter()
                            .map(|&q| {
                                p.iter()
                                    .map(|&mu| gaussian_quantile(mu, scale, q))
                                    .collect()
                            })
                            .collect();
                        let w = wql_from_quantile_matrix(&matrix, test_v);
                        if w.is_finite() {
                            wql_sum[0] += w;
                        }
                        n_ok[0] += 1;
                    }
                }
            }
            fit_us_sum[0] += t0.elapsed().as_micros();
        }
        // Model 1: AutoTheta — point-only. Same period-passing fix.
        if enabled[1] {
            let t0 = Instant::now();
            let mut m = if ds.period >= 2 {
                AutoTheta::seasonal(ds.period)
            } else {
                AutoTheta::new()
            };
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[1] += mae(p, test_v) / scale;
                        let matrix: Vec<Vec<f64>> = WQL_QUANTILES
                            .iter()
                            .map(|&q| {
                                p.iter()
                                    .map(|&mu| gaussian_quantile(mu, scale, q))
                                    .collect()
                            })
                            .collect();
                        let w = wql_from_quantile_matrix(&matrix, test_v);
                        if w.is_finite() {
                            wql_sum[1] += w;
                        }
                        n_ok[1] += 1;
                    }
                }
            }
            fit_us_sum[1] += t0.elapsed().as_micros();
        }
        // Model 2: Laplace + auto — mixture quantiles for WQL.
        // TEMP: testing .with_stacking() as an accuracy improvement.
        #[cfg(feature = "distributional")]
        if enabled[2] {
            use anofox_forecast::models::DistributionalForecaster;
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new()
                .auto()
                .with_stacking()
                .auto_with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(mixtures) = m.forecast_dist(ds.horizon) {
                    if mixtures.len() == test_v.len() {
                        let p: Vec<f64> = mixtures.iter().map(|g| g.mean()).collect();
                        mase_sum[2] += mae(&p, test_v) / scale;
                        let matrix: Vec<Vec<f64>> = WQL_QUANTILES
                            .iter()
                            .map(|&q| mixtures.iter().map(|g| g.quantile(q)).collect())
                            .collect();
                        let w = wql_from_quantile_matrix(&matrix, test_v);
                        if w.is_finite() {
                            wql_sum[2] += w;
                        }
                        n_ok[2] += 1;
                    }
                }
            }
            fit_us_sum[2] += t0.elapsed().as_micros();
        }
        // Model 3: Laplace + auto_aid — mixture quantiles.
        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        if enabled[3] {
            use anofox_forecast::models::DistributionalForecaster;
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new()
                .auto_aid()
                .auto_with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(mixtures) = m.forecast_dist(ds.horizon) {
                    if mixtures.len() == test_v.len() {
                        let p: Vec<f64> = mixtures.iter().map(|g| g.mean()).collect();
                        mase_sum[3] += mae(&p, test_v) / scale;
                        let matrix: Vec<Vec<f64>> = WQL_QUANTILES
                            .iter()
                            .map(|&q| mixtures.iter().map(|g| g.quantile(q)).collect())
                            .collect();
                        let w = wql_from_quantile_matrix(&matrix, test_v);
                        if w.is_finite() {
                            wql_sum[3] += w;
                        }
                        n_ok[3] += 1;
                    }
                }
            }
            fit_us_sum[3] += t0.elapsed().as_micros();
        }
        // Model 4: SmartForecaster — point-only via Forecaster trait; Gaussian fallback for WQL.
        #[cfg(all(feature = "distributional", feature = "postprocess"))]
        if enabled[4] {
            let t0 = Instant::now();
            let mut m = SmartForecaster::new().with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(fc) = m.predict(ds.horizon) {
                    let p = fc.primary();
                    if p.len() == test_v.len() {
                        mase_sum[4] += mae(p, test_v) / scale;
                        let matrix: Vec<Vec<f64>> = WQL_QUANTILES
                            .iter()
                            .map(|&q| {
                                p.iter()
                                    .map(|&mu| gaussian_quantile(mu, scale, q))
                                    .collect()
                            })
                            .collect();
                        let w = wql_from_quantile_matrix(&matrix, test_v);
                        if w.is_finite() {
                            wql_sum[4] += w;
                        }
                        n_ok[4] += 1;
                    }
                }
            }
            fit_us_sum[4] += t0.elapsed().as_micros();
        }
        // Model 5: Laplace + skaters() — full extended engine.
        // Fixed skaters-shaped candidate pool, sticky lattice, terminal
        // scale-mixture, XGBoost-shrunk softmax updates. Post-#180.
        #[cfg(feature = "distributional")]
        if enabled[5] {
            use anofox_forecast::models::DistributionalForecaster;
            let t0 = Instant::now();
            let mut m = LaplaceForecaster::new()
                .skaters()
                .auto_with_seasonal_period(ds.period.max(2));
            if m.fit(&train_ts).is_ok() {
                if let Ok(mixtures) = m.forecast_dist(ds.horizon) {
                    if mixtures.len() == test_v.len() {
                        let p: Vec<f64> = mixtures.iter().map(|g| g.mean()).collect();
                        mase_sum[5] += mae(&p, test_v) / scale;
                        let matrix: Vec<Vec<f64>> = WQL_QUANTILES
                            .iter()
                            .map(|&q| mixtures.iter().map(|g| g.quantile(q)).collect())
                            .collect();
                        let w = wql_from_quantile_matrix(&matrix, test_v);
                        if w.is_finite() {
                            wql_sum[5] += w;
                        }
                        n_ok[5] += 1;
                    }
                }
            }
            fit_us_sum[5] += t0.elapsed().as_micros();
        }
    }

    let mut mase_mean = [0.0f64; N_MODELS];
    let mut wql_mean = [0.0f64; N_MODELS];
    let mut total_s = [0.0f64; N_MODELS];
    for i in 0..N_MODELS {
        if n_ok[i] > 0 {
            mase_mean[i] = mase_sum[i] / n_ok[i] as f64;
            wql_mean[i] = wql_sum[i] / n_ok[i] as f64;
        } else {
            mase_mean[i] = f64::NAN;
            wql_mean[i] = f64::NAN;
        }
        total_s[i] = fit_us_sum[i] as f64 / 1_000_000.0;
    }

    Some(DatasetResult {
        name: ds.name,
        n_series,
        mase_mean,
        wql_mean,
        n_ok,
        total_s,
    })
}

fn geometric_mean(xs: &[f64]) -> f64 {
    let xs: Vec<f64> = xs
        .iter()
        .filter(|x| x.is_finite() && **x > 0.0)
        .copied()
        .collect();
    if xs.is_empty() {
        return f64::NAN;
    }
    let log_sum: f64 = xs.iter().map(|x| x.ln()).sum();
    (log_sum / xs.len() as f64).exp()
}

fn main() {
    let sample_per: usize = std::env::var("SAMPLE_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    // Filter which models to run (accuracy-audit workflow). Comma-separated
    // list of names or "all". Names match MODEL_NAMES (case-insensitive,
    // spaces ignored). E.g. `MODELS=laplace+skaters,laplace+auto` skips
    // AutoETS (which eats ~90% of runtime). Default: all.
    let models_filter: Vec<bool> = {
        let raw = std::env::var("MODELS").unwrap_or_else(|_| "all".into());
        if raw.eq_ignore_ascii_case("all") {
            vec![true; N_MODELS]
        } else {
            let requested: Vec<String> = raw
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase().replace(' ', ""))
                .collect();
            (0..N_MODELS)
                .map(|i| {
                    let name = MODEL_NAMES[i].to_ascii_lowercase().replace(' ', "");
                    requested.iter().any(|r| *r == name)
                })
                .collect()
        }
    };
    let enabled_count = models_filter.iter().filter(|b| **b).count();
    eprintln!(
        "fev-style benchmark — sample={} series/dataset, {}/{} models enabled",
        sample_per, enabled_count, N_MODELS
    );
    for (i, on) in models_filter.iter().enumerate() {
        if *on {
            eprintln!("  ✓ {}", MODEL_NAMES[i]);
        }
    }

    let mut results: Vec<DatasetResult> = Vec::new();
    for ds in DATASETS.iter() {
        if let Some(r) = run_dataset(ds, sample_per, &models_filter) {
            results.push(r);
        }
    }

    println!("\n=== fev-style MASE per dataset ===");
    print!("{:<20}{:>8}", "dataset", "n");
    for name in MODEL_NAMES {
        print!("{:>18}", name);
    }
    println!();
    for r in &results {
        print!("{:<20}{:>8}", r.name, r.n_series);
        for i in 0..N_MODELS {
            print!("{:>18.3}", r.mase_mean[i]);
        }
        println!();
    }

    println!("\n=== fev-style WQL per dataset ===");
    print!("{:<20}{:>8}", "dataset", "n");
    for name in MODEL_NAMES {
        print!("{:>18}", name);
    }
    println!();
    for r in &results {
        print!("{:<20}{:>8}", r.name, r.n_series);
        for i in 0..N_MODELS {
            print!("{:>18.3}", r.wql_mean[i]);
        }
        println!();
    }

    println!("\n=== geometric mean across datasets ===");
    print!("{:<20}", "");
    for name in MODEL_NAMES {
        print!("{:>18}", name);
    }
    println!();
    print!("{:<20}", "geomean MASE");
    for i in 0..N_MODELS {
        let vals: Vec<f64> = results.iter().map(|r| r.mase_mean[i]).collect();
        print!("{:>18.4}", geometric_mean(&vals));
    }
    println!();
    print!("{:<20}", "geomean WQL");
    for i in 0..N_MODELS {
        let vals: Vec<f64> = results.iter().map(|r| r.wql_mean[i]).collect();
        print!("{:>18.4}", geometric_mean(&vals));
    }
    println!();

    println!("\n=== total fit time per model (s) ===");
    for i in 0..N_MODELS {
        let total: f64 = results.iter().map(|r| r.total_s[i]).sum();
        println!("  {:<20}{:>10.1}s", MODEL_NAMES[i], total);
    }
}
