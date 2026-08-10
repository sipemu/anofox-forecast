//! Monash TSF dataset loader with per-series metadata.
//!
//! Provides typed access to Monash `.tsf` competition datasets (M3, M4,
//! Tourism, NN5), gated on the `ANOFOX_DATASET_DIR` environment variable.
//! Tests skip cleanly when the var is not set (ACCUR-01).
//!
//! Main types:
//! - [`DatasetSeries`] — one time series with frequency, horizon, and values
//! - [`parse_tsf_with_meta`] — parse a single `.tsf` file into typed series
//! - [`load_m3`] — load M3 yearly/quarterly/monthly from a corpus directory
//! - [`dataset_dir_from_env`] — read `ANOFOX_DATASET_DIR` (returns `None` when unset)
//! - [`mase_scale`] — competition-correct training-slice seasonal-naive denominator

use std::fs;
use std::path::{Path, PathBuf};

/// One time series from a Monash `.tsf` file, with file-level metadata.
#[derive(Debug, Clone)]
pub struct DatasetSeries {
    /// Series identifier (first colon-delimited field of each data line).
    pub id: String,
    /// Frequency tag from the `@frequency` header (e.g. `"monthly"`).
    pub frequency: String,
    /// Horizon from the `@horizon` header.
    pub horizon: usize,
    /// Observed values (non-finite tokens are silently dropped).
    pub values: Vec<f64>,
}

/// Parse a Monash `.tsf` file and return `(frequency, horizon, series)`.
///
/// The file is read as raw bytes and decoded as Latin-1 (ISO-8859-1) via
/// `b as char`, which is MANDATORY — Monash `.tsf` files contain accented
/// category labels that are not valid UTF-8. Using `std::fs::read_to_string`
/// would panic or error on these files.
///
/// Value tokens that fail to parse as `f64`, or that are non-finite (NaN/Inf
/// from malformed TSF), are silently dropped (security guard V5, T-02-NAN).
pub fn parse_tsf_with_meta(
    path: &Path,
) -> std::io::Result<(String, usize, Vec<DatasetSeries>)> {
    // Latin-1 decode: MANDATORY — Monash .tsf is not UTF-8.
    // std::fs::read_to_string panics on accented category labels in the header.
    let bytes = fs::read(path)?;
    let content: String = bytes.iter().map(|&b| b as char).collect();

    let mut frequency = String::new();
    let mut horizon: usize = 0;
    let mut series: Vec<DatasetSeries> = Vec::new();
    let mut in_data = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if !in_data {
            if let Some(val) = trimmed.strip_prefix("@frequency") {
                frequency = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("@horizon") {
                horizon = val.trim().parse().unwrap_or(0);
            } else if trimmed.starts_with("@data") {
                in_data = true;
            }
            continue;
        }

        // Data line format: id:start_ts:v1,v2,v3,...
        let mut parts = trimmed.splitn(3, ':');
        let id = match parts.next() {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };
        let _start_ts = parts.next();
        let vals_str = match parts.next() {
            Some(s) => s,
            None => continue,
        };

        let values: Vec<f64> = vals_str
            .split(',')
            .filter_map(|tok| {
                let v: f64 = tok.trim().parse().ok()?;
                // Drop NaN/Inf from malformed TSF (security guard T-02-NAN)
                v.is_finite().then_some(v)
            })
            .collect();

        if values.is_empty() {
            continue;
        }

        series.push(DatasetSeries {
            id,
            frequency: frequency.clone(),
            horizon,
            values,
        });
    }

    Ok((frequency, horizon, series))
}

/// Load M3 yearly, quarterly, and monthly series from a corpus directory.
///
/// Reads `m3_yearly.tsf`, `m3_quarterly.tsf`, and `m3_monthly.tsf` from
/// `dir` and concatenates their [`DatasetSeries`].
///
/// Security (T-02-PATH): `dir` is canonicalized and only the three fixed
/// hard-coded filenames are joined onto it — no caller-supplied filename
/// component can introduce a `..` path-traversal escape.
pub fn load_m3(dir: &Path) -> std::io::Result<Vec<DatasetSeries>> {
    // Canonicalize to resolve symlinks and strip any .. components.
    let canonical = std::fs::canonicalize(dir)?;

    let filenames = ["m3_yearly.tsf", "m3_quarterly.tsf", "m3_monthly.tsf"];
    let mut all = Vec::new();

    for name in &filenames {
        // Only fixed hard-coded filenames joined — never a caller-supplied path.
        let path = canonical.join(name);
        let (_freq, _horizon, mut series) = parse_tsf_with_meta(&path)?;
        all.append(&mut series);
    }

    Ok(all)
}

/// Read `ANOFOX_DATASET_DIR` from the environment.
///
/// Returns `None` when the variable is not set, allowing callers to skip
/// dataset-dependent operations cleanly (ACCUR-01).
pub fn dataset_dir_from_env() -> Option<PathBuf> {
    std::env::var("ANOFOX_DATASET_DIR").ok().map(PathBuf::from)
}

/// Competition-correct training-slice seasonal-naive MASE denominator.
///
/// Computes the mean absolute difference between same-phase observations in
/// `train` at the given `period`. This is the in-sample naive MAE used as the
/// MASE denominator in M3/M4 competition scoring, per the convention in
/// `examples/skaters_m3_monthly_benchmark.rs`.
///
/// The ACCUR-08 anchor (AutoETS M3-monthly MASE ≈ 0.93) depends on this exact
/// formula. Do NOT use `ForecastMetrics::compute`/`calculate_mase` for
/// competition MASE — those scale on the test slice, not the training slice.
///
/// Returns `1.0` when `train.len() <= period` (too short for seasonal diff).
/// The `.max(1e-9)` floor prevents division by zero on near-constant series.
pub fn mase_scale(train: &[f64], period: usize) -> f64 {
    if train.len() <= period {
        return 1.0;
    }
    let n = train.len() - period;
    let sum: f64 = (period..train.len())
        .map(|i| (train[i] - train[i - period]).abs())
        .sum();
    (sum / n as f64).max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // Build a minimal in-memory .tsf fixture and write it to a temp file.
    fn write_temp_tsf(content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "anofox_test_{}.tsf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let mut f = std::fs::File::create(&path).expect("create temp tsf");
        f.write_all(content.as_bytes()).expect("write temp tsf");
        path
    }

    #[test]
    fn parse_tsf_extracts_metadata_and_drops_nan() {
        // Minimal .tsf: two series, one token is "NaN" and must be dropped.
        let content = "\
@attribute series_name string
@attribute start_timestamp date
@attribute series_value numeric
@frequency monthly
@horizon 18
@missing false
@equallength false
@data
S1:1990-01:1.0,2.0,NaN,4.0,5.0
S2:1990-01:10.0,20.0,30.0
";
        let path = write_temp_tsf(content);
        let (freq, horizon, series) = parse_tsf_with_meta(&path)
            .expect("parse should succeed");
        std::fs::remove_file(&path).ok();

        assert_eq!(freq, "monthly", "frequency tag");
        assert_eq!(horizon, 18, "horizon tag");
        assert_eq!(series.len(), 2, "two series");

        // NaN token must be dropped
        assert_eq!(series[0].id, "S1");
        assert_eq!(
            series[0].values,
            vec![1.0, 2.0, 4.0, 5.0],
            "NaN token dropped from S1"
        );
        assert_eq!(series[1].id, "S2");
        assert_eq!(series[1].values, vec![10.0, 20.0, 30.0]);

        // Each series carries the file-level metadata
        for s in &series {
            assert_eq!(s.frequency, "monthly");
            assert_eq!(s.horizon, 18);
        }
    }

    #[test]
    fn mase_scale_ramp_period1() {
        // Ramp [0,1,2,3,4,5] with period=1: each diff is 1.0, mean = 1.0.
        let ramp: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let scale = mase_scale(&ramp, 1);
        assert!(
            (scale - 1.0).abs() < 1e-10,
            "expected 1.0, got {scale}"
        );
    }

    #[test]
    fn mase_scale_too_short_returns_one() {
        // When train.len() <= period, return 1.0 (collapse guard).
        let short = vec![1.0, 2.0, 3.0];
        assert_eq!(mase_scale(&short, 5), 1.0);
        assert_eq!(mase_scale(&short, 3), 1.0);
    }

    #[test]
    fn mase_scale_seasonal_period12() {
        // Repeating monthly pattern: each seasonal diff is 0.1, mean = 0.1.
        let mut vals: Vec<f64> = Vec::new();
        for season in 0..2 {
            for m in 0..12_usize {
                vals.push(m as f64 + season as f64 * 0.1);
            }
        }
        // season 0: [0,1,...,11]; season 1: [0.1,1.1,...,11.1]
        // diffs at lag 12: all 0.1 => mean = 0.1
        let scale = mase_scale(&vals, 12);
        assert!(
            (scale - 0.1).abs() < 1e-9,
            "expected 0.1, got {scale}"
        );
    }

    #[test]
    fn dataset_dir_from_env_absent() {
        // When ANOFOX_DATASET_DIR is not set, returns None.
        // (We cannot unset it in a safe way if set by an outer test runner, so
        // we test only the parse path via direct std::env::var.)
        let original = std::env::var("ANOFOX_DATASET_DIR").ok();
        // Only test absence when not set by outer environment
        if original.is_none() {
            assert!(dataset_dir_from_env().is_none());
        }
    }
}
