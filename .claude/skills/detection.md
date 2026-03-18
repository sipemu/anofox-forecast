---
name: detection
description: How to detect outliers and seasonal periods in time series data
user_invocable: true
---

# Detection in anofox-forecast

## 1. Outlier Detection

```rust
use anofox_forecast::detection::{
    detect_outliers, detect_outliers_auto,
    OutlierConfig, OutlierMethod, OutlierResult,
};

let series: Vec<f64> = vec![1.0, 2.0, 1.5, 100.0, 1.8, 2.1];

// Auto detection (IQR with multiplier 1.5)
let result: OutlierResult = detect_outliers_auto(&series);

// Custom configuration
let config = OutlierConfig::iqr(1.5);            // IQR method
let config = OutlierConfig::z_score(3.0);         // Z-score method
let config = OutlierConfig::modified_z_score(3.5); // MAD-based (robust)

let result = detect_outliers(&series, &config);

// Inspect results
println!("Outliers: {:?}", result.outlier_indices);  // e.g., [3]
println!("Count: {}", result.outlier_count());
println!("Percentage: {:.1}%", result.outlier_percentage());
println!("Is index 3 outlier? {}", result.is_outlier(3));

// Anomaly scores (higher = more anomalous)
for (i, score) in result.scores.iter().enumerate() {
    println!("  i={}: score={:.2}", i, score);
}
```

### Outlier Methods

| Method | Best For | Threshold Meaning |
|---|---|---|
| `IQR` | General use, skewed data | Multiplier of IQR beyond Q1/Q3 |
| `ZScore` | Normally distributed data | Number of standard deviations |
| `ModifiedZScore` | Data with existing outliers | MAD-based standard deviations |

## 2. Period Detection

```rust
use anofox_forecast::detection::{
    detect_periods, detect_dominant_period,
    PeriodDetectionConfig, Period,
};

let signal: Vec<f64> = (0..365)
    .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin())
    .collect();

// Quick: get the strongest period
let dominant: Option<usize> = detect_dominant_period(&signal);
// → Some(7)

// Full: get multiple periods with validation metadata
let config = PeriodDetectionConfig {
    min_period: 2,
    max_period: Some(180),
    max_periods: 5,
    min_power_ratio: 3.0,
    min_strength: 0.0,
    min_cycles: 2,
    window_size: None,  // auto
};

let periods: Vec<Period> = detect_periods(&signal, &config);

for p in &periods {
    println!("Period {}: strength={:.2}, acf={:.2}, cycles={}",
        p.period, p.strength, p.acf, p.n_cycles);
}
```

### Period Struct Fields

| Field | Description |
|---|---|
| `period` | Detected period (integer) |
| `power` | Spectral power at this frequency |
| `strength` | Seasonal differencing strength (0–1; >0.6 = strong) |
| `acf` | Autocorrelation at this lag |
| `n_cycles` | Complete cycles observed in data |

### Configuration Tips

- **`min_strength`**: Set to `0.6` to only return strong seasonalities.
- **`min_cycles`**: Default `2`. Increase for more confidence.
- **`max_period`**: Defaults to `signal.len() / 3`. Set explicitly for long series.
- **`min_power_ratio`**: How much stronger a peak must be vs mean spectral power. Default `3.0`.

## 3. Period Detection → Model Fitting

```rust
use anofox_forecast::detection::{detect_periods, PeriodDetectionConfig};
use anofox_forecast::models::mstl_forecaster::MSTLForecaster;
use anofox_forecast::models::Forecaster;

// Detect periods
let config = PeriodDetectionConfig::default();
let periods = detect_periods(ts.primary_values(), &config);

// Use detected periods with MSTL
let seasonal_periods: Vec<usize> = periods.iter()
    .filter(|p| p.strength > 0.6)
    .map(|p| p.period)
    .collect();

if !seasonal_periods.is_empty() {
    let mut model = MSTLForecaster::new(seasonal_periods);
    model.fit(&ts).unwrap();
    let forecast = model.predict(14).unwrap();
}
```

## 4. Welch Periodogram (Low-Level)

```rust
use anofox_forecast::detection::welch_periodogram;

// Returns Vec<(period, power)> sorted by period (largest first)
let spectrum = welch_periodogram(&signal, 64, 0.5);
for (period, power) in &spectrum {
    println!("Period {}: power={:.4}", period, power);
}
```

## Key Rules

- **Outlier detection** operates on raw `&[f64]` slices, not `TimeSeries`.
- **Period detection** also operates on `&[f64]`. Use `ts.primary_values()` to extract.
- Data needs at least `2 * period` observations for reliable period detection.
- `detect_dominant_period` returns `None` if no significant periodicity found.
- Modified Z-score is most robust when outliers are already present in the data.
