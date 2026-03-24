//! Calendar Feature Engineering
//!
//! Demonstrates cyclical sin/cos encoding, binary indicators, and advanced
//! calendar features — plus generating future timestamps for prediction.
//!
//! Run with: cargo run --example calendar_features

use anofox_forecast::core::time_series::{generate_future_timestamps, Frequency};
use anofox_forecast::core::TimeSeries;
use anofox_forecast::features::{
    AdvancedFeature, BinaryIndicator, FeatureGenerator, TimeComponent,
};
use chrono::{Datelike, Duration, TimeZone, Utc};

fn main() {
    println!("=== Calendar Feature Engineering ===\n");

    // ── 1. Cyclical sin/cos encoding ─────────────────────────────────
    println!("--- 1. Cyclical Encoding ---\n");

    let gen = FeatureGenerator::new()
        .cyclical(TimeComponent::Month)
        .cyclical(TimeComponent::DayOfWeek)
        .cyclical(TimeComponent::Hour);

    let timestamps: Vec<_> = (0..24)
        .map(|h| Utc.with_ymd_and_hms(2024, 6, 15, h, 0, 0).unwrap())
        .collect();
    let features = gen.generate(&timestamps);

    println!("Generated columns: {:?}", {
        let mut names: Vec<_> = features.keys().collect();
        names.sort();
        names
    });
    println!(
        "Hour 0:  hour_sin={:.3}, hour_cos={:.3}",
        features["hour_sin"][0], features["hour_cos"][0]
    );
    println!(
        "Hour 6:  hour_sin={:.3}, hour_cos={:.3}",
        features["hour_sin"][6], features["hour_cos"][6]
    );
    println!(
        "Hour 12: hour_sin={:.3}, hour_cos={:.3}",
        features["hour_sin"][12], features["hour_cos"][12]
    );

    // ── 2. Binary indicators ─────────────────────────────────────────
    println!("\n--- 2. Binary Indicators ---\n");

    let gen = FeatureGenerator::new()
        .binary(BinaryIndicator::Weekend)
        .binary(BinaryIndicator::MonthStart)
        .binary(BinaryIndicator::MonthEnd)
        .binary(BinaryIndicator::YearEnd);

    let dates: Vec<_> = (0..365)
        .map(|d| Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + Duration::days(d))
        .collect();
    let features = gen.generate(&dates);

    let weekend_count: usize = features["weekend"].iter().map(|&v| v as usize).sum();
    let month_starts: usize = features["month_start"].iter().map(|&v| v as usize).sum();
    let month_ends: usize = features["month_end"].iter().map(|&v| v as usize).sum();
    let year_ends: usize = features["year_end"].iter().map(|&v| v as usize).sum();

    println!("Over 365 days in 2024:");
    println!("  Weekends:     {}", weekend_count);
    println!("  Month starts: {}", month_starts);
    println!("  Month ends:   {}", month_ends);
    println!("  Year ends:    {}", year_ends);

    // ── 3. Advanced features ─────────────────────────────────────────
    println!("\n--- 3. Advanced Features ---\n");

    let gen = FeatureGenerator::new()
        .advanced(AdvancedFeature::DaysInMonth)
        .advanced(AdvancedFeature::LeapYear);

    let monthly: Vec<_> = (0..12)
        .map(|m| Utc.with_ymd_and_hms(2024, m + 1, 15, 0, 0, 0).unwrap())
        .collect();
    let features = gen.generate(&monthly);

    println!("{:<10} {:>12} {:>10}", "Month", "DaysInMonth", "LeapYear");
    for i in 0..12 {
        println!(
            "{:<10} {:>12.0} {:>10.0}",
            monthly[i].format("%B"),
            features["days_in_month"][i],
            features["leap_year"][i]
        );
    }

    // ── 4. Generate future timestamps ────────────────────────────────
    println!("\n--- 4. Future Timestamps ---\n");

    let last = Utc.with_ymd_and_hms(2024, 1, 31, 0, 0, 0).unwrap();

    // Monthly (calendar-aware: handles month-end correctly)
    let monthly_future = generate_future_timestamps(&last, &Frequency::Months(1), 6);
    println!("Monthly from Jan 31:");
    for ts in &monthly_future {
        println!("  {} (day {})", ts.format("%Y-%m-%d"), ts.day());
    }

    // Weekly
    let weekly_future =
        generate_future_timestamps(&last, &Frequency::Duration(Duration::weeks(1)), 4);
    println!("\nWeekly from Jan 31:");
    for ts in &weekly_future {
        println!("  {}", ts.format("%Y-%m-%d"));
    }

    // Auto-infer from TimeSeries
    let ts = TimeSeries::univariate(
        (0..24)
            .map(|m| Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap() + Duration::days(m * 30))
            .collect(),
        (0..24).map(|i| i as f64).collect(),
    )
    .unwrap();
    let auto_future = ts.future_timestamps(6).unwrap();
    println!("\nAuto-inferred from TimeSeries (6 steps):");
    for ts in &auto_future {
        println!("  {}", ts.format("%Y-%m-%d"));
    }

    // ── 5. Combined: features for forecast horizon ───────────────────
    println!("\n--- 5. Features for Forecast Horizon ---\n");

    let gen = FeatureGenerator::new()
        .cyclical(TimeComponent::Month)
        .binary(BinaryIndicator::MonthEnd)
        .advanced(AdvancedFeature::DaysInMonth);

    let future_dates = generate_future_timestamps(&last, &Frequency::Months(1), 4);
    let future_features = gen.generate(&future_dates);

    println!(
        "{:<12} {:>10} {:>10} {:>12}",
        "Date", "month_sin", "month_end", "days_in_mo"
    );
    for i in 0..4 {
        println!(
            "{:<12} {:>10.3} {:>10.0} {:>12.0}",
            future_dates[i].format("%Y-%m-%d"),
            future_features["month_sin"][i],
            future_features["month_end"][i],
            future_features["days_in_month"][i]
        );
    }

    println!("\nDone!");
}
