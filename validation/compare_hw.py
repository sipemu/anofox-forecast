#!/usr/bin/env python3
"""Compare HoltWinters forecasts between Rust and statsforecast."""

import csv

# Read statsforecast results
statsforecast_forecasts = []
with open('/home/simonm/projects/rust/forecast/validation/results/statsforecast/point_forecasts.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['series_type'] == 'multiplicative_seasonal' and row['model'] == 'HoltWinters':
            statsforecast_forecasts.append({
                'step': int(row['step']),
                'forecast': float(row['forecast'])
            })

# Read Rust results
rust_forecasts = []
with open('/home/simonm/projects/rust/forecast/validation/results/rust/point_forecasts.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['series_type'] == 'multiplicative_seasonal' and row['model'] == 'HoltWinters':
            rust_forecasts.append({
                'step': int(row['step']),
                'forecast': float(row['forecast'])
            })

# Sort by step
statsforecast_forecasts.sort(key=lambda x: x['step'])
rust_forecasts.sort(key=lambda x: x['step'])

print("HoltWinters Forecast Comparison: multiplicative_seasonal")
print("=" * 80)
print()
print(f"{'Step':>5} | {'statsforecast':>15} | {'Rust':>15} | {'Diff':>10} | {'% Diff':>8}")
print("-" * 80)

total_abs_diff = 0.0
for sf, rust in zip(statsforecast_forecasts, rust_forecasts):
    assert sf['step'] == rust['step']
    diff = rust['forecast'] - sf['forecast']
    pct_diff = (diff / sf['forecast']) * 100 if sf['forecast'] != 0 else 0
    total_abs_diff += abs(diff)

    print(f"{sf['step']:5d} | {sf['forecast']:15.6f} | {rust['forecast']:15.6f} | {diff:10.4f} | {pct_diff:7.2f}%")

print("-" * 80)
print(f"Mean Absolute Difference (MAD): {total_abs_diff / len(statsforecast_forecasts):.4f}")
print()

# Load data to compute some statistics
data = []
with open('/home/simonm/projects/rust/forecast/validation/data/multiplicative_seasonal.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(float(row['value']))

print(f"Data summary:")
print(f"  Length: {len(data)}")
print(f"  Mean: {sum(data)/len(data):.2f}")
print(f"  Min: {min(data):.2f}")
print(f"  Max: {max(data):.2f}")
print(f"  Last value: {data[-1]:.2f}")
print()

print("Last 12 values (for seasonal pattern inspection):")
for i, val in enumerate(data[-12:], 1):
    print(f"  {i:2d}: {val:.6f}")
