#!/usr/bin/env python3
"""Debug HoltWinters initialization differences."""

import csv

# Load multiplicative_seasonal data
data = []
with open('/home/simonm/projects/rust/forecast/validation/data/multiplicative_seasonal.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(float(row['value']))

print("Multiplicative Seasonal Data Analysis")
print("=" * 60)
print()

# Analyze first season for initialization
period = 12
first_season = data[:period]
print(f"First season ({period} values):")
for i, val in enumerate(first_season, 1):
    print(f"  {i:2d}: {val:.6f}")
print()

# Initial level (mean of first season)
initial_level = sum(first_season) / period
print(f"Initial level (mean of first season): {initial_level:.6f}")
print()

# Initial trend (if 2+ seasons available)
if len(data) >= 2 * period:
    trend_sum = sum((data[period + i] - data[i]) / period for i in range(period))
    initial_trend = trend_sum / period
    print(f"Initial trend: {initial_trend:.6f}")
else:
    initial_trend = 0.0
    print("Initial trend: 0.0 (not enough data)")
print()

# Initial seasonal indices (additive)
print("Initial seasonal indices (ADDITIVE):")
seasonal_additive = [y - initial_level for y in first_season]
for i, s in enumerate(seasonal_additive, 1):
    print(f"  {i:2d}: {s:10.6f}")
print()

# Initial seasonal indices (multiplicative)
print("Initial seasonal indices (MULTIPLICATIVE):")
seasonal_mult = [y / initial_level if abs(initial_level) > 1e-10 else 1.0 for y in first_season]
for i, s in enumerate(seasonal_mult, 1):
    print(f"  {i:2d}: {s:10.6f}")
print()

# Check if sum of additive is close to 0
additive_sum = sum(seasonal_additive)
print(f"Sum of additive seasonal indices: {additive_sum:.6f} (should be ~0)")

# Check if mean of multiplicative is close to 1
mult_mean = sum(seasonal_mult) / len(seasonal_mult)
print(f"Mean of multiplicative seasonal indices: {mult_mean:.6f} (should be ~1)")
