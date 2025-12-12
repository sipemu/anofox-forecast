#!/usr/bin/env python3
"""Trace HoltWinters execution to debug differences."""

import csv
import sys

try:
    from statsforecast.models import HoltWinters
except ImportError as e:
    print(f"Error: {e}")
    print("Please install: pip install statsforecast")
    sys.exit(1)

# Load data
data = []
with open('/home/simonm/projects/rust/forecast/validation/data/multiplicative_seasonal.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(float(row['value']))

print("Data length:", len(data))
print()

# Create model
model = HoltWinters(season_length=12, error_type='A')  # Additive, same as statsforecast validation
model.fit(data)

# Extract model state
print("=" * 80)
print("STATSFORECAST HoltWinters Model State (ADDITIVE)")
print("=" * 80)
print()

print("Optimized Parameters:")
print(f"  alpha: {model.model_['alpha']:.10f}")
print(f"  beta:  {model.model_['beta']:.10f}")
print(f"  gamma: {model.model_['gamma']:.10f}")
print()

print("Final State (after fitting all data):")
print(f"  level (l): {model.model_['l']:.10f}")
print(f"  trend (b): {model.model_['b']:.10f}")
print()

print("Final Seasonal Components (s):")
seasonal = model.model_['s']
for i, s in enumerate(seasonal):
    print(f"  s[{i:2d}]: {s:12.6f}")
print()

# Generate forecasts
forecasts = model.predict(h=12)['mean']
print("Forecasts (h=12):")
for i, pred in enumerate(forecasts, 1):
    print(f"  Step {i:2d}: {pred:.10f}")
print()

# Manually compute first forecast to verify formula
n = len(data)
print("Manual verification of first forecast:")
print(f"  n = {n}")
print(f"  h = 1")
print(f"  season_idx = (n + h - 1) % 12 = ({n} + 1 - 1) % 12 = {(n + 1 - 1) % 12}")
print(f"  s = seasonal[{(n + 1 - 1) % 12}] = {seasonal[(n + 1 - 1) % 12]:.10f}")
print(f"  l = {model.model_['l']:.10f}")
print(f"  b = {model.model_['b']:.10f}")
print(f"  forecast = l + h*b + s")
print(f"           = {model.model_['l']:.10f} + 1*{model.model_['b']:.10f} + {seasonal[(n + 1 - 1) % 12]:.10f}")
manual_forecast = model.model_['l'] + model.model_['b'] + seasonal[(n + 1 - 1) % 12]
print(f"           = {manual_forecast:.10f}")
print(f"  model.predict()[0] = {forecasts[0]:.10f}")
print(f"  Match: {abs(manual_forecast - forecasts[0]) < 1e-6}")
