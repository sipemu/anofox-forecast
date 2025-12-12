#!/usr/bin/env python3
"""Quick test of statsforecast HoltWinters on multiplicative_seasonal data."""

import sys
import numpy as np

try:
    import pandas as pd
    from statsforecast.models import HoltWinters
except ImportError as e:
    print(f"Error: {e}")
    print("Please install: pip install statsforecast pandas")
    sys.exit(1)

# Load data
df = pd.read_csv("/home/simonm/projects/rust/forecast/validation/data/multiplicative_seasonal.csv")
y = df['value'].values

print("Data shape:", y.shape)
print("First 12 values:", y[:12])
print()

# Test with multiplicative error type (like Rust implementation)
print("=== HoltWinters with error_type='M' (multiplicative) ===")
model = HoltWinters(season_length=12, error_type='M')
model.fit(y)

print("Parameters:")
print(f"  alpha: {model.model_['alpha']}")
print(f"  beta: {model.model_['beta']}")
print(f"  gamma: {model.model_['gamma']}")
print()

print("Initial state (after fitting):")
print(f"  level (l): {model.model_['l']}")
print(f"  trend (b): {model.model_['b']}")
print(f"  seasonal (s): {model.model_['s']}")
print()

predictions = model.predict(h=12)['mean']
print("Predictions (h=12):")
for i, pred in enumerate(predictions, 1):
    print(f"  Step {i:2d}: {pred:.6f}")
print()

# Test with additive error type
print("=== HoltWinters with error_type='A' (additive) ===")
model_a = HoltWinters(season_length=12, error_type='A')
model_a.fit(y)

print("Parameters:")
print(f"  alpha: {model_a.model_['alpha']}")
print(f"  beta: {model_a.model_['beta']}")
print(f"  gamma: {model_a.model_['gamma']}")
print()

predictions_a = model_a.predict(h=12)['mean']
print("Predictions (h=12):")
for i, pred in enumerate(predictions_a, 1):
    print(f"  Step {i:2d}: {pred:.6f}")
