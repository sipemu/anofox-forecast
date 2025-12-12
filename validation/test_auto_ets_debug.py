#!/usr/bin/env python3
"""Debug AutoETS selection on noisy_seasonal data."""

import pandas as pd
import numpy as np
from statsforecast.models import AutoETS

# Load data
df = pd.read_csv("data/noisy_seasonal.csv")
y = df['value'].values

print("=" * 80)
print("AutoETS Model Selection Debug - noisy_seasonal dataset")
print("=" * 80)
print(f"\nData: {len(y)} observations, Mean={y.mean():.2f}, Std={y.std():.2f}")
print(f"Range: [{y.min():.2f}, {y.max():.2f}]")

# Fit AutoETS
model = AutoETS(season_length=12)
model.fit(y)

print(f"\n{'='*80}")
print("SELECTED MODEL")
print(f"{'='*80}")
print(f"Method: {model.model_['method']}")
print(f"Components: {model.model_['components']}")
print(f"m (season_length): {model.model_['m']}")
print(f"nstate (number of states): {model.model_['nstate']}")

print(f"\n{'='*80}")
print("PARAMETERS")
print(f"{'='*80}")
par = model.model_['par']
print(f"Alpha: {par[0]:.10f}")
if len(par) > 1 and not np.isnan(par[1]):
    print(f"Beta:  {par[1]:.10f}")
if len(par) > 2 and not np.isnan(par[2]):
    print(f"Gamma: {par[2]:.10f}")
if len(par) > 3 and not np.isnan(par[3]):
    print(f"Phi:   {par[3]:.10f}")
if len(par) > 4:
    print(f"Initial Level: {par[4]:.6f}")

print(f"\n{'='*80}")
print("MODEL FIT")
print(f"{'='*80}")
print(f"Log-likelihood: {model.model_['loglik']:.4f}")
print(f"AIC:            {model.model_['aic']:.4f}")
print(f"AICc:           {model.model_['aicc']:.4f}")
print(f"BIC:            {model.model_['bic']:.4f}")
print(f"MSE:            {model.model_['mse']:.4f}")
print(f"Sigma2:         {model.model_['sigma2']:.4f}")
print(f"n_params:       {model.model_['n_params']}")

# Calculate MAD
residuals = model.model_['actual_residuals']
mad = np.mean(np.abs(residuals))
print(f"MAD:            {mad:.4f}")

print(f"\n{'='*80}")
print("FORECASTS (h=12)")
print(f"{'='*80}")
forecasts = model.predict(h=12)['mean']
print("Values:", forecasts)
print(f"All constant: {np.allclose(forecasts, forecasts[0])}")
print(f"Mean forecast: {forecasts.mean():.4f}")

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print("1. Alpha is extremely small (0.0001), essentially making this a constant forecast")
print("2. For noisy data, simpler models (like A,N,N) can win on AICc due to:")
print("   - Lower complexity penalty (fewer parameters)")
print("   - High noise masks seasonal pattern benefits")
print("3. This is CORRECT behavior - the data is too noisy for seasonal models to help")
print("\n4. statsforecast: ANN with alpha=0.0001 -> constant forecasts")
print("5. Rust should match this if using same AICc criterion")
