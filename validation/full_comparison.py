"""Full validation: statsforecast reference values for comparison with Rust."""
import numpy as np
import json
from statsforecast.models import (
    AutoARIMA, AutoETS, Naive, SeasonalNaive,
    RandomWalkWithDrift, SimpleExponentialSmoothing,
    Holt, HoltWinters, Theta, OptimizedTheta,
    DynamicOptimizedTheta, CrostonSBA, MFLES
)

np.random.seed(42)

# === Generate deterministic test series ===
def make_series(n, trend=0.0, seasonal_amp=0.0, period=1, level=50.0):
    t = np.arange(n, dtype=np.float64)
    y = level + trend * t
    if seasonal_amp > 0 and period > 1:
        y += seasonal_amp * np.sin(2 * np.pi * t / period)
    # Deterministic pseudo-noise (no randomness)
    y += np.array([(int(i * 7 + 3) % 11) * 0.3 - 1.5 for i in range(n)])
    return y

series = {
    "flat": make_series(100, trend=0.0, level=50.0),
    "trend": make_series(100, trend=0.5, level=10.0),
    "seasonal_p12": make_series(120, trend=0.2, seasonal_amp=8.0, period=12, level=50.0),
    "seasonal_p7": make_series(140, trend=0.1, seasonal_amp=5.0, period=7, level=30.0),
    "intermittent": np.array([0,0,3,0,0,2,0,4,0,0,1,0,0,0,5,0,0,3,0,0,
                              2,0,0,0,4,0,1,0,0,3,0,0,2,0,0,0,5,0,0,1,
                              0,0,3,0,0,2,0,4,0,0,1,0,0,0,5,0,0,3,0,0], dtype=np.float64),
}

horizon = 12
results = {}

for name, y in series.items():
    results[name] = {"values": y.tolist(), "horizon": horizon, "models": {}}

    # Models to run per series type
    models_to_run = []

    if name == "intermittent":
        models_to_run = [
            ("CrostonSBA", CrostonSBA()),
        ]
    elif "seasonal" in name:
        period = 12 if "p12" in name else 7
        models_to_run = [
            ("Naive", Naive()),
            ("SeasonalNaive", SeasonalNaive(season_length=period)),
            ("SES", SimpleExponentialSmoothing(alpha=0.3)),
            ("Holt", Holt()),
            ("AutoETS", AutoETS(season_length=period)),
            ("AutoARIMA", AutoARIMA(season_length=period)),
            ("Theta", Theta(season_length=period)),
            ("OptimizedTheta", OptimizedTheta(season_length=period)),
            ("DOTM", DynamicOptimizedTheta(season_length=period)),
            ("MFLES", MFLES(season_length=period)),
        ]
    else:
        models_to_run = [
            ("Naive", Naive()),
            ("RandomWalkWithDrift", RandomWalkWithDrift()),
            ("SES", SimpleExponentialSmoothing(alpha=0.3)),
            ("Holt", Holt()),
            ("AutoETS", AutoETS(season_length=1)),
            ("AutoARIMA", AutoARIMA(season_length=1)),
            ("Theta", Theta(season_length=1)),
            ("OptimizedTheta", OptimizedTheta(season_length=1)),
        ]

    for model_name, model in models_to_run:
        try:
            model.fit(y)
            pred = model.predict(h=horizon)
            fc = pred["mean"].tolist() if hasattr(pred, "keys") else pred.tolist()

            info = {"forecasts": fc}

            # Extract parameters where available
            if hasattr(model, 'alpha_') and model.alpha_ is not None:
                info["alpha"] = float(model.alpha_)

            if model_name == "AutoARIMA":
                if hasattr(model, 'model_'):
                    info["order"] = str(model.model_)

            if model_name == "AutoETS":
                if hasattr(model, 'model_'):
                    info["model"] = str(model.model_)

            results[name]["models"][model_name] = info
            print(f"  {name}/{model_name}: {fc[:3]}...")

        except Exception as e:
            print(f"  {name}/{model_name}: FAILED - {e}")
            results[name]["models"][model_name] = {"error": str(e)}

# Save results
with open("validation/data/statsforecast_reference.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved reference values to validation/data/statsforecast_reference.json")
print(f"Series: {list(results.keys())}")
for name, data in results.items():
    models = [m for m in data["models"] if "error" not in data["models"][m]]
    print(f"  {name}: {len(models)} models OK")
