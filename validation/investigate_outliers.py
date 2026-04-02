"""Deep investigation: what does statsforecast select for each outlier series?"""
import json
import numpy as np
from statsforecast.models import AutoARIMA

# Load outlier data
with open("tests/data/m4_outliers.json") as f:
    data = json.load(f)

print("=== statsforecast AutoARIMA on outlier series ===\n")
print(f"{'Series':<8} {'n':>6} {'SF_MAE':>10} {'Order':>20} {'d':>3} {'Forecast[0:3]':>30}")

results = {}
for uid, series_data in sorted(data.items()):
    train = np.array(series_data["train"], dtype=np.float64)
    test = np.array(series_data["test"], dtype=np.float64)
    n = len(train)

    m = AutoARIMA(season_length=7)
    m.fit(train)
    pred = m.predict(h=14)
    fc = pred["mean"]
    mae = np.mean(np.abs(fc - test[:14]))

    # Extract model info
    model_str = str(m.model_) if hasattr(m, 'model_') else "?"
    d = int(m.model_["arima"]["order"][1]) if hasattr(m, 'model_') and "arima" in str(m.model_) else "?"

    results[uid] = {
        "n": n,
        "mae": float(mae),
        "model": model_str,
        "forecasts": fc.tolist(),
    }

    print(f"{uid:<8} {n:>6} {mae:>10.1f} {model_str:>20} {str(d):>3} {str(fc[:3].round(1)):>30}")

# Save for Rust comparison
with open("validation/data/outlier_sf_reference.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to validation/data/outlier_sf_reference.json")
