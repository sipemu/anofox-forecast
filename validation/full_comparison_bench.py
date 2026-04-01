"""Benchmark statsforecast models for speed comparison with Rust."""
import numpy as np
import time
from statsforecast.models import (
    AutoARIMA, AutoETS, Naive, SeasonalNaive,
    RandomWalkWithDrift, SimpleExponentialSmoothing,
    Holt, HoltWinters, Theta, OptimizedTheta,
    DynamicOptimizedTheta, CrostonSBA, MFLES
)

np.random.seed(42)

def make_series(n, trend=0.0, seasonal_amp=0.0, period=1, level=50.0):
    t = np.arange(n, dtype=np.float64)
    y = level + trend * t
    if seasonal_amp > 0 and period > 1:
        y += seasonal_amp * np.sin(2 * np.pi * t / period)
    y += np.array([(int(i * 7 + 3) % 11) * 0.3 - 1.5 for i in range(n)])
    return y

y_seasonal = make_series(120, trend=0.2, seasonal_amp=8.0, period=12, level=50.0)
y_nonseasonal = make_series(200, trend=0.3, level=50.0)

print("=== Python statsforecast benchmark ===\n")

benchmarks = [
    ("Naive n=200", lambda: Naive(), y_nonseasonal, 1),
    ("SES n=200", lambda: SimpleExponentialSmoothing(alpha=0.3), y_nonseasonal, 1),
    ("Holt n=200", lambda: Holt(), y_nonseasonal, 1),
    ("Theta n=200", lambda: Theta(season_length=1), y_nonseasonal, 1),
    ("OptimizedTheta n=200", lambda: OptimizedTheta(season_length=1), y_nonseasonal, 1),
    ("AutoETS n=120 p=12", lambda: AutoETS(season_length=12), y_seasonal, 12),
    ("AutoARIMA n=200", lambda: AutoARIMA(season_length=1), y_nonseasonal, 1),
    ("AutoARIMA n=120 p=12", lambda: AutoARIMA(season_length=12), y_seasonal, 12),
    ("CrostonSBA n=60", lambda: CrostonSBA(), np.array([0,0,3,0,0,2,0,4,0,0,1,0]*5, dtype=np.float64), 1),
    ("MFLES n=120 p=12", lambda: MFLES(season_length=12), y_seasonal, 12),
]

for label, factory, y, period in benchmarks:
    # Warmup
    m = factory(); m.fit(y); m.predict(h=12)

    n_runs = 10
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        m = factory()
        m.fit(y)
        m.predict(h=12)
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    print(f"  {label:<30s}: {avg:>10.3f} ms")
