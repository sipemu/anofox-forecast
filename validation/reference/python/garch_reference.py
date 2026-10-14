#!/usr/bin/env python3
"""GARCH(1,1) reference generator for the Phase 11 interval-coverage diagnostic (UPST-05).

Generates, for each of four series (two simulated from a true GARCH(1,1) DGP
with known parameters, and two real validation series), statsforecast's fitted
coefficients, h=1..12 sigma^2 forecasts, and mean/lo/hi at the 80% and 95%
levels exactly as `statsforecast.models.GARCH` returns them.

The simulated series are used in the Rust diagnostic to compare the crate's
own fit against the TRUE generating parameters (ground truth), independent of
whatever statsforecast itself reports. statsforecast's own numbers are kept
as a secondary comparison point only (see H5 in the findings table — its own
`predict()`/`forecast()` interval formula multiplies the z-quantile by
`sigma2` i.e. variance, not by `sqrt(sigma2)`, which is confirmed by reading
`statsforecast/models.py` lines ~5355-5361 and is NOT treated as ground truth
here per the plan's interfaces note).

Run with (locked env, do not install anything):
    cd validation && uv run python reference/python/garch_reference.py
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from statsforecast.models import GARCH

SEED = 20261009
N = 2000
BURN_IN = 500
OMEGA_TRUE = 0.1
ALPHA_TRUE = 0.1
BETA_TRUE = 0.8
HORIZON = 12
LEVELS = [80, 95]

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = REPO_ROOT / "validation"
DATA_DIR = VALIDATION_DIR / "data"
OUT_PATH = REPO_ROOT / "tests" / "data" / "r_reference" / "garch_reference.json"


def simulate_garch11(
    rng: np.random.Generator, n: int, burn_in: int, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """Simulate a zero-mean GARCH(1,1) path: y_t = eps_t * sqrt(sigma2_t),
    sigma2_t = omega + alpha*y_{t-1}^2 + beta*sigma2_{t-1}. Returns the last `n`
    values after discarding `burn_in` to let the recursion settle away from the
    unconditional-variance initial condition."""
    total = n + burn_in
    sigma2 = np.empty(total)
    y = np.empty(total)
    sigma2[0] = omega / (1.0 - alpha - beta)  # unconditional variance
    y[0] = rng.standard_normal() * np.sqrt(sigma2[0])
    for t in range(1, total):
        sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
        y[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    return y[burn_in:]


def load_csv_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["value"]))
    return values


def fit_statsforecast(name: str, y: np.ndarray) -> dict:
    model = GARCH(p=1, q=1)
    model.fit(y)
    pred = model.predict(h=HORIZON, level=LEVELS)
    coeff = model.model_["coeff"]
    return {
        "series_type": name,
        "n": int(len(y)),
        "coeff_omega": float(coeff[0]),
        "coeff_alpha": float(coeff[1]),
        "coeff_beta": float(coeff[2]),
        "sigma2_forecast": [float(v) for v in pred["sigma2"]],
        "mean_forecast": [float(v) for v in pred["mean"]],
        "lo_80": [float(v) for v in pred["lo-80"]],
        "hi_80": [float(v) for v in pred["hi-80"]],
        "lo_95": [float(v) for v in pred["lo-95"]],
        "hi_95": [float(v) for v in pred["hi-95"]],
    }


def _ver(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    rng = np.random.default_rng(SEED)

    sim_zero_mean = simulate_garch11(rng, N, BURN_IN, OMEGA_TRUE, ALPHA_TRUE, BETA_TRUE)
    sim_mu5 = sim_zero_mean + 5.0

    heteroscedastic = np.array(load_csv_values(DATA_DIR / "heteroscedastic.csv"))
    ar1 = np.array(load_csv_values(DATA_DIR / "ar1.csv"))

    series = {
        "simulated_zero_mean": sim_zero_mean,
        "simulated_mu5": sim_mu5,
        "heteroscedastic": heteroscedastic,
        "ar1": ar1,
    }

    results = {name: fit_statsforecast(name, y) for name, y in series.items()}

    out = {
        "provenance": {
            "generator": "validation/reference/python/garch_reference.py",
            "regenerate_command": "cd validation && uv run python reference/python/garch_reference.py",
            "tool_versions": {
                "python": platform.python_version(),
                "numpy": _ver("numpy"),
                "statsforecast": _ver("statsforecast"),
            },
            "seed": SEED,
        },
        "true_dgp": {
            "omega": OMEGA_TRUE,
            "alpha": ALPHA_TRUE,
            "beta": BETA_TRUE,
            "n": N,
            "burn_in": BURN_IN,
            "note": "simulated_mu5 = simulated_zero_mean + 5.0 (same innovations, shifted mean)",
        },
        "series_values": {
            "simulated_zero_mean": [float(v) for v in sim_zero_mean],
            "simulated_mu5": [float(v) for v in sim_mu5],
        },
        "statsforecast": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    for name, r in results.items():
        print(
            f"  {name}: omega={r['coeff_omega']:.4f} alpha={r['coeff_alpha']:.4f} "
            f"beta={r['coeff_beta']:.4f} sigma2[0]={r['sigma2_forecast'][0]:.4f}"
        )


if __name__ == "__main__":
    main()
