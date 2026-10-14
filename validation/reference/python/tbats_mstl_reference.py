#!/usr/bin/env python3
"""TBATS/MSTL reference generator for the Phase 11 interval-fix plan (UPST-05, D-10).

Generates, for each of four real validation series (seasonal, noisy_seasonal,
multiplicative_seasonal, trend_seasonal; column `value`, period 12), exactly
what statsforecast's own `TBATS(season_length=12)`, `MSTL(season_length=12)`
(default `AutoETS(model="ZZN")` trend) and
`MSTL(season_length=12, trend_forecaster=Naive())` report at h=1..12: the
point forecast and the lo/hi bounds at the 80% and 95% levels, plus whatever
fitted TBATS parameters statsforecast exposes on its `model_` dict.

This fixture is a secondary comparison point (width ratio vs. statsforecast),
not ground truth — the primary correctness evidence for the crate's own TBATS
interval fix is `tbats_variance_matches_simulation` (Monte-Carlo simulation
through the crate's own state recursion), per the plan's interfaces note.

Run with (locked env, do not install anything):
    cd validation && uv run python reference/python/tbats_mstl_reference.py
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from statsforecast.models import MSTL, TBATS, Naive

PERIOD = 12
HORIZON = 12
LEVELS = [80, 95]

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = REPO_ROOT / "validation"
DATA_DIR = VALIDATION_DIR / "data"
OUT_PATH = REPO_ROOT / "tests" / "data" / "r_reference" / "tbats_mstl_reference.json"

SERIES_FILES = {
    "seasonal": "seasonal.csv",
    "noisy_seasonal": "noisy_seasonal.csv",
    "multiplicative_seasonal": "multiplicative_seasonal.csv",
    "trend_seasonal": "trend_seasonal.csv",
}


def load_csv_values(path: Path) -> list[float]:
    values: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["value"]))
    return values


def _ver(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def fit_tbats(y: np.ndarray) -> dict:
    model = TBATS(season_length=PERIOD)
    model.fit(y)
    pred = model.predict(h=HORIZON, level=LEVELS)
    mod = model.model_
    return {
        "mean_forecast": [float(v) for v in pred["mean"]],
        "lo_80": [float(v) for v in pred["lo-80"]],
        "hi_80": [float(v) for v in pred["hi-80"]],
        "lo_95": [float(v) for v in pred["lo-95"]],
        "hi_95": [float(v) for v in pred["hi-95"]],
        "box_cox_lambda": (
            None if mod["BoxCox_lambda"] is None else float(mod["BoxCox_lambda"])
        ),
        "sigma2": float(mod["sigma2"]),
        "k_vector": [int(k) for k in mod["k_vector"]],
    }


def fit_mstl(y: np.ndarray, *, naive_trend: bool) -> dict:
    kwargs = {"trend_forecaster": Naive()} if naive_trend else {}
    model = MSTL(season_length=PERIOD, **kwargs)
    model.fit(y)
    pred = model.predict(h=HORIZON, level=LEVELS)
    return {
        "mean_forecast": [float(v) for v in pred["mean"]],
        "lo_80": [float(v) for v in pred["lo-80"]],
        "hi_80": [float(v) for v in pred["hi-80"]],
        "lo_95": [float(v) for v in pred["lo-95"]],
        "hi_95": [float(v) for v in pred["hi-95"]],
    }


def main() -> None:
    results: dict[str, dict] = {}
    for name, filename in SERIES_FILES.items():
        values = load_csv_values(DATA_DIR / filename)
        y = np.array(values)
        results[name] = {
            "values": [float(v) for v in values],
            "tbats": fit_tbats(y),
            "mstl_autoets": fit_mstl(y, naive_trend=False),
            "mstl_naive": fit_mstl(y, naive_trend=True),
        }

    out = {
        "provenance": {
            "generator": "validation/reference/python/tbats_mstl_reference.py",
            "regenerate_command": (
                "cd validation && uv run python reference/python/tbats_mstl_reference.py"
            ),
            "tool_versions": {
                "python": platform.python_version(),
                "numpy": _ver("numpy"),
                "statsforecast": _ver("statsforecast"),
            },
            "seed": None,
            "note": (
                "No simulation here (real validation CSVs, deterministic fit) — "
                "seed is null."
            ),
        },
        "period": PERIOD,
        "horizon": HORIZON,
        "levels": LEVELS,
        "series": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    for name, r in results.items():
        t = r["tbats"]
        print(
            f"  {name}: tbats lambda={t['box_cox_lambda']} sigma2={t['sigma2']:.4f} "
            f"mean[0]={t['mean_forecast'][0]:.4f} width95[0]="
            f"{t['hi_95'][0] - t['lo_95'][0]:.4f}"
        )


if __name__ == "__main__":
    main()
