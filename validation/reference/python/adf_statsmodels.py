"""ADF/statsmodels reference fixture generator (DIAG-01, DIAG-02).

Loads the series and Monte-Carlo blocks from
tests/data/r_reference/adf_kpss_r.json (same data as the R generator) and
writes tests/data/r_reference/adf_statsmodels.json with:

  - fixed-lag adfuller(maxlag=k, autolag=None) statistics for
    regression in {"n", "c", "ct"}, lag in {0, 1, 4} (cross-checks urca)
  - autolag AIC/BIC/t-stat selection (maxlag = floor((n-1)^(1/3))) per
    regression: adfstat, usedlag, nobs, pvalue, critical values
  - the 40+40 Monte-Carlo series: autolag AIC, regression "c", maxlag 5
  - mackinnonp / mackinnoncrit grids for the parity tests
  - the N=1 rows of the MacKinnon coefficient tables, dumped programmatically

Regenerate with:
  cd validation && uv run python reference/python/adf_statsmodels.py
(uses the already-locked validation/.venv — installs nothing)
"""

import json
import math
from pathlib import Path

import numpy
import scipy
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.adfvalues import (
    mackinnonp,
    mackinnoncrit,
    tau_max_nc,
    tau_max_c,
    tau_max_ct,
    tau_min_nc,
    tau_min_c,
    tau_min_ct,
    tau_star_nc,
    tau_star_c,
    tau_star_ct,
    tau_nc_smallp,
    tau_c_smallp,
    tau_ct_smallp,
    tau_nc_largep,
    tau_c_largep,
    tau_ct_largep,
    tau_2010s,
)

GENERATOR_PATH = "validation/reference/python/adf_statsmodels.py"
REGENERATE_CMD = "cd validation && uv run python reference/python/adf_statsmodels.py"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "data" / "r_reference"

REGRESSIONS = ["n", "c", "ct"]
FIXED_LAGS = [0, 1, 4]


def provenance():
    return {
        "generator": GENERATOR_PATH,
        "regenerate": REGENERATE_CMD,
        "tool_versions": {
            "python": __import__("platform").python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }


def load_r_fixture():
    path = FIXTURES_DIR / "adf_kpss_r.json"
    with open(path) as f:
        return json.load(f)


def fixed_lag_block(y):
    out = {}
    for reg in REGRESSIONS:
        lag_out = {}
        for lag in FIXED_LAGS:
            adfstat, pvalue, usedlag, nobs, critvalues = adfuller(
                y, maxlag=lag, regression=reg, autolag=None
            )
            lag_out[f"lag{lag}"] = {
                "statistic": float(adfstat),
                "pvalue": float(pvalue),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
                "critvalues": {k: float(v) for k, v in critvalues.items()},
            }
        out[reg] = lag_out
    return out


def autolag_block(y):
    n = len(y)
    maxlag = int(math.floor((n - 1) ** (1.0 / 3.0)))
    out = {}
    for reg in REGRESSIONS:
        reg_out = {}
        for method in ["aic", "bic", "t-stat"]:
            adfstat, pvalue, usedlag, nobs, critvalues, icbest = adfuller(
                y, maxlag=maxlag, regression=reg, autolag=method
            )
            reg_out[method.replace("-", "_")] = {
                "statistic": float(adfstat),
                "pvalue": float(pvalue),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
                "critvalues": {k: float(v) for k, v in critvalues.items()},
                "icbest": float(icbest),
            }
        out[reg] = reg_out
    out["maxlag"] = maxlag
    return out


def mc_block(series_list):
    out = []
    for y in series_list:
        adfstat, pvalue, usedlag, nobs, critvalues, icbest = adfuller(
            y, maxlag=5, regression="c", autolag="aic"
        )
        out.append(
            {
                "statistic": float(adfstat),
                "pvalue": float(pvalue),
                "usedlag": int(usedlag),
                "nobs": int(nobs),
                "cv_5pct": float(critvalues["5%"]),
                "reject_5pct": bool(adfstat < critvalues["5%"]),
            }
        )
    return out


def mackinnon_tables():
    """N=1 rows of every coefficient table, dumped programmatically."""
    return {
        "tau_max": {"n": float(tau_max_nc[0]), "c": float(tau_max_c[0]), "ct": float(tau_max_ct[0])},
        "tau_min": {"n": float(tau_min_nc[0]), "c": float(tau_min_c[0]), "ct": float(tau_min_ct[0])},
        "tau_star": {"n": float(tau_star_nc[0]), "c": float(tau_star_c[0]), "ct": float(tau_star_ct[0])},
        "tau_smallp": {
            "n": [float(v) for v in tau_nc_smallp[0]],
            "c": [float(v) for v in tau_c_smallp[0]],
            "ct": [float(v) for v in tau_ct_smallp[0]],
        },
        "tau_largep": {
            "n": [float(v) for v in tau_nc_largep[0]],
            "c": [float(v) for v in tau_c_largep[0]],
            "ct": [float(v) for v in tau_ct_largep[0]],
        },
        "tau_2010": {
            "n": [[float(v) for v in row] for row in tau_2010s["n"][0]],
            "c": [[float(v) for v in row] for row in tau_2010s["c"][0]],
            "ct": [[float(v) for v in row] for row in tau_2010s["ct"][0]],
        },
    }


def mackinnonp_grid():
    stats = [round(-7.0 + 0.1 * i, 1) for i in range(0, int((3.0 - (-7.0)) / 0.1) + 1)]
    out = {}
    for reg in REGRESSIONS:
        out[reg] = [{"statistic": s, "pvalue": float(mackinnonp(s, regression=reg, N=1))} for s in stats]
    return out


def mackinnoncrit_grid():
    nobs_grid = [20, 50, 100, 250, 500, 10000]
    out = {}
    for reg in REGRESSIONS:
        rows = []
        for nobs in nobs_grid:
            cv = mackinnoncrit(N=1, regression=reg, nobs=nobs)
            rows.append(
                {
                    "nobs": nobs,
                    "cv_1pct": float(cv[0]),
                    "cv_5pct": float(cv[1]),
                    "cv_10pct": float(cv[2]),
                }
            )
        out[reg] = rows
    return out


def json_safe(obj):
    """Recursively replace non-finite floats with the quoted sentinels
    jsonlite already uses on the R side ("Inf"/"-Inf"/"NaN"), so the Rust
    loader's existing non-finite handling covers both fixtures the same way.
    Plain json.dump would otherwise emit the bare (non-standard) tokens
    Infinity/-Infinity/NaN, which serde_json rejects.
    """
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        if math.isnan(obj):
            return "NaN"
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def main():
    r_fixture = load_r_fixture()
    series = r_fixture["series"]
    mc = r_fixture["monte_carlo"]

    fixture = {
        "provenance": provenance(),
        "adf_fixed_lag": {name: fixed_lag_block(y) for name, y in series.items()},
        "adf_autolag": {name: autolag_block(y) for name, y in series.items()},
        "monte_carlo": {
            "random_walk": mc_block(mc["random_walk_series"]),
            "ima_theta_neg_0_5": mc_block(mc["ima_theta_neg_0_5_series"]),
        },
        "mackinnon_tables": mackinnon_tables(),
        "mackinnonp_grid": mackinnonp_grid(),
        "mackinnoncrit_grid": mackinnoncrit_grid(),
    }

    fixture = json_safe(fixture)

    out_path = FIXTURES_DIR / "adf_statsmodels.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=False, allow_nan=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
