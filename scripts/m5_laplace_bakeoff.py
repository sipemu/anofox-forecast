#!/usr/bin/env python3
"""M5 Laplace bakeoff: skaters.laplace (Python) vs anofox-forecast
LaplaceForecaster (Rust).

Reads Rust-side predictions from a JSONL file produced by
`examples/m5_bakeoff_export.rs`, runs `skaters.laplace` on the SAME
series with the SAME rolling-one-step protocol, and scores both streams
uniformly with log-likelihood and CRPS on the FIRST-DIFFERENCED target.

Per the skaters bakeoff protocol:
    - Target = one-step change Δy_t = y_t - y_{t-1}
    - Burn-in = 300 obs
    - Rolling one-step prediction with the same stride

Metrics:
    - LL:   log-likelihood (higher is better) — the mixture density at
            the actual observation, in nats
    - CRPS: continuous ranked probability score (lower is better),
            Grimit et al. (2006) closed-form for Gaussian mixtures

Per-series win-rate on both metrics + wall-clock cost comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from timemachines import laplace  # `pip install timemachines`


# ---------------------------------------------------------------------------
# Distribution scoring (identical formulas applied to both sides).
# ---------------------------------------------------------------------------


class GaussianMixtureDist:
    """Simple Gaussian mixture — components as list of (w, mu, sigma)."""

    __slots__ = ("components",)

    def __init__(self, components: List[Tuple[float, float, float]]):
        # Normalize weights defensively.
        s = sum(w for w, _, _ in components)
        if s <= 0:
            self.components = [(1.0, 0.0, 1.0)]
        else:
            self.components = [(w / s, mu, max(sigma, 1e-9)) for w, mu, sigma in components]

    def logpdf(self, y: float) -> float:
        """log Σ_i w_i · Normal(y | μ_i, σ_i) via log-sum-exp."""
        log_pdfs = np.array(
            [math.log(w) + norm.logpdf(y, mu, sigma) for w, mu, sigma in self.components]
        )
        m = log_pdfs.max()
        return float(m + math.log(np.exp(log_pdfs - m).sum()))

    def crps(self, y: float) -> float:
        """Closed-form Gaussian-mixture CRPS (Grimit et al. 2006)."""

        def A(mu_diff: float, sigma_sum: float) -> float:
            z = mu_diff / sigma_sum
            return sigma_sum * (
                z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi)
            )

        w = np.array([c[0] for c in self.components])
        mu = np.array([c[1] for c in self.components])
        sig = np.array([c[2] for c in self.components])

        # term1 = Σ_i w_i · A(y - μ_i, σ_i)
        term1 = float((w * np.array([A(y - mu[i], sig[i]) for i in range(len(w))])).sum())

        # term2 = Σ_i Σ_j w_i · w_j · A(μ_i - μ_j, √(σ_i² + σ_j²))
        term2 = 0.0
        for i in range(len(w)):
            for j in range(len(w)):
                s_sum = math.sqrt(sig[i] ** 2 + sig[j] ** 2)
                term2 += w[i] * w[j] * A(mu[i] - mu[j], s_sum)
        return term1 - 0.5 * term2


def normal_dist(mu: float, sigma: float) -> GaussianMixtureDist:
    return GaussianMixtureDist([(1.0, mu, max(sigma, 1e-9))])


# ---------------------------------------------------------------------------
# Load Rust exports + run Python skaters.laplace on the same series/points.
# ---------------------------------------------------------------------------


def load_rust_predictions(path: Path) -> Dict[Tuple[str, int], dict]:
    """Return {(series_id, t): row_dict}."""
    rows: Dict[Tuple[str, int], dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows[(row["series_id"], int(row["t"]))] = row
    return rows


def load_m5(csv_path: Path, series_ids: List[str]) -> Dict[str, np.ndarray]:
    """Load only the requested M5 series columns and trim leading NaN.

    Mirrors the Rust exporter: skip leading NaN (M5 SKUs weren't all
    introduced at the start of the panel), reject series with any
    interior NaN. Both sides therefore operate on identical values.
    """
    df = pd.read_csv(csv_path)
    out: Dict[str, np.ndarray] = {}
    for sid in series_ids:
        if sid not in df.columns:
            continue
        v = df[sid].values.astype(float)
        # First non-NaN index; drop everything before it.
        real_idx = np.where(~np.isnan(v))[0]
        if len(real_idx) == 0:
            continue
        v = v[real_idx[0]:]
        if np.isnan(v).any():
            continue  # interior NaN — skip series entirely
        out[sid] = v
    return out


def rolling_skaters_laplace(
    values: np.ndarray, points: List[int], burn_in: int
):
    """Run skaters.laplace on the first-differenced series. Returns
    `{i: skaters_Dist}` — the skaters Dist itself, since it already
    provides `.logpdf` / `.crps` matching our GaussianMixtureDist."""
    changes = np.diff(values).tolist()
    # objective="likelihood" avoids the NaN pruning bug in skaters'
    # default `crps_leaf` terminal. `sticky=True` is skaters' default —
    # matches our Rust `.with_sticky()` in `.skaters()`.
    f = laplace(k=1, objective="likelihood", sticky=True)
    state = None
    pending = None
    point_set = set(points)
    out = {}
    for i, y in enumerate(changes):
        if pending is not None and i in point_set:
            # `pending` is a list of Dist (one per horizon); k=1 → 1 element.
            dist = pending[0] if isinstance(pending, list) else pending
            out[i] = dist
        pending, state = f(float(y), state)
    return out


# ---------------------------------------------------------------------------
# Score + report.
# ---------------------------------------------------------------------------


def score_pair(
    rust_rows: Dict[Tuple[str, int], dict],
    py_rows,
    series_id: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return (rs_ll, rs_crps, py_ll, py_crps) matched at the same (series, t)."""
    rs_ll: List[float] = []
    rs_crps: List[float] = []
    py_ll: List[float] = []
    py_crps: List[float] = []
    for (sid, t), row in rust_rows.items():
        if sid != series_id:
            continue
        py_dist = py_rows.get((sid, t))
        if py_dist is None:
            continue
        comps = [(c["w"], c["mu"], c["sigma"]) for c in row["components"]]
        rs_dist = GaussianMixtureDist(comps)
        y = float(row["actual"])
        rs_ll.append(rs_dist.logpdf(y))
        rs_crps.append(rs_dist.crps(y))
        py_ll.append(py_dist.logpdf(y))
        py_crps.append(py_dist.crps(y))
    return rs_ll, rs_crps, py_ll, py_crps


def main() -> int:
    p = argparse.ArgumentParser(description="M5 Laplace bakeoff — Rust vs Python")
    p.add_argument("--rust-jsonl", type=Path, default=Path("/tmp/bakeoff_rs.jsonl"))
    p.add_argument("--m5-csv", type=Path, default=Path("validation/data/m5_full.csv"))
    p.add_argument("--burn-in", type=int, default=300)
    p.add_argument("--out-summary", type=Path, default=Path("/tmp/bakeoff_summary.json"))
    args = p.parse_args()

    if not args.rust_jsonl.exists():
        print(f"missing: {args.rust_jsonl}", file=sys.stderr)
        return 1
    if not args.m5_csv.exists():
        print(f"missing: {args.m5_csv}", file=sys.stderr)
        return 1

    print(f"Loading Rust predictions from {args.rust_jsonl}...")
    rs_rows = load_rust_predictions(args.rust_jsonl)
    series_ids = sorted({sid for sid, _ in rs_rows.keys()})
    points_by_series: Dict[str, List[int]] = defaultdict(list)
    for (sid, t) in rs_rows.keys():
        points_by_series[sid].append(t)
    print(f"  {len(rs_rows)} predictions across {len(series_ids)} series")

    print(f"Loading M5 series...")
    values_by_id = load_m5(args.m5_csv, series_ids)
    print(f"  {len(values_by_id)} series loaded")

    # Run skaters.laplace on the SAME series/points as the Rust exporter.
    print(f"Running skaters.laplace (burn_in={args.burn_in})...")
    t0 = time.time()
    py_dists = {}
    for sid, values in values_by_id.items():
        pts = points_by_series[sid]
        per_series = rolling_skaters_laplace(values, pts, args.burn_in)
        for t, d in per_series.items():
            py_dists[(sid, t)] = d
    py_wall = time.time() - t0
    print(f"  {len(py_dists)} skaters predictions in {py_wall:.1f}s")

    # Score both streams on the same matched (series, t) pairs.
    print("Scoring...")
    per_series_agg = []
    all_rs_ll: List[float] = []
    all_rs_crps: List[float] = []
    all_py_ll: List[float] = []
    all_py_crps: List[float] = []
    rs_wins_ll = py_wins_ll = 0
    rs_wins_crps = py_wins_crps = 0
    for sid in series_ids:
        rs_ll, rs_crps, py_ll, py_crps = score_pair(rs_rows, py_dists, sid)
        if not rs_ll:
            continue
        rs_ll_mean = float(np.mean(rs_ll))
        py_ll_mean = float(np.mean(py_ll))
        rs_crps_mean = float(np.mean(rs_crps))
        py_crps_mean = float(np.mean(py_crps))
        per_series_agg.append(
            {
                "series_id": sid,
                "n": len(rs_ll),
                "rs_ll": rs_ll_mean,
                "py_ll": py_ll_mean,
                "rs_crps": rs_crps_mean,
                "py_crps": py_crps_mean,
            }
        )
        all_rs_ll += rs_ll
        all_rs_crps += rs_crps
        all_py_ll += py_ll
        all_py_crps += py_crps
        if rs_ll_mean > py_ll_mean:
            rs_wins_ll += 1
        else:
            py_wins_ll += 1
        if rs_crps_mean < py_crps_mean:
            rs_wins_crps += 1
        else:
            py_wins_crps += 1

    total_series = len(per_series_agg)
    total_preds = len(all_rs_ll)

    summary = {
        "n_series": total_series,
        "n_predictions": total_preds,
        "burn_in": args.burn_in,
        "aggregate": {
            "rs_ll_mean_nats": float(np.mean(all_rs_ll)) if all_rs_ll else None,
            "py_ll_mean_nats": float(np.mean(all_py_ll)) if all_py_ll else None,
            "rs_crps_mean": float(np.mean(all_rs_crps)) if all_rs_crps else None,
            "py_crps_mean": float(np.mean(all_py_crps)) if all_py_crps else None,
        },
        "per_series_win_rate": {
            "ll_rs_wins": rs_wins_ll,
            "ll_py_wins": py_wins_ll,
            "ll_rs_win_rate": rs_wins_ll / total_series if total_series else None,
            "crps_rs_wins": rs_wins_crps,
            "crps_py_wins": py_wins_crps,
            "crps_rs_win_rate": rs_wins_crps / total_series if total_series else None,
        },
        "runtime": {
            "python_skaters_wall_s": py_wall,
            "python_per_pred_ms": (py_wall / max(total_preds, 1)) * 1000,
        },
    }
    print("\n=== M5 Laplace Bakeoff Summary ===")
    print(json.dumps(summary, indent=2))
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
