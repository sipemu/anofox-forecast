#!/usr/bin/env python3
"""FRED-md continuous-data Laplace bakeoff.

Same shape as `scripts/m5_laplace_bakeoff.py` but on FRED-md monthly
macroeconomic series (continuous smooth data, no discrete-repeat
structure).

Answers: does sticky-lattice's WQL blowup on continuous panels come
from our port or from an inherent skaters design issue?

Both sides run `objective="likelihood", sticky=False` — the fair
continuous-data comparison against our `.skaters().no_sticky()`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm
from timemachines import laplace


class GaussianMixtureDist:
    __slots__ = ("components",)

    def __init__(self, components: List[Tuple[float, float, float]]):
        s = sum(w for w, _, _ in components)
        if s <= 0:
            self.components = [(1.0, 0.0, 1.0)]
        else:
            self.components = [(w / s, mu, max(sigma, 1e-9)) for w, mu, sigma in components]

    def logpdf(self, y: float) -> float:
        log_pdfs = np.array(
            [math.log(w) + norm.logpdf(y, mu, sigma) for w, mu, sigma in self.components]
        )
        m = log_pdfs.max()
        return float(m + math.log(np.exp(log_pdfs - m).sum()))

    def crps(self, y: float) -> float:
        def A(mu_diff: float, sigma_sum: float) -> float:
            z = mu_diff / sigma_sum
            return sigma_sum * (
                z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi)
            )

        w = np.array([c[0] for c in self.components])
        mu = np.array([c[1] for c in self.components])
        sig = np.array([c[2] for c in self.components])
        term1 = float((w * np.array([A(y - mu[i], sig[i]) for i in range(len(w))])).sum())
        term2 = 0.0
        for i in range(len(w)):
            for j in range(len(w)):
                s_sum = math.sqrt(sig[i] ** 2 + sig[j] ** 2)
                term2 += w[i] * w[j] * A(mu[i] - mu[j], s_sum)
        return term1 - 0.5 * term2


def load_rust_predictions(path: Path) -> Dict[Tuple[str, int], dict]:
    rows: Dict[Tuple[str, int], dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows[(row["series_id"], int(row["t"]))] = row
    return rows


def parse_tsf(path: Path) -> Dict[str, np.ndarray]:
    """Load .tsf format — same as our Rust parser. tsf files use Latin-1 encoding."""
    text = path.read_text(encoding="latin-1")
    lines = text.splitlines()
    in_data = False
    out: Dict[str, np.ndarray] = {}
    for line in lines:
        if not in_data:
            if line.strip().startswith("@data"):
                in_data = True
            continue
        toks = line.split(":")
        if len(toks) < 2:
            continue
        name = toks[0]
        try:
            vals = np.array(
                [float(x) for x in toks[-1].split(",") if x.strip()],
                dtype=float,
            )
        except ValueError:
            continue
        if vals.size > 0:
            out[name] = vals
    return out


def rolling_skaters_laplace(values: np.ndarray, points: List[int]):
    """Run skaters.laplace on the first-differenced continuous series.

    `sticky=False` matches our `.no_sticky()` on the Rust side —
    the fair continuous-data comparison.
    """
    changes = np.diff(values).tolist()
    f = laplace(k=1, objective="likelihood", sticky=False)
    state = None
    pending = None
    point_set = set(points)
    out = {}
    for i, y in enumerate(changes):
        if pending is not None and i in point_set:
            dist = pending[0] if isinstance(pending, list) else pending
            out[i] = dist
        try:
            pending, state = f(float(y), state)
        except Exception:
            # skaters bug: `NoneType` indexing in Dist.prune when a
            # candidate produces NaN. Skip this series' remaining
            # points rather than crash the whole run.
            break
    return out


def score_pair(
    rust_rows: Dict[Tuple[str, int], dict],
    py_rows,
    series_id: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
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
    p = argparse.ArgumentParser(description="FRED-md continuous-data Laplace bakeoff")
    p.add_argument("--rust-jsonl", type=Path, default=Path("/tmp/bakeoff_fred_rs.jsonl"))
    p.add_argument("--tsf", type=Path, default=Path("validation/data/fred_md.tsf"))
    p.add_argument("--burn-in", type=int, default=100)
    args = p.parse_args()

    if not args.rust_jsonl.exists():
        print(f"missing: {args.rust_jsonl}", file=sys.stderr)
        return 1
    if not args.tsf.exists():
        print(f"missing: {args.tsf}", file=sys.stderr)
        return 1

    print(f"Loading Rust predictions from {args.rust_jsonl}...")
    rs_rows = load_rust_predictions(args.rust_jsonl)
    series_ids = sorted({sid for sid, _ in rs_rows.keys()})
    points_by_series: Dict[str, List[int]] = defaultdict(list)
    for (sid, t) in rs_rows.keys():
        points_by_series[sid].append(t)
    print(f"  {len(rs_rows)} predictions across {len(series_ids)} series")

    print(f"Loading TSF from {args.tsf}...")
    values_by_id = parse_tsf(args.tsf)
    print(f"  {len(values_by_id)} series loaded from TSF")

    print(f"Running skaters.laplace (sticky=False, burn_in={args.burn_in})...")
    t0 = time.time()
    py_dists = {}
    for sid in series_ids:
        values = values_by_id.get(sid)
        if values is None:
            continue
        pts = points_by_series[sid]
        per_series = rolling_skaters_laplace(values, pts)
        for t, d in per_series.items():
            py_dists[(sid, t)] = d
    py_wall = time.time() - t0
    print(f"  {len(py_dists)} skaters predictions in {py_wall:.1f}s")

    print("Scoring...")
    all_rs_ll: List[float] = []
    all_rs_crps: List[float] = []
    all_py_ll: List[float] = []
    all_py_crps: List[float] = []
    rs_wins_ll = py_wins_ll = 0
    rs_wins_crps = py_wins_crps = 0
    n_series_ok = 0
    for sid in series_ids:
        rs_ll, rs_crps, py_ll, py_crps = score_pair(rs_rows, py_dists, sid)
        if not rs_ll:
            continue
        n_series_ok += 1
        rs_ll_mean = float(np.mean(rs_ll))
        py_ll_mean = float(np.mean(py_ll))
        rs_crps_mean = float(np.mean(rs_crps))
        py_crps_mean = float(np.mean(py_crps))
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

    total_preds = len(all_rs_ll)
    summary = {
        "n_series": n_series_ok,
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
            "crps_rs_wins": rs_wins_crps,
            "crps_py_wins": py_wins_crps,
        },
        "runtime": {"python_skaters_wall_s": py_wall},
    }
    print("\n=== FRED-md Laplace Bakeoff Summary (continuous data, sticky=False both sides) ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
