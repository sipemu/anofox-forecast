#!/usr/bin/env python3
"""Generate parity-test fixtures from the ruptures Python library.

Run with:
    /tmp/ruptures-venv/bin/python3 scripts/generate_ruptures_fixtures.py

Pinned to ruptures==1.1.9. Emits one JSON file per (algorithm, cost,
mode) combination under tests/data/ruptures_fixtures/. The Rust
integration test tests/changepoint_ruptures_parity.rs loads these and
asserts that the Rust port produces identical results within
numerical tolerance.

Conventions
-----------
- Breakpoints (`bkps`) follow ruptures' convention: sorted list of
  segment-end indices (exclusive), with the terminal `n` included.
- All seeds are explicit (numpy.random.seed) for reproducibility.
- Cost values are reported to 12 significant digits; breakpoints are
  exact integer matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ruptures as rpt

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "data" / "ruptures_fixtures"
RUPTURES_VERSION = rpt.__version__


def write_fixture(name: str, payload: dict) -> None:
    """Serialise a fixture, including a header so the test can refuse
    if the on-disk version differs from the runtime."""
    payload = {
        "_ruptures_version": RUPTURES_VERSION,
        "_numpy_version": np.__version__,
        **payload,
    }
    out = FIXTURES_DIR / f"{name}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda x: float(x))
    print(f"  wrote {out.relative_to(ROOT)}")


def total_cost(algo, signal: np.ndarray, bkps: list[int]) -> float:
    """Compute Σ cost(seg) for a fitted algorithm."""
    algo.cost.fit(signal)
    cost_sum = 0.0
    start = 0
    for end in bkps:
        if end > start:
            cost_sum += algo.cost.error(start, end)
        start = end
    return float(cost_sum)


def gen_pelt_l2_level_shift() -> None:
    """Pelt + L2 on a deterministic level-shift signal."""
    np.random.seed(42)
    n_per = 100
    series = np.concatenate(
        [
            0.0 * np.ones(n_per),
            5.0 * np.ones(n_per),
            -3.0 * np.ones(n_per),
        ]
    )
    n = series.size
    algo = rpt.Pelt(model="l2", min_size=5, jump=1).fit(series.reshape(-1, 1))
    bkps = algo.predict(pen=3.0)
    cost = total_cost(algo, series.reshape(-1, 1), bkps)
    write_fixture(
        "pelt_l2_3_level_shifts",
        dict(
            algo="Pelt",
            cost="l2",
            mode="pen",
            params=dict(min_size=5, jump=1, pen=3.0),
            signal=series.tolist(),
            n=int(n),
            bkps=[int(b) for b in bkps],
            total_cost=cost,
        ),
    )


def gen_dynp_l2_n_bkps() -> None:
    """Dynp + L2, fixed n_bkps."""
    np.random.seed(7)
    n_per = 40
    series = np.concatenate(
        [
            np.zeros(n_per),
            5.0 * np.ones(n_per),
            np.zeros(n_per),
            5.0 * np.ones(n_per),
        ]
    )
    n = series.size
    algo = rpt.Dynp(model="l2", min_size=5, jump=1).fit(series.reshape(-1, 1))
    bkps = algo.predict(n_bkps=3)
    cost = total_cost(algo, series.reshape(-1, 1), bkps)
    write_fixture(
        "dynp_l2_n_bkps_3",
        dict(
            algo="Dynp",
            cost="l2",
            mode="n_bkps",
            params=dict(min_size=5, jump=1, n_bkps=3),
            signal=series.tolist(),
            n=int(n),
            bkps=[int(b) for b in bkps],
            total_cost=cost,
        ),
    )


def gen_binseg_l2_pen() -> None:
    """Binseg + L2 with penalty."""
    np.random.seed(13)
    n_per = 50
    series = np.concatenate(
        [
            np.zeros(n_per),
            8.0 * np.ones(n_per),
            np.zeros(n_per),
        ]
    )
    n = series.size
    algo = rpt.Binseg(model="l2", min_size=5, jump=1).fit(series.reshape(-1, 1))
    bkps = algo.predict(pen=5.0)
    cost = total_cost(algo, series.reshape(-1, 1), bkps)
    write_fixture(
        "binseg_l2_pen_5",
        dict(
            algo="Binseg",
            cost="l2",
            mode="pen",
            params=dict(min_size=5, jump=1, pen=5.0),
            signal=series.tolist(),
            n=int(n),
            bkps=[int(b) for b in bkps],
            total_cost=cost,
        ),
    )


def gen_bottom_up_l2_n_bkps() -> None:
    np.random.seed(21)
    n_per = 40
    series = np.concatenate(
        [
            np.zeros(n_per),
            5.0 * np.ones(n_per),
            -2.0 * np.ones(n_per),
        ]
    )
    n = series.size
    algo = rpt.BottomUp(model="l2", min_size=5, jump=5).fit(series.reshape(-1, 1))
    bkps = algo.predict(n_bkps=2)
    cost = total_cost(algo, series.reshape(-1, 1), bkps)
    write_fixture(
        "bottom_up_l2_n_bkps_2",
        dict(
            algo="BottomUp",
            cost="l2",
            mode="n_bkps",
            params=dict(min_size=5, jump=5, n_bkps=2),
            signal=series.tolist(),
            n=int(n),
            bkps=[int(b) for b in bkps],
            total_cost=cost,
        ),
    )


def gen_window_l2_n_bkps() -> None:
    """Window + L2 with explicit width."""
    np.random.seed(33)
    n_per = 50
    series = np.concatenate(
        [
            np.zeros(n_per),
            7.0 * np.ones(n_per),
            np.zeros(n_per),
        ]
    )
    n = series.size
    algo = rpt.Window(width=20, model="l2", min_size=5, jump=2).fit(
        series.reshape(-1, 1)
    )
    bkps = algo.predict(n_bkps=2)
    cost = total_cost(algo, series.reshape(-1, 1), bkps)
    write_fixture(
        "window_l2_n_bkps_2",
        dict(
            algo="Window",
            cost="l2",
            mode="n_bkps",
            params=dict(min_size=5, jump=2, width=20, n_bkps=2),
            signal=series.tolist(),
            n=int(n),
            bkps=[int(b) for b in bkps],
            total_cost=cost,
        ),
    )


def main() -> None:
    print(f"ruptures=={RUPTURES_VERSION}, numpy=={np.__version__}")
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    gen_pelt_l2_level_shift()
    gen_dynp_l2_n_bkps()
    gen_binseg_l2_pen()
    gen_bottom_up_l2_n_bkps()
    gen_window_l2_n_bkps()
    print(f"\nWrote fixtures to {FIXTURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
