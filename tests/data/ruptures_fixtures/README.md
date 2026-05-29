# Ruptures parity fixtures

These JSON files capture the input signal, parameters, and detected
breakpoints + total cost as produced by the
[`ruptures`](https://github.com/deepcharles/ruptures) Python library
version **1.1.9** running against numpy 2.4.6.

They power `tests/changepoint_ruptures_parity.rs`, which loads each
fixture, runs the Rust port with identical parameters, and asserts:

- **exact integer match** on the breakpoint list for the deterministic
  detectors (Pelt, Dynp, Binseg, BottomUp);
- **±2·jump** tolerance for the Window detector (approximate scoring);
- **total cost** matching to 1e-6 relative tolerance across all five.

## Regenerating

```bash
# One-time venv setup
uv venv /tmp/ruptures-venv --python 3.12
/tmp/ruptures-venv/bin/python3 -m pip install "ruptures==1.1.9" numpy

# Regenerate
/tmp/ruptures-venv/bin/python3 scripts/generate_ruptures_fixtures.py
```

Each JSON includes a `_ruptures_version` and `_numpy_version` header so
the Rust test can refuse if the runtime drifts from what produced the
fixture.

## Coverage

| Fixture | Algorithm | Cost | Mode | Notes |
|---|---|---|---|---|
| `pelt_l2_3_level_shifts.json` | Pelt | L2 | pen=3.0 | 3 segments, deterministic level shift |
| `dynp_l2_n_bkps_3.json` | Dynp | L2 | n_bkps=3 | 4 segments, exact DP |
| `binseg_l2_pen_5.json` | Binseg | L2 | pen=5.0 | greedy binary segmentation |
| `bottom_up_l2_n_bkps_2.json` | BottomUp | L2 | n_bkps=2 | agglomerative merge |
| `window_l2_n_bkps_2.json` | Window | L2 | n_bkps=2 | sliding-window scoring, width=20 |

Adding wider parity (other costs, noisy signals, multivariate, kernel
costs) is straightforward — add a `gen_*` function to the script and a
matching `#[test]` to the Rust file.
