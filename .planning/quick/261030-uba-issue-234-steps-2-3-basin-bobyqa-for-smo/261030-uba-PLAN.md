---
quick_id: 261030-uba
issue: 234
branch: feat/basin-optimizer
---

# Plan: GH #234 steps 2-3 (BOBYQA evaluation, M4 benchmark, switch decision)

## What
Step 1 (quick 261030-go2) put basin Nelder-Mead / L-BFGS-B behind the `basin` feature +
`--cfg anofox_basin_optimizer` and measured parity. Remaining per the issue:
2. Try basin BOBYQA for the box-bounded smoothing parameters (ETS / SES / Holt / HW / Theta) vs
   the current Nelder-Mead: fit quality and speed.
3. M4 accuracy + time benchmark, legacy vs basin. Switch over and delete the hand-rolled
   optimizers + `lbfgs` ONLY if parity AND speed hold; otherwise record numbers and keep legacy.

## How
- `bobyqa_basin` in `basin_backend.rs` (unit-cube scaling for mixed-scale ETS boxes, finite
  penalty for `f64::MAX` inadmissible values, NM fallback without a finite box) and a
  `minimize_bounded` router used by the ES / Theta smoothing optimizers; BOBYQA only with
  `--features basin` + `--cfg anofox_basin_bobyqa`, so the default build is unchanged.
- `examples/m4_optimizer_benchmark.rs` on the locally available M4 CSVs
  (`~/.datasetsforecast/m4/datasets`, no download): 300 series per frequency, sMAPE / MASE /
  in-sample MSE / time, paired compare mode. Run legacy, basin, BOBYQA, basin+BOBYQA, legacy
  again (timing noise).
- Full test suite under default and BOBYQA routing.

## Verification
- Default routing: full suite green, clippy, WASM (incl. basin + BOBYQA cfg).
- Numbers and decision in `docs/audits/basin-optimizer-parity.md`.
