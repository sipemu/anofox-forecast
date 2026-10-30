---
quick_id: 261030-uba
issue: 234
branch: feat/basin-optimizer
status: complete
commits: [1386b27]
---

# Summary: GH #234 steps 2-3 -- BOBYQA evaluation, M4 benchmark, decision

## Outcome
**Recommendation: do not switch to basin; keep the hand-rolled Nelder-Mead / L-BFGS and the
`lbfgs` dependency.** The issue's rule (switch only if parity AND speed hold) is not met. Nothing
was deleted. Full numbers: `docs/audits/basin-optimizer-parity.md` (steps 2, 3, decision).

## Done
- `bobyqa_basin` + `minimize_bounded` router (SES / Holt / HW / ETS / Theta family call sites);
  BOBYQA only with `--features basin` + `RUSTFLAGS="--cfg anofox_basin_bobyqa"`. Default build
  behaviour unchanged. 4 unit tests for the adapter (bounded quadratic, active bound with mixed
  scales, inadmissible `f64::MAX` region, fallback without a finite box / empty point).
- `examples/m4_optimizer_benchmark.rs` (run / compare) on the local M4 CSVs; 1800 series
  (300 per frequency), AutoARIMA on 150.

## Key numbers
| | legacy | basin NM/L-BFGS-B | basin BOBYQA |
|---|--:|--:|--:|
| full test suite (all features) | 3795 pass | 3791 pass (step 1) | 3788 pass, **3 fail** (`ets_r_reference`: MAM loglik -705.8 vs R -682.4 / legacy -681.1; AirPassengers AutoETS AICc 1415.5 vs R 1400.6) |
| M4 AutoETS sMAPE / MASE | 11.714 / 2.500 | 11.714 / 2.500 | 10.781 / 2.305 (gain only on Hourly; worse on Y/Q/M/D) |
| M4 Holt sMAPE / MASE | 26.139 / 9.567 | identical | 30.096 / 11.747 |
| M4 HoltWinters sMAPE / MASE | 13.160 / 1.254 | identical | 15.635 / 1.882 |
| M4 SES, Theta, DOTM, AutoARIMA | -- | identical | ties (within 0.005 sMAPE) |
| M4 total wall time | 574 s (repeat 515 s) | 575 s | 444 s |
| time vs legacy per model | -- | 1.00-1.19x | AutoETS 0.72x, SES 1.82x, HW 1.57x, Theta 1.03-1.12x |

Legacy-vs-legacy repeat: bit-identical forecasts, 8-16% timing difference (shared machine), so
time ratios within about +-15% are noise.

## Deviations
1. The PLAN was written after the implementation, not before (process slip; the plan content
   matches what was done).
2. Machine was shared with other jobs (DuckDB builds) during all timing runs; the legacy repeat
   run quantifies the noise instead of assuming a quiet machine.
3. BOBYQA tuning was limited to a sweep of `rho_beg`, penalty scale and `npt` on the failing
   ETS reference fits; none reached legacy quality on all multiplicative specs. The shipped
   adapter keeps `rho_beg = 0.1`.
4. The step-1 cfg (`anofox_basin_optimizer`) and the new `anofox_basin_bobyqa` are independent;
   `--all-features` CI builds compile and unit-test both backends without routing to them.

## Open / for the maintainer
- Whether to keep the `basin` evaluation code on this branch (merge as an opt-in experiment) or
  drop the branch, given the negative switch decision.
- If BOBYQA is pursued: investigate the AutoETS M4-Hourly gain and try BOBYQA as an *additional*
  start inside ETS's multi-start (best of NM and BOBYQA) instead of a replacement.

## Gates
- `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings` (default and
  `--cfg anofox_basin_bobyqa`): clean.
- `cargo test --all-features --no-fail-fast` (default routing): 3795 passed, 0 failed.
- WASM: `cargo check --target wasm32-unknown-unknown --features postprocess,js` and with
  `basin` + `--cfg anofox_basin_bobyqa`: OK.
