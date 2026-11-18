---
quick_id: 261118-tx2
status: complete
issues: [245, 246, 247]
commits: [5f224a9, 4c36001, 615845d, 4e9db9f]
branch: fix/245-246-247-autoets-autoarima
---

# Summary — 261118-tx2

## #246 AutoETS scale dependence — fixed (5f224a9)
- The cause the issue names (seasonal likelihood on n − period residuals) was already gone:
  removed in v0.16.0 (2561223); the reporter ran 0.15.x.
- A remaining scale dependence was found with the new 60-series panel test (×1000 / ÷1000):
  1/60 series switched ETS(M,A,N) → ETS(A,A,N) at ÷1000. Cause: the multi-start early exit
  `init > 3 * best` on the negative log-likelihood, whose sign flips with the data's units.
  Replaced with a log-likelihood-unit margin (`2n`).
- Before: 1 switch (complete pool). After: 0 switches, complete and damped-only pools.

## #245 Seasonal minimum — implemented (4c36001)
- R forecast 9.0.2 (checked locally): seasonal dropped only when `n <= m`; `n <= npars + 4`
  falls back to HoltWinters; below 3m R initialises seasonals with a Fourier regression.
- `AutoETSConfig::min_seasonal_cycles` (default and floor 2) + `with_min_seasonal_cycles`;
  ANOVA strength check now runs from 2 cycles. Tests: 24/30/35-obs strongly seasonal series
  select seasonal; 23 obs never fit seasonal; `with_min_seasonal_cycles(3)` restores old rule.

## #247 Seasonal AutoARIMA — 1.6x, plus R's approximation knob (615845d)
- Profile (timing counters): ~99% of time in the `ARIMA_Like` Kalman filter port, ~10.4k
  likelihood evaluations; `getQ0` ~40% of the (·)(1,1,2) fits.
- Filter loops made column-contiguous, zero `delta` coefficients skipped; bitwise identical
  to the old filter on 160 randomised orders (temporary test), identical evaluation counts.
- Interleaved, same machine/load, best of 5: baseline 3.94–4.29 s → 2.55–2.68 s.
  R `auto.arima` on the same series/machine: ~6.5 s (same model selected).
- `AutoARIMAConfig::approximation` (R's argument): forced on = 35 ms, selects
  ARIMA(1,0,1)(0,1,1)[12] with drift — same as R with `approximation = TRUE`.
- Not reached: "well under 1 s" with the default exact-likelihood search; that would need
  fewer evaluations (changing R's optimiser trajectory) or parallel gradients.

## Verification
- `cargo test --release --all-features`: 3786 passed, 0 failed.
- `cargo fmt --check` clean. Clippy (`--all-targets --all-features -D warnings`) on the
  1.100 toolchain fails on pre-existing lints in untouched code (`needless_bool` in
  `src/models/arima/model.rs:717`, `needless_range_loop` in `examples/fev_benchmark.rs`);
  with those allowed, no findings in changed files.
