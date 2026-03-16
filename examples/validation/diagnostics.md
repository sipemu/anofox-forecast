# Model Diagnostics

**Run:** `cargo run --example diagnostics`

## What this example demonstrates

Statistical diagnostic tests for validating time series models. It covers residual autocorrelation tests (Ljung-Box, Box-Pierce, Durbin-Watson) and stationarity tests (ADF, KPSS, combined), with interpretation guidance for each.

## Sections

1. **Ljung-Box test** -- Tests both white-noise and autocorrelated residuals for significant autocorrelation using `ljung_box`, checking `is_white_noise` at the 0.05 level.
2. **Box-Pierce test** -- Runs the simpler Box-Pierce variant via `box_pierce` and notes the difference from Ljung-Box.
3. **Durbin-Watson test** -- Detects first-order autocorrelation in residuals using `durbin_watson`, demonstrating positive, negative, and no-autocorrelation cases with the `AutocorrelationType` enum.
4. **ADF test** -- Runs `adf_test` on stationary, random walk, and trending series, printing test statistics, p-values, and critical values.
5. **KPSS test** -- Runs `kpss_test` on stationary and trending series, noting the opposite null hypothesis compared to ADF.
6. **Combined ADF + KPSS test** -- Uses `test_stationarity` to run both tests together and produce a unified conclusion string.
7. **Interpretation guide and workflow** -- Explains the four possible ADF/KPSS outcome combinations and outlines a practical pre-/post-modeling diagnostic workflow.

## Key types

- `ljung_box`, `box_pierce` -- residual autocorrelation tests
- `durbin_watson` -- first-order autocorrelation test, returns `AutocorrelationType`
- `adf_test`, `kpss_test` -- unit-root / stationarity tests with critical values
- `test_stationarity` -- combined ADF + KPSS convenience function
