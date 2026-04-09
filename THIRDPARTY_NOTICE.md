# Third-Party Notices

This project incorporates and builds upon work from the following sources.

## PostForecasts.jl

The postprocessing module (`src/postprocess/`) is a Rust port of functionality from [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl), a Julia package for probabilistic forecast postprocessing.

**Original Authors**: Bartosz Lipiecki and contributors

**License**: MIT License

**Reference**: The design and algorithms are based on the following research:

- Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

### Features Ported

- Conformal Prediction (Split, Cross-Validation, Jackknife+)
- Historical Simulation
- Normal Predictor
- Isotonic Distributional Regression (IDR)
- Quantile Regression Averaging (QRA)
- Conformalize (recalibration of quantile forecasts)

## Conformal Prediction

The conformal prediction implementation is based on:

- Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems*, 32.

- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive Inference with the Jackknife+. *The Annals of Statistics*, 49(1), 486-507.

## Isotonic Distributional Regression

The IDR implementation is based on:

- Henzi, A., Ziegel, J. F., & Gneiting, T. (2021). Isotonic Distributional Regression. *Journal of the Royal Statistical Society: Series B*, 83(5), 963-993.

## Quantile Regression Averaging

The QRA implementation is based on:

- Nowotarski, J., & Weron, R. (2015). Computing Electricity Spot Price Prediction Intervals Using Quantile Regression and Forecast Averaging. *Computational Statistics*, 30(3), 791-803.

## Forecasting: Principles and Practice (FPP3)

The ETS (Error-Trend-Seasonal) model taxonomy and validation rules are based on [Forecasting: Principles and Practice, 3rd edition](https://otexts.com/fpp3/) by Rob J. Hyndman and George Athanasopoulos.

**Reference**:

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition, OTexts: Melbourne, Australia. https://otexts.com/fpp3/

### Features Based on FPP3

- ETS model notation (e.g., "ANN", "AAA", "MAM", "AAdM")
- ETS model validation (rejection of unstable combinations MAA, MAdA)
- ETS taxonomy with 16 valid model specifications
- Exponential smoothing state-space framework

## StatsForecast

Some forecasting models and baseline methods are inspired by [StatsForecast](https://github.com/Nixtla/statsforecast) by Nixtla.

**License**: Apache License 2.0

## changepoint.forecast

The sequential monitoring module (`src/monitor/`) is a Rust port of the R package [changepoint.forecast](https://github.com/grundy95/changepoint.forecast), which performs online changepoint detection on forecast errors via Page's CUSUM and the original CUSUM detectors.

**Original Authors**: Thomas Grundy and Rebecca Killick (Lancaster University)

**License**: MIT License

### Features Ported

- Four CUSUM detectors (`PageCusum`, `PageCusum1`, `Cusum`, `Cusum1`) with raw / squared / both error transformations
- Time-varying threshold via the `√m · (1 + k/m) · (k/(k+m))^γ` weight function
- Online state update with constant-size state per stream (`SequentialDetector::update`)
- Pre-simulated critical-value lookup table (4 detectors × 19 γ × 3 α) and Wiener Monte-Carlo simulator
- Full `Forecaster` trait integration via `monitor_forecaster` (in-sample residuals) and `monitor_forecaster_cv` (rolling-origin CV residuals)

**References**:

- Fremdt, S. (2014). Page's sequential procedure for change-point detection in time series regression. *Statistics*, 49(1), 128–155. <https://doi.org/10.1080/02331888.2014.921899>
- Grundy, T., Killick, R., & Mihaylov, G. (2020). High-dimensional changepoint detection via a geometrically inspired mapping. *Statistics and Computing*, 30, 1155–1166. <https://doi.org/10.1007/s11222-020-09940-y>
- Aue, A., & Horváth, L. (2004). Delay time in sequential detection of change. *Statistics & Probability Letters*, 67(3), 221–231. <https://doi.org/10.1016/j.spl.2004.01.001>

## tsfresh

The time series feature extraction module (`src/features/`) is inspired by [tsfresh](https://github.com/blue-yonder/tsfresh), a Python library for automatic extraction of relevant features from time series.

**Original Authors**: Blue Yonder GmbH and contributors

**License**: MIT License

**Reference**:

- Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package). *Neurocomputing*, 307, 72-77.

### Features Inspired By tsfresh

- Approximate entropy, sample entropy, permutation entropy
- Complexity measures (C3, CID, Lempel-Ziv)
- Autocorrelation and partial autocorrelation features
- Distribution features (skewness, kurtosis)
- Trend and stationarity tests

## Other Dependencies

This project uses several Rust crates as dependencies. See `Cargo.toml` for the complete list. Each dependency is used under its respective license:

- `chrono` - MIT/Apache-2.0
- `faer` - MIT
- `statrs` - MIT
- `thiserror` - MIT/Apache-2.0
- `rand` - MIT/Apache-2.0
- `rustfft` - MIT/Apache-2.0
- `rayon` - MIT/Apache-2.0
- `anofox-regression` - MIT

---

If you believe any attribution is missing or incorrect, please open an issue or pull request.
