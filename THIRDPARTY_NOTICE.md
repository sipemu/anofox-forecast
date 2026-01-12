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

## StatsForecast

Some forecasting models and baseline methods are inspired by [StatsForecast](https://github.com/Nixtla/statsforecast) by Nixtla.

**License**: Apache License 2.0

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
