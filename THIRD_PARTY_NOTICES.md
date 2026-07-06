# Third-Party Notices

`anofox-forecast` is MIT-licensed. This file records upstream projects whose
design or concepts inspired parts of the crate, and reproduces their licenses
where required. It does **not** include upstream Rust dependencies — those
carry their own licenses reachable from the `Cargo.lock` file and are picked up
by `cargo license` / `cargo about`.

---

## LaplaceForecaster / distributional shell (`src/models/laplace/`)

The `distributional` feature adds a `LaplaceForecaster` — a streaming,
likelihood-weighted Gaussian mixture over small "leaf" predictors. The
overall shape (streaming leaves, per-observation log-likelihood weighting,
per-horizon `GaussianMixture` output) is inspired by the design in
[`microprediction/skaters`](https://github.com/microprediction/skaters), a
Python/JavaScript distributional forecaster released under the MIT license
by Peter Cotton. The reference paper "Laplace beats (almost) everything"
motivates the leaf composition and the "model first, conform last" split.

`anofox-forecast` implements a Rust port of a small subset of the design
(currently EMA, drift, AR(1), damped-Holt, AR(2), seasonal-EMA leaves; no
CRPS-tuned terminal leaf, no OU / fractional-differencing / Yeo-Johnson
leaves). The implementation was written from scratch against the public
description; no source was copied verbatim. Empirical defaults (which
leaves are on by default; leaf hyperparameters) are chosen based on
`anofox-forecast`'s own benchmarks (see `examples/skaters_m5_benchmark.rs`)
and may diverge materially from skaters' defaults.

Upstream project links:

- Repository: https://github.com/microprediction/skaters
- Documentation & live demos: https://skaters.microprediction.org/
- License: MIT

### skaters — MIT license

```
MIT License

Copyright (c) 2024 Peter Cotton and the skaters contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
