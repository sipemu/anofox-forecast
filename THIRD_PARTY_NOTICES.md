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

---

## MacKinnon ADF response-surface tables (`src/validation/stationarity.rs`)

`mackinnon_p_value` and `mackinnon_critical_values` (used internally by
`adf_test` / `adf_test_with_options`) port the `N = 1` (single series
believed `I(1)`) rows of the MacKinnon response-surface coefficient tables
as shipped in [`statsmodels`](https://github.com/statsmodels/statsmodels)'
`statsmodels.tsa.adfvalues` module (`tau_max_*`, `tau_min_*`, `tau_star_*`,
`tau_*_smallp`, `tau_*_largep` for the MacKinnon (1994) p-value surfaces,
and `tau_2010s` for the MacKinnon (2010) finite-sample critical-value
surfaces). The coefficient values themselves are dumped programmatically
from the installed `statsmodels` package into
`tests/data/r_reference/adf_statsmodels.json` by
`validation/reference/python/adf_statsmodels.py` and proven equal to the
Rust constants by `tests/adf_kpss_reference.rs::mackinnon_tables_match_statsmodels`
— they are not hand-transcribed from the paper.

References:

- MacKinnon, J.G. 1994. "Approximate Asymptotic Distribution Functions for
  Unit-Root and Cointegration Tests." *Journal of Business & Economic
  Statistics*, 12.2, 167–76.
- MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests." Queen's
  University, Dept. of Economics Working Paper 1227.

`statsmodels` issue [#10271](https://github.com/statsmodels/statsmodels/issues/10271)
documents a handful of transcription discrepancies between `statsmodels`'
shipped `tau_2010["c"]` values and the original MacKinnon (2010) paper. This
crate reproduces `statsmodels`' shipped values exactly (not the paper),
since `statsmodels`' own `adfuller`/`mackinnonp`/`mackinnoncrit` is the
oracle this crate's tests are proven against.

Upstream project links:

- Repository: https://github.com/statsmodels/statsmodels
- License: BSD-3-Clause (reproduced below, from `statsmodels` 0.14.6's
  `LICENSE.txt`)

### statsmodels — BSD-3-Clause license

```
Copyright (C) 2006, Jonathan E. Taylor
All rights reserved.

Copyright (c) 2006-2008 Scipy Developers.
All rights reserved.

Copyright (c) 2009-2018 statsmodels Developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of statsmodels nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```
