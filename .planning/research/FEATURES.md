# Feature Research: Forecasting Library Validation & Benchmark Suite

**Domain:** Forecasting library validation/benchmark harness (accuracy metrics, reference datasets, probabilistic calibration)
**Researched:** 2026-08-09
**Confidence:** MEDIUM (web sources, cross-verified against multiple competition papers and peer library docs)

---

## Context

This document answers: what must a best-in-class forecasting library's validation and benchmark suite measure? The scope is the **measurement harness** for `anofox-forecast` — a mature library (v0.15.8, 30+ models) that already ships cross-validation, conformal prediction, calibration, and quantile methods. The question is not what to build into models, but what to measure, how to measure it rigorously, and what reference datasets to run it against.

The library already has `calculate_metrics` (MASE, MAE, RMSE, sMAPE) and `cross_validate`. This research defines what that surface must grow into to meet industry standards for a published library.

---

## Feature Landscape

### Table Stakes (Peer Libraries Compute These — Missing = Credibility Gap)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **MASE** (Mean Absolute Scaled Error) | Hyndman & Koehler (2006); primary metric in all M-competitions; only scale-free metric that is stable on zero-containing series; used by R `forecast`, `fable`, statsforecast, sktime as default | LOW — formula is trivial; denominator is in-sample seasonal naive MAE | Already partly implemented in `calculate_metrics`; verify it handles seasonal baseline correctly (seasonal period divides in-sample MAE, not just lag-1 naive) |
| **sMAPE** (Symmetric Mean Absolute Percentage Error) | Used in M1–M4 competitions; bounded 0–200%; symmetric treatment of over/under forecasts; universally expected alongside MASE | LOW | Already in `calculate_metrics`; confirm denominator is `(|y| + |ŷ|)/2` not just `|y|` |
| **RMSE** | Scale-dependent but interpretable in original units; needed when comparing across time (same series, different models); standard in all accuracy reports | LOW | Already in `calculate_metrics` |
| **MAE** | Most robust to outliers; interpretable; required alongside RMSE for a complete point-accuracy picture | LOW | Already in `calculate_metrics` |
| **Naive2 baseline** | The Naive2 model (seasonal naive if autocorrelation test passes, otherwise naive) is the mandatory baseline in M-competitions; OWA is computed relative to it; any result without a naive comparison is uninterpretable | MEDIUM — requires autocorrelation test + conditional seasonality; anofox-forecast already has `SeasonalNaive` and `Naive` — the needed addition is the autocorrelation test gate | Without this, MASE denominators may use wrong baseline for yearly/non-seasonal series |
| **Empirical interval coverage** | For any model that produces prediction intervals, the fraction of test observations falling within the interval must be reported and compared to the nominal level (e.g., 95%); missing this means interval claims are unverified | LOW — a proportion test | The library already has conformal/calibration methods; coverage is the most basic sanity check on them |
| **MSIS** (Mean Scaled Interval Score) | Introduced in M4 probabilistic track; rewards both sharpness (narrow intervals) and coverage jointly; the standard for evaluating prediction intervals in competition settings; statsforecast and sktime both compute it; formula: `(1/h) * Σ[(U_t−L_t) + (2/α)(L_t−Y_t)·1(Y_t<L_t) + (2/α)(Y_t−U_t)·1(Y_t>U_t)]` divided by in-sample naive MAE | LOW — given MASE denominator is already available | Not currently in `calculate_metrics`; must add for interval-producing models |
| **Rolling-origin (time-series) cross-validation** | Standard for unbiased hold-out evaluation; random k-fold is wrong for time series (leaks future); all serious libraries use rolling-origin or expanding-window CV; the library already has `cross_validate` — what must be verified is that it never uses future data in training windows | LOW to validate (read the split logic); fixing leakage would be MEDIUM if found | Existing `cross_validate` in `src/utils/cross_validation.rs` — must confirm it generates folds in temporal order with no overlap |
| **Per-frequency stratification** | M-competition results are always reported per frequency (Yearly / Quarterly / Monthly / Weekly / Daily / Hourly) because model behavior differs dramatically by frequency; aggregating across frequencies misleads | LOW — grouping logic | Dataset loaders must tag each series with its frequency; benchmark runner must aggregate per-group |
| **Naive seasonal baseline on evaluation corpus** | All reported numbers must include the SeasonalNaive baseline score on the same dataset/metric for interpretability; a MASE of 0.9 means nothing without knowing the baseline MASE is 1.0 | LOW | Part of reporting design; not a separate metric |

### Differentiators (Peer Libraries Partially Cover These — Doing Them Well Distinguishes anofox-forecast)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **OWA** (Overall Weighted Average) | M4 primary competition ranking metric; `OWA = 0.5*(MASE/MASE_Naive2 + sMAPE/sMAPE_Naive2)`; lets anofox-forecast publish directly comparable numbers against M4 leaderboard; statsforecast computes it for M4 comparisons | MEDIUM — requires both Naive2 MASE and Naive2 sMAPE on the same dataset | Implement only after Naive2 baseline is correct |
| **RMSSE** (Root Mean Squared Scaled Error) | M5 competition primary metric; RMSSE = sqrt(MSE / in-sample naive MSE); preferred over MASE for intermittent/Poisson demand (zero-heavy) because it penalizes variance more; anofox-forecast has Croston/IMAPA/TSB for intermittent demand — RMSSE is the right evaluation metric for those | LOW to implement; MEDIUM to wire to correct baseline | Especially relevant for `SmartForecaster` with intermittent routing |
| **CRPS** (Continuous Ranked Probability Score) | Proper scoring rule for full predictive distributions; equal to integral of pinball loss over [0,1]; evaluates entire distribution, not just intervals; `fabletools` provides this for distributional models; anofox-forecast's `LaplaceForecaster` and `BootstrapPredictor` produce full distributions — CRPS is the correct metric for them | MEDIUM — requires numerical integration over quantiles, or closed-form for Gaussian | The library has `DistributionalForecaster` trait; CRPS measures it correctly |
| **Pinball loss** (quantile score) | Proper scoring rule per quantile; `L_τ(y, q) = (y−q)·τ·1(y≥q) + (q−y)·(1−τ)·1(y<q)`; needed to evaluate conformal/QRA quantile outputs at specific levels (e.g., 10th, 50th, 90th percentiles); statsforecast reports it | LOW | Directly relevant to existing `postprocess` (QRA, conformal) outputs |
| **PIT histogram / uniform test** | Probability Integral Transform test: for a well-calibrated distributional forecast, U = F(Y) where F is the forecast CDF should be uniform on [0,1]; a PIT histogram that deviates from flat reveals bias or variance miscalibration; relevant for LaplaceForecaster's Gaussian/GaussianMixture outputs | MEDIUM — requires CDF evaluation + histogram test (chi-squared or Kolmogorov-Smirnov) | Only needed for models that expose a full CDF; skip for conformal-only intervals |
| **Per-horizon accuracy decomposition** | Cross-validation results broken down by forecast horizon h=1,2,...,H; accuracy typically degrades with horizon; reporting only the average hides systematic horizon-specific model failures; R `accuracy()` on `fable` objects does this automatically | LOW — already available in `CVResults` (has per-horizon mean) — verify aggregation is horizon-wise not just overall | Very useful for diagnosing model-specific degradation patterns |
| **Diebold-Mariano significance test** | Statistical test for whether two models' accuracy difference is significant (not due to chance); required for credible published comparisons against reference implementations; null hypothesis: equal forecast accuracy | MEDIUM — requires asymptotic variance estimation with HAC correction for multi-step horizons | Do not claim "model A beats model B" without a significance test; use DM as gating condition for cross-library comparison claims |
| **M3 competition accuracy results** | Running anofox-forecast against the full M3 corpus (3003 series, all frequencies) and publishing the results is the single most credible external validation possible; peer libraries (statsforecast, sktime) all do this | HIGH — requires M3 loader, per-frequency evaluation runner, Naive2 baseline, MASE+sMAPE+OWA computation, likely 30+ min run time | The M3 dataset is available via Mcomp R package (GPL-3) or Monash Archive (CC-BY 4.0); a subset can be vendored |
| **Conditional coverage by series characteristics** | Separate coverage statistics for short vs. long series, high vs. low volatility, seasonal vs. non-seasonal; reveals where conformal/calibration methods degrade | HIGH — requires series metadata tagging + stratified analysis | Only valuable after basic marginal coverage is confirmed |

### Anti-Features (Do Not Build or Claim These)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **MAPE** (Mean Absolute Percentage Error, asymmetric version) | Familiar to business analysts; percentage format seems intuitive | Undefined when actuals are zero or near-zero; asymmetric (penalizes over-forecasts more than under-forecasts); gives misleading results for intermittent demand — which is a core use case; M4 committee explicitly dropped it in favor of sMAPE | Report sMAPE instead; sMAPE has the same "percentage-like" feel without the pathologies |
| **R²** on forecast horizon | Borrowed from regression evaluation; seems like a natural fit | Negative R² values are common and misleading for multi-step forecasts; not scale-free; cannot be averaged across series; no established forecasting competition uses it | Use MASE instead; it answers the same question (does the model beat a baseline?) with a clearer interpretation |
| **Single hold-out split accuracy** | Simplest to implement; familiar from ML | A single train/test split produces extremely high-variance accuracy estimates for time series (results depend heavily on which historical window happened to be "easy" or "hard"); not comparable to competition results which use the full series | Always use rolling-origin CV or the official competition train/test split; single hold-out is only acceptable when matching competition evaluation protocol exactly |
| **Accuracy on synthetic data only** | Easy to generate; controllable | Synthetic series may not capture real-world patterns (intermittency, fat tails, regime changes); a model that looks excellent on synthetic benchmarks can fail badly on real data; peer libraries all use real competition datasets | Use M3/M4/NN5 real data for accuracy claims; synthetic data is appropriate only for correctness tests (do residuals sum to zero? does the model recover known parameters?) |
| **Automatic seasonal period detection integration into evaluation** | Seems like a fair comparison — let each model choose its own period | Introduces uncontrolled variation into benchmark results making cross-run comparison non-reproducible; period detection is itself an algorithm with bugs; the benchmark should fix the seasonal period externally | Fix seasonal periods from dataset metadata during benchmarking; document the fixed periods used |
| **Aggregate accuracy across all frequencies without stratification** | Single number is easier to communicate | Aggregated numbers hide the fact that most models perform very differently on yearly vs. hourly series; can mask catastrophic failure in one frequency group | Always report per-frequency-group results alongside any aggregate |
| **Wall-clock speed as primary benchmark metric** | Speed matters for production use | Speed without accuracy is meaningless; and speed benchmarks are environment-dependent (CPU, parallelism level); peer libraries publish speed as secondary to accuracy | Report MASE/sMAPE as primary; wall-clock as secondary context; use criterion for reproducible timing |

---

## Dataset Reference Table

All datasets needed for the accuracy harness, with licensing and retrieval path:

| Dataset | Series Count | Frequencies | Primary Use | License | How to Retrieve |
|---------|-------------|-------------|-------------|---------|-----------------|
| **M3** | 3,003 | Yearly, Quarterly, Monthly, Other | Core accuracy benchmark; smallest credible full-competition corpus | GPL-3 (via Mcomp R package) | `install.packages("Mcomp")` then export to CSV; or via Monash Archive (forecastingdata.org, CC-BY 4.0) |
| **M4 (monthly subset)** | 48,000 of 100,000 | Monthly | High-volume monthly benchmark; large enough to be statistically robust | No explicit license on GitHub repo; widely treated as freely usable for research; Kaggle mirror has CC-BY | Mcompetitions/M4-methods GitHub or Nixtla `datasetsforecast` Python package |
| **NN5** | 111 | Daily, Weekly | Short daily series; ATM cash withdrawals; tests daily-frequency behavior | CC-BY 4.0 | Zenodo record 4656110 (with missing values) or 3889740 (without) |
| **Tourism** | 1,311 | Yearly, Quarterly, Monthly | Tourism competition dataset by Athanasopoulos & Hyndman; used by statsforecast | Research use; available from Hyndman's site and Tcomp R package | Tcomp R package or Monash Archive |
| **M1** | 1,001 | Yearly, Quarterly, Monthly | Smaller M-competition dataset; good for fast smoke-tests | GPL-3 (via Mcomp R package) | Same as M3 via Mcomp |
| **M5** | 42,840 (hierarchical) | Daily | Intermittent/retail demand; RMSSE metric; most relevant for Croston/IMAPA models | Kaggle competition terms (restrict to competition use); Zenodo copy CC-BY 4.0 | Zenodo record 12636070 (CC-BY); Kaggle requires account + rule acceptance |

**Recommended vendored corpus for the harness (license-safe, manageable size):**
1. **M3** via Monash Archive (CC-BY 4.0 version) — gold standard, every peer library uses it
2. **NN5** from Zenodo (CC-BY 4.0, tiny 111 series) — daily frequency, tests missing-value handling
3. **M4 monthly subset** — largest, most powerful for statistical significance; use the GitHub copy and document the source in the harness

---

## Metric Formulas (Implementation Reference)

All formulas are given for horizon H, with training set size T and seasonal period m.

**MASE (seasonal):**
```
denominator = (1/(T-m)) * sum_{t=m+1}^{T} |y_t - y_{t-m}|
MASE = (1/H) * sum_{h=1}^{H} |y_{T+h} - ŷ_{T+h}| / denominator
```
For non-seasonal (or yearly) series, use m=1 (lag-1 naive).

**sMAPE:**
```
sMAPE = (2/H) * sum_{h=1}^{H} |y_{T+h} - ŷ_{T+h}| / (|y_{T+h}| + |ŷ_{T+h}|)
```
Bounded [0, 2]. Report as percentage (multiply by 100).

**OWA:**
```
OWA = 0.5 * (MASE / MASE_Naive2) + 0.5 * (sMAPE / sMAPE_Naive2)
```
where Naive2 applies the autocorrelation test: if the series is seasonal (|r_m| > 0.1), use SeasonalNaive; otherwise use Naive.

**RMSSE:**
```
denominator = (1/(T-1)) * sum_{t=2}^{T} (y_t - y_{t-1})^2
RMSSE = sqrt((1/H) * sum_{h=1}^{H} (y_{T+h} - ŷ_{T+h})^2 / denominator)
```

**MSIS (95% interval, α=0.05):**
```
IS_t = (U_t - L_t) + (2/α)(L_t - y_t) * 1(y_t < L_t) + (2/α)(y_t - U_t) * 1(y_t > U_t)
MIS = (1/H) * sum_{h=1}^{H} IS_{T+h}
MSIS = MIS / denominator_MASE   [same denominator as MASE]
```

**Pinball loss at quantile τ:**
```
L_τ(y, q) = (y - q) * τ         if y >= q
           = (q - y) * (1 - τ)   if y < q
```

**Empirical coverage (nominal α):**
```
Coverage = (1/H) * sum_{h=1}^{H} 1(L_{T+h} <= y_{T+h} <= U_{T+h})
```
Expected: 1 - α. Acceptable tolerance: ±0.02 for small H; tighten for large panels.

---

## Feature Dependencies

```
Naive2 baseline
    └──required by──> MASE (correct denominator for yearly series)
    └──required by──> OWA

MASE
    └──required by──> OWA

sMAPE
    └──required by──> OWA

Rolling-origin CV (temporally correct)
    └──required by──> all accuracy metrics (invalid if splits leak future)

Dataset loader (M3/M4/NN5 with frequency tags)
    └──required by──> per-frequency stratification
    └──required by──> MASE denominator (seasonal period from dataset metadata)

Empirical coverage check
    └──required by──> MSIS interpretation (MSIS can look good if model cheats on width)

Interval output (point + lower + upper)
    └──required by──> MSIS, empirical coverage, pinball at 5th/95th quantile

Distributional output (CDF/quantiles)
    └──required by──> CRPS
    └──required by──> PIT histogram

DM test
    └──required by──> cross-library accuracy claims (gating condition)
```

---

## Prioritization Matrix

| Capability | User Value | Implementation Cost | Priority |
|------------|------------|---------------------|----------|
| Verify existing MASE/sMAPE/RMSE/MAE correctness (formula audit) | HIGH | LOW | P1 |
| Add MSIS to metric suite | HIGH | LOW | P1 |
| Empirical coverage check (binomial test vs. nominal) | HIGH | LOW | P1 |
| M3 dataset loader + rolling-origin harness | HIGH | MEDIUM | P1 |
| Naive2 baseline computation (with autocorrelation gate) | HIGH | MEDIUM | P1 |
| Per-frequency stratification in benchmark runner | HIGH | LOW | P1 |
| OWA computation | MEDIUM | LOW (once Naive2 exists) | P2 |
| RMSSE (for intermittent demand models) | MEDIUM | LOW | P2 |
| Pinball loss at target quantiles | MEDIUM | LOW | P2 |
| NN5 dataset loader (daily, missing values) | MEDIUM | LOW | P2 |
| M4 monthly subset loader | MEDIUM | MEDIUM (large download) | P2 |
| CRPS for distributional outputs (LaplaceForecaster) | MEDIUM | MEDIUM | P2 |
| Per-horizon accuracy decomposition in reporting | MEDIUM | LOW | P2 |
| Diebold-Mariano significance test | MEDIUM | MEDIUM | P3 |
| PIT histogram / KS test for Gaussian calibration | LOW | MEDIUM | P3 |
| Conditional coverage by series characteristic | LOW | HIGH | P3 |
| M5 dataset loader (intermittent retail, CC-BY Zenodo) | LOW | MEDIUM | P3 |

---

## Cross-Library Comparison Methodology

**The fair benchmark protocol (how statsforecast, sktime, and R forecast do it):**

1. Use the official M3 or M4 competition train/test split — do not re-split
2. Fit each model on the official training set, predict the official horizon
3. Compute MASE and sMAPE using the training set as the baseline denominator (not the test set)
4. Stratify results by frequency group
5. Compare against Naive2 baseline on the same split to compute OWA
6. Report wall-clock time on a fixed machine as secondary context

**What makes a comparison invalid:**
- Re-splitting the data differently from the competition protocol
- Using future data in any preprocessing step (scaling, decomposition) that spans train/test boundary
- Comparing averaged-across-frequency OWA when series counts differ (weight by series count or report per-group)
- Claiming superiority without a DM significance test when the difference is < 5%

**Recommended reference implementation for cross-library comparison:**
Nixtla `statsforecast` AutoETS and AutoARIMA on M3 monthly — their numbers are published, reproducible, and widely cited. Use their published M3 results (MASE ~0.93 for AutoETS monthly) as the target to match.

---

## Sources

- Hyndman & Koehler (2006) — MASE original paper — cited via multiple web sources (MEDIUM confidence, cross-verified)
- M4 Competitors Guide (UNIC, 2018) — MSIS formula — [M4-Competitors-Guide.pdf](https://www.unic.ac.cy/test/wp-content/uploads/sites/2/2018/09/M4-Competitors-Guide.pdf)
- Makridakis et al. M4 Competition results — [ScienceDirect](https://www.sciencedirect.com/article/pii/S0169207021001874) (MEDIUM)
- Nixtla datasetsforecast M4 API — [nixtlaverse.nixtla.io/datasetsforecast/m4.html](https://nixtlaverse.nixtla.io/datasetsforecast/m4.html) (MEDIUM)
- Monash Time Series Forecasting Archive — [forecastingdata.org](https://forecastingdata.org/) — CC-BY 4.0 confirmed (MEDIUM)
- Mcomp R package (M1, M3 data, GPL-3) — [pkg.robjhyndman.com/Mcomp](https://pkg.robjhyndman.com/Mcomp/) (MEDIUM)
- NN5 Zenodo CC-BY 4.0 — [zenodo.org/records/4656110](https://zenodo.org/records/4656110) (MEDIUM)
- M4-methods GitHub — [github.com/Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods)
- Forecast Evaluation Pitfalls — [arxiv.org/pdf/2203.10716](https://arxiv.org/pdf/2203.10716) (MEDIUM)
- Diebold-Mariano test overview — [wisdomlib.org/concept/diebold-mariano-test](https://www.wisdomlib.org/concept/diebold-mariano-test) (LOW)
- sktime M4 replication paper — [researchgate.net](https://www.researchgate.net/publication/341477809) (MEDIUM)

---

*Feature research for: forecasting library validation & benchmark suite*
*Researched: 2026-08-09*
