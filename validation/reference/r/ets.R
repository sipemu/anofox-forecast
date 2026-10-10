# R reference fixture generator for ETS likelihood / interval parity
# (branch fix/ets-likelihood-intervals, phase 11-06/11-07/11-08).
#
# This branch has no shared validation/reference/r/common.R of its own
# (each Phase 11 fix branch is cut independently from the same BASE_REF
# and must stay independently mergeable -- see PLAN D-01), so this script
# is self-contained. Its helpers mirror the pattern used by the DuckDB
# extension repo's validation/reference/r/common.R (script-dir resolution
# via --file=, a provenance block, deterministic seeding) and the sibling
# fix/auto-arima-order-selection branch's validation/reference/r/auto_arima.R.
#
# Regenerate with:
#   Rscript validation/reference/r/ets.R
#
# Requires: R packages forecast, jsonlite.

suppressMessages({
  library(forecast)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Script-dir resolution (cwd-independent).
# ---------------------------------------------------------------------------
ref_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    stop("ref_script_dir(): could not find --file= in commandArgs(); run via Rscript")
  }
  script_path <- sub("^--file=", "", file_arg[1])
  normalizePath(dirname(script_path), mustWork = TRUE)
}

fixture_path <- function() {
  file.path(ref_script_dir(), "..", "..", "..", "tests", "data", "r_reference", "ets_r.json")
}

# ---------------------------------------------------------------------------
# Deterministic seeding. `pos` is drawn immediately after the seed is set,
# before AirPassengers (a constant dataset) is touched, so regeneration is
# byte-identical.
# ---------------------------------------------------------------------------
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261009)

pos <- 50 + cumsum(rnorm(120))
if (min(pos) <= 0) {
  stop("pos series is not strictly positive; adjust the generator")
}

air <- AirPassengers

# ---------------------------------------------------------------------------
# Shared forecast-block extractor: mean + 80%/95% interval bounds.
# Positional, not by column name -- hw()/ets() label columns "80%"/"95%",
# which matches here since every caller passes level = c(80, 95).
# ---------------------------------------------------------------------------
fc_block <- function(f) {
  list(
    mean = as.numeric(f$mean),
    lower_80 = as.numeric(f$lower[, 1]),
    upper_80 = as.numeric(f$upper[, 1]),
    lower_95 = as.numeric(f$lower[, 2]),
    upper_95 = as.numeric(f$upper[, 2])
  )
}

# ---------------------------------------------------------------------------
# One (error, trend, seasonal) ETS fit on a given series, with full
# likelihood/IC/state/forecast detail.
# ---------------------------------------------------------------------------
ets_fit_block <- function(y, error, trend, seasonal) {
  model_str <- paste0(error, sub("Ad", "A", trend), seasonal)
  damped <- (trend == "Ad")
  fit <- ets(y, model = model_str, damped = damped, restrict = FALSE)
  fc <- forecast(fit, h = 24, level = c(80, 95))
  states <- fit$states
  list(
    error = error,
    trend = trend,
    seasonal = seasonal,
    method = fit$method,
    components = as.character(fit$components),
    par = as.list(fit$par),
    state_names = colnames(states),
    initial_state = as.numeric(states[1, ]),
    final_state = as.numeric(states[nrow(states), ]),
    loglik = as.numeric(fit$loglik),
    aic = as.numeric(fit$aic),
    aicc = as.numeric(fit$aicc),
    bic = as.numeric(fit$bic),
    sigma2 = as.numeric(fit$sigma2),
    np = length(fit$par) + 1,
    fitted = as.numeric(fitted(fit)),
    residuals = as.numeric(residuals(fit)),
    forecast = fc_block(fc)
  )
}

ets_auto_block <- function(y, label) {
  fit <- ets(y)
  fc <- forecast(fit, h = 24, level = c(80, 95))
  states <- fit$states
  list(
    label = label,
    method = fit$method,
    components = as.character(fit$components),
    par = as.list(fit$par),
    state_names = colnames(states),
    initial_state = as.numeric(states[1, ]),
    final_state = as.numeric(states[nrow(states), ]),
    loglik = as.numeric(fit$loglik),
    aic = as.numeric(fit$aic),
    aicc = as.numeric(fit$aicc),
    bic = as.numeric(fit$bic),
    sigma2 = as.numeric(fit$sigma2),
    np = length(fit$par) + 1,
    fitted = as.numeric(fitted(fit)),
    residuals = as.numeric(residuals(fit)),
    forecast = fc_block(fc)
  )
}

es_family_block <- function(fc, label) {
  m <- fc$model
  states <- m$states
  list(
    label = label,
    method = m$method,
    par = as.list(m$par),
    state_names = colnames(states),
    initial_state = as.numeric(states[1, ]),
    final_state = as.numeric(states[nrow(states), ]),
    sigma2 = as.numeric(m$sigma2),
    forecast = fc_block(fc)
  )
}

# ---------------------------------------------------------------------------
# 18 AirPassengers models: E in {A, M} x T in {N, A, Ad} x S in {N, A, M}.
# ---------------------------------------------------------------------------
errors <- c("A", "M")
trends <- c("N", "A", "Ad")
seasons <- c("N", "A", "M")

air_fits <- list()
for (e in errors) {
  for (t in trends) {
    for (s in seasons) {
      # Notation key matches ETSSpec::from_notation (e.g. "AAdN"), NOT the
      # 3-char ets() `model` argument (which collapses "Ad" to "A" + damped=TRUE).
      key <- paste0(e, t, s)
      air_fits[[key]] <- ets_fit_block(air, e, t, s)
    }
  }
}

# ---------------------------------------------------------------------------
# 6 non-seasonal models on the positive fixture series.
# ---------------------------------------------------------------------------
pos_fits <- list()
for (e in errors) {
  for (t in trends) {
    key <- paste0(e, t, "N")
    pos_fits[[key]] <- ets_fit_block(pos, e, t, "N")
  }
}

# ---------------------------------------------------------------------------
# AutoETS winners.
# ---------------------------------------------------------------------------
air_auto <- ets_auto_block(air, "air_auto")
pos_auto <- ets_auto_block(pos, "pos_auto")

# ---------------------------------------------------------------------------
# Exponential-smoothing family (consumed by plan 11-08).
# ---------------------------------------------------------------------------
ses_fc <- ses(pos, h = 24, level = c(80, 95))
holt_fc <- holt(pos, h = 24, level = c(80, 95))
holt_damped_fc <- holt(pos, damped = TRUE, h = 24, level = c(80, 95))
hw_add_fc <- hw(air, seasonal = "additive", h = 24, level = c(80, 95))
hw_mult_fc <- hw(air, seasonal = "multiplicative", h = 24, level = c(80, 95))

es_family <- list(
  ses = es_family_block(ses_fc, "ses"),
  holt = es_family_block(holt_fc, "holt"),
  holt_damped = es_family_block(holt_damped_fc, "holt_damped"),
  hw_additive = es_family_block(hw_add_fc, "hw_additive"),
  hw_multiplicative = es_family_block(hw_mult_fc, "hw_multiplicative")
)

# ---------------------------------------------------------------------------
# Assemble and write fixture.
# ---------------------------------------------------------------------------
fixture <- list(
  provenance = list(
    family = "ets",
    generator = "validation/reference/r/ets.R",
    regenerate_command = "Rscript validation/reference/r/ets.R",
    tool_versions = list(
      R = R.version.string,
      forecast = as.character(packageVersion("forecast")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    rng_kind = paste(RNGkind(), collapse = "/"),
    seed = 20261009
  ),
  air = as.numeric(air),
  pos = as.numeric(pos),
  air_fits = air_fits,
  pos_fits = pos_fits,
  air_auto = air_auto,
  pos_auto = pos_auto,
  es_family = es_family
)

out_path <- fixture_path()
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")

cat("Wrote fixture to", out_path, "\n")
cat("air_auto$method =", air_auto$method, " aicc =", air_auto$aicc, "\n")
cat("pos_auto$method =", pos_auto$method, " aicc =", pos_auto$aicc, "\n")
