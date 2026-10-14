# R reference fixture generator for AutoARIMA order-selection parity
# (branch fix/auto-arima-order-selection, phase 11-04/11-05).
#
# This branch has no shared validation/reference/r/common.R of its own
# (each Phase 11 fix branch is cut independently from the same BASE_REF
# and must stay independently mergeable — see PLAN D-01), so this script
# is self-contained. Its helpers mirror the pattern used by the DuckDB
# extension repo's validation/reference/r/common.R (script-dir resolution
# via --file=, a provenance block, deterministic seeding) without
# depending on that file.
#
# Regenerate with:
#   Rscript validation/reference/r/auto_arima.R
#
# Requires: R packages forecast, urca, jsonlite.

suppressMessages({
  library(forecast)
  library(urca)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Script-dir resolution (cwd-independent), mirrors the extension repo's
# validation/reference/r/common.R::ref_script_dir().
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
  file.path(ref_script_dir(), "..", "..", "..", "tests", "data", "r_reference", "auto_arima_r.json")
}

# ---------------------------------------------------------------------------
# Deterministic seeding (fixed order of generation matters for byte-identical
# regeneration): RW set (50), then AR(0.7) set (50), then the 4 seasonal
# series, in that exact order.
# ---------------------------------------------------------------------------
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261009)

n_each <- 50
series_n <- 200

rw_list <- vector("list", n_each)
for (i in seq_len(n_each)) {
  rw_list[[i]] <- cumsum(rnorm(series_n))
}

ar1_list <- vector("list", n_each)
for (i in seq_len(n_each)) {
  ar1_list[[i]] <- as.numeric(arima.sim(list(ar = 0.7), n = series_n))
}

air <- as.numeric(AirPassengers)
log_air <- log(air)

seasonal_rw <- numeric(144)
seasonal_rw[1:12] <- rnorm(12)
for (t in 13:144) {
  seasonal_rw[t] <- seasonal_rw[t - 12] + rnorm(1)
}

deterministic_seasonal <- 10 + sin(2 * pi * (1:144) / 12) * 5 + rnorm(144)

# ---------------------------------------------------------------------------
# Per-series KPSS statistic (urca::ur.kpss, type = "mu", use.lag per
# forecast::ndiffs' default: trunc(3*sqrt(n)/13)) + auto.arima summary.
# ---------------------------------------------------------------------------
kpss_stat_and_lag <- function(y) {
  n <- length(y)
  use_lag <- trunc(3 * sqrt(n) / 13)
  fit <- ur.kpss(y, type = "mu", use.lag = use_lag)
  list(stat = as.numeric(fit@teststat), use_lag = use_lag)
}

has_constant <- function(fit) {
  nm <- names(fit$coef)
  any(grepl("intercept|drift", nm))
}

arima_order_block <- function(fit, seasonal) {
  ord <- arimaorder(fit)
  if (seasonal) {
    list(
      p = unname(ord["p"]), d = unname(ord["d"]), q = unname(ord["q"]),
      cap_p = unname(ord["P"]), cap_d = unname(ord["D"]), cap_q = unname(ord["Q"]),
      m = unname(ord["Frequency"])
    )
  } else {
    list(
      p = unname(ord["p"]), d = unname(ord["d"]), q = unname(ord["q"]),
      cap_p = 0, cap_d = 0, cap_q = 0, m = 1
    )
  }
}

nonseasonal_entry <- function(y) {
  kpss <- kpss_stat_and_lag(y)
  fit <- auto.arima(y)
  ord <- arima_order_block(fit, seasonal = FALSE)
  c(
    list(
      values = y,
      ndiffs = ndiffs(y),
      kpss_stat = kpss$stat,
      kpss_use_lag = kpss$use_lag,
      has_constant = has_constant(fit),
      aicc = as.numeric(fit$aicc)
    ),
    ord
  )
}

seasonal_entry <- function(name, y_raw, period) {
  y <- ts(y_raw, frequency = period)
  kpss <- kpss_stat_and_lag(y_raw)
  nsd <- nsdiffs(y)
  seas_diffed <- if (nsd > 0) diff(y_raw, lag = period, differences = nsd) else y_raw
  d_after_seasonal <- ndiffs(seas_diffed)
  fit <- auto.arima(y)
  ord <- arima_order_block(fit, seasonal = TRUE)
  c(
    list(
      name = name,
      values = y_raw,
      period = period,
      nsdiffs = nsd,
      d_after_seasonal = d_after_seasonal,
      kpss_stat = kpss$stat,
      kpss_use_lag = kpss$use_lag,
      has_constant = has_constant(fit),
      aicc = as.numeric(fit$aicc)
    ),
    ord
  )
}

rw_entries <- lapply(rw_list, nonseasonal_entry)
ar1_entries <- lapply(ar1_list, nonseasonal_entry)

seasonal_entries <- list(
  seasonal_entry("air_passengers", air, 12),
  seasonal_entry("log_air_passengers", log_air, 12),
  seasonal_entry("seasonal_random_walk", seasonal_rw, 12),
  seasonal_entry("deterministic_seasonal", deterministic_seasonal, 12)
)

rw_010_share <- mean(sapply(rw_entries, function(e) {
  e$p == 0 && e$d == 1 && e$q == 0 && e$cap_p == 0 && e$cap_q == 0
}))

ar1_d_ge1_share <- mean(sapply(ar1_entries, function(e) e$d >= 1))

fixture <- list(
  provenance = list(
    family = "auto_arima",
    generator = "validation/reference/r/auto_arima.R",
    regenerate_command = "Rscript validation/reference/r/auto_arima.R",
    tool_versions = list(
      R = R.version.string,
      forecast = as.character(packageVersion("forecast")),
      urca = as.character(packageVersion("urca")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    rng_kind = paste(RNGkind(), collapse = "/"),
    seed = 20261009
  ),
  rw = rw_entries,
  ar1 = ar1_entries,
  seasonal = seasonal_entries,
  rw_010_share = rw_010_share,
  ar1_d_ge1_share = ar1_d_ge1_share
)

out_path <- fixture_path()
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")

cat("Wrote fixture to", out_path, "\n")
cat("rw_010_share =", rw_010_share, "\n")
cat("ar1_d_ge1_share =", ar1_d_ge1_share, "\n")
