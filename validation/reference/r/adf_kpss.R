# ADF/KPSS reference fixture generator (DIAG-01, DIAG-02).
#
# Writes tests/data/r_reference/adf_kpss_r.json — urca::ur.df fixed-lag
# test statistics (type n/c/ct) and tseries::kpss.test (null="Level",
# lshort=TRUE) statistics/p-values/lags for a fixed battery of series, plus
# a Monte-Carlo block (40 random walks + 40 IMA(1,1) theta=-0.5 series)
# giving urca's AIC-selected statistic and 5% rejection decision.
#
# Regenerate with:
#   Rscript validation/reference/r/adf_kpss.R
# (works from any cwd — the script resolves its own directory via --file=).
#
# No shared common.R in this repo (per Phase 11 Plan 02 conventions) — all
# helpers are inlined below.

suppressMessages({
  library(urca)
  library(tseries)
  library(jsonlite)
})

# ---- script-dir resolution -------------------------------------------------

.script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    stop("could not find --file= in commandArgs(); run via Rscript")
  }
  normalizePath(dirname(sub("^--file=", "", file_arg[1])), mustWork = TRUE)
})

.fixtures_dir <- normalizePath(
  file.path(.script_dir, "..", "..", "..", "tests", "data", "r_reference"),
  mustWork = FALSE
)

GENERATOR_PATH <- "validation/reference/r/adf_kpss.R"
REGENERATE_CMD <- "Rscript validation/reference/r/adf_kpss.R"
SEED <- 20261009

provenance <- function() {
  list(
    generator = GENERATOR_PATH,
    regenerate = REGENERATE_CMD,
    tool_versions = list(
      R = R.version.string,
      urca = as.character(packageVersion("urca")),
      tseries = as.character(packageVersion("tseries")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    rng_kind = paste(RNGkind(), collapse = "/"),
    seed = SEED
  )
}

write_fixture <- function(obj, rel_path) {
  full_path <- file.path(.fixtures_dir, rel_path)
  dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(obj, full_path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
  invisible(full_path)
}

# ---- deterministic series battery ------------------------------------------

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)

rw <- cumsum(rnorm(200))
ar07 <- as.numeric(arima.sim(list(ar = 0.7), n = 200))
ima <- cumsum(c(0, arima.sim(list(ma = -0.5), n = 199)))
trend_stat <- 0.05 * (1:150) + as.numeric(arima.sim(list(ar = 0.5), n = 150))
air <- as.numeric(AirPassengers)
lair <- log(air)

series_list <- list(
  rw = rw,
  ar07 = ar07,
  ima = ima,
  trend_stat = trend_stat,
  air = air,
  lair = lair
)

# Extra AR(phi) series for KPSS region coverage (drawn after the series
# above, so RNG stream order is: rw, ar07, ima, trend_stat, then this block,
# in exactly this order — the trailing 0.62/0.7/0.93 entries are empirically
# chosen, at this exact RNG stream position, to land each of the five tseries
# p-value regions (verified by the stopifnot() below); reordering or
# inserting values here changes every later draw).
phi_grid <- c(0, 0.5, 0.8, 0.9, 0.95, 0.98, 1, 0.62, 0.7, 0.93)
ar_phi_series <- list()
for (idx in seq_along(phi_grid)) {
  phi <- phi_grid[idx]
  key <- paste0("ar_phi_", idx, "_", gsub("\\.", "", sprintf("%.2f", phi)))
  if (phi >= 1) {
    ar_phi_series[[key]] <- cumsum(rnorm(200))
  } else {
    ar_phi_series[[key]] <- as.numeric(arima.sim(list(ar = phi), n = 200))
  }
}

kpss_series <- c(series_list, ar_phi_series)

# ---- ur.df fixed-lag block --------------------------------------------------

urdf_types <- c(none = "none", drift = "drift", trend = "trend")
urdf_lags <- c(0, 1, 4)

urdf_block <- function(y) {
  out <- list()
  for (type_name in names(urdf_types)) {
    type_val <- urdf_types[[type_name]]
    lag_out <- list()
    for (lag in urdf_lags) {
      u <- ur.df(y, type = type_val, lags = lag)
      teststat_vec <- as.numeric(u@teststat[1, ])
      lag_out[[paste0("lag", lag)]] <- list(teststat = teststat_vec)
    }
    out[[type_name]] <- lag_out
  }
  out
}

adf_fixed_lag <- lapply(series_list, urdf_block)

# ---- KPSS block --------------------------------------------------------------

kpss_block <- function(y) {
  k <- suppressWarnings(tseries::kpss.test(y, null = "Level", lshort = TRUE))
  list(
    values = as.numeric(y),
    statistic = as.numeric(k$statistic),
    p_value = as.numeric(k$p.value),
    lag = as.numeric(k$parameter)
  )
}

kpss_results <- lapply(kpss_series, kpss_block)

kpss_stats <- sapply(kpss_results, function(x) x$statistic)
stopifnot(
  "KPSS fixture must cover stat < 0.347" = any(kpss_stats < 0.347),
  "KPSS fixture must cover 0.347 <= stat < 0.463" = any(kpss_stats >= 0.347 & kpss_stats < 0.463),
  "KPSS fixture must cover 0.463 <= stat < 0.574" = any(kpss_stats >= 0.463 & kpss_stats < 0.574),
  "KPSS fixture must cover 0.574 <= stat < 0.739" = any(kpss_stats >= 0.574 & kpss_stats < 0.739),
  "KPSS fixture must cover stat >= 0.739" = any(kpss_stats >= 0.739)
)

# ---- Monte-Carlo block -------------------------------------------------------

n_mc <- 40
n_obs <- 200

mc_rw <- vector("list", n_mc)
mc_rw_series <- vector("list", n_mc)
for (i in seq_len(n_mc)) {
  y <- cumsum(rnorm(n_obs))
  mc_rw_series[[i]] <- as.numeric(y)
  u <- ur.df(y, type = "drift", lags = 5, selectlags = "AIC")
  stat <- as.numeric(u@teststat[1, 1])
  cv5 <- as.numeric(u@cval["tau2", "5pct"])
  mc_rw[[i]] <- list(statistic = stat, reject_5pct = stat < cv5)
}

mc_ima <- vector("list", n_mc)
mc_ima_series <- vector("list", n_mc)
for (i in seq_len(n_mc)) {
  y <- cumsum(c(0, arima.sim(list(ma = -0.5), n = n_obs - 1)))
  mc_ima_series[[i]] <- as.numeric(y)
  u <- ur.df(y, type = "drift", lags = 5, selectlags = "AIC")
  stat <- as.numeric(u@teststat[1, 1])
  cv5 <- as.numeric(u@cval["tau2", "5pct"])
  mc_ima[[i]] <- list(statistic = stat, reject_5pct = stat < cv5)
}

# ---- assemble and write ------------------------------------------------------

fixture <- list(
  provenance = provenance(),
  series = lapply(series_list, as.numeric),
  adf_fixed_lag = adf_fixed_lag,
  kpss = kpss_results,
  monte_carlo = list(
    random_walk = mc_rw,
    ima_theta_neg_0_5 = mc_ima,
    random_walk_series = mc_rw_series,
    ima_theta_neg_0_5_series = mc_ima_series
  )
)

write_fixture(fixture, "adf_kpss_r.json")
cat("Wrote adf_kpss_r.json\n")
