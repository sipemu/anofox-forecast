# R reference fixture generator for ARIMA / SARIMA prediction intervals.
#
# R computes ARIMA prediction variances with `KalmanForecast` on the
# state-space form built by `makeARIMA` (differencing folded into the
# state via `Delta`), starting from the filtered state left behind by the
# final `ARIMA_Like` pass; `se = sqrt(var * sigma2)` with
# `sigma2 = ssq / n.used` from that same pass. This script records, for
# a handful of fixed-order models, R's coefficients, sigma2, point
# forecasts, standard errors and 95% intervals (h = 1..24) so the crate
# can be checked at R's own coefficients.
#
# Regenerate with:
#   Rscript validation/reference/r/arima_intervals.R
#
# Requires: R packages forecast, jsonlite.

suppressMessages({
  library(forecast)
  library(jsonlite)
})

ref_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) stop("run via Rscript")
  normalizePath(dirname(sub("^--file=", "", file_arg[1])), mustWork = TRUE)
}
out_path <- file.path(ref_script_dir(), "..", "..", "..", "tests", "data",
                      "r_reference", "arima_intervals_r.json")

H <- 24
take <- function(coef, prefix) {
  v <- unname(coef[grep(paste0("^", prefix, "[0-9]+$"), names(coef))])
  if (length(v) == 0) numeric(0) else v
}

case <- function(name, y, order, seasonal = c(0, 0, 0), drift = FALSE) {
  fit <- Arima(y, order = order, seasonal = seasonal, include.drift = drift,
               method = "CSS-ML")
  fc <- forecast(fit, h = H, level = 95)
  # predict.Arima cannot re-evaluate forecast::Arima's drift xreg, so the
  # drift case recovers se from forecast()'s interval (exact up to rounding).
  se <- if (drift) {
    as.numeric((fc$upper[, 1] - fc$mean) / qnorm(0.975))
  } else {
    as.numeric(predict(fit, n.ahead = H)$se)
  }
  co <- coef(fit)
  list(
    name = name,
    values = as.numeric(y),
    period = frequency(y),
    order = as.integer(order),
    seasonal = as.integer(seasonal),
    drift = drift,
    ar = take(co, "ar"),
    ma = take(co, "ma"),
    sar = take(co, "sar"),
    sma = take(co, "sma"),
    drift_coef = if (drift) unname(co["drift"]) else 0,
    sigma2 = fit$sigma2,
    loglik = fit$loglik,
    mean = as.numeric(fc$mean),
    se = se,
    lower95 = as.numeric(fc$lower[, 1]),
    upper95 = as.numeric(fc$upper[, 1])
  )
}

ap <- AirPassengers
auto <- auto.arima(ap)
cases <- list(
  case("air_211_010", ap, c(2, 1, 1), c(0, 1, 0)),
  case("log_air_airline", log(ap), c(0, 1, 1), c(0, 1, 1)),
  case("log_air_110_110", log(ap), c(1, 1, 0), c(1, 1, 0)),
  case("wwwusage_111", WWWusage, c(1, 1, 1)),
  case("wwwusage_111_drift", WWWusage, c(1, 1, 1), drift = TRUE)
)

out <- list(
  provenance = list(
    family = "arima_intervals",
    generator = "validation/reference/r/arima_intervals.R",
    regenerate_command = "Rscript validation/reference/r/arima_intervals.R",
    tool_versions = list(
      R = R.version.string,
      forecast = as.character(packageVersion("forecast")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    level = 95,
    horizon = H
  ),
  auto_arima_air_passengers = list(
    arma = as.integer(auto$arma),
    mean = as.numeric(forecast(auto, h = H, level = 95)$mean),
    se = as.numeric(predict(auto, n.ahead = H)$se),
    sigma2 = auto$sigma2
  ),
  cases = cases
)

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), out_path)
cat("wrote", normalizePath(out_path), "\n")
