# Theta (STM) reference fixture generator (UPST-04).
#
# Writes tests/data/r_reference/theta_thetaf.json — forecast::thetaf-equivalent
# fits (forecast:::theta_model + forecast:::forecast.theta_model) for four
# series: a non-seasonal random walk, a trending series, an AR(0.7) series,
# and the seasonal AirPassengers series (multiplicative decomposition).
#
# Regenerate with:
#   Rscript validation/reference/r/theta.R
# (works from any cwd -- the script resolves its own directory via --file=).
#
# No shared common.R in this repo (per Phase 11 Plan 02/03 conventions) --
# all helpers are inlined below.

suppressMessages({
  library(forecast)
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

GENERATOR_PATH <- "validation/reference/r/theta.R"
REGENERATE_CMD <- "Rscript validation/reference/r/theta.R"
SEED <- 20261009

provenance <- function() {
  list(
    generator = GENERATOR_PATH,
    regenerate = REGENERATE_CMD,
    tool_versions = list(
      R = R.version.string,
      forecast = as.character(packageVersion("forecast")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    rng_kind = paste(RNGkind(), collapse = "/"),
    seed = SEED
  )
}

write_fixture <- function(obj, rel_path) {
  full_path <- file.path(.fixtures_dir, rel_path)
  dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(obj, full_path, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")
  invisible(full_path)
}

# ---- deterministic series battery ------------------------------------------
#
# Fixed order after the seed: rw, trend, ar, air (air itself is not RNG-drawn,
# so its position in the RNG stream does not matter, but it is still listed
# last to keep the three drawn series' RNG consumption order documented here).

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(SEED)

rw <- cumsum(rnorm(200))
trend <- 10 + 0.5 * (1:120) + rnorm(120, sd = 3)
ar <- 50 + as.numeric(arima.sim(list(ar = 0.7), n = 300))
air <- AirPassengers

series_defs <- list(
  rw = list(y = rw, frequency = 1),
  trend = list(y = trend, frequency = 1),
  ar = list(y = ar, frequency = 1),
  air = list(y = air, frequency = frequency(air))
)

# ---- fit theta_model + forecast.theta_model for each series ----------------

H <- 24

theta_block <- function(name, def) {
  y <- def$y
  n <- length(y)
  fit <- forecast:::theta_model(y)
  fc <- forecast(fit, h = H, level = c(80, 95))

  ses_model <- fit$ses_model
  ses_res <- residuals(ses_model)
  sse <- sum(as.numeric(ses_res)^2)

  states <- as.numeric(ses_model$states[, "l"])
  l0 <- as.numeric(ses_model$par["l"])
  final_level <- states[length(states)]
  alpha <- as.numeric(fit$alpha)
  b_coef <- as.numeric(lsfit(0:(n - 1), if (!is.null(fit$seas_component)) {
    # deseasonalized series used internally by theta_model for the slope;
    # reconstruct it the same way theta_model does (seasadj via decompose).
    seasadj(decompose(y, type = "multiplicative"))
  } else {
    y
  })$coefficients[2])
  drift <- as.numeric(fit$drift)

  seas_component <- if (!is.null(fit$seas_component)) as.numeric(fit$seas_component) else NULL

  list(
    name = name,
    n = n,
    frequency = def$frequency,
    seasonal = !is.null(fit$seas_component),
    alpha = alpha,
    l0 = l0,
    final_level = final_level,
    b = b_coef,
    drift = drift,
    sigma2 = as.numeric(fit$sigma2),
    sse = sse,
    seas_component = seas_component,
    point = as.numeric(fc$mean),
    lower_80 = as.numeric(fc$lower[, 1]),
    upper_80 = as.numeric(fc$upper[, 1]),
    lower_95 = as.numeric(fc$lower[, 2]),
    upper_95 = as.numeric(fc$upper[, 2]),
    fitted = as.numeric(fit$fitted),
    values = as.numeric(y)
  )
}

blocks <- lapply(names(series_defs), function(nm) theta_block(nm, series_defs[[nm]]))
names(blocks) <- names(series_defs)

fixture <- list(
  provenance = provenance(),
  horizon = H,
  series = blocks
)

write_fixture(fixture, "theta_thetaf.json")
cat("Wrote theta_thetaf.json\n")
