# R reference fixture generator for exact-likelihood AutoARIMA candidate
# scoring parity (branch fix/auto-arima-order-selection, phase 11-11).
#
# Unlike auto_arima.R (11-04, selection-frequency summary only), this
# generator records R's exact per-candidate forecast::Arima() loglik/AICc
# (CSS-ML) and, for series R scores under approximation=TRUE, the CSS
# criterion too -- the ground truth 11-11's exact_likelihood module is
# checked against candidate-by-candidate (D-06).
#
# This branch has no shared validation/reference/r/common.R of its own
# (each Phase 11 fix branch is cut independently from the same BASE_REF
# and must stay independently mergeable -- see PLAN D-01); self-contained,
# mirroring auto_arima.R's conventions.
#
# Regenerate with:
#   Rscript validation/reference/r/auto_arima_candidates.R
#
# Requires: R packages forecast, jsonlite.

suppressMessages({
  library(forecast)
  library(jsonlite)
})

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
  file.path(ref_script_dir(), "..", "..", "..", "tests", "data", "r_reference", "auto_arima_candidates_r.json")
}

m4_outliers_path <- function() {
  file.path(ref_script_dir(), "..", "..", "..", "tests", "data", "m4_outliers.json")
}

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(20261009)

# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------
air_raw <- as.numeric(AirPassengers)
air <- ts(air_raw, frequency = 12)

m4 <- fromJSON(m4_outliers_path(), simplifyVector = FALSE)
d4047_train <- as.numeric(unlist(m4[["D4047"]][["train"]]))
d4047 <- ts(d4047_train, frequency = 7)

n_each <- 50
series_n <- 200
rw_list <- vector("list", n_each)
for (i in seq_len(n_each)) {
  rw_list[[i]] <- cumsum(rnorm(series_n))
}
rw5 <- rw_list[1:5]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
approx_flag <- function(n, m) (n > 150) || (m > 12)

safe_fit <- function(y, order, seasonal = NULL, include_drift = FALSE, method) {
  tryCatch(
    {
      if (is.null(seasonal)) {
        Arima(y, order = order, include.drift = include_drift, include.mean = FALSE, method = method)
      } else {
        Arima(y, order = order, seasonal = seasonal, include.drift = include_drift, include.mean = FALSE, method = method)
      }
    },
    error = function(e) NULL,
    warning = function(w) NULL
  )
}

fit_entry <- function(y, order, seasonal, include_drift, m) {
  cml <- safe_fit(y, order, seasonal, include_drift, method = "CSS-ML")
  entry <- list(
    order = list(p = order[1], d = order[2], q = order[3]),
    seasonal = if (is.null(seasonal)) list(P = 0, D = 0, Q = 0, period = 1) else
      list(P = seasonal$order[1], D = seasonal$order[2], Q = seasonal$order[3], period = seasonal$period),
    drift = include_drift
  )
  if (is.null(cml)) {
    entry$css_ml <- NULL
  } else {
    coefs <- coef(cml)
    entry$css_ml <- list(
      loglik = as.numeric(cml$loglik),
      aicc = as.numeric(cml$aicc),
      aic = as.numeric(cml$aic),
      bic = as.numeric(cml$bic),
      sigma2 = as.numeric(cml$sigma2),
      nobs = as.numeric(cml$nobs),
      coef_names = names(coefs),
      coef_values = as.numeric(coefs)
    )
  }

  if (approx_flag(length(y), m)) {
    css <- safe_fit(y, order, seasonal, include_drift, method = "CSS")
    if (is.null(css)) {
      entry$css <- NULL
    } else {
      npar <- length(coef(css)) + 1
      nstar <- as.numeric(css$nobs)
      sigma2 <- as.numeric(css$sigma2)
      ic <- nstar * log(sigma2) + 2 * npar + (2 * npar * (npar + 1)) / (nstar - npar - 1)
      entry$css <- list(
        sigma2 = sigma2,
        nobs = nstar,
        npar = npar,
        criterion_offset_free = as.numeric(ic),
        coef_names = names(coef(css)),
        coef_values = as.numeric(coef(css))
      )
    }
  } else {
    entry$css <- NULL
  }
  entry
}

trace_entry <- function(line) {
  # Parse one line of auto.arima(trace=TRUE) capture.output, e.g.:
  # " ARIMA(2,1,2)(1,1,1)[12]                    : 1023.456"
  # " ARIMA(0,1,0)                               : Inf"
  m <- regmatches <- regexec(
    "ARIMA\\(([0-9]+),([0-9]+),([0-9]+)\\)(\\(([0-9]+),([0-9]+),([0-9]+)\\)\\[([0-9]+)\\])?( with non-zero mean| with drift| with zero mean)?\\s*:\\s*(-?[0-9.]+|Inf)",
    line
  )
  g <- regmatches(line, m)[[1]]
  if (length(g) < 11 || g[1] == "") {
    return(NULL)
  }
  p <- as.integer(g[2])
  d <- as.integer(g[3])
  q <- as.integer(g[4])
  has_seasonal <- nchar(g[5]) > 0
  cap_p <- if (has_seasonal) as.integer(g[6]) else 0L
  cap_d <- if (has_seasonal) as.integer(g[7]) else 0L
  cap_q <- if (has_seasonal) as.integer(g[8]) else 0L
  period <- if (has_seasonal) as.integer(g[9]) else 1L
  const_label <- trimws(g[10])
  ic_str <- g[11]
  ic <- if (ic_str == "Inf") NA else as.numeric(ic_str)
  list(
    p = p, d = d, q = q, cap_p = cap_p, cap_d = cap_d, cap_q = cap_q, period = period,
    constant = if (const_label == "") "none" else const_label,
    ic = ic
  )
}

parse_trace <- function(y) {
  lines <- capture.output(invisible(auto.arima(y, trace = TRUE)))
  entries <- lapply(lines, trace_entry)
  entries <- entries[!sapply(entries, is.null)]
  entries
}

# ---------------------------------------------------------------------------
# AirPassengers: trace + pick + candidates (exact, no approximation: n=144<=150, m=12<=12)
# ---------------------------------------------------------------------------
air_trace <- parse_trace(air)
air_fit <- auto.arima(air)
air_ord <- arimaorder(air_fit)
air_pick <- list(
  p = unname(air_ord["p"]), d = unname(air_ord["d"]), q = unname(air_ord["q"]),
  cap_p = unname(air_ord["P"]), cap_d = unname(air_ord["D"]), cap_q = unname(air_ord["Q"]),
  m = unname(air_ord["Frequency"]),
  aicc = as.numeric(air_fit$aicc),
  loglik = as.numeric(air_fit$loglik)
)

air_candidates_spec <- list(
  list(order = c(2, 1, 1), seasonal = list(order = c(0, 1, 0), period = 12)),
  list(order = c(0, 1, 1), seasonal = list(order = c(0, 1, 1), period = 12)),
  list(order = c(3, 1, 0), seasonal = list(order = c(2, 1, 0), period = 12)),
  list(order = c(0, 1, 3), seasonal = list(order = c(0, 1, 0), period = 12)),
  list(order = c(1, 1, 0), seasonal = list(order = c(0, 1, 0), period = 12)),
  list(order = c(0, 1, 0), seasonal = list(order = c(0, 1, 0), period = 12)),
  list(order = c(1, 1, 1), seasonal = list(order = c(0, 1, 0), period = 12)),
  list(order = c(2, 1, 1), seasonal = list(order = c(1, 1, 0), period = 12)),
  list(order = c(2, 1, 2), seasonal = list(order = c(1, 1, 1), period = 12)),
  list(order = c(0, 1, 1), seasonal = list(order = c(0, 1, 0), period = 12))
)

air_candidates <- lapply(air_candidates_spec, function(spec) {
  fit_entry(air, spec$order, spec$seasonal, include_drift = FALSE, m = 12)
})

# ---------------------------------------------------------------------------
# D4047 (period 7): trace + pick + candidates (approximation: n=162>150)
# ---------------------------------------------------------------------------
d4047_trace <- parse_trace(d4047)
d4047_fit <- auto.arima(d4047)
d4047_ord <- arimaorder(d4047_fit)
d4047_pick <- list(
  p = unname(d4047_ord["p"]), d = unname(d4047_ord["d"]), q = unname(d4047_ord["q"]),
  cap_p = unname(d4047_ord["P"]), cap_d = unname(d4047_ord["D"]), cap_q = unname(d4047_ord["Q"]),
  m = unname(d4047_ord["Frequency"]),
  aicc = as.numeric(d4047_fit$aicc),
  loglik = as.numeric(d4047_fit$loglik)
)

d4047_candidates_spec <- list(
  list(order = c(0, 1, 0), seasonal = NULL),
  list(order = c(1, 1, 0), seasonal = NULL),
  list(order = c(0, 1, 1), seasonal = NULL),
  list(order = c(1, 1, 1), seasonal = NULL),
  list(order = c(2, 1, 2), seasonal = NULL),
  list(order = c(0, 1, 0), seasonal = list(order = c(1, 0, 0), period = 7)),
  list(order = c(0, 1, 0), seasonal = list(order = c(0, 0, 1), period = 7)),
  list(order = c(1, 1, 1), seasonal = list(order = c(1, 0, 1), period = 7)),
  list(order = c(2, 1, 2), seasonal = list(order = c(1, 0, 1), period = 7))
)

d4047_candidates <- list()
for (spec in d4047_candidates_spec) {
  d_plus_cap_d <- spec$order[2] + (if (is.null(spec$seasonal)) 0 else spec$seasonal$order[2])
  drift_variants <- if (d_plus_cap_d == 1) c(FALSE, TRUE) else c(FALSE)
  for (dr in drift_variants) {
    d4047_candidates[[length(d4047_candidates) + 1]] <- fit_entry(d4047, spec$order, spec$seasonal, include_drift = dr, m = 7)
  }
}

# ---------------------------------------------------------------------------
# rw series (first five): candidates (approximation: n=200>150)
# ---------------------------------------------------------------------------
rw_candidates_spec <- list(
  c(0, 1, 0),
  c(1, 1, 0),
  c(0, 1, 1),
  c(1, 1, 1),
  c(2, 1, 2)
)

rw_entries <- lapply(seq_along(rw5), function(i) {
  y <- ts(rw5[[i]], frequency = 1)
  cands <- list()
  for (ord in rw_candidates_spec) {
    for (dr in c(FALSE, TRUE)) {
      cands[[length(cands) + 1]] <- fit_entry(y, ord, NULL, include_drift = dr, m = 1)
    }
  }
  list(values = rw5[[i]], candidates = cands)
})

# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------
fixture <- list(
  provenance = list(
    family = "auto_arima_candidates",
    generator = "validation/reference/r/auto_arima_candidates.R",
    regenerate_command = "Rscript validation/reference/r/auto_arima_candidates.R",
    tool_versions = list(
      R = R.version.string,
      forecast = as.character(packageVersion("forecast")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    rng_kind = paste(RNGkind(), collapse = "/"),
    seed = 20261009
  ),
  air_passengers = list(
    n = length(air_raw), m = 12, approximation = approx_flag(length(air_raw), 12),
    values = air_raw, trace = air_trace, pick = air_pick, candidates = air_candidates
  ),
  d4047 = list(
    n = length(d4047_train), m = 7, approximation = approx_flag(length(d4047_train), 7),
    values = d4047_train, trace = d4047_trace, pick = d4047_pick, candidates = d4047_candidates
  ),
  rw = rw_entries
)

out_path <- fixture_path()
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, out_path, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")

cat("Wrote fixture to", out_path, "\n")
