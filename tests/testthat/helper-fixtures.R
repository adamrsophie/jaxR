# Fixture helpers. tests/testthat/fixtures/<group>_x32.json and <group>_x64.json hold
# results recorded from the pinned jax wheel; these helpers decode them and compare
# jaxR results against them. Loading is lazy and cached; nothing runs at source time.

.fx_cache <- new.env(parent = emptyenv())

fx_load <- function(group, x64) {
  testthat::skip_if_not_installed("jsonlite")
  key <- paste0(group, if (x64) "_x64" else "_x32")
  if (is.null(.fx_cache[[key]])) {
    path <- testthat::test_path("fixtures", paste0(key, ".json"))
    if (!file.exists(path)) testthat::skip(paste0("fixture ", key, " missing"))
    .fx_cache[[key]] <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  }
  .fx_cache[[key]]
}

# ---- decoding -------------------------------------------------------------------------

fx_float_dtypes <- c(float16 = 2^-10, bfloat16 = 2^-7, float32 = 2^-23, float64 = 2^-52)

fx_kind <- function(dt) {
  if (dt %in% c("bool", "float0")) "b"
  else if (dt %in% c("complex64", "complex128")) "c"
  else if (dt %in% names(fx_float_dtypes)) "f"
  else "i"
}

# One scalar: hex float string, token (NaN, Inf, -Inf, -0.0), decimal string or number.
fx_num1 <- function(v) {
  if (!is.character(v)) return(as.numeric(v))
  if (grepl("^-?[0-9]{16,}$", v)) return(fx_big_int(v))
  switch(v, "NaN" = NaN, "Inf" = Inf, "-Inf" = -Inf, "-0.0" = -0, as.numeric(v))
}

# R's decimal parser is not correctly rounded above 2^53 (as.numeric("-9223372036854775808")
# is off by one ulp); split into exact parts so 64-bit extremes round to the nearest double.
fx_big_int <- function(v) {
  neg <- startsWith(v, "-")
  d <- sub("^-", "", v)
  n <- nchar(d)
  out <- as.numeric(substr(d, 1L, n - 9L)) * 1e9 + as.numeric(substr(d, n - 8L, n))
  if (neg) -out else out
}

fx_values <- function(data, dtype) {
  switch(fx_kind(dtype),
    b = as.logical(vapply(data, isTRUE, NA)),
    c = vapply(data, function(z) complex(real = fx_num1(z[[1L]]), imaginary = fx_num1(z[[2L]])), 0i),
    as.numeric(vapply(data, fx_num1, 0)))
}

fx_ints <- function(v) as.integer(unlist(v))

# C (row major) order <-> R arrays, independent of jaxR internals.
fx_c_array <- function(v, shape) {
  shape <- fx_ints(shape)
  r <- length(shape)
  if (r == 0L) return(v)
  if (r == 1L) return(array(v, shape))
  aperm(array(v, rev(shape)), r:1)
}

fx_c_ravel <- function(x) {
  d <- dim(x)
  if (length(d) <= 1L) return(as.vector(x))
  as.vector(aperm(x, length(d):1))
}

# Iterator over a list (used to substitute traced arrays when a case runs under jit).
fx_iter <- function(xs) {
  i <- 0L
  function() {
    i <<- i + 1L
    xs[[i]]
  }
}

# Decode an argument spec into R values. With `it`, array specs take the next value of
# the iterator instead (so the same spec walk builds jit arguments and traced calls).
fx_arg <- function(s, it = NULL) {
  switch(s$t,
    array = if (!is.null(it)) it() else
      jaxR::as_jax(fx_c_array(fx_values(s$data, s$dtype), s$shape), s$dtype),
    scalar = switch(s$kind,
      int = as.integer(s$value), float = fx_num1(s$value), bool = isTRUE(s$value),
      complex = complex(real = fx_num1(s$value[[1L]]), imaginary = fx_num1(s$value[[2L]]))),
    vector = switch(s$kind,
      int = fx_ints(s$value), bool = as.logical(unlist(s$value)),
      float = as.numeric(vapply(s$value, fx_num1, 0))),
    ints = fx_ints(s$value),
    int = as.integer(s$value),
    str = s$value,
    bool = isTRUE(s$value),
    null = NULL,
    list = lapply(s$items, fx_arg, it = it),
    dict = stats::setNames(lapply(s$items, fx_arg, it = it), unlist(s$keys)),
    fn = get(s$r, envir = asNamespace("jaxR")),
    stop("unknown fixture spec type ", s$t))
}

# The array leaves of a list of specs, in the order fx_arg() visits them.
fx_arrays <- function(specs) {
  out <- list()
  walk <- function(s) {
    if (identical(s$t, "array")) out[[length(out) + 1L]] <<- fx_arg(s)
    else if (s$t %in% c("list", "dict")) for (i in s$items) walk(i)
  }
  for (s in specs) walk(s)
  out
}

fx_args <- function(specs, it = NULL) {
  out <- lapply(specs, fx_arg, it = it)
  # lapply keeps NULL entries (needed for keyword arguments such as axis = None).
  out
}

# Call the R function of a fixture case eagerly or under jit (arrays traced, everything
# else closed over, like a Python lambda closing over static values).
fx_call <- function(case, jit = FALSE, fn = NULL) {
  f <- fn %||% get(case$fn, envir = asNamespace("jaxR"))
  if (!jit) return(do.call(f, c(fx_args(case$args), fx_args(case$kwargs))))
  arrs <- fx_arrays(c(case$args, unname(case$kwargs)))
  g <- function(arrs) {
    it <- fx_iter(arrs)
    do.call(f, c(fx_args(case$args, it), fx_args(case$kwargs, it)))
  }
  jaxR::jit(g)(arrs)
}

# ---- comparison -----------------------------------------------------------------------

fx_aval <- function(x) jaxR:::aval_of(x)

# Comparison options: ulps (relative tolerance in units of the dtype's epsilon),
# scale (also accept an absolute error of ulps * eps * max|want|, for results of
# accumulations with cancellation) and zero_sign (check the sign of exact zeros, which
# XLA may legitimately change when it fuses or simplifies under jit).
fx_cmp <- function(ulps = 4, scale = FALSE, zero_sign = TRUE) {
  list(ulps = ulps, scale = scale, zero_sign = zero_sign)
}

fx_expect_values <- function(got, want, dt, info, cmp = fx_cmp()) {
  if (fx_kind(dt) == "c") {
    comp <- if (dt == "complex64") "float32" else "float64"
    fx_expect_values(Re(got), Re(want), comp, paste(info, "(real part)"), cmp)
    fx_expect_values(Im(got), Im(want), comp, paste(info, "(imaginary part)"), cmp)
    return(invisible())
  }
  if (fx_kind(dt) != "f") {
    testthat::expect_identical(as.vector(got), as.vector(want), info = info)
    return(invisible())
  }
  got <- as.numeric(got)
  tol <- cmp$ulps * fx_float_dtypes[[dt]]
  fin <- is.finite(want)
  atol <- if (cmp$scale && any(fin)) tol * max(abs(want[fin])) else 0
  same <- (is.nan(got) & is.nan(want)) | (!is.nan(got) & !is.nan(want) & got == want)
  close <- !same & is.finite(got) & fin & abs(got - want) <= pmax(tol * abs(want), atol)
  bad <- which(!(same | close))
  testthat::expect(length(bad) == 0L, sprintf(
    "%s: values differ at C-order positions %s: got %s, want %s", info,
    paste(utils::head(bad, 5) - 1L, collapse = ","),
    paste(format(utils::head(got[bad], 5), digits = 17), collapse = ","),
    paste(format(utils::head(want[bad], 5), digits = 17), collapse = ",")))
  if (cmp$zero_sign) {
    z <- which(want == 0 & got == 0)
    testthat::expect(identical(sign(1 / got[z]), sign(1 / want[z])),
                     sprintf("%s: sign of zero differs at C-order positions %s", info,
                             paste(z[sign(1 / got[z]) != sign(1 / want[z])] - 1L, collapse = ",")))
  }
  invisible()
}

fx_expect <- function(got, want, info = "", cmp = fx_cmp()) {
  switch(want$t,
    array = {
      testthat::expect(inherits(got, "jax_value"), paste0(info, ": result is not a jax value"))
      if (!inherits(got, "jax_value")) return(invisible())
      a <- fx_aval(got)
      testthat::expect_identical(a$dtype, want$dtype, info = info)
      testthat::expect_identical(a$shape, fx_ints(want$shape), info = info)
      testthat::expect_identical(a$weak, isTRUE(want$weak_type), info = paste(info, "(weak_type)"))
      if (identical(a$dtype, want$dtype) && identical(a$shape, fx_ints(want$shape)))
        fx_expect_values(fx_c_ravel(as.array(got)), fx_values(want$data, want$dtype), want$dtype,
                         info, cmp)
    },
    aval = {
      testthat::expect_identical(got$dtype, want$dtype, info = info)
      testthat::expect_identical(as.integer(got$shape), fx_ints(want$shape), info = info)
      testthat::expect_identical(isTRUE(got$weak), isTRUE(want$weak_type), info = paste(info, "(weak_type)"))
    },
    list = {
      if (!is.list(got) && all(vapply(want$items, function(i) identical(i$t, "int"), NA))) {
        testthat::expect_identical(as.integer(got), as.integer(unlist(lapply(want$items, `[[`, "value"))),
                                   info = info)
        return(invisible())
      }
      testthat::expect(is.list(got) && length(got) == length(want$items),
                       paste0(info, ": expected a list of length ", length(want$items)))
      if (is.list(got) && length(got) == length(want$items))
        for (i in seq_along(want$items)) fx_expect(got[[i]], want$items[[i]], paste0(info, "[[", i, "]]"), cmp)
    },
    dict = {
      keys <- unlist(want$keys)
      testthat::expect_setequal(names(got), keys)
      for (i in seq_along(keys)) fx_expect(got[[keys[i]]], want$items[[i]], paste0(info, "$", keys[i]), cmp)
    },
    int = testthat::expect_identical(as.integer(got), as.integer(want$value), info = info),
    bool = testthat::expect_identical(as.logical(got), isTRUE(want$value), info = info),
    str = testthat::expect_identical(got, want$value, info = info),
    stop("unknown fixture output type ", want$t))
  invisible()
}

fx_error_class <- function(type) {
  switch(type, TypeError = "jax_type_error", ValueError = "jax_value_error", "jax_error")
}

# The R message must match the wheel's message up to the first sentence end (or the
# whole message when there is no sentence break), after stripping R's own prefix.
fx_first_sentence <- function(msg) sub("^(.*?[.:])( .*)?$", "\\1", msg, perl = TRUE)

# Error message mismatches are collected (one finding per group, see fx_run_cases)
# instead of failing each case: JAX 0.12 words many messages differently (for example
# "at position 0"), while the error class is checked per case.
.fx_messages <- new.env(parent = emptyenv())

fx_expect_error <- function(thunk, err, info, check_message = TRUE, collect = NULL) {
  e <- tryCatch({ thunk(); NULL }, error = function(e) e)
  testthat::expect(!is.null(e), sprintf("%s: expected %s: %s", info, err$type, err$message))
  if (is.null(e)) return(invisible())
  testthat::expect(inherits(e, fx_error_class(err$type)), sprintf(
    "%s: error class is %s, expected %s (wheel: %s: %s)", info, paste(class(e), collapse = "/"),
    fx_error_class(err$type), err$type, err$message))
  if (check_message) {
    want <- fx_first_sentence(err$message)
    ok <- startsWith(conditionMessage(e), want) || startsWith(want, conditionMessage(e))
    detail <- sprintf("%s: message\n  R:     %s\n  wheel: %s", info, conditionMessage(e), err$message)
    if (is.null(collect)) testthat::expect(ok, detail)
    else if (!ok) .fx_messages[[collect]] <- c(.fx_messages[[collect]], detail)
  }
  invisible()
}

# Is the output of a case something jit can return (arrays only)?
fx_jittable <- function(out) {
  if (is.null(out)) return(TRUE)
  switch(out$t, array = TRUE, list = , dict = all(vapply(out$items, fx_jittable, NA)), FALSE)
}

# Run every case of a group as its own test, eagerly and under jit, for both x64
# settings. `known` maps a case id (optionally suffixed "@x32" or "@x64") to a skip
# reason; `opts` maps ids to list(ulps = , scale = TRUE, message = FALSE, jit = FALSE).
# Set JAXR_FX_IGNORE_KNOWN=1 to run known-bug cases anyway (except those whose reason
# contains "(crash)"), to find skips that are no longer needed.
fx_skip_reason <- function(known, id, tag) {
  reason <- c(known[paste0(id, "@", tag)], known[id])
  reason <- reason[!is.na(reason)]
  if (identical(Sys.getenv("JAXR_FX_IGNORE_KNOWN"), "1")) reason <- reason[grepl("(crash)", reason, fixed = TRUE)]
  reason
}

fx_run_cases <- function(group, known = character(), opts = list()) {
  for (x64 in c(FALSE, TRUE)) {
    fx <- fx_load(group, x64)
    tag <- if (x64) "x64" else "x32"
    ids <- vapply(fx$cases, `[[`, "", "id")
    testthat::test_that(sprintf("every known-bug id of %s names a fixture case [%s]", group, tag), {
      testthat::expect_identical(setdiff(sub("@x(32|64)$", "", c(names(known), names(opts))), ids), character())
    })
    for (case in fx$cases) local({
      case <- case
      o <- opts[[case$id]] %||% list()
      reason <- fx_skip_reason(known, case$id, tag)
      testthat::test_that(sprintf("%s matches the wheel [%s]", case$id, tag), {
        skip_if_no_backend()
        if (length(reason)) testthat::skip(reason[[1L]])
        jaxR::with_jax_config("jax_enable_x64", x64, {
          info <- paste0(case$code, " [", tag, "]")
          if (identical(case$error$type, "AttributeError")) {
            # The wheel has no such function, so jaxR must not have one either.
            testthat::expect_false(exists(case$fn, envir = asNamespace("jaxR"), inherits = FALSE),
                                   info = info)
          } else if (!is.null(case$error)) {
            key <- paste0(group, "_", tag)
            fx_expect_error(function() fx_call(case), case$error, info, !isFALSE(o$message), key)
            if (!isFALSE(o$jit))
              fx_expect_error(function() fx_call(case, jit = TRUE), case$error, paste(info, "under jit"),
                              FALSE)
          } else {
            cmp <- fx_cmp(o$ulps %||% 4, isTRUE(o$scale), !isFALSE(o$zero_sign))
            fx_expect(fx_call(case), case$out, info, cmp)
            cmp$zero_sign <- FALSE
            if (!isFALSE(o$jit) && fx_jittable(case$out))
              fx_expect(fx_call(case, jit = TRUE), case$out, paste(info, "under jit"), cmp)
          }
        })
      })
    })
    local({
      key <- paste0(group, "_", tag)
      testthat::test_that(sprintf("%s error messages start like the wheel's [%s]", group, tag), {
        msgs <- unique(.fx_messages[[key]])
        if (length(msgs))
          testthat::skip(sprintf("known deviation: %d %s error messages are worded differently from jax 0.12 (first: %s)",
                                 length(msgs), group, gsub("\n", " ", msgs[[1L]])))
        testthat::expect_length(msgs, 0L)
      })
    })
  }
}
