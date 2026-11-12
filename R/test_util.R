# Port of jax.test_util (jax/_src/public_test_util.py): checks of automatic
# differentiation against central finite differences in one random direction.

tu_eps <- 1e-4

tu_default_tolerance <- c(
  float0 = 0, bool = 0, int1 = 0, int2 = 0, int4 = 0, int8 = 0, int16 = 0, int32 = 0, int64 = 0,
  uint1 = 0, uint2 = 0, uint4 = 0, uint8 = 0, uint16 = 0, uint32 = 0, uint64 = 0,
  float4_e2m1fn = 1, float8_e3m4 = 1e-1, float8_e4m3 = 1e-1, float8_e8m0fnu = 1,
  float8_e4m3b11fnuz = 1e-1, float8_e4m3fn = 1e-1, float8_e4m3fnuz = 1e-1, float8_e5m2 = 1e-1,
  float8_e5m2fnuz = 1e-1, bfloat16 = 1e-2, float16 = 1e-3, float32 = 1e-6, float64 = 1e-15,
  complex64 = 1e-6, complex128 = 1e-15)

tu_default_gradient_tolerance <- c(
  float4_e2m1fn = 1, float8_e3m4 = 1e-1, float8_e4m3 = 1e-1, float8_e8m0fnu = 1,
  float8_e4m3b11fnuz = 1e-1, float8_e4m3fn = 1e-1, float8_e4m3fnuz = 1e-1, float8_e5m2 = 1e-1,
  float8_e5m2fnuz = 1e-1, bfloat16 = 1e-1, float16 = 1e-2, float32 = 2e-3, float64 = 1e-5,
  complex64 = 1e-3, complex128 = 1e-5)

tu_assert <- function(...) jax_error(..., class = "jax_assertion_error")

tu_dtype <- function(x) aval_of(as_jax(x))$dtype

# A tolerance is NULL, one number, or numbers named by dtype.
tu_is_dict <- function(tol) !is.null(names(tol))

tu_tolerance <- function(dt, tol = NULL) {
  if (!is.null(tol) && !tu_is_dict(tol)) return(tol[[1L]])
  dt <- canonicalize_dtype(dt)
  if (dt %in% names(tol)) tol[[dt]] else tu_default_tolerance[[dt]]
}

tu_merge_tolerance <- function(tol, default) {
  if (is.null(tol)) return(default)
  if (!tu_is_dict(tol)) return(tol)
  out <- default
  for (k in names(tol)) out[[dtype(k)]] <- tol[[k]]
  out
}

tu_host <- function(x) if (is_jax_value(x)) as.array(x) else x

# numpy.testing.assert_allclose (equal_nan = TRUE) on host data.
tu_assert_allclose <- function(a, b, atol, rtol, err_msg) {
  a <- as.vector(a)
  b <- as.vector(b)
  fail <- function(what) {
    tu_assert("\nNot equal to tolerance rtol=", format(rtol), ", atol=", format(atol), "\n",
              err_msg, "\n", what)
  }
  if (!identical(is.nan(Re(a)) | is.nan(Im(a)), is.nan(Re(b)) | is.nan(Im(b))))
    fail("x and y nan location mismatch")
  ok <- !(is.nan(Re(a)) | is.nan(Im(a)))
  a <- a[ok]
  b <- b[ok]
  inf_a <- is.infinite(a)
  if (!identical(inf_a, is.infinite(b)) || !isTRUE(all(a[inf_a] == b[inf_a])))
    fail("x and y inf location or sign mismatch")
  a <- a[!inf_a]
  b <- b[!inf_a]
  bad <- Mod(a - b) > atol + rtol * Mod(b)
  if (any(bad)) {
    fail(paste0("Mismatched elements: ", sum(bad), " / ", length(bad),
                "\nMax absolute difference: ", format(max(Mod(a - b))),
                "\nMax relative difference: ", format(max(Mod(a - b) / Mod(b)))))
  }
  invisible(TRUE)
}

tu_assert_close <- function(a, b, atol = NULL, rtol = NULL, err_msg = "") {
  da <- tu_dtype(a)
  db <- tu_dtype(b)
  ha <- tu_host(a)
  hb <- tu_host(b)
  if (!identical(as.integer(r_shape(ha)), as.integer(r_shape(hb))))
    tu_assert(err_msg, ": shapes differ: (", paste(r_shape(ha), collapse = ", "), ") and (",
              paste(r_shape(hb), collapse = ", "), ")")
  if (da == "float0" && db == "float0") return(invisible(TRUE))
  atol <- max(tu_tolerance(da, atol), tu_tolerance(db, atol)) * length(ha)
  rtol <- max(tu_tolerance(da, rtol), tu_tolerance(db, rtol)) * length(hb)
  # assert_allclose defaults apply when a tolerance is zero.
  tu_assert_allclose(ha, hb, atol = if (atol) atol else 0, rtol = if (rtol) rtol else 1e-7,
                     err_msg = err_msg)
}

tu_check_close <- function(xs, ys, atol = NULL, rtol = NULL, err_msg = "") {
  tree_map(function(x, y) tu_assert_close(x, y, atol, rtol, err_msg), xs, ys)
  invisible(TRUE)
}

tu_check_dtypes_match <- function(xs, ys) {
  tree_map(function(x, y) {
    if (canonicalize_dtype(tu_dtype(x)) != canonicalize_dtype(tu_dtype(y)))
      tu_assert("dtypes differ: ", tu_dtype(x), " and ", tu_dtype(y))
    NULL
  }, xs, ys)
  invisible(TRUE)
}

# Leaves that are R data stay R data (the analog of Python scalars); device arrays keep
# their dtype.
tu_like <- function(result, x) {
  if (is_jax_value(x)) lax_convert_element_type(as_jax(result), aval_of(x)$dtype) else result
}

tu_binary <- function(op) function(xs, ys) {
  tree_map(function(x, y) {
    if (is_jax_value(x) || is_jax_value(y)) tu_like(op(as_jax(x), as_jax(y)), as_jax(x)) else op(x, y)
  }, xs, ys)
}

tu_add <- tu_binary(function(x, y) x + y)
tu_sub <- tu_binary(function(x, y) x - y)

# Subtraction with inf - inf == 0.
tu_safe_sub <- function(xs, ys) {
  tree_map(function(x, y) {
    if (is_jax_value(x) || is_jax_value(y)) {
      x <- as_jax(x)
      y <- as_jax(y)
      tu_like(lax_select(x == y, lax_full_like(x - y, 0), x - y), x)
    } else {
      ifelse(x == y, 0, x - y)
    }
  }, xs, ys)
}

tu_conj <- function(xs) {
  tree_map(function(x) {
    if (is_jax_value(x)) { if (is_complex_dtype(aval_of(x)$dtype)) lax_conj(x) else x } else Conj(x)
  }, xs)
}

tu_scalar_mul <- function(xs, a) {
  tree_map(function(x) if (is_jax_value(x)) tu_like(x * a, x) else x * a, xs)
}

tu_rand_like <- function(x) {
  if (is_jax_value(x)) {
    a <- aval_of(x)
    n <- prod_int(a$shape)
    v <- if (is_complex_dtype(a$dtype)) complex(real = stats::rnorm(n), imaginary = stats::rnorm(n))
         else if (is_inexact_dtype(a$dtype)) stats::rnorm(n)
         else trunc(stats::rnorm(n))
    lax_reshape(as_jax(v, a$dtype), a$shape)
  } else {
    v <- if (is.complex(x)) complex(real = stats::rnorm(length(x)), imaginary = stats::rnorm(length(x)))
         else stats::rnorm(length(x))
    if (!is.null(dim(x))) dim(v) <- dim(x)
    v
  }
}

# Run `code` with a fixed seed and leave the caller's random stream untouched.
tu_with_seed <- function(code) {
  genv <- globalenv()
  old <- if (exists(".Random.seed", envir = genv, inherits = FALSE)) get(".Random.seed", envir = genv)
  on.exit({
    if (is.null(old)) {
      if (exists(".Random.seed", envir = genv, inherits = FALSE)) rm(".Random.seed", envir = genv)
    } else {
      assign(".Random.seed", old, envir = genv)
    }
  })
  set.seed(0L)
  code
}

tu_inner_prod <- function(xs, ys) {
  dts <- character(0)
  total <- 0
  fx <- tree_flatten(xs)$leaves
  fy <- tree_flatten(ys)$leaves
  if (length(fx) != length(fy)) tu_assert("inner_prod: tree structures differ")
  for (i in seq_along(fx)) {
    dx <- tu_dtype(fx[[i]])
    dy <- tu_dtype(fy[[i]])
    dts <- c(dts, promote_types_dtype(dx, dy))
    total <- total + Re(sum(Conj(as.vector(tu_host(fx[[i]]))) * as.vector(tu_host(fy[[i]]))))
  }
  dt <- Reduce(promote_types_dtype, dts)
  if (is_complex_dtype(dt)) dt <- complex_component(dt)
  as_jax(total, dt)
}

tu_numerical_jvp <- function(f, primals, tangents, eps = tu_eps) {
  delta <- tu_scalar_mul(tangents, eps)
  f_pos <- do.call(f, tu_add(primals, delta))
  f_neg <- do.call(f, tu_sub(primals, delta))
  tu_scalar_mul(tu_safe_sub(f_pos, f_neg), 0.5 / eps)
}

tu_msg <- function(err_msg, what) if (nzchar(err_msg)) paste(err_msg, what) else what

#' Check derivatives against finite differences
#'
#' Ports of `jax.test_util.check_grads`, `check_jvp` and `check_vjp`. Derivatives are
#' compared with a central finite difference in one random direction, so the cost does
#' not grow with the number of inputs or outputs. `test_util_check_grads()` checks
#' forward and reverse mode up to `order`.
#'
#' Tolerances default to JAX's per dtype gradient tolerances (`2e-3` for `float32`,
#' `1e-5` for `float64`); as in JAX they are multiplied by the number of elements
#' compared. Pass one number, or numbers named by dtype to override single dtypes.
#' The random direction is drawn with a fixed seed and the caller's random number
#' stream is left untouched.
#' @param f Function to check at `do.call(f, args)`.
#' @param args List of argument values.
#' @param order Derivatives up to this order are checked.
#' @param modes Modes to check: any of `"fwd"`, `"rev"` and `"lin"`.
#' @param atol,rtol Absolute and relative tolerance.
#' @param eps Step of the finite difference (default `1e-4`).
#' @param f_jvp Function of `(primals, tangents)` computing the JVP of `f`, typically
#'   `function(p, t) jvp(f, p, t)`.
#' @param f_vjp Function of the arguments computing the VJP of `f`, typically
#'   `function(...) vjp(f, ...)`.
#' @param err_msg Text added to the error message.
#' @return `TRUE`, invisibly; an error of class `jax_assertion_error` when a check fails.
#' @examples
#' \dontrun{
#' test_util_check_grads(function(x) sin(x) * x, list(as_jax(c(1, 2, 3))), order = 2)
#' }
#' @export
test_util_check_grads <- function(f, args, order, modes = c("fwd", "rev"), atol = NULL, rtol = NULL,
                                  eps = NULL) {
  args <- as.list(args)
  eps <- if (is.null(eps) || !eps) tu_eps else eps
  bad <- setdiff(modes, c("fwd", "rev", "lin"))
  if (length(bad)) jax_error("check_grads: unknown mode '", bad[[1L]], "'", class = "jax_value_error")
  jvp_from_lin <- function(g) function(primals, tangents) {
    r <- do.call(linearize, c(list(g), primals))
    list(r[[1L]], do.call(r[[2L]], tangents))
  }
  check <- function(f, args, order, err_msg = "") {
    if ("fwd" %in% modes) {
      msg <- if (nzchar(err_msg)) paste("JVP of", err_msg) else "JVP"
      f_jvp <- function(primals, tangents) jvp(f, primals, tangents)
      test_util_check_jvp(f, f_jvp, args, atol = atol, rtol = rtol, eps = eps, err_msg = msg)
      if (order > 1) check(f_jvp, list(args, args), order - 1, msg)
    }
    if ("rev" %in% modes) {
      msg <- if (nzchar(err_msg)) paste("VJP of", err_msg) else "VJP"
      test_util_check_vjp(f, function(...) vjp(f, ...), args, atol = atol, rtol = rtol, eps = eps,
                          err_msg = msg)
      if (order > 1) {
        f_vjp <- function(...) {
          r <- vjp(f, ...)
          r[[2L]](r[[1L]])
        }
        check(f_vjp, args, order - 1, msg)
      }
    }
    if ("lin" %in% modes) {
      msg <- if (nzchar(err_msg)) paste("LIN of", err_msg) else "LIN"
      f_lin <- jvp_from_lin(f)
      test_util_check_jvp(f, f_lin, args, atol = atol, rtol = rtol, eps = eps, err_msg = msg)
      if (order > 1) check(f_lin, list(args, args), order - 1, msg)
    }
  }
  check(f, args, order)
  invisible(TRUE)
}

#' @rdname test_util_check_grads
#' @export
test_util_check_jvp <- function(f, f_jvp, args, atol = NULL, rtol = NULL, eps = 1e-4, err_msg = "") {
  args <- as.list(args)
  atol <- tu_merge_tolerance(atol, tu_default_gradient_tolerance)
  rtol <- tu_merge_tolerance(rtol, tu_default_gradient_tolerance)
  tangent <- tu_with_seed(tree_map(tu_rand_like, args))
  r <- f_jvp(args, tangent)
  v_out <- r[[1L]]
  t_out <- r[[2L]]
  tu_check_dtypes_match(v_out, t_out)
  v_out_expected <- do.call(f, args)
  tu_check_dtypes_match(v_out, v_out_expected)
  t_out_expected <- tu_numerical_jvp(f, args, tangent, eps = eps)
  tu_check_close(v_out, v_out_expected, atol = atol, rtol = rtol, err_msg = tu_msg(err_msg, "primal"))
  tu_check_close(t_out, t_out_expected, atol = atol, rtol = rtol, err_msg = tu_msg(err_msg, "tangent"))
  invisible(TRUE)
}

#' @rdname test_util_check_grads
#' @export
test_util_check_vjp <- function(f, f_vjp, args, atol = NULL, rtol = NULL, eps = 1e-4, err_msg = "") {
  args <- as.list(args)
  atol <- tu_merge_tolerance(atol, tu_default_gradient_tolerance)
  rtol <- tu_merge_tolerance(rtol, tu_default_gradient_tolerance)
  r <- do.call(f_vjp, args)
  v_out <- r[[1L]]
  vjpfun <- r[[2L]]
  v_out_expected <- do.call(f, args)
  tu_check_close(v_out, v_out_expected, atol = atol, rtol = rtol, err_msg = tu_msg(err_msg, "primal"))
  tu_with_seed({
    tangent <- tree_map(tu_rand_like, args)
    tangent_out <- tu_numerical_jvp(f, args, tangent, eps = eps)
    cotangent <- tree_map(tu_rand_like, v_out)
  })
  cotangent_out <- tu_conj(vjpfun(tu_conj(cotangent)))
  ip <- tu_inner_prod(tangent, cotangent_out)
  ip_expected <- tu_inner_prod(tangent_out, cotangent)
  tu_check_close(ip, ip_expected, atol = atol, rtol = rtol,
                 err_msg = tu_msg(err_msg, "cotangent projection"))
  invisible(TRUE)
}
