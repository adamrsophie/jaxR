# Shared test helpers.

skip_if_no_backend <- function() {
  if (!jaxR:::backend_available()) testthat::skip("no PJRT plugin (run jaxR::install_pjrt())")
}

# Relative tolerance of `ulps` units in the last place of a floating dtype. The expected
# values come from JAX on one machine, and XLA rounds floating point math differently
# between builds and CPUs (fused multiply-add, vector math functions): float results
# agree to a few units in the last place, not bit for bit. Integer and bit valued
# results are still compared exactly.
ulp_tol <- function(dtype, ulps = 16) {
  ulps * switch(dtype, float16 = 2^-10, bfloat16 = 2^-7, float32 = , complex64 = 2^-23, 2^-52)
}

# Compare a jax array (or R data) with expected R data, checking shape and dtype.
expect_jax_equal <- function(object, expected, tolerance = NULL, dtype = NULL, shape = NULL) {
  act <- testthat::quasi_label(rlang::enquo(object), arg = "object")
  x <- act$val
  if (!is.null(dtype)) testthat::expect_identical(jaxR::dtype(x), dtype)
  got <- if (inherits(x, "jax_value")) as.array(x) else x
  if (!is.null(shape)) testthat::expect_identical(as.integer(dim(x)), as.integer(shape))
  tol <- tolerance %||% if (inherits(x, "jax_value") &&
                             jaxR::dtype(x) %in% c("float32", "complex64")) 1e-5 else 1e-12
  if (is.logical(expected) || is.logical(got)) {
    testthat::expect_equal(as.vector(got), as.vector(expected))
  } else {
    testthat::expect_equal(as.vector(got), as.vector(expected), tolerance = tol)
  }
  if (!is.null(dim(expected)) && length(dim(expected)) > 1L)
    testthat::expect_identical(as.integer(dim(got)), as.integer(dim(expected)))
  invisible(act$val)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# Central finite difference check of a scalar function's gradient (float64 required).
check_grad_fd <- function(f, x, eps = 1e-6, tolerance = 1e-5) {
  g <- as.array(jaxR::grad(f)(x))
  xv <- as.array(x)
  fd <- vapply(seq_along(xv), function(i) {
    e <- xv * 0
    e[i] <- eps
    (as.numeric(as.array(f(jaxR::as_jax(xv + e, "float64")))) -
       as.numeric(as.array(f(jaxR::as_jax(xv - e, "float64"))))) / (2 * eps)
  }, 1)
  testthat::expect_equal(as.vector(g), fd, tolerance = tolerance)
}
