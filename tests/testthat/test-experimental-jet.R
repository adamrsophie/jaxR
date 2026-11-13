# jax.experimental.jet: results recorded from JAX by tools/gen_jet_fixture.py.

jet_ref <- function() {
  skip_if_no_backend()
  testthat::skip_if_not_installed("jsonlite")
  jsonlite::fromJSON(testthat::test_path("fixtures", "jet.json"), simplifyVector = FALSE)
}

jet_close <- function(got, want, info) {
  g <- as.numeric(as.array(jnp_ravel(got)))
  w <- as.numeric(unlist(want))
  ok <- length(g) == length(w) && all(abs(g - w) <= 2e-4 * pmax(1, abs(w)))
  expect_true(ok, info = paste0(info, ": got ", paste(format(g, digits = 8), collapse = " "),
                                ", want ", paste(format(w, digits = 8), collapse = " ")))
}

jet_check <- function(got, want, info) {
  gp <- tree_leaves(got[[1L]])
  gs <- tree_leaves(got[[2L]])
  expect_identical(length(gp), length(want$primal), info = info)
  order <- if (length(gp)) length(gs) %/% length(gp) else 0L
  for (i in seq_along(gp)) {
    jet_close(gp[[i]], want$primal[[i]], paste(info, "primal", i))
    expect_identical(order, length(want$series[[i]]), info = info)
    for (k in seq_len(order))
      jet_close(gs[[(i - 1L) * order + k]], want$series[[i]][[k]], paste(info, "term", k, "of output", i))
  }
}

jet_x <- function() jnp_array(c(0.3, 0.7, 1.2), dtype = "float32")
jet_xs <- function() lapply(list(c(1, 0.5, -0.3), c(0.2, -0.1, 0.4), c(0.05, 0.3, -0.2)),
                            jnp_array, dtype = "float32")
jet_y <- function() jnp_array(c(1.5, 0.4, 2), dtype = "float32")
jet_ys <- function() lapply(list(c(0.3, -0.2, 0.1), c(-0.4, 0.6, 0.25), c(0.1, 0.1, -0.5)),
                            jnp_array, dtype = "float32")

test_that("jet of unary functions matches JAX", {
  ref <- jet_ref()$unary
  funs <- list(
    sin = jnp_sin, cos = jnp_cos, exp = jnp_exp, log = jnp_log, tanh = jnp_tanh,
    logistic = nn_sigmoid, sinh = jnp_sinh, cosh = jnp_cosh, sqrt = jnp_sqrt,
    rsqrt = lax_rsqrt, square = jnp_square, expm1 = jnp_expm1, log1p = jnp_log1p,
    exp2 = jnp_exp2, log2 = jnp_log2, erf = lax_erf, erfc = lax_erfc,
    erf_inv = function(x) lax_erf_inv(0.5 * x), asinh = jnp_arcsinh,
    acosh = function(x) jnp_arccosh(x + 1), atanh = function(x) jnp_arctanh(0.5 * x),
    abs = function(x) jnp_abs(x - 0.5), neg = function(x) -x,
    ipow3 = function(x) x^3, ipow_m2 = function(x) x^-2,
    ipow0 = function(x) lax_integer_pow(x, 0L),
    one_minus_square = lax_one_minus_square,
    sum = jnp_sum, reshape = function(x) jnp_reshape(x, c(3L, 1L)),
    rev = function(x) x[sl(step = -1)],
    concat = function(x) jnp_concatenate(list(x, 2 * x)), cumsum = jnp_cumsum,
    cumprod = jnp_cumprod, cummax = function(x) lax_cummax(jnp_sin(3 * x)),
    cummin = function(x) lax_cummin(jnp_sin(3 * x)),
    reduce_max = function(x) jnp_max(jnp_sin(3 * x)), reduce_min = jnp_min,
    gather = function(x) x[jnp_array(c(2L, 0L, 0L))], index = function(x) x[1],
    scatter_add = function(x) at_add(jnp_zeros(4L, "float32"), jnp_array(c(1L, 1L, 3L)), value = x),
    where = function(x) jnp_where(x > 0.5, jnp_sin(x), jnp_cos(x)),
    clip = function(x) jnp_clip(x, 0.5, 1),
    pad = function(x) jnp_pad(x, 1L), dot_self = function(x) jnp_dot(x, x),
    outer = function(x) jnp_outer(x, jnp_exp(x)),
    composite = function(x) jnp_sin(x) * jnp_exp(-x^2) / (1 + x),
    polynomial = function(x) lax_polynomial(x, c(1, 2, 3)),
    tree = function(x) list(a = jnp_sin(x), b = list(x * x, 2)),
    constant = function(x) jnp_ones(2L, "float32"))
  expect_setequal(names(funs), names(ref))
  for (name in names(funs)) {
    got <- experimental_jet_jet(funs[[name]], list(jet_x()), list(jet_xs()))
    jet_check(got, ref[[name]], name)
  }
})

test_that("jet of binary functions matches JAX", {
  ref <- jet_ref()$binary
  funs <- list(
    add = jnp_add, sub = jnp_subtract, mul = jnp_multiply, div = jnp_divide,
    pow = jnp_power, atan2 = jnp_arctan2, max = jnp_maximum, min = jnp_minimum,
    rem = jnp_remainder, dot = jnp_dot, matmul = function(x, y) jnp_outer(x, y) %*% y,
    max_tie = function(x, y) jnp_maximum(x, x + 0 * y))
  expect_setequal(names(funs), names(ref))
  for (name in names(funs)) {
    got <- experimental_jet_jet(funs[[name]], list(jet_x(), jet_y()), list(jet_xs(), jet_ys()))
    jet_check(got, ref[[name]], name)
  }
})

test_that("jet options, jit and scalars match JAX", {
  ref <- jet_ref()$extra
  x <- jet_x()
  xs <- jet_xs()
  jet_check(experimental_jet_jet(jnp_sin, list(x), list(xs), factorial_scaled = FALSE),
            ref$unscaled, "unscaled")
  jet_check(experimental_jet_jet(jnp_tanh, list(x), list(xs[1L])), ref$order1, "order1")
  zs <- rep(list(jnp_zeros(3L)), 3L)
  jet_check(experimental_jet_jet(function(x, y) x * jnp_exp(y), list(x, jet_y()), list(xs, zs)),
            ref$second_zero, "second_zero")
  f <- function(x) jnp_sin(x) * x
  jet_check(jit(function(x) experimental_jet_jet(f, list(x), list(xs)))(x), ref$jit, "jit")
  jet_check(experimental_jet_jet(jit(f), list(x), list(xs)), ref$inner_jit, "inner_jit")
  out <- experimental_jet_jet(jnp_sin, list(0.5^3), list(list(3 * 0.5^2, 6 * 0.5)))
  jet_check(out, ref$scalar, "scalar")
  expect_identical(dim(out[[1L]]), integer(0))
  expect_length(out[[2L]], 2L)
})

test_that("jet agrees with nested first order differentiation", {
  skip_if_no_backend()
  f <- function(x) jnp_sin(x) * jnp_exp(x)
  d1 <- grad(f)
  d2 <- grad(d1)
  d3 <- grad(d2)
  out <- experimental_jet_jet(f, list(0.4), list(list(1, 0, 0)))
  want <- c(as.numeric(as.array(d1(0.4))), as.numeric(as.array(d2(0.4))), as.numeric(as.array(d3(0.4))))
  got <- vapply(out[[2L]], function(t) as.numeric(as.array(t)), 0)
  expect_equal(got, want, tolerance = 1e-5)
})

test_that("jet ignores custom derivative rules", {
  skip_if_no_backend()
  f <- custom_jvp_defjvp(custom_jvp(function(x) jnp_sin(x)), function(primals, tangents) {
    list(jnp_sin(primals[[1L]]), 100 * tangents[[1L]])
  })
  got <- experimental_jet_jet(function(x) f(x) * 2, list(jet_x()), list(jet_xs()))
  want <- experimental_jet_jet(function(x) jnp_sin(x) * 2, list(jet_x()), list(jet_xs()))
  expect_equal(as.array(got[[1L]]), as.array(want[[1L]]))
  for (k in 1:3) expect_equal(as.array(got[[2L]][[k]]), as.array(want[[2L]][[k]]))
})

test_that("jet reports bad arguments and missing rules like JAX", {
  ref <- jet_ref()$extra
  x <- jet_x()
  xs <- jet_xs()
  expect_error(experimental_jet_jet(jnp_add, list(x, x), list(xs, xs[1:2])),
               "jet terms have inconsistent lengths for different arguments", class = "jax_value_error")
  expect_error(experimental_jet_jet(jnp_sin, list(list(x)), list(xs)),
               "primal value at position 0 is not an array", class = "jax_value_error")
  expect_error(experimental_jet_jet(jnp_sin, list(x), list(list(x, list(x)))),
               "term 1 for argument 0 is not an array", class = "jax_value_error")
  expect_identical(ref$err_tan, "KeyError")
  expect_error(experimental_jet_jet(jnp_tan, list(x), list(xs)), "tan", class = "jax_key_error")
  expect_identical(ref$err_cond, "KeyError")
  expect_error(experimental_jet_jet(function(x) lax_cond(x[0] > 0, jnp_sin, jnp_cos, x), list(x), list(xs)),
               "cond", class = "jax_key_error")
})
