# Base R operators and the Math, Summary and other S3 methods on jax values. Each
# method must give the same value, dtype and weak type as the jnp function it maps to
# (those are checked against the wheel in test-core-jnp.R), eagerly and under jit.

# Same dtype and weak type; values equal up to float32 rounding (XLA may fuse or fold
# constants differently under jit, as it does for JAX).
expect_same_jax <- function(got, want, info = NULL) {
  expect_identical(jaxR:::aval_of(got)$dtype, jaxR:::aval_of(want)$dtype, info = info)
  expect_identical(jaxR:::aval_of(got)$weak, jaxR:::aval_of(want)$weak, info = info)
  expect_equal(as.array(got), as.array(want), tolerance = 1e-6, info = info)
}

test_that("arithmetic, comparison and logical operators map to jnp ufuncs", {
  skip_if_no_backend()
  x <- as_jax(c(-7.5, 0, 5, 3))
  y <- as_jax(c(2, -3, 4, 3))
  i <- as_jax(c(-7L, 0L, 5L, 3L))
  j <- as_jax(c(2L, -3L, 4L, 3L))
  b <- as_jax(c(TRUE, FALSE, TRUE, FALSE))
  ops <- list(
    "+" = jnp_add, "-" = jnp_subtract, "*" = jnp_multiply, "/" = jnp_true_divide,
    "^" = jnp_power, "%%" = jnp_remainder, "%/%" = jnp_floor_divide, "==" = jnp_equal,
    "!=" = jnp_not_equal, "<" = jnp_less, "<=" = jnp_less_equal, ">" = jnp_greater,
    ">=" = jnp_greater_equal)
  for (op in names(ops)) {
    f <- match.fun(op)
    for (pr in list(list(x, y), list(i, j), list(x, 2), list(3, x), list(i, 2L), list(i, 2.5),
                    list(c(1, 2, 3, 4), x))) {
      a <- pr[[1L]]; bb <- pr[[2L]]
      info <- paste(op, deparse(class(a)), deparse(class(bb)))
      expect_same_jax(f(a, bb), ops[[op]](a, bb), info)
      expect_same_jax(jit(function(u, v) f(u, v))(a, bb), ops[[op]](a, bb), paste(info, "jit"))
    }
  }
  expect_same_jax(i & j, jnp_bitwise_and(i, j))
  expect_same_jax(i | j, jnp_bitwise_or(i, j))
  expect_same_jax(b & TRUE, jnp_bitwise_and(b, TRUE))
  expect_same_jax(!b, jnp_logical_not(b))
  expect_same_jax(!i, jnp_logical_not(i))
  expect_same_jax(-x, jnp_negative(x))
  expect_same_jax(+x, x)
  expect_same_jax(jit(function(u) -u)(x), jnp_negative(x))
})

test_that("operator results match the wheel", {
  skip_if_no_backend()
  # jnp.remainder / jnp.floor_divide follow Python (and R) sign conventions.
  x <- as_jax(c(7, -7, 7, -7))
  y <- as_jax(c(2, 2, -2, -2))
  expect_jax_equal(x %% y, c(1, 1, -1, -1), dtype = "float32")
  expect_jax_equal(x %/% y, c(3, -4, -4, 3), dtype = "float32")
  expect_jax_equal(as_jax(c(7L, -7L)) %/% 2L, c(3, -4), dtype = "int32")
  expect_jax_equal(as_jax(c(1, 2, 3))^2, c(1, 4, 9), dtype = "float32")
  expect_jax_equal(2^as_jax(c(1, 2, 3)), c(2, 4, 8), dtype = "float32")
  expect_jax_equal(as_jax(c(2L, 3L))^2L, c(4, 9), dtype = "int32")
  expect_error(as_jax(c(2L, 3L))^-1L, "Integers cannot be raised to negative powers", class = "jax_type_error")
  expect_jax_equal(as_jax(1:3) / 2L, c(0.5, 1, 1.5), dtype = "float32")
  expect_identical(jaxR:::aval_of(as_jax(2) * 3L)$weak, TRUE)
  expect_identical(dtype(as_jax(1:2) * 2.5), "float32")
})

test_that("%*% is jnp.matmul", {
  skip_if_no_backend()
  a <- as_jax(matrix(c(1, 2, 3, 4, 5, 6), 2, byrow = TRUE))
  b <- as_jax(matrix(c(1, -1, 0.5, 2, -3, 1), 3, byrow = TRUE))
  expect_jax_equal(a %*% b, matrix(c(1 + 1 - 9, -1 + 4 + 3, 4 + 2.5 - 18, -4 + 10 + 6), 2, byrow = TRUE))
  expect_jax_equal(a %*% c(1, 1, 1), c(6, 15))
  expect_jax_equal(matrix(1, 1, 2) %*% a, matrix(c(5, 7, 9), 1))
  expect_jax_equal(jit(function(u, v) u %*% v)(a, b), as.array(jnp_matmul(a, b)))
})

test_that("Math group functions map to jnp functions", {
  skip_if_no_backend()
  x <- as_jax(c(0.25, 0.5, 1.5, 3))
  u <- as_jax(c(-0.75, -0.25, 0.25, 0.75))
  m <- list(
    abs = list(jnp_abs, x), sign = list(jnp_sign, u), sqrt = list(jnp_sqrt, x), floor = list(jnp_floor, x),
    ceiling = list(jnp_ceil, x), trunc = list(jnp_trunc, u), exp = list(jnp_exp, x),
    expm1 = list(jnp_expm1, x), log1p = list(jnp_log1p, x), log2 = list(jnp_log2, x),
    log10 = list(jnp_log10, x), log = list(jnp_log, x), cos = list(jnp_cos, x), sin = list(jnp_sin, x),
    tan = list(jnp_tan, x), acos = list(jnp_arccos, u), asin = list(jnp_arcsin, u),
    atan = list(jnp_arctan, x), cosh = list(jnp_cosh, x), sinh = list(jnp_sinh, x),
    tanh = list(jnp_tanh, x), acosh = list(jnp_arccosh, as_jax(c(1, 2, 3))), asinh = list(jnp_arcsinh, x),
    atanh = list(jnp_arctanh, u), round = list(jnp_round, as_jax(c(0.5, 1.5, 2.5, -0.5))))
  for (nm in names(m)) {
    f <- match.fun(nm)
    expect_same_jax(f(m[[nm]][[2L]]), m[[nm]][[1L]](m[[nm]][[2L]]), nm)
    expect_same_jax(jit(function(v) f(v))(m[[nm]][[2L]]), m[[nm]][[1L]](m[[nm]][[2L]]), paste(nm, "jit"))
  }
  expect_same_jax(log(x, 2), jnp_true_divide(jnp_log(x), jnp_log(2)))
  expect_jax_equal(round(as_jax(c(1.2345, -2.5678)), 2), c(1.23, -2.57), tolerance = 1e-6)
  expect_jax_equal(cospi(as_jax(c(0, 0.5, 1))), c(1, 0, -1), tolerance = 1e-6)
  expect_jax_equal(sinpi(as_jax(c(0, 0.5))), c(0, 1), tolerance = 1e-6)
  expect_error(signif(x), "signif is not supported")
})

test_that("round() accepts the digits argument by name", {
  skip_if_no_backend()
  expect_jax_equal(round(as_jax(c(1.2345, -2.5678)), digits = 2), c(1.23, -2.57), tolerance = 1e-6)
})

test_that("Math functions backed by other modules (lgamma, digamma, gamma, cumsum, ...)", {
  skip_if_no_backend()
  x <- as_jax(c(0.5, 1.5, 3))
  expect_same_jax(lgamma(x), scipy_special_gammaln(x))
  expect_same_jax(digamma(x), scipy_special_digamma(x))
  expect_jax_equal(gamma(x), gamma(c(0.5, 1.5, 3)), tolerance = 1e-5)
  expect_jax_equal(cumsum(x), c(0.5, 2, 5))
  expect_jax_equal(cumprod(x), c(0.5, 0.75, 2.25))
})

test_that("Summary group functions map to jnp reductions", {
  skip_if_no_backend()
  x <- as_jax(matrix(c(1, -2, 3.5, 0.5, 4, -1), 2, byrow = TRUE))
  y <- as_jax(c(10, 20))
  b <- as_jax(c(TRUE, FALSE, TRUE))
  expect_same_jax(sum(x), jnp_sum(x))
  expect_same_jax(prod(x), jnp_prod(x))
  expect_same_jax(max(x), jnp_max(x))
  expect_same_jax(min(x), jnp_min(x))
  expect_same_jax(all(b), jnp_all(b))
  expect_same_jax(any(b), jnp_any(b))
  expect_jax_equal(sum(x, y), sum(as.array(x)) + 30)
  expect_jax_equal(max(x, y), 20)
  expect_jax_equal(range(x), c(-2, 4))
  expect_same_jax(mean(x), jnp_mean(x))
  expect_same_jax(t(x), jnp_transpose(x))
  expect_same_jax(jit(function(v) sum(v) + max(v) * min(v))(x), jnp_sum(x) + jnp_max(x) * jnp_min(x))
  expect_identical(dtype(sum(as_jax(c(TRUE, TRUE)))), "int32")
  expect_jax_equal(sum(as_jax(c(TRUE, TRUE))), 2)
})

test_that("all.equal compares values", {
  skip_if_no_backend()
  expect_true(all.equal(as_jax(matrix(1:4, 2)), matrix(1:4, 2)))
  expect_true(all.equal(as_jax(c(1, 2)), as_jax(c(1, 2))))
  expect_type(all.equal(as_jax(c(1, 2)), as_jax(c(1, 3))), "character")
})

test_that("all.equal compares a rank 1 jax array with an R vector", {
  skip_if_no_backend()
  expect_true(all.equal(as_jax(c(1, 2)), c(1, 2)))
  expect_true(all.equal(as_jax(c(1, 2)), as_jax(c(1, 2))))
  expect_type(all.equal(as_jax(c(1, 2)), c(1, 3)), "character")
})

test_that("base functions that would drop metadata raise jax_type_error", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2, 3))
  expect_error(c(x, x), class = "jax_type_error")
  expect_error(rev(x), class = "jax_type_error")
  expect_error(rep(x, 2), class = "jax_type_error")
  expect_error(aperm(x), class = "jax_type_error")
  expect_error(x[[1]], class = "jax_type_error")
  expect_error(dim(x) <- c(3L, 1L), class = "jax_type_error")
  expect_error(length(x) <- 2L, class = "jax_type_error")
  expect_error(rev(x), "jnp_flip")
})
