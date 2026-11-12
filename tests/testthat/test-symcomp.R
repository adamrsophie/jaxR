# Interoperability with symcompR.

test_that("jax_lambdify evaluates, differentiates and agrees with symbolic derivatives", {
  skip_if_no_backend()
  skip_if_not_installed("symcompR")
  old <- jax_config_update("jax_enable_x64", TRUE)
  on.exit(jax_config_update("jax_enable_x64", old))
  x <- symcompR::Symbol("x")
  y <- symcompR::Symbol("y")
  e <- sin(x) * y^2 + exp(x / y) + symcompR::Rational(1, 3) * x
  f <- jax_lambdify(list(x, y), e)
  X <- as_jax(c(0.5, 1, 2), "float64")
  Y <- as_jax(c(1, 2, 3), "float64")
  xv <- c(0.5, 1, 2); yv <- c(1, 2, 3)
  expect_jax_equal(f(X, Y), sin(xv) * yv^2 + exp(xv / yv) + xv / 3)
  d <- jax_lambdify(list(x, y), symcompR::doit(symcompR::Derivative(e, x)))
  expect_jax_equal(jaxR::grad(function(a) sum(f(a, Y)))(X), as.array(d(X, Y)))
  pw <- jax_lambdify(x, symcompR::Piecewise(list(x^2, x < 1), list(2 * x, TRUE)))
  expect_jax_equal(jit(pw)(X), c(0.25, 2, 4))
})

test_that("grad and hessian delegate to symcompR for symbolic input", {
  skip_if_not_installed("symcompR")
  x <- symcompR::Symbol("x")
  y <- symcompR::Symbol("y")
  expect_s3_class(jaxR::hessian(x^2 * y, list(x, y)), "symmatrix")
  expect_false(is.function(jaxR::grad(x^2 * y, list(x, y))))
})

test_that("symbolic numbers combine with jax arrays", {
  skip_if_no_backend()
  skip_if_not_installed("symcompR")
  expect_jax_equal(as_jax(c(1, 2)) * symcompR::Rational(1, 2), c(0.5, 1))
  expect_error(as_jax(c(1, 2)) * symcompR::Symbol("x"), "free symbols")
})
