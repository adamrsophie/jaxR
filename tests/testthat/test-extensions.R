# jax.errors, jax.custom_batching and jax.test_util.

test_that("errors_* name the condition classes that jaxR signals", {
  skip_if_no_backend()
  expect_identical(errors_TracerBoolConversionError, "jax_tracer_bool_conversion_error")
  e <- tryCatch(jit(function(x) if (as.logical(x > 0)) x else -x)(1), error = identity)
  expect_s3_class(e, errors_TracerBoolConversionError)
  expect_s3_class(e, errors_ConcretizationTypeError)
  expect_s3_class(e, errors_JAXTypeError)
  e <- tryCatch(jit(function(x) as.array(x))(1), error = identity)
  expect_s3_class(e, errors_TracerArrayConversionError)
  e <- tryCatch(jit(function(x) as.integer(x))(1L), error = identity)
  expect_s3_class(e, errors_TracerIntegerConversionError)
  e <- tryCatch(jit(function(x, m) x[m])(as_jax(c(1, 2)), as_jax(c(TRUE, FALSE))), error = identity)
  expect_s3_class(e, errors_NonConcreteBooleanIndexError)
  leaked <- NULL
  jit(function(x) { leaked <<- x; x })(1)
  e <- tryCatch(leaked + 1, error = identity)
  expect_s3_class(e, errors_UnexpectedTracerError)
  expect_s3_class(tryCatch(as_jax(c(1, 2))[5L, 1L], error = identity), errors_JAXIndexError)
})

test_that("custom_vmap uses its rule under vmap and the function elsewhere", {
  skip_if_no_backend()
  calls <- 0L
  sizes <- NULL
  f <- custom_vmap(function(x, y) x * y, function(axis_size, in_batched, xs, ys) {
    calls <<- calls + 1L
    sizes <<- axis_size
    list(xs * ys + 100, TRUE)
  })
  x <- as_jax(c(1, 2, 3))
  y <- as_jax(c(4, 5, 6))
  expect_jax_equal(f(x, y), c(4, 10, 18))
  expect_jax_equal(vmap(f)(x, y), c(104, 110, 118))
  expect_identical(calls, 1L)
  expect_identical(as.integer(sizes), 3L)
  expect_jax_equal(jit(vmap(f))(x, y), c(104, 110, 118))
  # in_batched reports which arguments are mapped
  seen <- NULL
  g <- custom_vmap(function(x, y) x + y, function(axis_size, in_batched, xs, y) {
    seen <<- in_batched
    list(xs + y, TRUE)
  })
  expect_jax_equal(vmap(g, in_axes = list(0L, NULL))(x, 10), c(11, 12, 13))
  expect_identical(seen, list(TRUE, FALSE))
  # forward mode differentiates through the rule
  expect_jax_equal(jvp(vmap(f), list(x, y), list(jnp_ones_like(x), jnp_zeros_like(y)))[[2L]], c(4, 5, 6))
  # the rule can be attached later; registration returns a new function
  h <- custom_vmap_def_vmap(custom_vmap(function(x) x),
                            function(axis_size, in_batched, xs) list(xs * 2, TRUE))
  expect_jax_equal(vmap(h)(x), c(2, 4, 6))
  expect_error(vmap(custom_vmap(function(x) x))(x), "No batching rule defined")
  expect_error(custom_vmap(1), class = "jax_type_error")
  expect_error(custom_vmap_def_vmap(function(x) x, function(...) NULL), class = "jax_type_error")
  expect_error(vmap(custom_vmap(function(x) x, function(axis_size, in_batched, xs) xs))(x),
               "must return list")
})

test_that("sequential_vmap maps one element at a time", {
  skip_if_no_backend()
  f <- sequential_vmap(function(x) jnp_sum(x) * 2)
  m <- as_jax(matrix(c(1, 2, 3, 4, 5, 6), 2))
  expect_jax_equal(f(as_jax(c(1, 2))), 6)
  expect_jax_equal(vmap(f)(m), c(18, 24))
  expect_jax_equal(jit(vmap(f))(m), c(18, 24))
  g <- sequential_vmap(function(x, y) list(x + y, x * y))
  r <- vmap(g)(as_jax(c(1, 2)), as_jax(c(3, 4)))
  expect_jax_equal(r[[1L]], c(4, 6))
  expect_jax_equal(r[[2L]], c(3, 8))
})

test_that("check_grads, check_jvp and check_vjp accept correct derivatives and reject wrong ones", {
  skip_if_no_backend()
  old <- jax_config_update("jax_enable_x64", TRUE)
  on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(0.3, 0.7), "float64")
  f <- function(x) jnp_sum(jnp_sin(x) * x)
  expect_true(test_util_check_grads(f, list(x), order = 2L))
  expect_true(test_util_check_grads(f, list(x), order = 1L, modes = "rev"))
  expect_true(test_util_check_grads(function(a, b) jnp_sum(a * jnp_exp(b)), list(x, x * 2), order = 2L))
  wrong <- custom_jvp(function(x) jnp_sin(x), jvp = function(primals, tangents) {
    list(jnp_sin(primals[[1L]]), 2 * jnp_cos(primals[[1L]]) * tangents[[1L]])
  })
  expect_error(test_util_check_grads(wrong, list(x), order = 1L), class = "jax_assertion_error")
  expect_error(test_util_check_grads(f, list(x), order = 1L, modes = "bad"), class = "jax_value_error")
  expect_true(test_util_check_jvp(jnp_sin, function(p, t) jvp(jnp_sin, p, t), list(x)))
  expect_true(test_util_check_vjp(jnp_sin, function(...) vjp(jnp_sin, ...), list(x)))
  expect_error(test_util_check_jvp(jnp_sin, function(p, t) list(jnp_sin(p[[1L]]), t[[1L]]), list(x)),
               class = "jax_assertion_error")
})
