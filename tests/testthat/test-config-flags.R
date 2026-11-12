# Configuration context managers.

test_that("debug_nans and debug_infs raise on non-finite results", {
  skip_if_no_backend()
  x <- as_jax(c(0, 1))
  expect_error(debug_nans(TRUE, jnp_log(x - 1)), "invalid value \\(nan\\)", class = "jax_floating_point_error")
  expect_silent(debug_nans(TRUE, jnp_log(x + 1)))
  expect_error(debug_infs(TRUE, jnp_log(x)), "inf", class = "jax_floating_point_error")
  expect_error(debug_nans(TRUE, jit(function(v) jnp_sqrt(v - 2))(x)), "nan")
  expect_false(jax_config("jax_debug_nans"))
})

test_that("strict dtype promotion forbids mixing strong dtypes", {
  skip_if_no_backend()
  a <- as_jax(c(1L, 2L))
  b <- as_jax(c(1, 2))
  expect_error(numpy_dtype_promotion("strict", a + b), class = "jax_type_promotion_error")
  expect_jax_equal(numpy_dtype_promotion("strict", b + 1), c(2, 3))
  expect_jax_equal(a + b, c(2, 4))
})

test_that("log_compiles, explain_cache_misses, no_tracing and no_execution", {
  skip_if_no_backend()
  f <- jit(function(v) v * 3)
  expect_message(log_compiles(TRUE, f(as_jax(c(1, 2, 3, 4, 5)))), "Compiling")
  g <- jit(function(v) v - 1)
  expect_message(explain_cache_misses(TRUE, g(as_jax(c(1, 2)))), "TRACING CACHE MISS")
  expect_error(no_tracing(g(as_jax(c(1, 2, 3)))), "no_tracing")
  expect_error(no_execution(as_jax(c(1, 2)) * 2), "no_execution")
})

test_that("transfer guards", {
  skip_if_no_backend()
  expect_error(transfer_guard("disallow", as.array(as_jax(c(1, 2)) + 1)), "Disallowed")
  expect_jax_equal(transfer_guard("disallow", device_get(device_put(c(1, 2)) * 2)), c(2, 4))
  expect_error(transfer_guard("disallow_explicit", device_get(as_jax(1) * 2)), "Disallowed")
  expect_message(transfer_guard_device_to_host("log", as.array(as_jax(c(3, 4)))), "transfer")
})
