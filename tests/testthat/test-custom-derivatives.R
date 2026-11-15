# custom_jvp and custom_vjp functions that close over traced values.
test_that("custom derivative functions closing over jit tracers evaluate and batch as in JAX", {
  skip_if_no_backend()
  h <- jit(function(x, y) {
    f <- custom_jvp(function(a) a * y, function(p, t) list(p[[1]] * y, t[[1]] * y))
    f(x)
  })
  h2 <- jit(function(x, y) {
    f <- custom_vjp(function(a) a * y, function(a) list(a * y, NULL), function(r, g) list(g * y))
    f(x)
  })
  expect_jax_equal(h(2, 3), 6)
  expect_jax_equal(h2(2, 3), 6)
  expect_jax_equal(vmap(h, in_axes = list(0L, NULL))(jnp_arange(3), 3), c(0, 3, 6))
  expect_jax_equal(vmap(h2, in_axes = list(NULL, 0L))(2, jnp_arange(3)), c(0, 2, 4))
  # JAX cannot differentiate them either.
  expect_error(grad(h)(2, 3), "closes over a traced value", class = "jax_type_error")
  expect_error(grad(h2, argnums = 1L)(2, 3), "closes over a traced value", class = "jax_type_error")
  # Unbatched outputs of custom functions stay unbatched.
  f <- custom_jvp(function(x, c) list(x * 2, c + 1),
                  function(p, t) list(list(p[[1]] * 2, p[[2]] + 1), list(t[[1]] * 2, t[[2]])))
  o <- vmap(f, in_axes = list(0L, NULL), out_axes = list(0L, NULL))(jnp_arange(3), 5)
  expect_jax_equal(o[[2]], 6)
})
