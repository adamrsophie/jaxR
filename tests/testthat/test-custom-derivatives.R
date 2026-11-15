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
  # JAX cannot differentiate them either: the rules use a tracer of the jit trace (a
  # TypeError in JAX), and differentiating with respect to the closed over value raises.
  expect_error(grad(h)(2, 3), class = "jax_type_error")
  expect_error(grad(h2, argnums = 1L)(2, 3), "closes over a traced value", class = "jax_type_error")
  # Unbatched outputs of custom functions stay unbatched.
  f <- custom_jvp(function(x, c) list(x * 2, c + 1),
                  function(p, t) list(list(p[[1]] * 2, p[[2]] + 1), list(t[[1]] * 2, t[[2]])))
  o <- vmap(f, in_axes = list(0L, NULL), out_axes = list(0L, NULL))(jnp_arange(3), 5)
  expect_jax_equal(o[[2]], 6)
})

test_that("closed over values: rules for the explicit arguments still apply (JAX values)", {
  skip_if_no_backend()
  hh <- jit(function(x, y) {
    f <- custom_vjp(function(a) a + jnp_astype(y - y, "float32"),
                    function(a) list(a, NULL), function(r, g) list(g))
    f(x)
  })
  expect_equal(as.numeric(grad(hh)(2, 3L)), 1)
  h <- function(x, y) {
    f <- custom_jvp(function(a) a * y, function(p, t) list(p[[1]] * y, t[[1]] * y))
    checkpoint(f)(x)
  }
  expect_equal(as.numeric(vmap(grad(h), in_axes = list(NULL, 0L))(2, as_jax(c(1, 2, 3)))), c(1, 2, 3))
  hj <- function(y) {
    f <- custom_jvp(function(x) x * y, function(p, t) list(p[[1]] * y, t[[1]] * y))
    f(2)
  }
  hv <- function(y) {
    f <- custom_vjp(function(x) x * y, function(x) list(x * y, NULL), function(r, g) list(g * y))
    f(2)
  }
  y <- as_jax(c(1, 2, 3))
  expect_equal(as.numeric(vmap(hj)(y)), c(2, 4, 6))
  expect_equal(as.numeric(jit(vmap(hj))(y)), c(2, 4, 6))
  expect_equal(as.numeric(vmap(hv)(y)), c(2, 4, 6))
})
