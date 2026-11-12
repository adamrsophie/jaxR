# Named axis collectives under vmap.

test_that("psum, pmean, pmax and axis_index under vmap", {
  skip_if_no_backend()
  x <- as_jax(matrix(c(1, 2, 3, 4, 5, 6), 3, 2))  # 3 lanes of length 2
  f <- vmap(function(v) v - lax_pmean(v, "i"), axis_name = "i")
  expect_jax_equal(f(x), matrix(c(-1, 0, 1, -1, 0, 1), 3, 2))
  expect_jax_equal(vmap(function(v) lax_psum(v, "i"), axis_name = "i")(x), matrix(c(6, 6, 6, 15, 15, 15), 3, 2))
  expect_jax_equal(vmap(function(v) lax_pmax(v, "i"), axis_name = "i")(x), matrix(c(3, 3, 3, 6, 6, 6), 3, 2))
  expect_jax_equal(vmap(function(v) lax_axis_index("i") + 0L * lax_convert_element_type(lax_index_in_dim(v, 0L, keepdims = FALSE), "int32"), axis_name = "i")(x),
                   c(0L, 1L, 2L))
  expect_jax_equal(vmap(function(v) lax_psum(1, "i"), axis_name = "i", axis_size = 4L)(as_jax(1:4)), rep(4, 4))
  expect_error(lax_psum(as_jax(1), "i"), "unbound axis name")
})

test_that("collectives are differentiable and jit compatible", {
  skip_if_no_backend()
  old <- jax_config_update("jax_enable_x64", TRUE); on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(1, 2, 3), "float64")
  g <- grad(function(v) sum(vmap(function(z) z * lax_psum(z, "i"), axis_name = "i")(v)))(x)
  expect_jax_equal(g, rep(sum(1:3), 3) + 2 * (1:3) * 0 + rep(sum(1:3), 3))
  expect_jax_equal(vmap(jit(function(v) v / lax_psum(v, "i")), axis_name = "i")(x), (1:3) / 6)
})

test_that("all_gather, ppermute and all_to_all", {
  skip_if_no_backend()
  x <- as_jax(c(10, 20, 30))
  ag <- vmap(function(v) lax_all_gather(v, "i"), axis_name = "i")(x)
  expect_jax_equal(ag, matrix(c(10, 10, 10, 20, 20, 20, 30, 30, 30), 3, 3))
  pp <- vmap(function(v) lax_ppermute(v, "i", list(c(0, 1), c(1, 2), c(2, 0))), axis_name = "i")(x)
  expect_jax_equal(pp, c(30, 10, 20))
  m <- as_jax(matrix(1:9, 3, 3, byrow = TRUE) * 1)
  a2a <- vmap(function(v) lax_all_to_all(v, "i", 0L, 0L), axis_name = "i")(m)
  expect_jax_equal(a2a, t(matrix(1:9, 3, 3, byrow = TRUE)))
})
