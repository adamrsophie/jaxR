# Ports of tests/xla_metadata_test.py: metadata appear as frontend attributes of the
# lowered operations.

lowered_lines <- function(f, ...) strsplit(jit(f)$lower(...)$as_text(), "\n")[[1]]
attr_lines <- function(lines, op) grep(paste0("\"stablehlo\\.", op, "\""), lines, value = TRUE)
sm <- experimental_xla_metadata_set_xla_metadata

test_that("set_xla_metadata tags the operations staged in its context", {
  skip_if_no_backend()
  x <- jnp_ones(3)
  l <- lowered_lines(function(x) sm(code = jnp_sin(x) * 2, a = "b", n = 1L, f = 1.5, t = TRUE), x)
  want <- 'mhlo.frontend_attributes = {a = "b", f = "1.5", n = "1", t = "true"}'
  expect_true(grepl(want, attr_lines(l, "sine"), fixed = TRUE))
  expect_true(grepl(want, attr_lines(l, "multiply"), fixed = TRUE))
  # Operations staged outside are untagged.
  l <- lowered_lines(function(x) { y <- sm(code = jnp_sin(x), k = "v"); jnp_cos(y) }, x)
  expect_true(grepl('k = "v"', attr_lines(l, "sine"), fixed = TRUE))
  expect_false(grepl("frontend", attr_lines(l, "cosine")))
})

test_that("nested contexts merge and NULL removes a key", {
  skip_if_no_backend()
  x <- jnp_ones(3)
  l <- lowered_lines(function(x) {
    sm(code = list(jnp_sin(x), sm(code = jnp_cos(x), a = "inner", z = NULL)), a = "outer", z = 3)
  }, x)
  expect_true(grepl('{a = "outer", z = "3.0"}', attr_lines(l, "sine"), fixed = TRUE))
  expect_true(grepl('{a = "inner"}', attr_lines(l, "cosine"), fixed = TRUE))
})

test_that("tagging values, decorators, fuse_limit and must_fuse_call", {
  skip_if_no_backend()
  x <- jnp_ones(3)
  l <- lowered_lines(function(x) sm(jnp_sin(x), k = "v") * 2, x)
  expect_true(grepl('{k = "v"}', attr_lines(l, "sine"), fixed = TRUE))
  expect_false(grepl("frontend", attr_lines(l, "multiply")))
  l <- lowered_lines(function(x) experimental_xla_metadata_fuse_limit(jnp_exp(x)), x)
  expect_true(grepl('{FUSE_LIMIT = "true"}', attr_lines(l, "exponential"), fixed = TRUE))
  l <- lowered_lines(function(x) experimental_xla_metadata_must_fuse_call("1")(function(x) jnp_sum(x^2))(x), x)
  expect_true(grepl('{MUST_FUSE = "1"}', attr_lines(l, "multiply"), fixed = TRUE))
  g <- sm(a = "d")(function(x) jnp_tanh(x))
  expect_true(grepl('{a = "d"}', attr_lines(lowered_lines(g, x), "tanh"), fixed = TRUE))
  h <- experimental_xla_metadata_xla_metadata_call2(list(s = "l3"))(function(x) jnp_log(x))
  expect_true(grepl('{s = "l3"}', attr_lines(lowered_lines(h, x), "log"), fixed = TRUE))
})

test_that("existing frontend attributes are kept", {
  skip_if_no_backend()
  l <- lowered_lines(function(m) sm(code = jnp_linalg_cholesky(m), q = "w"), jnp_eye(3))
  cc <- grep("custom_call", l, value = TRUE)
  expect_true(grepl('num_batch_dims = "0"', cc, fixed = TRUE))
})

test_that("metadata do not change values and compose with transformations", {
  skip_if_no_backend()
  x <- jnp_ones(3)
  f <- jit(function(x) jnp_sin(x))
  expect_equal(as.numeric(sm(code = f(x), a = "c")), rep(sin(1), 3), tolerance = 1e-6)
  expect_equal(as.numeric(grad(function(x) jnp_sum(sm(jnp_sin(x), k = "v")))(x)), rep(cos(1), 3),
               tolerance = 1e-6)
  expect_equal(dim(vmap(function(x) sm(jnp_sin(x), k = "v"))(jnp_ones(c(2L, 3L)))), c(2L, 3L))
  expect_error(sm(NULL, 2, code = 1), "named", class = "jax_type_error")
  expect_error(sm(code = 1, a = list(1)), "strings, logicals or numbers", class = "jax_type_error")
})
