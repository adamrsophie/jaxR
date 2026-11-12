# Windowed reductions. Expected values from jax 0.12.0.1 (x64 enabled).

rw_x <- function() {
  matrix(c(0.3, 1.7, -2.1, 4.4, 0.9, 2.2, -0.5, 3.3, 1.1, -1.8,
           0.6, 2.9, -0.7, 1.4, 3.8, -1.2, 0.1, 2.5, -3.6, 1.9), 4, byrow = TRUE)
}
rw_w <- function() matrix(0:19, 4, byrow = TRUE) / 10

stack_loop <- function(f, xs, axis) {
  n <- dim(xs)[axis + 1L]
  outs <- lapply(seq_len(n) - 1L, function(i) f(jaxR::lax_index_in_dim(xs, i, axis, keepdims = FALSE)))
  jnp_stack(outs)
}

test_that("padtype_to_pads and reduce_window_shape_tuple match JAX", {
  expect_identical(lax_padtype_to_pads(c(5L, 6L), c(2L, 3L), c(2L, 2L), "SAME"),
                   list(c(0L, 1L), c(0L, 1L)))
  expect_identical(lax_padtype_to_pads(c(5L, 6L), c(2L, 3L), c(2L, 2L), "SAME_LOWER"),
                   list(c(1L, 0L), c(1L, 0L)))
  expect_identical(lax_padtype_to_pads(c(5L, 6L), c(2L, 3L), c(2L, 2L), "valid"),
                   list(c(0L, 0L), c(0L, 0L)))
  expect_error(lax_padtype_to_pads(5L, 2L, 1L, "FOO"), "Unrecognized padding type")
  expect_identical(lax_reduce_window_shape_tuple(c(5L, 6L), c(2L, 3L), c(2L, 1L),
                                                 list(c(1L, 0L), c(0L, 2L)), c(2L, 1L), c(1L, 2L)),
                   c(5L, 4L))
})

test_that("reduce_window sum, max, min and generic match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(rw_x(), "float64")
  expect_jax_equal(lax_reduce_window(x, 0, lax_add, c(2L, 3L), c(1L, 2L), "SAME"),
                   matrix(c(3.7, 7.9, 4.6, 5.2, 7.5, 4.5, 2.4, 2.6, 3.5, -1.1, -1.0, -1.7), 4,
                          byrow = TRUE), dtype = "float64")
  expect_jax_equal(lax_reduce_window(x, 0, lax_add, c(2L, 2L), c(2L, 1L),
                                     list(c(1L, 0L), c(0L, 2L)), c(1L, 2L), c(2L, 1L)),
                   matrix(c(2.2, -0.5, -0.5, 3.3, 3.3, 1.1, 1.1, -1.8, -1.8, 0.0,
                            1.0, -0.4, -0.4, 5.8, 5.8, -2.5, -2.5, 0.1, 0.1, 0.0), 2, byrow = TRUE))
  expect_jax_equal(lax_reduce_window(x, -Inf, lax_max, c(2L, 3L), c(1L, 2L), "SAME"),
                   matrix(c(2.2, 4.4, 4.4, 2.9, 3.3, 3.8, 2.9, 2.9, 3.8, 0.1, 2.5, 1.9), 4,
                          byrow = TRUE))
  expect_jax_equal(lax_reduce_window(x, Inf, lax_min, c(3L, 2L), c(1L, 1L), "VALID"),
                   matrix(c(-0.5, -2.1, -2.1, -1.8, -1.2, -0.7, -3.6, -3.6), 2, byrow = TRUE))
  expect_jax_equal(lax_reduce_window(x, -Inf, lax_max, c(2L, 2L), c(2L, 2L), "SAME_LOWER",
                                     c(2L, 1L)),
                   matrix(c(0.3, 1.7, 4.4, 2.2, 3.3, 1.1, 0.6, 2.9, 3.8), 3, byrow = TRUE))
  expect_jax_equal(lax_reduce_window(x, 1, lax_mul, c(2L, 2L), c(1L, 2L), "VALID"),
                   matrix(c(-0.561, -33.5412, -1.914, -3.5574, -0.2088, 8.82), 3, byrow = TRUE))
  xi <- as_jax(matrix(c(3, -7, 2, 9, -4, 5, 1, -8), 2, byrow = TRUE), "int64")
  expect_jax_equal(lax_reduce_window(xi, 9223372036854775807, lax_min, c(1L, 2L), c(1L, 1L), "SAME"),
                   matrix(c(-7, -7, 2, 9, -4, 1, -8, -8), 2, byrow = TRUE), dtype = "int64")
  expect_jax_equal(lax_reduce_window(xi, -2^63, lax_max, c(2L, 1L), c(1L, 1L),
                                     list(c(1L, 1L), c(0L, 0L))),
                   matrix(c(3, -7, 2, 9, 3, 5, 2, 9, -4, 5, 1, -8), 3, byrow = TRUE), dtype = "int64")
  # variadic reduction: window argmax
  iota <- lax_broadcasted_iota("int64", c(4L, 5L), 1L)
  red <- function(a, b) {
    pick <- lax_ge(a[[1L]], b[[1L]])
    list(lax_select(pick, a[[1L]], b[[1L]]), lax_select(pick, a[[2L]], b[[2L]]))
  }
  out <- lax_reduce_window(list(x, iota), list(-Inf, as_jax(0, "int64")), red, c(1L, 3L),
                           c(1L, 1L), "VALID")
  expect_jax_equal(out[[1L]], matrix(c(1.7, 4.4, 4.4, 3.3, 3.3, 3.3, 2.9, 2.9, 3.8, 2.5, 2.5, 2.5),
                                     4, byrow = TRUE))
  expect_jax_equal(out[[2L]], matrix(c(1, 3, 3, 2, 2, 2, 1, 1, 4, 2, 2, 2), 4, byrow = TRUE),
                   dtype = "int64")
})

test_that("reduce_window under jit and dtype rules", {
  skip_if_no_backend()
  x <- as_jax(rw_x(), "float32")
  f <- function(x) lax_reduce_window(x, -Inf, lax_max, c(2L, 3L), c(1L, 2L), "SAME")
  expect_jax_equal(jit(f)(x), f(x), dtype = "float32")
  g <- function(x) lax_reduce_window(x, 1, lax_mul, c(2L, 2L), c(1L, 2L), "VALID")
  expect_jax_equal(jit(g)(x), g(x), tolerance = 1e-6)
  xi <- as_jax(matrix(c(3L, -7L, 2L, 9L, -4L, 5L, 1L, -8L), 2, byrow = TRUE))
  expect_jax_equal(lax_reduce_window(xi, 0, lax_add, c(2L, 2L), c(1L, 1L), "SAME"),
                   matrix(c(-3, 1, 4, 1, 1, 6, -7, -8), 2, byrow = TRUE), dtype = "int32")
  expect_jax_equal(lax_reduce_window(as_jax(c(1, 2, 3)), 0, lax_add, 2L, 1L, "VALID"), c(3, 5),
                   dtype = "float32")
  cj <- make_jaxpr(function(x) lax_reduce_window(x, 0, lax_add, c(2L, 2L), c(1L, 1L)))(xi)
  expect_identical(cj$jaxpr$eqns[[1L]]$prim$name, "reduce_window_sum")
  cj <- make_jaxpr(g)(x)
  expect_identical(cj$jaxpr$eqns[[length(cj$jaxpr$eqns)]]$prim$name, "reduce_window")
})

test_that("reduce_window errors match JAX", {
  skip_if_no_backend()
  x <- as_jax(c(0, 1, 2, 3, 4))
  expect_error(lax_reduce_window(x, 0, lax_add, c(2L, 2L), 1L), "wrong number of window_dimensions")
  expect_error(lax_reduce_window(x, 0, lax_add, 2L, c(1L, 1L)), "inconsistent window_strides")
  expect_error(lax_reduce_window(x, 0, lax_add, 0L, 1L), "must have every element be strictly positive")
  expect_error(lax_reduce_window(x, 0, lax_add, 2L, 1L, "FOO"), "Unrecognized padding type")
  expect_error(lax_reduce_window(x, as_jax(c(0, 0)), lax_add, 2L, 1L),
               "expected init_values to be scalars")
  expect_error(lax_reduce_window(x, 1, lax_mul, 2L, 1L, "VALID", base_dilation = 0L),
               "All dilations must be positive")
  expect_error(lax_reduce_window(list(x, x), list(0), function(a, b) a, 2L, 1L),
               "same tree structure")
  expect_error(lax_reduce_window(as_jax(c(TRUE, FALSE)), FALSE, lax_add, 2L, 1L),
               "operand to reduce_window_sum must have a number dtype, got bool")
  expect_error(lax_reduce_window(as_jax(c(1L, 2L)), as_jax(1, "float32"), lax_mul, 2L, 1L),
               "inconsistent dtypes for operands and init_values")
})

test_that("reduce_window gradients match finite differences and JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(rw_x(), "float64")
  f_sum <- function(x) jnp_sum(jnp_sin(lax_reduce_window(x, 0, lax_add, c(2L, 2L), c(2L, 1L),
                                                         list(c(1L, 0L), c(0L, 2L)), c(1L, 2L),
                                                         c(2L, 1L))))
  check_grad_fd(f_sum, x)
  expect_jax_equal(grad(f_sum)(x), matrix(c(
    0, 0, 0, 0, 0,
    -0.048198811387206275, 3.5972871117865157, -0.2039205059350918, -0.6950949882427128, 1.5356041411698773,
    0, 0, 0, 0, 0,
    0.5403023058681395, 1.8421219880057702, 1.7710390338826378, -1.6022872310938674, 1.9900083305560516),
    4, byrow = TRUE))
  f_max <- function(x) jnp_sum(jnp_sin(lax_reduce_window(x, -Inf, lax_max, c(2L, 3L), c(1L, 2L), "SAME")))
  check_grad_fd(f_max, x)
  expect_jax_equal(grad(f_max)(x), matrix(c(
    0, 0, 0, -0.6146657399568387, 0,
    -0.5885011172553458, 0, -0.9874797699088649, 0, 0,
    0, -2.9128744954487718, 0, 0, -1.5819354238288337,
    0, 0.9950041652780258, -0.8011436155469337, 0, -0.32328956686350335), 4, byrow = TRUE))
  f_maxbd <- function(x) jnp_sum(jnp_sin(lax_reduce_window(x, -Inf, lax_max, c(2L, 2L), c(2L, 2L),
                                                           "SAME_LOWER", c(2L, 1L))))
  expect_jax_equal(grad(f_maxbd)(x), matrix(c(
    0.955336489125606, -0.12884449429552464, 0, -0.30733286997841935, 0,
    -0.5885011172553458, 0, -0.9874797699088649, 0.4535961214255773, 0,
    0.8253356149096783, -0.9709581651495905, 0, 0, -0.7909677119144168,
    0, 0, 0, 0, 0), 4, byrow = TRUE))
  f_min <- function(x) jnp_sum(jnp_cos(lax_reduce_window(x, Inf, lax_min, c(3L, 2L), c(1L, 1L), "VALID")))
  check_grad_fd(f_min, x)
  expect_jax_equal(grad(f_min)(x), matrix(c(
    0, 0, 1.7264187332977474, 0, 0,
    0, 0.479425538604203, 0, 0, 0.9738476308781951,
    0, 0, 0.644217687237691, 0, 0,
    0.9320390859672263, 0, 0, -0.8850408865897049, 0), 4, byrow = TRUE))
  # generic reduce_window: forward mode only, as in JAX
  w <- as_jax(rw_w(), "float64")
  fp <- function(x) lax_reduce_window(x, 1, lax_mul, c(2L, 2L), c(1L, 2L), "VALID")
  expect_jax_equal(jvp(fp, list(x), list(w))[[2L]],
                   matrix(c(0.5127, -30.6009, -2.0542, -0.5467, -3.507, -5.3424), 3, byrow = TRUE))
  expect_error(grad(function(x) jnp_sum(fp(x)))(x))
  # max pooling with window dilation: JVP works, VJP is not implemented (as in JAX)
  fwd <- function(x) lax_reduce_window(x, -Inf, lax_max, c(2L, 2L), c(1L, 1L), "VALID", NULL, c(2L, 1L))
  expect_jax_equal(jvp(fwd, list(x), list(w))[[2L]],
                   matrix(c(1.1, 1.1, 0.3, 0.3, 0.5, 0.7, 0.7, 1.9), 2, byrow = TRUE))
  expect_error(grad(function(x) jnp_sum(fwd(x)))(x),
               "VJP not implemented for select_and_gather \\(MaxPool\\) with window dilation")
})

test_that("reduce_window second order derivatives", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(rw_x(), "float64")
  w <- as_jax(rw_w(), "float64")
  f2 <- function(x) jnp_sum(lax_reduce_window(x, -Inf, lax_max, c(2L, 3L), c(1L, 2L), "SAME")^3)
  expected <- matrix(c(0, 0, 0, 15.84, 0, 6.6, 0, 13.86, 0, 0, 0, 57.42, 0, 0, 63.84,
                       0, 0.96, 25.5, 0, 21.66), 4, byrow = TRUE)
  expect_jax_equal(jvp(grad(f2), list(x), list(w))[[2L]], expected)
  expect_jax_equal(grad(function(x) jnp_sum(grad(f2)(x) * w))(x), expected)
  fs <- function(x) jnp_sum(lax_reduce_window(x, 0, lax_add, c(2L, 2L), c(1L, 1L), "SAME")^2)
  check_grad_fd(function(x) jnp_sum(grad(fs)(x) * w), x)
})

test_that("reduce_window batching matches a stacked loop", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  xs <- as_jax(array(sin(seq_len(3 * 4 * 5)) * 3, c(3L, 4L, 5L)), "float64")
  fs <- list(
    function(x) lax_reduce_window(x, 0, lax_add, c(2L, 2L), c(1L, 2L), "SAME", c(1L, 2L), c(2L, 1L)),
    function(x) lax_reduce_window(x, -Inf, lax_max, c(2L, 3L), c(1L, 2L), "SAME"),
    function(x) lax_reduce_window(x, Inf, lax_min, c(3L, 2L), c(1L, 1L), "VALID"),
    function(x) lax_reduce_window(x, 1, lax_mul, c(2L, 2L), c(1L, 2L), "VALID"),
    function(x) grad(function(y) jnp_sum(jnp_sin(lax_reduce_window(y, -Inf, lax_max, c(2L, 2L),
                                                                     c(2L, 1L), "SAME"))))(x))
  for (f in fs) {
    for (ax in 0:2) {
      xa <- jnp_moveaxis(xs, 0L, ax)
      expect_jax_equal(vmap(f, in_axes = ax)(xa), stack_loop(f, xa, ax))
    }
  }
  # generic reduction with one batched and one unbatched operand
  iota <- lax_broadcasted_iota("int64", c(4L, 5L), 1L)
  red <- function(a, b) {
    pick <- lax_ge(a[[1L]], b[[1L]])
    list(lax_select(pick, a[[1L]], b[[1L]]), lax_select(pick, a[[2L]], b[[2L]]))
  }
  fv <- function(x) lax_reduce_window(list(x, iota), list(-Inf, as_jax(0, "int64")), red,
                                      c(1L, 3L), c(1L, 1L), "VALID")[[2L]]
  expect_jax_equal(vmap(fv, in_axes = 2L)(jnp_moveaxis(xs, 0L, 2L)), stack_loop(fv, xs, 0L))
  # vmap of the JVP (select_and_gather_add batching) and of grad of a sum
  w <- as_jax(rw_w(), "float64")
  fj <- function(x) jvp(function(y) lax_reduce_window(y, -Inf, lax_max, c(2L, 2L), c(1L, 1L), "SAME"),
                        list(x), list(w))[[2L]]
  expect_jax_equal(vmap(fj, in_axes = 1L)(jnp_moveaxis(xs, 0L, 1L)), stack_loop(fj, xs, 0L))
})
