# Tests for the jax.numpy array functions a to l (R/numpy_array.R). Expected values
# were generated with the jax 0.12.0.1 wheel.

# A jax array from values in NumPy (row-major) order.
np_arr <- function(values, shape = length(values), dtype = NULL) {
  data <- if (length(shape) > 1L) aperm(array(values, rev(shape))) else array(values, shape)
  as_jax(data, dtype)
}

# Compare with expected values given in NumPy (row-major) order.
expect_np <- function(object, values, shape = length(values), dtype = NULL, tolerance = NULL) {
  expected <- if (length(shape) > 1L) aperm(array(values, rev(shape))) else values
  expect_jax_equal(object, expected, dtype = dtype, shape = shape, tolerance = tolerance)
}

# Bit exact comparison with values in NumPy order.
expect_bits <- function(object, values, shape = length(values), dtype = NULL) {
  if (!is.null(dtype)) expect_identical(dtype(object), dtype)
  expect_identical(as.integer(dim(object)), as.integer(shape))
  got <- as.array(object)
  if (length(shape) > 1L) got <- aperm(got)
  expect_identical(as.vector(got), values)
}

with_x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)

row_of <- function(a, k) lax_index_in_dim(a, k, 0L, keepdims = FALSE)

# Apply f to each slice along axis 0 of the arguments and stack the results.
stack_loop <- function(f, n, ...) {
  args <- list(...)
  jnp_stack(lapply(seq_len(n) - 1L, function(k) do.call(f, lapply(args, row_of, k = k))))
}

expect_same <- function(got, want) {
  expect_identical(dim(got), dim(want))
  expect_identical(dtype(got), dtype(want))
  expect_equal(as.vector(as.array(got)), as.vector(as.array(want)), tolerance = 1e-6)
}

is_weak <- function(x) jaxR:::aval_of(x)$weak

test_that("isclose and allclose match the wheel", {
  skip_if_no_backend()
  a <- np_arr(c(1, 1.00001, NaN, Inf, 2, -Inf))
  b <- np_arr(c(1, 1, NaN, Inf, 2.1, Inf))
  expect_np(jnp_isclose(a, b), c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE), 6L, "bool")
  expect_np(jnp_isclose(a, b, equal_nan = TRUE), c(TRUE, FALSE, TRUE, TRUE, FALSE, FALSE), 6L, "bool")
  expect_np(jit(function(a, b) jnp_isclose(a, b))(a, b), c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE), 6L, "bool")
  expect_np(jnp_isclose(1:3, 2L), c(FALSE, TRUE, FALSE), 3L, "bool")
  expect_np(jnp_isclose(c(1, 1.1, 1.2), 1, rtol = 0, atol = 0.15), c(TRUE, TRUE, FALSE), 3L, "bool")
  expect_np(jnp_isclose(c(1 + 1i, 1 + 2i), c(1 + 1.000001i, 1 + 2.1i)), c(TRUE, FALSE), 2L, "bool")
  expect_np(jnp_allclose(c(1, 2), c(1, 2.000001)), TRUE, integer(0), "bool")
  expect_np(jnp_allclose(c(1, 2), c(1, 2.1)), FALSE, integer(0), "bool")
  expect_np(jit(jnp_allclose)(c(1, 2), c(1, 2.1)), FALSE, integer(0), "bool")
})

test_that("angle matches the wheel", {
  skip_if_no_backend()
  z <- as_jax(c(1 + 1i, -1 + 0i, 1i, -1 - 1i), "complex64")
  expect_np(jnp_angle(z), c(0.7853981852531433, 3.141592502593994, 1.5707963705062866, -2.356194496154785),
            4L, "float32")
  expect_np(jnp_angle(z, deg = TRUE), c(45, 180, 90, -135), 4L, "float32")
  expect_np(jit(jnp_angle, static_argnames = "deg")(z, deg = TRUE), c(45, 180, 90, -135), 4L, "float32")
  expect_np(jnp_angle(c(1L, -1L, 0L)), c(0, 3.141592502593994, 0), 3L, "float32")
  s <- jnp_angle(1.5)
  expect_identical(dtype(s), "float32")
  expect_false(is_weak(s))
  expect_np(jnp_angle(c(1.5, -2)), c(0, 3.141592502593994), 2L, "float32")
  with_x64(expect_identical(dtype(jnp_angle(c(1L, -1L))), "float64"))
})

test_that("append matches the wheel", {
  skip_if_no_backend()
  expect_np(jnp_append(1:2, c(3.5, 4)), c(1, 2, 3.5, 4), 4L, "float32")
  m <- np_arr(0:5, c(2L, 3L), "float32")
  expect_np(jnp_append(m, jnp_ones(c(1L, 3L)), axis = 0L), c(0:5, 1, 1, 1), c(3L, 3L), "float32")
  expect_np(jit(jnp_append, static_argnames = "axis")(m, jnp_ones(c(1L, 3L)), axis = 0L),
            c(0:5, 1, 1, 1), c(3L, 3L), "float32")
  expect_np(jnp_append(m, 9), c(0:5, 9), 7L, "float32")
  expect_error(jnp_append(m, jnp_ones(4L), axis = 0L))
})

test_that("apply_along_axis and apply_over_axes match the wheel", {
  skip_if_no_backend()
  a <- np_arr(0:23, c(2L, 3L, 4L), "float32")
  expect_np(jnp_apply_along_axis(function(v) jnp_sum(v), 1L, a), c(12, 15, 18, 21, 48, 51, 54, 57),
            c(2L, 4L), "float32")
  expect_np(jnp_apply_along_axis(function(v, s) jnp_cumsum(v) * s, 2L, a, 2),
            c(0, 2, 6, 12, 8, 18, 30, 44, 16, 34, 54, 76, 24, 50, 78, 108, 32, 66, 102, 140, 40, 82,
              126, 172), c(2L, 3L, 4L), "float32")
  outer2 <- function(v) jnp_outer(v, lax_slice(v, 0L, 2L))
  expect_np(jnp_apply_along_axis(outer2, 0L, a),
            c(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 0, 13, 28, 45, 64, 85, 108, 133, 160,
              189, 220, 253, 0, 13, 28, 45, 64, 85, 108, 133, 160, 189, 220, 253, 144, 169, 196,
              225, 256, 289, 324, 361, 400, 441, 484, 529), c(2L, 2L, 3L, 4L), "float32")
  expect_np(jit(function(a) jnp_apply_along_axis(function(v) jnp_sum(v), -1L, a))(a),
            c(6, 22, 38, 54, 70, 86), c(2L, 3L), "float32")
  expect_np(jnp_apply_over_axes(function(x, ax) jnp_sum(x, axis = ax), a, c(0L, 2L)),
            c(60, 92, 124), c(1L, 3L, 1L), "float32")
  expect_np(jnp_apply_over_axes(function(x, ax) jnp_sum(x, axis = ax, keepdims = TRUE), a, 1L),
            c(12, 15, 18, 21, 48, 51, 54, 57), c(2L, 1L, 4L), "float32")
  expect_error(jnp_apply_over_axes(function(x, ax) jnp_sum(x), a, 0L), class = "jax_value_error")
  with_x64({
    x <- as_jax(matrix(c(1, 2, 3, 4, 5, 6), 2), "float64")
    check_grad_fd(function(x) jnp_sum(jnp_apply_along_axis(function(v) jnp_sum(v^2) * v[0], 1L, x)), x)
  })
})

test_that("argwhere and flatnonzero match the wheel", {
  skip_if_no_backend()
  m <- np_arr(c(0, 1, 0, 2, 0, 3), c(2L, 3L), "float32")
  expect_np(jnp_argwhere(m), c(0, 1, 1, 0, 1, 2), c(3L, 2L), "int32")
  expect_np(jnp_argwhere(m, size = 4L, fill_value = -1L), c(0, 1, 1, 0, 1, 2, -1, -1), c(4L, 2L), "int32")
  expect_np(jit(function(m) jnp_argwhere(m, size = 4L, fill_value = -1L))(m),
            c(0, 1, 1, 0, 1, 2, -1, -1), c(4L, 2L), "int32")
  expect_identical(dim(jnp_argwhere(as_jax(3, "float32"))), c(1L, 0L))
  expect_identical(dim(jnp_argwhere(as_jax(0, "float32"))), c(0L, 0L))
  expect_np(jnp_flatnonzero(m), c(1, 3, 5), 3L, "int32")
  expect_np(jnp_flatnonzero(m, size = 2L), c(1, 3), 2L, "int32")
  expect_np(jnp_flatnonzero(m, size = 5L, fill_value = 7L), c(1, 3, 5, 7, 7), 5L, "int32")
  expect_np(vmap(function(r) jnp_flatnonzero(r, size = 2L))(m), c(1, 0, 0, 2), c(2L, 2L), "int32")
  expect_np(jnp_flatnonzero(c(TRUE, FALSE, TRUE)), c(0, 2), 2L, "int32")
  expect_error(jit(jnp_flatnonzero)(m), class = "jax_concretization_error")
  expect_error(jit(jnp_argwhere)(m), class = "jax_concretization_error")
  expect_error(jaxR:::npa_nonzero(as_jax(1)), class = "jax_value_error")
  with_x64(expect_np(jnp_argwhere(m), c(0, 1, 1, 0, 1, 2), c(3L, 2L), "int64"))
})

test_that("around matches the wheel", {
  skip_if_no_backend()
  x <- c(1.25, 2.5, -0.5, 3.14159, 0.5)
  expect_np(jnp_around(x, 1L), c(1.2000000476837158, 2.5, -0.5, 3.1000001430511475, 0.5), 5L, "float32")
  expect_np(jnp_around(x), c(1, 2, -0, 3, 0), 5L, "float32")
  expect_np(jit(jnp_around, static_argnames = "decimals")(x, decimals = 1L),
            c(1.2000000476837158, 2.5, -0.5, 3.1000001430511475, 0.5), 5L, "float32")
  expect_np(jnp_around(c(125, 251, -1450), -2L), c(100, 300, -1400), 3L, "float32")
  expect_np(jnp_around(1:2, 1L), c(1, 2), 2L, "int32")
  expect_error(jnp_around(1:2, -1L), class = "jax_not_implemented_error")
  expect_error(jnp_around(x, out = x), class = "jax_not_implemented_error")
  with_x64(check_grad_fd(function(x) jnp_sum(jnp_around(x, 1L) * x), as_jax(c(0.33, 1.71), "float64")))
})

test_that("array_equal and array_equiv match the wheel", {
  skip_if_no_backend()
  expect_np(jnp_array_equal(1:2, c(1, 2)), TRUE, integer(0), "bool")
  expect_np(jnp_array_equal(c(1, NaN), c(1, NaN), equal_nan = TRUE), TRUE, integer(0), "bool")
  expect_np(jnp_array_equal(c(1, NaN), c(1, NaN)), FALSE, integer(0), "bool")
  expect_np(jnp_array_equal(jnp_ones(2L), jnp_ones(c(1L, 2L))), FALSE, integer(0), "bool")
  expect_np(jit(jnp_array_equal)(c(1, 2), c(1, 3)), FALSE, integer(0), "bool")
  expect_np(jnp_array_equiv(1:2, np_arr(c(1L, 2L, 1L, 2L), c(2L, 2L))), TRUE, integer(0), "bool")
  # The wheel raises a TypeError from lax.eq here; the source intends False (NumPy).
  expect_np(jnp_array_equiv(1:2, 1:3), FALSE, integer(0), "bool")
})

test_that("array_split, hsplit and dsplit match the wheel", {
  skip_if_no_backend()
  a <- jnp_arange(10L, dtype = "float32")
  s <- jnp_array_split(a, 3L)
  expect_np(s[[1]], 0:3, 4L, "float32")
  expect_np(s[[2]], 4:6, 3L, "float32")
  expect_np(s[[3]], 7:9, 3L, "float32")
  s <- jnp_array_split(a, c(2, 5, 20))
  expect_length(s, 4L)
  expect_np(s[[2]], 2:4, 3L, "float32")
  expect_np(s[[3]], 5:9, 5L, "float32")
  expect_identical(dim(s[[4]]), 0L)
  s <- jnp_array_split(a, list(4L))
  expect_np(s[[2]], 4:9, 6L, "float32")
  expect_error(jnp_array_split(a, c(5, 3)), class = "jax_value_error")
  b <- np_arr(0:23, c(2L, 4L, 3L), "float32")
  h <- jnp_hsplit(b, 2L)
  expect_np(h[[1]], c(0:5, 12:17), c(2L, 2L, 3L), "float32")
  expect_np(h[[2]], c(6:11, 18:23), c(2L, 2L, 3L), "float32")
  expect_np(jnp_hsplit(a, 2L)[[2]], 5:9, 5L, "float32")
  expect_error(jnp_hsplit(a, 3L), class = "jax_value_error")
  d <- jnp_dsplit(b, list(1L))
  expect_np(d[[1]], c(0, 3, 6, 9, 12, 15, 18, 21), c(2L, 4L, 1L), "float32")
  expect_identical(dim(d[[2]]), c(2L, 4L, 2L))
  j <- jit(function(x) jnp_array_split(x, 3L))(a)
  expect_np(j[[3]], 7:9, 3L, "float32")
  with_x64(check_grad_fd(function(x) jnp_sum(jnp_array_split(x, 3L)[[2]]^2) + jnp_sum(jnp_array_split(x, 3L)[[1]]),
                         as_jax(c(1, 2, 3, 4, 5, 6, 7), "float64")))
})

test_that("atleast_1d, atleast_2d and atleast_3d match the wheel", {
  skip_if_no_backend()
  shapes <- list(integer(0), 3L, c(2L, 3L), c(2L, 3L, 4L))
  want <- list(list(1L, 3L, c(2L, 3L), c(2L, 3L, 4L)),
               list(c(1L, 1L), c(1L, 3L), c(2L, 3L), c(2L, 3L, 4L)),
               list(c(1L, 1L, 1L), c(1L, 3L, 1L), c(2L, 3L, 1L), c(2L, 3L, 4L)))
  fs <- list(jnp_atleast_1d, jnp_atleast_2d, jnp_atleast_3d)
  for (k in 1:3) for (i in 1:4) {
    expect_identical(dim(fs[[k]](jnp_ones(shapes[[i]]))), want[[k]][[i]])
  }
  expect_true(is_weak(jnp_atleast_1d(1)))
  expect_true(is_weak(jnp_atleast_2d(1)))
  out <- jnp_atleast_2d(1, c(1, 2))
  expect_length(out, 2L)
  expect_identical(dim(out[[2]]), c(1L, 2L))
  expect_identical(dim(jit(jnp_atleast_3d)(c(1, 2))), c(1L, 2L, 1L))
})

test_that("bincount matches the wheel", {
  skip_if_no_backend()
  x <- c(0L, 1L, 1L, 3L, 2L, 1L, 7L)
  expect_np(jnp_bincount(x), c(1, 3, 1, 1, 0, 0, 0, 1), 8L, "int32")
  expect_np(jnp_bincount(c(0L, 1L, 1L, 3L), weights = c(0.5, 1, 1.5, 2)), c(0.5, 2.5, 0, 2), 4L, "float32")
  expect_np(jnp_bincount(c(0L, 1L, 1L), minlength = 5L), c(1, 2, 0, 0, 0), 5L, "int32")
  expect_np(jnp_bincount(c(0L, 1L, 1L, 5L, -2L), length = 3L), c(2, 2, 0), 3L, "int32")
  expect_np(jnp_bincount(c(TRUE, FALSE, TRUE)), c(1, 2), 2L, "int32")
  expect_identical(dim(jnp_bincount(integer(0))), 0L)
  expect_np(jit(function(x) jnp_bincount(x, length = 3L))(c(0L, 1L, 1L, 5L, -2L)), c(2, 2, 0), 3L, "int32")
  expect_np(vmap(function(x) jnp_bincount(x, length = 3L))(np_arr(c(0L, 1L, 2L, 2L), c(2L, 2L))),
            c(1, 1, 0, 0, 0, 2), c(2L, 3L), "int32")
  expect_error(jit(jnp_bincount)(x), class = "jax_concretization_error")
  expect_error(jnp_bincount(c(0.5, 1)), class = "jax_type_error")
  expect_error(jnp_bincount(matrix(1L, 2, 2)), class = "jax_value_error")
  expect_error(jnp_bincount(1:3, weights = c(1, 2)), class = "jax_value_error")
  with_x64({
    expect_np(jnp_bincount(x), c(1, 3, 1, 1, 0, 0, 0, 1), 8L, "int64")
    check_grad_fd(function(w) jnp_sum(jnp_bincount(c(0L, 1L, 1L, 3L), weights = w)^2),
                  as_jax(c(0.5, 1, 1.5, 2), "float64"))
  })
})

test_that("block matches the wheel", {
  skip_if_no_backend()
  A <- jnp_ones(c(2L, 2L))
  B <- jnp_zeros(c(2L, 3L))
  C <- jnp_full(c(1L, 5L), 2)
  expect_np(jnp_block(list(list(A, B), list(C))), c(1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2),
            c(3L, 5L), "float32")
  expect_np(jit(jnp_block)(list(list(A, B), list(C))), c(1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2),
            c(3L, 5L), "float32")
  expect_np(jnp_block(list(c(1, 2), 3)), c(1, 2, 3), 3L, "float32")
  expect_np(jnp_block(list(list(1, 2), list(3, 4))), c(1, 2, 3, 4), c(2L, 2L), "float32")
  expect_error(jnp_block(list()), class = "jax_value_error")
  expect_error(jnp_block(list(list(A), B)), class = "jax_value_error")
})

test_that("broadcast_shapes and object queries match the wheel", {
  skip_if_no_backend()
  expect_identical(jnp_broadcast_shapes(c(2L, 1L), 3L), c(2L, 3L))
  expect_identical(jnp_broadcast_shapes(), integer(0))
  expect_identical(jnp_broadcast_shapes(3, c(2, 1)), c(2L, 3L))
  expect_error(jnp_broadcast_shapes(2L, 3L), class = "jax_value_error")
  expect_true(jnp_iscomplexobj(1i))
  expect_true(jnp_iscomplexobj(as_jax(c(1i, 2))))
  expect_false(jnp_iscomplexobj(1))
  expect_false(jnp_iscomplexobj(NULL))
  expect_false(jnp_iscomplexobj("a"))
  expect_true(jnp_isrealobj(jnp_ones(2L)))
  expect_true(jnp_isscalar(1))
  expect_true(jnp_isscalar(as_jax(1)))
  expect_false(jnp_isscalar(c(1, 2)))
  expect_false(jnp_isscalar(jnp_ones(1L)))
  expect_false(jnp_isscalar(list(1)))
  expect_true(jnp_iterable(c(1, 2)))
  expect_true(jnp_iterable(list(1)))
  expect_true(jnp_iterable(jnp_ones(1L)))
  expect_false(jnp_iterable(as_jax(1)))
  expect_false(jnp_iterable(1))
  expect_false(jnp_iterable(NULL))
})

test_that("can_cast, issubdtype, finfo and iinfo (R/numpy_misc.R) match the wheel", {
  skip_if_not(exists("jnp_can_cast", mode = "function"), "jnp_can_cast is defined in R/numpy_misc.R")
  cases <- list(c("int32", "float32", FALSE, TRUE), c("int32", "float64", TRUE, TRUE),
                c("float32", "float16", FALSE, TRUE), c("int8", "bfloat16", TRUE, TRUE),
                c("uint8", "int8", FALSE, TRUE), c("float64", "float32", FALSE, TRUE))
  for (cs in cases) {
    expect_identical(jnp_can_cast(cs[1], cs[2]), as.logical(cs[3]))
    expect_identical(jnp_can_cast(cs[1], cs[2], "same_kind"), as.logical(cs[4]))
    expect_false(jnp_can_cast(cs[1], cs[2], "no"))
    expect_true(jnp_can_cast(cs[1], cs[2], "unsafe"))
  }
  if (jaxR:::backend_available()) expect_true(jnp_can_cast(jnp_ones(2L, "int8"), "int16"))
  expect_error(jnp_can_cast("int8", "int16", "sometimes"), class = "jax_value_error")
  expect_true(jnp_issubdtype("float32", "floating"))
  expect_false(jnp_issubdtype("int32", "inexact"))
  expect_identical(as.numeric(jnp_finfo("float32")$eps), 2^-23)
  expect_identical(as.numeric(jnp_iinfo("int8")$max), 127)
})

test_that("choose matches the wheel", {
  skip_if_no_backend()
  ch <- list(c(0, 1, 2, 3), c(10, 11, 12, 13), c(20, 21, 22, 23))
  expect_np(jnp_choose(c(2L, 0L, 1L, 2L), ch), c(20, 1, 12, 23), 4L, "float32")
  expect_np(jnp_choose(c(3L, -1L, 1L, 5L), ch, mode = "wrap"), c(0, 21, 12, 23), 4L, "float32")
  expect_np(jnp_choose(c(3L, -1L, 1L, 5L), ch, mode = "clip"), c(20, 1, 12, 23), 4L, "float32")
  expect_np(jit(function(a) jnp_choose(a, ch, mode = "wrap"))(c(3L, -1L, 1L, 5L)), c(0, 21, 12, 23),
            4L, "float32")
  expect_np(jnp_choose(np_arr(c(0L, 1L), c(2L, 1L)), list(c(1, 2), 5)), c(1, 2, 5, 5), c(2L, 2L), "float32")
  expect_np(jnp_choose(c(1L, 0L), np_arr(1:4, c(2L, 2L))), c(3, 2), 2L, "int32")
  expect_error(jnp_choose(c(3L, 0L), ch), class = "jax_value_error")
  expect_error(jnp_choose(c(0.5, 1), ch), class = "jax_value_error")
  expect_error(jnp_choose(c(0L, 1L), ch, mode = "bogus"), class = "jax_value_error")
  expect_error(jit(function(a) jnp_choose(a, ch))(c(0L, 1L, 2L, 0L)), class = "jax_concretization_error")
  g <- grad(function(c0) jnp_sum(jnp_choose(c(2L, 0L, 0L, 2L), list(c0, ch[[2]], ch[[3]]))^2))(as_jax(ch[[1]]))
  expect_np(g, c(0, 2, 4, 0), 4L, "float32")
  expect_np(vmap(function(a) jnp_choose(a, ch, mode = "clip"))(np_arr(c(0L, 1L, 2L, 3L, 2L, 1L, 0L, 0L), c(2L, 4L))),
            c(0, 11, 22, 23, 20, 11, 2, 3), c(2L, 4L), "float32")
})

test_that("clip matches the wheel", {
  skip_if_no_backend()
  x <- c(-2, 0.5, 3)
  expect_np(jnp_clip(x, 0, 1), c(0, 0.5, 1), 3L, "float32")
  expect_np(jnp_clip(c(-2L, 5L, 3L), max = 4L), c(-2, 4, 3), 3L, "int32")
  expect_np(jnp_clip(c(-2L, 5L, 3L), min = 0.5), c(0.5, 5, 3), 3L, "float32")
  expect_np(jnp_clip(x, c(0, 1, 2), 2.5), c(0, 1, 2.5), 3L, "float32")
  expect_np(jnp_clip(x), x, 3L, "float32")
  expect_np(jit(jnp_clip)(x, 0, 1), c(0, 0.5, 1), 3L, "float32")
  expect_error(jnp_clip(), class = "jax_value_error")
  expect_error(jnp_clip(c(1i, 2)), class = "jax_value_error")
  expect_error(jnp_clip(x, max = 1i), class = "jax_value_error")
  X <- np_arr(c(-1, 0.2, 2, 0.7, -3, 4), c(2L, 3L), "float32")
  expect_same(vmap(function(r) jnp_clip(r, 0, 1))(X), stack_loop(function(r) jnp_clip(r, 0, 1), 2L, X))
  with_x64(check_grad_fd(function(x) jnp_sum(jnp_clip(x, -1, 1)^2), as_jax(c(-2, 0.5, 0.3, 3), "float64")))
})

test_that("hstack, dstack, column_stack and concat match the wheel", {
  skip_if_no_backend()
  a <- c(1, 2)
  b <- c(3, 4)
  expect_np(jnp_hstack(list(a, b)), c(1, 2, 3, 4), 4L, "float32")
  expect_np(jnp_hstack(list(jnp_ones(c(2L, 1L)), jnp_zeros(c(2L, 2L)))), c(1, 0, 0, 1, 0, 0), c(2L, 3L), "float32")
  M <- np_arr(0:5, c(2L, 3L), "float32")
  expect_np(jnp_hstack(M), 0:5, 6L, "float32")
  expect_np(jnp_hstack(list(1:2, 3:4), dtype = "float32"), 1:4, 4L, "float32")
  expect_np(jnp_hstack(list(1, 2)), c(1, 2), 2L, "float32")
  expect_np(jnp_dstack(list(a, b)), c(1, 3, 2, 4), c(1L, 2L, 2L), "float32")
  expect_np(jnp_dstack(M), c(0, 3, 1, 4, 2, 5), c(1L, 3L, 2L), "float32")
  expect_np(jnp_column_stack(list(a, b)), c(1, 3, 2, 4), c(2L, 2L), "float32")
  expect_np(jnp_column_stack(list(np_arr(0:3, c(2L, 2L), "float32"), a)), c(0, 1, 1, 2, 3, 2), c(2L, 3L), "float32")
  expect_np(jnp_column_stack(M), c(0, 3, 1, 4, 2, 5), c(3L, 2L), "float32")
  expect_np(jnp_column_stack(np_arr(0:11, c(2L, 3L, 2L), "float32")), c(0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11),
            c(3L, 4L), "float32")
  expect_np(jnp_concat(list(jnp_ones(c(1L, 2L)), jnp_zeros(c(1L, 2L))), axis = 1L), c(1, 1, 0, 0), c(1L, 4L), "float32")
  expect_np(jnp_concat(list(jnp_ones(c(1L, 2L)), jnp_zeros(2L)), axis = NULL), c(1, 1, 0, 0), 4L, "float32")
  expect_np(jit(function(a, b) jnp_hstack(list(a, b)))(a, b), c(1, 2, 3, 4), 4L, "float32")
  expect_np(jit(jnp_column_stack)(M), c(0, 3, 1, 4, 2, 5), c(3L, 2L), "float32")
  expect_error(jnp_concat(list()), class = "jax_value_error")
  expect_error(jnp_concat(list(1, 2)), class = "jax_value_error")
  with_x64(check_grad_fd(function(x) jnp_sum(jnp_column_stack(list(x, x^2))[1, ]^2), as_jax(c(1, 2), "float64")))
})

test_that("compress and extract match the wheel", {
  skip_if_no_backend()
  x <- np_arr(0:11, c(3L, 4L), "float32")
  expect_np(jnp_compress(c(TRUE, FALSE, TRUE), x, axis = 0L), c(0:3, 8:11), c(2L, 4L), "float32")
  expect_np(jnp_compress(c(0, 1, 1, 0), x, axis = 1L), c(1, 2, 5, 6, 9, 10), c(3L, 2L), "float32")
  expect_np(jnp_compress(c(TRUE, FALSE, TRUE), x), c(0, 2), 2L, "float32")
  expect_np(jnp_compress(c(TRUE, FALSE, TRUE), x, axis = 0L, size = 3L, fill_value = -1),
            c(0:3, 8:11, -1, -1, -1, -1), c(3L, 4L), "float32")
  expect_np(jnp_compress(c(FALSE, FALSE, TRUE), x, axis = 0L, size = 1L), 8:11, c(1L, 4L), "float32")
  expect_np(jnp_compress(c(TRUE, FALSE, TRUE, FALSE, FALSE), jnp_arange(4L, dtype = "float32")), c(0, 2), 2L, "float32")
  expect_np(jnp_extract(x > 6, x), 7:11, 5L, "float32")
  expect_np(jnp_extract(x > 9, x, size = 4L, fill_value = 0.5), c(10, 11, 0.5, 0.5), 4L, "float32")
  expect_np(jit(function(c, x) jnp_compress(c, x, axis = 0L, size = 3L, fill_value = -1))(c(TRUE, FALSE, TRUE), x),
            c(0:3, 8:11, -1, -1, -1, -1), c(3L, 4L), "float32")
  expect_np(vmap(function(c, r) jnp_compress(c, r, size = 2L, fill_value = -1))(
    np_arr(c(TRUE, FALSE, TRUE, FALSE, FALSE, TRUE), c(2L, 3L)), np_arr(0:5, c(2L, 3L), "float32")),
    c(0, 2, 5, -1), c(2L, 2L), "float32")
  expect_error(jit(function(c, x) jnp_compress(c, x))(c(TRUE, FALSE, TRUE), x), class = "jax_concretization_error")
  expect_error(jit(function(c, x) jnp_extract(c, x))(x > 6, x), class = "jax_concretization_error")
  expect_error(jnp_compress(c(TRUE, FALSE, TRUE, TRUE), jnp_arange(3L)), class = "jax_value_error")
  expect_error(jnp_compress(x > 1, x), class = "jax_value_error")
  expect_error(jnp_compress(c(TRUE, FALSE), x, axis = 0L, size = 4L), class = "jax_value_error")
  with_x64(check_grad_fd(function(v) jnp_sum(jnp_compress(c(TRUE, FALSE, TRUE), v, size = 2L)^2),
                         as_jax(c(1.5, 2, -3), "float64")))
})

test_that("copy and empty match the wheel", {
  skip_if_no_backend()
  x <- jnp_arange(3L)
  expect_same(jnp_copy(x), x)
  expect_np(jnp_copy(c(1, 2)), c(1, 2), 2L, "float32")
  expect_error(jnp_copy(x, order = "F"), class = "jax_not_implemented_error")
  e <- jnp_empty(c(2L, 3L))
  expect_identical(dim(e), c(2L, 3L))
  expect_identical(dtype(e), "float32")
  expect_identical(dtype(jnp_empty(c(2L, 3L), "int32")), "int32")
  expect_identical(dim(jnp_empty(3L)), 3L)
  e <- jnp_empty_like(jnp_ones(3L, "int32"))
  expect_identical(dtype(e), "int32")
  expect_false(is_weak(e))
  e <- jnp_empty_like(2)
  expect_identical(dim(e), integer(0))
  expect_true(is_weak(e))
  expect_identical(dim(jnp_empty_like(jnp_ones(3L), shape = c(2L, 2L), dtype = "int8")), c(2L, 2L))
})

test_that("cov and corrcoef (R/numpy_reductions.R) match the wheel", {
  skip_if_no_backend()
  skip_if_not(exists("jnp_cov", mode = "function"), "jnp_cov is defined in R/numpy_reductions.R")
  m <- np_arr(c(1, 2, 4, 7, 3, 2, 1, 0, 5, 5), c(2L, 5L), "float32")
  expect_np(jnp_cov(m), c(5.3, 2.45, 2.45, 5.3), c(2L, 2L), "float32")
  expect_np(jnp_cov(jnp_transpose(m), rowvar = FALSE), c(5.3, 2.45, 2.45, 5.3), c(2L, 2L), "float32")
  expect_np(jnp_cov(m, bias = TRUE), c(4.24, 1.96, 1.96, 4.24), c(2L, 2L), "float32")
  expect_np(jnp_cov(m, ddof = 2L), c(7.066666603088379, 3.266666889190674, 3.266666889190674, 7.066667079925537),
            c(2L, 2L), "float32")
  expect_np(jnp_cov(m[0], m[1]), c(5.3, 2.45, 2.45, 5.3), c(2L, 2L), "float32")
  expect_np(jnp_cov(m[0]), 5.3, integer(0), "float32")
  fw <- c(1L, 2L, 1L, 1L, 3L)
  aw <- c(0.5, 1, 2, 1, 1)
  expect_np(jnp_cov(m, fweights = fw), c(3.267857313156128, 1.571428656578064, 1.571428656578064, 4.857142925262451),
            c(2L, 2L), "float32")
  expect_np(jnp_cov(m, aweights = aw), c(4.326086521148682, 1.97826087474823, 1.97826087474823, 6.41304349899292),
            c(2L, 2L), "float32")
  expect_np(jnp_cov(m, fweights = fw, aweights = aw),
            c(2.8709676265716553, 1.0806450843811035, 1.0806450843811035, 5.725805759429932), c(2L, 2L), "float32")
  expect_np(jnp_cov(m, fweights = fw, aweights = aw, ddof = 0L),
            c(2.463667869567871, 0.9273356795310974, 0.9273356199264526, 4.91349458694458), c(2L, 2L), "float32")
  expect_np(jnp_cov(jnp_ones(c(2L, 0L))), rep(NaN, 4), c(2L, 2L), "float32")
  expect_np(jnp_cov(jnp_ones(c(1L, 0L))), NaN, integer(0), "float32")
  expect_np(jnp_cov(np_arr(c(1L, 2L, 4L, 3L, 3L, 1L), c(2L, 3L))),
            c(2.3333332538604736, -1.6666666269302368, -1.6666666269302368, 1.3333333730697632), c(2L, 2L), "float32")
  cm <- np_arr(c(1 + 1i, 2, 3 - 1i, 2, 1i, 1), c(2L, 3L), "complex64")
  expect_np(jnp_cov(cm), c(2, -0.5 + 0.5i, -0.5 - 0.5i, 1.3333333), c(2L, 2L), "complex64")
  expect_np(jnp_corrcoef(m), c(1, 0.46226415038108826, 0.46226415038108826, 1), c(2L, 2L), "float32")
  expect_np(jnp_corrcoef(m[0]), 1, integer(0), "float32")
  expect_np(jnp_corrcoef(cm), c(1, -0.30618623 + 0.30618623i, -0.30618623 - 0.30618623i, 1), c(2L, 2L), "complex64")
  expect_np(jit(jnp_cov)(m), c(5.3, 2.45, 2.45, 5.3), c(2L, 2L), "float32")
  expect_np(jit(function(m, w) jnp_cov(m, aweights = w))(m, aw),
            c(4.326086521148682, 1.97826087474823, 1.97826087474823, 6.41304349899292), c(2L, 2L), "float32")
  expect_error(jnp_cov(jnp_ones(c(2L, 2L, 2L))), class = "jax_value_error")
  expect_error(jnp_cov(m, dtype = "int32"), class = "jax_value_error")
  expect_error(jnp_cov(m, fweights = c(1.5, 1, 1, 1, 1)), class = "jax_type_error")
  expect_error(jnp_cov(m, aweights = c(1, 1)), class = "jax_runtime_error")
  expect_error(jnp_corrcoef(m, dtype = "int32"), class = "jax_value_error")
  with_x64({
    m64 <- np_arr(c(1, 2, 4, 7, 3, 2, 1, 0, 5, 5), c(2L, 5L), "float64")
    expect_bits(jnp_cov(m64), c(0x1.5333333333333p+2, 0x1.399999999999ap+1, 0x1.399999999999ap+1, 0x1.5333333333333p+2),
                c(2L, 2L), "float64")
    expect_np(grad(function(z) jnp_sum(jnp_cov(z)^2))(m64),
              c(-14.19, -11.34, -3.19, 24.96, 3.76, -9.06, -11.91, -12.31, 21.54, 11.74), c(2L, 5L), "float64",
              tolerance = 1e-10)
    expect_np(grad(function(z) jnp_corrcoef(z)[0, 1])(m64),
              c(0.024029903880384496, -0.04494482022071912, -0.13572445710217163, 0.034709861160555334,
                0.12192951228195091, -0.10012459950160202, -0.031149875400498392, 0.08499466002135994,
                0.11747953008187968, -0.07119971520113923), c(2L, 5L), "float64", tolerance = 1e-10)
    check_grad_fd(function(z) jnp_sum(jnp_cov(z, aweights = c(0.5, 1, 2, 1, 1))), m64)
  })
})

test_that("cross matches the wheel", {
  skip_if_no_backend()
  a <- np_arr(c(1, 2, 3, 4, 5, 6), c(2L, 3L), "float32")
  b <- np_arr(c(7, 8, 9, 1, 0, 2), c(2L, 3L), "float32")
  expect_np(jnp_cross(a, b), c(-6, 12, -6, 10, -2, -5), c(2L, 3L), "float32")
  expect_np(jnp_cross(a, c(0, 0, 1)), c(2, -1, 0, 5, -4, 0), c(2L, 3L), "float32")
  expect_np(jnp_cross(jnp_transpose(a), jnp_transpose(b), axisa = 0L, axisb = 0L, axisc = 0L),
            c(-6, 10, 12, -2, -6, -5), c(3L, 2L), "float32")
  expect_np(jnp_cross(jnp_transpose(a), jnp_transpose(b), axis = 0L), c(-6, 10, 12, -2, -6, -5), c(3L, 2L), "float32")
  expect_np(jnp_cross(c(1L, 0L, 0L), c(0L, 1L, 0L)), c(0, 0, 1), 3L, "int32")
  expect_np(jit(jnp_cross)(a, b), c(-6, 12, -6, 10, -2, -5), c(2L, 3L), "float32")
  expect_same(vmap(jnp_cross)(a, b), stack_loop(jnp_cross, 2L, a, b))
  expect_error(jnp_cross(c(1, 2), c(3, 4)), class = "jax_value_error")
  with_x64({
    expect_np(grad(function(a) jnp_sum(jnp_cross(a, c(1, 2, 3))^2))(as_jax(c(0.5, -1, 2), "float64")),
              c(5, -46, 29), 3L, "float64")
    check_grad_fd(function(a) jnp_sum(jnp_cross(a, a^2)), as_jax(c(0.5, -1, 2), "float64"))
  })
})

test_that("delete and insert match the wheel", {
  skip_if_no_backend()
  x <- jnp_arange(10L, dtype = "float32")
  expect_np(jnp_delete(x, 3L), c(0:2, 4:9), 9L, "float32")
  expect_np(jnp_delete(x, -1), 0:8, 9L, "float32")
  expect_np(jnp_delete(x, c(1, 3, 3, -2)), c(0, 2, 4, 5, 6, 7, 9), 7L, "float32")
  expect_np(jnp_delete(jnp_arange(4L, dtype = "float32"), c(TRUE, FALSE, TRUE, FALSE)), c(1, 3), 2L, "float32")
  expect_np(jnp_delete(x, c(1L, 3L, -2L), assume_unique_indices = TRUE), c(0, 2, 4, 5, 6, 7, 9), 7L, "float32")
  expect_np(jnp_delete(x, sl(1L, 8L, 3L)), c(0, 2, 3, 5, 6, 8, 9), 7L, "float32")
  m <- np_arr(0:11, c(3L, 4L), "float32")
  expect_np(jnp_delete(m, 1L, axis = 1L), c(0, 2, 3, 4, 6, 7, 8, 10, 11), c(3L, 3L), "float32")
  expect_np(jnp_delete(m, c(0L, 2L), axis = 0L), 4:7, c(1L, 4L), "float32")
  expect_np(jnp_delete(m, as_jax(5L)), c(0:4, 6:11), 11L, "float32")
  expect_np(jit(function(x, i) jnp_delete(x, i, assume_unique_indices = TRUE))(x, c(1L, 3L, -2L)),
            c(0, 2, 4, 5, 6, 7, 9), 7L, "float32")
  expect_np(jit(function(x) jnp_delete(x, 3L))(x), c(0:2, 4:9), 9L, "float32")
  expect_error(jit(function(x, i) jnp_delete(x, i))(x, c(1L, 3L)), class = "jax_concretization_error")
  expect_error(jnp_delete(x, 10L), class = "jax_value_error")
  expect_error(jnp_delete(x, c(1L, 12L)), class = "jax_index_error")
  expect_error(jnp_delete(x, c(TRUE, FALSE)), class = "jax_value_error")
  expect_error(jnp_delete(x, c(0.5, 1)), class = "jax_value_error")
  y <- jnp_arange(5L, dtype = "float32")
  expect_np(jnp_insert(y, 2L, 99), c(0, 1, 99, 2, 3, 4), 6L, "float32")
  expect_np(jnp_insert(y, c(1L, 3L, 3L), c(-1, -2, -3)), c(0, -1, 1, 2, -2, -3, 3, 4), 8L, "float32")
  expect_np(jnp_insert(m, 1L, c(7, 8, 9), axis = 1L), c(0, 7, 1, 2, 3, 4, 8, 5, 6, 7, 8, 9, 9, 10, 11),
            c(3L, 5L), "float32")
  expect_np(jnp_insert(m, as_jax(array(1L, 1L)), c(7, 8, 9), axis = 1L),
            c(0, 7, 8, 9, 1, 2, 3, 4, 7, 8, 9, 5, 6, 7, 8, 7, 8, 9, 9, 10, 11), c(3L, 7L), "float32")
  expect_np(jnp_insert(m, 3L, 5, axis = 0L), c(0:11, 5, 5, 5, 5), c(4L, 4L), "float32")
  expect_np(jnp_insert(y, sl(0L, 4L, 2L), c(-1, -2)), c(-1, 0, 1, -2, 2, 3, 4), 7L, "float32")
  expect_np(jnp_insert(y, -1L, 50), c(0, 1, 2, 3, 50, 4), 6L, "float32")
  expect_np(jnp_insert(1:3, 1L, 7.5), c(1, 7, 2, 3), 4L, "int32")
  expect_np(jit(function(y, i, v) jnp_insert(y, i, v))(y, c(1L, 3L, 3L), c(-1, -2, -3)),
            c(0, -1, 1, 2, -2, -3, 3, 4), 8L, "float32")
  expect_error(jnp_insert(y, matrix(1L, 2, 2), 1), class = "jax_value_error")
  expect_error(jnp_insert(y, c(0.5, 1.5), 1), class = "jax_value_error")
  with_x64({
    check_grad_fd(function(v) jnp_sum(jnp_insert(v, c(1L, 3L), v[c(0L, 1L)]^2)^2), as_jax(c(1, 2, 3, 4), "float64"))
    check_grad_fd(function(v) jnp_sum(jnp_delete(v, c(1L, 3L))^3), as_jax(c(1, 2, 3, 4), "float64"))
  })
})

test_that("diagonal, diag, diagflat, diag_indices and fill_diagonal match the wheel", {
  skip_if_no_backend()
  m <- np_arr(0:11, c(3L, 4L), "float32")
  expect_np(jnp_diagonal(m), c(0, 5, 10), 3L, "float32")
  expect_np(jnp_diagonal(m, 1L), c(1, 6, 11), 3L, "float32")
  expect_np(jnp_diagonal(m, -1L), c(4, 9), 2L, "float32")
  t3 <- np_arr(0:23, c(2L, 3L, 4L), "float32")
  expect_np(jnp_diagonal(t3, 0L, 1L, 2L), c(0, 5, 10, 12, 17, 22), c(2L, 3L), "float32")
  expect_np(jnp_diagonal(t3, 1L, 0L, 2L), c(1, 14, 5, 18, 9, 22), c(3L, 2L), "float32")
  expect_identical(dim(jnp_diagonal(m, 5L)), 0L)
  expect_np(jit(jnp_diagonal, static_argnames = "offset")(m, offset = 1L), c(1, 6, 11), 3L, "float32")
  expect_error(jnp_diagonal(c(1, 2)), class = "jax_value_error")
  expect_np(jnp_diag(c(1, 2, 3)), c(1, 0, 0, 0, 2, 0, 0, 0, 3), c(3L, 3L), "float32")
  expect_np(jnp_diag(c(1, 2), k = 1L), c(0, 1, 0, 0, 0, 2, 0, 0, 0), c(3L, 3L), "float32")
  expect_np(jnp_diag(c(1, 2), k = -2L), c(rep(0, 8), 1, 0, 0, 0, 0, 2, 0, 0), c(4L, 4L), "float32")
  expect_np(jnp_diag(m, k = 1L), c(1, 6, 11), 3L, "float32")
  expect_np(jit(jnp_diag)(c(1, 2, 3)), c(1, 0, 0, 0, 2, 0, 0, 0, 3), c(3L, 3L), "float32")
  expect_error(jnp_diag(t3), class = "jax_value_error")
  expect_np(jnp_diagflat(np_arr(1:4, c(2L, 2L))), c(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4),
            c(4L, 4L), "int32")
  expect_np(jnp_diagflat(c(1, 2), 1L), c(0, 1, 0, 0, 0, 2, 0, 0, 0), c(3L, 3L), "float32")
  expect_np(jnp_diagflat(c(1, 2), -1L), c(0, 0, 0, 1, 0, 0, 0, 2, 0), c(3L, 3L), "float32")
  di <- jnp_diag_indices(3L, 3L)
  expect_length(di, 3L)
  expect_np(di[[3]], 0:2, 3L, "int32")
  expect_length(jnp_diag_indices_from(jnp_ones(c(2L, 2L))), 2L)
  expect_error(jnp_diag_indices(-1L), class = "jax_value_error")
  expect_error(jnp_diag_indices_from(jnp_ones(c(2L, 3L))), class = "jax_value_error")
  expect_error(jnp_diag_indices_from(jnp_ones(2L)), class = "jax_value_error")
  expect_np(jnp_fill_diagonal(m, 9, inplace = FALSE), c(9, 1, 2, 3, 4, 9, 6, 7, 8, 9, 9, 11), c(3L, 4L), "float32")
  expect_np(jnp_fill_diagonal(jnp_zeros(c(4L, 4L)), c(1, 2), inplace = FALSE),
            c(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2), c(4L, 4L), "float32")
  expect_np(jnp_fill_diagonal(jnp_zeros(c(2L, 2L, 2L), "int32"), c(5L, 6L, 7L), inplace = FALSE),
            c(5, 0, 0, 0, 0, 0, 0, 6), c(2L, 2L, 2L), "int32")
  expect_np(jit(function(a, v) jnp_fill_diagonal(a, v, inplace = FALSE))(m, 9), c(9, 1, 2, 3, 4, 9, 6, 7, 8, 9, 9, 11),
            c(3L, 4L), "float32")
  expect_error(jnp_fill_diagonal(m, 9), class = "jax_not_implemented_error")
  expect_error(jnp_fill_diagonal(m, 9, wrap = TRUE, inplace = FALSE), class = "jax_not_implemented_error")
  expect_error(jnp_fill_diagonal(jnp_ones(c(2L, 2L, 3L)), 1, inplace = FALSE), class = "jax_value_error")
  M <- np_arr(c(1, 2, 3, 4, 5, 6, 7, 8), c(2L, 2L, 2L), "float32")
  expect_same(vmap(jnp_diagonal)(M), stack_loop(jnp_diagonal, 2L, M))
  expect_same(vmap(jnp_diag)(m), stack_loop(jnp_diag, 3L, m))
  with_x64({
    check_grad_fd(function(x) jnp_sum(jnp_diag(x, 1L)^2), as_jax(c(1, -2, 3), "float64"))
    check_grad_fd(function(x) jnp_sum(jnp_diagonal(jnp_reshape(x, c(2L, 3L)), -1L)^2), as_jax(c(1, -2, 3, 4, 5, 6), "float64"))
    check_grad_fd(function(x) jnp_sum(jnp_diagflat(x, -1L) * 3), as_jax(c(1, -2), "float64"))
  })
})

test_that("diff and ediff1d match the wheel", {
  skip_if_no_backend()
  v <- c(1, 4, 9, 16, 25)
  expect_np(jnp_diff(v), c(3, 5, 7, 9), 4L, "float32")
  expect_np(jnp_diff(v, n = 2L), c(2, 2, 2), 3L, "float32")
  expect_np(jnp_diff(v, prepend = 0, append = c(30, 40)), c(1, 3, 5, 7, 9, 5, 10), 7L, "float32")
  m <- np_arr(0:11, c(3L, 4L), "float32")
  expect_np(jnp_diff(m, axis = 0L), rep(4, 8), c(2L, 4L), "float32")
  expect_np(jnp_diff(c(TRUE, TRUE, FALSE, TRUE)), c(FALSE, TRUE, TRUE), 3L, "bool")
  expect_np(jnp_diff(c(1L, 5L, 2L)), c(4, -3), 2L, "int32")
  expect_identical(dim(jnp_diff(c(1, 2), n = 3L)), 0L)
  expect_same(jnp_diff(v, n = 0L), as_jax(v))
  expect_np(jit(jnp_diff, static_argnames = "n")(v, n = 2L), c(2, 2, 2), 3L, "float32")
  expect_error(jnp_diff(v, n = -1L), class = "jax_value_error")
  expect_error(jnp_diff(as_jax(1)), class = "jax_value_error")
  expect_np(jnp_ediff1d(m), rep(1, 11), 11L, "float32")
  expect_np(jnp_ediff1d(c(1L, 3L, 6L), to_end = c(100.5, 200), to_begin = -9L), c(-9, 2, 3, 100, 200), 5L, "int32")
  expect_np(jit(jnp_ediff1d)(v), c(3, 5, 7, 9), 4L, "float32")
  expect_same(vmap(jnp_diff)(m), stack_loop(jnp_diff, 3L, m))
  with_x64({
    check_grad_fd(function(x) jnp_sum(jnp_diff(x, n = 2L)^2), as_jax(c(1, 4, 2, 8, 3), "float64"))
    check_grad_fd(function(x) jnp_sum(jnp_ediff1d(x, to_begin = 1)^2), as_jax(c(1, 4, 2), "float64"))
  })
})

test_that("digitize matches the wheel", {
  skip_if_no_backend()
  bins <- c(0, 1, 2.5, 4)
  x <- c(-1, 0, 0.5, 1, 2.5, 3, 4, 5)
  expect_np(jnp_digitize(x, bins), c(0, 1, 1, 2, 3, 3, 4, 4), 8L, "int32")
  expect_np(jnp_digitize(x, bins, right = TRUE), c(0, 0, 1, 1, 2, 3, 3, 4), 8L, "int32")
  expect_np(jnp_digitize(x, rev(bins)), c(4, 3, 3, 2, 1, 1, 0, 0), 8L, "int32")
  expect_np(jnp_digitize(x, rev(bins), right = TRUE), c(4, 4, 3, 3, 2, 1, 1, 0), 8L, "int32")
  expect_np(jnp_digitize(x, numeric(0)), rep(0, 8), 8L, "int32")
  expect_np(jit(jnp_digitize)(x, bins), c(0, 1, 1, 2, 3, 3, 4, 4), 8L, "int32")
  expect_np(jnp_digitize(x, bins, method = "compare_all"), c(0, 1, 1, 2, 3, 3, 4, 4), 8L, "int32")
  expect_error(jnp_digitize(x, jnp_ones(c(2L, 2L))), class = "jax_value_error")
  with_x64(expect_np(jnp_digitize(x, bins), c(0, 1, 1, 2, 3, 3, 4, 4), 8L, "int32"))
})

test_that("flip, fliplr and flipud match the wheel", {
  skip_if_no_backend()
  m <- np_arr(0:11, c(3L, 4L), "float32")
  expect_np(jnp_flip(m), 11:0, c(3L, 4L), "float32")
  expect_np(jnp_flip(m, 0L), c(8:11, 4:7, 0:3), c(3L, 4L), "float32")
  t3 <- np_arr(0:23, c(2L, 3L, 4L), "float32")
  expect_np(jnp_flip(t3, c(0L, 2L)), c(15:12, 19:16, 23:20, 3:0, 7:4, 11:8), c(2L, 3L, 4L), "float32")
  expect_np(jnp_fliplr(m), c(3:0, 7:4, 11:8), c(3L, 4L), "float32")
  expect_np(jnp_flipud(m), c(8:11, 4:7, 0:3), c(3L, 4L), "float32")
  expect_np(jit(jnp_flip)(m), 11:0, c(3L, 4L), "float32")
  expect_error(jnp_fliplr(c(1, 2)), class = "jax_value_error")
  with_x64(check_grad_fd(function(x) jnp_sum(jnp_flip(x) * c(1, 2, 3)), as_jax(c(1, 4, 2), "float64")))
})

test_that("fmax and fmin match the wheel", {
  skip_if_no_backend()
  a <- c(1, NaN, 3, NaN)
  b <- c(2, 1, NaN, NaN)
  expect_np(jnp_fmax(a, b), c(2, 1, 3, NaN), 4L, "float32")
  expect_np(jnp_fmin(a, b), c(1, 1, 3, NaN), 4L, "float32")
  expect_np(jnp_fmax(c(1L, 5L), 3L), c(3, 5), 2L, "int32")
  expect_np(jit(jnp_fmin)(a, b), c(1, 1, 3, NaN), 4L, "float32")
  with_x64(check_grad_fd(function(x) jnp_sum(jnp_fmax(x, c(0, 1, 2))^2), as_jax(c(1.5, -1, 2.5), "float64")))
})

test_that("frombuffer, fromstring, fromfunction, fromfile and fromiter match the wheel", {
  skip_if_no_backend()
  expect_np(jnp_frombuffer(as.raw(c(0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 64, 64)), dtype = "float32"),
            c(1.5, -2.25, 3), 3L, "float32")
  expect_np(jnp_frombuffer(as.raw(c(0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 2, 192))), c(1.5, -2.25),
            2L, "float32")
  expect_np(jnp_frombuffer(as.raw(c(1, 0, 254, 255, 44, 1, 0, 128)), dtype = "int16", count = 3L, offset = 2L),
            c(-2, 300, -32768), 3L, "int16")
  expect_np(jnp_frombuffer(as.raw(c(0, 40, 107, 238, 7, 0, 0, 0)), dtype = "uint32"), c(4e9, 7), 2L, "uint32")
  expect_np(jnp_frombuffer(as.raw(c(0, 62, 142, 134, 255, 123)), dtype = "float16"),
            c(1.5, -0.00010001659393310547, 65504), 3L, "float16")
  expect_np(jnp_frombuffer(as.raw(c(0, 1, 2, 0)), dtype = "bool"), c(FALSE, TRUE, TRUE, FALSE), 4L, "bool")
  expect_np(jnp_frombuffer(as.raw(c(0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 192, 0, 0, 0, 63)), dtype = "complex64"),
            c(1 + 2i, -3 + 0.5i), 2L, "complex64")
  expect_np(jnp_frombuffer(as.raw(c(192, 63, 64, 192)), dtype = "bfloat16"), c(1.5, -3), 2L, "bfloat16")
  expect_np(jnp_frombuffer(as.raw(c(0, 0, 192, 63)), dtype = "float32"), 1.5, 1L, "float32")
  with_x64(expect_np(jnp_frombuffer(as.raw(c(251, rep(255, 7), 20, 26, 153, 190, 28, 0, 0, 0)), dtype = "int64"),
                     c(-5, 123456789012), 2L, "int64"))
  expect_error(jnp_frombuffer(as.raw(1:3), dtype = "float32"), class = "jax_value_error")
  expect_error(jnp_frombuffer(as.raw(1:4), dtype = "float32", count = 2L), class = "jax_value_error")
  expect_error(jnp_frombuffer(c(1, 2)), class = "jax_type_error")
  expect_np(jnp_fromstring("1 2   3.5", sep = " "), c(1, 2, 3.5), 3L, "float32")
  expect_np(jnp_fromstring("1, 2,3", dtype = "int32", sep = ","), 1:3, 3L, "int32")
  expect_np(jnp_fromstring("1, 2,3", dtype = "int32", count = 2L, sep = ","), 1:2, 2L, "int32")
  expect_error(jnp_fromstring("1 2"), class = "jax_type_error")
  expect_error(jnp_fromstring("1 a", sep = " "), class = "jax_value_error")
  expect_error(jnp_fromstring("1 2", count = 3L, sep = " "), class = "jax_value_error")
  expect_np(jnp_fromfunction(function(i, j) i * 10 + j, c(2L, 3L)), c(0, 1, 2, 10, 11, 12), c(2L, 3L), "float32")
  expect_np(jnp_fromfunction(function(i, j) i - j, c(2L, 2L), dtype = "int32"), c(0, -1, 1, 0), c(2L, 2L), "int32")
  expect_np(jnp_fromfunction(function(i) i^2, 4L), c(0, 1, 4, 9), 4L, "float32")
  expect_np(jnp_fromfunction(function(i, j, scale) (i + j) * scale, c(2L, 2L), scale = 3), c(0, 3, 3, 6), c(2L, 2L),
            "float32")
  expect_error(jnp_fromfile("x"), class = "jax_not_implemented_error")
  expect_error(jnp_fromiter(1:3), class = "jax_not_implemented_error")
})

test_that("gcd and lcm (R/numpy_ufuncs.R) match the wheel", {
  skip_if_no_backend()
  skip_if_not(exists("jnp_gcd", mode = "function"), "jnp_gcd is defined in R/numpy_ufuncs.R")
  expect_np(jnp_gcd(c(12L, -18L, 0L, 7L, 0L), c(18L, 12L, 5L, 0L, 0L)), c(6, 6, 5, 7, 0), 5L, "int32")
  expect_np(jnp_gcd(np_arr(c(12L, 9L), c(2L, 1L)), c(4L, 6L, 15L)), c(4, 6, 3, 1, 3, 3), c(2L, 3L), "int32")
  g <- jnp_gcd(12L, 18L)
  expect_np(g, 6, integer(0), "int32")
  expect_np(jnp_lcm(c(4L, -6L, 0L, 7L), c(6L, 4L, 5L, 3L)), c(12, 12, 0, 21), 4L, "int32")
  expect_np(jnp_lcm(as_jax(c(4L, 6L), "uint8"), as_jax(c(6L, 10L), "uint8")), c(12, 30), 2L, "uint8")
  expect_np(jit(jnp_gcd)(c(12L, -18L), c(18L, 12L)), c(6, 6), 2L, "int32")
  expect_np(jit(jnp_lcm)(c(4L, -6L), c(6L, 4L)), c(12, 12), 2L, "int32")
  A <- np_arr(c(12L, 18L, 8L, 9L), c(2L, 2L))
  B <- np_arr(c(8L, 12L, 6L, 6L), c(2L, 2L))
  expect_same(vmap(jnp_gcd)(A, B), stack_loop(jnp_gcd, 2L, A, B))
  expect_error(jnp_gcd(c(1.5, 2), 2L), class = "jax_value_error")
  expect_error(jnp_lcm(c(1.5, 2), 2L), class = "jax_value_error")
  with_x64(expect_np(jnp_gcd(c(12L, -18L), c(18L, 12L)), c(6, 6), 2L, "int32"))
})

test_that("gradient matches the wheel", {
  skip_if_no_backend()
  f <- c(1, 2, 4, 7, 11)
  expect_np(jnp_gradient(f), c(1, 1.5, 2.5, 3.5, 4), 5L, "float32")
  expect_np(jnp_gradient(f, 2), c(0.5, 0.75, 1.25, 1.75, 2), 5L, "float32")
  xs <- c(0, 1, 1.5, 3.5, 4)
  expect_np(jnp_gradient(f, xs), c(1, 3.000000238418579, 3.5, 6.700000762939453, 8), 5L, "float32")
  F <- np_arr(c(1, 2, 6, 3, 4, 5), c(2L, 3L), "float32")
  g <- jnp_gradient(F)
  expect_length(g, 2L)
  expect_np(g[[1]], c(2, 2, -1, 2, 2, -1), c(2L, 3L), "float32")
  expect_np(g[[2]], c(1, 2.5, 4, 1, 1, 1), c(2L, 3L), "float32")
  g <- jnp_gradient(F, 2, c(0, 1, 3))
  expect_np(g[[1]], c(1, 1, -0.5, 1, 1, -0.5), c(2L, 3L), "float32")
  expect_np(g[[2]], c(1, 1.3333333730697632, 2, 1, 0.8333333134651184, 0.5), c(2L, 3L), "float32")
  expect_np(jnp_gradient(F, axis = 1L), c(1, 2.5, 4, 1, 1, 1), c(2L, 3L), "float32")
  expect_np(jnp_gradient(F, axis = -2L), c(2, 2, -1, 2, 2, -1), c(2L, 3L), "float32")
  expect_np(jnp_gradient(c(1L, 4L, 9L)), c(3, 4, 5), 3L, "float32")
  expect_np(jit(function(f, x) jnp_gradient(f, x))(f, xs), c(1, 3.000000238418579, 3.5, 6.700000762939453, 8),
            5L, "float32")
  expect_np(vmap(jnp_gradient)(np_arr(c(1, 2, 4, 0, 3, 9), c(2L, 3L), "float32")), c(1, 1.5, 2, 3, 4.5, 6),
            c(2L, 3L), "float32")
  expect_error(jnp_gradient(as_jax(array(1, 1L))), class = "jax_value_error")
  expect_error(jnp_gradient(f, c(1, 2)), class = "jax_value_error")
  expect_error(jnp_gradient(F, 1, 2, 3), class = "jax_type_error")
  expect_error(jnp_gradient(f, edge_order = 2L), class = "jax_not_implemented_error")
  expect_error(jnp_gradient(f, jnp_ones(c(5L, 1L))), class = "jax_value_error")
  with_x64({
    expect_np(grad(function(f) jnp_sum(jnp_gradient(f, c(0, 1, 1.5, 3.5, 4))^2))(as_jax(c(1, 2, 4, 7, 11), "float64")),
              c(-4, -15.2, 17.16, -51.4, 53.44), 5L, "float64", tolerance = 1e-10)
    check_grad_fd(function(x) jnp_sum(jnp_gradient(c(1, 2, 4, 7, 11), x)^2), as_jax(c(0, 1, 1.5, 3.5, 4), "float64"))
  })
})

test_that("histogram functions match the wheel", {
  skip_if_no_backend()
  a <- c(0.5, 1.5, 1.5, 2, 3.5, 4, 0)
  expect_np(jnp_histogram_bin_edges(a, bins = 4L), c(0, 1, 2, 3, 4), 5L, "float32")
  expect_np(jnp_histogram_bin_edges(a, bins = 3L, range = c(1, 2)), c(1, 1.3333332538604736, 1.6666667461395264, 2),
            4L, "float32")
  expect_np(jnp_histogram_bin_edges(a, bins = c(0L, 1L, 4L)), c(0, 1, 4), 3L, "float32")
  expect_np(jnp_histogram_bin_edges(c(2, 2), bins = 2L), c(1.5, 2, 2.5), 3L, "float32")
  expect_error(jnp_histogram_bin_edges(a, bins = "auto"), class = "jax_not_implemented_error")
  expect_error(jnp_histogram_bin_edges(a, range = c(1, 2, 3)), class = "jax_value_error")
  h <- jnp_histogram(a, bins = 4L)
  expect_np(h[[1]], c(2, 2, 1, 2), 4L, "float32")
  expect_np(h[[2]], c(0, 1, 2, 3, 4), 5L, "float32")
  expect_np(jnp_histogram(a, bins = c(0, 1, 2, 5), weights = jnp_arange(7L, dtype = "float32"))[[1]], c(6, 3, 12),
            3L, "float32")
  expect_np(jnp_histogram(a, bins = 4L, density = TRUE)[[1]],
            c(0.2857142984867096, 0.2857142984867096, 0.1428571492433548, 0.2857142984867096), 4L, "float32")
  h <- jnp_histogram(c(1L, 2L, 2L, 3L), bins = 2L)
  expect_np(h[[1]], c(1, 3), 2L, "float32")
  expect_np(h[[2]], c(1, 2, 3), 3L, "float32")
  h <- jit(function(a) jnp_histogram(a, bins = 4L))(a)
  expect_np(h[[1]], c(2, 2, 1, 2), 4L, "float32")
  expect_error(jnp_histogram(a, weights = c(1, 2)), class = "jax_value_error")
  x <- c(0, 1, 1, 2, 3)
  y <- c(0, 0, 1, 2, 2)
  h <- jnp_histogram2d(x, y, bins = 2L)
  expect_np(h[[1]], c(2, 1, 0, 2), c(2L, 2L), "int32")
  expect_np(h[[2]], c(0, 1.5, 3), 3L, "float32")
  expect_np(h[[3]], c(0, 1, 2), 3L, "float32")
  h <- jnp_histogram2d(x, y, bins = list(2L, 3L), range = list(c(0, 4), NULL), density = TRUE)
  expect_np(h[[1]], c(0.3, 0.15, 0, 0, 0, 0.3), c(2L, 3L), "float32")
  expect_np(h[[2]], c(0, 2, 4), 3L, "float32")
  expect_np(h[[3]], c(0, 0.6666666865348816, 1.3333333730697632, 2), 4L, "float32")
  h <- jnp_histogram2d(x, y, bins = c(0, 1, 2, 3), weights = c(1, 2, 3, 4, 5))
  expect_np(h[[1]], c(1, 0, 0, 2, 3, 0, 0, 0, 9), c(3L, 3L), "float32")
  s <- jnp_stack(list(as_jax(x), as_jax(y), as_jax(x + y)), axis = 1L)
  h <- jnp_histogramdd(s, bins = 2L)
  expect_np(h[[1]], c(2, 0, 1, 0, 0, 0, 0, 2), c(2L, 2L, 2L), "int32")
  expect_length(h[[2]], 3L)
  expect_np(h[[2]][[3]], c(0, 2.5, 5), 3L, "float32")
  h <- jnp_histogramdd(jnp_stack(list(as_jax(x), as_jax(y)), axis = 1L), bins = 2L, weights = c(1, 2, 3, 4, 5),
                       density = TRUE)
  expect_np(h[[1]], c(0.13333336, 0.13333336, 0, 0.40000004), c(2L, 2L), "float32")
  expect_error(jnp_histogramdd(s, bins = list(2L, 2L)), class = "jax_value_error")
  expect_error(jnp_histogramdd(s, weights = c(1, 2)), class = "jax_value_error")
  expect_error(jnp_histogramdd(s, range = list(c(0, 1))), class = "jax_value_error")
  with_x64(check_grad_fd(function(w) jnp_sum(jnp_histogram(a, bins = 3L, weights = w)[[1]]^2),
                         as_jax(c(1, 2, 3, 4, 5, 6, 7), "float64")))
})

test_that("i0 matches the wheel", {
  skip_if_no_backend()
  skip_if_not(exists("bessel_i0e_p", envir = asNamespace("jaxR")), "lax_special.R is not in this build")
  x <- c(0, 0.5, -1, 3, 20)
  expect_np(jnp_i0(x), c(1.0000001192092896, 1.063483476638794, 1.2660659551620483, 4.880792617797852, 43558276),
            5L, "float32", tolerance = 1e-6)
  expect_np(jit(jnp_i0)(x), c(1.0000001192092896, 1.063483476638794, 1.2660659551620483, 4.880792617797852, 43558276),
            5L, "float32", tolerance = 1e-6)
  xs <- c(0, 0.5, -1, 3)
  expect_np(vmap(grad(jnp_i0))(xs), c(0, 0.2578943371772766, -0.5651590824127197, 3.9533705711364746), 4L, "float32")
  expect_np(vmap(grad(grad(jnp_i0)))(xs), c(0.5, 0.547694742679596, 0.7009068131446838, 3.563002347946167), 4L, "float32")
  expect_np(jnp_i0(1L), 1.2660659551620483, integer(0), "float32")
  expect_error(jnp_i0(1i), class = "jax_value_error")
  with_x64({
    expect_np(jnp_i0(x), c(1, 1.0634833707413236, 1.2660658777520084, 4.880792585865024, 43558282.559553534), 5L,
              "float64", tolerance = 1e-14)
    check_grad_fd(function(x) jnp_sum(jnp_i0(x)), as_jax(c(0.5, -1, 3), "float64"))
  })
})

test_that("indices and ix_ match the wheel", {
  skip_if_no_backend()
  expect_np(jnp_indices(c(2L, 3L)), c(0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2), c(2L, 2L, 3L), "int32")
  s <- jnp_indices(c(2L, 3L), sparse = TRUE)
  expect_np(s[[1]], c(0, 1), c(2L, 1L), "int32")
  expect_np(s[[2]], 0:2, c(1L, 3L), "int32")
  expect_np(jnp_indices(2L, dtype = "float32"), c(0, 1), c(1L, 2L), "float32")
  expect_identical(dim(jnp_indices(integer(0))), 0L)
  expect_identical(dtype(jnp_indices(integer(0))), "int32")
  with_x64(expect_identical(dtype(jnp_indices(c(2L, 3L))), "int64"))
  ix <- jnp_ix_(c(0L, 2L), c(1L, 3L, 4L))
  expect_np(ix[[1]], c(0, 2), c(2L, 1L), "int32")
  expect_np(ix[[2]], c(1, 3, 4), c(1L, 3L), "int32")
  ix <- jnp_ix_(c(0, 2), integer(0))
  expect_np(ix[[1]], c(0, 2), c(2L, 1L), "int32")
  expect_identical(dim(ix[[2]]), c(1L, 0L))
  m <- np_arr(0:19, c(4L, 5L), "float32")
  expect_np(do.call(function(i, j) m[i, j], jnp_ix_(c(0L, 2L), c(1L, 3L, 4L))), c(1, 3, 4, 11, 13, 14),
            c(2L, 3L), "float32")
  expect_error(jnp_ix_(jnp_ones(c(2L, 2L))), class = "jax_value_error")
  expect_error(jnp_ix_(c(TRUE, FALSE)), class = "jax_not_implemented_error")
})

test_that("interp matches the wheel", {
  skip_if_no_backend()
  xp <- c(0, 1, 2.5, 4)
  fp <- c(1, 3, 2, 0)
  x <- c(-1, 0, 0.5, 2, 2.5, 3.9, 5)
  expect_np(jnp_interp(x, xp, fp), c(1, 1, 2, 2.3333332538604736, 2, 0.1333332061767578, 0), 7L, "float32")
  expect_np(jnp_interp(x, xp, fp, left = -10, right = 10),
            c(-10, 1, 2, 2.3333332538604736, 2, 0.1333332061767578, 10), 7L, "float32")
  expect_np(jnp_interp(x, xp, fp, left = "extrapolate", right = "extrapolate"),
            c(-1, 1, 2, 2.3333332538604736, 2, 0.1333332061767578, -1.3333332538604736), 7L, "float32")
  expect_np(jnp_interp(c(-1, 0.5, 5, 370), c(350, 10, 90, 180), c(1, 2, 3, 4), period = 360),
            c(1.4500000476837158, 1.524999976158142, 1.75, 2), 4L, "float32")
  expect_np(jnp_interp(c(1L, 2L), c(0L, 4L), c(0L, 8L)), c(2, 4), 2L, "float32")
  expect_np(jnp_interp(c(1, 1.5), c(0, 1, 1, 2), c(0, 1, 5, 6)), c(5, 5.5), 2L, "float32")
  expect_np(jit(jnp_interp)(x, xp, fp), c(1, 1, 2, 2.3333332538604736, 2, 0.1333332061767578, 0), 7L, "float32")
  expect_np(vmap(function(x) jnp_interp(x, xp, fp))(np_arr(c(0.5, 2, 3, -1), c(2L, 2L), "float32")),
            c(2, 2.3333333, 1.3333333, 1), c(2L, 2L), "float32")
  expect_error(jnp_interp(x, xp, fp[1:3]), class = "jax_value_error")
  expect_error(jnp_interp(x, xp, fp, left = "bogus"), class = "jax_value_error")
  expect_error(jnp_interp(1i, xp, fp), class = "jax_value_error")
  expect_error(jnp_interp(x, xp, fp, period = c(1, 2)), class = "jax_value_error")
  with_x64({
    xp64 <- as_jax(xp, "float64")
    fp64 <- as_jax(fp, "float64")
    expect_np(grad(function(x) jnp_sum(jnp_interp(x, xp64, fp64)^2))(as_jax(c(0.5, 2, 3), "float64")),
              c(8, -3.111111111111111, -3.555555555555556), 3L, "float64")
    expect_np(grad(function(f) jnp_sum(jnp_interp(c(0.5, 2, 3), xp64, f)^2))(fp64),
              c(2, 3.555555555555556, 4.888888888888889, 0.888888888888889), 4L, "float64")
    check_grad_fd(function(x) jnp_sum(jnp_interp(x, xp64, fp64)^2), as_jax(c(0.5, 2, 3), "float64"))
  })
})

test_that("iscomplex and isreal match the wheel", {
  skip_if_no_backend()
  z <- c(1 + 1i, 2 + 0i, 0i)
  expect_np(jnp_iscomplex(z), c(TRUE, FALSE, FALSE), 3L, "bool")
  expect_np(jnp_isreal(z), c(FALSE, TRUE, TRUE), 3L, "bool")
  expect_np(jnp_iscomplex(c(1, 2)), c(FALSE, FALSE), 2L, "bool")
  expect_np(jnp_isreal(1:2), c(TRUE, TRUE), 2L, "bool")
  expect_np(jit(jnp_iscomplex)(z), c(TRUE, FALSE, FALSE), 3L, "bool")
})

test_that("kron matches the wheel", {
  skip_if_no_backend()
  expect_np(jnp_kron(np_arr(1:4, c(2L, 2L)), np_arr(c(0L, 1L, 1L, 0L), c(2L, 2L))),
            c(0, 1, 0, 2, 1, 0, 2, 0, 0, 3, 0, 4, 3, 0, 4, 0), c(4L, 4L), "int32")
  expect_np(jnp_kron(c(1, 10), np_arr(c(1, 2, 3, 4), c(2L, 2L), "float32")), c(1, 2, 10, 20, 3, 4, 30, 40),
            c(2L, 4L), "float32")
  expect_np(jnp_kron(2, c(1, 2)), c(2, 4), 2L, "float32")
  expect_np(jnp_kron(as_jax(2, "float32"), as_jax(3, "float32")), 6, integer(0), "float32")
  expect_identical(dim(jnp_kron(jnp_ones(c(2L, 1L, 2L)), jnp_ones(c(2L, 2L)))), c(2L, 2L, 4L))
  expect_np(jit(jnp_kron)(c(1, 10), c(1, 2)), c(1, 2, 10, 20), 4L, "float32")
  A <- np_arr(1:4, c(2L, 2L), "float32")
  expect_same(vmap(function(a) jnp_kron(a, c(1, -1)))(A), stack_loop(function(a) jnp_kron(a, c(1, -1)), 2L, A))
  with_x64(check_grad_fd(function(a) jnp_sum(jnp_kron(a, c(1, 2, 3))^2), as_jax(c(0.5, -1), "float64")))
})

test_that("logspace and geomspace match the wheel bit for bit", {
  skip_if_no_backend()
  expect_bits(jnp_logspace(0, 2, 5L), c(0x1p+0, 0x1.94c584p+1, 0x1.4p+3, 0x1.f9f6e4p+4, 0x1.9p+6), 5L, "float32")
  expect_bits(jnp_logspace(0, 3, 4L, base = 2), c(1, 2, 4, 8), 4L, "float32")
  expect_bits(jnp_logspace(1, 2, 4L, endpoint = FALSE), c(0x1.4p+3, 0x1.1c8654p+4, 0x1.f9f6e4p+4, 0x1.c1df8p+5),
              4L, "float32")
  expect_bits(jnp_logspace(c(0, 1), c(1, 3), 3L, axis = 1L),
              c(0x1p+0, 0x1.94c584p+1, 0x1.4p+3, 0x1.4p+3, 0x1.9p+6, 0x1.f4p+9), c(2L, 3L), "float32")
  expect_np(jnp_logspace(0L, 3L, 4L, dtype = "int32"), c(1, 10, 100, 1000), 4L, "int32")
  # Under an outer jit the plugin may fuse pow differently from jaxlib (last bit).
  expect_np(jit(function(a, b) jnp_logspace(a, b, 5L))(0, 2), c(1, 0x1.94c584p+1, 10, 0x1.f9f6e4p+4, 100), 5L, "float32")
  expect_bits(jnp_geomspace(1, 1000, 4L), c(1, 10, 100, 1000), 4L, "float32")
  expect_bits(jnp_geomspace(2, 50, 5L), c(2, 0x1.1e377ap+2, 10, 0x1.65c55ap+4, 0x1.900004p+5), 5L, "float32")
  expect_bits(jnp_geomspace(-1, -100, 3L), c(-1, -10, -100), 3L, "float32")
  expect_bits(jnp_geomspace(1, 16, 4L, endpoint = FALSE), c(1, 2, 0x1.000002p+2, 8), 4L, "float32")
  expect_bits(jnp_geomspace(c(1, 2), c(8, 32), 3L, axis = -1L), c(1, 0x1.6a09e6p+1, 8, 2, 8, 0x1.fffffep+4),
              c(2L, 3L), "float32")
  expect_np(jnp_geomspace(1i, 16i, 3L), c(1i, 4.0000005i, 16.000002i), 3L, "complex64")
  expect_np(jnp_geomspace(1L, 1000L, 4L, dtype = "int32"), c(1, 10, 100, 1000), 4L, "int32")
  expect_error(jnp_logspace(0, 1, -1L), class = "jax_value_error")
  expect_error(jit(function(n) jnp_logspace(0, 1, n))(3L), class = "jax_concretization_error")
  with_x64({
    expect_bits(jnp_logspace(0, 2, 5L), c(1, 0x1.94c583ada5b53p+1, 10, 0x1.f9f6e4990f227p+4, 100), 5L, "float64")
    expect_bits(jnp_geomspace(2, 50, 5L), c(2, 0x1.1e3779b97f4a8p+2, 10, 0x1.65c55827df1d1p+4, 0x1.8ffffffffffffp+5),
                5L, "float64")
    expect_identical(dtype(jnp_logspace(as_jax(0L, "int32"), as_jax(2L, "int32"), 3L)), "float32")
    expect_identical(dtype(jnp_logspace(0L, 2L, 3L)), "float64")
    check_grad_fd(function(s) jnp_sum(jnp_geomspace(s, 50, 4L)), as_jax(2, "float64"))
    check_grad_fd(function(s) jnp_sum(jnp_logspace(s, 1, 4L)), as_jax(0.5, "float64"))
  })
})

test_that("the private linspace port matches the wheel bit for bit", {
  skip_if_no_backend()
  lin <- jaxR:::npa_linspace
  expect_bits(lin(0.1, 1.3, 7L), c(0x1.99999ap-4, 0x1.333334p-2, 0x1p-1, 0x1.666666p-1, 0x1.ccccccp-1, 0x1.19999ap+0,
                                   0x1.4cccccp+0), 7L, "float32")
  r <- lin(0, 1, 5L, endpoint = FALSE, retstep = TRUE)
  expect_np(r[[2]], 0.2, integer(0), "float32")
  expect_identical(dim(lin(c(0, 1), 2, 0L)), c(0L, 2L))
  expect_np(lin(2, 3, 1L), 2, 1L, "float32")
})

test_that("jit agrees with eager evaluation for the remaining functions", {
  skip_if_no_backend()
  a <- np_arr(0:23, c(2L, 3L, 4L), "float32")
  x <- c(0, 1, 1, 2, 3)
  y <- c(0, 0, 1, 2, 2)
  cases <- list(
    list(function(a) jnp_apply_over_axes(function(x, ax) jnp_sum(x, axis = ax), a, c(0L, 2L)), list(a)),
    list(function(p, q) jnp_array_equiv(p, q), list(1:2, np_arr(c(1L, 2L, 1L, 3L), c(2L, 2L)))),
    list(function(a) jnp_dsplit(a, 2L)[[2]], list(a)),
    list(function(a) jnp_hsplit(a, list(1L))[[2]], list(a)),
    list(function(p, q) jnp_dstack(list(p, q)), list(c(1, 2), c(3, 4))),
    list(function(p, q) jnp_concat(list(p, q), axis = NULL), list(jnp_ones(c(1L, 2L)), jnp_zeros(2L))),
    list(jnp_copy, list(c(1, 2))),
    list(function(v) jnp_diagflat(v, -1L), list(c(1, 2))),
    list(jnp_fmax, list(c(1, NaN, 3, NaN), c(2, 1, NaN, NaN))),
    list(function(x, y) jnp_histogram2d(x, y, bins = 2L)[[1]], list(x, y)),
    list(function(s) jnp_histogramdd(s, bins = 2L)[[2]][[1]], list(jnp_stack(list(as_jax(x), as_jax(y)), axis = 1L))),
    list(function(v) jnp_histogram_bin_edges(v, bins = 3L), list(x)),
    list(function(p, q) jnp_ix_(p, q)[[2]], list(c(0L, 2L), c(1L, 3L, 4L))),
    list(jnp_isreal, list(c(1 + 1i, 2 + 0i))),
    list(function(p, q) jnp_geomspace(p, q, 4L), list(1, 1000)),
    list(function(c, v) jnp_extract(c, v, size = 2L), list(c(TRUE, FALSE, TRUE), c(1, 2, 3))),
    list(function(v) jnp_delete(v, sl(1L, 3L)), list(c(1, 2, 3, 4))),
    list(function(v) jnp_insert(v, sl(0L, 4L, 2L), c(-1, -2)), list(c(1, 2, 3, 4))),
    list(function(v) jnp_atleast_1d(v), list(1)),
    list(function(v) jnp_angle(v), list(c(1i, -1)))
  )
  for (cs in cases) {
    expect_same(do.call(jit(cs[[1]]), cs[[2]]), do.call(cs[[1]], lapply(cs[[2]], as_jax)))
  }
})

test_that("second order derivatives match finite differences", {
  skip_if_no_backend()
  with_x64({
    x <- as_jax(c(0.5, -1, 2), "float64")
    check_grad_fd(function(x) jnp_sum(grad(function(y) jnp_sum(jnp_cross(y, y^2)^2))(x)), x, tolerance = 1e-4)
    check_grad_fd(function(x) jnp_sum(grad(function(y) jnp_sum(jnp_kron(y, y)^2))(x)), x, tolerance = 1e-4)
    check_grad_fd(function(f) jnp_sum(grad(function(g) jnp_sum(jnp_gradient(g)^3))(f)), as_jax(c(1, 2, 4, 7), "float64"),
                  tolerance = 1e-4)
  })
})

test_that("weak types follow the wheel for scalar inputs", {
  skip_if_no_backend()
  a1 <- function(v) jnp_atleast_1d(v)
  weak <- list(jnp_clip(2.5, 0, 1), jnp_clip(5L, 0L, 3L), jnp_fmax(1, 2), jnp_around(2.567, 1L), jnp_around(3L),
               jnp_copy(2), jnp_append(1, 2), jnp_kron(2, 3), jnp_diff(a1(1.5)), jnp_ediff1d(a1(1)), jnp_flip(1),
               jnp_atleast_3d(1), jnp_block(list(1, 2)), jnp_diag(a1(2)), jnp_extract(TRUE, a1(2)),
               jnp_insert(a1(1), 0L, 2), jnp_delete(a1(1), 0L), jnp_hstack(list(1, 2)), jnp_concat(list(a1(1), a1(2))))
  for (x in weak) expect_true(is_weak(x))
  strong <- list(jnp_clip(as_jax(2.5, "float32"), 0, 1), jnp_fmin(1, as_jax(2, "float32")), jnp_diff(c(1, 2), prepend = 0),
                 jnp_interp(1.5, c(0, 2), c(0, 4)), jnp_logspace(0, 1, 3L), jnp_angle(1), jnp_isclose(1, 1),
                 jnp_gradient(c(1, 2, 4), 2), jnp_choose(0L, list(1, 2), mode = "clip"), jnp_cross(c(1, 0, 0), c(0, 1, 0)),
                 jnp_digitize(1.5, c(0, 1, 2)), jnp_fill_diagonal(jnp_zeros(c(2L, 2L)), 1, inplace = FALSE))
  for (x in strong) expect_false(is_weak(x))
  with_x64({
    # int32 and bool inputs become float32 even with x64 enabled, as in the wheel.
    expect_identical(dtype(jnp_gradient(c(1L, 4L, 9L))), "float32")
    expect_identical(dtype(jnp_interp(c(1L, 2L), c(0L, 4L), c(0L, 8L))), "float32")
    expect_identical(dtype(jnp_histogram(c(1L, 2L, 2L, 3L), bins = 2L)[[2]]), "float32")
    expect_identical(dtype(jnp_histogramdd(matrix(c(1L, 2L, 2L, 3L), 2, byrow = TRUE), bins = 2L)[[2]][[1]]), "float32")
    expect_identical(dtype(jnp_logspace(as_jax(0L, "int32"), as_jax(1L, "int32"), 3L)), "float32")
  })
})
