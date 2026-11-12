# Tests for NumPy style indexing (`[`, `[<-`), the at_* update functions, take and put
# along an axis and the segment reductions (R/numpy_indexing.R). Expected values were
# generated with the jax 0.12.0.1 wheel.

# A jax array from values in NumPy (row-major) order.
np_arr <- function(values, shape = length(values), dtype = NULL) {
  data <- if (length(shape) > 1L) aperm(array(values, rev(shape))) else array(values, shape)
  as_jax(data, dtype)
}

# Compare with expected values given in NumPy (row-major) order.
expect_np <- function(object, values, shape = length(values), dtype = NULL) {
  expected <- if (length(shape) > 1L) aperm(array(values, rev(shape))) else values
  expect_jax_equal(object, expected, dtype = dtype, shape = shape)
}

expect_same <- function(got, want) {
  expect_identical(dim(got), dim(want))
  expect_identical(dtype(got), dtype(want))
  expect_equal(as.vector(as.array(got)), as.vector(as.array(want)), tolerance = 1e-6)
}

with_x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)

row_of <- function(a, k) lax_index_in_dim(a, k, 0L, keepdims = FALSE)

# Apply f to each slice along axis 0 of the arguments and stack the results.
stack_loop <- function(f, n, ...) {
  args <- list(...)
  jnp_stack(lapply(seq_len(n) - 1L, function(k) do.call(f, lapply(args, row_of, k = k))))
}

indexing_fixtures <- function(env = parent.frame()) {
  force(env)
  local({
    v <- np_arr(c(0, 10, 20, 30, 40), 5L, "float32")
    x <- np_arr(0:23, c(2L, 3L, 4L), "float32")
    m <- np_arr(0:11, c(3L, 4L))
    f3 <- np_arr(c(1, 2, 3), 3L, "float32")
    u2 <- np_arr(c(4, 5), 2L, "float32")
    X <- np_arr(1:6, c(2L, 3L), "float32")
    i3 <- np_arr(c(0L, 5L, -1L), c(3L, 1L))
    i34 <- np_arr(c(3L, 0L, 1L, 2L, 0L, 0L, 0L, 0L, 1L, 1L, 2L, 2L), c(3L, 4L))
    i_x <- np_arr(c(2L, 0L, 1L, 1L, 0L, 0L, 2L, 2L), c(2L, 1L, 4L))
    d6 <- np_arr(0:5, 6L, "float32")
    d62 <- np_arr(0:11, c(6L, 2L), "float32")
    ids <- np_arr(c(0L, 0L, 1L, 1L, 3L, 3L), 6L)
  }, envir = env)
}

test_that("acceptance gate: gradients, jit and vmap through x[c(0, 0, 2)]", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2, 3))
  f <- function(x) sum(x[c(0, 0, 2)])
  expect_np(f(x), 5, integer(0), "float32")
  expect_np(grad(f)(x), c(2, 0, 1), 3L, "float32")
  expect_np(jit(grad(f))(x), c(2, 0, 1), 3L, "float32")
  expect_np(grad(jit(f))(x), c(2, 0, 1), 3L, "float32")
  X <- np_arr(1:6, c(2L, 3L), "float32")
  expect_np(vmap(f)(X), c(5, 14), 2L, "float32")
  expect_np(vmap(grad(f))(X), c(2, 0, 1, 2, 0, 1), c(2L, 3L), "float32")
  expect_np(grad(function(X) sum(vmap(f)(X)))(X), c(2, 0, 1, 2, 0, 1), c(2L, 3L), "float32")
  expect_np(jit(vmap(grad(f)))(X), c(2, 0, 1, 2, 0, 1), c(2L, 3L), "float32")
  # Second order gradient through gather.
  h <- function(a) sum(grad(function(b) sum(b[c(0, 0, 2)]^3))(a))
  expect_np(grad(h)(x), c(12, 0, 18), 3L, "float32")
  expect_np(jit(grad(h))(x), c(12, 0, 18), 3L, "float32")
  # vmap of indexing with batched indices.
  I <- np_arr(c(0L, 1L, 2L, 2L, 2L, 0L), c(2L, 3L))
  expect_np(vmap(function(i) x[i])(I), c(1, 2, 3, 3, 3, 1), c(2L, 3L), "float32")
  expect_np(vmap(function(a, i) a[i])(X, I), c(1, 2, 3, 6, 6, 4), c(2L, 3L), "float32")
  expect_np(vmap(function(a, i) a[i])(X, as_jax(c(2L, 0L))), c(3, 4), 2L, "float32")
  # Out of bounds indices clamp when reading, as in the wheel.
  expect_np(x[c(0, 7, -7)], c(1, 3, 1), 3L, "float32")
  expect_np(x[7], 3, integer(0), "float32")
  expect_np(vmap(function(a, i) a[i])(X, as_jax(c(7L, -7L))), c(3, 4), 2L, "float32")
  expect_np(grad(function(a) sum(a[c(0, 7)]^2))(x), c(2, 0, 0), 3L, "float32")
})

test_that("basic indexing matches the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(v[1], 10, integer(0), "float32")
  expect_np(v[-1], 40, integer(0), "float32")
  expect_np(x[1], c(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), c(3L, 4L), "float32")
  expect_np(x[1, 2], c(20, 21, 22, 23), 4L, "float32")
  expect_np(x[1, 2, 3], 23, integer(0), "float32")
  expect_np(x[, 1], c(4, 5, 6, 7, 16, 17, 18, 19), c(2L, 4L), "float32")
  expect_np(x[, , 0], c(0, 4, 8, 12, 16, 20), c(2L, 3L), "float32")
  expect_np(x[ellipsis, 0], c(0, 4, 8, 12, 16, 20), c(2L, 3L), "float32")
  expect_np(x[-1, ellipsis, -1], c(15, 19, 23), 3L, "float32")
  expect_np(v[], c(0, 10, 20, 30, 40), 5L, "float32")
  expect_np(x[newaxis],
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23), c(1L, 2L, 3L, 4L), "float32")
  expect_np(x[, newaxis, 1], c(4, 5, 6, 7, 16, 17, 18, 19), c(2L, 1L, 4L), "float32")
  expect_np(x[0, ellipsis, newaxis],
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), c(3L, 4L, 1L), "float32")
  expect_np(v[newaxis, , newaxis], c(0, 10, 20, 30, 40), c(1L, 5L, 1L), "float32")
  expect_np(v[sl(1, 4)], c(10, 20, 30), 3L, "float32")
  expect_np(v[sl(-3)], c(20, 30, 40), 3L, "float32")
  expect_np(v[sl(stop = 2)], c(0, 10), 2L, "float32")
  expect_np(v[sl(step = 2)], c(0, 20, 40), 3L, "float32")
  expect_np(v[sl(step = -1)], c(40, 30, 20, 10, 0), 5L, "float32")
  expect_np(v[sl(4, 1, -1)], c(40, 30, 20), 3L, "float32")
  expect_np(v[sl(10)], numeric(0), 0L, "float32")
  expect_np(v[sl(-100, 2)], c(0, 10), 2L, "float32")
  expect_np(v[sl(3, 1)], numeric(0), 0L, "float32")
  expect_np(v[sl(5, 0, -2)], c(40, 20), 2L, "float32")
  expect_np(v[sl(step = -2)], c(40, 20, 0), 3L, "float32")
  expect_np(v[sl(1, -1)], c(10, 20, 30), 3L, "float32")
  expect_np(v[sl(NULL, NULL, 3)], c(0, 30), 2L, "float32")
  expect_np(v[sl(-1, NULL, -3)], c(40, 10), 2L, "float32")
  expect_np(x[sl(1), sl(step = 2)], c(12, 13, 14, 15, 20, 21, 22, 23), c(1L, 2L, 4L), "float32")
  expect_np(x[, sl(step = -1), sl(1, NULL, 2)],
            c(9, 11, 5, 7, 1, 3, 21, 23, 17, 19, 13, 15), c(2L, 3L, 2L), "float32")
  expect_np(x[sl(step = -1), sl(1, 3), sl(step = -2)],
            c(19, 17, 23, 21, 7, 5, 11, 9), c(2L, 2L, 2L), "float32")
  expect_np(x[0, sl(1, 3), 2], c(6, 10), 2L, "float32")
  expect_np(m[2, sl(step = -1)], c(11, 10, 9, 8), 4L, "int32")
  expect_np(as_jax(2)[], 2, integer(0), "float32")
  expect_np(as_jax(2)[ellipsis], 2, integer(0), "float32")
  expect_np(as_jax(2)[newaxis], 2, 1L, "float32")
})

test_that("advanced integer indexing matches the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(v[c(0, 0, 2)], c(0, 0, 20), 3L, "float32")
  expect_np(v[c(-1, 1, -5)], c(40, 10, 0), 3L, "float32")
  expect_np(v[matrix(c(0L, 4L, 1L, 3L), 2, 2)], c(0, 10, 40, 30), c(2L, 2L), "float32")
  expect_np(x[c(0, 1), c(1, 2)], c(4, 5, 6, 7, 20, 21, 22, 23), c(2L, 4L), "float32")
  expect_np(x[c(0, 1), , c(1, 2)], c(1, 5, 9, 14, 18, 22), c(2L, 3L), "float32")
  expect_np(x[, c(0, 2), c(1, 3)], c(1, 11, 13, 23), c(2L, 2L), "float32")
  expect_np(x[matrix(c(0L, 1L), 2, 1), c(0, 2)],
            c(0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23), c(2L, 2L, 4L), "float32")
  expect_np(x[0, c(1, 2)], c(4, 5, 6, 7, 8, 9, 10, 11), c(2L, 4L), "float32")
  expect_np(x[c(0, 1), sl(1, 3)],
            c(4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23), c(2L, 2L, 4L), "float32")
  expect_np(x[ellipsis, c(0, 3)],
            c(0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23), c(2L, 3L, 2L), "float32")
  expect_np(x[newaxis, c(0, 1)],
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23), c(1L, 2L, 3L, 4L), "float32")
  expect_np(x[c(0, 1), newaxis, 1], c(4, 5, 6, 7, 16, 17, 18, 19), c(2L, 1L, 4L), "float32")
  expect_np(x[1, , c(0, 1)], c(12, 16, 20, 13, 17, 21), c(2L, 3L), "float32")
  expect_np(x[c(0, 1), sl(step = 2), sl(step = -1)],
            c(3, 2, 1, 0, 11, 10, 9, 8, 15, 14, 13, 12, 23, 22, 21, 20), c(2L, 2L, 4L), "float32")
  expect_np(x[0, sl(step = 2), c(1, 3)], c(1, 9, 3, 11), c(2L, 2L), "float32")
  expect_np(x[sl(step = -1), c(0, 2), sl(1, 3)],
            c(13, 14, 21, 22, 1, 2, 9, 10), c(2L, 2L, 2L), "float32")
  expect_np(m[c(0, 2)][, c(1, 3)], c(1, 3, 9, 11), c(2L, 2L), "int32")
  expect_np(v[integer(0)], numeric(0), 0L, "float32")
  expect_np(jnp_zeros(0L)[integer(0)], numeric(0), 0L, "float32")
  expect_np(v[as_jax(c(1, 4), "uint8")], c(10, 40), 2L, "float32")
  expect_np(v[as_jax(c(1, -1), "int8")], c(10, 40), 2L, "float32")
  expect_np(v[as_jax(2L)], 20, integer(0), "float32")
  expect_np(v[as_jax(-2L)], 30, integer(0), "float32")
  expect_np(x[as_jax(1L), c(0, 2)], c(12, 13, 14, 15, 20, 21, 22, 23), c(2L, 4L), "float32")
  expect_np(x[as_jax(1L), sl(1), as_jax(-1L)], c(19, 23), 2L, "float32")
})

test_that("boolean mask indexing matches the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(v[v > 15], c(20, 30, 40), 3L, "float32")
  expect_np(m[m > 5L], c(6, 7, 8, 9, 10, 11), 6L, "int32")
  expect_np(m[c(TRUE, FALSE, TRUE), 1], c(1, 9), 2L, "int32")
  expect_np(m[c(TRUE, FALSE, TRUE)], c(0, 1, 2, 3, 8, 9, 10, 11), c(2L, 4L), "int32")
  expect_np(m[, c(TRUE, FALSE, TRUE, TRUE)], c(0, 2, 3, 4, 6, 7, 8, 10, 11), c(3L, 3L), "int32")
  expect_np(x[matrix(c(TRUE, FALSE, FALSE, TRUE, TRUE, FALSE), 2, 3)],
            c(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19), c(3L, 4L), "float32")
  expect_np(m[m > 100L], numeric(0), 0L, "int32")
  expect_np(v[TRUE], c(0, 10, 20, 30, 40), c(1L, 5L), "float32")
  expect_np(v[FALSE], numeric(0), c(0L, 5L), "float32")
  expect_np(v[as_jax(TRUE)], c(0, 10, 20, 30, 40), c(1L, 5L), "float32")
  expect_np(x[, TRUE],
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23), c(2L, 1L, 3L, 4L), "float32")
  expect_np(x[0, TRUE], c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), c(1L, 3L, 4L), "float32")
  expect_np(v[TRUE, c(0, 1)], c(0, 10), 2L, "float32")
  expect_np(v[newaxis, TRUE], 0, c(1L, 1L, 1L), "float32")
})

test_that("out of bounds reads, modes and strategies match the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(v[10], 40, integer(0), "float32")
  expect_np(v[-10], 0, integer(0), "float32")
  expect_np(v[5], 40, integer(0), "float32")
  expect_np(v[-6], 0, integer(0), "float32")
  expect_np(v[c(1, 7, -7)], c(10, 40, 0), 3L, "float32")
  expect_np(x[5, 1], c(16, 17, 18, 19), 4L, "float32")
  expect_np(x[1, 5], c(20, 21, 22, 23), 4L, "float32")
  expect_np(x[1, -5, 0], 12, integer(0), "float32")
  expect_np(v[as_jax(10L)], 40, integer(0), "float32")
  expect_np(v[as_jax(-10L)], 0, integer(0), "float32")
  expect_np(at_get(v, c(1, 7, -7), mode = "fill"), c(10, NaN, NaN), 3L, "float32")
  expect_np(at_get(v, c(1, 7, -7), mode = "fill", fill_value = -1), c(10, -1, -1), 3L, "float32")
  expect_np(at_get(v, c(1, 7, -7), mode = "clip"), c(10, 40, 0), 3L, "float32")
  expect_np(at_get(v, c(1, 7, -7), mode = "promise_in_bounds"), c(10, 40, 0), 3L, "float32")
  expect_np(at_get(v, 10, mode = "fill"), NaN, integer(0), "float32")
  expect_np(at_get(v, 10, mode = "clip"), 40, integer(0), "float32")
  expect_np(at_get(v, -10, mode = "fill"), NaN, integer(0), "float32")
  expect_np(at_get(jnp_arange(5L), c(1, 7), mode = "fill"), c(1, -2147483648), 2L, "int32")
  expect_np(at_get(as_jax(0:4, "uint8"), c(1, 7), mode = "fill"), c(1, 255), 2L, "uint8")
  expect_np(at_get(as_jax(c(FALSE, FALSE)), c(1, 7), mode = "fill"), c(FALSE, TRUE), 2L, "bool")
  expect_np(at_get(x, 1, c(0, 9), mode = "fill"),
            c(12, 13, 14, 15, NaN, NaN, NaN, NaN), c(2L, 4L), "float32")
  expect_np(at_get(v, -1, wrap_negative_indices = FALSE), 0, integer(0), "float32")
  expect_np(at_get(v, c(-1, 1), wrap_negative_indices = FALSE), c(0, 10), 2L, "float32")
  expect_np(at_get(v, c(-1, 1), wrap_negative_indices = FALSE, mode = "fill"),
            c(NaN, 10), 2L, "float32")
  expect_np(at_get(v, sl(1, 3), strategy = "dynamic_slice"), c(10, 20), 2L, "float32")
  expect_np(at_get(v, sl(1, 3), strategy = "gather"), c(10, 20), 2L, "float32")
  expect_np(at_get(v, sl(1, 3), strategy = "static_slice"), c(10, 20), 2L, "float32")
  expect_np(at_get(x, 1, sl(step = -1), strategy = "gather"),
            c(20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15), c(3L, 4L), "float32")
  expect_np(at_get(x, 1, sl(step = -1), strategy = "dynamic_slice"),
            c(20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15), c(3L, 4L), "float32")
  expect_np(at_get(x, -1, sl(2, 0, -1), newaxis, strategy = "dynamic_slice"),
            c(20, 21, 22, 23, 16, 17, 18, 19), c(2L, 1L, 4L), "float32")
})

test_that("functional updates match the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(at_set(v, 2, value = 7), c(0, 10, 7, 30, 40), 5L, "float32")
  expect_np(at_set(v, -1, value = 1), c(0, 10, 20, 30, 1), 5L, "float32")
  expect_np(at_set(v, c(0, 2), value = c(7, 8)), c(7, 10, 8, 30, 40), 5L, "float32")
  expect_np(at_set(v, sl(1, 4), value = 0), c(0, 0, 0, 0, 40), 5L, "float32")
  expect_np(at_set(v, sl(step = -1), value = c(0, 1, 2, 3, 4)), c(4, 3, 2, 1, 0), 5L, "float32")
  expect_np(at_set(v, sl(4, 1, -2), value = c(7, 8)), c(0, 10, 8, 30, 7), 5L, "float32")
  expect_np(at_set(x, , 1, value = c(1, 2, 3, 4)),
            c(0, 1, 2, 3, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 20, 21, 22, 23),
            c(2L, 3L, 4L), "float32")
  expect_np(at_set(x, 0, ellipsis, newaxis, value = 1),
            c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
            c(2L, 3L, 4L), "float32")
  expect_np(at_set(x, c(0, 1), , c(1, 2), value = -1),
            c(0, -1, 2, 3, 4, -1, 6, 7, 8, -1, 10, 11, 12, 13, -1, 15, 16, 17, -1, 19, 20, 21, -1,
            23), c(2L, 3L, 4L), "float32")
  expect_np(at_set(x, 1, 2, 3, value = 99),
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            99), c(2L, 3L, 4L), "float32")
  expect_np(at_set(v, value = 3), c(3, 3, 3, 3, 3), 5L, "float32")
  expect_np(at_set(m, m > 5L, value = 0), c(0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0), c(3L, 4L), "int32")
  expect_np(at_set(m, m > 100L, value = 0),
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), c(3L, 4L), "int32")
  expect_np(at_set(v, TRUE, value = 3), c(3, 3, 3, 3, 3), 5L, "float32")
  expect_np(at_set(v, FALSE, value = 3), c(0, 10, 20, 30, 40), 5L, "float32")
  expect_np(at_set(v, 0, value = 7L), c(7, 10, 20, 30, 40), 5L, "float32")
  expect_np(at_add(v, c(0, 0, 2), value = 1), c(2, 10, 21, 30, 40), 5L, "float32")
  expect_np(at_add(v, sl(step = 2), value = c(1, 2, 3)), c(1, 10, 22, 30, 43), 5L, "float32")
  expect_np(at_add(m, c(TRUE, FALSE, TRUE), value = 100),
            c(100, 101, 102, 103, 4, 5, 6, 7, 108, 109, 110, 111), c(3L, 4L), "int32")
  expect_np(at_add(x, c(0, 0), c(1, 1), value = 1),
            c(0, 1, 2, 3, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23), c(2L, 3L, 4L), "float32")
  expect_np(at_subtract(v, sl(1, 4), value = 1), c(0, 9, 19, 29, 40), 5L, "float32")
  expect_np(at_multiply(v, sl(step = 2), value = 2), c(0, 10, 40, 30, 80), 5L, "float32")
  expect_np(at_multiply(v, sl(step = -2), value = c(2, 3, 4)), c(0, 10, 60, 30, 80), 5L, "float32")
  expect_np(at_multiply(v, c(1, 1), value = c(2, 3)), c(0, 60, 20, 30, 40), 5L, "float32")
  expect_np(at_divide(v, c(1, 2), value = 4), c(0, 2.5, 5, 30, 40), 5L, "float32")
  expect_np(at_divide(jnp_arange(5L), 1, value = 2), c(0, 0.5, 2, 3, 4), 5L, "float32")
  expect_np(at_power(v, c(1, 2), value = 2), c(0, 100, 400, 30, 40), 5L, "float32")
  expect_np(at_min(v, c(1, 1), value = c(3, 5)), c(0, 3, 20, 30, 40), 5L, "float32")
  expect_np(at_max(v, c(1, 1), value = c(30, 50)), c(0, 50, 20, 30, 40), 5L, "float32")
  expect_np(at_max(as_jax(c(TRUE, FALSE)), 1, value = TRUE), c(TRUE, TRUE), 2L, "bool")
  expect_np(at_apply(v, c(0, 0, 2), func = function(t) t + 1), c(2, 10, 21, 30, 40), 5L, "float32")
  expect_np(at_apply(x, 1, sl(1), func = function(t) -t),
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16, -17, -18, -19, -20, -21,
            -22, -23), c(2L, 3L, 4L), "float32")
})

test_that("out of bounds updates match the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(at_set(v, c(1, 7, -7), value = 1), c(0, 1, 20, 30, 40), 5L, "float32")
  expect_np(at_set(v, c(1, 7, -7), value = c(1, 2, 3), mode = "clip"),
            c(3, 1, 20, 30, 2), 5L, "float32")
  expect_np(at_set(v, 10, value = 1), c(0, 10, 20, 30, 40), 5L, "float32")
  expect_np(at_set(v, -10, value = 1), c(0, 10, 20, 30, 40), 5L, "float32")
  expect_np(at_set(v, 10, value = 1, mode = "clip"), c(0, 10, 20, 30, 1), 5L, "float32")
  expect_np(at_add(v, c(1, 7, -7), value = 1), c(0, 11, 20, 30, 40), 5L, "float32")
  expect_np(at_add(v, c(1, 7, -7), value = 1, mode = "promise_in_bounds"),
            c(0, 11, 20, 30, 40), 5L, "float32")
  expect_np(at_add(v, c(1, 7, -7), value = 1, mode = "clip"), c(1, 11, 20, 30, 41), 5L, "float32")
  expect_np(at_set(v, sl(3, 10), value = 1), c(0, 10, 20, 1, 1), 5L, "float32")
  expect_np(at_set(v, c(-1, 1), value = 0, wrap_negative_indices = FALSE),
            c(0, 0, 20, 30, 40), 5L, "float32")
})

test_that("take, take_along_axis and put_along_axis match the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(jnp_take(m, c(0, 5, -1)), c(0, 5, 11), 3L, "int32")
  expect_np(jnp_take(m, c(0, 5, -1), axis = 1),
            c(0, -2147483648, 3, 4, -2147483648, 7, 8, -2147483648, 11), c(3L, 3L), "int32")
  expect_np(jnp_take(m, c(0, 5, -1), axis = 1, mode = "clip"),
            c(0, 3, 0, 4, 7, 4, 8, 11, 8), c(3L, 3L), "int32")
  expect_np(jnp_take(m, c(0, 5, -1), axis = 1, mode = "wrap"),
            c(0, 1, 3, 4, 5, 7, 8, 9, 11), c(3L, 3L), "int32")
  expect_np(jnp_take(m, c(0, 5, -1), axis = 1, mode = "fill"),
            c(0, -2147483648, 3, 4, -2147483648, 7, 8, -2147483648, 11), c(3L, 3L), "int32")
  expect_np(jnp_take(m, c(0, 50), fill_value = -5), c(0, -5), 2L, "int32")
  expect_np(jnp_take(x, matrix(c(0L, 2L, 1L, 2L), 2, 2), axis = 1),
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 20, 21, 22, 23), c(2L, 2L, 2L, 4L), "float32")
  expect_np(jnp_take(x, 1, axis = -1), c(1, 5, 9, 13, 17, 21), c(2L, 3L), "float32")
  expect_np(jnp_take(x, c(0, 30)), c(0, NaN), 2L, "float32")
  expect_np(jnp_take(x, integer(0), axis = 1), numeric(0), c(2L, 0L, 4L), "float32")
  expect_np(jnp_take_along_axis(m, i3, axis = 1), c(0, -2147483648, 11), c(3L, 1L), "int32")
  expect_np(jnp_take_along_axis(m, i3, axis = 1, mode = "clip"), c(0, 7, 11), c(3L, 1L), "int32")
  expect_np(jnp_take_along_axis(m, i3, axis = 1, fill_value = -9), c(0, -9, 11), c(3L, 1L), "int32")
  expect_np(jnp_take_along_axis(m, i34),
            c(3, 0, 1, 2, 4, 4, 4, 4, 9, 9, 10, 10), c(3L, 4L), "int32")
  expect_np(jnp_take_along_axis(m, i34[sl(stop = 2)], axis = 0),
            c(-2147483648, 1, 6, 11, 0, 1, 2, 3), c(2L, 4L), "int32")
  expect_np(jnp_take_along_axis(m, c(0, 11, -1, 12), axis = NULL),
            c(0, 11, 11, -2147483648), 4L, "int32")
  expect_np(jnp_take_along_axis(m, matrix(c(0L, 3L, -1L), 3, 1), axis = 1, mode = "one_hot"),
            c(0, 7, 11), c(3L, 1L), "int32")
  expect_np(jnp_take_along_axis(m, i34, mode = "one_hot"),
            c(3, 0, 1, 2, 4, 4, 4, 4, 9, 9, 10, 10), c(3L, 4L), "int32")
  expect_np(jnp_take_along_axis(v, c(4, 9, 0), axis = 0, mode = "one_hot"),
            c(40, 0, 0), 3L, "float32")
  expect_np(jnp_take_along_axis(x, i_x, axis = 1),
            c(8, 1, 6, 7, 12, 13, 22, 23), c(2L, 1L, 4L), "float32")
  expect_np(jnp_take_along_axis(x, array(c(1L, 0L), c(2, 1, 1)), axis = 2),
            c(1, 5, 9, 12, 16, 20), c(2L, 3L, 1L), "float32")
  expect_np(jnp_take_along_axis(m, i3, axis = 1, wrap_negative_indices = FALSE),
            c(0, -2147483648, -2147483648), c(3L, 1L), "int32")
  expect_np(jnp_put_along_axis(m, matrix(c(0L, 1L, 2L), 3, 1), 99L, axis = 1, inplace = FALSE),
            c(99, 1, 2, 3, 4, 99, 6, 7, 8, 9, 99, 11), c(3L, 4L), "int32")
  expect_np(jnp_put_along_axis(m, i34[sl(stop = 1)], c(-1L, -2L, -3L, -4L), axis = 0, inplace = FALSE),
            c(0, -2, 2, 3, 4, 5, -3, 7, 8, 9, 10, -4), c(3L, 4L), "int32")
  expect_np(jnp_put_along_axis(m, c(0, 11), -1L, axis = NULL, inplace = FALSE),
            c(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1), c(3L, 4L), "int32")
  expect_np(jnp_put_along_axis(m, matrix(c(0L, 9L, 2L), 3, 1), 99L, axis = 1, inplace = FALSE, mode = "clip"),
            c(99, 1, 2, 3, 4, 5, 6, 99, 8, 9, 99, 11), c(3L, 4L), "int32")
})

test_that("segment reductions match the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(ops_segment_sum(d6, ids, 5), c(1, 5, 0, 9, 0), 5L, "float32")
  expect_np(ops_segment_sum(d6, c(0, 0, 1, 1, 3, 7), 5), c(1, 5, 0, 4, 0), 5L, "float32")
  expect_np(ops_segment_sum(d6, c(0, -1, 1, 1, 3, 3), 5), c(0, 5, 0, 9, 0), 5L, "float32")
  expect_np(ops_segment_sum(d6, ids), c(1, 5, 0, 9), 4L, "float32")
  expect_np(ops_segment_sum(d62, ids, 4), c(2, 4, 10, 12, 0, 0, 18, 20), c(4L, 2L), "float32")
  expect_np(ops_segment_sum(d6, ids, 5, bucket_size = 4), c(1, 5, 0, 9, 0), 5L, "float32")
  expect_np(ops_segment_prod(jnp_arange(1L, 7L), ids, 5), c(2, 12, 1, 30, 1), 5L, "int32")
  expect_np(ops_segment_prod(d6 + 1, ids, 5, bucket_size = 2), c(2, 12, 1, 30, 1), 5L, "float32")
  expect_np(ops_segment_max(d6, ids, 5), c(1, 3, -Inf, 5, -Inf), 5L, "float32")
  expect_np(ops_segment_max(jnp_arange(6L), ids, 5),
            c(1, 3, -2147483648, 5, -2147483648), 5L, "int32")
  expect_np(ops_segment_max(d6, ids, 5, bucket_size = 4), c(1, 3, -Inf, 5, -Inf), 5L, "float32")
  expect_np(ops_segment_min(d6, ids, 5), c(0, 2, Inf, 4, Inf), 5L, "float32")
  expect_np(ops_segment_min(jnp_arange(6L), ids, 5),
            c(0, 2, 2147483647, 4, 2147483647), 5L, "int32")
  expect_np(ops_segment_min(d6 > 2, ids, 5), c(FALSE, FALSE, TRUE, TRUE, TRUE), 5L, "bool")
  expect_np(ops_segment_max(d6 > 2, ids, 5), c(FALSE, TRUE, FALSE, TRUE, FALSE), 5L, "bool")
  expect_np(ops_segment_sum(d6, c(0, 0, 1, 1, 3, 7), 5, mode = "clip"),
            c(1, 5, 0, 4, 5), 5L, "float32")
  expect_np(ops_segment_sum(d6, ids, 0), numeric(0), 0L, "float32")
})

test_that("derivatives through indexing match the wheel", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_np(grad(function(a) sum(a[c(0, 0, 2)]))(f3), c(2, 0, 1), 3L, "float32")
  expect_np(grad(function(a) sum(grad(function(b) sum(b[c(0, 0, 2)]^3))(a)))(f3),
            c(12, 0, 18), 3L, "float32")
  expect_np(grad(function(a) a[1]^2)(f3), c(0, 4, 0), 3L, "float32")
  expect_np(grad(function(a) a[-1]^2)(f3), c(0, 0, 6), 3L, "float32")
  expect_np(grad(function(a) a[as_jax(2L)]^2)(f3), c(0, 0, 6), 3L, "float32")
  expect_np(grad(function(a) sum(a[sl(1)]^2))(f3), c(0, 4, 6), 3L, "float32")
  expect_np(grad(function(a) sum(a[sl(step = -1)] * f3))(f3 * 2), c(3, 2, 1), 3L, "float32")
  expect_np(grad(function(a) sum(a[c(0, 7)]^2))(f3), c(2, 0, 0), 3L, "float32")
  expect_np(grad(function(a) sum(at_get(a, c(0, 7), mode = "fill", fill_value = 0)^2))(f3),
            c(2, 0, 0), 3L, "float32")
  expect_np(grad(function(a) sum(at_get(a, c(0, 7), mode = "clip")^2))(f3),
            c(2, 0, 6), 3L, "float32")
  expect_np(grad(function(a) sum(a[, c(0, 2)]^2))(X), c(2, 0, 6, 8, 0, 12), c(2L, 3L), "float32")
  expect_np(grad(function(a) sum(a[c(1, 1), c(0, 2)]^2))(X),
            c(0, 0, 0, 8, 0, 12), c(2L, 3L), "float32")
  expect_np(grad(function(a) sum(a[a > 2]^2))(f3 + 1), c(0, 6, 8), 3L, "float32")
  expect_np(grad(function(a) sum(at_set(a, c(0, 2), value = 5)^2))(f3), c(0, 4, 0), 3L, "float32")
  expect_np(grad(function(b) sum(at_set(f3, c(0, 2), value = b)^2))(u2), c(8, 10), 2L, "float32")
  expect_np(grad(function(b) sum(at_set(f3, c(0, 0), value = b)^2))(u2), c(0, 10), 2L, "float32")
  expect_np(grad(function(b) sum(at_add(f3, c(0, 0), value = b)^2))(u2), c(20, 20), 2L, "float32")
  expect_np(grad(function(b) sum(at_subtract(f3, c(0, 2), value = b)^2))(u2),
            c(6, 4), 2L, "float32")
  expect_np(grad(function(b) sum(at_multiply(f3, c(0, 2), value = b, unique_indices = TRUE)^2))(u2),
            c(8, 90), 2L, "float32")
  expect_np(grad(function(a) sum(at_multiply(a, c(0, 0), value = u2)^2))(f3),
            c(800, 4, 6), 3L, "float32")
  expect_np(grad(function(b) sum(at_min(f3, c(0, 2), value = b)^2))(c(0.5, 5)),
            c(1, 0), 2L, "float32")
  expect_np(grad(function(a) sum(at_max(a, c(0, 2), value = c(0.5, 5))^2))(f3),
            c(2, 4, 0), 3L, "float32")
  expect_np(grad(function(a) sum(at_divide(a, c(0, 2), value = u2, unique_indices = TRUE)^2))(f3),
            c(0.125, 4, 0.24000001), 3L, "float32")
  expect_np(grad(function(a) sum(at_set(a, sl(1), value = 0)^2))(f3), c(2, 0, 0), 3L, "float32")
  expect_np(grad(function(a) sum(at_set(a, 1, value = a[0] * 3)^2))(f3), c(20, 0, 6), 3L, "float32")
  expect_np(grad(function(a) sum(jnp_take(a, c(0, 0, 5))))(f3), c(2, 0, 0), 3L, "float32")
  expect_np(grad(function(a) sum(jnp_take(a, c(0, 0, 5), mode = "wrap")))(f3),
            c(2, 0, 1), 3L, "float32")
  expect_np(grad(function(a) sum(jnp_take(a, c(0, 0, 5), mode = "clip")))(f3),
            c(2, 0, 1), 3L, "float32")
  expect_np(grad(function(a) sum(jnp_take_along_axis(a, matrix(c(0L, 2L), 2, 1), 1)^2))(X),
            c(2, 0, 0, 0, 0, 12), c(2L, 3L), "float32")
  expect_np(grad(function(a) sum(jnp_take_along_axis(a, matrix(c(0L, 2L), 2, 1), 1, mode = "one_hot")^2))(X),
            c(2, 0, 0, 0, 0, 12), c(2L, 3L), "float32")
  expect_np(grad(function(a) sum(jnp_put_along_axis(a, matrix(c(0L, 2L), 2, 1), 0, axis = 1, inplace = FALSE)^2))(X),
            c(0, 4, 6, 8, 10, 0), c(2L, 3L), "float32")
  expect_np(grad(function(a) sum(ops_segment_sum(a, c(0, 0, 1), 2)^2))(f3),
            c(6, 6, 6), 3L, "float32")
  expect_np(grad(function(a) sum(ops_segment_max(a, c(0, 0, 1), 2)^2))(f3),
            c(0, 4, 6), 3L, "float32")
  expect_np(grad(function(a) sum(ops_segment_min(a, c(0, 0, 1), 2)^2))(f3),
            c(2, 0, 6), 3L, "float32")
  expect_np(grad(function(a) sum(ops_segment_prod(a, c(0, 1, 2), 3, unique_indices = TRUE)^2))(f3),
            c(2, 4, 6), 3L, "float32")
  expect_np(jvp(function(a) a[c(0, 0, 2)], list(f3), list(f3 * 10))[[2L]],
            c(10, 10, 30), 3L, "float32")
  expect_np(vjp(function(a) a[, c(0, 0)], X)[[2L]](jnp_ones(c(2L, 2L)))[[1L]],
            c(2, 0, 0, 2, 0, 0), c(2L, 3L), "float32")
})

test_that("indexing errors", {
  skip_if_no_backend()
  indexing_fixtures()
  expect_error(v[1.5], "Indexer must have integer or boolean type, got indexer with type float64",
               class = "jax_type_error")
  expect_error(v[as_jax(c(0.5, 1))], class = "jax_type_error",
               "Indexer must have integer or boolean type, got indexer with type float32")
  expect_error(v["a"], "JAX does not support string indexing", class = "jax_type_error")
  expect_error(v[list(0, 1)], "Using a list for multidimensional indexing is not allowed",
               class = "jax_type_error")
  expect_error(v[NULL], "use `newaxis`", class = "jax_index_error")
  expect_error(v[0, drop = FALSE], "unsupported named index argument", class = "jax_type_error")
  expect_error(v[0, 0], "Too many indices: array is 1-dimensional, but 2 were indexed",
               class = "jax_index_error")
  expect_error(x[ellipsis, ellipsis], "an index can only have a single ellipsis",
               class = "jax_index_error")
  expect_error(v[c(TRUE, FALSE)], class = "jax_index_error", fixed = TRUE, paste0(
    "boolean index did not match shape of indexed array in index 0: got (2,), expected [5]"))
  expect_error(jit(function(a) a[a > 1])(v), "Array boolean indices must be concrete; got bool[5]",
               fixed = TRUE, class = "jax_non_concrete_boolean_index_error")
  expect_error(jit(function(a) at_set(a, a > 1, value = 0))(v), class = "jax_index_error",
               "Array boolean indices must be concrete")
  expect_error(jit(function(a, i) a[sl(i)])(v, 1L), class = "jax_index_error",
               "Array slice indices must have static start/stop/step")
  expect_error(v[sl(step = 0)], "slice step cannot be zero", class = "jax_value_error")
  expect_error(jnp_zeros(0L)[0], "index is out of bounds for axis 0 with size 0",
               class = "jax_index_error")
  expect_error(jnp_zeros(c(0L, 3L))[0], "index is out of bounds for axis 0 with size 0",
               class = "jax_index_error")
  expect_error(jnp_zeros(0L)[array(0L, 1L)], class = "jax_type_error", fixed = TRUE,
               "Slice size at index 0 in gather op is out of range, must be within [0, 0 + 1), got 1.")
  expect_error(at_get(v, array(0L, 1L), strategy = "static_slice"),
               "static_slice: indices must be static scalars or slices", class = "jax_type_error")
  expect_error(at_get(v, sl(step = 2), strategy = "dynamic_slice"),
               "dynamic_slice: only unit steps supported in slice", class = "jax_type_error")
  expect_error(at_get(v, 1, strategy = "static_slice", mode = "fill"), class = "jax_value_error",
               "static_slice requires mode='promise_in_bounds' or mode='clip'")
  expect_error(at_get(v, 1, strategy = "scatter"), "not supported for take",
               class = "jax_value_error")
  expect_error(at_get(v, 1, strategy = "foo"), "Unknown indexing strategy",
               class = "jax_value_error")
  expect_error(at_get(v, array(9L, 1L), mode = "fill", fill_value = c(1, 2)),
               "fill_value argument to indexed get\\(\\) must be a scalar", class = "jax_value_error")
  expect_error(at_get(v, 1, mode = "foo"), "Unknown gather mode", class = "jax_value_error")
  expect_error(at_set(v, sl(1, 3), value = jnp_ones(3L)), "Incompatible shapes for broadcasting")
  expect_error(at_add(as_jax(c(TRUE, FALSE)), 0, value = TRUE),
               "add does not accept dtype bool at position 0", class = "jax_type_error")
  expect_error(at_multiply(as_jax(c(TRUE, FALSE)), 0, value = FALSE),
               "mul does not accept dtype bool at position 0", class = "jax_type_error")
  expect_error(jnp_take(m, array(0L, 1L), mode = "raise"), class = "jax_not_implemented_error",
               "The 'raise' mode to jnp.take is not supported.")
  expect_error(jnp_take(m, array(0L, 1L), mode = "foo"), "Invalid mode 'foo' for np.take",
               class = "jax_value_error")
  expect_error(jnp_take(m, as_jax(c(0, 1.5))), "indices must have an integer type",
               class = "jax_value_error")
  expect_error(jnp_take(m, 0, out = m), "'out' argument", class = "jax_not_implemented_error")
  expect_error(jnp_take(jnp_zeros(c(0L, 2L)), array(0L, 1L), axis = 0),
               "Cannot do a non-empty jnp.take\\(\\) from an empty axis", class = "jax_index_error")
  expect_error(jnp_take_along_axis(m, c(0, 1), axis = 1), class = "jax_value_error",
               "indices and arr must have the same number of dimensions; 1 vs. 2")
  expect_error(jnp_take_along_axis(m, as_jax(matrix(0.5, 1, 1)), axis = 1), class = "jax_type_error",
               "take_along_axis indices must be of integer type, got float32")
  expect_error(jnp_take_along_axis(m, i34, axis = NULL), class = "jax_value_error",
               "take_along_axis indices must be 1D if axis=None")
  expect_error(jnp_put_along_axis(m, matrix(0L, 3, 1), 99L, axis = 1),
               "cannot modify arrays in-place", class = "jax_value_error")
  expect_error(jnp_put_along_axis(m, c(0, 1), 99L, axis = 1, inplace = FALSE),
               "must have same ndim", class = "jax_value_error")
  expect_error(jnp_put_along_axis(m, matrix(0L, 3, 1), c(1L, 2L), axis = 1, inplace = FALSE),
               "must be broadcastable to 'indices'", class = "jax_value_error")
  expect_error(ops_segment_sum(d6, ids, -1), "num_segments must be non-negative.",
               class = "jax_value_error")
  expect_error(jit(function(d, i) ops_segment_sum(d, i))(d6, ids), class = "jax_concretization_error")
})

test_that("`[<-` returns an updated copy", {
  skip_if_no_backend()
  indexing_fixtures()
  y <- f3
  y[1] <- 10
  expect_np(y, c(1, 10, 3), 3L, "float32")
  expect_np(f3, c(1, 2, 3), 3L, "float32")
  y[c(0, 2)] <- c(7, 8)
  expect_np(y, c(7, 10, 8), 3L, "float32")
  y[sl(1)] <- 0
  expect_np(y, c(7, 0, 0), 3L, "float32")
  y[] <- 5
  expect_np(y, c(5, 5, 5), 3L, "float32")
  y[y > 4] <- c(1, 2, 3)
  expect_np(y, c(1, 2, 3), 3L, "float32")
  M <- np_arr(0:11, c(3L, 4L), "float32")
  M[, 1] <- 100
  expect_np(M, c(0, 100, 2, 3, 4, 100, 6, 7, 8, 100, 10, 11), c(3L, 4L), "float32")
  M[1, ] <- c(1, 2, 3, 4)
  expect_np(M, c(0, 100, 2, 3, 1, 2, 3, 4, 8, 100, 10, 11), c(3L, 4L), "float32")
  M[-1, sl(step = -1)] <- M[0, ]
  expect_np(M, c(0, 100, 2, 3, 1, 2, 3, 4, 3, 2, 100, 0), c(3L, 4L), "float32")
  M[ellipsis, newaxis, 0] <- 9
  expect_np(M[, 0], c(9, 9, 9), 3L, "float32")
  # Under jit and grad, with a traced index.
  g <- jit(function(a, i) {
    a[0] <- a[2] * 2
    a[i] <- -1
    a
  })
  expect_np(g(f3, 1L), c(6, -1, 3), 3L, "float32")
  expect_np(grad(function(a) {
    a[1] <- a[0]^2
    sum(a * c(1, 10, 100))
  })(f3), c(21, 0, 100), 3L, "float32")
  # Out of bounds updates are dropped.
  z <- f3
  z[7] <- 1
  z[c(-9, 9)] <- 1
  expect_np(z, c(1, 2, 3), 3L, "float32")
})

test_that("indexing under jit matches eager evaluation", {
  skip_if_no_backend()
  indexing_fixtures()
  funs <- list(
    function(a) a[1], function(a) a[-1, 2], function(a) a[, 1], function(a) a[ellipsis, 0],
    function(a) a[newaxis, , sl(1)], function(a) a[sl(step = -1), sl(1, 3), sl(step = -2)],
    function(a) a[c(0, 1), , c(1, 2)], function(a) a[, c(0, 2), c(1, 3)],
    function(a) a[0, sl(step = 2), c(1, 3)], function(a) a[matrix(c(0L, 1L), 2, 1), c(0, 2)],
    function(a) a[c(TRUE, FALSE)], function(a) a[, c(TRUE, FALSE, TRUE)], function(a) a[5, 1],
    function(a) a[c(0, 7, -7)], function(a) at_get(a, c(0, 7, -7), mode = "fill"),
    function(a) at_get(a, 1, c(0, 9), mode = "clip"), function(a) a[TRUE],
    function(a) at_set(a, c(0, 1), , c(1, 2), value = -1),
    function(a) at_add(a, c(0, 0), c(1, 1), value = 1),
    function(a) at_multiply(a, , sl(step = -1), 0, value = c(2, 3, 4)),
    function(a) at_min(a, 1, value = 5), function(a) at_max(a, ellipsis, 2, value = 9),
    function(a) at_divide(a, 0, 1, value = 4), function(a) at_power(a, 0, 1, value = 2),
    function(a) at_subtract(a, c(TRUE, FALSE), value = 1),
    function(a) at_apply(a, 1, sl(1), func = function(t) -t),
    function(a) at_set(a, c(1, 7, -7), value = 1),
    function(a) at_set(a, c(1, 7, -7), value = 1, mode = "clip"),
    function(a) jnp_take(a, c(0, 5, -1), axis = 2),
    function(a) jnp_take(a, c(0, 5, -1), axis = 2, mode = "wrap"),
    function(a) jnp_take(a, c(0, 30)), function(a) jnp_take_along_axis(a, i_x, axis = 1),
    function(a) jnp_take_along_axis(a, i_x, axis = 1, mode = "one_hot"),
    function(a) jnp_put_along_axis(a, i_x, 0, axis = 1, inplace = FALSE),
    function(a) ops_segment_sum(a, c(1, 1), 3), function(a) ops_segment_max(a, c(2, 0), 3),
    function(a) ops_segment_prod(a, c(0, 0), 2, bucket_size = 1))
  for (f in funs) expect_same(jit(f)(x), f(x))
})

test_that("traced indices work under jit", {
  skip_if_no_backend()
  indexing_fixtures()
  at <- jit(function(a, i) a[i])
  expect_np(at(v, 3L), 30, integer(0), "float32")
  expect_np(at(v, 7L), 40, integer(0), "float32")
  expect_np(at(v, -7L), 0, integer(0), "float32")
  expect_np(at(v, -2L), 30, integer(0), "float32")
  expect_np(at(v, np_arr(c(4L, 0L, -1L, 9L), 4L)), c(40, 0, 40, 40), 4L, "float32")
  at2 <- jit(function(a, i, j) a[i, sl(1), j])
  expect_np(at2(x, 1L, -1L), c(19, 23), 2L, "float32")
  expect_np(jit(function(a, i) a[i, c(0, 2)])(x, 1L), c(12, 13, 14, 15, 20, 21, 22, 23),
            c(2L, 4L), "float32")
  expect_np(jit(function(a, i) at_get(a, i, mode = "fill"))(v, 9L), NaN, integer(0), "float32")
  set <- jit(function(a, i, b) at_set(a, i, value = b))
  expect_np(set(v, 1L, 5), c(0, 5, 20, 30, 40), 5L, "float32")
  expect_np(set(v, 9L, 5), c(0, 10, 20, 30, 40), 5L, "float32")
  expect_np(set(v, np_arr(c(0L, -1L), 2L), 5), c(5, 10, 20, 30, 5), 5L, "float32")
  expect_np(jit(function(a, i) at_add(a, i, sl(2), value = 100))(x, 1L)[1, , 3],
            c(15, 19, 123), 3L, "float32")
  # A concrete mask captured by a jitted function is fine.
  mask <- c(TRUE, FALSE, TRUE, FALSE, TRUE)
  expect_np(jit(function(a) a[mask])(v), c(0, 20, 40), 3L, "float32")
  expect_np(jit(function(a) at_set(a, mask, value = -1))(v), c(-1, 10, -1, 30, -1), 5L, "float32")
  # Static slice bounds may be concrete jax scalars.
  expect_np(v[sl(as_jax(1L), as_jax(4L))], c(10, 20, 30), 3L, "float32")
  expect_np(jit(function(d, i) ops_segment_sum(d, i, 5))(d6, ids), c(1, 5, 0, 9, 0), 5L, "float32")
})

test_that("vmap of indexing matches a stacked loop", {
  skip_if_no_backend()
  indexing_fixtures()
  A <- np_arr(0:23, c(2L, 3L, 4L), "float32")
  I <- np_arr(c(0L, 2L, 5L, -1L), c(2L, 2L))
  S <- np_arr(c(1L, -9L), 2L)
  U <- np_arr(c(-1, -2, -3, -4), c(2L, 2L), "float32")
  funs1 <- list(function(a) a[1], function(a) a[, c(0, 3)], function(a) a[c(0, 2), sl(1, 3)],
                function(a) a[sl(step = -1), newaxis, 2], function(a) a[c(TRUE, FALSE, TRUE)],
                function(a) at_set(a, 1, value = 0), function(a) at_add(a, c(0, 0), 1, value = 1),
                function(a) jnp_take(a, c(0, 9), axis = 1),
                function(a) ops_segment_sum(a, c(0, 0, 1), 2))
  for (f in funs1) expect_same(vmap(f)(A), stack_loop(f, 2L, A))
  funs2 <- list(function(a, i) a[i], function(a, i) a[i, 1], function(a, i) a[sl(1), i],
                function(a, i) a[i, i], function(a, i) at_get(a, i, mode = "fill"),
                function(a, i) at_get(a, i, mode = "clip"),
                function(a, i) at_set(a, i, value = 0),
                function(a, i) at_add(a, i, 0, value = 1),
                function(a, i) at_multiply(a, 1, i, value = 3),
                function(a, i) at_max(a, i, value = 100),
                function(a, i) jnp_take(a, i, axis = 1),
                function(a, i) jnp_take(a, i, axis = 1, mode = "wrap"),
                function(a, i) ops_segment_sum(row_of(a, 0L)[i], i, 3))
  for (f in funs2) {
    expect_same(vmap(f)(A, I), stack_loop(f, 2L, A, I))
    expect_same(vmap(f)(A, S), stack_loop(f, 2L, A, S))
  }
  a0 <- row_of(A, 0L)
  for (f in funs2) expect_same(vmap(function(i) f(a0, i))(I),
                               stack_loop(function(i) f(a0, i), 2L, I))
  f3 <- function(a, i, b) at_set(a, i, 1, value = b)
  expect_same(vmap(f3)(A, I, U), stack_loop(f3, 2L, A, I, U))
  expect_same(vmap(function(b) f3(a0, c(0, 2), b))(U),
              stack_loop(function(b) f3(a0, c(0, 2), b), 2L, U))
  expect_same(vmap(function(i, b) f3(a0, i, b))(I, U),
              stack_loop(function(i, b) f3(a0, i, b), 2L, I, U))
  taa <- function(a, i) jnp_take_along_axis(a, i, axis = 0)
  It <- np_arr(c(0L, 2L, 1L, 1L, 2L, 2L, 0L, 5L), c(2L, 1L, 4L))
  expect_same(vmap(taa)(A, It), stack_loop(taa, 2L, A, It))
  expect_same(jit(vmap(funs2[[1L]]))(A, I), stack_loop(funs2[[1L]], 2L, A, I))
  expect_same(vmap(function(a, i) a[i], in_axes = list(1L, 0L))(jnp_transpose(A, c(1L, 0L, 2L)), I),
              stack_loop(function(a, i) a[i], 2L, A, I))
  expect_same(vmap(grad(function(a, i) sum(a[i, 1]^2)))(A, I),
              stack_loop(grad(function(a, i) sum(a[i, 1]^2)), 2L, A, I))
})

test_that("gradients of indexing agree with finite differences", {
  skip_if_no_backend()
  indexing_fixtures()
  with_x64({
    a0 <- as_jax(c(0.3, -1.2, 2.5, 0.7, 1.9, -0.4), "float64")
    b0 <- as_jax(c(0.9, 1.7), "float64")
    r <- function(a) jnp_reshape(a, c(2L, 3L))
    fs <- list(
      function(a) sum(a[c(0, 0, 4)]^3), function(a) sum(sin(a[sl(1, 5, 2)])),
      function(a) sum(r(a)[, c(0, 2)]^3), function(a) sum(r(a)[c(1, 1), c(0, 2)]^3),
      function(a) sum(r(a)[sl(step = -1), 1] * c(1, 2)), function(a) a[as_jax(3L)]^3,
      function(a) sum(at_get(a, c(0, 9), mode = "fill", fill_value = 0)^3),
      function(a) sum(at_get(a, c(0, 9), mode = "clip")^3),
      function(a) sum(at_set(a, c(0, 2), value = 5)^3),
      function(a) sum(at_set(a, 1, value = a[0] * a[2])^3),
      function(a) sum(at_add(a, c(0, 0), value = a[sl(4)])^3),
      function(a) sum(at_subtract(a, sl(step = 2), value = 2)^3),
      function(a) sum(at_multiply(a, c(0, 0), value = c(2, 3))^3),
      function(a) sum(at_min(a, c(1, 3), value = c(0.5, 0.5))^3),
      function(a) sum(at_max(a, c(1, 3), value = c(0.5, 0.5))^3),
      function(a) sum(at_divide(a, c(0, 1), value = c(2, 4), unique_indices = TRUE)^3),
      function(a) sum(jnp_take(a, c(0, 0, 9), fill_value = 0)^3),
      function(a) sum(jnp_take(a, c(0, 0, 9), mode = "wrap")^3),
      function(a) sum(jnp_take_along_axis(r(a), matrix(c(0L, 2L), 2, 1), 1)^3),
      function(a) sum(jnp_take_along_axis(r(a), matrix(c(0L, 2L), 2, 1), 1, mode = "one_hot")^3),
      function(a) sum(jnp_put_along_axis(r(a), matrix(c(0L, 2L), 2, 1), 2, axis = 1, inplace = FALSE)^3),
      function(a) sum(ops_segment_sum(a, c(0, 0, 1, 1, 3, 3), 4)^3),
      function(a) sum(ops_segment_max(a, c(0, 0, 1, 1, 3, 3), 4)[c(0, 1, 3)]^3),
      function(a) sum(ops_segment_min(a, c(0, 0, 1, 1, 3, 3), 4)[c(0, 1, 3)]^3),
      function(a) sum(ops_segment_sum(a, c(0, 0, 1, 1, 3, 3), 4, bucket_size = 4)^3),
      # second order
      function(a) sum(grad(function(b) sum(b[c(0, 0, 4)]^3))(a) * a),
      function(a) sum(grad(function(b) sum(at_add(b, c(0, 0), value = b[sl(4)])^3))(a) * a))
    for (f in fs) check_grad_fd(f, a0)
    gs <- list(
      function(b) sum(at_set(a0, c(0, 2), value = b)^3),
      function(b) sum(at_set(a0, c(0, 0), value = b)^3),
      function(b) sum(at_add(a0, c(1, 1), value = b)^3),
      function(b) sum(at_subtract(a0, c(1, 4), value = b)^3),
      function(b) sum(at_multiply(a0, c(1, 4), value = b, unique_indices = TRUE)^3),
      function(b) sum(at_min(a0, c(0, 2), value = b)^3),
      function(b) sum(at_max(a0, c(0, 2), value = b)^3),
      function(b) sum(at_set(a0, c(0, 9), value = b, mode = "clip")^3),
      function(b) sum(at_divide(a0, c(1, 4), value = b, unique_indices = TRUE)^3))
    for (g in gs) check_grad_fd(g, b0)
    # Index arrays of every integer width work with 64-bit types enabled.
    expect_identical(dtype(a0[as_jax(c(0, 2), "int64")]), "float64")
    expect_equal(as.vector(a0[as_jax(c(0, -1), "int64")]), c(0.3, -0.4))
    expect_equal(as.vector(at_set(a0, as_jax(1, "int64"), value = 0)), c(0.3, 0, 2.5, 0.7, 1.9, -0.4))
  })
})

test_that("dtype and weak type rules of indexing", {
  skip_if_no_backend()
  indexing_fixtures()
  w <- lax_broadcast(1, 3L)
  expect_true(jaxR:::aval_of(w[c(0, 1)])$weak)
  expect_true(jaxR:::aval_of(w[1])$weak)
  expect_true(jaxR:::aval_of(at_set(w, 0, value = 2))$weak)
  expect_np(at_set(w, 0, value = 2), c(2, 1, 1), 3L, "float32")
  expect_false(jaxR:::aval_of(v[c(0, 1)])$weak)
  i3a <- np_arr(0:2, 3L)
  # An integer valued R scalar is an integer update for integer arrays.
  expect_np(expect_no_warning(at_add(i3a, 0, value = 2)), c(2, 1, 2), 3L, "int32")
  expect_np(at_add(as_jax(0:2, "int8"), 0, value = 300), c(44, 1, 2), 3L, "int8")
  expect_np(at_set(as_jax(0:2, "uint8"), 0, value = -1), c(255, 1, 2), 3L, "uint8")
  expect_np(expect_no_warning(at_set(v, c(0, 1), value = c(7L, 8L))), c(7, 8, 20, 30, 40), 5L,
            "float32")
  # Integer valued R vectors are integer updates too.
  expect_np(expect_no_warning(at_set(i3a, c(0, 1), value = c(7, 8))), c(7, 8, 2), 3L, "int32")
  expect_np(expect_no_warning(at_add(m, , 1, value = c(10, 20, 30))),
            c(0, 11, 2, 3, 4, 25, 6, 7, 8, 39, 10, 11), c(3L, 4L), "int32")
  # Unsafe casts warn and keep the dtype of the array, as in the wheel.
  expect_warning(r <- at_set(i3a, 0, value = 1.5), paste0(
    "scatter inputs have incompatible types: cannot safely cast value from dtype=float32 to ",
    "dtype=int32 with jax_numpy_dtype_promotion=standard"))
  expect_np(r, c(1, 1, 2), 3L, "int32")
  expect_warning(r <- at_set(v, 0, value = 1i), "dtype=complex64 to dtype=float32")
  expect_np(r, c(0, 10, 20, 30, 40), 5L, "float32")
  expect_identical(dtype(at_divide(i3a, 1, value = 2)), "float32")
  expect_identical(dtype(jnp_take(m, c(0, 1))), "int32")
  expect_identical(dtype(ops_segment_sum(m, c(0, 0, 1), 2)), "int32")
})
