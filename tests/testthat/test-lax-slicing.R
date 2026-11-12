# Tests for lax gather and scatter, index_take and the dynamic *_in_dim wrappers
# (R/lax_slicing.R). Expected values were generated with the jax 0.12.0.1 wheel.

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

slicing_fixtures <- function(env = parent.frame()) {
  force(env)
  local({
    v <- np_arr(c(10, 11, 12), 3L, "float32")
    m <- np_arr(0:11, c(3L, 4L), "float32")
    x3 <- np_arr(0:23, c(2L, 3L, 4L), "float32")
    u <- np_arr(c(1, 2, 3), 3L, "float32")
    u2 <- np_arr(c(100, 2, 3), 3L, "float32")
    u_2 <- np_arr(c(1, 2), 2L, "float32")
    tv <- np_arr(c(1, 10, 100), 3L, "float32")
    tu <- np_arr(c(0.5, 0.25, 2), 3L, "float32")
    iv <- np_arr(c(0L, 0L, 2L), c(3L, 1L))
    iu <- np_arr(c(0L, 1L, 2L), c(3L, 1L))
    iu2 <- np_arr(c(0L, 1L), c(2L, 1L))
    oob <- np_arr(c(0L, 5L, -1L), c(3L, 1L))
    irow <- np_arr(c(1L, 0L, 1L), c(3L, 1L))
    irow2 <- np_arr(c(2L, 0L), c(2L, 1L))
    imid <- np_arr(c(2L, 0L), c(2L, 1L))
    iwin <- np_arr(c(0L, 1L, 1L, 2L), c(2L, 2L))
    iwin_oob <- np_arr(c(2L, 3L, -1L, 1L), c(2L, 2L))
    ibat <- np_arr(c(0L, 2L, 1L, 1L), c(2L, 2L, 1L))
    urow <- np_arr(100 + 0:7, c(2L, 4L), "float32")
    uwin <- np_arr(rep(1, 8), c(2L, 2L, 2L), "float32")
    ubat <- np_arr(0:15, c(2L, 2L, 4L), "float32")
    a5 <- np_arr(0:4, 5L)
    f5 <- np_arr(0:4, 5L, "float32")
    m34 <- np_arr(0:11, c(3L, 4L))
    z6 <- np_arr(rep(0, 6), 6L, "float32")
    o3 <- np_arr(rep(1, 3), 3L, "float32")
    dn1 <- lax_GatherDimensionNumbers(integer(0), 0L, 0L)
    dn_rows <- lax_GatherDimensionNumbers(c(1L, 2L), 0L, 0L)
    dn_win <- lax_GatherDimensionNumbers(c(1L, 2L), integer(0), c(0L, 1L))
    dn_mid <- lax_GatherDimensionNumbers(c(0L, 2L), 1L, 1L)
    dn_bat <- lax_GatherDimensionNumbers(2L, 1L, 1L, operand_batching_dims = 0L,
                                     start_indices_batching_dims = 0L)
    sd1 <- lax_ScatterDimensionNumbers(integer(0), 0L, 0L)
    sd_rows <- lax_ScatterDimensionNumbers(1L, 0L, 0L)
    sd_win <- lax_ScatterDimensionNumbers(c(1L, 2L), integer(0), c(0L, 1L))
    sd_bat <- lax_ScatterDimensionNumbers(2L, 1L, 1L, operand_batching_dims = 0L,
                                      scatter_indices_batching_dims = 0L)
  }, envir = env)
}

test_that("dimension numbers and modes are plain values", {
  d <- lax_GatherDimensionNumbers(offset_dims = c(1, 2), collapsed_slice_dims = 0,
                              start_index_map = 0)
  expect_identical(d, list(offset_dims = c(1L, 2L), collapsed_slice_dims = 0L,
                           start_index_map = 0L, operand_batching_dims = integer(0),
                           start_indices_batching_dims = integer(0)))
  s <- lax_ScatterDimensionNumbers(update_window_dims = 1, inserted_window_dims = 0,
                               scatter_dims_to_operand_dims = 0, operand_batching_dims = 2,
                               scatter_indices_batching_dims = 0)
  expect_identical(s, list(update_window_dims = 1L, inserted_window_dims = 0L,
                           scatter_dims_to_operand_dims = 0L, operand_batching_dims = 2L,
                           scatter_indices_batching_dims = 0L))
  from_any <- lax_GatherScatterMode$from_any
  expect_identical(from_any("clip"), lax_GatherScatterMode$CLIP)
  expect_identical(from_any("fill"), lax_GatherScatterMode$FILL_OR_DROP)
  expect_identical(from_any("drop"), lax_GatherScatterMode$FILL_OR_DROP)
  expect_identical(from_any(NULL), lax_GatherScatterMode$FILL_OR_DROP)
  expect_identical(from_any("promise_in_bounds"), lax_GatherScatterMode$PROMISE_IN_BOUNDS)
  expect_identical(from_any("one_hot"), lax_GatherScatterMode$ONE_HOT)
  expect_identical(from_any(lax_GatherScatterMode$CLIP), lax_GatherScatterMode$CLIP)
  expect_error(from_any("foo"), "Unknown gather mode \"foo\"", class = "jax_value_error")
})

test_that("gather matches the wheel", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(lax_gather(v, iv, dn1, 1L), c(10, 10, 12), 3L, "float32")
  expect_np(lax_gather(x3, irow, dn_rows, c(1L, 3L, 4L)),
            c(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), c(3L, 3L, 4L), "float32")
  expect_np(lax_gather(m, iwin, dn_win, c(2L, 2L)),
            c(1, 2, 5, 6, 6, 7, 10, 11), c(2L, 2L, 2L), "float32")
  expect_np(lax_gather(x3, ibat, dn_bat, c(1L, 1L, 4L)),
            c(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 16, 17, 18, 19), c(2L, 2L, 4L), "float32")
  expect_np(lax_gather(x3, imid, dn_mid, c(2L, 1L, 4L)),
            c(8, 9, 10, 11, 0, 1, 2, 3, 20, 21, 22, 23, 12, 13, 14, 15), c(2L, 2L, 4L), "float32")
})

test_that("gather out of bounds modes match the wheel", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(lax_gather(v, oob, dn1, 1L), c(10, 12, 10), 3L, "float32")
  expect_np(lax_gather(v, oob, dn1, 1L, mode = "clip"), c(10, 12, 10), 3L, "float32")
  expect_np(lax_gather(v, oob, dn1, 1L, mode = "fill"), c(10, NaN, NaN), 3L, "float32")
  expect_np(lax_gather(v, oob, dn1, 1L, mode = "drop"), c(10, NaN, NaN), 3L, "float32")
  expect_np(lax_gather(v, oob, dn1, 1L, mode = "fill", fill_value = -7),
            c(10, -7, -7), 3L, "float32")
  expect_np(lax_gather(jnp_arange(3L), oob, dn1, 1L, mode = "fill"),
            c(0, -2147483648, -2147483648), 3L, "int32")
  expect_np(lax_gather(as_jax(0:2, "uint8"), oob, dn1, 1L, mode = "fill"),
            c(0, 255, 255), 3L, "uint8")
  expect_np(lax_gather(as_jax(c(FALSE, FALSE, FALSE)), oob, dn1, 1L, mode = "fill"),
            c(FALSE, TRUE, TRUE), 3L, "bool")
  expect_np(lax_gather(m, iwin_oob, dn_win, c(2L, 2L), mode = "clip"),
            c(6, 7, 10, 11, 1, 2, 5, 6), c(2L, 2L, 2L), "float32")
  expect_np(lax_gather(m, iwin_oob, dn_win, c(2L, 2L), mode = "fill"),
            c(NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN), c(2L, 2L, 2L), "float32")
  expect_np(lax_gather(m, iwin_oob, dn_win, c(2L, 2L)),
            c(6, 7, 10, 11, 1, 2, 5, 6), c(2L, 2L, 2L), "float32")
})

test_that("gather under jit matches eager evaluation and the wheel lowering", {
  skip_if_no_backend()
  slicing_fixtures()
  cases <- list(
    list(function(a, i) lax_gather(a, i, dn1, 1L), v, iv),
    list(function(a, i) lax_gather(a, i, dn_rows, c(1L, 3L, 4L)), x3, irow),
    list(function(a, i) lax_gather(a, i, dn_win, c(2L, 2L), mode = "fill"), m, iwin_oob),
    list(function(a, i) lax_gather(a, i, dn_win, c(2L, 2L), mode = "clip"), m, iwin_oob),
    list(function(a, i) lax_gather(a, i, dn_bat, c(1L, 1L, 4L)), x3, ibat),
    list(function(a, i) lax_gather(a, i, dn1, 1L, mode = "fill", fill_value = -7), v, oob))
  for (case in cases) expect_same(jit(case[[1L]])(case[[2L]], case[[3L]]),
                                  case[[1L]](case[[2L]], case[[3L]]))
  # The attribute text is the one printed by the wheel (empty lists are omitted).
  txt <- lower_text(function(a, i) lax_gather(a, i, dn_bat, c(1L, 1L, 4L)), x3, ibat)
  expect_match(txt, paste0(
    "\"stablehlo.gather\"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<",
    "offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], ",
    "start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, ",
    "indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 4>}>"), fixed = TRUE)
  txt <- lower_text(function(a, i) lax_gather(a, i, dn1, 1L, indices_are_sorted = TRUE), v, iv)
  expect_match(txt, paste0(
    "#stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], ",
    "index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1>"),
    fixed = TRUE)
})

test_that("gather dtype, weak type and shape rules", {
  skip_if_no_backend()
  slicing_fixtures()
  # The result has the dtype and weak type of the operand, for any integer index dtype.
  for (dt in c("int8", "uint8", "int16", "uint32")) {
    expect_np(lax_gather(v, as_jax(iv, dt), dn1, 1L), c(10, 10, 12), 3L, "float32")
  }
  w <- lax_broadcast(1, 3L)
  expect_true(jaxR:::aval_of(w)$weak)
  expect_true(jaxR:::aval_of(lax_gather(w, iv, dn1, 1L))$weak)
  expect_false(jaxR:::aval_of(lax_gather(v, iv, dn1, 1L))$weak)
  expect_identical(dtype(lax_gather(a5, iv, dn1, 1L)), "int32")
  # Integer valued R numerics are accepted as indices.
  expect_np(lax_gather(v, matrix(c(0, 0, 2), 3, 1), dn1, 1L), c(10, 10, 12), 3L, "float32")
  # An empty result needs no gather.
  expect_np(lax_gather(v, np_arr(integer(0), c(0L, 1L)), dn1, 1L), numeric(0), 0L, "float32")
  expect_identical(eval_shape(function(a, i) lax_gather(a, i, dn_mid, c(2L, 1L, 4L)),
                              x3, imid)$shape, c(2L, 2L, 4L))
})

test_that("gather errors", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_error(lax_gather(v, as_jax(matrix(0.5, 1, 1)), dn1, 1L),
               "indices must have an integer type", class = "jax_value_error")
  expect_error(lax_gather(v, matrix(0.5, 1, 1), dn1, 1L),
               "indices must have an integer type", class = "jax_value_error")
  expect_error(lax_gather(v, iv, dn1, 4L), class = "jax_type_error", fixed = TRUE,
               "Slice size at index 0 in gather op is out of range, must be within [0, 3 + 1), got 4.")
  expect_error(lax_gather(m, np_arr(0L, c(1L, 1L)), lax_GatherDimensionNumbers(1L, 0L, 0L), c(2L, 4L)),
               class = "jax_type_error", fixed = TRUE,
               "Gather op can only collapse slice dims with bound 1, but bound is 2 for index 0 at position 0.")
  expect_error(lax_gather(x3, np_arr(1L, c(1L, 1L)), lax_GatherDimensionNumbers(c(2L, 1L), 0L, 0L),
                          c(1L, 3L, 4L)),
               "offset_dims in gather op must be sorted; got (2, 1)", fixed = TRUE,
               class = "jax_type_error")
  expect_error(lax_gather(v, np_arr(c(0L, 1L), c(1L, 2L)), dn1, 1L), class = "jax_type_error",
               fixed = TRUE, paste0(
                 "Gather op has 1 elements in start_index_map and the bound of dimension ",
                 "index_vector_dim=1 of indices is 2. These two numbers must be equal."))
  expect_error(lax_gather(v, iv, lax_GatherDimensionNumbers(integer(0), 0L, 3L), 1L),
               "Invalid start_index_map; domain is [0, 1), got: 0->3.", fixed = TRUE,
               class = "jax_type_error")
  expect_error(lax_gather(v, iv, dn1, c(1L, 1L)), class = "jax_type_error", fixed = TRUE,
               paste0("Gather op must have one slice size for every input dimension; got: ",
                      "len(slice_sizes)=2, input_shape.rank=1"))
  expect_error(lax_gather(m, np_arr(0L, c(1L, 1L)), dn1, c(1L, 4L)), class = "jax_type_error",
               fixed = TRUE, paste0(
                 "All components of the offset index in a gather op must either be a offset ",
                 "dimension or explicitly collapsed/batching; got len(slice_sizes)=2, ",
                 "output_slice_sizes=(), collapsed_slice_dims=(0,), operand_batching_dims=()."))
  expect_error(lax_gather(v, as_jax(0L), dn1, 1L), class = "jax_type_error", fixed = TRUE,
               paste0("Gather index leaf dimension must be within [0, rank(indices) + 1). ",
                      "rank(indices) is 0 and gather index leaf dimension is -1."))
  expect_error(lax_gather(x3, np_arr(rep(0L, 6), c(3L, 2L, 1L)), dn_bat, c(1L, 1L, 4L)),
               class = "jax_type_error", fixed = TRUE, paste0(
                 "Gather op requires operand batching dimensions and indices batching ",
                 "dimensions to have the same shape, got (2,) and (3,)."))
  expect_error(lax_gather(v, iv, dn1, 1L, mode = "foo"), "Unknown gather mode \"foo\"",
               class = "jax_value_error")
  expect_error(lax_gather(v, iv, dn1, 1L, mode = "one_hot"), "one_hot")
})

test_that("gather derivatives match the wheel and finite differences", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(jvp(function(x) lax_gather(x, iv, dn1, 1L), list(v), list(tv))[[2L]],
            c(1, 1, 100), 3L, "float32")
  expect_np(vjp(function(x) lax_gather(x, iv, dn1, 1L), v)[[2L]](tv)[[1L]],
            c(11, 0, 100), 3L, "float32")
  expect_np(grad(function(x) sum(lax_gather(x, oob, dn1, 1L, mode = "fill")^2))(v),
            c(20, 0, 0), 3L, "float32")
  expect_np(grad(function(x) sum(lax_gather(x, oob, dn1, 1L, mode = "clip")^2))(v),
            c(40, 0, 24), 3L, "float32")
  expect_np(grad(function(x) sum(lax_gather(x, oob, dn1, 1L)^2))(v), c(20, 0, 0), 3L, "float32")
  expect_np(grad(function(x) sum(grad(function(y) sum(lax_gather(y, iv, dn1, 1L)^3))(x)))(v),
            c(120, 0, 72), 3L, "float32")
  expect_np(grad(function(x) sum(lax_gather(x, ibat, dn_bat, c(1L, 1L, 4L))^2))(x3),
            c(0, 2, 4, 6, 0, 0, 0, 0, 16, 18, 20, 22, 0, 0, 0, 0, 64, 68, 72, 76, 0, 0, 0, 0),
            c(2L, 3L, 4L), "float32")
  # The transpose of a gather is a scatter-add.
  ct <- linear_transpose(function(a) lax_gather(a, iv, dn1, 1L), v)(tv)[[1L]]
  expect_np(ct, c(11, 0, 100), 3L, "float32")
  expect_same(jit(grad(function(a) sum(lax_gather(a, iv, dn1, 1L)^2)))(v),
              grad(function(a) sum(lax_gather(a, iv, dn1, 1L)^2))(v))
  with_x64({
    x <- as_jax(c(0.3, -1.2, 2.5, 0.7, 1.9, -0.4, 0.1, 3.2, -2.2, 1.1, 0.6, -0.9), "float64")
    xm <- jnp_reshape(x, c(3L, 4L))
    check_grad_fd(function(a) sum(lax_gather(a, iv, dn1, 1L)^3), as_jax(c(0.3, -1.2, 2.5), "float64"))
    check_grad_fd(function(a) sum(sin(lax_gather(jnp_reshape(a, c(3L, 4L)), iwin, dn_win, c(2L, 2L)))), x)
    check_grad_fd(function(a) {
      sum(lax_gather(jnp_reshape(a, c(3L, 4L)), iwin_oob, dn_win, c(2L, 2L), mode = "clip")^2)
    }, x)
    check_grad_fd(function(a) {
      sum(lax_gather(jnp_reshape(a, c(3L, 4L)), iwin_oob, dn_win, c(2L, 2L), mode = "fill",
                     fill_value = 0)^2)
    }, x)
    # Second order: Hessian vector product through gather.
    check_grad_fd(function(a) {
      sum(grad(function(b) sum(lax_gather(b, iv, dn1, 1L)^3))(a) * c(1, 2, 3))
    }, as_jax(c(0.3, -1.2, 2.5), "float64"))
  })
})

test_that("gather vmap matches a stacked loop", {
  skip_if_no_backend()
  slicing_fixtures()
  X <- np_arr(c(0:11, 20 + 0:11), c(2L, 3L, 4L), "float32")
  I <- np_arr(c(0L, 2L, 2L, 1L, 1L, 5L), c(2L, 3L, 1L))
  i0 <- row_of(I, 0L)
  for (mode in list(NULL, "clip", "fill")) {
    f <- function(a, i) lax_gather(a, i, dn_rows_m, c(1L, 4L), mode = mode)
    dn_rows_m <- lax_GatherDimensionNumbers(1L, 0L, 0L)
    # operand batched, indices batched, both batched
    expect_same(vmap(function(a) f(a, i0))(X), stack_loop(function(a) f(a, i0), 2L, X))
    expect_same(vmap(function(i) f(m, i))(I), stack_loop(function(i) f(m, i), 2L, I))
    expect_same(vmap(f)(X, I), stack_loop(f, 2L, X, I))
  }
  dn_rows_m <- lax_GatherDimensionNumbers(1L, 0L, 0L)
  f <- function(a, i) lax_gather(a, i, dn_rows_m, c(1L, 4L))
  # Batch dimensions that are not leading, and a non-leading output axis.
  Xt <- jnp_transpose(X, c(1L, 2L, 0L))
  expect_same(vmap(f, in_axes = list(2L, 0L))(Xt, I), stack_loop(f, 2L, X, I))
  It <- jnp_transpose(I, c(1L, 0L, 2L))
  expect_same(vmap(f, in_axes = list(0L, 1L))(X, It), stack_loop(f, 2L, X, I))
  expect_same(vmap(f, out_axes = 1L)(X, I), jnp_moveaxis(stack_loop(f, 2L, X, I), 0L, 1L))
  # A gather that already has batching dimensions, under two levels of vmap.
  g <- function(a, i) lax_gather(a, i, dn_bat, c(1L, 1L, 4L))
  X4 <- np_arr(0:47, c(2L, 2L, 3L, 4L), "float32")
  I4 <- np_arr(c(0L, 2L, 1L, 1L, 2L, 2L, 0L, 1L), c(2L, 2L, 2L, 1L))
  expect_same(vmap(g)(X4, I4), stack_loop(g, 2L, X4, I4))
  expect_same(vmap(function(a) g(a, ibat))(X4), stack_loop(function(a) g(a, ibat), 2L, X4))
  expect_same(vmap(function(i) g(x3, i))(I4), stack_loop(function(i) g(x3, i), 2L, I4))
  h <- function(a, i) lax_gather(a, i, dn1, 1L)
  X2 <- np_arr(0:11, c(2L, 2L, 3L), "float32")
  I2 <- np_arr(c(0L, 0L, 2L, 1L, 1L, 2L, 0L, 1L), c(2L, 2L, 2L, 1L))
  expect_same(vmap(vmap(h))(X2, I2), stack_loop(function(a, i) stack_loop(h, 2L, a, i), 2L, X2, I2))
  # jit of vmap and grad of vmap
  expect_same(jit(vmap(f))(X, I), stack_loop(f, 2L, X, I))
  expect_np(grad(function(a) sum(vmap(f)(a, I)))(X),
            c(1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0),
            c(2L, 3L, 4L), "float32")
})

test_that("scatter family matches the wheel", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(lax_scatter_add(v, iv, u, sd1), c(13, 11, 15), 3L, "float32")
  expect_np(lax_scatter_sub(v, iv, u, sd1), c(7, 11, 9), 3L, "float32")
  expect_np(lax_scatter_mul(v, iv, u, sd1), c(20, 11, 36), 3L, "float32")
  expect_np(lax_scatter_min(v, iv, u2, sd1), c(2, 11, 3), 3L, "float32")
  expect_np(lax_scatter_max(v, iv, u2, sd1), c(100, 11, 12), 3L, "float32")
  expect_np(lax_scatter(v, iu, u, sd1), c(1, 2, 3), 3L, "float32")
  expect_np(lax_scatter_apply(v, iv, function(t) t * 2, sd1, update_shape = 3L),
            c(40, 11, 24), 3L, "float32")
  expect_np(lax_scatter(m, irow2, urow, sd_rows),
            c(104, 105, 106, 107, 4, 5, 6, 7, 100, 101, 102, 103), c(3L, 4L), "float32")
  expect_np(lax_scatter_add(m, irow2, urow, sd_rows),
            c(104, 106, 108, 110, 4, 5, 6, 7, 108, 110, 112, 114), c(3L, 4L), "float32")
  expect_np(lax_scatter_add(m, iwin, uwin, sd_win),
            c(0, 2, 3, 3, 4, 6, 8, 8, 8, 9, 11, 12), c(3L, 4L), "float32")
  expect_np(lax_scatter_add(x3, ibat, ubat, sd_bat),
            c(0, 2, 4, 6, 4, 5, 6, 7, 12, 14, 16, 18, 12, 13, 14, 15, 36, 39, 42, 45, 20, 21, 22,
            23), c(2L, 3L, 4L), "float32")
  expect_np(lax_scatter_mul(jnp_arange(1L, 4L), iv, c(2L, 3L, 4L), sd1), c(6, 2, 12), 3L, "int32")
})

test_that("scatter out of bounds modes match the wheel", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(lax_scatter_add(v, oob, u, sd1), c(11, 11, 12), 3L, "float32")
  expect_np(lax_scatter_add(v, oob, u, sd1, mode = "clip"), c(14, 11, 14), 3L, "float32")
  expect_np(lax_scatter_add(v, oob, u, sd1, mode = "promise_in_bounds"),
            c(11, 11, 12), 3L, "float32")
  expect_np(lax_scatter(v, oob, u, sd1), c(1, 11, 12), 3L, "float32")
  expect_np(lax_scatter(v, oob, u, sd1, mode = "clip"), c(3, 11, 2), 3L, "float32")
  expect_np(lax_scatter(v, oob, u, sd1, mode = "promise_in_bounds"), c(1, 11, 12), 3L, "float32")
  expect_np(lax_scatter_add(m, iwin_oob, uwin, sd_win),
            c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), c(3L, 4L), "float32")
  expect_np(lax_scatter_add(m, iwin_oob, uwin, sd_win, mode = "clip"),
            c(0, 2, 3, 3, 4, 6, 8, 8, 8, 9, 11, 12), c(3L, 4L), "float32")
})

test_that("scatter under jit matches eager evaluation and the wheel lowering", {
  skip_if_no_backend()
  slicing_fixtures()
  ops <- list(lax_scatter_add, lax_scatter_sub, lax_scatter_mul, lax_scatter_min,
              lax_scatter_max)
  for (op in ops) {
    expect_same(jit(function(a, i, b) op(a, i, b, sd1))(v, iv, u2), op(v, iv, u2, sd1))
    expect_same(jit(function(a, i, b) op(a, i, b, sd_bat))(x3, ibat, ubat),
                op(x3, ibat, ubat, sd_bat))
    expect_same(jit(function(a, i, b) op(a, i, b, sd_win, mode = "clip"))(m, iwin_oob, uwin),
                op(m, iwin_oob, uwin, sd_win, mode = "clip"))
  }
  expect_same(jit(function(a, i, b) lax_scatter(a, i, b, sd_rows))(m, irow2, urow),
              lax_scatter(m, irow2, urow, sd_rows))
  apply2 <- function(a, i) lax_scatter_apply(a, i, function(t) t * 2 + 1, sd1, update_shape = 3L)
  expect_same(jit(apply2)(v, iv), apply2(v, iv))
  expect_np(apply2(v, iv), c(43, 11, 25), 3L, "float32")
  # A function closing over a (concrete) array constant.
  k <- as_jax(c(2, 3))
  expect_np(lax_scatter_apply(v, iv, function(t) t * jnp_sum(k), sd1, update_shape = 3L),
            c(250, 11, 60), 3L, "float32")
  txt <- lower_text(function(a, i, b) lax_scatter_add(a, i, b, sd_bat), x3, ibat, ubat)
  expect_match(txt, paste0(
    "\"stablehlo.scatter\"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, ",
    "scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], ",
    "inserted_window_dims = [1], input_batching_dims = [0], ",
    "scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], ",
    "index_vector_dim = 2>, unique_indices = false}>"), fixed = TRUE)
  expect_match(txt, "\"stablehlo.add\"", fixed = TRUE)
  txt <- lower_text(function(a, i, b) lax_scatter(a, i, b, sd1, mode = "clip"), v, iv, u)
  expect_match(txt, "\"stablehlo.clamp\"", fixed = TRUE)
})

test_that("scatter dtype and weak type rules and errors", {
  skip_if_no_backend()
  slicing_fixtures()
  w <- lax_broadcast(1, 3L)
  expect_true(jaxR:::aval_of(lax_scatter_add(w, iv, u, sd1))$weak)
  expect_false(jaxR:::aval_of(lax_scatter_add(v, iv, w, sd1))$weak)
  expect_identical(dtype(lax_scatter_add(a5, np_arr(1L, c(1L, 1L)), np_arr(7L, 1L), sd1)), "int32")
  expect_np(lax_scatter_max(as_jax(c(TRUE, FALSE, FALSE)), iv, as_jax(c(FALSE, FALSE, TRUE)), sd1),
            c(TRUE, FALSE, TRUE), 3L, "bool")
  expect_error(lax_scatter_add(v, iv, jnp_ones(c(3L, 1L)), sd1),
               "Updates tensor must be of rank 1; got 2.", class = "jax_type_error")
  expect_error(lax_scatter_add(v, iv, jnp_ones(2L), sd1), class = "jax_type_error", fixed = TRUE,
               paste0("Bounds of the scatter dimensions of updates must be the same as the ",
                      "bounds of the corresponding dimensions of scatter indices. For scatter ",
                      "dimension 0, updates bound is 2, indices bound is 3."))
  expect_error(lax_scatter_add(v, iv, as_jax(c(1L, 2L, 3L)), sd1), class = "jax_type_error",
               "lax.scatter requires arguments to have the same dtypes, got float32, int32.",
               fixed = TRUE)
  expect_error(lax_scatter_add(v, as_jax(matrix(c(0.5, 1, 2), 3, 1)), u, sd1),
               "indices must have an integer type", class = "jax_value_error")
  expect_error(lax_scatter_add(m, iv, u, sd1), class = "jax_type_error",
               "Scatter op has window of size 1; doesn't match operand of rank 2.", fixed = TRUE)
  expect_error(lax_scatter_add(m, np_arr(0L, c(1L, 1L)), jnp_ones(c(1L, 5L)), sd_rows),
               class = "jax_type_error", fixed = TRUE, paste0(
                 "Bounds of the window dimensions of updates must not exceed the bounds of ",
                 "the corresponding dimensions of operand. For dimension 1, updates bound is ",
                 "5, operand bound is 4."))
  expect_error(lax_scatter_add(v, iv, u, lax_ScatterDimensionNumbers(integer(0), 0L, 5L)),
               "Invalid scatter_dims_to_operand_dims mapping; domain is [0, 1), got: 0->5.",
               fixed = TRUE, class = "jax_type_error")
  b3 <- as_jax(c(TRUE, FALSE, TRUE))
  expect_error(lax_scatter_add(b3, iv, b3, sd1), "add does not accept dtype bool at position 0",
               class = "jax_type_error")
  expect_error(lax_scatter_apply(v, iv, function(t) t > 0, sd1, update_shape = 3L),
               "scatter_apply func must map a scalar", class = "jax_type_error")
  expect_error(jit(function(a, k) {
    lax_scatter_apply(a, iv, function(t) t * k, sd1, update_shape = 3L)
  })(v, 2), "can't close over Tracers", class = "jax_not_implemented_error")
})

test_that("scatter derivatives match the wheel and finite differences", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(jvp(function(a, b) lax_scatter_add(a, iv, b, sd1), list(v, u), list(tv, tu))[[2L]],
            c(1.75, 10, 102), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter_add(a, iv, u, sd1)^2))(v),
            c(26, 22, 30), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_add(v, iv, b, sd1)^2))(u),
            c(26, 26, 30), 3L, "float32")
  expect_np(jvp(function(a, b) lax_scatter_sub(a, iv, b, sd1), list(v, u), list(tv, tu))[[2L]],
            c(0.25, 10, 98), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter_sub(a, iv, u, sd1)^2))(v),
            c(14, 22, 18), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_sub(v, iv, b, sd1)^2))(u),
            c(-14, -14, -18), 3L, "float32")
  expect_np(jvp(function(a, b) lax_scatter_min(a, iv, b, sd1), list(v, u2), list(tv, tu))[[2L]],
            c(0.25, 10, 2), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter_min(a, iv, u2, sd1)^2))(v), c(0, 22, 0), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_min(v, iv, b, sd1)^2))(u2), c(0, 4, 6), 3L, "float32")
  expect_np(jvp(function(a, b) lax_scatter_max(a, iv, b, sd1), list(v, u2), list(tv, tu))[[2L]],
            c(0.5, 10, 100), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter_max(a, iv, u2, sd1)^2))(v),
            c(0, 22, 24), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_max(v, iv, b, sd1)^2))(u2),
            c(200, 0, 0), 3L, "float32")
  expect_np(jvp(function(a, b) lax_scatter(a, iv, b, sd1), list(v, u), list(tv, tu))[[2L]],
            c(0.25, 10, 2), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter(a, iv, u, sd1)^2))(v), c(0, 22, 0), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter(v, iv, b, sd1)^2))(u), c(0, 4, 6), 3L, "float32")
  expect_np(jvp(function(a, b) lax_scatter_mul(a, iu, b, sd1, unique_indices = TRUE), list(v, u), list(tv, tu))[[2L]],
            c(6, 22.75, 324), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter_mul(a, iv, u, sd1)^2))(v),
            c(80, 22, 216), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_mul(v, iu, b, sd1, unique_indices = TRUE)^2))(u),
            c(200, 484, 864), 3L, "float32")
  expect_np(grad(function(a) sum(lax_scatter(a, iu2, u_2, sd1, unique_indices = TRUE)^2))(v),
            c(0, 0, 24), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter(v, iu2, b, sd1, unique_indices = TRUE)^2))(u_2),
            c(2, 4), 2L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_add(v, oob, b, sd1)^2))(u), c(22, 0, 0), 3L, "float32")
  expect_np(grad(function(b) sum(lax_scatter_add(v, oob, b, sd1, mode = "clip")^2))(u),
            c(28, 28, 28), 3L, "float32")
  expect_np(grad(function(a) sum(grad(function(b) sum(lax_scatter_add(a, iv, b, sd1)^3))(u)))(v),
            c(156, 0, 90), 3L, "float32")
  expect_np(vjp(function(b) lax_scatter_add(v, iv, b, sd1), u)[[2L]](tv)[[1L]],
            c(1, 1, 100), 3L, "float32")
  expect_np(vjp(function(b) lax_scatter_sub(v, iv, b, sd1), u)[[2L]](tv)[[1L]],
            c(-1, -1, -100), 3L, "float32")
  expect_error(grad(function(b) sum(lax_scatter_mul(v, iv, b, sd1)))(u),
               "scatter_mul gradients are only implemented if `unique_indices=True`",
               class = "jax_not_implemented_error")
  expect_error(grad(function(a) {
    sum(lax_scatter_apply(a, iv, function(t) t * 2, sd1, update_shape = 3L))
  })(v), "scatter_apply JVP not implemented", class = "jax_not_implemented_error")
  expect_error(linear_transpose(function(a) lax_scatter(a, iv, u, sd1), v)(v),
               "scatter transpose is only implemented where unique_indices=True",
               class = "jax_not_implemented_error")
  # Transposes: scatter-add is the transpose of gather; a unique scatter zeroes the
  # updated entries of the operand cotangent.
  expect_np(linear_transpose(function(b) lax_scatter_add(jnp_zeros(3L), iv, b, sd1), u)(tv)[[1L]],
            c(1, 1, 100), 3L, "float32")
  expect_np(linear_transpose(function(a) lax_scatter(a, iu2, jnp_zeros(2L), sd1,
                                                     unique_indices = TRUE), v)(tv)[[1L]],
            c(0, 0, 100), 3L, "float32")
  expect_np(linear_transpose(function(b) lax_scatter(jnp_zeros(3L), iu2, b, sd1,
                                                     unique_indices = TRUE), u_2)(tv)[[1L]],
            c(1, 10), 2L, "float32")
  with_x64({
    a0 <- as_jax(c(0.3, -1.2, 2.5), "float64")
    b0 <- as_jax(c(0.9, 1.7, -0.6), "float64")
    for (op in list(lax_scatter_add, lax_scatter_sub, lax_scatter_min, lax_scatter_max,
                    lax_scatter)) {
      check_grad_fd(function(a) sum(op(a, iv, b0, sd1)^3), a0)
      check_grad_fd(function(b) sum(op(a0, iv, b, sd1)^3), b0)
      check_grad_fd(function(b) sum(op(a0, oob, b, sd1, mode = "clip")^3), b0)
      check_grad_fd(function(b) sum(op(a0, oob, b, sd1)^3), b0)
    }
    check_grad_fd(function(a) sum(lax_scatter_mul(a, iv, b0, sd1)^3), a0)
    check_grad_fd(function(b) sum(lax_scatter_mul(a0, iu, b, sd1, unique_indices = TRUE)^3), b0)
    check_grad_fd(function(a) sum(lax_scatter(a, iu, b0, sd1, unique_indices = TRUE)^3), a0)
    check_grad_fd(function(b) sum(lax_scatter(a0, iu, b, sd1, unique_indices = TRUE)^3), b0)
    # Windows and batching dimensions.
    m0 <- as_jax(seq(-1, 1, length.out = 12), "float64")
    w0 <- as_jax(seq(0.5, 2, length.out = 8), "float64")
    check_grad_fd(function(a) {
      sum(lax_scatter_add(jnp_reshape(a, c(3L, 4L)), iwin, jnp_reshape(w0, c(2L, 2L, 2L)), sd_win)^3)
    }, m0)
    check_grad_fd(function(b) {
      sum(lax_scatter(jnp_reshape(m0, c(3L, 4L)), iwin, jnp_reshape(b, c(2L, 2L, 2L)), sd_win)^3)
    }, w0)
    x0 <- as_jax(seq(-2, 2, length.out = 24), "float64")
    ub0 <- as_jax(seq(0.1, 1.6, length.out = 16), "float64")
    check_grad_fd(function(b) {
      sum(lax_scatter_add(jnp_reshape(x0, c(2L, 3L, 4L)), ibat, jnp_reshape(b, c(2L, 2L, 4L)),
                          sd_bat)^3)
    }, ub0)
    # Second order through scatter-add.
    check_grad_fd(function(a) {
      sum(grad(function(b) sum(lax_scatter_add(a, iv, b, sd1)^3))(b0) * c(1, 2, 3))
    }, a0)
  })
})

test_that("scatter vmap matches a stacked loop", {
  skip_if_no_backend()
  slicing_fixtures()
  full <- list(np_arr(c(0:11, 20 + 0:11), c(2L, 3L, 4L), "float32"),
               np_arr(c(0L, 2L, 2L, 1L, 1L, 5L), c(2L, 3L, 1L)),
               np_arr(seq(0.5, 12, by = 0.5), c(2L, 3L, 4L), "float32"))
  one <- lapply(full, row_of, k = 0L)
  masks <- list(c(TRUE, FALSE, FALSE), c(FALSE, TRUE, FALSE), c(FALSE, FALSE, TRUE),
                c(TRUE, TRUE, FALSE), c(TRUE, FALSE, TRUE), c(FALSE, TRUE, TRUE),
                c(TRUE, TRUE, TRUE))
  check <- function(f) {
    for (mask in masks) {
      args <- Map(function(a, b, batched) if (batched) a else b, full, one, mask)
      in_axes <- lapply(mask, function(batched) if (batched) 0L else NULL)
      want <- jnp_stack(lapply(0:1, function(k) {
        do.call(f, Map(function(a, b, batched) if (batched) row_of(a, k) else b, full, one, mask))
      }))
      expect_same(do.call(vmap(f, in_axes = in_axes), args), want)
    }
  }
  for (op in list(lax_scatter_add, lax_scatter_sub, lax_scatter_mul, lax_scatter_min,
                  lax_scatter_max)) {
    check(function(a, i, b) op(a, i, b, sd_rows))
  }
  check(function(a, i, b) lax_scatter_add(a, i, b, sd_rows, mode = "clip"))
  # Overwriting scatter with indices that stay unique per example.
  full[[2L]] <- np_arr(c(0L, 2L, 1L, 1L, 0L, 5L), c(2L, 3L, 1L))
  one <- lapply(full, row_of, k = 0L)
  check(function(a, i, b) lax_scatter(a, i, b, sd_rows))
  check(function(a, i, b) lax_scatter(a, i, b, sd_rows, mode = "clip"))
  check(function(a, i, b) {
    lax_scatter_apply(a, i, function(t) t * 3, sd_rows, update_shape = c(3L, 4L)) + 0 * sum(b)
  })
  # Non leading batch dimensions and a scatter that already has batching dimensions.
  f <- function(a, i, b) lax_scatter_add(a, i, b, sd_rows)
  expect_same(vmap(f, in_axes = list(1L, 0L, 2L))(jnp_transpose(full[[1L]], c(1L, 0L, 2L)),
                                                   full[[2L]],
                                                   jnp_transpose(full[[3L]], c(1L, 2L, 0L))),
              stack_loop(f, 2L, full[[1L]], full[[2L]], full[[3L]]))
  g <- function(a, i, b) lax_scatter_add(a, i, b, sd_bat)
  X4 <- np_arr(0:47, c(2L, 2L, 3L, 4L), "float32")
  I4 <- np_arr(c(0L, 2L, 1L, 1L, 2L, 2L, 0L, 1L), c(2L, 2L, 2L, 1L))
  U4 <- np_arr(0:31, c(2L, 2L, 2L, 4L), "float32")
  expect_same(vmap(g)(X4, I4, U4), stack_loop(g, 2L, X4, I4, U4))
  expect_same(jit(vmap(g))(X4, I4, U4), stack_loop(g, 2L, X4, I4, U4))
  expect_same(vmap(function(i) g(x3, i, ubat))(I4), stack_loop(function(i) g(x3, i, ubat), 2L, I4))
  # grad of vmap of scatter
  expect_np(grad(function(b) sum(vmap(f)(full[[1L]], full[[2L]], b)^2))(full[[3L]])[1, 2],
            c(0, 0, 0, 0), 4L, "float32")
})

test_that("index_take and the dynamic in_dim wrappers match the wheel", {
  skip_if_no_backend()
  slicing_fixtures()
  expect_np(lax_index_take(m, list(c(0, 2, -1), c(1, 3, 5)), c(0, 1)), c(1, 11, 9), 3L, "float32")
  expect_np(lax_index_take(x3, list(c(1, 0, 1)), 0L),
            c(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), c(3L, 3L, 4L), "float32")
  expect_np(lax_index_take(x3, list(c(1, 0, 1), c(3, 0, 2)), c(0, 2)),
            c(15, 19, 23, 0, 4, 8, 14, 18, 22), c(3L, 3L), "float32")
  expect_np(lax_dynamic_slice_in_dim(a5, 1, 3), c(1, 2, 3), 3L, "int32")
  expect_np(lax_dynamic_slice_in_dim(a5, 4, 3), c(2, 3, 4), 3L, "int32")
  expect_np(lax_dynamic_slice_in_dim(a5, -2, 2), c(3, 4), 2L, "int32")
  expect_np(lax_dynamic_slice_in_dim(a5, as_jax(-2L), 2, allow_negative_indices = FALSE),
            c(0, 1), 2L, "int32")
  expect_np(lax_dynamic_slice_in_dim(m34, 1, 2, axis = 1), c(1, 2, 5, 6, 9, 10), c(3L, 2L), "int32")
  expect_np(lax_dynamic_slice_in_dim(m34, 1, 2, axis = -1),
            c(1, 2, 5, 6, 9, 10), c(3L, 2L), "int32")
  expect_np(lax_dynamic_index_in_dim(a5, 1), 1, 1L, "int32")
  expect_np(lax_dynamic_index_in_dim(a5, 1, keepdims = FALSE), 1, integer(0), "int32")
  expect_np(lax_dynamic_index_in_dim(m34, 1, axis = 1, keepdims = FALSE), c(1, 5, 9), 3L, "int32")
  expect_np(lax_dynamic_index_in_dim(m34, -1, axis = 0), c(8, 9, 10, 11), c(1L, 4L), "int32")
  expect_np(lax_dynamic_index_in_dim(a5, 9, keepdims = FALSE), 4, integer(0), "int32")
  expect_np(lax_dynamic_update_slice_in_dim(z6, o3, 2, 0), c(0, 0, 1, 1, 1, 0), 6L, "float32")
  expect_np(lax_dynamic_update_slice_in_dim(z6, o3, 3, 0), c(0, 0, 0, 1, 1, 1), 6L, "float32")
  expect_np(lax_dynamic_update_slice_in_dim(z6, o3, 5, 0), c(0, 0, 0, 1, 1, 1), 6L, "float32")
  expect_np(lax_dynamic_update_slice_in_dim(z6, o3, -4, 0), c(0, 0, 1, 1, 1, 0), 6L, "float32")
  expect_np(lax_dynamic_update_slice_in_dim(jnp_zeros(c(4L, 4L)), jnp_ones(c(2L, 3L)), 1, 0),
            c(0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0), c(4L, 4L), "float32")
  expect_np(lax_dynamic_update_slice_in_dim(jnp_zeros(c(4L, 4L)), jnp_ones(c(2L, 3L)), 2, 1),
            c(0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), c(4L, 4L), "float32")
  expect_np(lax_dynamic_update_index_in_dim(z6, 1, 2, 0), c(0, 0, 1, 0, 0, 0), 6L, "float32")
  expect_np(lax_dynamic_update_index_in_dim(z6, array(1, 1), 2, 0),
            c(0, 0, 1, 0, 0, 0), 6L, "float32")
  expect_np(lax_dynamic_update_index_in_dim(z6, 1, 10, 0), c(0, 0, 0, 0, 0, 1), 6L, "float32")
  expect_np(lax_dynamic_update_index_in_dim(jnp_zeros(c(4L, 4L)), jnp_ones(4L), 1, 0),
            c(0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), c(4L, 4L), "float32")
  expect_np(lax_dynamic_update_index_in_dim(jnp_zeros(c(4L, 4L)), jnp_ones(4L), -1, 1),
            c(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1), c(4L, 4L), "float32")
  expect_np(grad(function(a) sum(lax_dynamic_slice_in_dim(a, 4, 3)^2))(f5),
            c(0, 0, 4, 6, 8), 5L, "float32")
  expect_np(grad(function(a) sum(lax_dynamic_update_index_in_dim(a, 7, 1, 0)^2))(f5),
            c(0, 0, 4, 6, 8), 5L, "float32")
  expect_np(grad(function(b) sum(lax_dynamic_update_slice_in_dim(f5, b, 3, 0)^2))(o3),
            c(2, 2, 2), 3L, "float32")
  # index_take: gradient and vmap, against the wheel and a stacked loop.
  I2 <- np_arr(c(0L, 2L, 1L, 4L), c(2L, 2L))
  expect_np(grad(function(a) sum(lax_index_take(a, list(c(1, 0, 1), c(3, 0, 2)), c(0, 2))^2))(x3),
            c(0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 28, 30, 0, 0, 36, 38, 0, 0, 44, 46),
            c(2L, 3L, 4L), "float32")
  expect_np(vmap(function(a) lax_index_take(a, list(c(0, 2, -1)), 0L))(x3),
            c(0, 1, 2, 3, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21, 22,
            23), c(2L, 3L, 4L), "float32")
  expect_np(vmap(function(i) lax_index_take(m, list(i), 0L))(I2),
            c(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 4, 5, 6, 7), c(2L, 2L, 4L), "float32")
  take0 <- function(a) lax_index_take(a, list(c(0, 2, -1)), 0L)
  expect_same(vmap(take0)(x3), stack_loop(take0, 2L, x3))
  take_i <- function(i) lax_index_take(m, list(i), 0L)
  expect_same(vmap(take_i)(I2), stack_loop(take_i, 2L, I2))
  with_x64({
    check_grad_fd(function(a) {
      sum(lax_index_take(jnp_reshape(a, c(3L, 4L)), list(c(0, 2, -1), c(1, 3, 5)), c(0, 1))^3)
    }, as_jax(seq(-1, 1, length.out = 12), "float64"))
  })
  expect_same(jit(function(a) lax_index_take(a, list(c(1, 0, 1), c(3, 0, 2)), c(0, 2)))(x3),
              lax_index_take(x3, list(c(1, 0, 1), c(3, 0, 2)), c(0, 2)))
  expect_error(lax_dynamic_slice_in_dim(a5, -2, 2, allow_negative_indices = FALSE),
               "Index -2 is out of bounds for dimension 5 if allow_negative_indices=False",
               class = "jax_value_error")
  expect_error(lax_dynamic_update_index_in_dim(jnp_zeros(c(2L, 2L, 2L)), 1, 0, 0),
               "rank of operand or one less", class = "jax_type_error")
})

test_that("lax_dynamic_slice and lax_dynamic_update_slice batch through gather and scatter", {
  skip_if_no_backend()
  slicing_fixtures()
  ib <- np_arr(c(0L, 2L), 2L)
  ib2 <- np_arr(c(-1L, 5L), 2L)
  up <- np_arr(rep(-1, 4), c(2L, 2L), "float32")
  U2 <- np_arr(-(1:8), c(2L, 2L, 2L), "float32")
  Im <- np_arr(c(0L, 3L, -1L), 3L)
  expect_np(vmap(function(a, i) lax_dynamic_slice(a, list(i, 1L), c(2L, 2L)))(x3, ib),
            c(1, 2, 5, 6, 17, 18, 21, 22), c(2L, 2L, 2L), "float32")
  expect_np(vmap(function(a, i) lax_dynamic_slice(a, list(i, 1L), c(2L, 2L)))(x3, ib2),
            c(5, 6, 9, 10, 17, 18, 21, 22), c(2L, 2L, 2L), "float32")
  expect_np(vmap(function(i) lax_dynamic_slice(m, list(1L, i), c(1L, 3L)))(ib2),
            c(5, 6, 7, 5, 6, 7), c(2L, 1L, 3L), "float32")
  expect_np(vmap(function(i, j) lax_dynamic_slice(m, list(i, j), c(2L, 2L)))(ib, ib2),
            c(2, 3, 6, 7, 6, 7, 10, 11), c(2L, 2L, 2L), "float32")
  expect_np(vmap(function(a, i) lax_dynamic_update_slice(a, up, list(i, 1L)))(x3, ib),
            c(0, -1, -1, 3, 4, -1, -1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -1, 19, 20, -1,
            -1, 23), c(2L, 3L, 4L), "float32")
  expect_np(vmap(function(i) lax_dynamic_update_slice(m, up, list(1L, i)))(ib2),
            c(0, 1, 2, 3, 4, 5, -1, -1, 8, 9, -1, -1, 0, 1, 2, 3, 4, 5, -1, -1, 8, 9, -1, -1),
            c(2L, 3L, 4L), "float32")
  expect_np(vmap(function(b, i) lax_dynamic_update_slice(m, b, list(i, 0L)))(U2, ib),
            c(-1, -2, 2, 3, -3, -4, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, -5, -6, 6, 7, -7, -8, 10, 11),
            c(2L, 3L, 4L), "float32")
  expect_np(grad(function(a) sum(vmap(function(r, i) lax_dynamic_slice(r, list(i, 1L), c(2L, 2L)))(a, ib)^2))(x3),
            c(0, 2, 4, 0, 0, 10, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 36, 0, 0, 42, 44, 0), c(2L,
            3L, 4L), "float32")
  expect_np(vmap(function(a, i) lax_dynamic_update_index_in_dim(a, 7, i, 0L))(m, Im),
            c(7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7), c(3L, 4L), "float32")
  expect_np(vmap(function(i) lax_dynamic_update_index_in_dim(m, c(-1, -2, -3), i, 1L))(Im),
            c(-1, 1, 2, 3, -2, 5, 6, 7, -3, 9, 10, 11, 0, 1, 2, -1, 4, 5, 6, -2, 8, 9, 10, -3, 0,
            1, 2, -1, 4, 5, 6, -2, 8, 9, 10, -3), c(3L, 3L, 4L), "float32")
  f <- function(a, i) lax_dynamic_slice(a, list(i, 1L), c(2L, 2L))
  expect_same(vmap(f)(x3, ib2), stack_loop(f, 2L, x3, ib2))
  expect_same(jit(vmap(f))(x3, ib), stack_loop(f, 2L, x3, ib))
  expect_same(vmap(f, in_axes = list(1L, 0L))(jnp_transpose(x3, c(1L, 0L, 2L)), ib),
              stack_loop(f, 2L, x3, ib))
  g <- function(a, b, i) lax_dynamic_update_slice(a, b, list(i, 0L))
  expect_same(vmap(g)(x3, U2, ib), stack_loop(g, 2L, x3, U2, ib))
  expect_same(jit(vmap(g))(x3, U2, ib2), stack_loop(g, 2L, x3, U2, ib2))
  h <- function(a, i) lax_dynamic_update_index_in_dim(a, 7, i, 0L)
  expect_same(vmap(h)(m, Im), stack_loop(h, 3L, m, Im))
  expect_null(jaxR:::.jx$fallback_hits[["dynamic_slice"]])
  expect_null(jaxR:::.jx$fallback_hits[["dynamic_update_slice"]])
})

test_that("dynamic in_dim wrappers take traced indices under jit, grad and vmap", {
  skip_if_no_backend()
  slicing_fixtures()
  sl_dyn <- jit(function(a, i) lax_dynamic_slice_in_dim(a, i, 3L))
  expect_np(sl_dyn(a5, 1L), c(1, 2, 3), 3L, "int32")
  expect_np(sl_dyn(a5, 4L), c(2, 3, 4), 3L, "int32")
  expect_np(sl_dyn(a5, -2L), c(2, 3, 4), 3L, "int32")
  expect_np(sl_dyn(a5, -4L), c(1, 2, 3), 3L, "int32")
  ix_dyn <- jit(function(a, i) lax_dynamic_index_in_dim(a, i, axis = 1L, keepdims = FALSE))
  expect_np(ix_dyn(m34, 1L), c(1, 5, 9), 3L, "int32")
  expect_np(ix_dyn(m34, -1L), c(3, 7, 11), 3L, "int32")
  expect_np(ix_dyn(m34, 9L), c(3, 7, 11), 3L, "int32")
  no_neg <- jit(function(a, i) {
    lax_dynamic_index_in_dim(a, i, keepdims = FALSE, allow_negative_indices = FALSE)
  })
  expect_np(no_neg(a5, -1L), 0, integer(0), "int32")
  up_dyn <- jit(function(a, b, i) lax_dynamic_update_slice_in_dim(a, b, i, 0L))
  expect_np(up_dyn(z6, o3, 2L), c(0, 0, 1, 1, 1, 0), 6L, "float32")
  expect_np(up_dyn(z6, o3, 5L), c(0, 0, 0, 1, 1, 1), 6L, "float32")
  expect_np(up_dyn(z6, o3, -4L), c(0, 0, 1, 1, 1, 0), 6L, "float32")
  ui_dyn <- jit(function(a, i) lax_dynamic_update_index_in_dim(a, 7, i, 0L))
  expect_np(ui_dyn(z6, 10L), c(0, 0, 0, 0, 0, 7), 6L, "float32")
  expect_np(ui_dyn(z6, -6L), c(7, 0, 0, 0, 0, 0), 6L, "float32")
  # Gradients with a traced index.
  expect_np(grad(function(a, i) sum(lax_dynamic_slice_in_dim(a, i, 2L)^2))(f5, as_jax(3L)),
            c(0, 0, 0, 6, 8), 5L, "float32")
  expect_np(jit(grad(function(a, i) sum(lax_dynamic_update_index_in_dim(a, 1, i, 0L)^2)))(f5, 2L),
            c(0, 2, 0, 6, 8), 5L, "float32")
  # Batched indices use the gather and scatter batching rules, not a fallback.
  X <- np_arr(0:11, c(3L, 4L), "float32")
  I <- np_arr(c(0L, 3L, -1L), 3L)
  f <- function(a, i) lax_dynamic_slice_in_dim(a, i, 2L)
  expect_same(vmap(f)(X, I), stack_loop(f, 3L, X, I))
  expect_np(vmap(f)(X, I), c(0, 1, 6, 7, 10, 11), c(3L, 2L), "float32")
  expect_same(vmap(function(i) f(f5, i))(I), stack_loop(function(i) f(f5, i), 3L, I))
  expect_same(vmap(function(a) f(a, 1L))(X), stack_loop(function(a) f(a, 1L), 3L, X))
  X3 <- np_arr(0:23, c(3L, 2L, 4L), "float32")
  f2 <- function(a, i) lax_dynamic_index_in_dim(a, i, axis = 1L, keepdims = FALSE)
  expect_same(vmap(f2)(X3, I), stack_loop(f2, 3L, X3, I))
  expect_same(vmap(f2, in_axes = list(2L, 0L))(jnp_transpose(X3, c(1L, 2L, 0L)), I),
              stack_loop(f2, 3L, X3, I))
  U <- np_arr(-(1:6), c(3L, 2L), "float32")
  g <- function(a, b, i) lax_dynamic_update_slice_in_dim(a, b, i, 0L)
  expect_same(vmap(g)(X, U, I), stack_loop(g, 3L, X, U, I))
  expect_np(vmap(g)(X, U, I), c(-1, -2, 2, 3, 4, 5, -3, -4, 8, 9, -5, -6), c(3L, 4L), "float32")
  u0 <- row_of(U, 0L)
  expect_same(vmap(function(i) g(f5, u0, i))(I), stack_loop(function(i) g(f5, u0, i), 3L, I))
  expect_same(vmap(function(a, i) g(a, u0, i))(X, I),
              stack_loop(function(a, i) g(a, u0, i), 3L, X, I))
  expect_same(vmap(function(b, i) g(f5, b, i))(U, I),
              stack_loop(function(b, i) g(f5, b, i), 3L, U, I))
  expect_same(vmap(function(b) g(f5, b, 2L))(U), stack_loop(function(b) g(f5, b, 2L), 3L, U))
  expect_same(jit(vmap(g))(X, U, I), stack_loop(g, 3L, X, U, I))
  expect_np(grad(function(a) sum(vmap(f)(a, I)^2))(X),
            c(0, 2, 0, 0, 0, 0, 12, 14, 0, 0, 20, 22), c(3L, 4L), "float32")
  expect_null(jaxR:::.jx$fallback_hits[["dynamic_slice"]])
  expect_null(jaxR:::.jx$fallback_hits[["dynamic_update_slice"]])
  with_x64({
    x <- as_jax(c(0.3, -1.2, 2.5, 0.7, 1.9), "float64")
    check_grad_fd(function(a) sum(lax_dynamic_slice_in_dim(a, as_jax(4L), 3L)^3), x)
    check_grad_fd(function(a) sum(lax_dynamic_update_slice_in_dim(a, c(1, 2), as_jax(9L), 0L)^3), x)
    check_grad_fd(function(b) {
      sum(lax_dynamic_update_slice_in_dim(x, b, as_jax(1L), 0L)^3)
    }, as_jax(c(0.4, 0.8), "float64"))
  })
})
