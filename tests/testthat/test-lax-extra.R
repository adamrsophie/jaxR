# Remaining jax.lax functions, custom derivative extras and checkpoint extras
# (R/lax_extra.R). Expected values from jax 0.12.0.1; bit exact floats as hex literals.

# Row major data (as printed by NumPy) to an R array of the given shape.
rm_array <- function(v, ...) {
  s <- c(...)
  if (length(s) < 2L) return(v)
  aperm(array(v, rev(s)), rev(seq_along(s)))
}

lx_err <- function(expr, class, regexp = NULL) {
  if (is.null(regexp)) return(testthat::expect_error(expr, class = class))
  testthat::expect_error(expr, regexp = regexp, class = class, fixed = TRUE)
}

jaxpr_text <- function(cj) paste(utils::capture.output(print(cj)), collapse = "\n")

test_that("enum like objects are accepted by the existing lax functions", {
  skip_if_no_backend()
  expect_identical(lax_Precision$HIGHEST, "HIGHEST")
  expect_jax_equal(lax_dot(matrix(c(1, 3, 2, 4), 2), c(1, 1), precision = lax_Precision$HIGHEST), c(3, 7))
  expect_jax_equal(lax_conv_general_dilated(array(1, c(1, 1, 3, 3)), array(1, c(1, 1, 2, 2)), c(1L, 1L),
                                            "VALID", precision = lax_Precision$HIGH),
                   array(4, c(1, 1, 2, 2)))
  expect_jax_equal(lax_round(c(-1.5, -0.5, 0.5, 1.5, 2.5), rounding_method = lax_RoundingMethod$TO_NEAREST_EVEN),
                   c(-2, 0, 0, 2, 2))
  expect_jax_equal(lax_round(c(-1.5, -0.5, 0.5), rounding_method = lax_RoundingMethod$AWAY_FROM_ZERO),
                   c(-2, -1, 1))
  expect_identical(lax_AccuracyMode, list(HIGHEST = 1L, DEFAULT = 2L))
  expect_jax_equal(lax_dot_general(matrix(1:4, 2), matrix(1:4, 2), lax_DotDimensionNumbers(1L, 0L)),
                   matrix(c(7, 10, 15, 22), 2))
  expect_identical(lax_DotDimensionNumbers(1L, 0L, 2L, 3L), list(list(1L, 0L), list(2L, 3L)))
  out <- lax_conv_general_dilated(array(1, c(1, 3, 3, 1)), array(1, c(2, 2, 1, 1)), c(1L, 1L), "VALID",
                                  dimension_numbers = lax_ConvGeneralDilatedDimensionNumbers("NHWC", "HWIO", "NHWC"))
  expect_jax_equal(out, array(4, c(1, 2, 2, 1)))
  expect_s3_class(lax_ConvGeneralDilatedDimensionNumbers(0:3, 0:3, 0:3), "ConvDimensionNumbers")
  expect_identical(lax_Tolerance(atol = 1e-6)$atol, 1e-6)
  lx_err(lax_Tolerance(), "jax_value_error", "At least one of atol, rtol, or ulps must be set.")
  lx_err(lax_Tolerance(atol = -1), "jax_value_error", "Tolerances must be non-negative.")
  alg <- lax_DotAlgorithm("float32", "float32", "float32")
  expect_s3_class(alg, "jax_DotAlgorithm")
  expect_identical(alg$num_primitive_operations, 1L)
  expect_identical(names(lax_DotAlgorithmPreset)[c(1L, 15L, 16L)], c("DEFAULT", "F32_F32_F32", "F64_F64_F64"))
  # lax.canonicalize_precision
  expect_identical(lax_PrecisionLike("highest"), c("HIGHEST", "HIGHEST"))
  expect_identical(lax_PrecisionLike(c("default", "high")), c("DEFAULT", "HIGH"))
  expect_identical(lax_PrecisionLike(lax_Precision$HIGH), c("HIGH", "HIGH"))
  expect_identical(lax_PrecisionLike("F32_F32_F32"), lax_DotAlgorithmPreset$F32_F32_F32)
  expect_identical(lax_PrecisionLike(alg), alg)
  expect_null(lax_PrecisionLike(NULL))
  lx_err(lax_PrecisionLike("bogus"), "jax_value_error", "Precision argument must be one of")
  rdn <- lax_RaggedDotDimensionNumbers(list(list(1L, 1L), list(integer(0), integer(0))), 0L, 0L)
  expect_s3_class(rdn, "jax_RaggedDotDimensionNumbers")
  expect_identical(rdn$rhs_group_dimensions, 0L)
})

test_that("broadcast_shapes, broadcast_to_rank, collapse, batch_matmul, reciprocal match JAX", {
  skip_if_no_backend()
  expect_identical(lax_broadcast_shapes(c(3L, 1L), c(1L, 4L), c(5L, 1L, 1L)), c(5L, 3L, 4L))
  expect_identical(lax_broadcast_shapes(c(2L, 3L)), c(2L, 3L))
  lx_err(lax_broadcast_shapes(c(3L, 1L), c(4L, 1L)), "jax_value_error",
         "Incompatible shapes for broadcasting: shapes=[(3, 1), (4, 1)]")
  lx_err(lax_broadcast_shapes(c(-1L, 1L), 2L), "jax_type_error", "non-negative")
  expect_identical(dim(lax_broadcast_to_rank(c(1, 2, 3), 3L)), c(1L, 1L, 3L))
  expect_identical(dim(lax_broadcast_to_rank(matrix(1, 2, 3), 1L)), c(2L, 3L))
  expect_jax_equal(lax_broadcast_to_rank(c(1, 2), 2L), matrix(c(1, 2), 1))
  z <- array(0, c(2, 3, 4))
  expect_identical(dim(lax_collapse(z, 0L, 2L)), c(6L, 4L))
  expect_identical(dim(lax_collapse(z, 1L)), c(2L, 12L))
  expect_identical(dim(lax_collapse(z, -2L, -1L)), c(2L, 3L, 4L))
  expect_identical(dim(lax_collapse(z, 0L, 10L)), 24L)
  expect_identical(dim(lax_collapse(z, -5L, 2L)), c(6L, 4L))
  expect_jax_equal(lax_collapse(matrix(1:6, 2, byrow = TRUE), 0L), 1:6)
  lx_err(lax_collapse(z, 2L, 1L), "jax_value_error", "Invalid dimension range passed to collapse: (2, 3, 4)[2:1]")
  a <- lax_reshape(as_jax(0:11, "float32"), c(2L, 2L, 3L))
  b <- lax_reshape(as_jax(0:11, "float32"), c(2L, 3L, 2L))
  expect_jax_equal(lax_batch_matmul(a, b), rm_array(c(10, 13, 28, 40, 172, 193, 244, 274), 2, 2, 2),
                   dtype = "float32")
  expect_jax_equal(jit(lax_batch_matmul)(a, b), rm_array(c(10, 13, 28, 40, 172, 193, 244, 274), 2, 2, 2))
  lx_err(lax_batch_matmul(c(0, 0, 0), matrix(0, 3, 3)), "jax_value_error",
         "Arguments to batch_matmul must be at least 2D, got 1, 2")
  lx_err(lax_batch_matmul(array(0, c(2, 3, 3)), matrix(0, 3, 3)), "jax_value_error",
         "Arguments to batch_matmul must have same ndim, got 3, 2")
  expect_jax_equal(lax_reciprocal(c(1, 2, 4)), c(1, 0.5, 0.25), dtype = "float32")
  lx_err(lax_reciprocal(as_jax(1:2)), "jax_type_error", "Integers cannot be raised to negative powers")
})

test_that("dtype, shape_as_value and empty match JAX", {
  skip_if_no_backend()
  expect_identical(lax_dtype(1.5), "float32")
  expect_identical(lax_dtype(1L), "int32")
  expect_identical(lax_dtype(c(0, 0)), "float32")
  expect_identical(lax_dtype(TRUE), "bool")
  expect_identical(lax_dtype("float64"), "float32")
  expect_identical(lax_dtype(as_jax(1:2, "int8")), "int8")
  lx_err(lax_dtype(NULL), "jax_value_error", "Invalid argument to dtype: None.")
  lx_err(lax_dtype("bogus"), "jax_type_error", "not a valid JAX array type")
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(lax_dtype(1.5), "float64")
    expect_identical(lax_dtype(1L), "int64")
    expect_identical(lax_dtype(as_jax(1, "float32")), "float32")
  })
  expect_jax_equal(lax_shape_as_value(c(2L, 3L)), c(2L, 3L), dtype = "int32")
  expect_jax_equal(lax_shape_as_value(integer(0)), integer(0), dtype = "int32", shape = 0L)
  expect_jax_equal(lax_empty(c(2L, 2L), "float32"), matrix(0, 2, 2), dtype = "float32")
  expect_jax_equal(jit(function() lax_empty(3L, "int32"))(), c(0L, 0L, 0L), dtype = "int32")
})

test_that("log2, mulhi and one_minus_square match JAX", {
  skip_if_no_backend()
  x <- as_jax(c(0.5, 1, 3, 10), "float32")
  expect_equal(as.vector(as.array(lax_log2(x))), c(-1, 0, 0x1.95c01ap+0, 0x1.a934fp+1), tolerance = 0)
  expect_equal(as.vector(as.array(jit(lax_log2)(x))), c(-1, 0, 0x1.95c01ap+0, 0x1.a934fp+1), tolerance = 0)
  expect_jax_equal(lax_log2(as_jax(c(1 + 2i, -0.5 + 0.25i), "complex64")),
                   c(1.160964012145996 + 1.5972779989242554i, -0.8390359282493591 + 3.8634581565856934i),
                   dtype = "complex64")
  expect_equal(as.vector(as.array(vmap(grad(lax_log2))(x))),
               c(0x1.715476p+1, 0x1.715476p+0, 0x1.ec709ep-2, 0x1.2776c6p-3), tolerance = 1e-7)
  lx_err(lax_log2(as_jax(1:2)), "jax_type_error", "log2 does not accept dtype int32")
  expect_jax_equal(lax_mulhi(as_jax(c(2147483647, -5, 123456789, -2147483648), "int32"),
                             as_jax(c(2147483647, 7, 987654321, 3), "int32")),
                   c(1073741823L, -1L, 28389652L, -2L), dtype = "int32")
  expect_jax_equal(lax_mulhi(as_jax(c(4294967295, 5, 123456789), "uint32"), as_jax(c(4294967295, 5, 123456789), "uint32")),
                   c(4294967294, 0, 3548706), dtype = "uint32")
  expect_jax_equal(jit(lax_mulhi)(as_jax(c(100L, -100L, 127L), "int8"), as_jax(c(100L, 3L, -128L), "int8")),
                   c(39L, -2L, -64L), dtype = "int8")
  expect_jax_equal(vmap(lax_mulhi, in_axes = list(0L, NULL))(as_jax(matrix(c(2147483647, -5), 2), "int32"),
                                                            as_jax(2147483647, "int32")),
                   matrix(c(1073741823, -3), 2))
  lx_err(lax_mulhi(c(0, 0), c(0, 0)), "jax_type_error", "mulhi does not accept dtype float32")
  y <- as_jax(c(0.999, -0.5, 1e-3, 2), "float32")
  expect_equal(as.vector(as.array(lax_one_minus_square(y))), c(0x1.060272p-9, 0.75, 0x1.ffffep-1, -3), tolerance = 0)
  expect_equal(as.vector(as.array(vmap(grad(lax_one_minus_square))(y))),
               c(-0x1.ff7ceep+0, 1, -0x1.0624dep-9, -4), tolerance = 0)
  expect_jax_equal(lax_one_minus_square(as_jax(c(2L, -3L))), c(-3L, -8L), dtype = "int32")
  enable_x64()
  on.exit(enable_x64(FALSE))
  check_grad_fd(function(v) jnp_sum(lax_log2(v) * v), as_jax(c(0.5, 1.5, 3), "float64"))
  check_grad_fd(function(v) jnp_sum(lax_one_minus_square(v)^2), as_jax(c(-0.7, 0.2, 1.3), "float64"))
  # The derivative of 1 - x^2 is exact for tiny x (no cancellation).
  expect_identical(as.numeric(as.array(grad(lax_one_minus_square)(1e-300))), -2e-300)
})

test_that("polynomial matches JAX, with derivatives in x and the coefficients", {
  skip_if_no_backend()
  x <- as_jax(c(0.5, 1, 3, 10), "float32")
  p <- lax_polynomial(3, c(1, 5, 2))
  expect_jax_equal(p, 34, dtype = "float32")
  expect_true(jaxR:::aval_of(p)$weak)
  expect_false(jaxR:::aval_of(lax_polynomial(x, c(1, 5)))$weak)
  expect_jax_equal(lax_polynomial(x, c(1, -2, 0.5, 0.25)), c(0.15625, -0.25, 6.25, 281))
  expect_jax_equal(lax_polynomial(x, list(as_jax(c(1, 2, 3, 4), "float32"), 2, as_jax(c(0.5, -1, 1, 2), "float32"))),
                   c(2.125, 3, 18, 224))
  # More than 8 coefficients lower to a scan.
  ten <- (0:9) / 10
  expect_equal(as.vector(as.array(lax_polynomial(x, ten))),
               c(0x1.953334p-3, 0x1.2p+2, 0x1.881f9cp+14, 0x1.d6f346p+29), tolerance = 0)
  expect_equal(as.vector(as.array(jit(function(v) lax_polynomial(v, ten))(x))),
               c(0x1.953334p-3, 0x1.2p+2, 0x1.881f9cp+14, 0x1.d6f346p+29), tolerance = 0)
  expect_equal(as.vector(as.array(lax_polynomial(x, ten, unroll = 20L))),
               c(0x1.953334p-3, 0x1.2p+2, 0x1.881f9cp+14, 0x1.d6f346p+29), tolerance = 1e-6)
  expect_jax_equal(lax_polynomial(x, list()), c(0, 0, 0, 0))
  expect_jax_equal(lax_polynomial(x, list(as_jax(2, "float32"))), c(2, 2, 2, 2), shape = 4L)
  expect_jax_equal(vmap(grad(function(t) lax_polynomial(t, c(1, -2, 0.5, 0.25))))(x), c(-1.3125, -0.25, 7.75, 83))
  expect_jax_equal(grad(function(c) jnp_sum(lax_polynomial(x, list(1, c, c * c))))(2), 455.5)
  expect_jax_equal(vmap(function(t, c) lax_polynomial(t, list(c, 1)), in_axes = list(0L, NULL))(x, 2),
                   c(2.5, 3, 5, 12))
  expect_true(grepl("polynomial", jaxpr_text(make_jaxpr(function(t) lax_polynomial(t, c(1, 2)))(x))))
  lx_err(lax_polynomial(as_jax(TRUE), list(TRUE, FALSE)), "jax_type_error", "polynomial does not accept dtype bool")
  enable_x64()
  on.exit(enable_x64(FALSE))
  check_grad_fd(function(v) jnp_sum(lax_polynomial(v, c(1, -2, 0.5, 0.25))), as_jax(c(0.3, -1.2, 2), "float64"))
  check_grad_fd(function(v) jnp_sum(lax_polynomial(v, ten)), as_jax(c(0.3, -1.2, 2), "float64"), tolerance = 1e-4)
  cvec <- as_jax(c(0.5, -1.5, 2), "float64")
  check_grad_fd(function(c) jnp_sum(lax_polynomial(cvec, list(c[0L], c[1L] * c[2L], c))),
                as_jax(c(1, 2, -0.5), "float64"))
  # Second order: d2/dx2 of 1 - 2x + 0.5x^2 + 0.25x^3 is 1 + 1.5x.
  expect_jax_equal(grad(grad(function(t) lax_polynomial(t, c(1, -2, 0.5, 0.25))))(2), 4)
})

test_that("tile, split, stack and unstack match JAX under jit, grad and vmap", {
  skip_if_no_backend()
  m <- as_jax(matrix(1:4, 2, byrow = TRUE), "int32")
  want <- rm_array(c(1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4), 4, 6)
  expect_jax_equal(lax_tile(m, c(2L, 3L)), want, dtype = "int32")
  expect_jax_equal(jit(function(v) lax_tile(v, c(2L, 3L)))(m), want)
  expect_jax_equal(lax_tile(c(1, 2, 3), 2L), c(1, 2, 3, 1, 2, 3))
  expect_jax_equal(lax_tile(5, integer(0)), 5)
  expect_jax_equal(vmap(function(v) lax_tile(v, 2L), in_axes = 1L)(as_jax(matrix(1:6, 2), "float32")),
                   rm_array(c(1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6), 3, 4))
  lx_err(lax_tile(m, 2L), "jax_type_error", "reps length must be equal to the ndim of x, got len(reps)=1 and x.ndim=2.")
  a <- lax_reshape(as_jax(0:9, "float32"), c(5L, 2L))
  s <- lax_split(a, c(2L, 3L))
  expect_length(s, 2L)
  expect_jax_equal(s[[1L]], rm_array(c(0, 1, 2, 3), 2, 2))
  expect_jax_equal(s[[2L]], rm_array(4:9, 3, 2))
  s1 <- jit(function(v) lax_split(v, c(1L, 0L, 1L), axis = 1L))(a)
  expect_jax_equal(s1[[1L]], matrix(c(0, 2, 4, 6, 8), 5))
  expect_identical(dim(s1[[2L]]), c(5L, 0L))
  expect_jax_equal(s1[[3L]], matrix(c(1, 3, 5, 7, 9), 5))
  expect_jax_equal(vmap(function(v) lax_split(v, c(1L, 2L))[[2L]], in_axes = 1L)(as_jax(matrix(1:6, 3), "float32")),
                   rm_array(c(2, 3, 5, 6), 2, 2))
  lx_err(lax_split(m, c(1L, 2L)), "jax_value_error", "Sum of sizes 3 must be equal to dimension 0 of the operand shape [2, 2]")
  lx_err(lax_split(m, c(-1L, 3L)), "jax_value_error", "Sizes passed to split must be nonnegative, got [-1, 3]")
  expect_jax_equal(lax_stack(list(as_jax(1:2), as_jax(3:4)), axis = 1L), rm_array(c(1, 3, 2, 4), 2, 2), dtype = "int32")
  expect_jax_equal(lax_stack(list(c(1, 2), c(3, 4)), axis = -1L), rm_array(c(1, 3, 2, 4), 2, 2))
  many <- jit(function(v) lax_stack(lapply(1:20, function(i) v * i), 1L))(as_jax(c(1, 2), "float32"))
  expect_jax_equal(many, rbind(1:20, 2 * (1:20)))
  expect_jax_equal(vmap(function(u, w) lax_stack(list(u, w), 0L), in_axes = list(0L, NULL))(
    as_jax(matrix(1:6, 3), "float32"), as_jax(c(9, 8), "float32")),
    rm_array(c(1, 4, 9, 8, 2, 5, 9, 8, 3, 6, 9, 8), 3, 2, 2))
  expect_jax_equal(vmap(function(u, w) lax_stack(list(u, w), 1L), in_axes = list(1L, 1L))(
    as_jax(matrix(1:6, 2), "float32"), as_jax(matrix(7:12, 2), "float32")),
    rm_array(c(1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12), 3, 2, 2))
  lx_err(lax_stack(list(c(0, 0), c(0, 0, 0))), "jax_value_error", "All input arrays must have the same shape. Got (2,), (3,).")
  lx_err(lax_stack(list(c(0, 0), matrix(0, 2, 1))), "jax_value_error",
         "Cannot stack arrays with different numbers of dimensions: got (2,), (2, 1).")
  lx_err(lax_stack(list(as_jax(c(0, 0)), as_jax(c(0L, 0L)))), "jax_type_error",
         "lax.stack requires arguments to have the same dtypes, got float32, int32.")
  lx_err(lax_stack(list()), "jax_value_error", "stack expects at least one operand, got 0.")
  u <- lax_unstack(as_jax(matrix(1:6, 2, byrow = TRUE)), axis = 1L)
  expect_length(u, 3L)
  expect_jax_equal(u[[1L]], c(1L, 4L))
  expect_jax_equal(u[[3L]], c(3L, 6L))
  expect_jax_equal(vmap(function(v) lax_unstack(v, 0L)[[2L]], in_axes = 1L)(as_jax(matrix(1:6, 3), "float32")), c(2, 5))
  expect_jax_equal(vmap(function(v) lax_unstack(v, 0L)[[1L]])(as_jax(array(1:8, c(2, 2, 2)), "float32")),
                   matrix(c(1, 2, 5, 6), 2))
  lx_err(lax_unstack(1), "jax_value_error", "unstack requires arrays with rank > 0")
  enable_x64()
  on.exit(enable_x64(FALSE))
  w <- as_jax(matrix(seq(0.5, 12, by = 0.5)^2, 4), "float64")
  check_grad_fd(function(v) jnp_sum(lax_tile(v, c(2L, 3L)) * w), as_jax(matrix(c(1, 2, 3, 4), 2), "float64"))
  check_grad_fd(function(v) { p <- lax_split(v, c(1L, 2L)); jnp_sum(p[[1L]] * 2) + jnp_sum(p[[2L]]^2) },
                as_jax(c(1, 2, 3), "float64"))
  check_grad_fd(function(v) jnp_sum(lax_stack(list(v, v^2), 1L) * as_jax(matrix(1:4, 2), "float64")),
                as_jax(c(1.5, -2), "float64"))
  check_grad_fd(function(v) { u <- lax_unstack(v, 1L); jnp_sum(u[[1L]] * 3 + u[[2L]]^2) },
                as_jax(matrix(c(1, 2, 3, 4, 5, 6), 3), "float64"))
  expect_jax_equal(grad(function(v) jnp_sum(lax_split(v, c(2L, 1L))[[2L]]))(c(1, 2, 3)), c(0, 0, 1))
})

test_that("approx_max_k and approx_min_k match JAX", {
  skip_if_no_backend()
  x <- as_jax(rm_array(c(3, 1, 4, 0.5, 5, 9, 2, 6, 2.5, 7, 1.5, 8, -2, 8.5, 1, 0), 2, 8), "float32")
  r <- lax_approx_max_k(x, 3L)
  expect_jax_equal(r[[1L]], rm_array(c(9, 6, 5, 8.5, 8, 7), 2, 3), dtype = "float32")
  expect_jax_equal(r[[2L]], rm_array(c(5, 7, 4, 5, 3, 1), 2, 3), dtype = "int32")
  r <- jit(function(v) lax_approx_max_k(v, 3L))(x)
  expect_jax_equal(r[[2L]], rm_array(c(5, 7, 4, 5, 3, 1), 2, 3))
  r <- lax_approx_min_k(x, 2L)
  expect_jax_equal(r[[1L]], rm_array(c(0.5, 1, -2, 0), 2, 2))
  expect_jax_equal(r[[2L]], rm_array(c(3, 1, 4, 7), 2, 2))
  r <- lax_approx_max_k(x, 1L)
  expect_jax_equal(r[[1L]], matrix(c(9, 8.5), 2))
  expect_jax_equal(r[[2L]], matrix(c(5L, 5L), 2))
  r <- lax_approx_min_k(x, 1L)
  expect_jax_equal(r[[2L]], matrix(c(3L, 4L), 2))
  r <- lax_approx_max_k(x, 1L, reduction_dimension = 0L)
  expect_jax_equal(r[[1L]], matrix(c(3, 7, 4, 8, 5, 9, 2, 6), 1))
  expect_jax_equal(r[[2L]], matrix(c(0, 1, 0, 1, 0, 0, 0, 0), 1))
  r <- lax_approx_max_k(x, 2L, reduction_dimension = -2L)
  expect_jax_equal(r[[2L]], rm_array(c(0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1), 2, 8))
  r <- lax_approx_max_k(x, 3L, aggregate_to_topk = FALSE)
  expect_jax_equal(r[[1L]], rm_array(c(9, 6, 5, 4, 3, 2, 1, 0.5, 8.5, 8, 7, 2.5, 1.5, 1, 0, -2), 2, 8))
  expect_jax_equal(r[[2L]], rm_array(c(5, 7, 4, 2, 0, 6, 1, 3, 5, 3, 1, 0, 2, 6, 7, 4), 2, 8))
  big <- lax_reshape(as_jax(0:299, "float32"), c(1L, 300L))
  expect_identical(dim(lax_approx_max_k(big, 5L, aggregate_to_topk = FALSE)[[1L]]), c(1L, 256L))
  expect_identical(dim(lax_approx_max_k(big, 5L, recall_target = 1, aggregate_to_topk = FALSE)[[1L]]), c(1L, 300L))
  expect_jax_equal(grad(function(v) jnp_sum(lax_approx_max_k(v, 2L)[[1L]] * as_jax(c(1, 2), "float32")))(x),
                   rm_array(c(0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0), 2, 8))
  r <- vmap(function(v) lax_approx_max_k(v, 2L), in_axes = 1L)(lax_transpose(x, c(1L, 0L)))
  expect_jax_equal(r[[1L]], rm_array(c(9, 6, 8.5, 8), 2, 2))
  expect_jax_equal(r[[2L]], rm_array(c(5, 7, 5, 3), 2, 2))
  lx_err(lax_approx_max_k(x, 0L), "jax_value_error", "k must be positive, got 0")
  lx_err(lax_approx_max_k(x, 9L), "jax_value_error", "k must be smaller than the size of reduction_dim 8, got 9")
  lx_err(lax_approx_max_k(as_jax(1:2), 1L), "jax_value_error", "operand must be a floating type")
  lx_err(lax_approx_max_k(1, 1L), "jax_type_error", "approx_top_k operand must have >= 1 dimension, got ()")
  enable_x64()
  on.exit(enable_x64(FALSE))
  check_grad_fd(function(v) jnp_sum(lax_approx_min_k(v, 2L)[[1L]]^2), as_jax(c(0.3, -1.2, 2.5, 0.9), "float64"))
})

test_that("the approx_top_k output size port matches XLA", {
  f <- jaxR:::lxe_approx_top_k_output_size
  # (input_size, rank, k, recall_target, override) -> size, from jax._src.lib._jax.
  expect_identical(f(8, 2L, 3L, 0.95, -1L), 8L)
  expect_identical(f(300, 2L, 5L, 0.95, -1L), 256L)
  expect_identical(f(100000, 1L, 10L, 0.95, -1L), 2048L)
  expect_identical(f(129, 2L, 2L, 0.5, 200000L), 256L)
  expect_identical(f(1025, 2L, 2L, 0.5, 200000L), 256L)
  expect_identical(f(10000, 1L, 2L, 0.5, 200000L), 1024L)
  expect_identical(f(65536, 2L, 50L, 0.99, 200000L), 2048L)
  expect_identical(f(5000, 2L, 1L, 0.95, -1L), 128L)
  expect_identical(f(5000, 2L, 3L, 1, -1L), 5000L)
})

test_that("ragged_dot and ragged_dot_general match JAX in all three modes", {
  skip_if_no_backend()
  lhs <- lax_reshape(as_jax(0:9, "float32"), c(5L, 2L))
  rhs <- lax_reshape(as_jax(0:11, "float32"), c(3L, 2L, 2L))
  gs <- as_jax(c(2L, 1L, 2L))
  want <- rm_array(c(2, 3, 6, 11, 46, 55, 118, 131, 154, 171), 5, 2)
  expect_jax_equal(lax_ragged_dot(lhs, rhs, gs), want, dtype = "float32")
  expect_jax_equal(jit(function(a) lax_ragged_dot(a, rhs, gs))(lhs), want)
  expect_jax_equal(grad(function(a) jnp_sum(lax_ragged_dot(a, rhs, gs)^2))(lhs),
                   rm_array(c(6, 26, 22, 90, 918, 1322, 4246, 5242, 5542, 6842), 5, 2))
  expect_jax_equal(grad(function(b) jnp_sum(lax_ragged_dot(lhs, b, gs)^2))(rhs),
                   rm_array(c(24, 44, 40, 72, 368, 440, 460, 550, 3880, 4308, 4424, 4912), 3, 2, 2))
  dn2 <- lax_RaggedDotDimensionNumbers(list(list(1L, 0L), list(integer(0), integer(0))), 1L, integer(0))
  l2 <- lax_reshape(as_jax(0:11, "float32"), c(3L, 4L))
  r2 <- lax_reshape(as_jax(0:7, "float32"), c(4L, 2L))
  expect_jax_equal(lax_ragged_dot_general(l2, r2, as_jax(c(1L, 3L)), dn2),
                   rm_array(c(0, 0, 0, 4, 0, 8, 28, 34, 76, 94, 124, 154), 2, 3, 2))
  dn3 <- lax_RaggedDotDimensionNumbers(list(list(2L, 1L), list(0L, 0L)), 0L, integer(0))
  expect_jax_equal(lax_ragged_dot_general(lax_reshape(as_jax(0:11, "float32"), c(2L, 3L, 2L)),
                                          lax_reshape(as_jax(0:7, "float32"), c(2L, 2L, 2L)), as_jax(c(1L, 1L)), dn3),
                   rm_array(c(2, 3, 6, 11, 10, 19, 66, 79, 86, 103, 106, 127), 2, 3, 2))
  out <- vmap(function(a, b, g) lax_ragged_dot(a, b, g))(lax_stack(list(lhs, lhs + 1)), lax_stack(list(rhs, rhs * 2)),
                                                         as_jax(matrix(c(2L, 1L, 2L, 0L, 4L, 1L), 2, byrow = TRUE)))
  expect_jax_equal(out, rm_array(c(2, 3, 6, 11, 46, 55, 118, 131, 154, 171,
                                   32, 38, 72, 86, 112, 134, 152, 182, 344, 382), 2, 5, 2))
  # Unbatched operands are broadcast (JAX requires every operand batched at 0).
  out <- vmap(function(a) lax_ragged_dot(a, rhs, gs), in_axes = 1L)(lax_stack(list(lhs, lhs + 1), 1L))
  expect_jax_equal(lax_index_in_dim(out, 0L, 0L, keepdims = FALSE), want)
  # Contracting mode under vmap gets a leading group dimension after the batch one.
  out <- vmap(function(a) lax_ragged_dot_general(a, r2, as_jax(c(1L, 3L)), dn2))(lax_stack(list(l2, l2)))
  expect_identical(dim(out), c(2L, 2L, 3L, 2L))
  expect_jax_equal(lax_index_in_dim(out, 1L, 0L, keepdims = FALSE), rm_array(c(0, 0, 0, 4, 0, 8, 28, 34, 76, 94, 124, 154), 2, 3, 2))
  lx_err(grad(function(a) jnp_sum(lax_ragged_dot_general(a, r2, as_jax(c(1L, 3L)), dn2)))(l2),
         "jax_not_implemented_error", "Unimplemented grad_x_dims for ragged dot general in mode RAGGED_CONTRACTING.")
  lx_err(lax_ragged_dot(lhs, rhs, c(2, 1, 2)), "jax_type_error",
         "ragged_dot_general requires that group_sizes.dtype is subtype of np.integer.")
  lx_err(lax_ragged_dot(lhs, rhs, c(2L, 3L)), "jax_type_error", "expected rhs group dimension size to be 2, got 3.")
  lx_err(lax_ragged_dot(lhs, rhs, gs, group_offset = 0L), "jax_not_implemented_error", "Unimplemented group_offset support.")
  enable_x64()
  on.exit(enable_x64(FALSE))
  rhs64 <- lax_reshape(as_jax(seq(-1, 1.2, length.out = 12), "float64"), c(3L, 2L, 2L))
  check_grad_fd(function(a) jnp_sum(lax_ragged_dot(a, rhs64, gs)^2),
                lax_reshape(as_jax(seq(0.1, 1, by = 0.1), "float64"), c(5L, 2L)))
  lhs64 <- lax_reshape(as_jax(seq(0.1, 1, by = 0.1), "float64"), c(5L, 2L))
  check_grad_fd(function(b) jnp_sum(sin(lax_ragged_dot(lhs64, b, gs))), rhs64)
})

test_that("conv_general_dilated_patches and conv_general_dilated_local match JAX", {
  skip_if_no_backend()
  img <- lax_reshape(as_jax(0:31, "float32"), c(1L, 2L, 4L, 4L))
  p <- lax_conv_general_dilated_patches(img, c(2L, 2L), c(1L, 1L), "VALID")
  expect_jax_equal(p, rm_array(c(0, 1, 2, 4, 5, 6, 8, 9, 10, 1, 2, 3, 5, 6, 7, 9, 10, 11, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                                 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26, 17, 18, 19, 21, 22,
                                 23, 25, 26, 27, 20, 21, 22, 24, 25, 26, 28, 29, 30, 21, 22, 23, 25, 26, 27, 29, 30, 31),
                               1, 8, 3, 3))
  p2 <- jit(function(v) lax_conv_general_dilated_patches(v, c(2L, 3L), c(2L, 1L), list(c(1L, 0L), c(0L, 1L)),
                                                         dimension_numbers = c("NHWC", "HWIO", "NHWC")))(
    lax_transpose(img, c(0L, 2L, 3L, 1L)))
  expect_jax_equal(p2, rm_array(c(0, 0, 0, 0, 1, 2, 0, 0, 0, 16, 17, 18, 0, 0, 0, 1, 2, 3, 0, 0, 0, 17, 18, 19, 0, 0, 0,
                                  2, 3, 0, 0, 0, 0, 18, 19, 0, 4, 5, 6, 8, 9, 10, 20, 21, 22, 24, 25, 26, 5, 6, 7, 9,
                                  10, 11, 21, 22, 23, 25, 26, 27, 6, 7, 0, 10, 11, 0, 22, 23, 0, 26, 27, 0), 1, 2, 3, 12))
  k <- lax_reshape(as_jax(((0:215) %% 7 - 3) / 4, "float32"), c(3L, 8L, 3L, 3L))
  loc <- lax_conv_general_dilated_local(img, k, c(1L, 1L), "VALID", c(2L, 2L))
  expect_jax_equal(loc, rm_array(c(-9.25, 2.5, -11.5, 3, -4.5, 16.5, 12.5, -16, -2, -11, 3, -5.25, 15.5, 10.25, -13.75,
                                   -1, -13.25, 3, -5.75, 14, 8, -12.25, 0.5, -12.5, 3, -3.5, 18.5), 1, 3, 3, 3))
  k2 <- lax_reshape(as_jax(((0:95) %% 5 - 2) / 2, "float32"), c(2L, 2L, 8L, 3L))
  loc2 <- lax_conv_general_dilated_local(lax_transpose(img, c(0L, 2L, 3L, 1L)), k2, c(2L, 2L), "VALID", c(2L, 2L),
                                         dimension_numbers = c("NHWC", "HWIO", "NHWC"))
  expect_jax_equal(loc2, rm_array(c(-14, 15.5, 5, -13.5, -16, 16.5, 15, -13.5, -22, 0, 17, -13.5), 1, 2, 2, 3))
  enable_x64()
  on.exit(enable_x64(FALSE))
  k64 <- lax_reshape(as_jax(((0:215) %% 7 - 3) / 4, "float64"), c(3L, 8L, 3L, 3L))
  check_grad_fd(function(v) jnp_sum(lax_conv_general_dilated_local(v, k64, c(1L, 1L), "VALID", c(2L, 2L))^2),
                lax_reshape(as_jax(sin(1:32), "float64"), c(1L, 2L, 4L, 4L)), tolerance = 1e-4)
})

test_that("composite evaluates its decomposition and has no derivative or batching rules", {
  skip_if_no_backend()
  f <- lax_composite(function(x, k) lax_sin(x) * k, name = "my.op")
  x <- as_jax(c(0, 1, 2), "float32")
  want <- c(0, 0x1.aed548p+0, 0x1.d18f6ep+0)
  expect_jax_equal(f(x, k = 2), want, dtype = "float32")
  expect_jax_equal(jit(function(v) f(v, k = 2))(x), want)
  expect_true(grepl("composite[name=my.op", jaxpr_text(make_jaxpr(function(v) f(v, k = 2))(x)), fixed = TRUE))
  lx_err(vmap(function(v) f(v, k = 2))(x), "jax_error", "Batching rule for 'composite' not implemented")
  lx_err(grad(function(v) jnp_sum(f(v, k = 2)))(x), "jax_value_error", "JVP rule for composite not implemented")
  lx_err(linear_transpose(function(v) f(v, k = 2), x)(x), "jax_value_error", "Transpose rule for composite not implemented")
  lx_err(jit(function(v, s) f(v, k = s))(x, 2), "jax_escaped_tracer_error", "Found a JAX Tracer as an attribute")
})

test_that("scaled_dot matches JAX", {
  skip_if_no_backend()
  lhs <- lax_reshape(as_jax(0:7, "float32"), c(2L, 4L))
  rhs <- lax_reshape(as_jax(0:11, "float32"), c(4L, 3L))
  ls <- as_jax(rm_array(c(1, 2, 0.5, 4), 2, 2), "float32")
  rs <- as_jax(rm_array(c(1, 2, 3, 0.25, 1, 2), 2, 3), "float32")
  expect_jax_equal(lax_scaled_dot(lhs, rhs, lhs_scale = ls, rhs_scale = rs),
                   rm_array(c(22.5, 96, 211, 106.5, 472, 1048), 2, 3), dtype = "bfloat16")
  expect_jax_equal(lax_scaled_dot(lhs, rhs, lhs_scale = ls, rhs_scale = rs, preferred_element_type = "float32"),
                   rm_array(c(22.5, 96, 211, 106.5, 472, 1049.5), 2, 3), dtype = "float32")
  expect_jax_equal(jit(function(a) lax_scaled_dot(a, rhs, preferred_element_type = "float32"))(lhs),
                   rm_array(c(42, 48, 54, 114, 136, 158), 2, 3))
  out <- vmap(function(a) lax_scaled_dot(a, rhs, lhs_scale = ls, rhs_scale = rs, preferred_element_type = "float32"))(
    lax_stack(list(lhs, lhs * 2)))
  expect_jax_equal(out, rm_array(c(22.5, 96, 211, 106.5, 472, 1049.5, 45, 192, 422, 213, 944, 2099), 2, 2, 3))
  lx_err(lax_scaled_dot(lhs, rhs, lhs_scale = matrix(1, 2, 4), rhs_scale = rs), "jax_type_error",
         "The ratio of LHS contracting dim 1 to its scale's dim size (1) must be at least 2.")
  lx_err(lax_scaled_dot(lhs, rhs, lhs_scale = matrix(1, 2, 3), rhs_scale = rs), "jax_type_error",
         "LHS contracting dim 1 of size 4 must be divisible by its scale's dim size 3.")
  lx_err(lax_scaled_dot(lhs, rhs, lhs_scale = matrix(1, 3, 2), rhs_scale = rs), "jax_type_error",
         "LHS dim 0 of size 2 does not match scale dim size 3.")
  lx_err(lax_scaled_dot(c(1, 2), c(1, 2)), "jax_type_error", "scaled_dot does not support 0-rank and 1-rank operands.")
  lx_err(grad(function(a) jnp_sum(lax_scaled_dot(a, rhs, preferred_element_type = "float32")))(lhs), "jax_error",
         "Differentiation rule for 'scaled_dot' not implemented")
})

test_that("platform_dependent selects the cpu branch and pcast is the identity under vmap", {
  skip_if_no_backend()
  expect_jax_equal(lax_platform_dependent(c(0, 1, 2), cpu = function(v) v + 1, default = function(v) v - 1), c(1, 2, 3))
  expect_jax_equal(lax_platform_dependent(c(0, 1, 2), default = function(v) v * 10), c(0, 10, 20))
  expect_jax_equal(lax_platform_dependent(c(0, 1, 2), 2, cpu = function(a, b) a * b), c(0, 2, 4))
  expect_jax_equal(jit(function(v) lax_platform_dependent(v, cpu = sin, default = cos))(1), sin(1))
  expect_jax_equal(grad(function(v) lax_platform_dependent(v, cpu = sin, default = cos))(1), cos(1))
  expect_jax_equal(vmap(function(v) lax_platform_dependent(v, cpu = sin, default = cos))(c(0, 1, 2)), sin(c(0, 1, 2)))
  lx_err(lax_platform_dependent(1, tpu = function(v) v), "jax_not_implemented_error",
         "MLIR translation rule for primitive 'platform_index(platforms=(('tpu',),))' not found for platform cpu")
  lx_err(lax_platform_dependent(1, gpu = function(v) v), "jax_value_error", "Use 'cuda', 'rocm', or 'oneapi'")
  lx_err(lax_platform_dependent(1, cpu = 3), "jax_type_error", "the 'cpu' branch must be a callable.")
  lx_err(lax_platform_dependent(1, cpu = sin, default = 3), "jax_type_error", "the 'default' branch must be a callable.")
  expect_jax_equal(vmap(function(v) lax_pcast(v, "i", to = "varying"), axis_name = "i")(c(1, 2, 3)), c(1, 2, 3))
  expect_jax_equal(vmap(grad(function(v) lax_pcast(v, "i", to = "varying")^2), axis_name = "i")(c(1, 2, 3)), c(2, 4, 6))
  r <- vmap(function(v) lax_pcast(list(a = v, b = 2 * v), "i", to = "varying"), axis_name = "i")(c(1, 2))
  expect_jax_equal(r$b, c(2, 4))
  expect_identical(lax_pcast(1, character(0), to = "bogus"), 1)
  lx_err(lax_pcast(1, "i", to = "bogus"), "jax_value_error", "Got unexpected `to` value.")
  lx_err(lax_pcast(1, "i", to = "unreduced"), "jax_value_error", "Unsupported pcast from=invarying, to='unreduced'")
  lx_err(vmap(function(v) lax_pcast(v, "j", to = "varying"), axis_name = "i")(c(1, 2)), "jax_name_error", "unbound axis name: j")
})

test_that("tokens and dce_sink", {
  skip_if_no_backend()
  t <- lax_create_token()
  expect_s3_class(t, "jax_token")
  expect_s3_class(lax_after_all(t, lax_create_token(1)), "jax_token")
  expect_s3_class(lax_after_all(), "jax_token")
  expect_output(print(t), "Token()", fixed = TRUE)
  lx_err(lax_after_all(t, 1), "jax_type_error", "Arguments to after_all must be tokens")
  out <- jit(function(x) list(lax_after_all(lax_create_token(), lax_create_token()), x * 2))(3)
  expect_s3_class(out[[1L]], "jax_token")
  expect_jax_equal(out[[2L]], 6)
  expect_null(lax_dce_sink(list(1, c(2, 3))))
  expect_jax_equal(jit(function(x) { lax_dce_sink(x * 3); x + 1 })(1), 2)
  lx_err(lax_dce_sink("a"), "jax_type_error")
})

test_that("custom_root differentiates implicitly with respect to closed over values", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  sqrt_cr <- function(a) {
    f <- function(x) x^2 - a
    solve <- function(f, x0) lax_fori_loop(0L, 30L, function(i, x) x - f(x) / (2 * x), x0)
    lax_custom_root(f, 1, solve, function(g, y) y / g(1))
  }
  expect_equal(as.numeric(as.array(sqrt_cr(2))), 1.4142135623730951, tolerance = 1e-15)
  expect_equal(as.numeric(as.array(grad(sqrt_cr)(2))), 0.35355339059327373, tolerance = 1e-15)
  expect_equal(as.numeric(as.array(jit(grad(sqrt_cr))(2))), 0.35355339059327373, tolerance = 1e-15)
  expect_equal(as.numeric(as.array(grad(grad(sqrt_cr))(2))), -0.08838834764831842, tolerance = 1e-12)
  expect_jax_equal(vmap(sqrt_cr)(c(1, 4, 9)), c(1, 2, 3))
  expect_jax_equal(vmap(grad(sqrt_cr))(c(1, 4, 9)), c(0.5, 0.25, 1 / 6))
  expect_jax_equal(jvp(sqrt_cr, list(4), list(1))[[2L]], 0.25)
  check_grad_fd(function(a) sqrt_cr(a) * a, as_jax(3, "float64"))
  vec_cr <- function(a) {
    M <- as_jax(matrix(c(3, 1, 1, 2), 2, byrow = TRUE))
    f <- function(x) jnp_matmul(M, x) - a
    solve <- function(f, x0) lax_fori_loop(0L, 60L, function(i, x) x - 0.3 * f(x), x0)
    lax_custom_root(f, c(0, 0), solve, function(g, y) jnp_linalg_solve(jacobian(g)(y), y))
  }
  expect_jax_equal(vec_cr(as_jax(c(1, 2))), c(4.997197e-15, 1), tolerance = 1e-6)
  expect_jax_equal(jacobian(vec_cr)(as_jax(c(1, 2))), matrix(c(0.4, -0.2, -0.2, 0.6), 2), tolerance = 1e-12)
  aux_cr <- function(a) {
    f <- function(x) x^3 - a
    solve <- function(f, x0) {
      x <- lax_fori_loop(0L, 40L, function(i, x) x - f(x) / (3 * x * x), x0)
      list(x, f(x))
    }
    lax_custom_root(f, 1, solve, function(g, y) y / g(1), has_aux = TRUE)
  }
  r <- aux_cr(8)
  expect_jax_equal(r[[1L]], 2)
  expect_jax_equal(r[[2L]], 0)
  expect_equal(as.numeric(as.array(grad(function(a) aux_cr(a)[[1L]])(8))), 0.08333333333333333, tolerance = 1e-14)
  lx_err(lax_custom_root(function(x) list(x, x), 1, function(f, x) x, function(g, y) y), "jax_type_error",
         "f() output pytree structure must match initial_guess")
  lx_err(lax_custom_root(function(x) x, 1, function(f, x) list(x, x), function(g, y) y), "jax_type_error",
         "solve() output pytree structure must match initial_guess")
  lx_err(lax_custom_root(function(x) x, 1, function(f, x) x, function(g, y) y, has_aux = TRUE), "jax_value_error",
         "but a pytree tuple with auxiliary output was expected because has_aux was set to True.")
})

test_that("linear_call uses the custom transpose", {
  skip_if_no_backend()
  f <- function(r, x) x / r
  t <- function(r, ct) ct / r
  div_add <- function(x, denom) x + linear_call(f, t, denom, x)
  tr <- function(fn, ex) function(y) linear_transpose(fn, ex)(y)[[1L]]
  expect_jax_equal(div_add(9, 3), 12)
  expect_jax_equal(jit(function(x) div_add(x, 3))(9), 12)
  expect_jax_equal(tr(function(x) div_add(x, 3), 1)(18), 24)
  custom_id <- function(x) linear_call(function(r, x) x, function(r, t) 7, list(), x)
  expect_jax_equal(custom_id(1), 1)
  expect_jax_equal(tr(custom_id, 1)(1), 7)
  expect_jax_equal(tr(tr(custom_id, 1), 1)(1), 1)
  expect_jax_equal(tr(tr(tr(custom_id, 1), 1), 1)(1), 7)
  expect_jax_equal(grad(function(x) div_add(x, 3)^2)(9), 32)
  expect_jax_equal(jvp(function(x) div_add(x, 3), list(9), list(1))[[2L]], 4 / 3)
  lx_err(vmap(function(x) div_add(x, 3))(c(1, 1, 1)), "jax_error", "Batching rule for 'linear_call' not implemented")
  lx_err(grad(function(d) div_add(9, d))(3), "jax_assertion_error")
  lx_err(linear_transpose(function(x) linear_call(f, function(r, ct) list(ct, ct), 3, x), 1)(1), "jax_type_error",
         "transpose output pytree structure must match that of linear inputs")
})

test_that("SymbolicZero, zero_from_primal, CustomVJPPrimal and custom_vjp_primal_tree_values", {
  skip_if_no_backend()
  z <- SymbolicZero(ShapeDtypeStruct(c(2L, 3L), "float32"))
  expect_s3_class(z, "jax_SymbolicZero")
  expect_identical(dim(z), c(2L, 3L))
  expect_output(print(z), "SymbolicZero(f32[2,3])", fixed = TRUE)
  zs <- zero_from_primal(list(a = c(1, 2), b = 3L))
  expect_jax_equal(zs$a, c(0, 0), dtype = "float32")
  expect_identical(dtype(zs$b), "float0")
  zs <- zero_from_primal(list(a = c(1, 2)), symbolic_zeros = TRUE)
  expect_s3_class(zs$a, "jax_SymbolicZero")
  expect_identical(zs$a$aval$dtype, "float32")
  tree <- list(CustomVJPPrimal(1, TRUE), list(x = CustomVJPPrimal(2, FALSE)))
  expect_identical(custom_vjp_primal_tree_values(tree), list(1, list(x = 2)))
  expect_true(tree[[1L]]$perturbed)
  lx_err(custom_vjp_primal_tree_values(list(1)), "jax_type_error", "unexpected leaf type numeric")
})

test_that("remat policy values, checkpoint_name_fwd and saved_residuals", {
  skip_if_no_backend()
  expect_true(isTRUE(Saveable))
  expect_false(isTRUE(Recompute))
  off <- Offloadable("device", "pinned_host")
  expect_true(isTRUE(off))
  expect_identical(off$dst, "pinned_host")
  expect_output(print(off), "Offloadable(src='device', dst='pinned_host')", fixed = TRUE)
  expect_output(print(Saveable), "Saveable")
  pol <- function(prim, params) if (prim == "sin") Saveable else Recompute
  g <- grad(checkpoint(function(x) sin(sin(x)), policy = pol))(1)
  expect_jax_equal(g, 0.36003948908962097)
  g <- grad(checkpoint(function(x) sin(sin(x)), policy = function(p, params) if (p == "sin") off else Recompute))(1)
  expect_jax_equal(g, 0.36003948908962097)
  expect_jax_equal(checkpoint_name_fwd(c(1, 2), "a"), c(1, 2))
  g <- grad(checkpoint(function(x) jnp_sum(sin(checkpoint_name_fwd(cos(x), "c"))),
                       policy = checkpoint_policies$save_only_these_names("c")))(c(1, 2))
  expect_jax_equal(g, -sin(c(1, 2)) * cos(cos(c(1, 2))))
  enable_x64()
  on.exit(enable_x64(FALSE))
  srcs <- function(r) vapply(r, `[[`, "", "src")
  r <- saved_residuals(function(x) sin(x) * cos(x), 1)
  expect_identical(srcs(r), c("output of sin from unknown", "output of cos from unknown",
                              "output of cos from unknown", "output of sin from unknown"))
  expect_identical(jaxR:::aval_str(r[[1L]]$aval), "f64[]")
  r <- saved_residuals(function(x, y) jnp_sum(exp(x) * y), c(1, 1, 1), c(1, 1, 1))
  expect_identical(srcs(r), c("from the argument y", "output of exp from unknown"))
  expect_identical(saved_residuals(function(x) x * 2, 1), list())
  r <- saved_residuals(function(d) tanh(d$a) * d$b, list(a = c(1, 1), b = c(1, 1)))
  expect_identical(srcs(r)[1L], "from the argument d$b")
  expect_identical(srcs(saved_residuals(function(...) ..1 * ..2, 1, 2)),
                   c("from the argument ..1", "from the argument ..2"))
  expect_output(print_saved_residuals(function(x, y) jnp_sum(exp(x) * y), c(1, 1, 1), c(1, 1, 1)),
                "f64[3] from the argument y", fixed = TRUE)
})
