# Primitive objects, narrow dtypes, NumPy I/O and printing, refs, config contexts and
# other late additions. Expected values are from jax 0.12.0.1 / NumPy 2.

test_that("primitive objects are exported and bindable", {
  skip_if_no_backend()
  expect_s3_class(lax_add_p, "jax_primitive")
  expect_identical(lax_add_p$name, "add")
  expect_identical(lax_linalg_qr_p$name, "qr")
  expect_identical(custom_jvp_call_p$name, "custom_jvp_call")
  expect_identical(name_p$name, "name")
  expect_identical(experimental_sparse_bcoo_todense_p$name, "bcoo_todense")
  expect_jax_equal(lax_add_p$bind(1, 2), 3)
  expect_jax_equal(lax_reshape_p$bind(as_jax(1:6), new_sizes = c(2L, 3L)), matrix(1:6, 2, byrow = TRUE))
  jx <- make_jaxpr(function(x) lax_squeeze(lax_expand_dims(x, 0L), 0L))(c(1, 2))
  expect_identical(vapply(jx$jaxpr$eqns, function(e) e$prim$name, ""), c("broadcast_in_dim", "squeeze"))
  expect_jax_equal(grad(function(x) jnp_sum(lax_squeeze(jnp_reshape(x, c(1L, 3L)), 0L)^2))(c(1, 2, 3)), c(2, 4, 6))
  expect_error(lax_squeeze(c(1, 2), 0L), class = "jax_value_error")
  expect_jax_equal(lax_stage(3), 3)
  expect_jax_equal(lax_broadcast_like(1, c(1, 2, 3)), c(1, 1, 1))
  expect_jax_equal(vmap(function(x) lax_psum_scatter_start(x, "i")$done(), axis_name = "i")(
    matrix(c(1, 2, 3, 4), 2)), c(3, 7))
  expect_error(vmap(function(x) lax_psum_scatter(x, "i"), axis_name = "i")(c(1, 2)), "out of bounds")
  expect_error(lax_ragged_all_to_all(c(1, 2), c(0, 0), 0L, 1L, 0L, 1L, "i"), class = "jax_runtime_error")
})

test_that("gcd and lcm match JAX", {
  skip_if_no_backend()
  expect_jax_equal(jnp_gcd(c(12L, 18L, 0L, -7L, 0L), c(8L, 27L, 5L, 21L, 0L)), c(4, 9, 5, 7, 0), dtype = "int32")
  expect_jax_equal(jnp_lcm(c(4L, 6L, 0L, -3L), c(6L, 4L, 3L, 5L)), c(12, 12, 0, 15), dtype = "int32")
  expect_error(jnp_gcd(1.5, 2), class = "jax_value_error")
})

test_that("narrow dtypes convert and promote as in JAX", {
  skip_if_no_backend()
  x <- c(0.3, 1.7, -2.4, 20, 1e-3)
  want <- list(float8_e4m3fn = c(0.3125, 1.75, -2.5, 20, 0.001953125),
               float8_e5m2 = c(0.3125, 1.75, -2.5, 20, 0.0009765625),
               float8_e4m3fnuz = c(0.3125, 1.75, -2.5, 20, 0.0009765625),
               float8_e5m2fnuz = c(0.3125, 1.75, -2.5, 20, 0.0009765625),
               float8_e4m3b11fnuz = c(0.3125, 1.75, -2.5, 20, 0.0009765625),
               float8_e3m4 = c(0.296875, 1.6875, -2.375, Inf, 0),
               float8_e4m3 = c(0.3125, 1.75, -2.5, 20, 0.001953125),
               float8_e8m0fnu = c(0.25, 2, NaN, 16, 0.0009765625),
               float4_e2m1fn = c(0.5, 1.5, -2, 6, 0))
  for (dt in names(want)) {
    a <- as_jax(x, dt)
    expect_identical(dtype(a), dt)
    expect_identical(as.vector(as.array(a)), want[[dt]], info = dt)
  }
  expect_jax_equal(as_jax(c(-3L, 7L, 9L, 1L), "int4"), c(-3, 7, -7, 1), dtype = "int4")
  expect_jax_equal(as_jax(c(-3L, 7L, 9L, 1L), "uint4"), c(13, 7, 9, 1), dtype = "uint4")
  expect_jax_equal(as_jax(c(-3L, 7L, 9L, 1L), "int2"), c(1, -1, 1, 1), dtype = "int2")
  expect_jax_equal(as_jax(c(-3L, 7L, 9L, 1L), "uint2"), c(1, 3, 1, 1), dtype = "uint2")
  expect_identical(unlist(iinfo("int4")[c("min", "max")]), c(min = -8, max = 7))
  expect_identical(unlist(iinfo("uint2")[c("min", "max")]), c(min = 0, max = 3))
  f <- finfo("float8_e4m3fn")
  expect_identical(c(f$eps, f$max, f$tiny, f$resolution, f$nmant, f$nexp, f$bits), c(0.125, 448, 0.015625, 0.1015625, 3, 4, 8))
  f <- finfo("float8_e8m0fnu")
  expect_identical(c(f$eps, f$max, f$tiny), c(1, 2^127, 2^-127))
  expect_identical(finfo("float4_e2m1fn")$max, 6)
  expect_identical(result_type("int4", 1L), "int4")
  expect_identical(result_type("float8_e4m3fn", 1), "float8_e4m3fn")
  expect_identical(result_type("int4", TRUE), "int4")
  expect_error(result_type("uint4", "int8"), "1-bit, 2-bit and 4-bit integers", class = "jax_type_promotion_error")
  expect_error(result_type("float8_e5m2", "float32"), "8-bit floats", class = "jax_type_promotion_error")
  expect_jax_equal(jit(function(a) a * 2)(as_jax(c(1.5, 3), "float8_e4m3fn")), c(3, 6), dtype = "float8_e4m3fn")
  expect_jax_equal(jnp_float8_e5m2(c(1.1, 3)), c(1, 3), dtype = "float8_e5m2")
  expect_output(print(as_jax(1, "float8_e4m3fn")), "f8_e4m3fn")
})

test_that("save, savez and load round trip in NumPy formats", {
  skip_if_no_backend()
  f <- tempfile(fileext = ".npy")
  for (x in list(as_jax(matrix(c(1.5, 2, 3, 4), 2)), as_jax(c(TRUE, FALSE, TRUE)), as_jax(c(1+2i, 3-1i)),
                 as_jax(c(1.5, -2), "bfloat16"), as_jax(c(-1L, 300L, 7L), "int16"), as_jax(c(3L, 250L), "uint8"),
                 as_jax(2.5), as_jax(array(1:24, c(2L, 3L, 4L)), "int32"))) {
    jnp_save(f, x)
    y <- jnp_load(f)
    expect_identical(dtype(y), dtype(x))
    expect_identical(as.array(y), as.array(x))
  }
  # Header written as numpy does: magic, version 1.0, 64 byte aligned header.
  jnp_save(f, as_jax(c(1, 2, 3)))
  raw <- readBin(f, "raw", 200L)
  expect_identical(raw[1:8], as.raw(c(0x93, charToRaw("NUMPY"), 1L, 0L)))
  hlen <- as.integer(raw[9]) + 256L * as.integer(raw[10])
  expect_identical((10L + hlen) %% 64L, 0L)
  expect_match(rawToChar(raw[11:(10 + hlen)]), "\\{'descr': '<f4', 'fortran_order': False, 'shape': \\(3,\\), \\}")
  z <- tempfile(fileext = ".npz")
  jnp_savez(z, a = 1:3, c(2.5, 1))
  out <- jnp_load(z)
  expect_setequal(names(out), c("a", "arr_0"))
  expect_jax_equal(out$a, 1:3, dtype = "int32")
  expect_jax_equal(out$arr_0, c(2.5, 1), dtype = "float32")
})

test_that("array_repr and array_str follow NumPy", {
  skip_if_no_backend()
  expect_identical(jnp_array_repr(c(1.5, 2.25, 3)), "array([1.5 , 2.25, 3.  ], dtype=float32)")
  expect_identical(jnp_array_repr(c(1e-5, 1)), "array([1.e-05, 1.e+00], dtype=float32)")
  expect_identical(jnp_array_repr(c(1.5e-5, 1)), "array([1.5e-05, 1.0e+00], dtype=float32)")
  expect_identical(jnp_array_repr(as_jax(c(0.1, 0.25, 3), "float16")), "array([0.1 , 0.25, 3.  ], dtype=float16)")
  expect_identical(jnp_array_repr(c(100000.5, 2)), "array([1.000005e+05, 2.000000e+00], dtype=float32)")
  expect_identical(jnp_array_repr(c(-1.5, NaN, Inf)), "array([-1.5,  nan,  inf], dtype=float32)")
  expect_identical(jnp_array_repr(c(1+2i, 3.5-1i)), "array([1. +2.j, 3.5-1.j], dtype=complex64)")
  expect_identical(jnp_array_repr(as_jax(3.5, "float32")), "array(3.5, dtype=float32)")
  expect_identical(jnp_array_repr(jnp_zeros(0L)), "array([], dtype=float32)")
  expect_identical(jnp_array_repr(c(TRUE, FALSE)), "array([ True, False])")
  expect_identical(jnp_array_repr(matrix(1:4, 2)), "array([[1, 3],\n       [2, 4]], dtype=int32)")
  expect_identical(jnp_array_repr(jnp_arange(30, dtype = "float32")), paste0(
    "array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,\n",
    "       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,\n",
    "       26., 27., 28., 29.], dtype=float32)"))
  expect_identical(jnp_array_repr(jnp_reshape(jnp_arange(8L), c(2L, 2L, 2L))),
                   "array([[[0, 1],\n        [2, 3]],\n\n       [[4, 5],\n        [6, 7]]], dtype=int32)")
  expect_identical(jnp_array_str(matrix(c(1.5, 3, 2, 4), 2)), "[[1.5 2. ]\n [3.  4. ]]")
})

test_that("s_ and index_exp build indices", {
  skip_if_no_backend()
  m <- jnp_reshape(jnp_arange(12L), c(3L, 4L))
  expect_jax_equal(m[jnp_s_[1L, sl(0L, 2L)]], c(4, 5))
  expect_s3_class(jnp_s_[sl(1L, 3L)], "jax_slice")
  idx <- jnp_index_exp[, 1L]
  expect_s3_class(idx, "jax_index_tuple")
  expect_jax_equal(m[idx], c(1, 5, 9))
  expect_identical(jnp_ComplexWarning, "jax_complex_warning")
})

test_that("refs hold mutable state through transformations", {
  skip_if_no_backend()
  r <- new_ref(c(1, 2, 3))
  expect_s3_class(r, "jax_ref")
  expect_identical(r$shape, 3L)
  r[1L] <- 10
  expect_jax_equal(r[], c(1, 10, 3))
  r$addupdate(1)
  expect_jax_equal(r[], c(2, 11, 4))
  expect_jax_equal(r$swap(c(0, 0, 0)), c(2, 11, 4))
  r[] <- c(2, 11, 4)
  f <- jit(function(x) { r[] <- r[] * x; jnp_sum(r[]) })
  expect_jax_equal(f(2), 34)
  expect_jax_equal(r[], c(4, 22, 8))
  expect_jax_equal(f(2), 68)
  g <- jit(function(q, y) { q[0L] <- y; q[] + 1 })
  q <- new_ref(c(0, 0))
  expect_jax_equal(g(q, 5), c(6, 1))
  expect_jax_equal(q[], c(5, 0))
  h <- function(x) { loc <- new_ref(x); for (i in 1:3) loc[] <- loc[] * 2; jnp_sum(loc[]) }
  expect_jax_equal(grad(h)(c(1, 2)), c(8, 8))
  expect_jax_equal(jit(h)(c(1, 2)), 24)
  expect_jax_equal(vmap(h)(matrix(c(1, 2, 3, 4), 2)), c(32, 48))
  e <- empty_ref(ShapeDtypeStruct(2L, "float32"))
  expect_jax_equal(e[], c(0, 0))
  expect_jax_equal(freeze(r), c(8, 44, 16))
  expect_error({ r[] <- 0 }, "frozen", class = "jax_value_error")
  free_ref(q)
  expect_error(q[], class = "jax_value_error")
  k <- new_ref(jnp_arange(6L))
  expect_jax_equal(k[ds(1L, 3L)], c(1, 2, 3))
  expect_jax_equal(jit(function(i) k[ds(i, 2L)])(2L), c(2, 3))
  k[ds(0L, 2L)] <- c(9L, 9L)
  expect_jax_equal(k[], c(9, 9, 2, 3, 4, 5))
  expect_error(new_ref(c(1, 2))$set(as_jax(1L, "int32")), "Invalid dtype", class = "jax_value_error")
})

test_that("live_arrays tracks reachable arrays", {
  skip_if_no_backend()
  # No count comparison: a garbage collection between two calls shrinks the list.
  keep <- as_jax(c(1, 2)) + 1
  expect_true(any(vapply(live_arrays(), identical, NA, keep)))
})

test_that("configuration objects and contexts follow JAX", {
  skip_if_no_backend()
  expect_false(config$jax_enable_x64)
  old <- config$update("jax_debug_nans", FALSE)
  expect_false(config$read("jax_debug_nans"))
  expect_true(is.list(config$values))
  expect_true(enable_checks(TRUE, jax_config("jax_enable_checks")))
  expect_false(jax_config("jax_enable_checks"))
  expect_error(legacy_prng_key("error", random_uniform(random_PRNGKey(0L))), "Legacy uint32 key", class = "jax_value_error")
  expect_warning(legacy_prng_key("warn", random_uniform(random_PRNGKey(0L))), "Legacy uint32 key")
  expect_error(legacy_prng_key("bad", 1), class = "jax_value_error")
  expect_identical(thread_guard(TRUE, 1 + 1), 2)
  ctx <- make_user_context(1)
  expect_identical(ctx(), 1)
  traces <- 0L
  f <- jit(function(x) { traces <<- traces + 1L; x * 2 })
  ctx(2, f(1)); ctx(2, f(1)); ctx(3, f(1))
  expect_identical(traces, 2L)
  expect_identical(ctx(), 1)
})

test_that("custom_vjp optimize_remat keeps values and gradients", {
  skip_if_no_backend()
  f <- custom_vjp(function(x) sin(x))
  f <- custom_vjp_defvjp(f, function(x) list(sin(x), cos(x)), function(c, g) list(c * g), optimize_remat = TRUE)
  expect_jax_equal(f(1), sin(1))
  expect_jax_equal(grad(f)(1), cos(1))
  expect_jax_equal(jit(grad(f))(1), cos(1))
  expect_jax_equal(vmap(grad(f))(c(0, 1)), cos(c(0, 1)))
  jx <- make_jaxpr(function(x) grad(f)(x))(1)
  expect_true("remat_opt" %in% vapply(jx$jaxpr$eqns, function(e) e$prim$name, ""))
  expect_error(custom_vjp_defvjp(f, function(x) list(x, x), function(c, g) list(g), symbolic_zeros = TRUE,
                                 optimize_remat = TRUE), class = "jax_not_implemented_error")
})
