# as_jax conversions, host round trips, printing and shape broadcasting.

test_that("as_jax maps R types to JAX dtypes and weak types", {
  skip_if_no_backend()
  for (x64 in c(FALSE, TRUE)) with_jax_config("jax_enable_x64", x64, {
    f <- if (x64) "float64" else "float32"
    cplx <- if (x64) "complex128" else "complex64"
    cases <- list(
      list(c(1.5, 2), f, FALSE, 2L), list(2.5, f, TRUE, integer(0)), list(1:3, "int32", FALSE, 3L),
      list(c(TRUE, FALSE), "bool", FALSE, 2L),
      list(TRUE, "bool", FALSE, integer(0)), list(c(1i, 2), cplx, FALSE, 2L),
      list(1i, cplx, TRUE, integer(0)), list(matrix(1:6, 2), "int32", FALSE, c(2L, 3L)),
      list(array(0, c(2, 1, 3)), f, FALSE, c(2L, 1L, 3L)), list(numeric(0), f, FALSE, 0L))
    for (cs in cases) {
      a <- jaxR:::aval_of(as_jax(cs[[1L]]))
      expect_identical(a$dtype, cs[[2L]], info = deparse(cs[[1L]]))
      expect_identical(a$weak, cs[[3L]], info = deparse(cs[[1L]]))
      expect_identical(a$shape, cs[[4L]], info = deparse(cs[[1L]]))
    }
  })
})

test_that("R integer scalars are weak; their width is covered in test-core-dtypes.R", {
  skip_if_no_backend()
  expect_true(jaxR:::aval_of(as_jax(7L))$weak)
  expect_identical(with_jax_config("jax_enable_x64", FALSE, dtype(as_jax(7L))), "int32")
})

test_that("as_jax with an explicit dtype is strongly typed and converts like XLA", {
  skip_if_no_backend()
  expect_identical(jaxR:::aval_of(as_jax(2.5, "float32"))$weak, FALSE)
  expect_jax_equal(as_jax(c(1.7, -1.7, 127, -5), "int8"), c(1, -1, 127, -5), dtype = "int8")
  expect_jax_equal(as_jax(c(0.5, 0, -2), "bool"), c(TRUE, FALSE, TRUE), dtype = "bool")
  expect_jax_equal(as_jax(c(TRUE, FALSE), "float32"), c(1, 0), dtype = "float32")
  expect_jax_equal(as_jax(c(1.5 + 2i), "float32"), 1.5, dtype = "float32")
  expect_jax_equal(as_jax(c(4294967295, 1), "uint32"), c(4294967295, 1), dtype = "uint32")
  expect_jax_equal(as_jax(1:3, "float16"), c(1, 2, 3), dtype = "float16")
  expect_jax_equal(as_jax(c(0.1, 0.2), "bfloat16"), c(0.10009765625, 0.2001953125), dtype = "bfloat16")
  x <- as_jax(c(1, 2))
  expect_identical(as_jax(x), x)
  expect_identical(dtype(as_jax(x, "int32")), "int32")
})

test_that("as_jax keeps R layout: x[i, j] in R is element (i - 1, j - 1)", {
  skip_if_no_backend()
  m <- matrix(c(1, 2, 3, 4, 5, 6), 2, byrow = TRUE)
  x <- as_jax(m)
  expect_identical(dim(x), c(2L, 3L))
  expect_identical(as.array(x), m)
  expect_jax_equal(lax_index_in_dim(lax_index_in_dim(x, 0L, 0L, FALSE), 2L, 0L, FALSE), 3)
  a <- array(as.numeric(1:24), c(2, 3, 4))
  expect_identical(as.array(as_jax(a)), a)
  expect_jax_equal(lax_slice(as_jax(a), c(1L, 2L, 3L), c(2L, 3L, 4L)), array(a[2, 3, 4], c(1, 1, 1)))
  expect_identical(length(as_jax(a)), 24L)
})

test_that("as_jax rejects data that has no array meaning", {
  skip_if_no_backend()
  expect_error(as_jax("a"), class = "jax_type_error")
  expect_error(as_jax(list(1, 2)), class = "jax_type_error")
  expect_error(as_jax(factor("a")), class = "jax_type_error")
  expect_error(as_jax(NULL), class = "jax_type_error")
  expect_error(as_jax(function(x) x), class = "jax_type_error")
  expect_error(as_jax(c(1, NA)), "NA values")
  expect_error(as_jax(NA), "NA values")
  expect_true(is.nan(as.vector(as_jax(NaN))))
  expect_jax_equal(as_jax(c(Inf, -Inf)), c(Inf, -Inf))
})

test_that("host conversions return R data", {
  skip_if_no_backend()
  x <- as_jax(matrix(c(1.5, -2, 3, 4), 2))
  expect_identical(as.matrix(x), matrix(c(1.5, -2, 3, 4), 2))
  expect_identical(as.vector(x), c(1.5, -2, 3, 4))
  expect_identical(as.numeric(x), c(1.5, -2, 3, 4))
  expect_identical(as.integer(as_jax(c(2.7, -2.7))), c(2L, -2L))
  expect_identical(as.logical(as_jax(c(TRUE, FALSE))), c(TRUE, FALSE))
  expect_identical(as.complex(as_jax(1i)), 1i)
  expect_identical(device_get(list(a = as_jax(1:2), b = 3)), list(a = array(c(1, 2)), b = 3))
  y <- device_put(list(1, as_jax(2L)))
  expect_true(all(vapply(y, inherits, NA, "jax_array")))
  expect_identical(block_until_ready(x), x)
  expect_identical(as.vector(as_jax(c(-0, 0))), c(-0, 0))
  expect_identical(1 / as.vector(as_jax(-0)), -Inf)
})

test_that("printing shows dtype, shape and weak type", {
  skip_if_no_backend()
  expect_output(print(as_jax(c(1, 2, 3))), "^Array f32\\[3\\]\n\\[1\\] 1 2 3")
  expect_output(print(as_jax(2)), "Array f32\\[\\] weak_type")
  expect_output(print(as_jax(1:2)), "Array i32\\[2\\]\n\\[1\\] 1 2")
  expect_output(print(as_jax(c(TRUE, FALSE))), "Array bool\\[2\\]\n\\[1\\]  TRUE FALSE")
  expect_output(print(as_jax(array(1:8, c(2, 2, 2)))), "\\[0, , \\]")
  expect_output(print(as_jax(array(1:8, c(2, 2, 2)))), "\\[1, , \\]")
  expect_output(print(as_jax(1:3, "uint8")), "Array u8\\[3\\]")
  expect_output(print(jnp_zeros(c(0L, 2L, 2L))), "<empty>")
  expect_identical(format(as_jax(c(1, 2))), "Array(f32[2])")
  expect_output(print(ShapeDtypeStruct(c(2L, 3L), "int32")), "ShapedArray\\(i32\\[2,3\\]\\)")
  expect_output(print(ShapeDtypeStruct(integer(0), "float32", TRUE)), "ShapedArray\\(f32\\[\\], weak_type=True\\)")
  expect_output(str(as_jax(1:3)), "Array\\(i32\\[3\\]\\)")
  fmt <- NULL
  jit(function(x) { fmt <<- format(x); x })(as_jax(c(1, 2)))
  expect_identical(fmt, "Traced<f32[2]>with<jaxpr_trace>")
  expect_output(print(jaxR:::zero_of(jaxR:::new_aval(2L, "float32"))), "Zero\\(f32\\[2\\]\\)")
})

test_that("broadcast_shapes follows NumPy rules", {
  expect_identical(broadcast_shapes(c(2L, 1L), 3L), c(2L, 3L))
  expect_identical(broadcast_shapes(integer(0), c(4L, 1L, 2L), c(3L, 1L)), c(4L, 3L, 2L))
  expect_identical(broadcast_shapes(c(0L, 1L), 5L), c(0L, 5L))
  expect_identical(broadcast_shapes(), integer(0))
  expect_error(broadcast_shapes(c(2L, 3L), c(4L)), class = "jax_value_error")
  expect_error(broadcast_shapes(c(2L, 3L), c(4L)), "Incompatible shapes for broadcasting: \\(2, 3\\), \\(4\\)")
})

test_that("jnp_broadcast_to and jnp_broadcast_arrays broadcast (and only broadcast)", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2, 3))
  expect_jax_equal(jnp_broadcast_to(x, c(2L, 3L)), matrix(c(1, 2, 3), 2, 3, byrow = TRUE))
  expect_error(jnp_broadcast_to(x, c(3L, 2L)), "Incompatible shapes for broadcasting")
  expect_error(jnp_broadcast_to(as_jax(matrix(1:6, 2)), 6L), "Incompatible shapes")
  r <- jnp_broadcast_arrays(x, as_jax(matrix(c(1, 2), 2)), 5)
  expect_identical(lapply(r, dim), rep(list(c(2L, 3L)), 3))
  expect_identical(dtype(r[[3L]]), "float32")
})

test_that("ShapeDtypeStruct canonicalizes its dtype", {
  expect_identical(with_jax_config("jax_enable_x64", FALSE, ShapeDtypeStruct(2L, "float64"))$dtype, "float32")
  expect_identical(with_jax_config("jax_enable_x64", TRUE, ShapeDtypeStruct(2L, "float64"))$dtype, "float64")
})

test_that("configuration options can be read, set and restored", {
  old <- jax_config("jax_enable_x64")
  expect_identical(with_jax_config("jax_enable_x64", !old, jax_config("jax_enable_x64")), !old)
  expect_identical(jax_config("jax_enable_x64"), old)
  expect_error(with_jax_config("jax_enable_x64", !old, stop("boom")), "boom")
  expect_identical(jax_config("jax_enable_x64"), old)
  expect_identical(enable_x64(old), old)
  expect_true(is.list(jax_config()))
  expect_error(jax_config("no_such_option"), "unknown config option")
  expect_error(jax_config_update("no_such_option", 1), "unknown config option")
})
