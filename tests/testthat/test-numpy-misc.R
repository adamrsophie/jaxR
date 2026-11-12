# jax.numpy scalar types, dtype functions, constants and metadata (R/numpy_misc.R).
# Tables and limits from jax 0.12.0.1 (numpy / ml_dtypes); rows and columns follow misc_dts.

misc_dts <- c("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128")
misc_cast_safe <- matrix(c(
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE), 15, byrow = TRUE)
misc_cast_same_kind <- matrix(c(
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE), 15, byrow = TRUE)
misc_promote_x32 <- matrix(c(
  "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int8", "int8", "int16", "int32", "int64", "int16", "int32", "int32", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int16", "int16", "int16", "int32", "int64", "int16", "int32", "int32", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int32", "int32", "int32", "int32", "int64", "int32", "int32", "int32", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint8", "int16", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint16", "int32", "int32", "int32", "int64", "uint16", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint32", "int32", "int32", "int32", "int64", "uint32", "uint32", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint64", "float64", "float64", "float64", "float64", "uint64", "uint64", "uint64", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32", "float64", "complex64", "complex128",
  "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float16", "float32", "float64", "complex64", "complex128",
  "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float64", "complex64", "complex128",
  "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "complex128", "complex128",
  "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex128", "complex64", "complex128",
  "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128"), 15, byrow = TRUE)
misc_result_x32 <- matrix(c(
  "bool", "int8", "int16", "int32", "int32", "uint8", "uint16", "uint32", "uint32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "int8", "int8", "int16", "int32", "int32", "int16", "int32", "int32", "int32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "int16", "int16", "int16", "int32", "int32", "int16", "int32", "int32", "int32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "int32", "int32", "int32", "int32", "int32", "int32", "int32", "int32", "int32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "int32", "int32", "int32", "int32", "int32", "int32", "int32", "int32", "int32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "uint8", "int16", "int16", "int32", "int32", "uint8", "uint16", "uint32", "uint32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "uint16", "int32", "int32", "int32", "int32", "uint16", "uint16", "uint32", "uint32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "uint32", "int32", "int32", "int32", "int32", "uint32", "uint32", "uint32", "uint32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "uint32", "int32", "int32", "int32", "int32", "uint32", "uint32", "uint32", "uint32", "bfloat16", "float16", "float32", "float32", "complex64", "complex64",
  "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32", "float32", "complex64", "complex64",
  "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float16", "float32", "float32", "complex64", "complex64",
  "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "complex64", "complex64",
  "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "complex64", "complex64",
  "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64",
  "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64"), 15, byrow = TRUE)
misc_promote_x64 <- matrix(c(
  "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int8", "int8", "int16", "int32", "int64", "int16", "int32", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int16", "int16", "int16", "int32", "int64", "int16", "int32", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int32", "int32", "int32", "int32", "int64", "int32", "int32", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint8", "int16", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint16", "int32", "int32", "int32", "int64", "uint16", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint32", "int64", "int64", "int64", "int64", "uint32", "uint32", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint64", "float64", "float64", "float64", "float64", "uint64", "uint64", "uint64", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32", "float64", "complex64", "complex128",
  "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float16", "float32", "float64", "complex64", "complex128",
  "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float64", "complex64", "complex128",
  "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "complex128", "complex128",
  "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex128", "complex64", "complex128",
  "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128"), 15, byrow = TRUE)
misc_result_x64 <- matrix(c(
  "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int8", "int8", "int16", "int32", "int64", "int16", "int32", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int16", "int16", "int16", "int32", "int64", "int16", "int32", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int32", "int32", "int32", "int32", "int64", "int32", "int32", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "float64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint8", "int16", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint16", "int32", "int32", "int32", "int64", "uint16", "uint16", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint32", "int64", "int64", "int64", "int64", "uint32", "uint32", "uint32", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "uint64", "float64", "float64", "float64", "float64", "uint64", "uint64", "uint64", "uint64", "bfloat16", "float16", "float32", "float64", "complex64", "complex128",
  "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "float32", "float64", "complex64", "complex128",
  "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float16", "float32", "float64", "complex64", "complex128",
  "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float64", "complex64", "complex128",
  "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "complex128", "complex128",
  "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex128", "complex64", "complex128",
  "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128"), 15, byrow = TRUE)
misc_cats <- c("generic", "number", "integer", "signedinteger", "unsignedinteger", "inexact", "floating", "complexfloating", "character", "flexible")
misc_subdtype <- matrix(c(
  TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
  TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE,
  TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE,
  TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE,
  TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE,
  TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE,
  TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE), 15, byrow = TRUE)
misc_kinds <- c("bool", "signed integer", "unsigned integer", "integral", "real floating", "complex floating", "numeric")
misc_isdtype <- matrix(c(
  TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE,
  FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE,
  FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE,
  FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE,
  FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE,
  FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE,
  FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE,
  FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE), 15, byrow = TRUE)
misc_finfo_float16 <- list(dtype = "float16", bits = 16, eps = 0x1.0000000000000p-10, epsneg = 0x1.0000000000000p-11, max = 0x1.ffc0000000000p+15, min = -0x1.ffc0000000000p+15, tiny = 0x1.0000000000000p-14, smallest_normal = 0x1.0000000000000p-14, smallest_subnormal = 0x1.0000000000000p-24, resolution = 0x1.0640000000000p-10, precision = 3, machep = -10, negep = -11, minexp = -14, maxexp = 16, nmant = 10, nexp = 5, iexp = 5)
misc_finfo_bfloat16 <- list(dtype = "bfloat16", bits = 16, eps = 0x1.0000000000000p-7, epsneg = 0x1.0000000000000p-8, max = 0x1.fe00000000000p+127, min = -0x1.fe00000000000p+127, tiny = 0x1.0000000000000p-126, smallest_normal = 0x1.0000000000000p-126, smallest_subnormal = 0x1.0000000000000p-133, resolution = 0x1.4800000000000p-7, precision = 2, machep = -7, negep = -8, minexp = -126, maxexp = 128, nmant = 7, nexp = 8, iexp = 8)
misc_finfo_float32 <- list(dtype = "float32", bits = 32, eps = 0x1.0000000000000p-23, epsneg = 0x1.0000000000000p-24, max = 0x1.fffffe0000000p+127, min = -0x1.fffffe0000000p+127, tiny = 0x1.0000000000000p-126, smallest_normal = 0x1.0000000000000p-126, smallest_subnormal = 0x1.0000000000000p-149, resolution = 0x1.0c6f7a0000000p-20, precision = 6, machep = -23, negep = -24, minexp = -126, maxexp = 128, nmant = 23, nexp = 8, iexp = 8)
misc_finfo_float64 <- list(dtype = "float64", bits = 64, eps = 0x1.0000000000000p-52, epsneg = 0x1.0000000000000p-53, max = 0x1.fffffffffffffp+1023, min = -0x1.fffffffffffffp+1023, tiny = 0x1.0000000000000p-1022, smallest_normal = 0x1.0000000000000p-1022, smallest_subnormal = 0x0.0000000000001p-1022, resolution = 0x1.203af9ee75616p-50, precision = 15, machep = -52, negep = -53, minexp = -1022, maxexp = 1024, nmant = 52, nexp = 11, iexp = 11)
misc_finfo_complex64 <- list(dtype = "float32", bits = 32, eps = 0x1.0000000000000p-23, epsneg = 0x1.0000000000000p-24, max = 0x1.fffffe0000000p+127, min = -0x1.fffffe0000000p+127, tiny = 0x1.0000000000000p-126, smallest_normal = 0x1.0000000000000p-126, smallest_subnormal = 0x1.0000000000000p-149, resolution = 0x1.0c6f7a0000000p-20, precision = 6, machep = -23, negep = -24, minexp = -126, maxexp = 128, nmant = 23, nexp = 8, iexp = 8)
misc_finfo_complex128 <- list(dtype = "float64", bits = 64, eps = 0x1.0000000000000p-52, epsneg = 0x1.0000000000000p-53, max = 0x1.fffffffffffffp+1023, min = -0x1.fffffffffffffp+1023, tiny = 0x1.0000000000000p-1022, smallest_normal = 0x1.0000000000000p-1022, smallest_subnormal = 0x0.0000000000001p-1022, resolution = 0x1.203af9ee75616p-50, precision = 15, machep = -52, negep = -53, minexp = -1022, maxexp = 1024, nmant = 52, nexp = 11, iexp = 11)
misc_iinfo_int8 <- list(dtype = "int8", kind = "i", bits = 8, min = -128, max = 127)
misc_iinfo_int16 <- list(dtype = "int16", kind = "i", bits = 16, min = -32768, max = 32767)
misc_iinfo_int32 <- list(dtype = "int32", kind = "i", bits = 32, min = -2147483648, max = 2147483647)
misc_iinfo_uint8 <- list(dtype = "uint8", kind = "u", bits = 8, min = 0, max = 255)
misc_iinfo_uint16 <- list(dtype = "uint16", kind = "u", bits = 16, min = 0, max = 65535)
misc_iinfo_uint32 <- list(dtype = "uint32", kind = "u", bits = 32, min = 0, max = 4294967295)

test_that("scalar types construct arrays of their dtype", {
  skip_if_no_backend()
  expect_jax_equal(jnp_float32(3), 3, dtype = "float32")
  expect_false(jnp_float32(3)$aval$weak)
  expect_jax_equal(jnp_float64(c(1.5, 2)), c(1.5, 2), dtype = "float32")
  expect_jax_equal(jnp_int_(3.7), 3, dtype = "int32")
  expect_jax_equal(jnp_int32(jnp_float32(-2.5)), -2, dtype = "int32")
  expect_jax_equal(jnp_bool_(c(0, 2)), c(FALSE, TRUE), dtype = "bool")
  expect_jax_equal(jnp_bool(1L), TRUE, dtype = "bool")
  expect_jax_equal(jnp_uint8(c(3, 250)), c(3, 250), dtype = "uint8")
  expect_jax_equal(jnp_complex64(2), 2 + 0i, dtype = "complex64")
  expect_jax_equal(jnp_float16(1 / 3), 0x1.554p-2, dtype = "float16", tolerance = 0)
  expect_jax_equal(jnp_bfloat16(1 / 3), 0x1.56p-2, dtype = "bfloat16", tolerance = 0)
  expect_jax_equal(jnp_single(1), 1, dtype = "float32")
  expect_jax_equal(jnp_csingle(1), 1 + 0i, dtype = "complex64")
  expect_jax_equal(jit(function(x) jnp_int16(x))(c(1.9, -1.9)), c(1, -1), dtype = "int16")
  enable_x64()
  on.exit(enable_x64(FALSE))
  expect_jax_equal(jnp_float64(1 / 3), 1 / 3, dtype = "float64", tolerance = 0)
  expect_jax_equal(jnp_int_(3), 3, dtype = "int64")
  expect_jax_equal(jnp_uint(3), 3, dtype = "uint64")
  expect_jax_equal(jnp_float_(3), 3, dtype = "float64")
  expect_jax_equal(jnp_complex_(3), 3 + 0i, dtype = "complex128")
  expect_jax_equal(jnp_double(2), 2, dtype = "float64")
  expect_jax_equal(jnp_cdouble(2), 2 + 0i, dtype = "complex128")
  expect_output(print(jnp_float32), "jax.numpy scalar type float32")
})

test_that("jnp_dtype resolves names like NumPy", {
  skip_if_no_backend()
  expect_identical(jnp_dtype("float"), "float64")
  expect_identical(jnp_dtype("int"), "int64")
  expect_identical(jnp_dtype("uint"), "uint64")
  expect_identical(jnp_dtype("complex"), "complex128")
  expect_identical(jnp_dtype("half"), "float16")
  expect_identical(jnp_dtype("single"), "float32")
  expect_identical(jnp_dtype("double"), "float64")
  expect_identical(jnp_dtype("csingle"), "complex64")
  expect_identical(jnp_dtype("cdouble"), "complex128")
  expect_identical(jnp_dtype("int_"), "int64")
  expect_identical(jnp_dtype(jnp_float64), "float64")
  expect_identical(jnp_dtype(jnp_int_), "int64")
  expect_identical(jnp_dtype(jnp_ones(2L)), "float32")
  expect_error(jnp_dtype(1.5), "Cannot interpret '1.5' as a data type", fixed = TRUE)
  expect_error(jnp_dtype("foo"), "data type 'foo' not understood", fixed = TRUE)
})

test_that("can_cast, promote_types and result_type match the JAX tables", {
  skip_if_no_backend()
  for (i in seq_along(misc_dts)) for (j in seq_along(misc_dts)) {
    a <- misc_dts[i]
    b <- misc_dts[j]
    info <- paste(a, b)
    expect_identical(jnp_can_cast(a, b), misc_cast_safe[i, j], info = info)
    expect_identical(jnp_can_cast(a, b, "same_kind"), misc_cast_same_kind[i, j], info = info)
    expect_identical(jnp_promote_types(a, b), misc_promote_x32[i, j], info = info)
    expect_identical(jnp_result_type(a, b), misc_result_x32[i, j], info = info)
  }
  expect_true(jnp_can_cast("int8", "int8", "no"))
  expect_false(jnp_can_cast("int8", "int16", "equiv"))
  expect_true(jnp_can_cast("float64", "int8", "unsafe"))
  expect_true(jnp_can_cast(jnp_float32, jnp_ones(2L, "complex64")))
  expect_error(jnp_can_cast("int8", "int16", "foo"), "casting must be one of")
  enable_x64()
  on.exit(enable_x64(FALSE))
  for (i in seq_along(misc_dts)) for (j in seq_along(misc_dts)) {
    info <- paste(misc_dts[i], misc_dts[j])
    expect_identical(jnp_promote_types(misc_dts[i], misc_dts[j]), misc_promote_x64[i, j], info = info)
    expect_identical(jnp_result_type(misc_dts[i], misc_dts[j]), misc_result_x64[i, j], info = info)
  }
})

test_that("result_type promotes values with weak types", {
  skip_if_no_backend()
  expect_identical(jnp_result_type(NULL), "float32")
  expect_identical(jnp_result_type(TRUE, 1L), "int32")
  expect_identical(jnp_result_type(1L, 1, 1i), "complex64")
  expect_identical(jnp_result_type(jnp_ones(2L, "int8"), 2), "float32")
  expect_identical(jnp_result_type(jnp_ones(2L, "int8"), 2L), "int8")
  expect_identical(jnp_result_type(1, "int8"), "float32")
  expect_identical(jnp_result_type(1L, "uint8"), "uint8")
  expect_identical(jnp_result_type(jnp_float16), "float16")
  expect_error(jnp_result_type(), "at least one array or dtype is required", fixed = TRUE)
})

test_that("issubdtype and isdtype match JAX", {
  skip_if_no_backend()
  cats <- list(jnp_generic, jnp_number, jnp_integer, jnp_signedinteger, jnp_unsignedinteger,
               jnp_inexact, jnp_floating, jnp_complexfloating, jnp_character, jnp_flexible)
  expect_identical(unlist(cats), misc_cats)
  for (i in seq_along(misc_dts)) {
    for (j in seq_along(cats)) {
      expect_identical(jnp_issubdtype(misc_dts[i], cats[[j]]), misc_subdtype[i, j],
                       info = paste(misc_dts[i], misc_cats[j]))
    }
    for (j in seq_along(misc_kinds)) {
      expect_identical(jnp_isdtype(misc_dts[i], misc_kinds[j]), misc_isdtype[i, j],
                       info = paste(misc_dts[i], misc_kinds[j]))
    }
  }
  expect_false(jnp_issubdtype("float64", "float32"))
  expect_true(jnp_issubdtype(jnp_float32, "float32"))
  expect_true(jnp_issubdtype(jnp_floating, jnp_inexact))
  expect_false(jnp_issubdtype(jnp_inexact, jnp_floating))
  expect_false(jnp_issubdtype(jnp_floating, "float32"))
  expect_true(jnp_issubdtype(jnp_ones(2L, "uint16"), jnp_unsignedinteger))
  expect_true(jnp_isdtype("int32", c("real floating", "signed integer")))
  expect_true(jnp_isdtype("int32", list("real floating", jnp_int32)))
  expect_true(jnp_isdtype("float32", "float32"))
  expect_false(jnp_isdtype("bool", "numeric"))
  expect_error(jnp_isdtype("int32", "foo"), paste0(
    "Unrecognized kind='foo' expected one of ['bool', 'signed integer', 'unsigned integer', ",
    "'integral', 'real floating', 'complex floating', 'numeric'], or a compatible input for jnp.dtype()"),
    fixed = TRUE)
})

test_that("finfo and iinfo match ml_dtypes", {
  skip_if_no_backend()
  for (d in c("float16", "bfloat16", "float32", "float64", "complex64", "complex128")) {
    expect_equal(jnp_finfo(d), get(paste0("misc_finfo_", d)), tolerance = 0, info = d)
  }
  expect_identical(jnp_finfo(jnp_float64)$eps, 2^-52)
  expect_identical(jnp_finfo(jnp_ones(2L))$dtype, "float32")
  for (d in c("int8", "int16", "int32", "uint8", "uint16", "uint32")) {
    expect_equal(jnp_iinfo(d), get(paste0("misc_iinfo_", d)), tolerance = 0, info = d)
  }
  expect_identical(jnp_iinfo("int64")$min, -2^63)
  expect_identical(jnp_iinfo(jnp_uint64)$bits, 64L)
  expect_error(jnp_finfo("int32"), "data type dtype('int32') not compatible with finfo", fixed = TRUE)
  expect_error(jnp_iinfo("float32"), "Invalid integer data type 'f'.", fixed = TRUE)
})

test_that("constants match jax.numpy", {
  skip_if_no_backend()
  expect_identical(jnp_pi, 0x1.921fb54442d18p+1)
  expect_identical(jnp_e, 0x1.5bf0a8b145769p+1)
  expect_identical(jnp_euler_gamma, 0x1.2788cfc6fb619p-1)
  expect_identical(jnp_inf, Inf)
  expect_true(is.nan(jnp_nan))
  expect_identical(jnp_newaxis, newaxis)
  x <- jnp_arange(3L)
  expect_identical(jnp_shape(x[jnp_newaxis, ]), c(1L, 3L))
  expect_jax_equal(jnp_sin(jnp_pi / 2), 1, dtype = "float32")
})

test_that("print options are stored like NumPy's", {
  skip_if_no_backend()
  opts <- jnp_get_printoptions()
  expect_identical(opts[c("edgeitems", "threshold", "floatmode", "precision", "suppress", "linewidth",
                          "nanstr", "infstr", "sign", "legacy")],
                   list(edgeitems = 3L, threshold = 1000L, floatmode = "maxprec", precision = 8L,
                        suppress = FALSE, linewidth = 75L, nanstr = "nan", infstr = "inf", sign = "-",
                        legacy = FALSE))
  expect_identical(jnp_printoptions(jnp_get_printoptions()$precision, precision = 3L), 3L)
  expect_identical(jnp_get_printoptions()$precision, 8L)
  old <- jnp_set_printoptions(suppress = TRUE)
  expect_true(jnp_get_printoptions()$suppress)
  jnp_set_printoptions(suppress = old$suppress)
  expect_false(jnp_get_printoptions()$suppress)
})

test_that("array API metadata matches JAX", {
  skip_if_no_backend()
  expect_identical(jnp_array_api_version, "2025.12")
  info <- jnp_array_namespace_info()
  expect_identical(info$capabilities(), list("boolean indexing" = FALSE, "data-dependent shapes" = FALSE,
                                             "max dimensions" = 64L))
  expect_null(info$default_device())
  expect_null(info$devices()[[1L]])
  expect_true(length(info$devices()) >= 2L)
  expect_identical(info$default_dtypes(), list("real floating" = "float32", "complex floating" = "complex64",
                                               integral = "int32", indexing = "int32"))
  expect_setequal(info$dtypes(), c("complex64", "uint32", "float32", "int8", "int32", "uint16", "int16",
                                   "uint8", "bool"))
  expect_setequal(info$dtypes(kind = "integral"), c("int8", "uint16", "uint32", "int16", "uint8", "int32"))
  expect_setequal(info$dtypes(kind = c("bool", "real floating")), c("bool", "float32"))
  enable_x64()
  on.exit(enable_x64(FALSE))
  expect_setequal(info$dtypes(kind = "integral"), c("int8", "int64", "uint16", "uint32", "uint64", "int16",
                                                    "uint8", "int32"))
  expect_identical(info$default_dtypes()$indexing, "int64")
})
