# Top level jax API (devices, typeof, compile time evaluation, custom_gradient,
# closure_convert, fwd_and_bwd, config context managers) and jax.dtypes. Expected
# values from jax 0.12.0.1.

test_that("named_call and named_scope are transparent", {
  f <- function(x) x + 1
  expect_identical(named_call(f, name = "add_one"), f)
  expect_identical(named_scope("block", 1 + 2), 3)
  expect_error(named_call(1), "fun must be a function")
  expect_error(named_scope(1, 2), "name argument must be a string")
})

test_that("jax_typeof returns the abstract value", {
  skip_if_no_backend()
  a <- jax_typeof(as_jax(c(1, 2, 3)))
  expect_identical(a$shape, 3L)
  expect_identical(a$dtype, "float32")
  expect_false(a$weak)
  w <- jax_typeof(1)
  expect_identical(w$shape, integer(0))
  expect_true(w$weak)
  expect_identical(jax_typeof(TRUE)$dtype, "bool")
  expect_identical(format(jax_typeof(as_jax(1L))), "i32[]")
  expect_error(jax_typeof("a"), "of type character is not a valid JAX type")
  shapes <- NULL
  jit(function(x) { shapes <<- jax_typeof(x); x })(as_jax(matrix(1, 2, 3)))
  expect_identical(shapes$shape, c(2L, 3L))
})

test_that("devices, processes and the backend", {
  skip_if_no_backend()
  expect_identical(default_backend(), "cpu")
  # the PJRT CPU plugin exposes several devices; jaxR computes on the first one
  expect_identical(device_count(), length(devices()))
  expect_identical(local_device_count(), device_count())
  expect_identical(local_devices(), devices())
  expect_error(local_devices(process_index = 1L), "out of range")
  expect_identical(process_index(), 0L)
  expect_identical(process_count(), 1L)
  expect_identical(process_indices(), 0L)
  expect_warning(expect_identical(host_id(), 0L), "renamed to jax.process_index")
  expect_warning(expect_identical(host_count(), 1L), "renamed to jax.process_count")
  expect_warning(expect_identical(host_ids(), 0L), "renamed to jax.process_indices")
  expect_identical(default_device(devices()[1L], 1 + 1), 2)
  expect_identical(default_device(NULL, 3), 3)
  expect_error(default_device("tpu:7", 1), "unknown device")
  info <- print_environment_info(return_string = TRUE)
  expect_match(info, "jaxR:")
  expect_match(info, paste0("device info: cpu-", device_count(), ", ", device_count(),
                            " local devices"), fixed = TRUE)
  expect_output(print_environment_info(), "process_count: 1")
  expect_null(effects_barrier())
  x <- as_jax(c(1, 2))
  expect_identical(copy_to_host_async(list(x, 3)), list(x, 3))
})

test_that("ensure_compile_time_eval folds constants inside jit", {
  skip_if_no_backend()
  seen <- NULL
  f <- jit(function(x) {
    z <- ensure_compile_time_eval({
      y <- jnp_sin(as_jax(3))
      seen <<- y
      x1 <- x + 1
      list(jnp_sin(y) > 0, x1)
    })
    expect_true(jaxR:::is_tracer(z[[2L]]))
    if (as.logical(z[[1L]])) jnp_sin(x) + z[[2L]] else jnp_cos(x)
  })
  expect_jax_equal(f(as_jax(0.5)), sin(0.5) + 1.5)
  expect_false(jaxR:::is_tracer(seen))
  expect_jax_equal(seen, 0.14112, tolerance = 1e-5)
  # control flow on traced values inside the block is staged (through partial evaluation)
  h <- jit(function(x) ensure_compile_time_eval({
    k <- jnp_cos(as_jax(0))
    lax_cond(x > 0, function(v) v * k, function(v) v - k, x) + lax_scan(function(c, e) list(c + e, c), k,
                                                                         jnp_ones(3L))[[1L]]
  }))
  expect_jax_equal(h(as_jax(2)), 2 + 4)
  expect_jax_equal(h(as_jax(-2)), -3 + 4)
  # outside of staging it just evaluates
  expect_jax_equal(ensure_compile_time_eval(as_jax(2) * 3), 6)
  # without it, the concrete computation is staged and cannot drive R control flow
  g <- jit(function(x) if (as.logical(jnp_sin(jnp_sin(as_jax(3))) > 0)) x else -x)
  expect_error(g(as_jax(1)), class = "jax_concretization_error")
})

test_that("custom_gradient", {
  skip_if_no_backend()
  f <- custom_gradient(function(x) list(x^2, function(g) list(g * x)))
  expect_jax_equal(f(as_jax(3)), 9)
  expect_jax_equal(grad(f)(as_jax(3)), 3)
  expect_jax_equal(jit(grad(f))(as_jax(3)), 3)
  expect_jax_equal(vmap(grad(f))(as_jax(c(1, 2, 3))), c(1, 2, 3))
  f1 <- custom_gradient(function(x) list(jnp_sin(x), function(g) g * 2))
  expect_jax_equal(grad(f1)(as_jax(0.3)), 2)
  h <- custom_gradient(function(x, y) list(x * y, function(g) list(g * y, g * x)))
  gr <- grad(h, argnums = 0:1)(as_jax(3), as_jax(4))
  expect_jax_equal(gr[[1L]], 4)
  expect_jax_equal(gr[[2L]], 3)
  # values computed inside the rule from closed-over primals
  k <- custom_gradient(function(x) list(jnp_sum(x^3), function(g) g * jnp_cos(x)))
  expect_jax_equal(grad(k)(as_jax(c(0, 1))), cos(c(0, 1)))
  expect_error(grad(custom_gradient(function(x) x))(as_jax(1)), "must return list\\(value, vjp_function\\)")
})

test_that("closure_convert hoists perturbed closed-over values", {
  skip_if_no_backend()
  outer <- function(x) {
    cc <- closure_convert(function(y) jnp_sin(x) * y, 1)
    list(do.call(cc[[1L]], c(list(3), cc[[2L]])), length(cc[[2L]]))
  }
  r <- outer(as_jax(2))
  expect_jax_equal(r[[1L]], 2.7278922)
  expect_identical(r[[2L]], 0L)
  nh <- NULL
  g <- grad(function(x) {
    cc <- closure_convert(function(y) jnp_sin(x) * y, 1)
    nh <<- length(cc[[2L]])
    do.call(cc[[1L]], c(list(3), cc[[2L]]))
  })(as_jax(2))
  expect_jax_equal(g, -1.2484405)
  expect_identical(nh, 1L)
  cc <- closure_convert(function(a, b) a + b[[1L]] + b[[2L]], 1, list(2, 3))
  expect_jax_equal(cc[[1L]](1, list(2, 3)), 6)
  expect_error(cc[[1L]](1, 2), "must have the same Pytree structure")
})

test_that("fwd_and_bwd splits a VJP into reusable halves", {
  skip_if_no_backend()
  x <- as_jax(matrix(c(1, 2, 3, 4), 2, byrow = TRUE))
  W <- as_jax(matrix(c(0.5, -1, 2, 1.5), 2, byrow = TRUE))
  ct <- as_jax(matrix(c(1, 0, 2, -1), 2, byrow = TRUE))
  fb <- fwd_and_bwd(function(x, W) jnp_sin(jnp_matmul(x, W)), argnums = c(0L, 1L))
  r <- fb[[1L]](x, W)
  expect_equal(as.vector(t(as.array(r[[1L]]))), c(-0.9775301, 0.9092974, -0.07515112, 0.14112), tolerance = 1e-6)
  for (i in 1:2) {
    g <- fb[[2L]](r[[2L]], ct)
    expect_equal(as.vector(t(as.array(g[[1L]]))), c(-0.1053979, -0.4215916, -1.9871647, -2.5037), tolerance = 1e-6)
    expect_equal(as.vector(t(as.array(g[[2L]]))), c(-6.193829, 2.969977, -8.398969, 3.95997), tolerance = 1e-6)
  }
  fb1 <- fwd_and_bwd(function(x, W) jnp_sum(jnp_sin(jnp_matmul(x, W))), argnums = 0L, jitted = FALSE)
  r1 <- fb1[[1L]](x, W)
  expect_jax_equal(r1[[1L]], -0.0022638291)
  expect_equal(as.vector(t(as.array(fb1[[2L]](r1[[2L]], 1)))), c(0.3107489, -1.0458119, 0.4914064, -3.4793332), tolerance = 1e-6)
  fa <- fwd_and_bwd(function(x) list(jnp_sum(x^2), jnp_max(x)), has_aux = TRUE)
  ra <- fa[[1L]](as_jax(c(1, 2)))
  expect_jax_equal(ra[[1L]], 5)
  expect_jax_equal(ra[[3L]], 2)
  expect_jax_equal(fa[[2L]](ra[[2L]], 1), c(2, 4))
})

test_that("configuration context managers set and restore options", {
  skip_if_no_backend()
  expect_error(numpy_rank_promotion("raise", jnp_add(jnp_ones(3L), jnp_ones(c(2L, 3L)))),
               "jax_numpy_rank_promotion='raise'")
  expect_identical(jax_config("jax_numpy_rank_promotion"), "allow")
  expect_error(numpy_rank_promotion("bad", 1), "must be one of")
  expect_identical(default_matmul_precision("highest", jax_config("jax_default_matmul_precision")), "highest")
  expect_null(jax_config("jax_default_matmul_precision"))
  expect_false(threefry_partitionable(FALSE, jax_config("jax_threefry_partitionable")))
  expect_true(jax_config("jax_threefry_partitionable"))
  expect_identical(default_prng_impl("rbg", jax_config("jax_default_prng_impl")), "rbg")
  expect_identical(jax_config("jax_default_prng_impl"), "threefry2x32")
})

test_that("jax.dtypes functions", {
  skip_if_no_backend()
  expect_identical(float0, "float0")
  expect_identical(dtypes_float0, "float0")
  expect_identical(dtypes_canonicalize_dtype("float64"), "float32")
  expect_identical(dtypes_canonicalize_dtype("int64"), "int32")
  expect_identical(dtypes_canonicalize_dtype("bool"), "bool")
  expect_error(dtypes_canonicalize_dtype("key<fry>"), "not a valid dtype")
  expect_identical(dtypes_canonicalize_dtype("key<fry>", allow_extended_dtype = TRUE), "key<fry>")
  expect_identical(dtypes_result_type(1, 2.5), "float32")
  expect_identical(dtypes_result_type(1, 2.5, return_weak_type_flag = TRUE), list("float32", TRUE))
  expect_identical(dtypes_result_type(as_jax(1L, "int8"), 2L), "int8")
  expect_identical(dtypes_result_type(as_jax(1L, "int8"), 2), "float32")
  expect_identical(dtypes_result_type("float32", "int32"), "float32")
  expect_true(dtypes_issubdtype("float32", "floating"))
  expect_true(dtypes_issubdtype("key<fry>", dtypes_prng_key))
  expect_true(dtypes_issubdtype("key<fry>", dtypes_extended))
  expect_false(dtypes_issubdtype("float32", dtypes_extended))
  expect_false(dtypes_issubdtype("key<fry>", "floating"))
  expect_identical(dtypes_scalar_type_of(jnp_ones(2L)), "double")
  expect_identical(dtypes_scalar_type_of(as_jax(1L, "int8")), "integer")
  expect_identical(dtypes_scalar_type_of(TRUE), "logical")
  expect_identical(dtypes_scalar_type_of(1i), "complex")
  expect_identical(dtypes_scalar_type_of(as_jax(1, "uint32")), "integer")
  expect_identical(dtypes_itemsize_bits("bool"), 8L)
  expect_identical(dtypes_itemsize_bits("float32"), 32L)
  expect_identical(dtypes_itemsize_bits(dtypes_bfloat16), 16L)
  expect_error(dtypes_itemsize_bits("float0"), "unexpected input")
  expect_identical(dtypes_finfo("bfloat16")$eps, 0.0078125)
  expect_identical(dtypes_iinfo("int8")$max, 127)
  cond <- dtypes_TypePromotionError("cannot promote")
  expect_s3_class(cond, "jax_type_promotion_error")
  expect_error(stop(cond), "cannot promote", class = "jax_type_promotion_error")
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(dtypes_canonicalize_dtype("float64"), "float64")
    expect_identical(dtypes_result_type(1, 2.5), "float64")
  })
})
