# Error classes: concretization, type errors, escaped tracers, value errors.

test_that("converting a traced value to R data raises jax_concretization_error", {
  skip_if_no_backend()
  x <- as_jax(c(1, -2))
  expect_error(jit(function(v) if (as.logical(jnp_sum(v) > 0)) v else -v)(x),
               class = "jax_concretization_error")
  expect_error(jit(function(v) as.array(v))(x), class = "jax_concretization_error")
  expect_error(jit(function(v) as.numeric(v))(x), class = "jax_concretization_error")
  expect_error(vmap(function(v) as.vector(v))(x), class = "jax_concretization_error")
  expect_error(jit(function(v) jnp_arange(jnp_sum(v)))(as_jax(3L)), class = "jax_concretization_error")
  e <- tryCatch(jit(function(v) as.logical(v))(x), error = identity)
  expect_match(conditionMessage(e), "Abstract tracer value encountered where concrete value is expected")
  expect_match(conditionMessage(e), "lax_cond")
})

test_that("grad and jvp tracers of concrete values can be used in R control flow", {
  skip_if_no_backend()
  f <- function(x) if (as.logical(x > 0)) x^2 else -x
  expect_jax_equal(grad(f)(as_jax(3)), 6)
  expect_jax_equal(grad(f)(as_jax(-3)), -1)
  expect_jax_equal(jvp(f, list(as_jax(2)), list(as_jax(1)))[[2L]], 4)
})

test_that("type errors carry class jax_type_error", {
  skip_if_no_backend()
  expect_error(lax_add(as_jax(1, "float32"), as_jax(1L, "int32")), class = "jax_type_error")
  expect_error(lax_add(as_jax(c(1, 2)), as_jax(c(1, 2, 3))), class = "jax_type_error")
  expect_error(lax_floor(as_jax(1L)), class = "jax_type_error")
  expect_error(grad(function(x) jnp_sum(x))(as_jax(1:2)), class = "jax_type_error")
  expect_error(grad(function(x) x)(as_jax(c(1, 2))), class = "jax_type_error")
  expect_error(lax_reshape(as_jax(1:6), c(4L, 2L)), class = "jax_type_error")
  no_batch <- list(integer(0), integer(0))
  expect_error(lax_dot_general(as_jax(matrix(1, 2, 3)), as_jax(matrix(1, 2, 3)), list(list(1L, 1L), no_batch)),
               NA)
  expect_error(lax_dot_general(as_jax(matrix(1, 2, 3)), as_jax(matrix(1, 2, 3)), list(list(1L, 0L), no_batch)),
               class = "jax_type_error")
  expect_error(jaxR:::promote_types(c("float0", "int32"), c(FALSE, FALSE)), class = "jax_type_error")
})

test_that("leaked tracers raise jax_escaped_tracer_error", {
  skip_if_no_backend()
  leaked <- NULL
  jit(function(x) { leaked <<- x; x * 2 })(as_jax(c(1, 2)))
  expect_s3_class(leaked, "jax_tracer")
  expect_error(leaked + 1, class = "jax_escaped_tracer_error")
  expect_error(jnp_sin(leaked), class = "jax_escaped_tracer_error")
  e <- tryCatch(lax_neg(leaked), error = identity)
  expect_match(conditionMessage(e), "Encountered an unexpected tracer")
  leaked <- NULL
  vmap(function(x) { leaked <<- x; x })(as_jax(c(1, 2)))
  expect_error(lax_mul(leaked, leaked), class = "jax_escaped_tracer_error")
  leaked <- NULL
  grad(function(x) { leaked <<- x; jnp_sum(x) })(as_jax(c(1, 2)))
  expect_error(lax_exp(leaked), class = "jax_escaped_tracer_error")
})

test_that("axis and shape errors carry class jax_value_error", {
  skip_if_no_backend()
  x <- as_jax(matrix(1, 2, 3))
  expect_error(jnp_sum(x, axis = 2L), class = "jax_value_error")
  expect_error(jnp_sum(x, axis = -3L), class = "jax_value_error")
  expect_error(jnp_sum(x, axis = 2L), "axis 2 is out of bounds for array of rank 2")
  expect_error(broadcast_shapes(2L, 3L), class = "jax_value_error")
  expect_error(jnp_add(x, as_jax(c(1, 2))), class = "jax_value_error")
})

test_that("all jaxR errors are conditions of class jax_error", {
  skip_if_no_backend()
  e <- tryCatch(lax_add(as_jax(1, "float32"), as_jax(1L, "int32")), error = identity)
  expect_s3_class(e, "jax_error")
  expect_null(conditionCall(e))
})
