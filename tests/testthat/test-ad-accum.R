# Ports of the with_logs / with_accums / with_refs tests of tests/custom_api_test.py.
# Reference values computed with JAX 0.12.

num <- function(x) as.numeric(as.array(x))

test_that("custom_gradient with_logs returns logs from the backward pass", {
  skip_if_no_backend()
  f <- custom_gradient(function(x, y) {
    list(x * y, function(g) list(list(g * y, g * x), list(f = list(ct_out = g, x = x))))
  }, with_logs = TRUE)
  expect_equal(num(f(3, 4)), 12)
  expect_equal(num(grad(f)(3, 4)), 4)
  fv <- vjp(f, 3, 4)[[2]]
  r <- fv$with_logs(1)
  expect_equal(num(r[[1]][[1]]), 4)
  expect_equal(num(r[[1]][[2]]), 3)
  expect_equal(num(r[[2]]$f$ct_out), 1)
  expect_equal(num(r[[2]]$f$x), 3)
  # A plain call drops the logs.
  expect_equal(lapply(fv(1), num), list(4, 3))
  # Under jit.
  h <- jit(function(x, y, ct) vjp(f, x, y)[[2]]$with_logs(ct))
  expect_equal(num(h(3, 4, 1)[[2]]$f$x), 3)
})

test_that("custom_gradient with_logs checks what the rule returns", {
  skip_if_no_backend()
  e <- custom_gradient(function(x) list(x, function(ct) list(ct)), with_logs = TRUE)
  expect_error(grad(e)(1), "pair", class = "jax_type_error")
  e2 <- custom_gradient(function(x) list(x, function(ct) list(ct, list(1))), with_logs = TRUE)
  expect_error(grad(e2)(1), "named list", class = "jax_type_error")
})

test_that("repeated log keys are an error even when logs are dropped", {
  skip_if_no_backend()
  d <- custom_gradient(function(x) list(x, function(ct) list(ct, list(k = ct))), with_logs = TRUE)
  expect_error(grad(function(x) d(d(x)))(1), "Duplicate backward-pass log key 'k'",
               class = "jax_value_error")
})

test_that("logs from control flow bodies raise a clear error when requested", {
  skip_if_no_backend()
  g <- custom_gradient(function(x) list(x^2, function(ct) list(ct * 2 * x, list(ct = ct))),
                       with_logs = TRUE)
  h <- function(x) lax_scan(function(c, z) list(g(c), NULL), x, NULL, length = 2L)[[1]]
  expect_error(vjp(h, 3)[[2]]$with_logs(1), "not supported", class = "jax_not_implemented_error")
  # Gradients are unaffected.
  expect_equal(num(grad(h)(3)), 108)
  expect_equal(num(grad(checkpoint(g))(3)), 6)
})

test_that("custom_vjp_defvjp_with_logs", {
  skip_if_no_backend()
  f <- custom_vjp(function(x, y) x * y)
  f <- custom_vjp_defvjp_with_logs(f, function(x, y) list(x * y, list(x, y)),
    function(res, g) list(list(g * res[[2]], g * res[[1]]), list(g = g)))
  r <- vjp(f, 3, 4)[[2]]$with_logs(2)
  expect_equal(lapply(r[[1]], num), list(8, 6))
  expect_equal(num(r[[2]]$g), 2)
  expect_equal(num(grad(f)(3, 4)), 4)
})

test_that("custom_vjp_defvjp_with_accums", {
  skip_if_no_backend()
  seen <- character(0)
  f <- custom_vjp(function(c, x) c * x[3L])
  f <- custom_vjp_defvjp_with_accums(f, function(c, x) list(f(c, x), list(c, x)),
    function(res, g, c_acc, x_acc) {
      seen <<- c(seen, c_acc$kind, x_acc$kind)
      c_acc$accum(g * res[[2]][3L])
      x_acc$accum(jnp_where(jnp_arange(10L) == 3L, g * res[[1]], 0))
      list(g = g)
    })
  x <- jnp_arange(10, dtype = "float32")
  expect_equal(num(grad(f, argnums = 1L)(2, x)), c(0, 0, 0, 2, 0, 0, 0, 0, 0, 0))
  expect_equal(seen, c("NullAccum", "ValAccum"))
  seen <- character(0)
  expect_equal(num(grad(f)(2, x)), 3)
  expect_equal(seen, c("ValAccum", "NullAccum"))
  ref <- new_ref(jnp_zeros(10))
  r <- vjp(f, 2, x)[[2]]$with_logs$with_refs(ad_DontWant(), ref)(1)
  expect_s3_class(r[[1]][[1]], "jax_DidntWant")
  expect_s3_class(r[[1]][[2]], "jax_GradRef")
  expect_equal(num(r[[2]]$g), 1)
  expect_equal(num(ref[]), c(0, 0, 0, 2, 0, 0, 0, 0, 0, 0))
})

test_that("custom_gradient with_accums", {
  skip_if_no_backend()
  seen <- character(0)
  x <- jnp_arange(10, dtype = "float32")
  f <- custom_gradient(function(c, x, i) list(c * x[i], function(g, c_acc, x_acc, i_acc) {
    seen <<- c(seen, c_acc$kind, x_acc$kind, i_acc$kind)
    c_acc$accum(g * x[i])
    if (x_acc$kind == "RefAccum") x_acc$ref[i] <- x_acc$ref[i] + g * c
    list(g = g)
  }), with_accums = list(ad_ValAccum, ad_RefAccum, ad_ValAccum))
  expect_equal(num(f(2, x, 3L)), 6)
  ref <- new_ref(jnp_zeros(10))
  r <- vjp(f, 2, x, 3L)[[2]]$with_logs$with_refs(ad_GradValue(), ref, ad_DontWant())(1)
  expect_equal(num(r[[1]][[1]]), 3)
  expect_equal(num(ref[]), c(0, 0, 0, 2, 0, 0, 0, 0, 0, 0))
  expect_equal(num(r[[2]]$g), 1)
  expect_equal(seen, c("ValAccum", "RefAccum", "NullAccum"))
  # A dict argument with mixed kinds.
  gd <- custom_gradient(function(d) list(d$a + 2 * d$b, function(ct, d_accs) {
    d_accs$a$accum(ct)
    d_accs$b$ref[] <- d_accs$b$ref[] + 2 * ct
    NULL
  }), with_accums = list(list(a = ad_ValAccum, b = ad_RefAccum)))
  b_ref <- new_ref(as_jax(0, "float32"))
  r <- vjp(gd, list(a = 1, b = 1))[[2]]$with_refs(list(a = ad_GradValue(), b = b_ref))(1)
  expect_equal(num(r[[1]]$a), 1)
  expect_equal(num(b_ref[]), 2)
  expect_error(custom_gradient(function(x) list(x, NULL), with_accums = 3L)(1),
               "with_accums must be", class = "jax_type_error")
})

test_that("with_accums rules compose with transformations", {
  skip_if_no_backend()
  sinf <- custom_gradient(function(x) list(jnp_sin(x), function(g, x_acc) x_acc$accum(g * jnp_cos(x))),
                          with_accums = ad_ValAccum)
  xs <- jnp_reshape(jnp_arange(6, dtype = "float32"), c(2L, 3L))
  loss <- function(x) jnp_sum(sinf(x))
  want <- num(jnp_cos(xs))
  expect_equal(num(vmap(grad(loss))(xs)), want, tolerance = 1e-6)
  expect_equal(num(jit(grad(loss))(xs)), want, tolerance = 1e-6)
  expect_equal(num(grad(function(xs) jnp_sum(vmap(loss)(xs)))(xs)), want, tolerance = 1e-6)
  mul <- custom_gradient(function(x, y) list(x * y, function(g, x_acc, y_acc) {
    x_acc$accum(g * y); y_acc$accum(g * x); NULL
  }), with_accums = list(ad_ValAccum, ad_ValAccum))
  expect_equal(num(grad(function(x) mul(x, 2))(1)), 2)
  expect_equal(num(grad(function(x) mul(x, jnp_sin(x)))(1)), sin(1) + cos(1), tolerance = 1e-6)
})

test_that("with_refs checks its arguments and accumulates", {
  skip_if_no_backend()
  fv <- vjp(function(x, y) x * y, 3, 4)[[2]]
  expect_error(fv$with_refs(ad_DontWant())(1), "unexpected tree structure",
               class = "jax_value_error")
  expect_error(fv$with_refs(new_ref(jnp_zeros(2)), ad_GradValue())(1), "gradient ref",
               class = "jax_value_error")
  ref <- new_ref(as_jax(1, "float32"))
  r <- fv$with_refs(ad_DontWant(), ref)(1)
  expect_s3_class(r[[1]], "jax_DidntWant")
  expect_s3_class(r[[2]], "jax_GradRef")
  expect_equal(num(ref[]), 4)
})

test_that("custom_vjp_defremat", {
  skip_if_no_backend()
  f <- custom_vjp(function(x) jnp_sin(x))
  f <- custom_vjp_defremat(f, function(x) list(jnp_sin(x), x),
                           function(res, x) list(jnp_sin(res), jnp_cos(res)),
                           function(res, g) list(g * res))
  expect_equal(num(grad(f)(1)), cos(1), tolerance = 1e-6)
  expect_equal(num(grad(checkpoint(f))(1)), cos(1), tolerance = 1e-6)
  h <- custom_vjp(function(x) 2 * x)
  h <- custom_vjp_defremat_with_logs(h, function(x) list(2 * x, NULL),
                                     function(res, x) list(2 * x, x),
                                     function(res, g) list(list(2 * g), list(x = res)))
  r <- vjp(h, 3)[[2]]$with_logs(1)
  expect_equal(num(r[[1]][[1]]), 2)
  expect_equal(num(r[[2]]$x), 3)
})
