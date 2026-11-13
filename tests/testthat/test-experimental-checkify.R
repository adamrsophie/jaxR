# jax.experimental.checkify: messages recorded from JAX by tools/gen_checkify_fixture.py.

ck_ref <- function() {
  skip_if_no_backend()
  testthat::skip_if_not_installed("jsonlite")
  jsonlite::fromJSON(testthat::test_path("fixtures", "checkify.json"), simplifyVector = FALSE)
}

# The message of the error of a checked call against the recorded one (NULL: no error).
ck_expect <- function(ref, name, res) {
  expect_true(name %in% names(ref), info = name)
  expect_identical(res[[1L]]$get(), ref[[name]], info = name)
}

ck_num <- function(x) as.numeric(as.array(x))
checkify <- experimental_checkify_checkify
check <- experimental_checkify_check
fc <- experimental_checkify_float_checks
ic <- experimental_checkify_index_checks

ck_f <- function(x) {
  x <- as_jax(x)
  check(x > 0, "{x} needs to be positive!", x = x)
  1 / x
}
ck_g <- function(x) x + jnp_sin(x)

test_that("user checks are functionalized and raise outside checkify", {
  ref <- ck_ref()
  r <- jit(checkify(ck_f))(-3)
  ck_expect(ref, "user_jit_fail", r)
  expect_equal(ck_num(r[[2L]]), ref$user_jit_val, tolerance = 1e-6)
  ck_expect(ref, "user_jit_ok", jit(checkify(ck_f))(3))
  ck_expect(ref, "user_eager_fail", checkify(ck_f)(-3))
  expect_null(experimental_checkify_init_error$get())
  expect_error(ck_f(as_jax(-1)), ref$eager_raise[[2L]], fixed = TRUE,
               class = experimental_checkify_JaxRuntimeError)
  expect_error(ck_f(as_jax(-1)), class = "jax_value_error")
  expect_error(jit(ck_f)(-1), ref$unfunctionalized[[2L]], fixed = TRUE, class = "jax_value_error")
  expect_error(check(jnp_ones(2L) > 0, "x"), "check takes a scalar pred as argument",
               class = "jax_type_error")
  expect_error(experimental_checkify_debug_check(jnp_ones(2L) > 0, "x"),
               "debug_check takes a scalar pred as argument", class = "jax_type_error")
  expect_error(experimental_checkify_check_error(3), "check_error takes an Error as argument",
               class = "jax_type_error")
  expect_error(check(as_jax(TRUE), "x {}", "str"), "need to be PyTrees of arrays",
               class = "jax_type_error")
  expect_error(checkify(ck_f, errors = "nans"), class = "jax_type_error")
})

test_that("throw, check_error and nested checkify follow JAX", {
  ref <- ck_ref()
  e1 <- jit(checkify(ck_f))(as_jax(-5))[[1L]]
  expect_error(e1$throw(), ref$throw[[2L]], fixed = TRUE, class = experimental_checkify_JaxRuntimeError)
  e2 <- jit(checkify(ck_f))(as_jax(5))[[1L]]
  expect_null(e2$throw())
  expect_identical(format(e1), ref$str_err)
  expect_identical(format(e2), "Error(NULL)")
  expect_s3_class(e1$get_exception(), "jax_checkify_exception")
  expect_null(e2$get_exception())

  outer <- function(x) {
    r <- jit(checkify(ck_f))(x)
    experimental_checkify_check_error(r[[1L]])
    r[[2L]]
  }
  expect_error(outer(as_jax(-1)), ref$check_error_raise[[2L]], fixed = TRUE,
               class = experimental_checkify_JaxRuntimeError)
  ck_expect(ref, "recheckify", checkify(outer)(as_jax(-1)))

  inner <- function(x) {
    check(x > 0, "inner positive {}", x)
    check(x < 10, "inner small {}", x)
    x
  }
  nested <- function(x) {
    check(x != 5, "outer five {}", x)
    r <- jit(checkify(inner))(x)
    experimental_checkify_check_error(r[[1L]])
    check(r[[2L]] != 7, "outer seven {}", r[[2L]])
    r[[2L]]
  }
  cases <- c(nest_five = 5, nest_neg = -1, nest_big = 20, nest_seven = 7, nest_ok = 1)
  for (n in names(cases)) ck_expect(ref, n, jit(checkify(nested))(as_jax(cases[[n]])))

  multi <- function(x) {
    x <- as_jax(x)
    check(x > 0, "first {}", x)
    y <- jnp_log(x)
    check(y > 0, "second {}", y)
    y
  }
  all_checks <- experimental_checkify_all_checks
  ck_expect(ref, "multi_first", jit(checkify(multi, errors = all_checks))(-1))
  ck_expect(ref, "multi_second", jit(checkify(multi, errors = all_checks))(0.5))
  ck_expect(ref, "multi_float_only", jit(checkify(multi, errors = fc))(-1))

  dbg <- function(x) {
    x <- as_jax(x)
    experimental_checkify_debug_check(x != 0, "cannot be zero!")
    x
  }
  expect_equal(ck_num(dbg(0)), ref$debug_plain)
  expect_equal(ck_num(jit(dbg)(0)), 0)
  ck_expect(ref, "debug_checked", jit(checkify(dbg))(0))
})

test_that("float, division and index checks follow JAX", {
  ref <- ck_ref()
  A <- as_jax
  ck_expect(ref, "nan_sin", checkify(ck_g, errors = fc)(A(Inf)))
  ck_expect(ref, "nan_ok", jit(checkify(ck_g, errors = fc))(1))
  div <- function(x, y) x / y
  ck_expect(ref, "div", checkify(div, errors = experimental_checkify_div_checks)(A(1), A(0)))
  ck_expect(ref, "div_nan", checkify(div, errors = fc)(A(0), A(0)))
  ck_expect(ref, "div_nan_only", checkify(div, errors = experimental_checkify_nan_checks)(A(0), A(0)))
  ck_expect(ref, "int_div", checkify(function(x, y) x %/% y,
                                     errors = experimental_checkify_div_checks)(A(3L), A(0L)))
  ck_expect(ref, "rem_nan", checkify(function(x, y) x %% y,
                                     errors = experimental_checkify_all_checks)(A(3), A(0)))
  # Operations whose result is not used are checked too.
  ck_expect(ref, "dead_code", checkify(function(x) list(jnp_log(x), x)[[2L]], errors = fc)(A(-1)))
  # A check that is not enabled leaves no error.
  expect_null(checkify(ck_g)(A(Inf))[[1L]]$get())
  expect_null(checkify(div, errors = ic)(A(1), A(0))[[1L]]$get())

  x3 <- jnp_arange(3, dtype = "float32")
  x5 <- jnp_arange(5, dtype = "float32")
  at <- function(x, i) x[i]
  ck_expect(ref, "gather_oob", checkify(at, errors = ic)(x3, 5L))
  ck_expect(ref, "gather_oob_neg", checkify(at, errors = ic)(x3, -7L))
  ck_expect(ref, "gather_ok", checkify(at, errors = ic)(x3, 2L))
  ck_expect(ref, "gather_oob_2d", checkify(function(x, i) x[i, 1L], errors = ic)(jnp_ones(c(2L, 3L)), 4L))
  ck_expect(ref, "scatter_oob", checkify(function(x, i) at_set(x, i, value = 1), errors = ic)(x3, 5L))
  ck_expect(ref, "scatter_nan",
            checkify(function(x, i) at_add(x, i, value = NaN),
                     errors = experimental_checkify_automatic_checks)(x3, 1L))
  ck_expect(ref, "dyn_slice_oob",
            checkify(function(x, i) lax_dynamic_slice(x, list(i), 2L), errors = ic)(x5, 4L))
  ck_expect(ref, "dyn_update_oob",
            checkify(function(x, i) lax_dynamic_update_slice(x, jnp_ones(2L), list(i)), errors = ic)(x5, 4L))
})

test_that("checks inside cond, scan and while_loop follow JAX", {
  ref <- ck_ref()
  A <- as_jax
  h <- function(x) lax_cond(x > 0, function(x) jnp_sin(x), function(x) jnp_log(x), x)
  ck_expect(ref, "cond_log", checkify(h, errors = fc)(A(-1)))
  ck_expect(ref, "cond_ok", jit(checkify(h, errors = fc))(1))
  ck_expect(ref, "cond_sin", checkify(h, errors = fc)(A(Inf)))
  hc <- function(x) {
    x <- as_jax(x)
    lax_cond(x > 0,
             function(x) { check(x < 10, "too big: {}", x); x },
             function(x) { check(x > -10, "too small: {}", x); -x }, x)
  }
  ck_expect(ref, "cond_user_true", jit(checkify(hc))(20))
  ck_expect(ref, "cond_user_false", jit(checkify(hc))(-20))
  ck_expect(ref, "cond_user_ok", jit(checkify(hc))(2))

  x3 <- jnp_arange(3, dtype = "float32")
  x4 <- jnp_arange(4, dtype = "float32")
  x5 <- jnp_arange(5, dtype = "float32")
  sc <- function(xs) lax_scan(function(c, x) {
    check(x < 3, "x={x} at carry {c}", x = x, c = c)
    list(c + x, jnp_log(x - 1))
  }, 0, xs)
  ck_expect(ref, "scan_user", jit(checkify(sc))(x5))
  ck_expect(ref, "scan_all", jit(checkify(sc, errors = experimental_checkify_all_checks))(x5))
  r <- jit(checkify(sc))(x3)
  ck_expect(ref, "scan_ok", r)
  expect_equal(ck_num(r[[2L]][[1L]]), ref$scan_ok_carry)
  si <- function(xs, idx) lax_scan(function(c, i) list(c + xs[i], xs[i]), 0, idx)
  ck_expect(ref, "scan_index", jit(checkify(si, errors = ic))(x4, A(c(0L, 3L, 7L, 9L))))
  ck_expect(ref, "scan_index_ok", jit(checkify(si, errors = ic))(x4, A(c(0L, 3L))))

  wl <- function(x) {
    x <- as_jax(x)
    lax_while_loop(function(s) s[[1L]] < 5L,
                   function(s) {
                     check(s[[2L]] < 100, "overflow {}", s[[2L]])
                     list(s[[1L]] + 1L, s[[2L]] * 4)
                   }, list(0L, x))
  }
  ck_expect(ref, "while_user", jit(checkify(wl))(1))
  r <- jit(checkify(wl))(0.01)
  ck_expect(ref, "while_ok", r)
  expect_equal(ck_num(r[[2L]][[2L]]), ref$while_ok_val, tolerance = 1e-6)
  wl2 <- function(x) lax_while_loop(function(s) jnp_log(s) < 2, function(s) s - 1, x)
  ck_expect(ref, "while_cond_nan", jit(checkify(wl2, errors = fc))(3))
  ck_expect(ref, "while_cond_first", jit(checkify(wl2, errors = fc))(-3))

  wl3 <- function(x) lax_while_loop(function(s) s < 3, function(s) s + 1, x)
  expect_error(checkify(vmap(wl3), errors = fc)(A(c(0, 1))), ref$checkify_vmap_while[[2L]],
               fixed = TRUE, class = "jax_value_error")
  ck_expect(ref, "vmap_checkify_while", vmap(checkify(wl3, errors = fc))(A(c(0, 1))))
})

test_that("checkify composes with vmap, grad, jvp, checkpoint and custom derivatives", {
  ref <- ck_ref()
  A <- as_jax
  ck_expect(ref, "vmap_user", vmap(checkify(ck_f))(A(c(1, -2, 3, -4))))
  ck_expect(ref, "vmap_nan", vmap(checkify(ck_g, errors = fc))(A(c(1, Inf))))
  ck_expect(ref, "vmap_ok", vmap(checkify(ck_f))(A(c(1, 2))))
  ck_expect(ref, "vmap2_user", vmap(vmap(checkify(ck_f)))(A(matrix(c(1, 3, -2, -4), 2))))
  expect_error(vmap(ck_f)(A(c(1, -2, 3))), ref$vmap_eager_check[[2L]], fixed = TRUE,
               class = experimental_checkify_JaxRuntimeError)
  expect_equal(ck_num(vmap(ck_f)(A(c(1, 2)))), unlist(ref$vmap_eager_ok))

  expect_equal(ck_num(grad(function(x) checkify(ck_g, errors = fc)(x)[[2L]])(1)), ref$grad_checked,
               tolerance = 1e-6)
  expect_equal(ck_num(grad(function(x) checkify(ck_f)(x)[[2L]])(A(2))), ref$grad_user, tolerance = 1e-6)
  ck_expect(ref, "checkify_of_grad", checkify(grad(function(x) jnp_log(x) * x), errors = fc)(A(0)))

  ck_expect(ref, "remat_nan", checkify(checkpoint(function(x) jnp_log(x) * 2), errors = fc)(A(-1)))
  ck_expect(ref, "remat_user", jit(checkify(checkpoint(ck_f)))(A(-2)))

  ck_expect(ref, "custom_jvp_nan", checkify(function(x) nn_relu(jnp_log(x)), errors = fc)(A(-1)))
  ck_expect(ref, "custom_jvp_inner",
            checkify(function(x) scipy_special_logsumexp(jnp_log(x)), errors = fc)(A(c(-1, 2))))
  cj <- custom_jvp(function(x) jnp_log(x),
                   jvp = function(p, t) list(jnp_log(p[[1L]]), t[[1L]] / p[[1L]]))
  ck_expect(ref, "custom_jvp_fun", checkify(cj, errors = fc)(A(-1)))
  ck_expect(ref, "custom_jvp_under_jvp",
            checkify(function(x) jvp(cj, list(x), list(jnp_ones_like(x)))[[2L]], errors = fc)(A(0)))
  r <- jvp(function(x) checkify(cj, errors = fc)(x)[[2L]], list(A(2)), list(A(1)))
  expect_equal(c(ck_num(r[[1L]]), ck_num(r[[2L]])), unlist(ref$jvp_of_checkify), tolerance = 1e-6)

  cv <- custom_vjp(function(x) jnp_log(x), fwd = function(x) list(jnp_log(x), x),
                   bwd = function(res, g) list(g / res))
  ck_expect(ref, "custom_vjp_nan", checkify(cv, errors = fc)(A(-1)))
  r <- checkify(grad(cv), errors = fc)(A(0))
  ck_expect(ref, "custom_vjp_grad", r)
  expect_identical(ck_num(r[[2L]]), Inf)
  r <- checkify(grad(cv), errors = fc)(A(2))
  ck_expect(ref, "custom_vjp_grad_ok", r)
  expect_equal(ck_num(r[[2L]]), ref$custom_vjp_grad_ok_val)
})

test_that("messages are formatted like Python and results keep their structure", {
  ref <- ck_ref()
  A <- as_jax
  x3 <- jnp_arange(3, dtype = "float32")
  fmt <- function(x) {
    check(jnp_sum(x) > 100, "sum {:.2f} of {} and {v}", jnp_sum(x), x, v = x[0L])
    x
  }
  ck_expect(ref, "fmt", checkify(fmt)(x3))
  fm <- function(x) {
    xi <- jnp_astype(x, "int32")
    check(jnp_sum(x) < 0, "{} {} {} {:d} {:5.1f}|{:e}", x > 1, xi, x[0L] > 5, xi[1L], x[2L], x[1L])
    x
  }
  ck_expect(ref, "fmt_types", checkify(fm)(A(c(1.5, 2, 2.5))))
  # Scalars are formatted as Python floats (the shortest decimal of the double).
  fm2 <- function(x) {
    check(jnp_sum(x) < 0, "{} {} {{x}}", x[0L], x[1L])
    x
  }
  expect_identical(checkify(fm2)(A(c(1e-7, 123456.789, 1e20, 0.1)))[[1L]]$get(),
                   "1.0000000116860974e-07 123456.7890625 {x} (`check` failed)")
  expect_error(checkify(function(x) { check(x > 0, "{} {}", x); x })(A(-1))[[1L]]$get(),
               class = "jax_index_error")
  expect_error(checkify(function(x) { check(x > 0, "{y}", x = x); x })(A(-1))[[1L]]$get(),
               class = "jax_key_error")

  tr <- function(x) {
    check(x$a > 0, "a={a} b={b}", a = x$a, b = x$b)
    list(s = x$a + 1L, t = list(x$b, x$a))
  }
  r <- jit(checkify(tr))(list(a = A(-1L), b = A(c(1L, 2L))))
  ck_expect(ref, "tree_int", r)
  expect_identical(names(r[[2L]]), c("s", "t"))
  expect_equal(ck_num(r[[2L]]$s), ref$tree_val)
})

test_that("an unfunctionalized check can raise from compiled code", {
  ref <- ck_ref()
  old <- jax_config_update("jax_experimental_unsafe_xla_runtime_errors", TRUE)
  on.exit(jax_config_update("jax_experimental_unsafe_xla_runtime_errors", old))
  expect_error(jit(ck_f)(as_jax(-1)), "-1.0 needs to be positive! (`check` failed)", fixed = TRUE)
  expect_equal(ck_num(jit(ck_f)(as_jax(2))), ref$xla_runtime_ok)
})
