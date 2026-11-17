# Exact values across lowering and transfers, callback result checks, cache and
# registry lifetimes, argument binding of the AOT stages, sparse derivatives with
# unused capacity, and numerically stable formulas.

vals <- function(x) as.vector(as.array(x))

x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)

test_that("int64 constants of custom derivative calls keep values beyond 2^53 under jit", {
  x64({
    base <- as_jax(2^53, dtype = "int64")
    w <- base + as_jax(1, dtype = "int64")
    z <- as_jax(0, dtype = "int64")
    fs <- list(
      custom_jvp(function(x) x + w, function(p, t) list(p[[1L]] + w, t[[1L]])),
      custom_vjp(function(x) x + w, function(x) list(x + w, NULL), function(r, ct) list(ct)))
    for (f in fs) {
      expect_equal(vals(f(z) - base), 1, tolerance = 0)
      expect_equal(vals(jit(f)(z) - base), 1, tolerance = 0)
    }
  })
})

test_that("negative int64 dense constants keep their bits", {
  x64({
    for (v in list(c(-1, -2), c(-1024, -1025), c(-2147483647, 2147483647), c(-2^52, 3))) {
      w <- as_jax(v, dtype = "int64")
      f <- custom_jvp(function(x) x + w, function(p, t) list(p[[1L]] + w, t[[1L]]))
      expect_equal(vals(jit(f)(jnp_zeros(2L, dtype = "int64"))), v, tolerance = 0)
      # Literal constants of the module itself.
      expect_equal(vals(jit(function(x) x + v)(jnp_zeros(2L, dtype = "int64"))), v, tolerance = 0)
    }
  })
})

test_that("compiled callbacks check their results as eager ones do", {
  for (case in list(list(256, "uint8"), list(-1, "uint8"), list(NaN, "int32"),
                    list(Inf, "uint8"), list(NA_real_, "float32"), list(NA, "bool"))) {
    value <- case[[1L]]
    f <- function(x) pure_callback(function(v) value, ShapeDtypeStruct(integer(0), case[[2L]]), x)
    expect_error(f(as_jax(1)))
    expect_error(jit(f)(as_jax(1)), "out of bounds|cannot convert|NA values")
  }
  f <- function(x) pure_callback(function(v) 255, ShapeDtypeStruct(integer(0), "uint8"), x)
  expect_equal(vals(jit(f)(as_jax(1))), 255)
})

test_that("float16 and bfloat16 transfers round once from double", {
  # Above the midpoint of 1 and 1 + 2^-10 by less than float32 resolves.
  v <- 1 + 2^-11 + 2^-25
  expect_identical(vals(as_jax(v, "float16")), 1 + 2^-10)
  expect_identical(vals(as_jax(1 + 2^-11, "float16")), 1)
  expect_identical(vals(as_jax(1 + 2^-8 + 2^-30, "bfloat16")), 1 + 2^-7)
  expect_identical(vals(as_jax(c(65504, 70000, -Inf, 2^-24, 2^-26), "float16")),
                   c(65504, Inf, -Inf, 2^-24, 0))
  expect_true(is.nan(vals(as_jax(NaN, "float16"))))
})

test_that("block_until_ready waits without copying to the host", {
  f <- jit(function(x) x + 1)
  y <- f(as_jax(c(1, 2)))
  expect_null(y$host)
  transfer_guard_device_to_host("disallow", block_until_ready(list(y, NULL, 3)))
  expect_null(y$host)
  expect_equal(vals(y), c(2, 3))
})

test_that("clear_caches clears the caches of live jit functions", {
  traces <- 0L
  f <- jit(function(x) {
    traces <<- traces + 1L
    x + x
  })
  x <- as_jax(c(1, 2, 3))
  f(x); f(x)
  expect_identical(traces, 1L)
  clear_caches()
  expect_equal(vals(f(x)), c(2, 4, 6))
  expect_identical(traces, 2L)
})

test_that("callback registry entries are released with their executables", {
  live <- function() sum(!vapply(jaxR:::.jx$cb_registry, is.null, NA))
  clear_caches()
  before <- live()
  local({
    for (i in 1:3) {
      cb <- local({ k <- i; function(v) v + k })
      f <- jit(function(x) pure_callback(cb, ShapeDtypeStruct(integer(0), "float32"), x))
      expect_equal(vals(f(as_jax(1, "float32"))), 1 + i)
    }
  })
  expect_identical(live(), before + 3L)
  clear_caches()
  expect_identical(live(), before)
  # A retained function retraces and registers again after its entry was released.
  cb <- function(v) v * 2
  g <- jit(function(x) pure_callback(cb, ShapeDtypeStruct(integer(0), "float32"), x))
  expect_equal(vals(g(as_jax(2, "float32"))), 4)
  clear_caches()
  expect_equal(vals(g(as_jax(3, "float32"))), 6)
})

test_that("AOT stages and eval_shape bind static arguments as calls do", {
  x64({
    f <- jit(function(x, n) x * n, static_argnames = "n")
    compiled <- f$lower(n = 3L, x = as_jax(c(1, 2, 3)))$compile()
    expect_equal(vals(compiled(x = as_jax(c(10, 20, 30)))), c(30, 60, 90))
    expect_length(f$lower(n = 3L, x = as_jax(c(1, 2, 3)))$in_avals, 1L)
    g <- jit(function(x, n) x + jnp_zeros(n), static_argnames = "n")
    info <- g$eval_shape(as_jax(c(1, 2, 3)), 3L)
    expect_identical(info$shape, 3L)
    expect_identical(g$eval_shape(n = 3L, x = as_jax(c(1, 2, 3)))$shape, 3L)
    h <- jit(function(n, x) x + n, static_argnums = 0L)
    expect_identical(h$eval_shape(2, as_jax(1:4))$shape, 4L)
  })
})

test_that("subclasses of a registered pytree class flatten with its handler", {
  register_pytree_node("jaxr_test_tree_base",
    flatten = function(x) list(children = list(x$value), aux = x$label),
    unflatten = function(aux, children)
      structure(list(value = children[[1L]], label = aux), class = "jaxr_test_tree_base"))
  node <- structure(list(value = c(2, 3), label = "a"), class = c("jaxr_test_tree_sub", "jaxr_test_tree_base"))
  fl <- tree_flatten(node)
  r <- tree_unflatten(fl$treedef, fl$leaves)
  expect_s3_class(r, "jaxr_test_tree_base")
  expect_identical(r$value, c(2, 3))
  expect_identical(r$label, "a")
  m <- tree_map_with_path(function(p, l) l + 1, node)
  expect_identical(m$value, c(3, 4))
})

test_that("tree_flatten keeps order and NULL leaves of large trees", {
  t <- as.list(seq_len(5000))
  t[[17]] <- list(a = 1, b = NULL)
  fl <- tree_flatten(t, is_leaf = is.null)
  expect_length(fl$leaves, 5001L)
  expect_null(fl$leaves[[18]])
  expect_identical(tree_unflatten(fl$treedef, fl$leaves), t)
})

test_that("fori_loop keeps exact int64 bounds of static loops", {
  x64({
    lo <- as_jax(2147483648, dtype = "int64")
    hi <- as_jax(2147483651, dtype = "int64")
    expect_equal(vals(lax_fori_loop(lo, hi, function(i, v) v + 1, as_jax(0))), 3)
    expect_equal(vals(lax_fori_loop(lo, hi, function(i, v) i, as_jax(0, dtype = "int64")) - lo), 2)
    expect_equal(vals(lax_fori_loop(hi, lo, function(i, v) v + 1, as_jax(0))), 0)
    neg <- as_jax(-2^40, dtype = "int64")
    expect_equal(vals(lax_fori_loop(neg, neg + 2L, function(i, v) v + 1, as_jax(0))), 2)
  })
})

test_that("vmap rejects output axes outside the rank of the result", {
  x <- as_jax(c(1, 2, 3))
  expect_error(vmap(identity, out_axes = 99L)(x), class = "jax_value_error")
  expect_error(vmap(identity, out_axes = -2L)(x), class = "jax_value_error")
  expect_error(vmap(function(v) 1, out_axes = 1L, axis_size = 3L)(x), class = "jax_value_error")
  expect_equal(vals(vmap(identity, out_axes = -1L)(x)), c(1, 2, 3))
  m <- vmap(function(v) jnp_stack(list(v, v)), out_axes = -1L)(x)
  expect_identical(dim(as.array(m)), c(2L, 3L))
})

test_that("a concrete array closed over several times is one constant", {
  a <- as_jax(c(1, 2, 3))
  cj <- make_jaxpr(function(x) x * a + x * a + x * a)(as_jax(c(1, 1, 1)))
  expect_length(cj$jaxpr$constvars, 1L)
})

test_that("padded CSR solves have the gradients of the matrix they hold", {
  x64({
    m <- experimental_sparse_CSR$fromdense(as_jax(diag(c(2, 3))), nse = 3L)
    solve_b <- function(b) experimental_sparse_linalg_spsolve(m$data, m$indices, m$indptr, b)
    b <- as_jax(c(4, 9))
    expect_equal(vals(grad(function(b) jnp_sum(solve_b(b)))(b)), c(0.5, 1 / 3))
    expect_equal(vals(jit(grad(function(b) jnp_sum(solve_b(b))))(b)), c(0.5, 1 / 3))
    A <- matrix(c(4, 1, 0, 2, 5, 0, 0, 1, 3), 3)
    s <- experimental_sparse_CSR$fromdense(as_jax(A), nse = 8L)
    g <- grad(function(b) jnp_sum(experimental_sparse_linalg_spsolve(s$data, s$indices, s$indptr, b)))
    expect_equal(vals(g(as_jax(c(1, 2, 3)))), solve(t(A), rep(1, 3)))
  })
})

test_that("COO and CSR fromdense and todense derivatives ignore unused capacity", {
  x64({
    C <- as_jax(matrix(c(2, 3, 0, 0), 2L))
    dC <- as_jax(matrix(c(7, 11, 0, 0), 2L))
    fcsr <- function(a) experimental_sparse_CSR$fromdense(a, nse = 3L)$data
    fcoo <- function(a) experimental_sparse_COO$fromdense(a, nse = 3L)$data
    expect_equal(vals(jvp(fcsr, list(C), list(dC))[[2L]]), c(7, 11, 0))
    expect_equal(vals(jvp(fcoo, list(C), list(dC))[[2L]]), c(7, 11, 0))
    expect_equal(as.array(grad(function(a) jnp_sum(fcoo(a)))(C)), matrix(c(1, 1, 0, 0), 2L))
    expect_equal(as.array(grad(function(a) jnp_sum(fcsr(a)))(C)), matrix(c(1, 1, 0, 0), 2L))
    cs <- experimental_sparse_CSR$fromdense(as_jax(diag(c(2, 3))), nse = 3L)
    td <- function(d) jnp_sum(experimental_sparse_CSR(list(d, cs$indices, cs$indptr), shape = c(2L, 2L))$todense())
    expect_equal(vals(grad(td)(cs$data)), c(1, 1, 0))
  })
})

test_that("line_search accepts a reversed bracket that satisfies the Wolfe conditions", {
  x64({
    r <- scipy_optimize_line_search(function(x) jnp_sum((x - 0.51)^2), as_jax(array(0, 1L)),
                                    as_jax(array(1, 1L)))
    expect_false(vals(r$failed))
    expect_equal(vals(r$a_k), 0.51)
    expect_equal(vals(r$status), 0)
  })
})

test_that("BFGS reaches the minimum of a quadratic", {
  x64({
    A <- crossprod(matrix(c(3, 1, 0.5, 1, 2, 0.3, 0.5, 0.3, 1), 3)) + diag(3)
    b <- c(1, -2, 0.5)
    f <- function(x) 0.5 * jnp_dot(x, jnp_matmul(as_jax(A), x)) - jnp_dot(as_jax(b), x)
    r <- scipy_optimize_minimize(f, as_jax(c(0, 0, 0)), method = "BFGS")
    expect_equal(vals(r$x), solve(A, b), tolerance = 1e-6)
    H <- as.array(r$hess_inv)
    expect_equal(H, t(H), tolerance = 1e-10)
  })
})

test_that("norm logpdf stays finite for extreme scales", {
  x64({
    expect_equal(vals(scipy_stats_norm_logpdf(0, scale = 1e200)), dnorm(0, sd = 1e200, log = TRUE))
    expect_equal(vals(scipy_stats_norm_logpdf(0, scale = 1e-200)), dnorm(0, sd = 1e-200, log = TRUE))
    expect_equal(vals(scipy_stats_norm_logpdf(c(-1, 0, 2.5), 0.5, 2)), dnorm(c(-1, 0, 2.5), 0.5, 2, log = TRUE))
    g <- grad(function(s) scipy_stats_norm_logpdf(1.5, 0.2, s))(as_jax(0.7))
    expect_equal(vals(g), -1 / 0.7 + 1.3^2 / 0.7^3)
  })
  expect_equal(vals(scipy_stats_norm_logpdf(as_jax(0, "float32"), scale = as_jax(1e20, "float32"))),
               dnorm(0, sd = 1e20, log = TRUE), tolerance = 1e-6)
})

test_that("solve_sylvester checks the eigenvalues of its factorizations", {
  x64({
    A <- matrix(c(4, 1, 0, 3), 2); B <- matrix(c(2, 0, 1, 5), 2); C <- matrix(1:4, 2)
    for (method in c("schur", "eigen")) {
      X <- as.array(scipy_linalg_solve_sylvester(A, B, C, method = method))
      expect_equal(A %*% X + X %*% B, C, tolerance = 1e-10)
    }
    # A + B singular (eigenvalues 1 and -1): NaN.
    expect_true(all(is.nan(as.array(scipy_linalg_solve_sylvester(diag(c(1, 2)), diag(c(-1, 3)), C)))))
  })
})
