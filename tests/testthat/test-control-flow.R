# Structured control flow, custom derivatives and checkpointing.

x64 <- function() { old <- jaxR::jax_config_update("jax_enable_x64", TRUE); old }

test_that("cond: values, jit, grad, jvp, vmap with batched predicate", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(1, 2, 3), "float64")
  f <- function(p, v) lax_cond(p > 0, function(v) sin(v) * 2, function(v) cos(v), v)
  expect_jax_equal(f(1, x), sin(1:3) * 2)
  expect_jax_equal(f(-1, x), cos(1:3))
  expect_jax_equal(jit(f)(1, x), sin(1:3) * 2)
  expect_jax_equal(grad(function(v) sum(f(1, v)))(x), 2 * cos(1:3))
  expect_jax_equal(grad(function(v) sum(f(-1, v)))(x), -sin(1:3))
  expect_jax_equal(jvp(function(v) f(1, v), list(x), list(jnp_ones_like(x)))[[2]], 2 * cos(1:3))
  p <- as_jax(c(1, -1, 1), "float64")
  expect_jax_equal(vmap(f)(p, x), c(2 * sin(1), cos(2), 2 * sin(3)))
  expect_jax_equal(grad(function(v) sum(vmap(f)(p, v)))(x), c(2 * cos(1), -sin(2), 2 * cos(3)))
  expect_jax_equal(hessian(function(v) sum(f(1, v)))(x), diag(-2 * sin(1:3)))
  expect_jax_equal(grad(function(z) lax_cond(TRUE, identity, function(y) -y, z))(as_jax(2, "float64")), 1)
})

test_that("switch clamps the index", {
  skip_if_no_backend()
  br <- list(function(v) v, function(v) v * 2, function(v) v * 3)
  expect_jax_equal(lax_switch(2L, br, as_jax(c(1, 2))), c(3, 6))
  expect_jax_equal(lax_switch(7L, br, as_jax(c(1, 2))), c(3, 6))
  expect_jax_equal(lax_switch(-3L, br, as_jax(c(1, 2))), c(1, 2))
})

test_that("while_loop: values, jit, jvp, vmap with batched predicate, reverse mode error", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(1, 2, 3), "float64")
  wl <- function(v) lax_while_loop(function(c) c[[1]] < 5L,
                                   function(c) list(c[[1]] + 1L, c[[2]] * 1.5), list(0L, v))[[2]]
  expect_jax_equal(wl(x), 1.5^5 * (1:3))
  expect_jax_equal(jit(wl)(x), 1.5^5 * (1:3))
  expect_jax_equal(jvp(wl, list(x), list(jnp_ones_like(x)))[[2]], rep(1.5^5, 3))
  n <- as_jax(array(c(1L, 2L, 3L), 3))
  expect_jax_equal(vmap(function(k, v) lax_while_loop(function(c) c[[1]] < k,
    function(c) list(c[[1]] + 1L, c[[2]] * 2), list(0L, v))[[2]])(n, x), c(2, 8, 24))
  expect_error(grad(function(v) sum(wl(v)))(x), "Reverse-mode differentiation does not work")
})

test_that("fori_loop with static bounds is reverse differentiable, also under jit", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(1, 2, 3), "float64")
  fl <- function(v) lax_fori_loop(0L, 5L, function(i, c) c * 1.5 + i, v)
  expect_jax_equal(grad(function(v) sum(fl(v)))(x), rep(1.5^5, 3))
  expect_jax_equal(jit(function(z) lax_fori_loop(0L, 3L, function(i, v) v + 1, z))(as_jax(1, "float64")), 4)
})

test_that("scan: forward, reverse, jit, vmap, grad and second order", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(1, 2, 3), "float64")
  sc <- function(v) lax_scan(function(c, xi) list(c * 0.5 + xi, c * xi), 0, v)
  r <- sc(x)
  expect_jax_equal(r[[1]], 4.25)
  expect_jax_equal(r[[2]], c(0, 2, 7.5))
  expect_jax_equal(jit(function(v) sc(v)[[2]])(x), c(0, 2, 7.5))
  expect_jax_equal(lax_scan(function(c, xi) list(c + xi, c), 0, x, reverse = TRUE)[[2]], c(5, 3, 0))
  expect_jax_equal(vmap(function(v) sc(v)[[1]])(jnp_stack(list(x, x * 2))), c(4.25, 8.5))
  expect_jax_equal(grad(function(v) sum(sc(v)[[2]]) + sc(v)[[1]])(x), c(3.75, 4.5, 3.5))
  expect_jax_equal(grad(function(s) sum(grad(function(v) sum(sc(v)[[2]]))(x * s)))(as_jax(1, "float64")), 10)
  expect_jax_equal(lax_scan(function(c, xi) list(c + xi, c), 0, jnp_zeros(0L, "float64"))[[1]], 0)
  expect_jax_equal(lax_map(function(v) v * 2, x), c(2, 4, 6))
})

test_that("custom_linear_solve: value and implicit gradient", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  M <- matrix(c(4, 1, 1, 3), 2)
  A <- as_jax(M, "float64")
  Ainv <- as_jax(solve(M), "float64")
  b <- as_jax(c(1, 2), "float64")
  s <- function(bb) lax_custom_linear_solve(function(v) A %*% v, bb,
                                            function(m, z) jnp_matmul(Ainv, z), symmetric = TRUE)
  expect_jax_equal(s(b), solve(M, c(1, 2)))
  expect_jax_equal(grad(function(bb) sum(s(bb)))(b), colSums(solve(M)))
})

test_that("custom_vjp gate: grad 6, grad(grad) 3, jacfwd(jacrev) 3, jvp errors", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  f <- custom_vjp(function(x) x * x, fwd = function(x) list(x * x, x),
                  bwd = function(res, g) list(3 * res * g))
  x <- as_jax(2, "float64")
  expect_jax_equal(grad(f)(x), 6)
  expect_jax_equal(grad(grad(f))(x), 3)
  expect_jax_equal(jacfwd(jacrev(f))(x), 3)
  expect_jax_equal(grad(jit(f))(x), 6)
  expect_jax_equal(jit(grad(f))(x), 6)
  expect_error(jvp(f, list(x), list(1)), "forward-mode autodiff")
  expect_jax_equal(vmap(grad(f))(as_jax(c(1, 2, 3), "float64")), c(3, 6, 9))
  fv <- custom_vjp(function(a, b) b, fwd = function(a, b) list(b, NULL),
                   bwd = function(res, ct) list(NULL, ct))
  g <- grad(function(a, b) fv(a, b), argnums = c(0L, 1L))(as_jax(1, "float64"), as_jax(2, "float64"))
  expect_jax_equal(g[[1]], 0)
  expect_jax_equal(g[[2]], 1)
})

test_that("custom_jvp rule is used under jit and vmap", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  g <- custom_jvp(function(x) sin(x), jvp = function(p, t) list(sin(p[[1]]), 10 * t[[1]]))
  x <- as_jax(2, "float64")
  expect_jax_equal(grad(g)(x), 10)
  expect_jax_equal(grad(jit(g))(x), 10)
  expect_jax_equal(vmap(grad(g))(as_jax(c(1, 2), "float64")), c(10, 10))
})

test_that("checkpoint recomputes and matches plain gradients", {
  skip_if_no_backend()
  old <- x64(); on.exit(jax_config_update("jax_enable_x64", old))
  x <- as_jax(c(0.5, 1, 2), "float64")
  f <- function(v) sin(sin(v)) * exp(v)
  cf <- checkpoint(f)
  expect_jax_equal(grad(function(v) sum(cf(v)))(x), as.array(grad(function(v) sum(f(v)))(x)))
  expect_jax_equal(jit(grad(function(v) sum(cf(v))))(x), as.array(grad(function(v) sum(f(v)))(x)))
  inner <- function(v) lax_scan(function(c, xi) list(c * xi, c), 1, v)[[1]]
  expect_jax_equal(grad(checkpoint(inner))(x), as.array(grad(inner)(x)))
  pol <- checkpoint(function(v) { a <- checkpoint_name(sin(v), "a"); cos(a) * a },
                    policy = checkpoint_policies$save_only_these_names("a"))
  v <- c(0.5, 1, 2)
  expect_jax_equal(grad(function(z) sum(pol(z)))(x),
                   -sin(sin(v)) * cos(v) * sin(v) + cos(sin(v)) * cos(v))
  jx <- make_jaxpr(function(v) vjp(cf, v)[[2]](jnp_ones_like(v)))(x)
  expect_true(any(vapply(jx$jaxpr$eqns, function(e) e$prim$name, "") == "checkpoint"))
})
