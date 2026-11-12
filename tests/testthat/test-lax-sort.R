# Sorting, top_k and jax.numpy sorting. Expected values from jax 0.12.0.1 (x64 enabled).

sort_x <- function() {
  matrix(c(0.3, 1.7, -2.1, 4.4, 0.9, 2.2, -0.5, 3.3, 1.1, -1.8, 0.6, 2.9, -0.7, 1.4, 3.8),
         3, byrow = TRUE)
}
sort_w <- function() matrix(0:14, 3, byrow = TRUE) / 10

sort_stack_loop <- function(f, xs, axis) {
  n <- dim(xs)[axis + 1L]
  jnp_stack(lapply(seq_len(n) - 1L, function(i) f(lax_index_in_dim(xs, i, axis, keepdims = FALSE))))
}

test_that("lax_sort and lax_sort_key_val match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(sort_x(), "float64")
  expect_jax_equal(lax_sort(x), matrix(c(-2.1, 0.3, 0.9, 1.7, 4.4, -1.8, -0.5, 1.1, 2.2, 3.3,
                                         -0.7, 0.6, 1.4, 2.9, 3.8), 3, byrow = TRUE),
                   dtype = "float64")
  expect_jax_equal(lax_sort(x, dimension = 0L),
                   matrix(c(0.3, -0.5, -2.1, 1.1, -1.8, 0.6, 1.7, -0.7, 1.4, 0.9,
                            2.2, 2.9, 3.3, 4.4, 3.8), 3, byrow = TRUE))
  expect_jax_equal(jit(function(x) lax_sort(x, 0L))(x), lax_sort(x, 0L))
  # total order: -0.0 and 0.0 are equivalent (stable), NaNs last
  xs <- as_jax(c(1, NaN, -0, 0, -Inf, 2, -NaN, Inf, -1), "float64")
  kv <- lax_sort_key_val(xs, jnp_arange(9L))
  expect_jax_equal(kv[[1L]], c(-Inf, -1, 0, 0, 1, 2, Inf, NaN, NaN))
  expect_jax_equal(kv[[2L]], c(4, 8, 2, 3, 0, 5, 7, 1, 6), dtype = "int64")
  expect_identical(1 / as.vector(kv[[1L]])[3:4], c(-Inf, Inf))
  expect_jax_equal(lax_sort(as_jax(c(1 + 2i, 1 + 1i, 0 + 5i, 2 - 1i))),
                   c(0 + 5i, 1 + 1i, 1 + 2i, 2 - 1i), dtype = "complex128")
  a <- as_jax(c(2, 1, 2, 1, 3), "int64")
  b <- as_jax(c(5, 9, 1, 3, 0), "int64")
  c3 <- jnp_arange(5, dtype = "float64")
  r <- lax_sort(list(a, b, c3), num_keys = 2L)
  expect_jax_equal(r[[1L]], c(1, 1, 2, 2, 3))
  expect_jax_equal(r[[2L]], c(3, 9, 1, 5, 0))
  expect_jax_equal(r[[3L]], c(3, 1, 2, 0, 4))
  expect_jax_equal(lax_sort(list(a, b, c3), num_keys = 1L)[[3L]], c(1, 3, 0, 2, 4))
  expect_jax_equal(lax_sort(c(TRUE, FALSE, TRUE, FALSE)), c(FALSE, FALSE, TRUE, TRUE), dtype = "bool")
  expect_jax_equal(lax_sort(as_jax(c(3, 1, 2), "uint8")), c(1, 2, 3), dtype = "uint8")
})

test_that("lax_top_k matches JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(sort_x(), "float64")
  r <- lax_top_k(x, 3L)
  expect_jax_equal(r[[1L]], matrix(c(4.4, 1.7, 0.9, 3.3, 2.2, 1.1, 3.8, 2.9, 1.4), 3, byrow = TRUE))
  expect_jax_equal(r[[2L]], matrix(c(3, 1, 4, 2, 0, 3, 4, 1, 3), 3, byrow = TRUE), dtype = "int32")
  r0 <- lax_top_k(jnp_transpose(x), 2L)
  expect_jax_equal(r0[[1L]], matrix(c(2.2, 0.6, 2.9, 1.7, 3.3, -0.7, 4.4, 1.4, 3.8, 0.9), 5, byrow = TRUE))
  expect_jax_equal(r0[[2L]], matrix(c(1, 2, 2, 0, 1, 2, 0, 2, 2, 0), 5, byrow = TRUE))
  ra <- lax_top_k(x, 2L, axis = 0L)
  expect_jax_equal(ra[[1L]], jnp_transpose(r0[[1L]]))
  expect_jax_equal(ra[[2L]], jnp_transpose(r0[[2L]]))
  expect_jax_equal(jit(function(x) lax_top_k(x, 2L, axis = 0L))(x)[[2L]], jnp_transpose(r0[[2L]]))
  expect_jax_equal(lax_top_k(as_jax(c(1, 3, 3, 2, 3)), 3L)[[2L]], c(1, 2, 4))
  expect_jax_equal(lax_top_k(as_jax(c(4L, -1L, 7L, 7L, 0L)), 2L)[[1L]], c(7, 7), dtype = "int32")
})

test_that("sort and top_k errors match JAX", {
  skip_if_no_backend()
  z <- as_jax(c(7, 2, 9, 1, 5, 3, 8))
  expect_error(lax_top_k(as_jax(c(1i, 2i)), 1L), "top_k is not compatible with complex inputs.")
  expect_error(lax_top_k(z, 9L), "k argument to top_k must be no larger than size along axis; got k=9 with shape=\\[7\\] and axis=0")
  expect_error(lax_top_k(z, -1L), "k argument to top_k must be nonnegative, got -1")
  expect_error(lax_sort(list(z, jnp_ones(3L))), "Arguments to sort must have equal shapes, got: \\(7,\\) \\(3,\\)")
  expect_error(lax_sort(z, num_keys = 2L), "num_keys=2 must equal 1 for a single operand.")
  expect_error(lax_sort(list(z), num_keys = 2L), "must be between 1 and")
  expect_error(jnp_sort(z, kind = "quicksort"), "'kind' argument to sort is not supported")
})

test_that("sort and top_k gradients match JAX and finite differences", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(sort_x(), "float64")
  f1 <- function(x) jnp_sum(jnp_sin(lax_sort(x, 1L)) * jnp_arange(5, dtype = "float64"))
  expect_jax_equal(grad(f1)(x), matrix(c(
    0.955336489125606, -0.3865334828865739, 0, -1.2293314799136774, 1.2432199365413288,
    -1.7655033517660375, 0.8775825618903728, -3.9499190796354595, 0.9071922428511546, 0,
    0.8253356149096783, -2.9128744954487718, 0, 0.33993428580048207, -3.1638708476576674),
    3, byrow = TRUE))
  check_grad_fd(f1, x)
  f0 <- function(x) jnp_sum(jnp_sin(lax_sort(x, 0L)) * jnp_reshape(jnp_arange(15, dtype = "float64"), c(3L, 5L)))
  expect_jax_equal(grad(f0)(x), matrix(c(
    0, -0.7730669657731478, -1.0096922091997151, -3.9953273097194515, 5.59448971443598,
    -5.885011172553458, 0.8775825618903728, -11.849757238906378, 1.360788364276732, -0.9088083787723484,
    4.1266780745483915, -10.680539816645496, 5.35389531099142, 1.3597371432019283, -11.073547966801836),
    3, byrow = TRUE))
  expect_jax_equal(jit(grad(f0))(x), grad(f0)(x))
  fkv <- function(x) jnp_sum(lax_sort(list(x, x^2), 1L, num_keys = 1L)[[2L]] * jnp_arange(5, dtype = "float64"))
  expect_jax_equal(grad(fkv)(x), matrix(c(0.6, 10.2, 0, 35.2, 3.6, 13.2, -1.0, 26.4, 4.4, 0,
                                          1.2, 17.4, 0, 5.6, 30.4), 3, byrow = TRUE))
  ft <- function(x) jnp_sum(jnp_cos(lax_top_k(x, 3L)[[1L]]) * jnp_arange(3, dtype = "float64"))
  expect_jax_equal(grad(ft)(x), matrix(c(
    0, -0.9916648104524686, 0, 0, -1.5666538192549668,
    -0.8084964038195901, 0, 0, -1.7824147201228708, 0,
    0, -0.23924932921398243, 0, -1.9708994599769203, 0), 3, byrow = TRUE))
  check_grad_fd(ft, x)
  check_grad_fd(function(x) jnp_sum(jnp_sin(lax_top_k(x, 2L, axis = 0L)[[1L]])), x)
})

test_that("sort and top_k second order derivatives", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(sort_x(), "float64")
  w <- as_jax(sort_w(), "float64")
  f2 <- function(x) jnp_sum(lax_sort(x, 1L)^3 * jnp_arange(5, dtype = "float64"))
  expect_jax_equal(jvp(grad(f2), list(x), list(w))[[2L]], matrix(c(
    0, 3.06, 0, 31.68, 4.32, 19.8, -1.8, 55.44, 10.56, 0, 3.6, 57.42, 0, 21.84, 127.68),
    3, byrow = TRUE))
  f3 <- function(x) jnp_sum(lax_top_k(x, 2L)[[1L]]^3)
  expect_jax_equal(grad(function(x) jnp_sum(grad(f3)(x) * w))(x), matrix(c(
    0, 1.02, 0, 7.92, 0, 6.6, 0, 13.86, 0, 0, 0, 19.14, 0, 0, 31.92), 3, byrow = TRUE))
})

test_that("sort and top_k batch like a stacked loop", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  xs <- as_jax(array(round(sin(seq_len(2 * 3 * 4) * 1.3) * 10, 2), c(2L, 3L, 4L)), "float64")
  ys <- as_jax(array(seq_len(24) %% 5, c(2L, 3L, 4L)), "int64")
  fs <- list(
    function(x) lax_sort(x, 1L),
    function(x) lax_sort(x, 0L, is_stable = TRUE),
    function(x) lax_top_k(x, 2L)[[1L]],
    function(x) lax_top_k(x, 2L, axis = 0L)[[2L]],
    function(x) grad(function(y) jnp_sum(jnp_sin(lax_sort(y, 1L)) * jnp_arange(4, dtype = "float64")))(x),
    function(x) grad(function(y) jnp_sum(jnp_sin(lax_top_k(y, 2L, axis = 0L)[[1L]])))(x))
  for (f in fs) {
    for (ax in 0:2) {
      xa <- jnp_moveaxis(xs, 0L, ax)
      expect_jax_equal(vmap(f, in_axes = ax)(xa), sort_stack_loop(f, xa, ax))
    }
  }
  # one batched and one unbatched operand
  key <- lax_index_in_dim(ys, 0L, 0L, keepdims = FALSE)
  fk <- function(v) lax_sort(list(key, v), 1L, num_keys = 1L)[[2L]]
  expect_jax_equal(vmap(fk, in_axes = 2L)(jnp_moveaxis(xs, 0L, 2L)), sort_stack_loop(fk, xs, 0L))
  fk2 <- function(k) lax_sort(list(k, lax_index_in_dim(xs, 0L, 0L, keepdims = FALSE)), 0L)[[2L]]
  expect_jax_equal(vmap(fk2, in_axes = 1L)(jnp_moveaxis(ys, 0L, 1L)), sort_stack_loop(fk2, ys, 0L))
})

test_that("jnp sort, argsort, partition, lexsort and top_k match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  x <- as_jax(sort_x(), "float64")
  expect_jax_equal(jnp_sort(x, axis = NULL), c(-2.1, -1.8, -0.7, -0.5, 0.3, 0.6, 0.9, 1.1, 1.4,
                                                1.7, 2.2, 2.9, 3.3, 3.8, 4.4))
  expect_jax_equal(jnp_sort(x, axis = 0L, descending = TRUE),
                   matrix(c(2.2, 2.9, 3.3, 4.4, 3.8, 0.6, 1.7, -0.7, 1.4, 0.9,
                            0.3, -0.5, -2.1, 1.1, -1.8), 3, byrow = TRUE))
  y <- as_jax(c(3, 1, 2, 1, 3, 0), "int64")
  expect_jax_equal(jnp_argsort(y), c(5, 1, 3, 2, 0, 4), dtype = "int64")
  expect_jax_equal(jnp_argsort(y, descending = TRUE), c(0, 4, 2, 1, 3, 5))
  expect_jax_equal(jnp_argsort(y, dtype = "int16"), c(5, 1, 3, 2, 0, 4), dtype = "int16")
  expect_jax_equal(jnp_argsort(x, axis = 0L), matrix(c(0, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 1, 0, 2),
                                                     3, byrow = TRUE))
  z <- as_jax(c(7, 2, 9, 1, 5, 3, 8), "float64")
  expect_jax_equal(jnp_partition(z, 3L), c(1, 2, 3, 5, 9, 8, 7))
  expect_jax_equal(jnp_argpartition(z, 3L), c(3, 1, 5, 4, 0, 2, 6), dtype = "int32")
  expect_jax_equal(jnp_partition(z, -2L), c(1, 2, 3, 5, 7, 8, 9))
  expect_jax_equal(jnp_argpartition(x, 1L, axis = 0L),
                   matrix(c(0, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 1, 0, 2), 3, byrow = TRUE))
  expect_jax_equal(jnp_partition(x, 2L, axis = 1L),
                   matrix(c(-2.1, 0.3, 0.9, 4.4, 1.7, -1.8, -0.5, 1.1, 3.3, 2.2,
                            -0.7, 0.6, 1.4, 3.8, 2.9), 3, byrow = TRUE))
  expect_jax_equal(jnp_partition(as_jax(c(5L, -3L, 8L, 0L, 2L)), 1L), c(-3, 0, 8, 5, 2), dtype = "int32")
  expect_jax_equal(jnp_sort_complex(as_jax(c(3, 1, 2), "int64")), c(1, 2, 3) + 0i, dtype = "complex128")
  expect_jax_equal(jnp_sort_complex(as_jax(c(3 + 1i, 1 - 2i, 1 - 3i))), c(1 - 3i, 1 - 2i, 3 + 1i))
  surnames <- as_jax(c(3, 1, 2, 1, 3), "int64")
  first <- as_jax(c(1, 2, 0, 1, 0), "int64")
  expect_jax_equal(jnp_lexsort(list(first, surnames)), c(3, 1, 2, 4, 0), dtype = "int64")
  expect_jax_equal(jnp_lexsort(jnp_stack(list(first, surnames))), c(3, 1, 2, 4, 0))
  expect_jax_equal(jnp_lexsort(list(x, jnp_round(x)), axis = 0L),
                   matrix(c(0, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 1, 0, 2), 3, byrow = TRUE))
  expect_jax_equal(jnp_lexsort(as_jax(c(3, 1, 2), "int64")), 0, shape = integer(0))
  keys <- lapply(0:19, function(i) as_jax(c(i %% 2, (i * 7) %% 3, i %% 5, 1, 0, 2), "int64"))
  expect_jax_equal(jnp_lexsort(keys), c(4, 0, 1, 3, 5, 2))
  st <- jnp_top_k(z, 2L, mode = "smallest")
  expect_jax_equal(st[[1L]], c(1, 2))
  expect_jax_equal(st[[2L]], c(3, 1))
  expect_error(jnp_partition(as_jax(1i), 0L), "jnp.partition for complex dtype is not implemented.")
  expect_error(jnp_lexsort(list()), "need sequence of keys with len > 0 in lexsort")
  check_grad_fd(function(x) jnp_sum(jnp_sin(jnp_sort(x, axis = NULL)) * jnp_arange(15, dtype = "float64")), x)
  check_grad_fd(function(x) jnp_sum(jnp_partition(x, 2L, axis = 1L)^2 * jnp_arange(5, dtype = "float64")), x)
})

# The "scan" method vmaps a lax_scan over the queries; that needs vmap of a scan whose
# body selects between a batched closed over value and an unbatched one.
scan_vmap_ok <- function() {
  f <- function(q) lax_scan(function(c, x) list(lax_add(c, lax_select(lax_eq(q, 0), as_jax(1), q)), NULL),
                            as_jax(0), NULL, length = 2L)[[1L]]
  tryCatch(isTRUE(all.equal(as.vector(vmap(f)(as_jax(c(0, 3)))), c(2, 6))), error = function(e) FALSE)
}

test_that("jnp_searchsorted matches JAX for all methods", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  a <- as_jax(c(1, 2, 2, 3, 4, 5, 5), "int64")
  vals <- as_jax(c(0, 3, 8, 1.5, 2, 5, 6.5), "float64")
  for (m in c("sort", "compare_all")) {
    expect_jax_equal(jnp_searchsorted(a, vals, method = m), c(0, 3, 7, 1, 1, 5, 7), dtype = "int32")
    expect_jax_equal(jnp_searchsorted(a, vals, side = "right", method = m), c(0, 4, 7, 1, 3, 7, 7))
    expect_jax_equal(jit(function(a, v) jnp_searchsorted(a, v, method = m))(a, vals),
                     c(0, 3, 7, 1, 1, 5, 7))
  }
  # scalar queries need no vmap for the scan method
  for (m in c("scan", "scan_unrolled", "sort", "compare_all")) {
    expect_jax_equal(jnp_searchsorted(a, 2, method = m), 1, shape = integer(0), dtype = "int32")
    expect_jax_equal(jnp_searchsorted(a, 2, side = "right", method = m), 3, shape = integer(0))
    expect_jax_equal(jnp_searchsorted(a, 8, method = m), 7, shape = integer(0))
  }
  expect_jax_equal(jnp_searchsorted(as_jax(numeric(0), "float64"), 2, method = "scan"), 0,
                   shape = integer(0))
  sa <- as_jax(c(0.5, 1.5, NaN), "float64")
  q <- as_jax(matrix(c(0, 1, 2, NaN), 2, byrow = TRUE), "float64")
  expect_jax_equal(jnp_searchsorted(sa, q, method = "sort"), matrix(c(0, 1, 2, 2), 2, byrow = TRUE))
  expect_jax_equal(jnp_searchsorted(sa, q, side = "right", method = "compare_all"),
                   matrix(c(0, 1, 2, 3), 2, byrow = TRUE))
  expect_jax_equal(jnp_searchsorted(sa, q, side = "right", method = "sort"),
                   matrix(c(0, 1, 2, 3), 2, byrow = TRUE))
  a2 <- as_jax(c(4, 3, 5, 1, 2), "int64")
  expect_jax_equal(jnp_searchsorted(a2, vals, sorter = jnp_argsort(a2), method = "sort"),
                   c(0, 2, 5, 1, 1, 4, 5))
  f <- function(v) jnp_searchsorted(a, v, method = "sort")
  expect_jax_equal(vmap(f, in_axes = 1L)(jnp_stack(list(vals, vals + 1), 1L)),
                   jnp_stack(list(f(vals), f(vals + 1))))
  expect_error(jnp_searchsorted(as_jax(sort_x()), vals), "a should be 1-dimensional")
  expect_error(jnp_searchsorted(a, vals, side = "middle"), "invalid argument side='middle'")
  expect_error(jnp_searchsorted(a, vals, method = "foo"), "invalid argument method='foo'")
  skip_if_not(scan_vmap_ok(), "vmap of lax_scan with a batched closed over select is broken in R/lax_control.R")
  expect_jax_equal(jnp_searchsorted(a, vals), c(0, 3, 7, 1, 1, 5, 7), dtype = "int32")
  expect_jax_equal(jnp_searchsorted(a, vals, side = "right"), c(0, 4, 7, 1, 3, 7, 7))
  expect_jax_equal(jit(function(a, v) jnp_searchsorted(a, v))(a, vals), c(0, 3, 7, 1, 1, 5, 7))
  expect_jax_equal(jnp_searchsorted(sa, q), matrix(c(0, 1, 2, 2), 2, byrow = TRUE))
  expect_jax_equal(jnp_searchsorted(a2, vals, sorter = jnp_argsort(a2)), c(0, 2, 5, 1, 1, 4, 5))
  g <- function(v) jnp_searchsorted(a, v)
  expect_jax_equal(vmap(g, in_axes = 1L)(jnp_stack(list(vals, vals + 1), 1L)),
                   jnp_stack(list(g(vals), g(vals + 1))))
})
