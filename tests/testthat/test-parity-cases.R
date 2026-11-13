# Edge cases compared with JAX 0.12.0.1: each expected value below is the output of the
# same call in the Python reference (dtype, shape and row-major values).

pc_expect <- function(got, dtype, shape, values) {
  testthat::expect_identical(jaxR::dtype(got), dtype)
  testthat::expect_identical(as.integer(dim(got)), as.integer(shape))
  a <- as.array(got)
  a <- as.vector(if (length(dim(a)) > 1L) aperm(a) else a)
  if (is.null(values)) values <- numeric(0)
  if (is.logical(values)) testthat::expect_identical(a, values)
  else testthat::expect_equal(a, values, tolerance = 2e-5)
}
pc_format <- function(v) {
  paste0(toupper(sub("jax_", "", class(v)[1L])), " shape=(", paste(v$shape, collapse = ","), ") ", v$dtype)
}
f32 <- function(v) as_jax(v, "float32")
test_that("array creation and manipulation edge cases match jax.numpy", {
  skip_if_no_backend()
  pc_expect(jnp_full_like(jnp_zeros(c(2L, 3L)), f32(c(1, 2, 3))), "float32", c(2L, 3L), c(1, 2, 3, 1, 2, 3))
  r <- jnp_linspace(0, 1, 5L, retstep = TRUE)
  pc_expect(r[[1L]], "float32", c(5L), c(0, 0.25, 0.5, 0.75, 1))
  pc_expect(r[[2L]], "float32", integer(0), c(0.25))
  pc_expect(jnp_linspace(f32(c(0, 1)), f32(c(1, 3)), 3L, axis = 1L), "float32", c(2L, 3L), c(0, 0.5, 1, 1, 2, 3))
  pc_expect(jnp_arange(0.5, 3), "float32", c(3L), c(0.5, 1.5, 2.5))
  pc_expect(jnp_arange(5), "int32", c(5L), c(0, 1, 2, 3, 4))
  pc_expect(jnp_arange(as_jax(3L, "int16")), "int16", c(3L), c(0, 1, 2))
  pc_expect(jnp_arange(1, 2, 0.25), "float32", c(4L), c(1, 1.25, 1.5, 1.75))
  expect_error(jnp_arange(1i), class = "jax_value_error")
  pc_expect(jnp_concatenate(list(f32(c(1, 2)), f32(c(3, 4))), dtype = "int32"), "int32", c(4L), c(1, 2, 3, 4))
  pc_expect(jnp_concatenate(jnp_ones(c(2L, 2L))), "float32", c(4L), c(1, 1, 1, 1))
  pc_expect(jnp_einsum("...i,->...i", f32(c(1, 2)), f32(3)), "float32", c(2L), c(3, 6))
  pc_expect(jnp_reshape(jnp_arange(6), c(2L, 3L), order = "F"), "int32", c(2L, 3L), c(0, 2, 4, 1, 3, 5))
  pc_expect(jnp_ravel(jnp_reshape(jnp_arange(6), c(2L, 3L)), order = "F"), "int32", c(6L), c(0, 3, 1, 4, 2, 5))
  pc_expect(jnp_array(list()), "float32", c(0L), c())
  pc_expect(jnp_array(c(1, 2), ndmin = 3L), "float32", c(1L, 1L, 2L), c(1, 2))
  pc_expect(jnp_trim_zeros(f32(c(0, NaN, 1, 0))), "float32", c(2L), c(NaN, 1))
  expect_error(jnp_zeros(2.5), class = "jax_type_error")
})

test_that("ufunc, reduction and linalg edge cases match jax.numpy", {
  skip_if_no_backend()
  pc_expect(jnp_sum(f32(c(1, 2, 3)), where = as_jax(c(TRUE, FALSE, TRUE))), "float32", integer(0), c(4))
  pc_expect(jnp_sum(as_jax(c(TRUE, TRUE, FALSE))), "int32", integer(0), c(2))
  pc_expect(jnp_max(f32(c(1, 2, 3)), initial = 5), "float32", integer(0), c(5))
  expect_error(jnp_max(jnp_zeros(0L)), class = "jax_value_error")
  pc_expect(jnp_var(f32(c(1, 2)), ddof = 2L), "float32", integer(0), c(NaN))
  pc_expect(jnp_argmax(jnp_reshape(jnp_arange(6), c(2L, 3L)), axis = 1L, keepdims = TRUE), "int32", c(2L, 1L), c(2, 2))
  pc_expect(jnp_mean(f32(c(1, 2, 3)), where = as_jax(c(TRUE, FALSE, TRUE))), "float32", integer(0), c(2))
  pc_expect(jnp_maximum(as_jax(c(1+2i, 1+1i)), as_jax(c(1+1i, 2+0i))), "complex64", c(2L), c(complex(real = 1.0, imaginary = 2.0), complex(real = 2.0, imaginary = 0.0)))
  pc_expect(jnp_less(as_jax(c(1+2i, 1+1i)), as_jax(c(1+3i, 0+5i))), "bool", c(2L), c(TRUE, FALSE))
  pc_expect(jnp_min(as_jax(c(1+2i, 1+1i, 3+0i))), "complex64", integer(0), c(complex(real = 1.0, imaginary = 1.0)))
  pc_expect(jnp_polyder(f32(c(1, 2, 3)), m = 2L), "float32", c(1L), c(2))
  pc_expect(jnp_round(f32(c(1.2345678, 2.5)), 50L), "float32", c(2L), c(1.23457, 2.5))
  pc_expect(jnp_round(as_jax(c(1.26+2.34i, 0.44+0.05i)), 1L), "complex64", c(2L), c(complex(real = 1.3000001, imaginary = 2.3), complex(real = 0.4, imaginary = 0.0)))
  expect_error(jnp_round(as_jax(c(15L, 25L)), -1L), class = "jax_not_implemented_error")
  pc_expect(jnp_arctan2(as_jax(1L), as_jax(1L)), "float32", integer(0), c(0.785398))
  pc_expect(jnp_power(as_jax(c(2L, 3L)), 2.5), "float32", c(2L), c(5.65685, 15.5885))
  pc_expect(jnp_power(f32(c(2, 3)), as_jax(2L)), "float32", c(2L), c(4, 9))
  expect_error(jnp_power(as_jax(c(2L, 3L)), as_jax(-1L)), class = "jax_type_error")
  pc_expect(jnp_linalg_vector_norm(as_jax(c(1L, -3L)), ord = Inf), "int32", integer(0), c(3))
  pc_expect(jnp_linalg_vector_norm(f32(c(1, 2)), ord = 3), "float32", integer(0), c(2.08008))
  pc_expect(jnp_log2(as_jax(0+0i)), "complex64", integer(0), c(complex(real = -Inf, imaginary = 0)))
  pc_expect(jnp_log10(as_jax(0+0i)), "complex64", integer(0), c(complex(real = -Inf, imaginary = 0)))
  pc_expect(jnp_quantile(f32(c(1, 2, 3)), 0.5, axis = integer(0)), "float32", c(3L), c(1, 2, 3))
  pc_expect(jnp_logical_not(as_jax(c(0+0i, 0+1i))), "bool", c(2L), c(TRUE, FALSE))
  pc_expect(jnp_isinf(as_jax(c(complex(real = Inf, imaginary = 0), 1+1i))), "bool", c(2L), c(TRUE, FALSE))
  expect_error(jnp_sum(jnp_ones(c(2L, 2L)), axis = c(0L, 0L)), class = "jax_value_error")
  expect_error(jnp_fft_fft(f32(c(1, 2, 3)), n = 2.5), class = "jax_type_error")
})

test_that("jax.scipy edge cases match JAX", {
  skip_if_no_backend()
  pc_expect(scipy_stats_cauchy_logcdf(f32(c(-2, 0, 3))), "float32", c(3L), c(-1.913361, -0.693147, -0.108049))
  pc_expect(scipy_signal_welch(jnp_sin(jnp_arange(64, dtype = "float32")), nperseg = 16L, window = "hamming")[[2L]][sl(stop = 3)], "float32", c(3L), c(0.101453, 0.0430465, 3.62542))
  pc_expect(scipy_fft_dctn(f32(3)), "float32", integer(0), c(3))
  expect_error(scipy_ndimage_map_coordinates(f32(c(1, 2, 3)), list(f32(c(0.5))), order = 0.5), class = "jax_not_implemented_error")
  expect_error(scipy_stats_sem(jnp_ones(c(2L, 2L)), axis = c(0L, 1L)), class = "jax_type_error")
})

test_that("nn, image, optimizer and random edge cases match JAX", {
  skip_if_no_backend()
  opt <- optimizers_sm3(0.1)
  pc_expect({ st <- opt$init_fn(jnp_array(c(1, 2))); st <- jit(opt$update_fn)(0L, jnp_array(c(0.2, -0.3)), st); opt$params_fn(st) }, "float32", c(2L), c(0.99, 2.01))
  expect_error(image_resize(jnp_array(c(1, 2)), 1.5, "nearest"), class = "jax_type_error")
  expect_error(nn_initializers_ones(random_key(0), c(2.5, 2)), class = "jax_type_error")
  expect_error(image_scale_and_translate(jnp_arange(3L, dtype = "float32"), 4L, 0L, jnp_array(c(1, 2)), jnp_array(c(0, 1)), "linear"), class = "jax_assertion_error")
  pc_expect(image_scale_and_translate(jnp_arange(3L, dtype = "float32"), 4L, 0L, jnp_array(array(1.5, 1)), jnp_array(array(0.5, 1)), "linear"), "float32", c(4L), c(0, 0.166667, 0.833333, 1.5))
  pc_expect(nn_softmax(as_jax(c(0+0i, 1+0i, 1+0i), "complex64")), "complex64", c(3L), c(complex(real = 0.15536241, imaginary = 0.0), complex(real = 0.42231882, imaginary = 0.0), complex(real = 0.42231882, imaginary = 0.0)))
  pc_expect(nn_log_softmax(as_jax(c(0+0i, 1+0i, 1+0i), "complex64")), "complex64", c(3L), c(complex(real = -1.8619947, imaginary = 0.0), complex(real = -0.8619948, imaginary = 0.0), complex(real = -0.8619948, imaginary = 0.0)))
  pc_expect(nn_sparse_plus(as_jax(c(-1+0i, 0+0i, 1+0i), "complex64")), "complex64", c(3L), c(complex(real = 0.0, imaginary = 0.0), complex(real = 0.25, imaginary = 0.0), complex(real = 1.0, imaginary = 0.0)))
  pc_expect(nn_elu(as_jax(c(-1+0i, 1+1i), "complex64")), "complex64", c(2L), c(complex(real = -0.63212055, imaginary = 0.0), complex(real = 1.0, imaginary = 1.0)))
  pc_expect(nn_relu6(as_jax(c(-1+0i, 0+0i, 1+0i), "complex64")), "complex64", c(3L), c(complex(real = 0.0, imaginary = 0.0), complex(real = 0.0, imaginary = 0.0), complex(real = 1.0, imaginary = 0.0)))
  expect_error(nn_standardize(as_jax(c(0+0i, 1+0i, 1+0i), "complex64")), class = "jax_value_error")
  expect_error(nn_sparse_sigmoid(as_jax(c(0+0i), "complex64")), class = "jax_value_error")
  pc_expect(nn_sparse_sigmoid(f32(c(-2, 0, 0.5, 3))), "float32", c(4L), c(0, 0.5, 0.75, 1))
  expect_error(random_randint(random_key(2L), 8L, 0, 4294967296, dtype = "uint32"), class = "jax_overflow_error")
  expect_error(image_resize(jnp_zeros(c(0L, 2L)), c(3L, 2L), "linear"), class = "jax_value_error")
  pc_expect(nn_softmax(f32(c(1, 2, 3)), where = as_jax(c(TRUE, FALSE, TRUE))), "float32", c(3L), c(0.119203, 0, 0.880797))
  pc_expect(nn_standardize(f32(c(1, 2, 4))), "float32", c(3L), c(-1.06904, -0.267261, 1.3363))
  r <- nn_logsumexp(f32(c(1, 2, 3)), b = f32(c(1, -1, 1)), return_sign = TRUE)
  pc_expect(r[[1L]], "float32", integer(0), c(2.73533))
  pc_expect(r[[2L]], "float32", integer(0), c(1))
})

test_that("sparse construction, vmap, autodiff and linalg match jax.experimental.sparse", {
  skip_if_no_backend()
  B <- experimental_sparse_BCOO
  S2 <- experimental_sparse_BCSR
  mi <- as_jax(matrix(c(0L, 2L, 1L, 0L), 2, byrow = TRUE), "int32")
  m <- as_jax(matrix(c(1, 0, 2, 0, 3, 0), 2, byrow = TRUE))
  m3 <- jnp_stack(list(m, m * 2))
  b <- B$fromdense(c(1, 0, 2))
  co <- experimental_sparse_COO$fromdense(matrix(c(1, 0, 0, 2), 2))
  cs <- experimental_sparse_CSR$fromdense(matrix(c(4, 1, 0, 1, 3, 0, 0, 0, 2), 3))
  rhs <- jnp_array(c(1, 2, 3))
  sol <- function(d, bb) experimental_sparse_linalg_spsolve(d, cs$indices, cs$indptr, bb)
  A <- jnp_diag(jnp_arange(20, dtype = "float32"))
  X <- jnp_reshape(jnp_sin(jnp_arange(40, dtype = "float32")), c(20L, 2L))
  pc_expect(B$fromdense(mi, nse = 5L)$todense(), "int32", c(2L, 2L), c(0, 2, 1, 0))
  pc_expect(B$fromdense(as_jax(c(TRUE, FALSE)))$data, "int32", c(1L), c(1))
  pc_expect(experimental_sparse_bcoo_slice(B(list(as_jax(c(1L, 2L), "int32"), as_jax(matrix(c(0L, 0L, 1L, 1L), 2, byrow = TRUE), "int32")), shape = c(2L, 2L)), c(0L, 0L), c(1L, 2L))$todense(), "int32", c(1L, 2L), c(1, 0))
  pc_expect(vmap(function(x) B$fromdense(x, nse = 3L))(m)$todense(), "float32", c(2L, 3L), c(1, 0, 2, 0, 3, 0))
  pc_expect(vmap(function(x) x$todense())(B$fromdense(m, n_batch = 1L)), "float32", c(2L, 3L), c(1, 0, 2, 0, 3, 0))
  expect_error(vmap(function(x) x$todense())(B$fromdense(m)), class = "jax_value_error")
  expect_identical(pc_format(vmap(function(x) x * 2)(B$fromdense(m, n_batch = 1L))), "BCOO shape=(2,3) float32")
  pc_expect(vmap(function(x) x$todense())(S2$fromdense(m3, n_batch = 1L)), "float32", c(2L, 2L, 3L), c(1, 0, 2, 0, 3, 0, 2, 0, 4, 0, 6, 0))
  pc_expect(vmap(function(x) S2$fromdense(x, nse = 3L))(m3)$todense(), "float32", c(2L, 2L, 3L), c(1, 0, 2, 0, 3, 0, 2, 0, 4, 0, 6, 0))
  expect_identical(pc_format(experimental_sparse_jacrev(function(x) x$todense())(co)), "COO shape=(2,2) float32")
  expect_identical(pc_format(experimental_sparse_jacrev(function(x) x$todense())(B$fromdense(matrix(c(1, 0, 0, 2), 2)))), "BCOO shape=(2,2,2,2) float32")
  r <- { r <- b$sum_duplicates(nse = 4L, remove_zeros = FALSE); list(r$data, r$indices) }
  pc_expect(r[[1L]], "float32", c(4L), c(1, 2, 0, 0))
  pc_expect(r[[2L]], "int32", c(4L, 1L), c(0, 2, 3, 3))
  pc_expect(sol(cs$data, rhs), "float32", c(3L), c(0.0909091, 0.636364, 1.5))
  pc_expect(jit(sol)(cs$data, rhs), "float32", c(3L), c(0.0909091, 0.636364, 1.5))
  pc_expect(grad(function(bb) jnp_sum(sol(cs$data, bb)^2))(rhs), "float32", c(3L), c(-0.0661157, 0.446281, 1.5))
  pc_expect(grad(function(d) jnp_sum(sol(d, rhs)^2))(cs$data), "float32", c(5L), c(0.00601052, 0.0420736, -0.040571, -0.283997, -2.25))
  pc_expect(experimental_sparse_linalg_lobpcg_standard(function(v) lax_dot(A, v), X)[[1L]], "float32", c(2L), c(19, 18))
  expect_error(experimental_sparse_linalg_lobpcg_standard(A, jnp_ones(c(20L, 4L))), class = "jax_value_error")
  for (fmt in c("BCSR", "COO", "CSR", "CSC")) {
    cls <- get(paste0("experimental_sparse_", fmt))
    pc_expect(cls$fromdense(mi)$todense(), "int32", c(2L, 2L), c(0, 2, 1, 0))
  }
  a <- matrix(c(1, 0, 2, 0, 3, 0), 2, byrow = TRUE)
  for (fmt in c("COO", "CSR", "CSC")) {
    mm <- get(paste0("experimental_sparse_", fmt))$fromdense(a)
    expect_error(mm %*% c(1, 2), class = "jax_assertion_error")
    pc_expect(mm %*% c(1, 2, 3), "float32", 2L, c(7, 6))
    pc_expect(mm %*% matrix(1, 3, 2), "float32", c(2L, 2L), c(3, 3, 3, 3))
  }
  # Gradients keep the container of a single sparse leaf.
  g <- experimental_sparse_grad(function(z) jnp_sum(z$x$todense()^2))(list(x = b))
  expect_named(g, "x")
  expect_s3_class(g$x, "jax_bcoo")
  expect_s3_class(experimental_sparse_grad(function(z) jnp_sum(z$todense()^2))(b), "jax_bcoo")
  # LOBPCG: the two largest eigenvalues of diag(0, ..., 19).
  r <- experimental_sparse_linalg_lobpcg_standard(A, X)
  pc_expect(r[[1L]], "float32", 2L, c(19, 18))
  expect_gt(as.integer(as.array(r[[3L]])), 0L)
  expect_lt(as.numeric(jnp_max(jnp_abs(lax_dot(A, r[[2L]]) - r[[2L]] * lax_expand_dims(r[[1L]], 0L)))), 1e-2)
})

test_that("sparse matrices of the Matrix package import without densification", {
  skip_if_no_backend()
  skip_if_not_installed("Matrix")
  M <- Matrix::Matrix(c(2, 1, 0, 1, 3, 0, 0, 0, 4), 3, sparse = TRUE)   # stored as symmetric
  bc <- experimental_sparse_BCOO$from_scipy_sparse(M)
  expect_identical(bc$nse, 5L)
  pc_expect(bc$todense(), "float32", c(3L, 3L), c(2, 1, 0, 1, 3, 0, 0, 0, 4))
  bs <- experimental_sparse_BCSR$from_scipy_sparse(Matrix::Matrix(c(0, 1, 2, 0, 0, 3), 2, sparse = TRUE))
  pc_expect(bs$indptr, "int32", 3L, c(0, 1, 3))
  pc_expect(bs$todense(), "float32", c(2L, 3L), c(0, 2, 0, 1, 0, 3))
  expect_error(experimental_sparse_BCOO$from_scipy_sparse(diag(2)), class = "jax_type_error")
  expect_error(experimental_sparse_BCOO$from_scipy_sparse(M, n_batch = 1L), class = "jax_not_implemented_error")
})

test_that("jit keys, pytree matching and static arguments match JAX", {
  skip_if_no_backend()
  # Two dicts whose printed structures coincide are different cache entries.
  f <- jit(function(d) if ("a" %in% names(d)) d[["a"]] else d[["c"]])
  d1 <- stats::setNames(list(1, 2), c("a", "b = *, c"))
  d2 <- stats::setNames(list(10, 20), c("a = *, b", "c"))
  expect_equal(c(as.array(f(d1)), as.array(f(d2))), c(1, 20))
  # A dict does not match a positional list, and node metadata is compared.
  pull <- vjp(function(x) list(a = x, b = 2 * x), 3)[[2L]]
  expect_error(pull(list(10, 20)), "unexpected tree structure", class = "jax_value_error")
  pc_expect(pull(list(b = 20, a = 10))[[1L]], "float32", integer(0), 50)
  expect_error(tree_map(function(x, y) x + y, list(1, 2), list(a = 1, b = 2)), class = "jax_value_error")
  expect_error(tree_map(function(x, y) x + y, list(1, list(2, 3)), list(1, NULL)), class = "jax_value_error")
  Box <- function(x, tag) structure(list(x = x, tag = tag), class = "pc_box")
  register_pytree_node("pc_box", function(b) list(children = list(b$x), aux = list(b$tag)),
                       function(a, ch) Box(ch[[1L]], a[[1L]]))
  expect_identical(tree_map(function(x, y) x + y, Box(1, "a"), Box(2, "a"))$x, 3)
  expect_error(tree_map(function(x, y) x + y, Box(1, "a"), Box(2, "b")), class = "jax_value_error")
  expect_error(vmap(function(b) b$x, in_axes = list(Box(0L, "a")))(Box(as_jax(c(1, 2)), "b")),
               "tree prefix", class = "jax_value_error")
  # Static arguments: nothing is inferred when both specifications are given.
  g <- jit(function(x, n) if (n == 2L) x * 2 else x, static_argnums = integer(0), static_argnames = "n")
  expect_error(g(3, 2L))
  pc_expect(g(3, n = 2L), "float32", integer(0), 6)
  h <- jit(function(...) { a <- list(...); if (a[[2L]] == 2L) a[[1L]] * 2 else a[[1L]] },
           static_argnums = -1L)
  pc_expect(h(2, 2L), "float32", integer(0), 4)
  ck <- checkpoint(function(x, n) if (n > 0) x * x else x, static_argnums = -1L)
  pc_expect(ck(2, 1L), "float32", integer(0), 4)
  expect_error(checkpoint(function(x) x, static_argnums = 3L)(1), class = "jax_value_error")
  # make_jaxpr(return_shape = TRUE) keeps the weak type.
  s <- make_jaxpr(function(x) x * x, return_shape = TRUE)(2)[[2L]]
  expect_true(s$weak)
  expect_identical(eval_shape(function(x) x + as_jax(1, "float16"), s)$dtype, "float16")
})

test_that("derivative options and custom rule conventions match JAX", {
  skip_if_no_backend()
  r <- linearize(function(x, y) x * y, 2, 3, in_nzs = list(TRUE, FALSE))
  pc_expect(r[[2L]](1, 0), "float32", integer(0), 3)
  v <- vjp(function(x, y) x * y, 2, 3, in_nzs = list(TRUE, FALSE))[[2L]](1)
  expect_equal(vapply(v, function(t) as.numeric(as.array(t)), 0), c(3, 0))
  expect_error(vjp(function(x, y) x * y, 2, 3, in_nzs = list(TRUE)), class = "jax_value_error")
  # A negative entry of nondiff_argnums is passed to the rule, as in JAX.
  f <- custom_jvp(function(x, n) x * n,
                  function(n, p, t) list(p[[1L]] * n, t[[1L]] * n), nondiff_argnums = -1L)
  pc_expect(grad(function(x) f(x, 3))(2), "float32", integer(0), 3)
  # The cotangent of a custom VJP rule keeps its dtype.
  cv <- custom_vjp(function(x) x, function(x) list(x, NULL),
                   function(res, g) list(as_jax(1 + 2i, "complex64")))
  gc <- grad(cv)(as_jax(2, "float32"))
  expect_identical(dtype(gc), "complex64")
  expect_equal(as.array(gc), 1 + 2i)
  # The gradient through a dtype conversion keeps the weak type of the input.
  gw <- grad(function(v) jnp_sum(jnp_full_like(jnp_zeros(3L), v)))(2)
  expect_identical(dtype(gw * jnp_array(3, dtype = "float16")), "float16")
  # A ref cannot be both an argument and closed over.
  rr <- new_ref(1)
  fa <- jit(function(a) { a[] <- a[] + 1; rr[] <- rr[] + 1; a[] })
  expect_error(fa(rr), "both closed over and passed as an argument", class = "jax_value_error")
  r2 <- new_ref(1)
  fb <- jit(function(a) { a[] <- a[] + 1; a[] })
  pc_expect(fb(r2), "float32", integer(0), 2)
  pc_expect(r2[], "float32", integer(0), 2)
})

test_that("array construction and file edge cases match NumPy and jax.numpy", {
  skip_if_no_backend()
  pc_expect(jnp_full_like(jnp_zeros(2L, "int8"), Inf), "int8", 2L, c(127, 127))
  pc_expect(jnp_full_like(jnp_zeros(2L, "int8"), -Inf), "int8", 2L, c(-128, -128))
  pc_expect(jnp_full_like(jnp_zeros(2L, "uint8"), c(-1L, 256L)), "uint8", 2L, c(255, 0))
  s <- function(x) jnp_array(x, dtype = "float32")
  pc_expect(jnp_arange(s(0), s(0.3), s(0.1)), "float32", 3L, c(0, 0.1, 0.2))
  pc_expect(jnp_arange(s(0), s(0.6), s(0.2)), "float32", 3L, c(0, 0.2, 0.4))
  pc_expect(jnp_frombuffer(raw(0), dtype = "complex128"), "complex64", 0L, complex(0))
  pc_expect(jnp_frombuffer(as.raw(c(0, 1, 56, 127)), dtype = "int4"), "int4", 4L, c(0, 1, -8, -1))
  pc_expect(jnp_frombuffer(as.raw(c(0, 1, 56, 127)), dtype = "uint4"), "uint4", 4L, c(0, 1, 8, 15))
  d <- tempfile()
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE))
  expect_error(jnp_savez(file.path(d, "clash.npz"), jnp_arange(3L), arr_0 = jnp_ones(2L)),
               "Cannot use un-named variables and keyword arr_0", class = "jax_value_error")
  jnp_savez(file.path(d, "empty.npz"))
  expect_identical(readBin(file.path(d, "empty.npz"), "raw", 30L),
                   as.raw(c(0x50, 0x4b, 0x05, 0x06, rep(0, 18L))))
  expect_identical(jnp_load(file.path(d, "empty.npz")), stats::setNames(list(), character(0)))
})

test_that("numpy edge cases of sets, powers, sorting and rounding match JAX", {
  skip_if_no_backend()
  # One surviving element is still a vector.
  pc_expect(jnp_unique(c(1, 1, 1)), "float32", 1L, 1)
  pc_expect(jnp_intersect1d(c(1, 1), c(1, 1)), "float32", 1L, 1)
  # A concrete unsigned exponent beyond the range of R integers.
  big <- jnp_array(2147483648, dtype = "uint32")
  pc_expect(jnp_power(jnp_array(c(1L, 1L)), big), "int32", 2L, c(1, 1))
  pc_expect(jit(function(a, b) jnp_power(a, b))(jnp_array(c(1L, 1L)), big), "int32", 2L, c(1, 1))
  expect_error(jnp_round(c(TRUE, FALSE)), "not compatible with finfo", class = "jax_value_error")
  expect_error(jnp_around(c(TRUE, FALSE)), "not compatible with finfo", class = "jax_value_error")
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(dtype(jnp_sort_complex(jnp_array(c(3L, 1L, 2L), dtype = "int32"))), "complex64")
    expect_identical(dtype(jnp_sort_complex(jnp_array(c(3, 1, 2), dtype = "float64"))), "complex128")
  })
})

test_that("statistics and spectral windows match jax.scipy", {
  skip_if_no_backend()
  # Integer arguments with the integer defaults give strongly typed float32.
  x <- jnp_array(c(0L, 2L), dtype = "int32")
  y <- scipy_stats_cauchy_cdf(x)
  pc_expect(y + jnp_zeros(2L, dtype = "float16"), "float32", 2L, c(0.5, 0.8524164))
  expect_identical(dtype(scipy_stats_norm_pdf(x) + jnp_zeros(2L, dtype = "float16")), "float32")
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(dtype(scipy_stats_cauchy_cdf(x)), "float32")
    expect_identical(dtype(jit(scipy_stats_cauchy_cdf)(x)), "float32")
    expect_identical(dtype(vmap(scipy_stats_cauchy_cdf)(x)), "float32")
  })
  m <- matrix(1:4, 2)
  expect_error(scipy_stats_sem(m, axis = list(0L)), "tuple indices", class = "jax_type_error")
  expect_error(scipy_stats_sem(m, axis = integer(0)), "tuple indices", class = "jax_type_error")
  # Windows of scipy.signal.get_window, as tuples of a name and its parameters.
  sig <- c(0, 1, 2, 3)
  w <- function(window) suppressWarnings(scipy_signal_welch(sig, window = window, nperseg = 4L)[[2L]])
  pc_expect(w(list("chebwin", 60)), "float32", 3L, c(0.41409943, 2.5232797, 0.110624))
  pc_expect(w(list("taylor", 4, 30)), "float32", 3L, c(0.25968626, 2.9371727, 0.25800374))
  pc_expect(w(list("dpss", 1)), "float32", 3L, c(0.17217423, 3.1960638, 0.38472024))
  pc_expect(w(list("exponential", 1, 2)), "float32", 3L, c(0.16445181, 2.7554376, 0.2316423))
  expect_error(suppressWarnings(scipy_signal_welch(numeric(0), window = "hamming", nperseg = 4L)),
               "Parameter Nx=0 is not a positive integer", class = "jax_value_error")
  expect_error(w(list("dpss", 2)), "NW must be less than M/2", class = "jax_value_error")
  expect_error(w(list("hamming", 1)), "does not allow parameters", class = "jax_value_error")
})

test_that("sparse imports, solves and products match jax.experimental.sparse", {
  skip_if_no_backend()
  m <- experimental_sparse_CSR$fromdense(matrix(c(2, 0, 0, 3), 2), nse = 3L)
  pc_expect(experimental_sparse_linalg_spsolve(m$data, m$indices, m$indptr, as_jax(c(4, 9))),
            "float32", 2L, c(2, 3))
  z <- experimental_sparse_linalg_spsolve(as_jax(array(1 + 1i, 1L), "complex64"), as_jax(array(0L, 1L), "int32"),
                                          as_jax(c(0L, 1L, 1L), "int32"), as_jax(c(1 + 0i, 1 + 0i), "complex64"))
  expect_identical(dtype(z), "complex64")
  expect_true(all(is.nan(Re(as.array(z)))))
  co <- experimental_sparse_COO$fromdense(matrix(0:5 + 0, 2, byrow = TRUE))
  g <- experimental_sparse_jacrev(function(x) x$todense())(co)
  pc_expect(g %*% as_jax(c(1, 1, 1)), "float32", 2L, c(2, 3))
  skip_if_not_installed("Matrix")
  one <- Matrix::sparseMatrix(i = 1L, j = 2L, x = 3, dims = c(2L, 2L))
  for (cls in list(experimental_sparse_BCOO, experimental_sparse_BCSR))
    pc_expect(cls$from_scipy_sparse(one)$todense(), "float32", c(2L, 2L), c(0, 3, 0, 0))
  with_jax_config("jax_enable_x64", TRUE, {
    d2 <- Matrix::sparseMatrix(i = 1:2, j = 1:2, x = c(1, 2))
    for (cls in list(experimental_sparse_BCOO, experimental_sparse_BCSR))
      expect_identical(dtype(cls$from_scipy_sparse(d2)$indices), "int32")
  })
})

test_that("lax typing, construction and validation edge cases match JAX", {
  skip_if_no_backend()
  half <- as_jax(c(1, 1), "float16")
  # The mean over a named axis of an integer array is a strongly typed float32.
  pm <- vmap(function(x) lax_pmean(x, "i"), axis_name = "i")
  expect_identical(dtype(jnp_add(pm(as_jax(1:2)), half)), "float32")
  expect_error(vmap(function(x) lax_all_to_all(x, "i", 0L, 0L, axis_index_groups = list(c(0L, 1L))),
                    axis_name = "i")(as_jax(matrix(1:4, 2, byrow = TRUE))),
               class = "jax_not_implemented_error")
  # full_like: narrow dtypes under jit, non finite fills.
  pc_expect(jit(function(x) lax_full_like(x, 1))(as_jax(c(0, 1), "int4")), "int4", 2L, c(1, 1))
  b <- lax_full_like(as_jax(FALSE), NaN)
  expect_identical(c(as.vector(as.array(b)), as.vector(as.array(jit(function(y) y)(b)))), c(TRUE, TRUE))
  i8 <- lax_full_like(as_jax(0L, "int8"), Inf)
  expect_equal(c(as.vector(as.array(i8)), as.vector(as.array(jit(function(y) y)(i8)))), c(127, 127))
  # R numbers given an integer dtype: truncated, range checked, narrow integers wrap.
  expect_error(as_jax(c(128.5, 2), "int8"), "Python integer 128 out of bounds for int8",
               class = "jax_overflow_error")
  expect_error(as_jax(c(-1.5, 2), "uint32"), "Python integer -1 out of bounds for uint32",
               class = "jax_overflow_error")
  expect_error(as_jax(NaN, "int32"), "cannot convert float NaN to integer", class = "jax_value_error")
  expect_error(as_jax(Inf, "int32"), "cannot convert float infinity to integer",
               class = "jax_overflow_error")
  pc_expect(as_jax(c(0, 1, 2, 3), "int2"), "int2", 4L, c(0, 1, -2, -1))
  pc_expect(as_jax(c(1.9, -1.9), "int8"), "int8", 2L, c(1, -1))
  # reduce: a strongly typed initial value makes the result strong; dtypes must match.
  wx <- lax_broadcast(as_jax(1), 3L)
  expect_identical(dtype(jnp_add(lax_reduce(wx, as_jax(0, "float32"), lax_add, 0L), as_jax(1, "float16"))),
                   "float32")
  expect_identical(dtype(jnp_add(lax_reduce(wx, 0, lax_add, 0L), as_jax(1, "float16"))), "float16")
  expect_error(lax_reduce(as_jax(c(1, 2), "float32"), as_jax(0L, "int32"), function(a, b) lax_add(a, b), 0L),
               "reduce operand dtypes should match", class = "jax_type_error")
  pc_expect(lax_reduce(as_jax(c(1, 2), "float32"), 0L, function(a, b) lax_add(a, b), 0L),
            "float32", integer(0), 3)
  # Slices and conditionals.
  expect_error(lax_dynamic_slice(as_jax(0:4), list(-1L), 2L, allow_negative_indices = FALSE),
               "Index -1 is out of bounds for dimension 5", class = "jax_value_error")
  pc_expect(lax_dynamic_slice(as_jax(0:4), list(-2L), 2L), "int32", 2L, c(3, 4))
  expect_error(lax_dynamic_slice(as_jax(0:4), list(1L), 2L, allow_negative_indices = c(TRUE, FALSE)),
               "one flag or one flag per axis", class = "jax_value_error")
  expect_error(disable_jit(lax_cond(1 + 1i, function(x) x, function(x) -x, 1)),
               "Pred type must be either boolean or number", class = "jax_type_error")
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(dtype(pm(as_jax(1:2))), "float32")
    # A 64 bit integer computed on the device keeps all its bits as a jit constant.
    big <- lax_add(as_jax(2^53, "int64"), as_jax(1L, "int64"))
    expect_equal(as.vector(as.array(lax_sub(jit(function() big)(), big))), 0)
  })
})

test_that("narrow floats, update slices and fill values follow JAX", {
  skip_if_no_backend()
  # Formats without infinity reduce from their extreme finite values.
  dt <- "float8_e4m3fn"
  pc_expect(lax_reduce_min(as_jax(c(1, 2), dt), 0L), dt, integer(0), 1)
  pc_expect(lax_reduce_max(as_jax(c(-1, -2), dt), 0L), dt, integer(0), -1)
  pc_expect(lax_argmin(as_jax(c(2, 1), dt), 0L, "int32"), "int32", integer(0), 1)
  pc_expect(lax_argmax(as_jax(c(1, 2), dt), 0L, "int32"), "int32", integer(0), 1)
  expect_error(lax_dynamic_update_slice(as_jax(0:4), as_jax(array(9L, 1L)), list(-1L),
                                        allow_negative_indices = FALSE),
               "Index -1 is out of bounds for dimension 5", class = "jax_value_error")
  expect_error(lax_full_like(as_jax(TRUE), c(TRUE, FALSE)),
               "full must be called with scalar fill_value", class = "jax_type_error")
  expect_error(vmap(function(y) lax_pswapaxes(y, "i", 0L, axis_index_groups = list(c(0L, 1L))),
                    axis_name = "i")(as_jax(matrix(1:4, 2, byrow = TRUE))),
               class = "jax_not_implemented_error")
  for (f in list(lax_add, function(a, b) lax_add(a, b)))
    expect_identical(dtype(jnp_add(lax_reduce(1, as_jax(0, "float32"), f, integer(0)),
                                   as_jax(1, "float16"))), "float32")
})

test_that("special functions, custom_root and stax sizes follow JAX", {
  skip_if_no_backend()
  # Empty reductions inside logsumexp and softmax have an identity.
  e <- jnp_zeros(c(2L, 0L))
  ls <- scipy_special_logsumexp(e, axis = 1L, keepdims = TRUE, return_sign = TRUE)
  pc_expect(ls[[1L]], "float32", c(2L, 1L), c(-Inf, -Inf))
  pc_expect(ls[[2L]], "float32", c(2L, 1L), c(0, 0))
  pc_expect(scipy_special_softmax(e, axis = 1L), "float32", c(2L, 0L), NULL)
  pc_expect(scipy_special_log_softmax(e, axis = 1L), "float32", c(2L, 0L), NULL)
  # Orders are integers.
  expect_error(scipy_special_bessel_jn(as_jax(3), v = 1.5), "cannot be interpreted as an integer",
               class = "jax_type_error")
  expect_error(scipy_special_bernoulli(1.5), "cannot be interpreted as an integer",
               class = "jax_type_error")
  pc_expect(scipy_special_bernoulli(2), "float32", 3L, c(1, -0.5, 1 / 6))
  # The gradient through custom_root keeps the weak type of a scalar.
  root <- function(a) lax_custom_root(function(x) x - a, 1, function(f, x0) a, function(g, y) y / g(1))
  with_jax_config("jax_enable_x64", TRUE, {
    for (g in list(grad(root), jit(grad(root))))
      expect_identical(dtype(g(2) + as_jax(1, "float32")), "float32")
  })
  # stax sizes are whole numbers.
  k <- random_key(0L)
  expect_error(stax_Dense(2.5)$init_fun(k, c(3L, 4L)), class = "jax_type_error")
  expect_error(stax_Conv(2.5, c(1L, 1L))$init_fun(k, c(1L, 3L, 3L, 1L)), class = "jax_type_error")
  expect_error(stax_MaxPool(c(1.5, 1))$init_fun(k, c(1L, 3L, 3L, 1L)), class = "jax_type_error")
  expect_identical(as.integer(stax_Dense(2)$init_fun(k, c(3L, 4L))$output_shape), c(3L, 2L))
  expect_error(nn_one_hot(jnp_array(c(0L, 1L, 2L)), 2.5), "cannot be interpreted as an integer",
               class = "jax_type_error")
  pc_expect(nn_one_hot(jnp_array(c(0L, 1L)), 2), "float32", c(2L, 2L), c(1, 0, 0, 1))
  with_jax_config("jax_enable_x64", TRUE, {
    expect_error(random_randint(random_key(2L), 1L, 0, 2^63, dtype = "int64"), class = "jax_overflow_error")
  })
})

test_that("lax constants, polygamma and whole number arguments follow JAX", {
  skip_if_no_backend()
  x5 <- as_jax(c(1, 2, 3, 4, 5), "float32")
  expect_error(lax_empty(2.5, "float32"), class = "jax_type_error")
  expect_error(lax_tile(x5, 1.5), class = "jax_type_error")
  expect_error(lax_approx_max_k(x5, 1.5), class = "jax_type_error")
  expect_error(lax_approx_min_k(x5, 1.5), class = "jax_type_error")
  expect_error(lax_split(x5, c(1.5, 4)), class = "jax_value_error")
  pc_expect(lax_tile(x5, 2), "float32", 10L, rep(c(1, 2, 3, 4, 5), 2))
  # The pole of digamma through polygamma of order zero.
  z <- as_jax(0, "float32")
  pc_expect(lax_polygamma(z, z), "float32", integer(0), -Inf)
  pc_expect(jit(function(v) scipy_special_polygamma(0L, v))(z), "float32", integer(0), -Inf)
  # A large static dimension keeps its int64 without 64 bit mode.
  pc_expect(lax_shape_as_value(2147483648), "int64", 1L, 2147483648)
  pc_expect(lax_shape_as_value(c(2L, 3L)), "int32", 2L, c(2, 3))
  with_jax_config("jax_enable_x64", TRUE, {
    # Constants made for a weakly typed array are strongly typed (lax._const).
    w <- lax_full(2L, 2)
    p <- jit(function(v) lax_polynomial(v, list(1e40)))(w)
    pc_expect(p + as_jax(c(0, 0), "float32"), "float64", 2L, c(1e40, 1e40))
    pc_expect(lax_polynomial(2, list(1, 3)) + as_jax(1, "float32"), "float32", integer(0), 8)
  })
})

test_that("typed ranges, wide exponents, products and weak cotangents follow JAX", {
  skip_if_no_backend()
  num <- function(x) as.numeric(as.array(x))
  # A whole R number next to a typed bound is an integer in the length arithmetic.
  pc_expect(jnp_arange(as_jax(16777216L, "int32"), 16777219, 1L), "int32", 3L,
            c(16777216, 16777217, 16777218))
  pc_expect(jnp_arange(as_jax(16777216L, "int32"), 16777219), "int32", 3L,
            c(16777216, 16777217, 16777218))

  with_jax_config("jax_enable_x64", TRUE, {
    # Exponents beyond 2^53 are used exactly (the values are those of jnp.power).
    n <- lax_add(as_jax(2^53, "int64"), as_jax(1L, "int64"))
    pw <- function(b, dt = "int64") jnp_power(as_jax(b, dt), n)
    expect_identical(num(pw(-1L)), -1)
    expect_identical(num(pw(0L)), 0)
    expect_identical(num(pw(1L)), 1)
    expect_identical(num(pw(2L)), 0)
    expect_identical(dtype(pw(3L)), "int64")
    expect_identical(num(jnp_remainder(pw(3L), 1000000L)), 406147)    # 6593269854470406147
    expect_identical(num(lax_add(pw(3L), pw(-3L))), 0)
    expect_identical(num(pw(3L, "int32")), 3)
    expect_identical(num(pw(-1, "float64")), -1)
    expect_identical(num(pw(-2, "float64")), -Inf)
    expect_identical(num(pw(1.0000000001, "float64")), Inf)
    u <- lax_sub(as_jax(0L, "uint64"), as_jax(1L, "uint64"))
    expect_identical(num(jnp_remainder(jnp_power(as_jax(3L, "uint64"), u), as_jax(1000000L, "uint64"))),
                     34411)                                             # 12297829382473034411
    expect_identical(num(jit(function(x, n) jnp_power(x, n))(as_jax(-1L, "int64"), n)), -1)
    expect_error(jnp_power(as_jax(-1L, "int64"), lax_neg(n)),
                 "Integers cannot be raised to negative powers", class = "jax_type_error")
    expect_identical(num(jnp_power(as_jax(-1, "float64"), lax_neg(n))), -1)

    # custom_vjp passes the cotangent of the rule on unchanged.
    f <- custom_vjp(function(x) x, function(x) list(x, NULL), function(res, g) list(1 / 3))
    g <- grad(f)(as_jax(2, "float16"))
    expect_identical(dtype(g), "float64")
    expect_true(jaxR:::aval_of(g)$weak)
    expect_identical(num(g), 1 / 3)

    # The gradient of a checkpointed function keeps its weak type under jit.
    g <- jit(grad(checkpoint(function(x) x^3L)))(2)
    expect_true(jaxR:::aval_of(g)$weak)
    expect_identical(dtype(g + as_jax(1, "float16")), "float16")
    expect_identical(num(g), 12)
  })

  # jnp.dot and jnp.matmul prefer the result type of their inputs: an integer product
  # stays an integer in a Taylor series with floating coefficients, as in JAX.
  x <- as_jax(c(2L, 3L))
  s <- as_jax(c(0.5, 0.5))
  r <- experimental_jet_jet(jnp_dot, list(x, x), list(list(s), list(s)))
  expect_identical(c(num(r[[1L]]), num(r[[2L]][[1L]])), c(13, 0))
  pc_expect(jnp_dot(x, s), "float32", integer(0), 2.5)
  pc_expect(jnp_dot(as_jax(2L), as_jax(c(1.5, 2))), "float32", 2L, c(3, 4))
  pc_expect(jnp_dot(jnp_ones(c(2L, 2L), "int32"), s), "float32", 2L, c(1, 1))
  pc_expect(jnp_matmul(jnp_ones(c(2L, 2L), "int32"), s), "float32", 2L, c(1, 1))
  pc_expect(jnp_matmul(x, x), "int32", integer(0), 13)
  expect_true(jaxR:::aval_of(jnp_dot(2L, 3L))$weak)
  expect_identical(num(jnp_dot(2L, 3L)), 6)
  pc_expect(jnp_dot(x, x, preferred_element_type = "float32"), "float32", integer(0), 13)

  # Formats without an infinity use their extreme finite value as the identity.
  x4 <- as_jax(c(2, 1), "float4_e2m1fn")
  expect_identical(num(jnp_astype(lax_reduce_min(x4, 0L), "float32")), 1)
  expect_identical(num(jnp_astype(jit(function(y) lax_reduce_min(y, 0L))(x4), "float32")), 1)
  expect_identical(num(jnp_astype(lax_reduce_max(lax_neg(x4), 0L), "float32")), -1)
  expect_identical(num(lax_argmin(x4, 0L, "int32")), 1)
  expect_identical(num(lax_argmax(lax_neg(x4), 0L, "int32")), 1)

  # A fill that is not a whole number is a Python float: it saturates. A whole number
  # is a Python int and wraps.
  z8 <- as_jax(0L, "int8")
  pc_expect(lax_full_like(z8, 300.7), "int8", integer(0), 127)
  pc_expect(lax_full_like(z8, -129.5), "int8", integer(0), -128)
  pc_expect(lax_full_like(z8, 2.9), "int8", integer(0), 2)
  pc_expect(lax_full_like(z8, 128), "int8", integer(0), -128)
  pc_expect(lax_full_like(as_jax(0L, "uint8"), -1), "uint8", integer(0), 255)

  # NULL in in_nzs matches an empty part of the arguments.
  r <- linearize(function(x, unused) x * x, 2, NULL, in_nzs = list(TRUE, NULL))
  expect_identical(num(r[[2L]](1, NULL)), 4)

  # A named array called arr_ does not clash with automatic names when there are none.
  d <- tempfile("npz")
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  jnp_savez(file.path(d, "named.npz"), arr_ = as_jax(1))
  expect_identical(names(jnp_load(file.path(d, "named.npz"))), "arr_")
})

test_that("named arguments, narrow window identities, jet weak types and containers follow JAX", {
  skip_if_no_backend()
  num <- function(x) as.numeric(as.array(x))
  # Named arguments are matched to the formals before flattening.
  f <- custom_vjp(function(x) x * x, function(x) list(x * x, x), function(r, g) list(2 * r * g))
  expect_identical(num(grad(function(x) f(x = x))(2)), 4)
  h <- custom_jvp(function(x, y) x * y)
  h <- custom_jvp_defjvp(h, function(p, t) list(p[[1L]] * p[[2L]], p[[2L]] * t[[1L]] + p[[1L]] * t[[2L]]))
  expect_identical(num(grad(function(x) h(y = 3, x = x))(2)), 3)
  # Missing arguments get their default, as in inspect.Signature.apply_defaults.
  fd <- custom_vjp(function(x, y = 3) x * y, function(x, y = 3) list(x * y, list(x, y)),
                   function(r, g) list(r[[2L]] * g, r[[1L]] * g))
  expect_identical(num(grad(function(x) fd(x = x))(2)), 3)
  expect_identical(num(grad(function(x) fd(x))(2)), 3)
  expect_identical(num(grad(function(x) fd(y = 4, x = x))(2)), 4)
  gn <- custom_vjp(function(n, x) x * n, nondiff_argnums = 0L)
  gn <- custom_vjp_defvjp(gn, function(n, x) list(x * n, NULL), function(n, r, ct) list(ct * n))
  expect_identical(num(grad(function(x) gn(x = x, n = 5))(2)), 5)

  # Window reductions of formats without an infinity use the finite identity.
  x8 <- as_jax(c(1, 2, 3), "float8_e4m3fn")
  r <- lax_reduce_window(x8, as_jax(-448, "float8_e4m3fn"), lax_max, 2L, 1L, "VALID")
  expect_identical(num(jnp_astype(r, "float32")), c(2, 3))
  r <- lax_reduce_window(x8, as_jax(448, "float8_e4m3fn"), lax_min, 2L, 1L, "VALID")
  expect_identical(num(jnp_astype(r, "float32")), c(1, 2))

  # Taylor terms are weak for integer constants and integer powers, strong for floats.
  with_jax_config("jax_enable_x64", TRUE, {
    weak <- function(f) {
      r <- experimental_jet_jet(f, list(2), list(list(1, 0.5)))
      vapply(r[[2L]], function(t) jaxR:::aval_of(t)$weak, NA)
    }
    expect_identical(weak(function(x) x^3), c(TRUE, TRUE))
    expect_identical(weak(function(x) x + 1L), c(TRUE, TRUE))
    expect_identical(weak(function(x) x * 1L), c(TRUE, TRUE))
    expect_identical(weak(function(x) 1L - x), c(TRUE, TRUE))
    expect_identical(weak(function(x) x * 2.5), c(FALSE, FALSE))
    r <- experimental_jet_jet(function(x) x^3, list(2), list(list(1, 0.5)))
    expect_identical(num(r[[2L]][[1L]]), 12)
    expect_equal(num(r[[2L]][[2L]]), 18, tolerance = 1e-14)   # JAX: 17.99999999999999
    expect_identical(dtype(r[[2L]][[1L]] + as_jax(1, "float16")), "float16")
  })

  # Registered containers are prefixes in in_nzs; NULL marks an empty argument.
  g <- vjp(function(x, y) x * x, 2, NULL, in_nzs = list(TRUE, NULL))[[2L]](1)
  expect_identical(num(g[[1L]]), 4)
  expect_null(g[[2L]])
  fn <- function(a, b) a + b
  r <- linearize(function(q) attr(q, "args")[[1L]] * 2, jax_Partial(fn, 1),
                 in_nzs = list(jax_Partial(fn, TRUE)))
  expect_identical(num(r[[2L]](jax_Partial(fn, 1))), 2)

  # The cotangent of a custom rule keeps its type; fractional fills saturate.
  with_jax_config("jax_enable_x64", FALSE, {
    fc <- custom_vjp(function(x) x, function(x) list(x, NULL), function(r, g) list(1 + 2i))
    gc <- grad(fc)(as_jax(2, "float32"))
    expect_identical(dtype(gc), "complex64")
  })
  pc_expect(lax_full_like(as_jax(0L, "int8"), 128.5), "int8", integer(0), 127)
  pc_expect(lax_full_like(as_jax(0L, "uint8"), -1.9), "uint8", integer(0), 0)
})

test_that("vjp saveable_args, args_res and out_nzs follow JAX", {
  skip_if_no_backend()
  num <- function(x) as.numeric(as.array(x))
  a <- as_jax(2, "float32")
  b <- as_jax(3, "float32")
  r <- vjp(function(x, y) x * y, a, b)
  expect_identical(lapply(r[[2L]]$args_res, num), list(2, 3))
  expect_true(r[[2L]]$out_nzs)
  # R numbers are new arrays, like Python scalars: nothing is saved verbatim.
  r <- vjp(function(x, y) x * y, 2, 3)
  expect_true(all(vapply(r[[2L]]$args_res, inherits, NA, "jax_NotNeeded")))
  expect_identical(lapply(r[[2L]](1), num), list(3, 2))
  r <- vjp(function(x) sin(x), a)
  expect_s3_class(r[[2L]]$args_res[[1L]], "jax_NotNeeded")
  r <- vjp(function(x) sin(x) * x, a)
  expect_identical(num(r[[2L]]$args_res[[1L]]), 2)

  # An argument that is not saveable must be restored before the call.
  r <- vjp(function(x, y) x * y, a, b, saveable_args = list(TRUE, FALSE))
  f <- r[[2L]]
  expect_s3_class(f$args_res[[2L]], "jax_NotSaveable")
  expect_error(f(as_jax(1, "float32")), "position args\\[1\\]", class = "jax_value_error")
  f$args_res[[2L]] <- as_jax(10, "float32")
  expect_identical(lapply(f(as_jax(1, "float32")), num), list(10, 2))
  r <- vjp(function(d) d$a * d$b, list(a = a, b = b), saveable_args = FALSE)
  f <- r[[2L]]
  expect_true(all(vapply(f$args_res[[1L]], inherits, NA, "jax_NotSaveable")))
  e <- tryCatch(f(as_jax(1, "float32")), jax_value_error = conditionMessage)
  expect_match(e, "args[0]['a']", fixed = TRUE)
  expect_match(e, "args[0]['b']", fixed = TRUE)
  f$args_res <- list(list(a = a, b = b))
  expect_identical(lapply(f(as_jax(1, "float32"))[[1L]], num), list(a = 3, b = 2))
  expect_error(vjp(function(x) x, a, saveable_args = list(TRUE, FALSE)), class = "jax_value_error")

  # out_nzs mirrors the outputs; in_nzs makes tangents symbolically zero.
  r <- vjp(function(x, y) list(x * 2, jnp_astype(y, "int32"), list(a = x)), a, b)
  expect_identical(r[[2L]]$out_nzs, list(TRUE, FALSE, list(a = TRUE)))
  r <- vjp(function(x, y) list(x * 2, y), 2, 3, in_nzs = list(TRUE, FALSE))
  expect_identical(r[[2L]]$out_nzs, list(TRUE, FALSE))
  expect_identical(lapply(r[[2L]](list(1, 1)), num), list(2, 0))
  l <- linearize(function(x, y) list(x * 2, y), 2, 3, in_nzs = list(TRUE, FALSE))
  expect_identical(l[[2L]]$out_nzs, list(TRUE, FALSE))

  # The VJP function sees the output types of its own call (jacrev runs it under vmap).
  g <- function(x) jnp_stack(list(x[0] * x[1], jnp_sin(x[2]), jnp_sum(x^2)))
  x <- as_jax(c(1, 2, 3), "float32")
  expect_equal(num(jacrev(g)(x)), num(jacfwd(g)(x)), tolerance = 1e-6)

  # An argument that is not saveable is not kept by the VJP function.
  released <- FALSE
  keep <- local({
    big <- as_jax(c(3, 4), "float32")
    reg.finalizer(big, function(e) released <<- TRUE)
    vjp(function(x, y) x * y, as_jax(c(1, 2), "float32"), big,
        saveable_args = list(TRUE, FALSE))[[2L]]
  })
  invisible(gc())
  invisible(gc())
  expect_true(released)
  keep$args_res[[2L]] <- as_jax(c(3, 4), "float32")
  expect_identical(num(keep(as_jax(c(1, 1), "float32"))[[1L]]), c(3, 4))
})

test_that("mixed products, wide power gradients, fills, ranges and weak types follow JAX", {
  skip_if_no_backend()
  num <- function(x) as.numeric(as.array(x))
  with_jax_config("jax_enable_x64", FALSE, {
    # jnp_dot and jnp_matmul keep weak float operands of integer arrays.
    a <- as_jax(c(1L, 2L))
    pc_expect(jnp_dot(a, 0.5), "float32", 2L, c(0.5, 1))
    pc_expect(jnp_matmul(a, jnp_broadcast_to(0.5, 2L)), "float32", integer(0), 1.5)
    expect_identical(num(grad(function(y) jnp_sum(jnp_dot(a, y)))(0.5)), 3)
    # Float8 formats are ranked by their largest exponent.
    r <- jnp_dot(as_jax(c(1, 2), "float8_e4m3b11fnuz"), as_jax(c(1, 2), "float8_e4m3fnuz"),
                 preferred_element_type = "float32")
    pc_expect(r, "float32", integer(0), 5)
    # A fractional fill in the range is truncated from the R double; outside it saturates,
    # and under jit the folded conversion wraps.
    pc_expect(lax_full_like(as_jax(0L, "int32"), 2147483646.5), "int32", integer(0), 2147483646)
    pc_expect(lax_full_like(as_jax(0L, "int32"), 16777217.9), "int32", integer(0), 16777217)
    pc_expect(jit(function(x) lax_full_like(x, 300.7))(as_jax(0L, "int8")), "int8", integer(0), 44)
    pc_expect(jit(function() lax_full_like(as_jax(0L, "int8"), 255.9))(), "int8", integer(0), -1)
    pc_expect(lax_full_like(as_jax(0L, "int8"), 300.7), "int8", integer(0), 127)
    # Under grad and vjp the conversion of the constant is not folded: it saturates.
    r <- vjp(function(x) list(x, lax_full_like(as_jax(0L, "int8"), 300.7)), 1)[[1L]][[2L]]
    pc_expect(r, "int8", integer(0), 127)
    # Residuals of cond and carried cotangents of scan keep their weak types.
    f <- checkpoint(function(x) lax_cond(x > 0, function(x) x * x, function(x) x * 3L, x))
    g <- jit(grad(f))(2)
    expect_true(jaxR:::aval_of(g)$weak)
    expect_identical(num(g), 4)
    h <- function(x) lax_scan(function(carry, y) list(carry * y, NULL), x,
                              as_jax(c(400, 400), "float32"))[[1L]]
    g <- jit(grad(h))(2)
    expect_false(jaxR:::aval_of(g)$weak)
    expect_identical(num(g + as_jax(1, "float16")), 160001)
    # Constants of lax functions and of jnp_stack get strongly typed Taylor terms.
    t <- experimental_jet_jet(function(x) lax_add(x, lax_full_like(x, as_jax(1L))),
                              list(2), list(list(70000)))[[2L]][[1L]]
    expect_identical(dtype(t + as_jax(1, "float16")), "float32")
    t <- experimental_jet_jet(function(x) jnp_add(x, lax_full_like(x, as_jax(1L))),
                              list(2), list(list(70000)))[[2L]][[1L]]
    expect_identical(dtype(t + as_jax(1, "float16")), "float32")
    t <- experimental_jet_jet(function(x) jnp_stack(list(x, 1L)), list(2),
                              list(list(100000)))[[2L]][[1L]]
    expect_false(jaxR:::aval_of(t)$weak)
  })
  with_jax_config("jax_enable_x64", TRUE, {
    # The derivative of a wide integer power is y * x^(y - 1).
    n <- lax_add(as_jax(2^53, "int64"), as_jax(1L, "int64"))
    x <- 1 + .Machine$double.eps
    expect_equal(num(grad(function(x) jnp_power(x, n))(x)), 6.6554699594620056e+16, tolerance = 1e-12)
    # Whole R numbers beyond the int32 range are int64 bounds of a typed range.
    start <- lax_add(as_jax(2^53, "int64"), as_jax(1L, "int64"))
    r <- jnp_arange(start, 2^53 + 4, 1L)
    expect_identical(dim(r), 3L)
    expect_identical(num(lax_sub(r, start)), c(0, 1, 2))
  })
  # Arguments are bound like inspect.Signature.bind with defaults applied.
  f <- custom_vjp(function(x = 2, y = 3) x * y, function(x = 2, y = 3) list(x * y, list(x, y)),
                  function(r, g) list(r[[2L]] * g, r[[1L]] * g))
  expect_identical(num(grad(function(y) f(y = y))(4)), 2)
  f3 <- custom_vjp(function(x = 2, y = 3, z = 4) x * y * z,
                   function(x = 2, y = 3, z = 4) list(x * y * z, list(x, y, z)),
                   function(r, g) list(g * r[[2L]] * r[[3L]], g * r[[1L]] * r[[3L]], g * r[[1L]] * r[[2L]]))
  expect_identical(num(grad(function(z) f3(x = 2, z = z))(4)), 6)
  fd <- custom_jvp(function(x, ...) x * 2)
  fd <- custom_jvp_defjvp(fd, function(p, t) list(p[[1L]] * 2, t[[1L]] * 2))
  expect_identical(num(grad(function(x) fd(x = x))(1)), 2)
  # A registered container is a prefix of any container with as many children.
  register_pytree_node("jaxr_test_Box", function(q) list(children = list(q$x), aux = NULL),
                       function(aux, kids) structure(list(x = kids[[1L]]), class = "jaxr_test_Box"))
  prefix <- structure(list(x = TRUE), class = "jaxr_test_Box")
  l <- linearize(function(q) q[[1L]] * 2, list(1), in_nzs = list(prefix))
  expect_identical(num(l[[2L]](list(1))), 2)
})
