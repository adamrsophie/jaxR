# jax.numpy set operations (R/numpy_setops.R). Expected values from jax 0.12.0.1; each
# entry is list(value, dtype).

set_exp <- list(
  u = list(c(1, 2, 3, 4, 5, 6, 9), "int32"),
  u_all1 = list(c(1, 2, 3, 4, 5, 6, 9), "int32"),
  u_all2 = list(c(1, 6, 0, 2, 4, 7, 5), "int32"),
  u_all3 = list(c(2, 0, 3, 0, 4, 6, 1, 5, 4, 2), "int32"),
  u_all4 = list(c(2, 1, 2, 1, 2, 1, 1), "int32"),
  u_size1 = list(c(1, 2, 3, 4, 5, 6, 9, -1, -1), "int32"),
  u_size2 = list(c(1, 6, 0, 2, 4, 7, 5, 1, 1), "int32"),
  u_size3 = list(c(2, 0, 3, 0, 4, 6, 1, 5, 4, 2), "int32"),
  u_size4 = list(c(2, 1, 2, 1, 2, 1, 1, 0, 0), "int32"),
  u_trunc1 = list(c(1, 2, 3, 4), "int32"),
  u_trunc2 = list(c(2, 1, 2, 1), "int32"),
  u_nofill = list(c(1, 2, 3, 4, 5, 6, 9, 1, 1), "int32"),
  u_f = list(c(-1, 0, 2.5, 7, NaN), "float32"),
  u_f_noeq1 = list(c(-1, 0, 2.5, 7, NaN, NaN), "float32"),
  u_f_noeq2 = list(c(1, 2, 2, 1, 1, 1), "int32"),
  u_m1 = list(array(c(0, 1, 3, 5, 2, 4), dim = c(3, 2)), "float32"),
  u_m2 = list(c(3, 0, 1), "int32"),
  u_m3 = list(c(1, 2, 1, 0, 2), "int32"),
  u_m4 = list(c(1, 2, 2), "int32"),
  u_col = list(array(c(1, 3, 1, 0, 3, 2, 4, 2, 5, 4), dim = c(5, 2)), "float32"),
  u_m31 = list(array(c(0, 1, 1, 2, 1, 0, 1, 2), dim = c(2, 2, 2)), "int32"),
  u_m32 = list(c(1, 1, 0), "int32"),
  u_m3_flat_inv1 = list(c(0, 1, 2), "int32"),
  u_m3_flat_inv2 = list(array(c(1, 1, 0, 2, 2, 1, 0, 0, 1, 2, 2, 1), dim = c(3, 2, 2)), "int32"),
  u_m_size = list(array(c(0, 1, 3, 0, 0, 5, 2, 4, 0, 0), dim = c(5, 2)), "float32"),
  u_bool = list(c(FALSE, TRUE), "bool"),
  ua1 = list(c(-1, 0, 2.5, 7, NaN, NaN), "float32"),
  ua2 = list(c(2, 5, 0, 7, 1, 4), "int32"),
  ua3 = list(c(2, 4, 0, 2, 5, 1, 1, 3), "int32"),
  ua4 = list(c(1, 2, 2, 1, 1, 1), "int32"),
  uc1 = list(c(1, 2, 3, 4, 5, 6, 9), "int32"),
  uc2 = list(c(2, 1, 2, 1, 2, 1, 1), "int32"),
  ui1 = list(c(1, 2, 3), "int32"),
  ui2 = list(array(c(1, 0, 0, 2), dim = c(2, 2)), "int32"),
  uv = list(c(-1, 0, 2.5, 7, NaN, NaN), "float32"),
  ua_size1 = list(c(1, 2, 3, 4, 5, 6, 9, 0), "int32"),
  ua_size2 = list(c(1, 6, 0, 2, 4, 7, 5, 1), "int32"),
  ua_size3 = list(c(2, 0, 3, 0, 4, 6, 1, 5, 4, 2), "int32"),
  ua_size4 = list(c(2, 1, 2, 1, 2, 1, 1, 0), "int32"),
  u_empty = list(c(7, 7, 7), "float32"),
  u_empty_counts1 = list(numeric(0), "float32"),
  u_empty_counts2 = list(numeric(0), "int32"),
  union = list(c(0, 1, 3, 5, 7, 8), "int32"),
  union_f = list(c(-2, 1.5, 3, 4), "float32"),
  union_size = list(c(0, 1, 3, 5, 7, 8, -1, -1), "int32"),
  inter = list(c(3, 5), "int32"),
  inter_idx1 = list(c(3, 5), "int32"),
  inter_idx2 = list(c(1, 2), "int32"),
  inter_idx3 = list(c(1, 0), "int32"),
  inter_au1 = list(c(3, 5), "int32"),
  inter_au2 = list(c(1, 2), "int32"),
  inter_au3 = list(c(1, 0), "int32"),
  inter_size1 = list(c(3, 5, 3, 3), "int32"),
  inter_size2 = list(c(1, 2, 6, 6), "int32"),
  inter_size3 = list(c(1, 0, 4, 4), "int32"),
  inter_size_fill = list(c(3, 5, -9, -9), "int32"),
  inter_size_au1 = list(c(3, 5, 3), "int32"),
  inter_size_au2 = list(c(1, 2, 3), "int32"),
  inter_size_au3 = list(c(1, 0, 3), "int32"),
  inter_f = list(c(-2, 3), "float32"),
  diff = list(c(1, 8), "int32"),
  diff_size = list(c(1, 8, 0, 0), "int32"),
  diff_size_fill = list(c(1, 8, -1, -1), "int32"),
  diff_au = list(c(1, 8), "int32"),
  diff_au_size = list(c(1, 8, 0), "int32"),
  diff_f = list(c(1.5), "float32"),
  xor = list(c(0, 1, 7, 8), "int32"),
  xor_size = list(c(0, 1, 7, 8, 0, 0), "int32"),
  xor_size_fill = list(c(0, 1, 7, 8, -1, -1), "int32"),
  xor_au_size = list(c(1, 7, 8, 1, 1), "int32"),
  xor_f = list(c(1.5, 4), "float32"),
  isin_compare_all = list(array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE), dim = c(2, 3)), "bool"),
  isin_inv_compare_all = list(array(c(TRUE, FALSE, FALSE, FALSE, FALSE, TRUE), dim = c(2, 3)), "bool"),
  isin_binary_search = list(array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE), dim = c(2, 3)), "bool"),
  isin_inv_binary_search = list(array(c(TRUE, FALSE, FALSE, FALSE, FALSE, TRUE), dim = c(2, 3)), "bool"),
  isin_sort = list(array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE), dim = c(2, 3)), "bool"),
  isin_inv_sort = list(array(c(TRUE, FALSE, FALSE, FALSE, FALSE, TRUE), dim = c(2, 3)), "bool"),
  isin_f = list(c(TRUE, FALSE, TRUE, TRUE), "bool"),
  isin_empty = list(array(c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE), dim = c(2, 3)), "bool"),
  jit_u1 = list(c(1, 2, 3, 4, 5, 6, 9, 0), "int32"),
  jit_u2 = list(c(1, 6, 0, 2, 4, 7, 5, 1), "int32"),
  jit_u3 = list(c(2, 0, 3, 0, 4, 6, 1, 5, 4, 2), "int32"),
  jit_u4 = list(c(2, 1, 2, 1, 2, 1, 1, 0), "int32"),
  jit_union = list(c(0, 1, 3, 5, 7, 8, 0, 0), "int32"),
  vmap_u = list(array(c(1, 5, 0, 2, -1, 1, 3, -1, 2, -1, -1, 3), dim = c(3, 4)), "int32"),
  vmap_isin = list(array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE), dim = c(2, 3)), "bool"),
  vmap_inter = list(array(c(2, 8, 4, -1, -1, -1), dim = c(2, 3)), "int32"),
  x64_u1 = list(c(1, 2, 3, 4, 5, 6, 9), "int64"),
  x64_u2 = list(c(1, 6, 0, 2, 4, 7, 5), "int64"),
  x64_u3 = list(c(2, 0, 3, 0, 4, 6, 1, 5, 4, 2), "int64"),
  x64_u4 = list(c(2, 1, 2, 1, 2, 1, 1), "int64"),
  x64_inter_idx1 = list(c(3, 5), "int64"),
  x64_inter_idx2 = list(c(1, 2), "int64"),
  x64_inter_idx3 = list(c(1, 0), "int64"),
  x64_inter_au_idx1 = list(c(3, 5), "int64"),
  x64_inter_au_idx2 = list(c(1, 2), "int32"),
  x64_inter_au_idx3 = list(c(1, 0), "int32"),
  x64_f1 = list(c(-1, 0, 0x1.4000000000000p+1, 7, NaN), "float64"),
  x64_f2 = list(c(2, 4, 0, 2, 4, 1, 1, 3), "int64"),
  g_unique = list(c(3, 2, 0, 4, 1, 0), "float64"),
  g_union = list(c(4, 2, 6), "float64"),
  end = NULL
)

set_x <- function() as_jax(c(3L, 1L, 4L, 1L, 5L, 9L, 2L, 6L, 5L, 3L))
set_f <- function() as_jax(c(2.5, NaN, -1, 2.5, NaN, 0, -0, 7))
set_m <- function() as_jax(matrix(c(1, 2, 3, 4, 1, 2, 0, 5, 3, 4), 5, byrow = TRUE))
set_m3 <- function() as_jax(array(c(1L, 1L, 0L, 2L, 2L, 1L, 0L, 0L, 1L, 2L, 2L, 1L), c(3L, 2L, 2L)))
set_a1 <- function() as_jax(c(1L, 3L, 5L, 3L, 8L, 8L))
set_a2 <- function() as_jax(c(5L, 3L, 7L, 0L))
set_fa <- c(1.5, -2, 3, 1.5)
set_fb <- c(3, 4, -2)
set_el <- function() as_jax(matrix(c(1L, 3L, 5L, 7L, 0L, 8L), 2, byrow = TRUE))

# Compare one result (array or list of arrays) with set_exp entries `name` or `name1`, ...
set_check <- function(name, got) {
  if (!is.list(got)) got <- list(got)
  keys <- if (length(got) == 1L && !is.null(set_exp[[name]])) name else paste0(name, seq_along(got))
  for (i in seq_along(got)) {
    e <- set_exp[[keys[i]]]
    expect_identical(dtype(got[[i]]), e[[2L]], info = keys[i])
    g <- as.array(got[[i]])
    expect_equal(as.vector(g), as.vector(e[[1L]]), info = keys[i])
    if (length(dim(e[[1L]])) > 1L) expect_identical(as.integer(dim(g)), as.integer(dim(e[[1L]])), info = keys[i])
  }
}

set_cases <- list(
  u = function() jnp_unique(set_x()),
  u_all = function() jnp_unique(set_x(), return_index = TRUE, return_inverse = TRUE, return_counts = TRUE),
  u_size = function() jnp_unique(set_x(), return_index = TRUE, return_inverse = TRUE, return_counts = TRUE,
                                 size = 9L, fill_value = -1L),
  u_trunc = function() jnp_unique(set_x(), return_counts = TRUE, size = 4L),
  u_nofill = function() jnp_unique(set_x(), size = 9L),
  u_f = function() jnp_unique(set_f()),
  u_f_noeq = function() jnp_unique(set_f(), equal_nan = FALSE, return_counts = TRUE),
  u_m = function() jnp_unique(set_m(), axis = 0L, return_index = TRUE, return_inverse = TRUE,
                              return_counts = TRUE),
  u_col = function() jnp_unique(set_m(), axis = 1L),
  u_m3 = function() jnp_unique(set_m3(), axis = 0L, return_inverse = TRUE),
  u_m3_flat_inv = function() jnp_unique(set_m3(), return_inverse = TRUE),
  u_m_size = function() jnp_unique(set_m(), axis = 0L, size = 5L, fill_value = 0),
  u_bool = function() jnp_unique(c(TRUE, FALSE, TRUE)),
  ua = function() jnp_unique_all(set_f()),
  uc = function() jnp_unique_counts(set_x()),
  ui = function() jnp_unique_inverse(matrix(c(2L, 1L, 1L, 3L), 2, byrow = TRUE)),
  uv = function() jnp_unique_values(set_f()),
  ua_size = function() jnp_unique_all(set_x(), size = 8L, fill_value = 0L),
  u_empty = function() jnp_unique(jnp_zeros(0L), size = 3L, fill_value = 7),
  u_empty_counts = function() jnp_unique(jnp_zeros(0L), return_counts = TRUE),
  union = function() jnp_union1d(set_a1(), set_a2()),
  union_f = function() jnp_union1d(set_fa, set_fb),
  union_size = function() jnp_union1d(set_a1(), set_a2(), size = 8L, fill_value = -1L),
  inter = function() jnp_intersect1d(set_a1(), set_a2()),
  inter_idx = function() jnp_intersect1d(set_a1(), set_a2(), return_indices = TRUE),
  inter_au = function() jnp_intersect1d(c(1L, 3L, 5L), c(5L, 3L, 7L), assume_unique = TRUE,
                                        return_indices = TRUE),
  inter_size = function() jnp_intersect1d(set_a1(), set_a2(), return_indices = TRUE, size = 4L),
  inter_size_fill = function() jnp_intersect1d(set_a1(), set_a2(), size = 4L, fill_value = -9L),
  inter_size_au = function() jnp_intersect1d(c(1L, 3L, 5L), c(5L, 3L, 7L), assume_unique = TRUE,
                                             return_indices = TRUE, size = 3L),
  inter_f = function() jnp_intersect1d(set_fa, set_fb),
  diff = function() jnp_setdiff1d(set_a1(), set_a2()),
  diff_size = function() jnp_setdiff1d(set_a1(), set_a2(), size = 4L),
  diff_size_fill = function() jnp_setdiff1d(set_a1(), set_a2(), size = 4L, fill_value = -1L),
  diff_au = function() jnp_setdiff1d(c(1L, 3L, 5L, 8L), c(5L, 3L, 7L), assume_unique = TRUE),
  diff_au_size = function() jnp_setdiff1d(c(1L, 3L, 5L, 8L), c(5L, 3L, 7L), assume_unique = TRUE,
                                          size = 3L, fill_value = 0L),
  diff_f = function() jnp_setdiff1d(set_fa, set_fb),
  xor = function() jnp_setxor1d(set_a1(), set_a2()),
  xor_size = function() jnp_setxor1d(set_a1(), set_a2(), size = 6L),
  xor_size_fill = function() jnp_setxor1d(set_a1(), set_a2(), size = 6L, fill_value = -1L),
  xor_au_size = function() jnp_setxor1d(c(1L, 3L, 5L, 8L), c(5L, 3L, 7L), assume_unique = TRUE, size = 5L),
  xor_f = function() jnp_setxor1d(set_fa, set_fb),
  isin_compare_all = function() jnp_isin(set_el(), set_a2(), method = "compare_all"),
  isin_inv_compare_all = function() jnp_isin(set_el(), set_a2(), method = "compare_all", invert = TRUE),
  isin_binary_search = function() jnp_isin(set_el(), set_a2(), method = "binary_search"),
  isin_inv_binary_search = function() jnp_isin(set_el(), set_a2(), method = "binary_search", invert = TRUE),
  isin_sort = function() jnp_isin(set_el(), set_a2(), method = "sort"),
  isin_inv_sort = function() jnp_isin(set_el(), set_a2(), method = "sort", invert = TRUE),
  isin_f = function() jnp_isin(set_fa, as_jax(c(1.5, 3), "float32")),
  isin_empty = function() jnp_isin(set_el(), jnp_zeros(0L, "int32"))
)

test_that("set operations match JAX", {
  skip_if_no_backend()
  for (nm in names(set_cases)) set_check(nm, set_cases[[nm]]())
  expect_named(jnp_unique_all(set_f()), c("values", "indices", "inverse_indices", "counts"))
  expect_named(jnp_unique_counts(set_x()), c("values", "counts"))
  expect_named(jnp_unique_inverse(set_x()), c("values", "inverse_indices"))
})

test_that("set operations with size work under jit and vmap", {
  skip_if_no_backend()
  set_check("jit_u", jit(function(v) jnp_unique(v, return_index = TRUE, return_inverse = TRUE,
                                                return_counts = TRUE, size = 8L, fill_value = 0L))(set_x()))
  set_check("jit_union", jit(function(a, b) jnp_union1d(a, b, size = 8L))(set_a1(), set_a2()))
  for (nm in c("inter_size", "inter_size_fill", "diff_size", "diff_size_fill", "xor_size",
               "xor_size_fill", "xor_au_size", "diff_au_size", "inter_size_au", "u_size", "ua_size",
               "isin_sort", "isin_binary_search")) {
    set_check(nm, jit(set_cases[[nm]])())
  }
  vu <- as_jax(matrix(c(1L, 2L, 2L, 1L, 3L, 5L, 5L, 5L, 5L, 5L, 4L, 3L, 2L, 1L, 0L), 3, byrow = TRUE))
  set_check("vmap_u", vmap(function(v) jnp_unique(v, size = 4L, fill_value = -1L))(vu))
  set_check("vmap_isin", vmap(function(v) jnp_isin(v, set_a2()))(set_el()))
  ia <- as_jax(matrix(c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L), 2, byrow = TRUE))
  ib <- as_jax(matrix(c(4L, 2L, 9L, 8L, 8L, 1L), 2, byrow = TRUE))
  set_check("vmap_inter", vmap(function(a, b) jnp_intersect1d(a, b, size = 3L, fill_value = -1L))(ia, ib))
})

test_that("set operations follow JAX index dtypes under x64", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  set_check("x64_u", jnp_unique(as_jax(set_x(), "int64"), return_index = TRUE, return_inverse = TRUE,
                                return_counts = TRUE))
  set_check("x64_inter_idx", jnp_intersect1d(as_jax(set_a1(), "int64"), as_jax(set_a2(), "int64"),
                                             return_indices = TRUE))
  set_check("x64_inter_au_idx", jnp_intersect1d(as_jax(c(1, 3, 5), "int64"), as_jax(c(5, 3, 7), "int64"),
                                                assume_unique = TRUE, return_indices = TRUE))
  set_check("x64_f", jnp_unique(as_jax(c(2.5, NaN, -1, 2.5, NaN, 0, -0, 7), "float64"), return_inverse = TRUE))
})

test_that("set operation gradients match JAX", {
  skip_if_no_backend()
  enable_x64()
  on.exit(enable_x64(FALSE))
  fu <- function(v) jnp_sum(jnp_unique(v, size = 6L, fill_value = 0) * jnp_arange(1, 7, dtype = "float64"))
  x <- as_jax(c(2, 1, 2, 3, 0.5, 1), "float64")
  expect_jax_equal(grad(fu)(x), set_exp$g_unique[[1L]])
  fn <- function(v) jnp_sum(jnp_union1d(v, c(1, 4), size = 5L, fill_value = 0)^2)
  expect_jax_equal(grad(fn)(as_jax(c(2, 1, 3), "float64")), set_exp$g_union[[1L]])
  # away from ties the outputs are a permutation of the inputs
  check_grad_fd(function(v) jnp_sum(jnp_union1d(v, c(1.5, 4), size = 5L, fill_value = 0)^2 * jnp_arange(1, 6, dtype = "float64")),
                as_jax(c(2, 1, 3), "float64"))
  check_grad_fd(function(v) jnp_sum(jnp_setdiff1d(v, c(1, 4), size = 2L)^2), as_jax(c(2.5, 1.2, 3), "float64"))
})

test_that("set operation errors match JAX", {
  skip_if_no_backend()
  expect_error(jit(jnp_unique)(set_x()), class = "jax_concretization_error")
  expect_error(jit(jnp_union1d)(set_a1(), set_a2()), class = "jax_concretization_error")
  expect_error(jit(jnp_setdiff1d)(set_a1(), set_a2()), class = "jax_concretization_error")
  expect_error(jit(jnp_intersect1d)(set_a1(), set_a2()), class = "jax_concretization_error")
  expect_error(jit(jnp_setxor1d)(set_a1(), set_a2()), class = "jax_concretization_error")
  expect_error(jit(function(v, s) jnp_unique(v, size = s))(set_x(), 3L), class = "jax_concretization_error")
  expect_error(jnp_unique(jnp_zeros(0L), size = 2L),
               "jnp.unique: for zero-sized input with nonzero size argument, fill_value must be specified",
               fixed = TRUE)
  expect_error(jnp_isin(set_el(), set_a2(), method = "foo"),
               "method='foo' is not implemented; options are 'compare_all', 'binary_search', 'sort', and 'auto'",
               fixed = TRUE)
  expect_error(jnp_unique(set_m(), axis = 2L), "out of bounds")
  expect_error(jnp_unique(set_x(), size = 2.5), "size must be an integer")
})
