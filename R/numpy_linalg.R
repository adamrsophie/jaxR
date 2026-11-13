# jax.numpy.linalg (port of jax/_src/numpy/linalg.py).
#
# JAX wraps most of these functions in jit; here they run primitive by primitive (the
# values are the same) and can be wrapped in jit() by the caller.

# ---- helpers ---------------------------------------------------------------------------

nl_inexact <- function(x) promote_dtypes_inexact(list(as_jax(x)))[[1L]]
nl_shape <- function(x) aval_of(x)$shape
nl_dtype <- function(x) aval_of(x)$dtype
nl_is_complex <- function(x) is_complex_dtype(aval_of(x)$dtype)
nl_num <- function(x, v) is.numeric(x) && length(x) == 1L && !is.na(x) && x == v

# Static 0-based indexing of the trailing dimensions (negative indices count from the
# end): a[..., i, j], a[..., i, :] and a[..., i].
nl_elem <- function(a, i, j) {
  lax_index_in_dim(lax_index_in_dim(a, i, axis = -2L, keepdims = FALSE), j, axis = -1L,
                   keepdims = FALSE)
}
nl_row <- function(a, i) lax_index_in_dim(a, i, axis = -2L, keepdims = FALSE)
nl_last <- function(a, i) lax_index_in_dim(a, i, axis = -1L, keepdims = FALSE)

# Port of jax/_src/numpy/linalg.py:_H.
nl_H <- function(x) jnp_conjugate(jnp_linalg_matrix_transpose(x))

# Port of jax/_src/numpy/linalg.py:_symmetrize.
nl_symmetrize <- function(x) (x + nl_H(x)) / 2

# jnp.diagonal(x, offset, axis1 = -2, axis2 = -1). JAX gathers the diagonal; here it is
# the row sum of the square block that holds it, masked by the identity (the method
# jnp.trace uses), so no gather primitive is needed.
nl_diagonal <- function(x, offset = 0L) {
  x <- as_jax(x)
  s <- nl_shape(x)
  r <- length(s)
  if (r < 2L)
    jax_error("diagonal requires an array of at least two dimensions.", class = "jax_value_error")
  offset <- as.integer(offset)
  k <- max(0L, min(s[r - 1L] + min(offset, 0L), s[r] - max(offset, 0L)))
  r0 <- max(-offset, 0L)
  c0 <- max(offset, 0L)
  batch <- s[seq_len(r - 2L)]
  if (k == 0L) return(lax_full(c(batch, 0L), 0, nl_dtype(x)))
  sub <- lax_slice(x, c(rep(0L, r - 2L), r0, c0), c(batch, r0 + k, c0 + k))
  masked <- lax_select(lax_broadcast(linalg_eye("bool", c(k, k)), batch), sub, lax_full_like(sub, 0))
  if (nl_dtype(x) == "bool") lax_reduce_or(masked, r - 1L) else lax_reduce_sum(masked, r - 1L)
}

# Port of jax/_src/numpy/lax_numpy.py:trace with axis1 = -2, axis2 = -1.
nl_trace <- function(x, offset = 0L, dtype = NULL) {
  x <- as_jax(x)
  s <- nl_shape(x)
  r <- length(s)
  if (r < 2L)
    jax_error("trace requires an array of at least two dimensions.", class = "jax_value_error")
  eye <- linalg_eye("bool", s[c(r - 1L, r)], as.integer(offset))
  jnp_sum(jnp_where(eye, x, jnp_zeros_like(x)), axis = c(-2L, -1L), dtype = dtype)
}

# reductions.amax(x, initial = 0) for non negative x.
nl_amax0 <- function(x, axis, keepdims = FALSE) {
  jnp_max(x, axis = axis, keepdims = keepdims, initial = 0)
}

named_result <- function(values, names) stats::setNames(values, names)

# ---- cholesky, svd ---------------------------------------------------------------------

#' Linear algebra (jax.numpy.linalg)
#'
#' Ports of the functions of `jax.numpy.linalg`. Inputs are promoted to an inexact
#' dtype and the matrix functions work on batches (leading dimensions). Functions that
#' return named tuples in JAX return named lists here: `jnp_linalg_svd()` gives
#' `list(U, S, Vh)`, `jnp_linalg_eigh()` and `jnp_linalg_eig()` give
#' `list(eigenvalues, eigenvectors)`, `jnp_linalg_qr()` gives `list(Q, R)` and
#' `jnp_linalg_slogdet()` gives `list(sign, logabsdet)`.
#'
#' @param a,x,M Input array.
#' @param b Right hand side.
#' @param x1,x2 Input arrays.
#' @param upper If `TRUE` return the upper triangular factor.
#' @param symmetrize_input If `TRUE` the matrix is symmetrized before the decomposition.
#' @param full_matrices Whether full or reduced matrices are returned.
#' @param compute_uv Whether to compute the singular vectors.
#' @param hermitian Assume a Hermitian matrix (uses the eigendecomposition).
#' @param subset_by_index Optional 0-based index range of singular values.
#' @param n Integer exponent.
#' @param rtol,tol,rcond Relative tolerance below which singular values count as zero
#'   (`tol` and `rcond` are older names of `rtol`).
#' @param method `"lu"` (default) or `"qr"`.
#' @param UPLO `"L"` or `"U"`: which triangle of the matrix is used.
#' @param ord Order of the norm.
#' @param axis Axis (0-based) or axes over which to operate.
#' @param keepdims Keep the reduced dimensions with size one.
#' @param mode `"reduced"`, `"complete"`, `"r"` or `"raw"`.
#' @param numpy_resid If `TRUE` the residuals follow the NumPy convention (empty unless
#'   the system is overdetermined and of full rank).
#' @param precision Precision of the underlying products.
#' @param preferred_element_type Optional result dtype.
#' @param axes Number of axes to contract, or `list(axes1, axes2)`; for
#'   `jnp_linalg_tensorsolve()` the axes of `a` to move to the end.
#' @param offset Offset of the diagonal.
#' @param ind Number of leading dimensions that form the rows of the inverted tensor.
#' @param arrays List of arrays.
#' @param p Order of the norm (see `jnp_linalg_norm()`).
#' @param dtype Optional result dtype.
#' @return An array or a (named) list of arrays.
#' @name jnp_linalg
NULL

#' @rdname jnp_linalg
#' @export
jnp_linalg_cholesky <- function(a, upper = FALSE, symmetrize_input = TRUE) {
  a <- as_jax(a)
  s <- nl_shape(a)
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L])
    jax_error("Argument to linalg.cholesky must have shape [..., n, n], got ", py_shape(s), ".",
              class = "jax_value_error")
  L <- lax_linalg_cholesky(nl_inexact(a), symmetrize_input = symmetrize_input)
  if (upper) nl_H(L) else L
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_svd <- function(a, full_matrices = TRUE, compute_uv = TRUE, hermitian = FALSE,
                           subset_by_index = NULL) {
  a <- nl_inexact(a)
  if (hermitian) {
    f <- lax_linalg_eigh(a, subset_by_index = subset_by_index)
    w <- f[[1L]]  # eigenvectors
    v <- f[[2L]]  # eigenvalues
    s <- lax_abs(v)
    last <- linalg_ndim(s) - 1L
    if (!compute_uv) return(lax_rev(lax_sort(s, dimension = -1L), last))
    sign <- jnp_where(v < 0, -1L, 1L)
    idxs <- lax_broadcasted_iota("uint32", nl_shape(s), last)
    sorted <- lax_sort(list(s, idxs, sign), dimension = -1L, num_keys = 1L)
    s <- lax_rev(sorted[[1L]], last)
    idxs <- lax_rev(sorted[[2L]], last)
    sign <- lax_rev(sorted[[3L]], last)
    u <- jnp_take_along_axis(w, lax_expand_dims(idxs, -2L), axis = -1L)
    vh <- nl_H(u * jnp_astype(lax_expand_dims(sign, -2L), nl_dtype(u)))
    return(list(U = u, S = s, Vh = vh))
  }
  res <- lax_linalg_svd(a, full_matrices = full_matrices, compute_uv = compute_uv,
                        subset_by_index = subset_by_index)
  if (compute_uv) named_result(res, c("U", "S", "Vh")) else res
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_svdvals <- function(x) jnp_linalg_svd(x, compute_uv = FALSE, hermitian = FALSE)

# ---- matrix_power, matrix_rank ---------------------------------------------------------

#' @rdname jnp_linalg
#' @export
jnp_linalg_matrix_power <- function(a, n) {
  arr <- as_jax(a)
  s <- nl_shape(arr)
  r <- length(s)
  if (r < 2L)
    jax_error(r, "-dimensional array given. Array must be at least two-dimensional",
              class = "jax_type_error")
  if (s[r - 1L] != s[r])
    jax_error("Last 2 dimensions of the array must be square", class = "jax_type_error")
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n != round(n))
    jax_error("exponent must be an integer, got ", format(n), class = "jax_type_error")
  n <- as.integer(n)
  if (n == 0L) return(jnp_broadcast_to(jnp_eye(s[r], dtype = nl_dtype(arr)), s))
  if (n < 0L) {
    arr <- jnp_linalg_inv(arr)
    n <- abs(n)
  }
  if (n == 1L) return(arr)
  if (n == 2L) return(arr %*% arr)
  if (n == 3L) return((arr %*% arr) %*% arr)
  z <- NULL
  result <- NULL
  while (n > 0L) {
    z <- if (is.null(z)) arr else z %*% z
    bit <- n %% 2L
    n <- n %/% 2L
    if (bit == 1L) result <- if (is.null(result)) z else result %*% z
  }
  result
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_matrix_rank <- function(M, rtol = NULL, hermitian = FALSE, tol = NULL) {
  M <- as_jax(M)
  if (!is.null(tol)) {
    if (!is.null(rtol))
      jax_error("matrix_rank: only one of tol or rtol may be specified.", class = "jax_value_error")
    rtol <- tol
  }
  M <- nl_inexact(M)
  if (linalg_ndim(M) < 2L) return(jnp_astype(jnp_any(M != 0), "int32"))
  S <- jnp_linalg_svd(M, full_matrices = FALSE, compute_uv = FALSE, hermitian = hermitian)
  if (is.null(rtol))
    rtol <- jnp_max(S, axis = -1L) * as_jax(max(utils::tail(nl_shape(M), 2L)), nl_dtype(S)) *
      finfo(nl_dtype(S))$eps
  rtol <- as_jax(rtol)
  rtol <- jnp_expand_dims(rtol, linalg_ndim(rtol))
  jnp_sum(S > rtol, axis = -1L)
}

# ---- slogdet and det -------------------------------------------------------------------

# Port of jax/_src/numpy/linalg.py:_slogdet_1x1 and _slogdet_2x2: closed forms.
slogdet_small_fn <- function(a) {
  n <- utils::tail(nl_shape(a), 1L)
  det <- if (n == 1L) {
    nl_elem(a, 0L, 0L)
  } else if (n == 2L) {
    nl_elem(a, 0L, 0L) * nl_elem(a, 1L, 1L) - nl_elem(a, 0L, 1L) * nl_elem(a, 1L, 0L)
  } else {
    jax_error("_slogdet_small only supports n in {1, 2}, got n=", n, class = "jax_value_error")
  }
  list(jnp_sign(det), jnp_real(jnp_log(jnp_abs(det))))
}

# sign_dot and the real part of ans_dot shared by the slogdet JVP rules.
slogdet_tangents <- function(x, sign, ans_dot) {
  if (nl_is_complex(x)) {
    sign_dot <- (ans_dot - jnp_astype(jnp_real(ans_dot), nl_dtype(ans_dot))) * sign
    list(sign_dot, jnp_real(ans_dot))
  } else {
    list(jnp_zeros_like(sign), ans_dot)
  }
}

# Port of jax/_src/numpy/linalg.py:_slogdet_small_jvp.
slogdet_small_jvp <- function(primals, tangents) {
  x <- primals[[1L]]
  g <- tangents[[1L]]
  out <- slogdet_small(x)
  if (utils::tail(nl_shape(x), 1L) == 1L) {
    ans_dot <- nl_elem(g, 0L, 0L) / nl_elem(x, 0L, 0L)
  } else {
    x00 <- nl_elem(x, 0L, 0L)
    x01 <- nl_elem(x, 0L, 1L)
    x10 <- nl_elem(x, 1L, 0L)
    x11 <- nl_elem(x, 1L, 1L)
    det <- x00 * x11 - x01 * x10
    ans_dot <- (nl_elem(g, 0L, 0L) * x11 - nl_elem(g, 0L, 1L) * x10 -
                  nl_elem(g, 1L, 0L) * x01 + nl_elem(g, 1L, 1L) * x00) / det
  }
  list(out, slogdet_tangents(x, out[[1L]], ans_dot))
}

slogdet_small <- custom_jvp(slogdet_small_fn, slogdet_small_jvp)

# Port of jax/_src/numpy/linalg.py:_slogdet_lu: slogdet from the LU decomposition.
slogdet_lu_fn <- function(a) {
  dt <- nl_dtype(a)
  f <- lax_linalg_lu(a)
  pivot <- f[[2L]]
  diag <- nl_diagonal(f[[1L]])
  is_zero <- jnp_any(diag == as_jax(0, dt), axis = -1L)
  iota <- lax_expand_dims(jnp_arange(utils::tail(nl_shape(a), 1L), dtype = nl_dtype(pivot)),
                          seq_len(linalg_ndim(pivot) - 1L) - 1L)
  parity <- jnp_sum(pivot != iota, axis = -1L)
  if (is_complex_dtype(dt)) {
    sign <- jnp_prod(diag / jnp_astype(jnp_abs(diag), dt), axis = -1L)
  } else {
    sign <- as_jax(1, dt)
    parity <- parity + jnp_sum(diag < 0, axis = -1L)
  }
  sign <- jnp_where(is_zero, as_jax(0, dt), sign * jnp_astype(-2L * (parity %% 2L) + 1L, dt))
  logdet <- jnp_where(is_zero, as_jax(-Inf, dt),
                      jnp_sum(jnp_astype(jnp_log(jnp_abs(diag)), dt), axis = -1L))
  list(sign, jnp_real(logdet))
}

# Port of jax/_src/numpy/linalg.py:_slogdet_qr: slogdet from the QR decomposition (no
# row pivoting).
slogdet_qr_fn <- function(a) {
  if (nl_is_complex(a))
    jax_error("slogdet method='qr' not implemented for complex inputs",
              class = "jax_not_implemented_error")
  n <- utils::tail(nl_shape(a), 1L)
  f <- geqrf(a)
  # The determinant of a triangular matrix is the product of its diagonal elements.
  a_diag <- nl_diagonal(f[[1L]])
  log_abs_det <- jnp_sum(jnp_log(jnp_abs(a_diag)), axis = -1L)
  sign_diag <- jnp_prod(jnp_sign(a_diag), axis = -1L)
  # The determinant of a Householder reflector is -1: every actual reflection
  # (tau != 0) flips the sign.
  taus <- lax_slice_in_dim(f[[2L]], 0L, max(n - 1L, 0L), axis = -1L)
  sign_taus <- jnp_astype(jnp_prod(jnp_where(taus != 0, -1L, 1L), axis = -1L), nl_dtype(sign_diag))
  list(sign_diag * sign_taus, log_abs_det)
}

# Port of jax/_src/numpy/linalg.py:_slogdet_jvp.
slogdet_jvp <- function(primals, tangents) {
  x <- primals[[1L]]
  g <- tangents[[1L]]
  out <- jnp_linalg_slogdet(x)
  ans_dot <- nl_trace(jnp_linalg_solve(x, g))
  list(unname(out), slogdet_tangents(x, out[[1L]], ans_dot))
}

slogdet_lu <- custom_jvp(slogdet_lu_fn, slogdet_jvp)
slogdet_qr <- custom_jvp(slogdet_qr_fn, slogdet_jvp)

#' @rdname jnp_linalg
#' @export
jnp_linalg_slogdet <- function(a, method = NULL) {
  a <- nl_inexact(a)
  s <- nl_shape(a)
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L])
    jax_error("Argument to slogdet() must have shape [..., n, n], got ", py_shape(s),
              class = "jax_value_error")
  res <- if (is.null(method) || identical(method, "lu")) {
    if (s[r] == 1L || s[r] == 2L) slogdet_small(a) else slogdet_lu(a)
  } else if (identical(method, "qr")) {
    slogdet_qr(a)
  } else {
    jax_error("Unknown slogdet method '", method, "'. Supported methods ",
              "are 'lu' (`None`), and 'qr'.", class = "jax_value_error")
  }
  named_result(res, c("sign", "logabsdet"))
}

# Port of jax/_src/numpy/linalg.py:_cofactor_solve: det(a) and x = det(a) * a^-1 b
# (a^-1 scaled by the determinant, the cofactor matrix transposed, applied to b),
# which is well defined for matrices of rank n and n - 1.
# JAX takes cumprod(diag) and uses its last two entries; here the two products are
# reductions. The rows of x are permuted with permute_rows instead of a gather.
cofactor_solve <- function(a, b) {
  a <- nl_inexact(a)
  b <- nl_inexact(b)
  a_shape <- nl_shape(a)
  b_shape <- nl_shape(b)
  nd <- length(a_shape)
  if (!(nd >= 2L && a_shape[nd] == a_shape[nd - 1L] && length(b_shape) >= 2L &&
        identical(utils::tail(b_shape, 2L), utils::tail(a_shape, 2L))))
    jax_error("The arguments to _cofactor_solve must have shapes ",
              "a=[..., m, m] and b=[..., m, m]; got a=", py_shape(a_shape), " and b=",
              py_shape(b_shape), class = "jax_value_error")
  n <- a_shape[nd]
  if (n == 1L) return(list(nl_elem(a, 0L, 0L), b))
  # lu contains u in the upper triangular matrix and l in the strict lower triangular
  # matrix. The diagonal of l is set to ones without loss of generality.
  f <- lax_linalg_lu(a)
  lu <- f[[1L]]
  pivots <- f[[2L]]
  dt <- nl_dtype(a)
  batch_dims <- broadcast_shapes_impl(list(utils::head(nl_shape(lu), -2L), utils::head(b_shape, -2L)))
  x <- jnp_broadcast_to(b, c(batch_dims, utils::tail(b_shape, 2L)))
  lu <- jnp_broadcast_to(lu, c(batch_dims, n, n))
  # Compute the (partial) determinant, ignoring the last diagonal entry of LU.
  diag <- nl_diagonal(lu)
  iota <- lax_expand_dims(jnp_arange(n, dtype = nl_dtype(pivots)),
                          seq_len(linalg_ndim(pivots) - 1L) - 1L)
  parity <- jnp_sum(pivots != iota, axis = -1L)
  sign <- jnp_astype(-2L * (parity %% 2L) + 1L, dt)
  det <- jnp_prod(diag, axis = -1L) * sign  # the full determinant
  det_but_last <- jnp_prod(lax_slice_in_dim(diag, 0L, n - 1L, axis = -1L), axis = -1L) * sign
  corner <- lax_broadcast(lax_bitwise_and(lax_eq(lax_broadcasted_iota("int32", c(n, n), 0L),
                                                 as_jax(n - 1L, "int32")),
                                          lax_eq(lax_broadcasted_iota("int32", c(n, n), 1L),
                                                 as_jax(n - 1L, "int32"))), batch_dims)
  lu <- jnp_where(corner, lax_expand_dims(1.0 / det_but_last, c(-2L, -1L)), lu)
  permutation <- jnp_broadcast_to(f[[3L]], c(batch_dims, n))
  # filter out any matrices that are not full rank
  d <- jnp_ones(utils::head(nl_shape(x), -1L), nl_dtype(x))
  d <- lax_linalg_triangular_solve(lu, d, left_side = TRUE, lower = FALSE)
  d <- jnp_any(jnp_logical_or(jnp_isnan(d), jnp_isinf(d)), axis = -1L)
  d <- jnp_broadcast_to(lax_expand_dims(d, c(-2L, -1L)), nl_shape(x))
  x <- jnp_where(d, jnp_zeros_like(x), x)  # first filter
  x <- permute_rows(x, permutation)
  x <- lax_linalg_triangular_solve(lu, x, left_side = TRUE, lower = TRUE, unit_diagonal = TRUE)
  x <- jnp_concatenate(list(lax_slice_in_dim(x, 0L, n - 1L, axis = -2L) *
                              lax_expand_dims(det, c(-2L, -1L)),
                            lax_slice_in_dim(x, n - 1L, n, axis = -2L)), axis = -2L)
  x <- lax_linalg_triangular_solve(lu, x, left_side = TRUE, lower = FALSE)
  x <- jnp_where(d, jnp_zeros_like(x), x)  # second filter
  list(det, x)
}

# Port of jax/_src/numpy/linalg.py:_det_jvp.
det_jvp <- function(primals, tangents) {
  f <- cofactor_solve(primals[[1L]], tangents[[1L]])
  list(f[[1L]], nl_trace(f[[2L]]))
}

# Port of jax/_src/numpy/linalg.py:_det.
det_general <- custom_jvp(function(a) {
  f <- jnp_linalg_slogdet(a)
  f[[1L]] * jnp_astype(jnp_exp(f[[2L]]), nl_dtype(f[[1L]]))
}, det_jvp)

# Port of jax/_src/numpy/linalg.py:_det_2x2: closed-form 2x2 LU with row pivoting
# (PA = LU). Direct a00*a11 - a01*a10 loses precision in float32 when both products
# are close; swapping rows so |r0[0]| >= |r1[0]| keeps the multiplier |l10| <= 1.
det_2x2 <- custom_jvp(function(a) {
  a0 <- nl_row(a, 0L)
  a1 <- nl_row(a, 1L)
  swap <- jnp_abs(nl_last(a1, 0L)) > jnp_abs(nl_last(a0, 0L))
  swap_row <- lax_expand_dims(swap, -1L)
  r0 <- jnp_where(swap_row, a1, a0)
  r1 <- jnp_where(swap_row, a0, a1)
  r00 <- nl_last(r0, 0L)
  safe_r00 <- jnp_where(r00 == 0, 1, r00)
  l10 <- jnp_where(r00 == 0, 0, nl_last(r1, 0L) / safe_r00)
  u11 <- nl_last(r1, 1L) - l10 * nl_last(r0, 1L)
  jnp_where(swap, -1L, 1L) * r00 * u11
}, det_jvp)

# Port of jax/_src/numpy/linalg.py:_det_3x3: closed-form 3x3 LU with row pivoting.
# Eliminates column by column while keeping the row swap multipliers <= 1.
det_3x3 <- custom_jvp(function(a) {
  col <- function(r, j) nl_last(r, j)
  ex <- function(v) lax_expand_dims(v, -1L)
  a0 <- nl_row(a, 0L)
  a1 <- nl_row(a, 1L)
  # Step 1: pivot column 0 so row 0 gets the largest magnitude element.
  s01 <- jnp_abs(col(a1, 0L)) > jnp_abs(col(a0, 0L))
  r0 <- jnp_where(ex(s01), a1, a0)
  r1 <- jnp_where(ex(s01), a0, a1)
  r2 <- nl_row(a, 2L)
  sign <- jnp_where(s01, -1L, 1L)
  s02 <- jnp_abs(col(r2, 0L)) > jnp_abs(col(r0, 0L))
  r0_new <- jnp_where(ex(s02), r2, r0)
  r2 <- jnp_where(ex(s02), r0, r2)
  r0 <- r0_new
  sign <- jnp_where(s02, -sign, sign)
  # Eliminate the column 0 entries of rows 1 and 2.
  r00 <- col(r0, 0L)
  safe_r00 <- jnp_where(r00 == 0, 1, r00)
  l10 <- jnp_where(r00 == 0, 0, col(r1, 0L) / safe_r00)
  l20 <- jnp_where(r00 == 0, 0, col(r2, 0L) / safe_r00)
  r1 <- r1 - ex(l10) * r0
  r2 <- r2 - ex(l20) * r0
  # Step 2: pivot column 1 across rows 1 and 2, then eliminate row 2.
  s12 <- jnp_abs(col(r2, 1L)) > jnp_abs(col(r1, 1L))
  r1_new <- jnp_where(ex(s12), r2, r1)
  r2 <- jnp_where(ex(s12), r1, r2)
  r1 <- r1_new
  sign <- jnp_where(s12, -sign, sign)
  r11 <- col(r1, 1L)
  safe_r11 <- jnp_where(r11 == 0, 1, r11)
  l21 <- jnp_where(r11 == 0, 0, col(r2, 1L) / safe_r11)
  u22 <- col(r2, 2L) - l21 * col(r1, 2L)
  sign * col(r0, 0L) * r11 * u22
}, det_jvp)

#' @rdname jnp_linalg
#' @export
jnp_linalg_det <- function(a) {
  a <- nl_inexact(a)
  s <- nl_shape(a)
  r <- length(s)
  if (r >= 2L && s[r] == 2L && s[r - 1L] == 2L) return(det_2x2(a))
  if (r >= 2L && s[r] == 3L && s[r - 1L] == 3L) return(det_3x3(a))
  if (r >= 2L && s[r] == s[r - 1L]) return(det_general(a))
  jax_error("Argument to _det() must have shape [..., n, n], got ", py_shape(s),
            class = "jax_value_error")
}

# ---- eigenvalue problems -----------------------------------------------------------------

#' @rdname jnp_linalg
#' @export
jnp_linalg_eig <- function(a) {
  named_result(lax_linalg_eig(nl_inexact(a), compute_left_eigenvectors = FALSE),
               c("eigenvalues", "eigenvectors"))
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_eigvals <- function(a) {
  lax_linalg_eig(nl_inexact(a), compute_left_eigenvectors = FALSE,
                 compute_right_eigenvectors = FALSE)[[1L]]
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_eigh <- function(a, UPLO = NULL, symmetrize_input = TRUE) {
  a <- as_jax(a)
  s <- nl_shape(a)
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L])
    jax_error("Argument to linalg.eigh must have shape [..., n, n], got ", py_shape(s), ".",
              class = "jax_value_error")
  lower <- if (is.null(UPLO) || identical(UPLO, "L")) {
    TRUE
  } else if (identical(UPLO, "U")) {
    FALSE
  } else {
    jax_error("UPLO must be one of None, 'L', or 'U', got ", format(UPLO), class = "jax_value_error")
  }
  f <- lax_linalg_eigh(nl_inexact(a), lower = lower, symmetrize_input = symmetrize_input)
  list(eigenvalues = f[[2L]], eigenvectors = f[[1L]])
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_eigvalsh <- function(a, UPLO = "L", symmetrize_input = TRUE) {
  jnp_linalg_eigh(nl_inexact(a), UPLO, symmetrize_input = symmetrize_input)[[1L]]
}

# ---- pinv, inv, solve, lstsq -------------------------------------------------------------

# Port of jax/_src/numpy/linalg.py:_pinv. Uses the same algorithm as NumPy.
pinv_fn <- function(a, rtol = NULL, hermitian = FALSE) {
  arr <- nl_inexact(a)
  s <- nl_shape(arr)
  r <- length(s)
  m <- s[r - 1L]
  n <- s[r]
  if (m == 0L || n == 0L) return(jnp_zeros(c(s[seq_len(r - 2L)], n, m), nl_dtype(arr)))
  arr <- jnp_conjugate(arr)
  if (is.null(rtol))
    rtol <- (10 * max(m, n)) * jnp_asarray(finfo(nl_dtype(arr))$eps, dtype = default_float())
  rtol <- as_jax(rtol)
  f <- jnp_linalg_svd(arr, full_matrices = FALSE, hermitian = hermitian)
  u <- f[[1L]]
  sv <- f[[2L]]
  vh <- f[[3L]]
  # Singular values less than or equal to rtol * largest_singular_value are set to
  # zero.
  rtol <- lax_expand_dims(rtol, -1L)
  rtol <- lax_expand_dims(rtol, seq_len(linalg_ndim(sv) - linalg_ndim(rtol)) - 1L)
  cutoff <- rtol * lax_slice_in_dim(sv, 0L, 1L, axis = -1L)
  sv <- jnp_astype(jnp_where(sv > cutoff, sv, Inf), nl_dtype(u))
  res <- jnp_matmul(jnp_linalg_matrix_transpose(vh),
                    jnp_true_divide(jnp_linalg_matrix_transpose(u), lax_expand_dims(sv, -1L)),
                    precision = "highest")
  lax_convert_element_type(res, nl_dtype(arr))
}

# Port of jax/_src/numpy/linalg.py:_pinv_jvp.
# The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose
# Variables Separate. G. H. Golub and V. Pereyra. SIAM Journal on Numerical Analysis,
# Vol. 10, No. 2 (Apr., 1973), pp. 413-432.
pinv_jvp <- function(hermitian, primals, tangents) {
  a <- primals[[1L]]
  a_dot <- tangents[[1L]]
  p <- jnp_linalg_pinv(a, rtol = primals[[2L]], hermitian = hermitian)
  if (hermitian) {
    # svd(..., hermitian = TRUE) symmetrizes its input, and the JVP must match.
    a <- nl_symmetrize(a)
    a_dot <- nl_symmetrize(a_dot)
  }
  s_ <- nl_shape(a)
  m <- s_[length(s_) - 1L]
  n <- s_[length(s_)]
  if (m >= n) {
    s <- (p %*% nl_H(p)) %*% nl_H(a_dot)
    t <- (nl_H(a_dot) %*% nl_H(p)) %*% p
    p_dot <- -(p %*% a_dot) %*% p + s - (s %*% a) %*% p + t - (p %*% a) %*% t
  } else {
    s <- p %*% (nl_H(p) %*% nl_H(a_dot))
    t <- nl_H(a_dot) %*% (nl_H(p) %*% p)
    p_dot <- -p %*% (a_dot %*% p) + s - s %*% (a %*% p) + t - p %*% (a %*% t)
  }
  list(p, p_dot)
}

pinv_custom <- custom_jvp(pinv_fn, pinv_jvp, nondiff_argnums = 2L)

#' @rdname jnp_linalg
#' @export
jnp_linalg_pinv <- function(a, rtol = NULL, hermitian = FALSE, rcond = NULL) {
  if (!is.null(rcond)) {
    if (!is.null(rtol))
      jax_error("pinv: only one of rtol and rcond may be specified.", class = "jax_value_error")
    rtol <- rcond
  }
  pinv_custom(nl_inexact(a), rtol, isTRUE(hermitian))
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_inv <- function(a) {
  arr <- as_jax(a)
  s <- nl_shape(arr)
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L])
    jax_error("Argument to inv must have shape [..., n, n], got ", py_shape(s), ".",
              class = "jax_value_error")
  arr <- nl_inexact(arr)
  jnp_linalg_solve(arr, lax_broadcast(jnp_eye(s[r], dtype = nl_dtype(arr)), s[seq_len(r - 2L)]))
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_solve <- function(a, b) {
  ab <- promote_dtypes_inexact(list(as_jax(a), as_jax(b)))
  a <- ab[[1L]]
  b <- ab[[2L]]
  as_ <- nl_shape(a)
  bs <- nl_shape(b)
  na <- length(as_)
  nb <- length(bs)
  if (na < 2L)
    jax_error("left hand array must be at least two dimensional; got a.shape=", py_shape(as_),
              class = "jax_value_error")
  # Check for invalid inputs that previously would have led to a batched 1D solve:
  if (nb > 1L && na == nb + 1L && as_[na] == bs[nb] && as_[na] != bs[nb - 1L])
    jax_error("Invalid shapes for solve: ", py_shape(as_), ", ", py_shape(bs),
              ". Prior to JAX v0.5.0, this would have been treated as a batched ",
              "1-dimensional solve. To recover this behavior, use ",
              "solve(a, b[..., None]).squeeze(-1).", class = "jax_value_error")
  # JAX maps lax.linalg._solve over the broadcast batch dimensions with jnp.vectorize
  # (signature (m,m),(m)->(m) for a vector b, else (m,m),(m,n)->(m,n)); the solve here
  # works on batches directly, so only the batch dimensions are broadcast.
  core_b <- if (nb == 1L) 1L else 2L
  if (as_[na] != as_[na - 1L] || bs[nb - core_b + 1L] != as_[na])
    jax_error("The arguments to solve must have shapes a=[..., m, m] and ",
              "b=[..., m, k] or b=[..., m]; got a=", py_shape(as_), " and b=", py_shape(bs),
              class = "jax_value_error")
  batch <- broadcast_shapes_impl(list(utils::head(as_, -2L), utils::head(bs, -core_b)))
  a <- jnp_broadcast_to(a, c(batch, utils::tail(as_, 2L)))
  b <- jnp_broadcast_to(b, c(batch, utils::tail(bs, core_b)))
  linalg_solve(a, b)
}

# Port of jax/_src/numpy/linalg.py:_lstsq.
lstsq_impl <- function(a, b, rcond, numpy_resid = FALSE) {
  ab <- promote_dtypes_inexact(list(a, b))
  a <- ab[[1L]]
  b <- ab[[2L]]
  as_ <- nl_shape(a)
  bs <- nl_shape(b)
  if (!length(as_) || !length(bs) || as_[1L] != bs[1L])
    jax_error("Leading dimensions of input arrays must match", class = "jax_value_error")
  b_orig_ndim <- length(bs)
  if (b_orig_ndim == 1L) b <- lax_expand_dims(b, 1L)
  if (length(as_) != 2L)
    jax_error(length(as_), "-dimensional array given. Array must be two-dimensional",
              class = "jax_type_error")
  if (linalg_ndim(b) != 2L)
    jax_error(linalg_ndim(b), "-dimensional array given. Array must be one or two-dimensional",
              class = "jax_type_error")
  m <- as_[1L]
  n <- as_[2L]
  dt <- nl_dtype(a)
  if (m * n == 0L) {
    s <- jnp_zeros(0L, complex_component(dt))
    rank <- as_jax(0L, default_int())
    x <- jnp_zeros(c(n, nl_shape(b)[-1L]), dt)
  } else {
    eps <- finfo(dt)$eps
    rcond <- if (is.null(rcond)) eps * max(n, m) else jnp_where(as_jax(rcond) < 0, eps, rcond)
    f <- jnp_linalg_svd(a, full_matrices = FALSE)
    s <- f[[2L]]
    mask <- (s > 0) & (s >= jnp_astype(rcond, nl_dtype(s)) * nl_last(s, 0L))
    rank <- jnp_sum(mask)
    safe_s <- jnp_astype(jnp_where(mask, s, 1), dt)
    s_inv <- lax_expand_dims(jnp_where(mask, 1 / safe_s, 0), 1L)
    uTb <- jnp_matmul(nl_H(f[[1L]]), b, precision = "highest")
    x <- jnp_matmul(nl_H(f[[3L]]), s_inv * uTb, precision = "highest")
  }
  # NumPy returns empty residuals in some cases. To allow compilation, the full
  # residuals are returned by default.
  if (numpy_resid && (isTRUE(as.logical(rank < n)) || m <= n)) {
    resid <- jnp_zeros(0L)
  } else {
    b_estimate <- jnp_matmul(a, x, precision = "highest")
    resid <- jnp_linalg_norm(b - b_estimate, axis = 0L)^2L
  }
  if (b_orig_ndim == 1L) x <- jnp_ravel(x)
  list(x, resid, rank, s)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_lstsq <- function(a, b, rcond = NULL, numpy_resid = FALSE) {
  lstsq_impl(as_jax(a), as_jax(b), rcond, numpy_resid = isTRUE(numpy_resid))
}

# ---- norms -------------------------------------------------------------------------------

#' @rdname jnp_linalg
#' @export
jnp_linalg_norm <- function(x, ord = NULL, axis = NULL, keepdims = FALSE) {
  x <- nl_inexact(x)
  ndim <- linalg_ndim(x)
  frobenius <- function(axis) {
    jnp_sqrt(jnp_sum(jnp_real(x * jnp_conjugate(x)), axis = axis, keepdims = keepdims))
  }
  if (is.null(axis)) {
    # NumPy admits arbitrary rank inputs if `ord` is None.
    if (is.null(ord)) return(frobenius(NULL))
    axis <- seq_len(ndim) - 1L
  } else {
    axis <- norm_axis(axis, ndim)
  }
  if (length(axis) == 1L)
    return(jnp_linalg_vector_norm(x, ord = if (is.null(ord)) 2 else ord, axis = axis,
                                  keepdims = keepdims))
  if (length(axis) != 2L)
    jax_error("Improper number of axes for norm: axis=", py_shape(axis), ". Pass one axis to",
              " compute a vector-norm, or two axes to compute a matrix-norm.",
              class = "jax_value_error")
  row_axis <- axis[1L]
  col_axis <- axis[2L]
  abs_sum <- function(ax) jnp_sum(jnp_abs(x), axis = ax, keepdims = keepdims)
  if (is.null(ord) || identical(ord, "f") || identical(ord, "fro")) return(frobenius(axis))
  if (nl_num(ord, 1) || nl_num(ord, -1)) {
    if (!keepdims && col_axis > row_axis) col_axis <- col_axis - 1L
    s <- abs_sum(row_axis)
    return(if (ord == 1) nl_amax0(s, col_axis, keepdims) else jnp_min(s, axis = col_axis, keepdims = keepdims))
  }
  if (nl_num(ord, Inf) || nl_num(ord, -Inf)) {
    if (!keepdims && row_axis > col_axis) row_axis <- row_axis - 1L
    s <- abs_sum(col_axis)
    return(if (ord > 0) nl_amax0(s, row_axis, keepdims) else jnp_min(s, axis = row_axis, keepdims = keepdims))
  }
  if (identical(ord, "nuc") || nl_num(ord, 2) || nl_num(ord, -2)) {
    s <- jnp_linalg_svd(jnp_moveaxis(x, axis, c(-2L, -1L)), compute_uv = FALSE)
    y <- if (nl_num(ord, 2)) {
      nl_amax0(s, -1L)
    } else if (nl_num(ord, -2)) {
      jnp_min(s, axis = -1L)
    } else {
      jnp_sum(s, axis = -1L)
    }
    if (keepdims) y <- jnp_expand_dims(y, axis)
    return(y)
  }
  jax_error("Invalid order '", format(ord), "' for matrix norm.", class = "jax_value_error")
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_matrix_norm <- function(x, keepdims = FALSE, ord = "fro") {
  jnp_linalg_norm(as_jax(x), ord = ord, keepdims = keepdims, axis = c(-2L, -1L))
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_vector_norm <- function(x, axis = NULL, keepdims = FALSE, ord = 2) {
  x <- as_jax(x)
  if (is.null(ord) || nl_num(ord, 2))
    return(jnp_sqrt(jnp_sum(jnp_real(x * jnp_conjugate(x)), axis = axis, keepdims = keepdims)))
  if (nl_num(ord, Inf)) return(nl_amax0(jnp_abs(x), axis, keepdims))
  if (nl_num(ord, -Inf)) return(jnp_min(jnp_abs(x), axis = axis, keepdims = keepdims))
  if (nl_num(ord, 0))
    return(jnp_sum(x != 0, dtype = finfo(nl_dtype(x))$dtype, axis = axis, keepdims = keepdims))
  # NumPy has a special case for ord == 1 with slightly different type promotion
  # semantics.
  if (nl_num(ord, 1)) return(jnp_sum(jnp_abs(x), axis = axis, keepdims = keepdims))
  if (is.character(ord)) {
    msg <- paste0("Invalid order '", ord, "' for vector norm.")
    if (ord == "inf") msg <- paste0(msg, "Use 'jax.numpy.inf' instead.")
    if (ord == "-inf") msg <- paste0(msg, "Use '-jax.numpy.inf' instead.")
    jax_error(msg, class = "jax_value_error")
  }
  abs_x <- jnp_abs(x)
  # lax._const: the order and its reciprocal take the dtype of abs_x (for integer
  # input the reciprocal truncates, as in the reference version).
  ord_arr <- as_jax(ord, nl_dtype(abs_x))
  inv <- 1 / as.numeric(as.vector(array_host(ord_arr)))
  # 1 / 0 in an integer dtype is a C cast of infinity upstream: convert on the device.
  ord_inv <- if (is.finite(inv)) as_jax(inv, nl_dtype(abs_x))
             else lax_convert_element_type(as_jax(inv), nl_dtype(abs_x))
  jnp_power(jnp_sum(jnp_power(abs_x, ord_arr), axis = axis, keepdims = keepdims), ord_inv)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_cond <- function(x, p = NULL) {
  arr <- as_jax(x)
  s <- nl_shape(arr)
  r <- length(s)
  if (r < 2L)
    jax_error("jnp.linalg.cond: input array must be at least 2D; got arr.shape=", py_shape(s),
              class = "jax_value_error")
  if (s[r] == 0L || s[r - 1L] == 0L)
    jax_error("jnp.linalg.cond: input array must not be empty; got arr.shape=", py_shape(s),
              class = "jax_value_error")
  if (is.null(p) || nl_num(p, 2) || nl_num(p, -2)) {
    sv <- jnp_linalg_svdvals(arr)
    res <- if (nl_num(p, -2)) nl_last(sv, -1L) / nl_last(sv, 0L) else nl_last(sv, 0L) / nl_last(sv, -1L)
  } else {
    if (s[r - 1L] != s[r])
      jax_error("jnp.linalg.cond: for p=", format(p), ", array must be square; got arr.shape=",
                py_shape(s), class = "jax_value_error")
    res <- jnp_linalg_norm(arr, ord = p, axis = c(-2L, -1L)) *
      jnp_linalg_norm(jnp_linalg_inv(arr), ord = p, axis = c(-2L, -1L))
  }
  # Convert NaNs to infs where the original array has no NaNs.
  jnp_where(jnp_isnan(res) & !jnp_any(jnp_isnan(arr), axis = c(-2L, -1L)), Inf, res)
}

# ---- qr ------------------------------------------------------------------------------------

#' @rdname jnp_linalg
#' @export
jnp_linalg_qr <- function(a, mode = "reduced") {
  a <- nl_inexact(a)
  if (identical(mode, "raw")) {
    f <- geqrf(a)
    return(list(Q = jnp_linalg_matrix_transpose(f[[1L]]), R = f[[2L]]))
  }
  full_matrices <- if (mode %in% c("reduced", "r", "full")) {
    FALSE
  } else if (identical(mode, "complete")) {
    TRUE
  } else {
    jax_error("Unsupported QR decomposition mode '", mode, "'", class = "jax_value_error")
  }
  f <- lax_linalg_qr(a, pivoting = FALSE, full_matrices = full_matrices)
  if (identical(mode, "r")) f[[2L]] else named_result(f, c("Q", "R"))
}

# ---- products, transposes, traces --------------------------------------------------------

#' @rdname jnp_linalg
#' @export
jnp_linalg_cross <- function(x1, x2, axis = -1L) {
  ab <- promote_dtypes(list(as_jax(x1), as_jax(x2)))
  n1 <- nl_shape(ab[[1L]])[norm_axis(axis, linalg_ndim(ab[[1L]])) + 1L]
  n2 <- nl_shape(ab[[2L]])[norm_axis(axis, linalg_ndim(ab[[2L]])) + 1L]
  if (n1 != 3L || n2 != 3L)
    jax_error("Both input arrays must be (arrays of) 3-dimensional vectors, ",
              "but they have x1.shape[axis]=", n1, " and x2.shape[axis]=", n2,
              class = "jax_value_error")
  # Port of jax/_src/numpy/lax_numpy.py:cross for 3-vectors.
  a <- jnp_moveaxis(ab[[1L]], axis, -1L)
  b <- jnp_moveaxis(ab[[2L]], axis, -1L)
  a0 <- nl_last(a, 0L); a1 <- nl_last(a, 1L); a2 <- nl_last(a, 2L)
  b0 <- nl_last(b, 0L); b1 <- nl_last(b, 1L); b2 <- nl_last(b, 2L)
  out <- jnp_stack(list(a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis = 0L)
  jnp_moveaxis(out, 0L, axis)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_outer <- function(x1, x2) {
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  if (linalg_ndim(x1) != 1L || linalg_ndim(x2) != 1L)
    jax_error("Input arrays must be one-dimensional, but they are x1.ndim=", linalg_ndim(x1),
              " x2.ndim=", linalg_ndim(x2), class = "jax_value_error")
  lax_expand_dims(x1, 1L) * lax_expand_dims(x2, 0L)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_matrix_transpose <- function(x) {
  x <- as_jax(x)
  ndim <- linalg_ndim(x)
  if (ndim < 2L)
    jax_error("matrix_transpose requires at least 2 dimensions; got ndim=", ndim,
              class = "jax_value_error")
  linalg_T(x)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_vecdot <- function(x1, x2, axis = -1L, precision = NULL, preferred_element_type = NULL) {
  # Port of jax/_src/numpy/tensor_contractions.py:vecdot (vdot mapped over the leading
  # dimensions).
  x1 <- as_jax(x1)
  x2 <- as_jax(x2)
  n1 <- nl_shape(x1)[norm_axis(axis, linalg_ndim(x1)) + 1L]
  n2 <- nl_shape(x2)[norm_axis(axis, linalg_ndim(x2)) + 1L]
  if (n1 != n2)
    jax_error("axes must match; got x1_arr.shape[axis]=", n1, " and x2_arr.shape[axis]=", n2,
              class = "jax_value_error")
  a <- lax_expand_dims(jnp_conjugate(jnp_moveaxis(x1, axis, -1L)), -2L)
  b <- lax_expand_dims(jnp_moveaxis(x2, axis, -1L), -1L)
  out <- jnp_matmul(a, b, precision = precision, preferred_element_type = preferred_element_type)
  lax_squeeze(out, c(-2L, -1L))
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_matmul <- function(x1, x2, precision = NULL, preferred_element_type = NULL) {
  jnp_matmul(as_jax(x1), as_jax(x2), precision = precision,
             preferred_element_type = preferred_element_type)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_tensordot <- function(x1, x2, axes = 2L, precision = NULL, preferred_element_type = NULL) {
  # Port of jax/_src/numpy/tensor_contractions.py:tensordot.
  a <- as_jax(x1)
  b <- as_jax(x2)
  if (is.null(preferred_element_type)) {
    ab <- promote_dtypes(list(a, b))
    a <- ab[[1L]]
    b <- ab[[2L]]
  }
  na <- linalg_ndim(a)
  nb <- linalg_ndim(b)
  if (is.numeric(axes) && length(axes) == 1L) {
    if (axes > min(na, nb))
      jax_error("Number of tensordot axes (axes ", axes, ") exceeds input ranks (",
                py_shape(nl_shape(a)), " and ", py_shape(nl_shape(b)), ")", class = "jax_type_error")
    contracting <- list(na - axes + seq_len(axes) - 1L, seq_len(axes) - 1L)
  } else if (is.list(axes) && length(axes) == 2L) {
    if (length(axes[[1L]]) != length(axes[[2L]]))
      jax_error("tensordot requires axes lists to have equal length, got ", length(axes[[1L]]),
                " and ", length(axes[[2L]]), ".", class = "jax_type_error")
    contracting <- list(norm_axis(axes[[1L]], na), norm_axis(axes[[2L]], nb))
  } else {
    jax_error("tensordot axes argument must be an int, a pair of ints, or a pair of ",
              "lists/tuples of ints.", class = "jax_type_error")
  }
  lax_dot_general(a, b, list(contracting, list(integer(0), integer(0))), precision,
                  preferred_element_type)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_diagonal <- function(x, offset = 0L) nl_diagonal(x, offset)

#' @rdname jnp_linalg
#' @export
jnp_linalg_trace <- function(x, offset = 0L, dtype = NULL) nl_trace(x, offset, dtype)

#' @rdname jnp_linalg
#' @export
jnp_linalg_tensorinv <- function(a, ind = 2L) {
  arr <- as_jax(a)
  if (!is.numeric(ind) || length(ind) != 1L || ind != round(ind))
    jax_error("ind must be an integer", class = "jax_type_error")
  ind <- as.integer(ind)
  if (ind <= 0L) jax_error("ind must be a positive integer; got ind=", ind, class = "jax_value_error")
  s <- nl_shape(arr)
  contracting_shape <- s[seq_len(min(ind, length(s)))]
  batch_shape <- drop_first(s, ind)
  flatshape <- c(prod_int(contracting_shape), prod_int(batch_shape))
  if (flatshape[1L] != flatshape[2L])
    jax_error("tensorinv is only possible when the product of the first",
              " `ind` dimensions equals that of the remaining dimensions.",
              " got arr.shape=", py_shape(s), " with ind=", ind, ".", class = "jax_value_error")
  jnp_reshape(jnp_linalg_inv(jnp_reshape(arr, flatshape)), c(batch_shape, contracting_shape))
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_tensorsolve <- function(a, b, axes = NULL) {
  a_arr <- as_jax(a)
  b_arr <- as_jax(b)
  if (!is.null(axes)) {
    # JAX's moveaxis inserts the axes at the end in increasing order of the source axis.
    axes <- sort(norm_axis(axes, linalg_ndim(a_arr)))
    a_arr <- jnp_moveaxis(a_arr, axes, rep(linalg_ndim(a_arr) - 1L, length(axes)))
  }
  as_ <- nl_shape(a_arr)
  bs <- nl_shape(b_arr)
  out_shape <- drop_first(as_, length(bs))
  if (length(as_) < length(bs) || !identical(as_[seq_along(bs)], bs))
    jax_error("After moving axes to end, leading shape of a must match shape of b.",
              " got a.shape=", py_shape(as_), ", b.shape=", py_shape(bs), class = "jax_value_error")
  if (prod_int(bs) != prod_int(out_shape))
    jax_error("Input arrays must have prod(a.shape[:b.ndim]) == prod(a.shape[b.ndim:]);",
              " got a.shape=", py_shape(as_), ", b.ndim=", length(bs), ".", class = "jax_value_error")
  a_arr <- jnp_reshape(a_arr, c(prod_int(bs), prod_int(out_shape)))
  jnp_reshape(jnp_linalg_solve(a_arr, jnp_ravel(b_arr)), out_shape)
}

#' @rdname jnp_linalg
#' @export
jnp_linalg_multi_dot <- function(arrays, precision = NULL) {
  arrs <- lapply(arrays, as_jax)
  n <- length(arrs)
  shapes <- lapply(arrs, nl_shape)
  ranks <- vapply(shapes, length, 1L)
  shapes_str <- paste0("[", paste(vapply(shapes, py_shape, ""), collapse = ", "), "]")
  if (n < 2L)
    jax_error("multi_dot requires at least two arrays; got len(arrays)=", n, class = "jax_value_error")
  if (!(ranks[1L] %in% 1:2 && ranks[n] %in% 1:2 && all(ranks[-c(1L, n)] == 2L)))
    jax_error("multi_dot: input arrays must all be two-dimensional, except for",
              " the first and last array which may be 1 or 2 dimensional.",
              " Got array shapes ", shapes_str, class = "jax_value_error")
  for (i in seq_len(n - 1L))
    if (utils::tail(shapes[[i]], 1L) != shapes[[i + 1L]][1L])
      jax_error("multi_dot: last dimension of each array must match first dimension",
                " of following array. Got array shapes ", shapes_str, class = "jax_value_error")
  # JAX contracts the chain with einsum(optimize = 'auto'); here the classic matrix
  # chain ordering picks the cheapest parenthesization and the factors are multiplied
  # with matmul.
  dims <- c(if (ranks[1L] == 1L) 1L else shapes[[1L]][1L],
            vapply(shapes, function(s) utils::tail(s, 1L), 1L))
  if (ranks[n] == 1L) dims[n + 1L] <- 1L
  cost <- matrix(0, n, n)
  split <- matrix(0L, n, n)
  for (len in seq_len(n - 1L)) for (i in seq_len(n - len)) {
    j <- i + len
    ks <- i:(j - 1L)
    q <- cost[i, ks] + cost[ks + 1L, j] + as.numeric(dims[i]) * dims[ks + 1L] * dims[j + 1L]
    cost[i, j] <- min(q)
    split[i, j] <- ks[which.min(q)]
  }
  mul <- function(i, j) {
    if (i == j) return(arrs[[i]])
    jnp_matmul(mul(i, split[i, j]), mul(split[i, j] + 1L, j), precision = precision)
  }
  mul(1L, n)
}
