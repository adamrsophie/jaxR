# jax.numpy polynomial functions (port of jax/_src/numpy/polynomial.py). Coefficients are
# ordered from the highest degree to the constant term, as in NumPy.

# ---- private helpers (`pnm_` prefix) --------------------------------------------------------

pnm_len <- function(x) {
  s <- aval_of(x)$shape
  if (!length(s)) jax_error("len() of unsized object", class = "jax_type_error")
  s[1L]
}

pnm_atleast_1d <- function(x) if (length(aval_of(x)$shape)) x else lax_reshape(x, 1L)

pnm_int <- function(v, what) {
  if (is_jax_value(v)) v <- as.vector(concrete_or_error(v, what))
  if (!is.numeric(v) || length(v) != 1L || v != round(v))
    jax_error(what, " must be an integer", class = "jax_type_error")
  as.integer(v)
}

# Port of lax_numpy.py:vander.
pnm_vander <- function(x, N) {
  iota <- lax_iota(aval_of(x)$dtype, N)
  iota <- lax_sub(lax_full_like(iota, N - 1L), iota)
  jnp_power(lax_expand_dims(x, 1L), lax_expand_dims(iota, 0L))
}

# Port of lax_numpy.py:trim_zeros for a 1-D array with trim = "f" (leading zeros).
pnm_trim_leading_zeros <- function(filt) {
  v <- as.vector(as.array(concrete_or_error(filt, "the `filt` argument of trim_zeros()")))
  nz <- which(v != 0)
  if (!length(nz)) return(lax_slice_in_dim(filt, 0L, 0L))
  lax_slice_in_dim(filt, nz[1L] - 1L, length(v))
}

# Port of lax_numpy.py:trim_zeros_tol with trim = "f".
pnm_trim_zeros_tol <- function(filt, tol) {
  v <- as.vector(as.array(concrete_or_error(filt, "the `filt` argument of trim_zeros_tol()")))
  keep <- which(!(abs(v) < tol))
  if (!length(keep)) return(lax_full(0L, 0, aval_of(filt)$dtype))
  lax_slice_in_dim(filt, keep[1L] - 1L, length(v))
}

# Port of lax_numpy.py:_roll_dynamic for a 1-D array.
pnm_roll_dynamic <- function(a, shift) {
  n <- pnm_len(a)
  x <- jnp_remainder(lax_convert_element_type(shift, "int32"), max(n, 1L))
  lax_dynamic_slice_in_dim(lax_concatenate(list(a, a), 0L), lax_sub(as_jax(n, "int32"), x), n)
}

# Horner evaluation of jax/_src/lax/lax.py:polynomial (lowering _polynomial_lower);
# `coeffs` run from the constant term upward.
pnm_horner <- function(x, coeffs, unroll = NULL) {
  dt <- aval_of(x)$dtype
  cs <- lapply(coeffs, lax_convert_element_type, new_dtype = dt)
  n <- length(cs)
  if (n <= (unroll %||% 8L)) {
    acc <- cs[[n]]
    for (c in rev(cs[-n])) acc <- lax_add(lax_mul(acc, x), c)
    return(acc)
  }
  shape <- broadcast_shapes_impl(lapply(c(list(x), cs), function(a) aval_of(a)$shape))
  acc <- lax_full(shape, 0, dt)
  for (c in rev(cs)) acc <- lax_add(lax_mul(acc, x), c)
  acc
}

pnm_rank_to <- function(x, r) {
  s <- aval_of(x)$shape
  if (length(s) >= r) x else lax_reshape(x, c(rep(1L, r - length(s)), s))
}

# Port of polynomial.py:_roots_no_zeros.
pnm_roots_no_zeros <- function(p) {
  n <- prod_int(aval_of(p)$shape)
  if (n < 2L) return(lax_full(0L, 0, complex_of(aval_of(p)$dtype)))
  dt <- aval_of(p)$dtype
  k <- n - 1L
  A <- lax_convert_element_type(lax_eq(lax_broadcasted_iota("int32", c(k, k), 0L),
                                       lax_add(lax_broadcasted_iota("int32", c(k, k), 1L), 1L)), dt)
  first <- jnp_true_divide(jnp_negative(lax_slice_in_dim(p, 1L, n)), lax_index_in_dim(p, 0L, keepdims = FALSE))
  A <- at_set(A, 0L, sl(), value = first)
  jnp_linalg_eigvals(A)
}

# Port of polynomial.py:_roots_with_zeros.
pnm_roots_with_zeros <- function(p, num_leading_zeros) {
  n <- pnm_len(p)
  p <- jnp_where(lax_eq(lax_convert_element_type(num_leading_zeros, default_int()), n), 1, p)
  roots <- pnm_roots_no_zeros(pnm_roll_dynamic(p, jnp_negative(num_leading_zeros)))
  roots <- lax_sort_key_val(lax_eq(roots, lax_full_like(roots, 0)), roots)[[2L]]
  m <- prod_int(aval_of(roots)$shape)
  keep <- lax_lt(lax_iota(default_int(), m),
                 lax_sub(lax_full(integer(0), m, default_int()),
                         lax_convert_element_type(num_leading_zeros, default_int())))
  jnp_where(keep, roots, lax_full(integer(0), complex(real = NaN, imaginary = NaN),
                                  aval_of(roots)$dtype))
}

# ---- public API ---------------------------------------------------------------------------------

#' Polynomials
#'
#' Ports of `jnp.poly`, `jnp.polyadd`, `jnp.polysub`, `jnp.polymul`, `jnp.polydiv`,
#' `jnp.polyder`, `jnp.polyint`, `jnp.polyval`, `jnp.polyfit` and `jnp.roots`.
#' Coefficient vectors run from the highest degree to the constant term.
#' @param seq_of_zeros 1-D array of roots, or a square matrix (its eigenvalues are used).
#' @param a1,a2,u,v,p Coefficient arrays.
#' @param x Points at which to evaluate (`polyval`) or sample coordinates (`polyfit`).
#' @param y Sample values (1-D, or 2-D with one column per data set).
#' @param deg Degree of the fitted polynomial.
#' @param rcond Relative condition number for the least squares solve.
#' @param full Also return the residuals, rank, singular values and `rcond`.
#' @param w Optional weights of the samples.
#' @param cov Also return the covariance matrix of the coefficients (`TRUE`, or
#'   `"unscaled"`).
#' @param m Order of the derivative or integral.
#' @param k Integration constants (a scalar or one per integration).
#' @param unroll Number of Horner steps unrolled before a loop is used.
#' @param trim_leading_zeros Whether to remove leading zeros (needs concrete values).
#' @param strip_zeros `roots`: strip leading zeros (needs concrete values); with
#'   `FALSE` the result has one NaN entry per leading zero.
#' @return An array; `polydiv` returns `list(quotient, remainder)`, `polyfit` with
#'   `full` or `cov` returns a list.
#' @name jnp_polynomial
NULL

# Port of polynomial.py:roots.
#' @rdname jnp_polynomial
#' @export
jnp_roots <- function(p, strip_zeros = TRUE) {
  p_arr <- pnm_atleast_1d(promote_dtypes_inexact(list(p))[[1L]])
  if (length(aval_of(p_arr)$shape) != 1L)
    jax_error("Input must be a rank-1 array.", class = "jax_value_error")
  n <- pnm_len(p_arr)
  if (n < 2L) return(lax_full(0L, 0, complex_of(aval_of(p_arr)$dtype)))
  is_zero <- lax_eq(p_arr, lax_full_like(p_arr, 0))
  num_leading_zeros <- jnp_where(red_all(is_zero), n, jnp_argmin(is_zero))
  if (strip_zeros) {
    num_leading_zeros <- as.integer(as.vector(concrete_or_error(num_leading_zeros, paste0(
      "the jnp.roots() function. To use this within a JIT-compiled context, pass ",
      "strip_zeros=False, but be aware that leading zeros will result in some returned ",
      "roots being set to NaN."))))
    return(pnm_roots_no_zeros(lax_slice_in_dim(p_arr, min(num_leading_zeros, n), n)))
  }
  pnm_roots_with_zeros(p_arr, num_leading_zeros)
}

# Port of polynomial.py:polyfit.
#' @rdname jnp_polynomial
#' @export
jnp_polyfit <- function(x, y, deg, rcond = NULL, full = FALSE, w = NULL, cov = FALSE) {
  x_arr <- as_jax(x)
  y_arr <- as_jax(y)
  deg <- pnm_int(deg, "deg")
  order <- deg + 1L
  if (deg < 0L) jax_error("expected deg >= 0", class = "jax_value_error")
  xs <- aval_of(x_arr)$shape
  ys <- aval_of(y_arr)$shape
  if (length(xs) != 1L) jax_error("expected 1D vector for x", class = "jax_type_error")
  if (xs[1L] == 0L) jax_error("expected non-empty vector for x", class = "jax_type_error")
  if (length(ys) < 1L || length(ys) > 2L)
    jax_error("expected 1D or 2D array for y", class = "jax_type_error")
  if (xs[1L] != ys[1L]) jax_error("expected x and y to have same length", class = "jax_type_error")
  if (is.null(rcond)) rcond <- xs[1L] * finfo(aval_of(x_arr)$dtype)$eps
  if (is_jax_value(rcond)) rcond <- as.vector(concrete_or_error(rcond, "rcond"))
  lhs <- pnm_vander(x_arr, order)
  rhs <- y_arr
  if (!is.null(w)) {
    w_arr <- promote_dtypes_inexact(list(w))[[1L]]
    if (length(aval_of(w_arr)$shape) != 1L)
      jax_error("expected a 1-d array for weights", class = "jax_type_error")
    if (aval_of(w_arr)$shape[1L] != ys[1L])
      jax_error("expected w and y to have the same length", class = "jax_type_error")
    lhs <- jnp_multiply(lhs, lax_expand_dims(w_arr, 1L))
    rhs <- jnp_multiply(rhs, if (length(ys) == 2L) lax_expand_dims(w_arr, 1L) else w_arr)
  }
  scale <- jnp_sqrt(red_sum(jnp_multiply(lhs, lhs), axis = 0L))
  lhs <- jnp_true_divide(lhs, lax_expand_dims(scale, 0L))
  sol <- jnp_linalg_lstsq(lhs, rhs, rcond)
  c <- sol[[1L]]
  resids <- sol[[2L]]
  c <- if (length(aval_of(c)$shape) > 1L) jnp_true_divide(c, lax_expand_dims(scale, 1L))
       else jnp_true_divide(c, scale)
  if (isTRUE(full)) return(list(c, resids, sol[[3L]], sol[[4L]], as_jax(rcond)))
  if (isFALSE(cov) || is.null(cov)) return(c)
  Vbase <- jnp_linalg_inv(jnp_dot(jnp_transpose(lhs), lhs))
  Vbase <- jnp_true_divide(Vbase, jnp_outer(scale, scale))
  if (identical(cov, "unscaled")) {
    fac <- as_jax(1)
  } else {
    if (xs[1L] <= order)
      jax_error("the number of data points must exceed order to scale the covariance matrix",
                class = "jax_value_error")
    fac <- jnp_true_divide(resids, xs[1L] - order)
  }
  fac <- pnm_atleast_1d(fac)
  if (length(ys) == 1L) return(list(c, jnp_multiply(Vbase, lax_expand_dims(fac, 0L))))
  list(c, jnp_multiply(lax_expand_dims(Vbase, 2L), lax_expand_dims(fac, c(0L, 1L))))
}

# Port of polynomial.py:poly.
#' @rdname jnp_polynomial
#' @export
jnp_poly <- function(seq_of_zeros) {
  z <- pnm_atleast_1d(promote_dtypes_inexact(list(seq_of_zeros))[[1L]])
  sh <- aval_of(z)$shape
  if (length(sh) == 2L && sh[1L] == sh[2L] && sh[1L] != 0L) z <- jnp_linalg_eigvals(z)
  if (length(aval_of(z)$shape) != 1L)
    jax_error("input must be 1d or non-empty square 2d array.", class = "jax_value_error")
  dt <- aval_of(z)$dtype
  n <- pnm_len(z)
  if (n == 0L) return(lax_full(integer(0), 1, dt))
  a <- lax_full(1L, 1, dt)
  for (k in seq_len(n) - 1L) {
    term <- lax_concatenate(list(lax_full(1L, 1, dt),
                                 lax_neg(lax_slice_in_dim(z, k, k + 1L))), 0L)
    a <- jnp_convolve(a, term, mode = "full")
  }
  a
}

# Port of polynomial.py:polyval.
#' @rdname jnp_polynomial
#' @export
jnp_polyval <- function(p, x, unroll = NULL) {
  px <- promote_dtypes(list(p, x))
  p_arr <- px[[1L]]
  x_arr <- px[[2L]]
  ps <- aval_of(p_arr)$shape
  shape <- broadcast_shapes_impl(list(ps[-1L], aval_of(x_arr)$shape))
  if (ps[1L] == 0L) return(lax_full(shape, 0, aval_of(x_arr)$dtype))
  r <- length(shape)
  xr <- length(aval_of(x_arr)$shape)
  if (xr > 0L && xr < r) x_arr <- pnm_rank_to(x_arr, r)
  coeffs <- lapply(rev(seq_len(ps[1L]) - 1L), function(i) {
    c <- lax_index_in_dim(p_arr, i, keepdims = FALSE)
    if (length(ps) - 1L > 0L && length(ps) - 1L < r) pnm_rank_to(c, r) else c
  })
  if (length(coeffs) == 1L) return(jnp_broadcast_to(coeffs[[1L]], shape))
  pnm_horner(x_arr, coeffs, unroll)
}

# Port of polynomial.py:polyadd.
#' @rdname jnp_polynomial
#' @export
jnp_polyadd <- function(a1, a2) {
  a <- promote_dtypes(list(a1, a2))
  n1 <- pnm_len(a[[1L]])
  n2 <- pnm_len(a[[2L]])
  if (n2 <= n1) at_add(a[[1L]], sl(-n2), value = a[[2L]])
  else at_add(a[[2L]], sl(-n1), value = a[[1L]])
}

# Port of polynomial.py:polysub.
#' @rdname jnp_polynomial
#' @export
jnp_polysub <- function(a1, a2) {
  a <- promote_dtypes(list(a1, a2))
  jnp_polyadd(a[[1L]], lax_neg(a[[2L]]))
}

# Port of polynomial.py:polyint.
#' @rdname jnp_polynomial
#' @export
jnp_polyint <- function(p, m = 1L, k = NULL) {
  m <- pnm_int(m, "'m' argument of jnp.polyint")
  pk <- promote_dtypes_inexact(list(p, if (is.null(k)) 0L else k))
  p_arr <- pk[[1L]]
  k_arr <- pk[[2L]]
  if (m < 0L) jax_error("Order of integral must be positive (see polyder)", class = "jax_value_error")
  k_arr <- pnm_atleast_1d(k_arr)
  if (identical(aval_of(k_arr)$shape, 1L)) k_arr <- jnp_broadcast_to(k_arr, m)
  if (!identical(aval_of(k_arr)$shape, m))
    jax_error("k must be a scalar or a rank-1 array of length 1 or m.", class = "jax_value_error")
  if (m == 0L) return(p_arr)
  dt <- aval_of(p_arr)$dtype
  grid <- lax_sub(lax_expand_dims(lax_iota(dt, pnm_len(p_arr) + m), 0L),
                  lax_expand_dims(lax_iota(dt, m), 1L))
  coeff <- lax_rev(red_prod(jnp_maximum(1, grid), axis = 0L), 0L)
  jnp_true_divide(lax_concatenate(list(p_arr, k_arr), 0L), coeff)
}

# Port of polynomial.py:polyder.
#' @rdname jnp_polynomial
#' @export
jnp_polyder <- function(p, m = 1L) {
  m <- pnm_int(m, "'m' argument of jnp.polyder")
  p_arr <- promote_dtypes_inexact(list(p))[[1L]]
  if (m < 0L) jax_error("Order of derivative must be positive", class = "jax_value_error")
  if (m == 0L) return(p_arr)
  dt <- aval_of(p_arr)$dtype
  n <- pnm_len(p_arr)
  lo <- lax_expand_dims(jnp_arange(m, max(n, m), dtype = dt), 0L)
  coeff <- red_prod(lax_sub(lo, lax_expand_dims(lax_iota(dt, m), 1L)), axis = 0L)
  jnp_multiply(lax_slice_in_dim(p_arr, 0L, max(n - m, 0L)), lax_rev(coeff, 0L))
}

# Port of polynomial.py:polymul.
#' @rdname jnp_polynomial
#' @export
jnp_polymul <- function(a1, a2, trim_leading_zeros = FALSE) {
  a <- promote_dtypes_inexact(list(a1, a2))
  a1_arr <- a[[1L]]
  a2_arr <- a[[2L]]
  if (trim_leading_zeros && (pnm_len(a1_arr) > 1L || pnm_len(a2_arr) > 1L)) {
    a1_arr <- pnm_trim_leading_zeros(a1_arr)
    a2_arr <- pnm_trim_leading_zeros(a2_arr)
  }
  if (pnm_len(a1_arr) == 0L) a1_arr <- lax_full(1L, 0, aval_of(a2_arr)$dtype)
  if (pnm_len(a2_arr) == 0L) a2_arr <- lax_full(1L, 0, aval_of(a1_arr)$dtype)
  jnp_convolve(a1_arr, a2_arr, mode = "full")
}

# Port of polynomial.py:polydiv.
#' @rdname jnp_polynomial
#' @export
jnp_polydiv <- function(u, v, trim_leading_zeros = FALSE) {
  uv <- promote_dtypes_inexact(list(u, v))
  u_arr <- uv[[1L]]
  v_arr <- uv[[2L]]
  m <- pnm_len(u_arr) - 1L
  n <- pnm_len(v_arr) - 1L
  scale <- jnp_true_divide(1, lax_index_in_dim(v_arr, 0L, keepdims = FALSE))
  q <- lax_full(max(m - n + 1L, 1L), 0, aval_of(u_arr)$dtype)
  for (k in seq_len(max(m - n + 1L, 0L)) - 1L) {
    d <- jnp_multiply(scale, lax_index_in_dim(u_arr, k, keepdims = FALSE))
    q <- at_set(q, k, value = d)
    u_arr <- at_add(u_arr, sl(k, k + n + 1L), value = jnp_multiply(lax_neg(d), v_arr))
  }
  if (trim_leading_zeros)
    u_arr <- pnm_trim_zeros_tol(u_arr, sqrt(finfo(aval_of(u_arr)$dtype)$eps))
  list(q, u_arr)
}
