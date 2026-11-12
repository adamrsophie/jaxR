# jax.scipy.linalg and jax.scipy.sparse.linalg (ports of jax/_src/scipy/linalg.py,
# jax/_src/third_party/scipy/linalg.py and jax/_src/scipy/sparse/linalg.py).
#
# JAX wraps most of these functions in jit; here they run primitive by primitive (the
# values are the same) and can be wrapped in jit() by the caller. Functions returning
# tuples in JAX return unnamed lists. The scipy arguments overwrite_*, check_finite,
# lwork, turbo, debug and lapack_driver are accepted and ignored, as in JAX.

# ---- helpers ---------------------------------------------------------------------------

# Port of jnp.vectorize(f, signature = ...) (jax/_src/numpy/vectorize.py) for the
# signatures used here: `core` holds the number of core dimensions of each input. The
# batch dimensions are broadcast together, collapsed to one axis and mapped with vmap;
# the outputs (one array or a list of arrays) get them back.
scl_vectorize <- function(f, core) {
  force(f)
  force(core)
  function(...) {
    args <- lapply(list(...), as_jax)
    shapes <- lapply(args, function(a) aval_of(a)$shape)
    batch <- broadcast_shapes_impl(Map(function(s, k) utils::head(s, length(s) - k),
                                       shapes, core))
    if (!length(batch)) return(do.call(f, args))
    nb <- prod_int(batch)
    args <- Map(function(a, s, k) {
      cs <- utils::tail(s, k)
      lax_reshape(linalg_broadcast_to(a, c(batch, cs)), c(nb, cs))
    }, args, shapes, core)
    out <- do.call(vmap(f), args)
    back <- function(o) lax_reshape(o, c(batch, aval_of(o)$shape[-1L]))
    if (is_jax_value(out)) back(out) else lapply(out, back)
  }
}

scl_shape <- function(x) aval_of(x)$shape
scl_ndim <- function(x) length(aval_of(x)$shape)
scl_dtype <- function(x) aval_of(x)$dtype
scl_hi_matmul <- function(a, b) jnp_matmul(a, b, precision = "highest")
scl_hi_dot <- function(a, b) jnp_dot(a, b, precision = "highest")

# jnp.atleast_1d / jnp.atleast_2d of one array.
scl_atleast_1d <- function(x) {
  x <- as_jax(x)
  if (scl_ndim(x) == 0L) lax_reshape(x, 1L) else x
}
scl_atleast_2d <- function(x) {
  x <- as_jax(x)
  switch(as.character(min(scl_ndim(x), 2L)),
         "0" = lax_reshape(x, c(1L, 1L)), "1" = lax_expand_dims(x, 0L), x)
}

# Port of dtypes.to_complex_dtype.
scl_complex_dtype <- function(dt) complex_of(to_inexact_dtype(dt))

# Port of jax/_src/numpy/util.py:promote_dtypes_complex.
scl_promote_complex <- function(args) {
  args <- promote_dtypes(args)
  dt <- scl_complex_dtype(scl_dtype(args[[1L]]))
  lapply(args, function(a) lax_convert_element_type(a, dt))
}

# The FutureWarning of the batched 1D solves of cho_solve, lu_solve and
# solve_triangular; returns the core rank of b for scl_vectorize.
scl_b_core <- function(a, b, name, call) {
  if (scl_ndim(b) == 1L) return(1L)
  if (scl_ndim(a) == scl_ndim(b) + 1L && utils::tail(scl_shape(a), 1L) == utils::tail(scl_shape(b), 1L)) {
    warning("jax.scipy.linalg.", name, ": batched 1D solves with b.ndim > 1 are deprecated, ",
            "and in the future will be treated as a batched 2D solve. Use ", call,
            " to avoid this warning.", call. = FALSE)
    return(1L)
  }
  2L
}

scl_check_square <- function(x, msg) {
  s <- scl_shape(x)
  r <- length(s)
  if (r < 2L || s[r] != s[r - 1L]) jax_error(msg, class = "jax_value_error")
}

# ---- decompositions and solvers ------------------------------------------------------

#' Linear algebra (jax.scipy.linalg)
#'
#' Ports of the decompositions and solvers of `jax.scipy.linalg`. Inputs are promoted
#' to an inexact dtype and most functions work on batches (leading dimensions).
#' Functions that return tuples in JAX return unnamed lists: `scipy_linalg_svd()` gives
#' `list(u, s, vh)`, `scipy_linalg_eigh()` gives `list(w, v)`, `scipy_linalg_lu()` gives
#' `list(p, l, u)` (or `list(pl, u)`), `scipy_linalg_qr()` gives `list(q, r)` (plus the
#' pivots, or only `list(r)` for `mode = "r"`), `scipy_linalg_schur()` gives `list(t, z)`.
#' The scipy arguments `overwrite_a`, `overwrite_b`, `overwrite_c`, `check_finite`,
#' `lwork`, `turbo`, `debug` and `lapack_driver` are accepted and ignored, as in JAX.
#' @param a,b Arrays (`a` is a matrix or a batch of matrices).
#' @param lower Whether the lower triangle is used (or returned).
#' @param overwrite_a,overwrite_b,overwrite_c,check_finite,lwork,turbo,debug,lapack_driver
#'   Ignored (scipy compatibility).
#' @param c_and_lower `list(c, lower)` as returned by `scipy_linalg_cho_factor()`.
#' @param full_matrices Whether to compute the full `u` and `vh`.
#' @param compute_uv Whether to compute `u` and `vh`.
#' @param eigvals_only Whether to return only the eigenvalues.
#' @param eigvals Must be `NULL` (only the full spectrum is implemented).
#' @param type Generalized problem type: 1 (`a v = w b v`), 2 (`a b v = w v`) or 3
#'   (`b a v = w v`).
#' @param output `"real"` or `"complex"` Schur form.
#' @param lu_and_piv `list(lu, piv)` as returned by `scipy_linalg_lu_factor()`.
#' @param trans 0 (`"N"`), 1 (`"T"`) or 2 (`"C"`): which system to solve.
#' @param permute_l Whether to return `list(p %*% l, u)`.
#' @param mode For `qr`: `"full"`, `"economic"` or `"r"`; for `qr_multiply`: `"right"`
#'   (`c %*% q`) or `"left"` (`q %*% c`).
#' @param pivoting Whether to use column pivoting (the pivots are 0-based).
#' @param c Array multiplied by `q`.
#' @param conjugate Whether to use the conjugate transpose of `q`.
#' @param assume_a `"gen"`, `"sym"`, `"her"` or `"pos"` (only `"pos"` uses a Cholesky
#'   solve).
#' @param unit_diagonal Whether the diagonal of `a` is assumed to be ones.
#' @param calc_q Whether to also return the unitary factor `q` (`list(h, q)`).
#' @return Arrays, or lists of arrays as described above.
#' @name scipy_linalg
NULL

scl_cholesky <- function(a, lower) {
  a <- nl_inexact(a)
  l <- lax_linalg_cholesky(if (lower) a else linalg_H(a), symmetrize_input = FALSE)
  if (lower) l else linalg_H(l)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_cholesky <- function(a, lower = FALSE, overwrite_a = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:cholesky
  scl_cholesky(a, isTRUE(lower))
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_cho_factor <- function(a, lower = FALSE, overwrite_a = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:cho_factor
  list(scipy_linalg_cholesky(a, lower = lower), lower)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_cho_solve <- function(c_and_lower, b, overwrite_b = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:_cho_solve
  lower <- isTRUE(c_and_lower[[2L]])
  cb <- promote_dtypes_inexact(list(c_and_lower[[1L]], b))
  c <- cb[[1L]]
  b <- cb[[2L]]
  core <- c(2L, scl_b_core(c, b, "cho_solve", "cho_solve(c_and_lower, b[..., None]).squeeze(-1)"))
  b <- scl_vectorize(function(c, b) {
    lax_linalg_triangular_solve(c, b, left_side = TRUE, lower = lower, transpose_a = !lower,
                                conjugate_a = !lower)
  }, core)(c, b)
  scl_vectorize(function(c, b) {
    lax_linalg_triangular_solve(c, b, left_side = TRUE, lower = lower, transpose_a = lower,
                                conjugate_a = lower)
  }, core)(c, b)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_svd <- function(a, full_matrices = TRUE, compute_uv = TRUE, overwrite_a = FALSE,
                             check_finite = TRUE, lapack_driver = "gesdd") {
  # Port of jax/_src/scipy/linalg.py:_svd
  lax_linalg_svd(nl_inexact(a), full_matrices = full_matrices, compute_uv = compute_uv)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_det <- function(a, overwrite_a = FALSE, check_finite = TRUE) jnp_linalg_det(a)

# Port of jax/_src/scipy/linalg.py:_eigh_generalized.
scl_eigh_generalized <- function(a, b, lower, eigvals_only, type) {
  l <- lax_linalg_cholesky(b)
  a <- (a + linalg_H(a)) / 2
  if (type == 1L) {
    c <- lax_linalg_triangular_solve(l, a, left_side = TRUE, lower = TRUE)
    c <- lax_linalg_triangular_solve(l, c, left_side = FALSE, lower = TRUE, transpose_a = TRUE,
                                     conjugate_a = TRUE)
  } else {
    c <- scl_hi_matmul(linalg_H(l), a)
    c <- scl_hi_matmul(c, l)
  }
  vw <- lax_linalg_eigh(c, lower = lower)
  v <- vw[[1L]]
  w <- vw[[2L]]
  if (eigvals_only) return(w)
  v <- if (type == 3L) {
    scl_hi_matmul(l, v)
  } else {
    lax_linalg_triangular_solve(l, v, left_side = TRUE, lower = TRUE, transpose_a = TRUE,
                                conjugate_a = TRUE)
  }
  list(w, v)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_eigh <- function(a, b = NULL, lower = TRUE, eigvals_only = FALSE,
                              overwrite_a = FALSE, overwrite_b = FALSE, turbo = TRUE,
                              eigvals = NULL, type = 1L, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:_eigh
  if (!(type %in% 1:3)) jax_error("type must be 1, 2, or 3.", class = "jax_value_error")
  if (!is.null(eigvals))
    jax_error("Only the eigvals=None case of eigh is implemented.",
              class = "jax_not_implemented_error")
  a <- as_jax(a)
  scl_check_square(a, paste0("Argument a to linalg.eigh must have shape [..., n, n], got ",
                             py_shape(scl_shape(a)), "."))
  lower <- isTRUE(lower)
  eigvals_only <- isTRUE(eigvals_only)
  if (!is.null(b)) {
    ab <- promote_dtypes_inexact(list(a, b))
    scl_check_square(ab[[2L]], paste0("Argument b to linalg.eigh must have shape [..., n, n], got ",
                                      py_shape(scl_shape(ab[[2L]])), "."))
    return(scl_vectorize(function(a, b) scl_eigh_generalized(a, b, lower, eigvals_only, type),
                         c(2L, 2L))(ab[[1L]], ab[[2L]]))
  }
  vw <- lax_linalg_eigh(nl_inexact(a), lower = lower)
  if (eigvals_only) vw[[2L]] else list(vw[[2L]], vw[[1L]])
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_schur <- function(a, output = "real") {
  # Port of jax/_src/scipy/linalg.py:schur
  if (!(identical(output, "real") || identical(output, "complex")))
    jax_error("Expected 'output' to be either 'real' or 'complex', got output='", output, "'.",
              class = "jax_value_error")
  a <- as_jax(a)
  if (output == "complex") a <- lax_convert_element_type(a, scl_complex_dtype(scl_dtype(a)))
  lax_linalg_schur(a)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_inv <- function(a, overwrite_a = FALSE, check_finite = TRUE) jnp_linalg_inv(a)

#' @rdname scipy_linalg
#' @export
scipy_linalg_lu_factor <- function(a, overwrite_a = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:lu_factor
  f <- lax_linalg_lu(nl_inexact(a))
  list(f[[1L]], f[[2L]])
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_lu_solve <- function(lu_and_piv, b, trans = 0L, overwrite_b = FALSE,
                                  check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:lu_solve
  pivots <- as_jax(lu_and_piv[[2L]])
  lb <- promote_dtypes_inexact(list(lu_and_piv[[1L]], b))
  lu <- lb[[1L]]
  b <- lb[[2L]]
  m <- utils::tail(scl_shape(lu), 2L)[1L]
  perm <- lax_linalg_lu_pivots_to_permutation(pivots, m)
  core <- c(2L, 1L, scl_b_core(lu, b, "lu_solve", "lu_solve(lu_and_piv, b[..., None]).squeeze(-1)"))
  scl_vectorize(function(lu, perm, b) lu_solve(lu, perm, b, trans = trans), core)(lu, perm, b)
}

# Port of jax/_src/scipy/linalg.py:_lu.
scl_lu_2d <- function(a, permute_l) {
  f <- lax_linalg_lu(a)
  lu <- f[[1L]]
  permutation <- f[[3L]]
  dt <- scl_dtype(a)
  s <- scl_shape(a)
  m <- s[1L]
  n <- s[2L]
  pdt <- scl_dtype(permutation)
  p <- lax_convert_element_type(
    lax_eq(lax_broadcast_in_dim(permutation, c(m, m), 1L), lax_broadcasted_iota(pdt, c(m, m), 0L)),
    dt)
  k <- min(m, n)
  l <- lax_slice(linalg_tril(lu, -1L), c(0L, 0L), c(m, k)) + jnp_eye(m, k, dtype = dt)
  u <- lax_slice(linalg_triu(lu), c(0L, 0L), c(k, n))
  if (permute_l) list(scl_hi_matmul(p, l), u) else list(jnp_real(p), l, u)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_lu <- function(a, permute_l = FALSE, overwrite_a = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:lu
  permute_l <- isTRUE(permute_l)
  scl_vectorize(function(a) scl_lu_2d(a, permute_l), 2L)(nl_inexact(a))
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_qr <- function(a, overwrite_a = FALSE, lwork = NULL, mode = "full",
                            pivoting = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:_qr
  full_matrices <- if (identical(mode, "full") || identical(mode, "r")) {
    TRUE
  } else if (identical(mode, "economic")) {
    FALSE
  } else {
    jax_error("Unsupported QR decomposition mode '", mode, "'", class = "jax_value_error")
  }
  pivoting <- isTRUE(pivoting)
  f <- lax_linalg_qr(nl_inexact(a), pivoting = pivoting, full_matrices = full_matrices)
  if (mode == "r") return(if (pivoting) list(f[[2L]], f[[3L]]) else list(f[[2L]]))
  if (pivoting) list(f[[1L]], f[[2L]], f[[3L]]) else list(f[[1L]], f[[2L]])
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_qr_multiply <- function(a, c, mode = "right", pivoting = FALSE, conjugate = FALSE,
                                     overwrite_a = FALSE, overwrite_c = FALSE) {
  # Port of jax/_src/scipy/linalg.py:qr_multiply
  ac <- promote_dtypes_inexact(list(a, c))
  a <- ac[[1L]]
  c <- ac[[2L]]
  if (!(identical(mode, "right") || identical(mode, "left")))
    jax_error("mode must be 'right' or 'left', got '", mode, "'", class = "jax_value_error")
  onedim <- scl_ndim(c) == 1L
  if (onedim) c <- lax_expand_dims(c, if (mode == "left") 1L else 0L)
  as_ <- scl_shape(a)
  cs <- scl_shape(c)
  m <- utils::tail(as_, 2L)[1L]
  n <- utils::tail(as_, 1L)
  k <- min(m, n)
  if (mode == "left") {
    if (utils::tail(cs, 2L)[1L] != k)
      jax_error("Array shapes are not compatible for Q @ c operation: a has shape ",
                py_shape(as_), " so Q has ", k, " columns, but c has ",
                utils::tail(cs, 2L)[1L], " rows (expected ", k, ").", class = "jax_value_error")
  } else if (utils::tail(cs, 1L) != m) {
    jax_error("Array shapes are not compatible for c @ Q operation: a has shape ", py_shape(as_),
              " so Q has ", m, " rows, but c has ", utils::tail(cs, 1L), " columns (expected ",
              m, ").", class = "jax_value_error")
  }
  batch <- broadcast_shapes_impl(list(utils::head(as_, -2L), utils::head(cs, -2L)))
  a <- jnp_broadcast_to(a, c(batch, utils::tail(as_, 2L)))
  c <- jnp_broadcast_to(c, c(batch, utils::tail(cs, 2L)))
  p <- NULL
  if (isTRUE(pivoting)) {
    jpvt <- jnp_zeros(c(batch, n), dtype = "int32")
    f <- geqp3(a, jpvt)
    r <- f[[1L]]
    p <- f[[2L]] - 1L
    taus <- f[[3L]]
  } else {
    f <- geqrf(a)
    r <- f[[1L]]
    taus <- f[[2L]]
  }
  if (m > n && mode == "left") {
    cs <- scl_shape(c)
    zeros <- jnp_zeros(c(utils::head(cs, -2L), m - k, utils::tail(cs, 1L)), dtype = scl_dtype(c))
    c <- jnp_concatenate(list(c, zeros), axis = -2L)
  }
  if (isTRUE(conjugate)) c <- jnp_swapaxes(c, -1L, -2L)
  left <- (mode == "left") != isTRUE(conjugate)
  cQ <- lax_linalg_ormqr(r, taus, c, left = left, transpose = conjugate)
  if (isTRUE(conjugate)) cQ <- jnp_swapaxes(cQ, -1L, -2L)
  if (mode == "right") cQ <- lax_slice_in_dim(cQ, 0L, k, axis = -1L)
  if (onedim) cQ <- jnp_ravel(cQ)
  r <- linalg_triu(lax_slice_in_dim(r, 0L, k, axis = -2L))
  if (isTRUE(pivoting)) list(cQ, r, p) else list(cQ, r)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_solve <- function(a, b, lower = FALSE, overwrite_a = FALSE, overwrite_b = FALSE,
                               debug = FALSE, check_finite = TRUE, assume_a = "gen") {
  # Port of jax/_src/scipy/linalg.py:solve and _solve
  valid <- c("gen", "sym", "her", "pos")
  if (!(is.character(assume_a) && length(assume_a) == 1L && assume_a %in% valid))
    jax_error("Expected assume_a to be one of ['gen', 'sym', 'her', 'pos']; got '",
              format(assume_a), "'", class = "jax_value_error")
  if (assume_a != "pos") return(jnp_linalg_solve(a, b))
  ab <- promote_dtypes_inexact(list(a, b))
  a <- ab[[1L]]
  b <- ab[[2L]]
  check_solve_shapes(a, b)
  factors <- scipy_linalg_cho_factor(lax_stop_gradient(a), lower = lower)
  custom_solve <- function(b) {
    lax_custom_linear_solve(
      function(x) broadcasted_matvec(a, x), b,
      solve = function(matvec, x) {
        lax_squeeze(scipy_linalg_cho_solve(factors, lax_expand_dims(x, -1L)), -1L)
      },
      symmetric = TRUE)
  }
  if (scl_ndim(a) == scl_ndim(b) + 1L) return(custom_solve(b))
  vmap(custom_solve, scl_ndim(b) - 1L, max(scl_ndim(a), scl_ndim(b)) - 1L)(b)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_solve_triangular <- function(a, b, trans = 0L, lower = FALSE, unit_diagonal = FALSE,
                                          overwrite_b = FALSE, debug = NULL, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:_solve_triangular
  ta <- if (identical(trans, 0) || identical(trans, 0L) || identical(trans, "N")) {
    c(FALSE, FALSE)
  } else if (identical(trans, 1) || identical(trans, 1L) || identical(trans, "T")) {
    c(TRUE, FALSE)
  } else if (identical(trans, 2) || identical(trans, 2L) || identical(trans, "C")) {
    c(TRUE, TRUE)
  } else {
    jax_error("Invalid 'trans' value ", format(trans), class = "jax_value_error")
  }
  ab <- promote_dtypes_inexact(list(a, b))
  a <- ab[[1L]]
  b <- ab[[2L]]
  core <- c(2L, scl_b_core(a, b, "solve_triangular", "solve_triangular(a, b[..., None]).squeeze(-1)"))
  scl_vectorize(function(a, b) {
    lax_linalg_triangular_solve(a, b, left_side = TRUE, lower = lower, transpose_a = ta[1L],
                                conjugate_a = ta[2L], unit_diagonal = unit_diagonal)
  }, core)(a, b)
}

#' @rdname scipy_linalg
#' @export
scipy_linalg_hessenberg <- function(a, calc_q = FALSE, overwrite_a = FALSE, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:hessenberg
  a <- as_jax(a)
  s <- scl_shape(a)
  n <- utils::tail(s, 1L)
  if (n == 0L) return(if (calc_q) list(jnp_zeros_like(a), jnp_zeros_like(a)) else jnp_zeros_like(a))
  f <- lax_linalg_hessenberg(a)
  a_out <- f[[1L]]
  taus <- f[[2L]]
  h <- linalg_triu(a_out, -1L)
  if (!calc_q) return(h)
  r <- length(s)
  sub <- lax_slice(a_out, c(rep(0L, r - 2L), 1L, 0L), c(utils::head(s, -2L), n, n - 1L))
  q <- lax_linalg_householder_product(sub, taus)
  batch <- utils::head(s, -2L)
  dt <- scl_dtype(a_out)
  top <- jnp_concatenate(list(jnp_ones(c(batch, 1L, 1L), dtype = dt),
                              jnp_zeros(c(batch, 1L, n - 1L), dtype = dt)), axis = -1L)
  bottom <- jnp_concatenate(list(jnp_zeros(c(batch, n - 1L, 1L), dtype = dt), q), axis = -1L)
  list(h, jnp_concatenate(list(top, bottom), axis = -2L))
}

# ---- eigh_tridiagonal ------------------------------------------------------------------

# jnp.nonzero(x, size = size)[[1]] for a 1-D boolean x (fill value 0): slot j holds the
# index i with x[i] and cumsum(x)[i] == j + 1.
scl_nonzero <- function(x, size) {
  n <- scl_shape(x)
  cs <- lax_cumsum(lax_convert_element_type(x, "int32"), 0L)
  hit <- lax_eq(lax_broadcast_in_dim(cs, c(size, n), 1L),
                lax_add(lax_broadcasted_iota("int32", c(size, n), 0L), as_jax(1L, "int32"))) &
    lax_broadcast_in_dim(x, c(size, n), 1L)
  idx <- lax_broadcasted_iota(default_int(), c(size, n), 1L)
  jnp_sum(jnp_where(hit, idx, lax_full_like(idx, 0)), axis = 1L)
}

#' Eigenvalues of a symmetric tridiagonal matrix
#'
#' Port of `jax.scipy.linalg.eigh_tridiagonal`: bisection with Sturm counts for the
#' eigenvalues and inverse iteration (with reorthogonalization of clusters) for the
#' eigenvectors.
#' @param d Diagonal (length `n`).
#' @param e Off-diagonal (length `n - 1`), same dtype as `d`.
#' @param eigvals_only Whether to return only the eigenvalues.
#' @param select `"a"` (all eigenvalues) or `"i"` (an index range); `"v"` is not
#'   implemented.
#' @param select_range For `select = "i"`, the 0-based inclusive index range `c(lo, hi)`.
#' @param tol Optional absolute tolerance of the eigenvalues.
#' @param key PRNG key for the random start vectors of the inverse iteration (default
#'   `random_key(42)`).
#' @return The eigenvalues, or `list(w, v)` with the eigenvectors in the columns of `v`.
#' @export
scipy_linalg_eigh_tridiagonal <- function(d, e, eigvals_only = FALSE, select = "a",
                                          select_range = NULL, tol = NULL, key = NULL) {
  # Port of jax/_src/scipy/linalg.py:eigh_tridiagonal
  if (!eigvals_only && is.null(key)) key <- random_key(42L)
  alpha <- as_jax(d)
  beta <- as_jax(e)
  if (scl_dtype(alpha) != scl_dtype(beta))
    jax_error("diagonal and off-diagonal values must have same dtype, got ", scl_dtype(alpha),
              " and ", scl_dtype(beta), class = "jax_type_error")
  if (!(scl_dtype(alpha) %in% c("float32", "float64", "complex64", "complex128")))
    jax_error("Only float32 and float64 inputs are supported as inputs to ",
              "jax.scipy.linalg.eigh_tridiagonal, got ", scl_dtype(alpha), " and ",
              scl_dtype(beta), class = "jax_type_error")
  n <- scl_shape(alpha)[1L]
  if (n <= 1L) {
    if (eigvals_only) return(jnp_real(alpha))
    return(list(jnp_real(alpha), jnp_eye(n, dtype = scl_dtype(alpha))))
  }
  if (is_complex_dtype(scl_dtype(alpha))) {
    alpha <- jnp_real(alpha)
    beta_sq <- jnp_real(beta * jnp_conjugate(beta))
    beta_abs <- jnp_sqrt(beta_sq)
  } else {
    beta_abs <- jnp_abs(beta)
    beta_sq <- jnp_square(beta)
  }
  m <- n - 1L
  off_diag_abs_row_sum <- jnp_concatenate(list(
    lax_slice(beta_abs, 0L, 1L),
    lax_slice(beta_abs, 0L, m - 1L) + lax_slice(beta_abs, 1L, m),
    lax_slice(beta_abs, m - 1L, m)), axis = 0L)
  lambda_est_max <- jnp_max(alpha + off_diag_abs_row_sum)
  lambda_est_min <- jnp_min(alpha - off_diag_abs_row_sum)
  t_norm <- jnp_maximum(jnp_abs(lambda_est_min), jnp_abs(lambda_est_max))
  dt <- scl_dtype(alpha)
  fi <- finfo(dt)
  eps <- as_jax(fi$eps, dt)
  safemin <- as_jax(max(1 / fi$max, (1 + fi$eps) * fi$tiny), dt)
  pivmin <- safemin * jnp_maximum(1, jnp_max(beta_sq))
  alpha0_perturbation <- jnp_square(eps * lax_index_in_dim(beta_abs, 0L, keepdims = FALSE))
  abs_tol <- eps * t_norm
  if (!is.null(tol)) abs_tol <- jnp_maximum(tol, abs_tol)
  max_it <- fi$nmant + 1L
  target_counts <- if (identical(select, "a")) {
    jnp_arange(n, dtype = "int32")
  } else if (identical(select, "i")) {
    if (is.null(select_range))
      jax_error("for select='i', select_range must be specified.", class = "jax_value_error")
    if (select_range[[1L]] > select_range[[2L]])
      jax_error("Got empty index range in select_range.", class = "jax_value_error")
    jnp_arange(select_range[[1L]], select_range[[2L]] + 1L, dtype = "int32")
  } else if (identical(select, "v")) {
    jax_error("eigh_tridiagonal(..., select='v') is not implemented",
              class = "jax_not_implemented_error")
  } else {
    jax_error("'select must have a value in {'a', 'i', 'v'}.", class = "jax_value_error")
  }
  fudge <- 2.1
  norm_slack <- as_jax(n, dt) * fudge * eps * t_norm
  lower <- lambda_est_min - norm_slack - (2 * fudge) * pivmin
  upper <- lambda_est_max + norm_slack + fudge * pivmin
  target_shape <- scl_shape(target_counts)
  lower <- jnp_broadcast_to(lower, target_shape)
  upper <- jnp_broadcast_to(upper, target_shape)
  mid <- 0.5 * (upper + lower)
  pivmin <- jnp_broadcast_to(pivmin, target_shape)
  alpha0_perturbation <- jnp_broadcast_to(alpha0_perturbation, target_shape)

  # Port of _sturm: the number of eigenvalues below each x.
  sturm <- function(x) {
    zeros <- jnp_zeros(scl_shape(x), dtype = "int32")
    ones <- jnp_ones(scl_shape(x), dtype = "int32")
    a0 <- alpha[0L]
    q <- a0 - x
    count <- jnp_where(q < 0, ones, zeros)
    q <- jnp_where(a0 == x, alpha0_perturbation, q)
    sturm_step <- function(i, q, count) {
      q <- alpha[i] - beta_sq[i - 1L] / q - x
      count <- jnp_where(q <= pivmin, count + 1L, count)
      q <- jnp_where(q <= pivmin, jnp_minimum(q, -pivmin), q)
      list(q, count)
    }
    unrolled_steps <- function(args, unroll_cnt) {
      start <- args[[1L]]
      q <- args[[2L]]
      count <- args[[3L]]
      for (j in seq_len(unroll_cnt) - 1L) {
        qc <- sturm_step(start + j, q, count)
        q <- qc[[1L]]
        count <- qc[[2L]]
      }
      list(start + unroll_cnt, q, count)
    }
    blocksize <- 16L
    st <- unrolled_steps(list(1L, q, count), (n - 1L) %% blocksize)
    lax_while_loop(function(s) s[[1L]] < n, function(s) unrolled_steps(s, blocksize), st)[[3L]]
  }

  res <- lax_while_loop(
    function(s) (s[[1L]] < max_it) & (abs_tol < jnp_max(s[[4L]] - s[[2L]])),
    function(s) {
      counts <- sturm(s[[3L]])
      lo <- jnp_where(counts <= target_counts, s[[3L]], s[[2L]])
      up <- jnp_where(counts > target_counts, s[[3L]], s[[4L]])
      list(s[[1L]] + 1L, lo, 0.5 * (lo + up), up)
    },
    list(0L, lower, mid, upper))
  mid <- res[[3L]]
  if (eigvals_only) return(mid)
  list(mid, jnp_transpose(scl_tridiagonal_eigenvectors(alpha, beta, mid, key)))
}

# Port of eigh_tridiagonal._compute_eigenvectors.
scl_tridiagonal_eigenvectors <- function(alpha, beta, eigvals, key) {
  n <- scl_shape(alpha)
  k <- scl_shape(eigvals)
  dl <- lax_pad(jnp_conjugate(beta), 0, list(c(1L, 0L, 0L)))
  du <- lax_pad(beta, 0, list(c(0L, 1L, 0L)))
  gap <- lax_slice(eigvals, 1L, k) - lax_slice(eigvals, 0L, k - 1L)
  edt <- scl_dtype(eigvals)
  eps <- as_jax(finfo(edt)$eps, edt)
  t_norm <- jnp_maximum(jnp_abs(eigvals[0L]), jnp_abs(eigvals[k - 1L]))
  gaptol <- jnp_sqrt(eps) * t_norm
  close <- gap < gaptol
  left_neighbor_close <- lax_pad(close, FALSE, list(c(1L, 0L, 0L)))
  right_neighbor_close <- lax_pad(close, FALSE, list(c(0L, 1L, 0L)))
  ortho_interval_start <- !left_neighbor_close & right_neighbor_close
  ortho_interval_end <- left_neighbor_close & !right_neighbor_close
  max_clusters <- k %/% 2L + 1L
  starts <- scl_nonzero(ortho_interval_start, max_clusters)
  ends <- scl_nonzero(ortho_interval_end, max_clusters) + 1L
  num_clusters <- jnp_sum(ortho_interval_start)
  arange_k <- jnp_arange(k)
  solve_dtype <- result_type(alpha, beta)
  base_dtype <- complex_component(solve_dtype)
  v <- jnp_astype(random_normal(key, c(k, n), dtype = base_dtype), solve_dtype)
  v <- v / jnp_astype(jnp_linalg_norm(v, axis = -1L, keepdims = TRUE), solve_dtype)

  orthogonalize_close_eigenvectors <- function(ev) {
    orthogonalize_cluster <- function(i, ev) {
      start <- starts[i]
      end <- ends[i]
      c <- end - start
      ev_padded <- lax_pad(ev, 0, list(c(0L, k, 0L), c(0L, 0L, 0L)))
      v_block <- lax_dynamic_slice(ev_padded, list(start, 0L), c(k, n))
      mask <- jnp_astype(arange_k < c, scl_dtype(ev))
      v_block_masked <- v_block * lax_expand_dims(mask, 1L)
      Q <- jnp_linalg_qr(jnp_transpose(v_block_masked))[[1L]]
      QT_masked <- jnp_transpose(Q) * lax_expand_dims(mask, 1L)
      big_zeros <- jnp_zeros(c(2L * k, n), dtype = scl_dtype(ev))
      big_update <- lax_dynamic_update_slice(big_zeros, QT_masked, list(start, 0L))
      update_full <- lax_slice(big_update, c(0L, 0L), c(k, n))
      is_in_cluster <- (arange_k >= start) & (arange_k < end)
      jnp_where(lax_expand_dims(is_in_cluster, 1L), update_full, ev)
    }
    lax_fori_loop(0L, num_clusters, orthogonalize_cluster, ev)
  }

  alpha_shifted <- lax_expand_dims(alpha, 0L) - lax_expand_dims(eigvals, 1L)
  dl_batched <- jnp_broadcast_to(lax_expand_dims(dl, 0L), c(k, n))
  du_batched <- jnp_broadcast_to(lax_expand_dims(du, 0L), c(k, n))

  inverse_iteration_step <- function(state) {
    v <- state[[2L]]
    v_new <- lax_linalg_tridiagonal_solve(
      jnp_astype(dl_batched, solve_dtype), jnp_astype(alpha_shifted, solve_dtype),
      jnp_astype(du_batched, solve_dtype), jnp_astype(lax_expand_dims(v, 2L), solve_dtype),
      perturb_singular = TRUE)
    v_new <- lax_squeeze(v_new, 2L)
    nrm_v_new <- jnp_linalg_norm(v_new, axis = -1L, keepdims = TRUE)
    v_new <- v_new / jnp_astype(nrm_v_new, scl_dtype(v_new))
    nrm_v_new <- lax_squeeze(nrm_v_new, 1L)
    v_new <- orthogonalize_close_eigenvectors(v_new)
    list(state[[1L]] + 1L, v_new, nrm_v_new, state[[3L]])
  }
  continue_iteration <- function(state) {
    max_it <- 5L
    min_norm_growth <- 0.1
    norm_growth_factor <- 1 + min_norm_growth
    (state[[1L]] < max_it) & jnp_any(state[[3L]] >= norm_growth_factor * state[[4L]])
  }
  lax_while_loop(continue_iteration, inverse_iteration_step,
                 list(0L, v, jnp_ones(k, dtype = edt), jnp_zeros(k, dtype = edt)))[[2L]]
}

# ---- matrix functions ------------------------------------------------------------------

#' Matrix functions (jax.scipy.linalg)
#'
#' Ports of `jax.scipy.linalg.expm` (Pade approximation with scaling and squaring),
#' `expm_frechet` (the Frechet derivative, computed as the forward mode derivative of
#' `expm`, as in JAX), `sqrtm` (via the complex Schur form), `funm` (Schur-Parlett),
#' `polar`, `rsf2csf` and `solve_sylvester`.
#' @param A,B,C Square matrices (`C` has the shape of `A %*% X`).
#' @param upper_triangular Whether `A` is upper triangular (a faster solve is used).
#' @param max_squarings Maximum number of squarings; larger norms give `NaN`.
#' @param E Direction of the Frechet derivative (same shape as `A`).
#' @param method For `expm_frechet`: ignored. For `polar`: `"qdwh"` or `"svd"`. For
#'   `solve_sylvester`: `"schur"` or `"eigen"`.
#' @param compute_expm Whether to also return `expm(A)` (`list(expm_A, frechet)`).
#' @param blocksize Must be 1 (the blocked algorithm is not implemented).
#' @param func Function applied elementwise to the eigenvalues (a vector).
#' @param disp If `FALSE`, also return an error estimate (`list(F, err)`).
#' @param a Matrix (or batch of matrices) to decompose.
#' @param side `"right"` (`a = u p`) or `"left"` (`a = p u`).
#' @param eps,max_iterations Convergence parameters of the QDWH iteration.
#' @param T,Z Real Schur form and Schur vectors (as returned by `scipy_linalg_schur()`).
#' @param check_finite Ignored.
#' @param tol Threshold on `|eig(A) + eig(B)|` below which the result is `NaN`.
#' @return The matrix function, or the lists described above: `polar` gives
#'   `list(u, p)` and `rsf2csf` gives `list(T, Z)` in complex Schur form.
#' @name scipy_linalg_matrix_functions
NULL

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_expm <- function(A, upper_triangular = FALSE, max_squarings = 16L) {
  # Port of jax/_src/scipy/linalg.py:expm
  A <- promote_dtypes_inexact(list(A))[[1L]]
  s <- scl_shape(A)
  if (length(s) < 2L || s[length(s)] != s[length(s) - 1L])
    jax_error("Expected A to be a (batched) square matrix, got A.shape=", py_shape(s), ".",
              class = "jax_value_error")
  if (length(s) > 2L)
    return(scl_vectorize(function(a) scipy_linalg_expm(a, upper_triangular, max_squarings), 2L)(A))
  pq <- scl_calc_P_Q(A)
  n_squarings <- pq[[3L]]
  lax_cond(n_squarings > max_squarings,
           function(args) jnp_full_like(args[[1L]], NaN),
           function(args) {
             R <- scl_solve_P_Q(args[[2L]], args[[3L]], upper_triangular)
             scl_squaring(R, n_squarings, max_squarings)
           },
           list(A, pq[[1L]], pq[[2L]]))
}

# Port of jax/_src/scipy/linalg.py:_calc_P_Q. The thresholds are written in hexadecimal
# so they are the exact doubles of the Python literals.
scl_calc_P_Q <- function(A) {
  s <- scl_shape(A)
  if (length(s) != 2L || s[1L] != s[2L]) jax_error("expected A to be a square matrix", class = "jax_value_error")
  A_L1 <- jnp_linalg_norm(A, 1)
  dt <- scl_dtype(A)
  if (dt %in% c("float64", "complex128")) {
    maxnorm <- 0x1.57cd8b04397b4p+2  # 5.371920351148152
    conds <- c(0x1.ea12c7ff600eep-7, 0x1.0408cdb7a7ca9p-2, 0x1.e69d2cc8c51d7p-1,
               0x1.0c864830ca293p+1)  # 0.01495585217958292 0.253939833006323 0.9504178996162932 2.097847961257068
    pades <- list(scl_pade3, scl_pade5, scl_pade7, scl_pade9, scl_pade13)
  } else if (dt %in% c("float32", "complex64")) {
    maxnorm <- 0x1.f67e265256afbp+1  # 3.92572478313866
    conds <- c(0x1.b4180d5a12223p-2, 0x1.e151af96a884ep+0)  # 0.4258730016922831 1.880152677804762
    pades <- list(scl_pade3, scl_pade5, scl_pade7)
  } else {
    jax_error("A.dtype=", dt, " is not supported.", class = "jax_type_error")
  }
  n_squarings <- jnp_maximum(0, jnp_floor(jnp_log2(A_L1 / maxnorm)))
  A <- A / jnp_power(2, jnp_astype(n_squarings, dt))
  conds <- as_jax(conds, scl_dtype(A_L1))
  # jnp.digitize(A_L1, conds): the number of thresholds at or below A_L1.
  idx <- jnp_sum(lax_convert_element_type(conds <= A_L1, "int32"))
  UV <- lax_switch(idx, pades, A)
  list(UV[[1L]] + UV[[2L]], -UV[[1L]] + UV[[2L]], n_squarings)
}

# Port of jax/_src/scipy/linalg.py:_solve_P_Q.
scl_solve_P_Q <- function(P, Q, upper_triangular = FALSE) {
  if (upper_triangular) scipy_linalg_solve_triangular(Q, P) else jnp_linalg_solve(Q, P)
}

# Port of jax/_src/scipy/linalg.py:_squaring.
scl_squaring <- function(R, n_squarings, max_squarings) {
  lax_scan(function(c, i) {
    list(lax_cond(i < n_squarings, function(x) scl_hi_dot(x, x), function(x) x, c), NULL)
  }, R, jnp_arange(max_squarings, dtype = scl_dtype(n_squarings)))[[1L]]
}

# Ports of jax/_src/scipy/linalg.py:_pade3 ... _pade13.
scl_pade_ident <- function(A) jnp_eye(scl_shape(A)[1L], scl_shape(A)[2L], dtype = scl_dtype(A))

scl_pade3 <- function(A) {
  b <- c(120., 60., 12., 1.)
  ident <- scl_pade_ident(A)
  A2 <- scl_hi_dot(A, A)
  U <- scl_hi_dot(A, b[4L] * A2 + b[2L] * ident)
  V <- b[3L] * A2 + b[1L] * ident
  list(U, V)
}

scl_pade5 <- function(A) {
  b <- c(30240., 15120., 3360., 420., 30., 1.)
  ident <- scl_pade_ident(A)
  A2 <- scl_hi_dot(A, A)
  A4 <- scl_hi_dot(A2, A2)
  U <- scl_hi_dot(A, b[6L] * A4 + b[4L] * A2 + b[2L] * ident)
  V <- b[5L] * A4 + b[3L] * A2 + b[1L] * ident
  list(U, V)
}

scl_pade7 <- function(A) {
  b <- c(17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
  ident <- scl_pade_ident(A)
  A2 <- scl_hi_dot(A, A)
  A4 <- scl_hi_dot(A2, A2)
  A6 <- scl_hi_dot(A4, A2)
  U <- scl_hi_dot(A, b[8L] * A6 + b[6L] * A4 + b[4L] * A2 + b[2L] * ident)
  V <- b[7L] * A6 + b[5L] * A4 + b[3L] * A2 + b[1L] * ident
  list(U, V)
}

scl_pade9 <- function(A) {
  b <- c(17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880.,
         3960., 90., 1.)
  ident <- scl_pade_ident(A)
  A2 <- scl_hi_dot(A, A)
  A4 <- scl_hi_dot(A2, A2)
  A6 <- scl_hi_dot(A4, A2)
  A8 <- scl_hi_dot(A6, A2)
  U <- scl_hi_dot(A, b[10L] * A8 + b[8L] * A6 + b[6L] * A4 + b[4L] * A2 + b[2L] * ident)
  V <- b[9L] * A8 + b[7L] * A6 + b[5L] * A4 + b[3L] * A2 + b[1L] * ident
  list(U, V)
}

scl_pade13 <- function(A) {
  # The two largest coefficients have 17 digits; written in hexadecimal to be exact.
  b <- c(0x1.cc2e660954400p+55, 0x1.cc2e660954400p+54, 7771770303897600., 1187353796428800.,
         129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
         40840800., 960960., 16380., 182., 1.)
  ident <- scl_pade_ident(A)
  A2 <- scl_hi_dot(A, A)
  A4 <- scl_hi_dot(A2, A2)
  A6 <- scl_hi_dot(A4, A2)
  U <- scl_hi_dot(A, scl_hi_dot(A6, b[14L] * A6 + b[12L] * A4 + b[10L] * A2) + b[8L] * A6 +
                    b[6L] * A4 + b[4L] * A2 + b[2L] * ident)
  V <- scl_hi_dot(A6, b[13L] * A6 + b[11L] * A4 + b[9L] * A2) + b[7L] * A6 + b[5L] * A4 +
    b[3L] * A2 + b[1L] * ident
  list(U, V)
}

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_expm_frechet <- function(A, E, method = NULL, compute_expm = TRUE) {
  # Port of jax/_src/scipy/linalg.py:expm_frechet
  A_arr <- as_jax(A)
  E_arr <- as_jax(E)
  As <- scl_shape(A_arr)
  Es <- scl_shape(E_arr)
  if (length(As) < 2L || As[length(As) - 1L] != As[2L])
    jax_error("expected A to be a (batched) square matrix, got A.shape=", py_shape(As),
              class = "jax_value_error")
  if (length(Es) < 2L || Es[length(Es) - 1L] != Es[length(Es)])
    jax_error("expected E to be a (batched) square matrix, got E.shape=", py_shape(Es),
              class = "jax_value_error")
  if (!identical(As, Es))
    jax_error("expected A and E to be the same shape, got A.shape=", py_shape(As), " E.shape=",
              py_shape(Es), class = "jax_value_error")
  out <- jvp(function(a) scipy_linalg_expm(a, upper_triangular = FALSE, max_squarings = 16L),
             list(A_arr), list(E_arr))
  if (isTRUE(compute_expm)) list(out[[1L]], out[[2L]]) else out[[2L]]
}

# Port of jax/_src/scipy/linalg.py:_polar_2d. JAX sets the default matmul precision to
# float32 here, which does not change CPU results.
scl_polar_2d <- function(a, side, method, eps, max_iterations) {
  s <- scl_shape(a)
  m <- s[1L]
  n <- s[2L]
  if (identical(method, "qdwh")) {
    if (m >= n && side == "right") {
      f <- lax_linalg_qdwh(a, is_hermitian = FALSE, eps = eps)
      unitary <- f[[1L]]
      posdef <- f[[2L]]
    } else if (m < n && side == "left") {
      f <- lax_linalg_qdwh(linalg_H(a), is_hermitian = FALSE, eps = eps)
      posdef <- linalg_H(f[[2L]])
      unitary <- linalg_H(f[[1L]])
    } else {
      jax_error("method='qdwh' only supports mxn matrices where m < n where side='right' and ",
                "m >= n side='left', got ", py_shape(s), " with side='", side, "'",
                class = "jax_not_implemented_error")
    }
  } else if (identical(method, "svd")) {
    usv <- lax_linalg_svd(a, full_matrices = FALSE)
    u <- usv[[1L]]
    vh <- usv[[3L]]
    sv <- lax_expand_dims(jnp_astype(usv[[2L]], scl_dtype(u)), 0L)
    unitary <- jnp_matmul(u, vh)
    posdef <- if (side == "right") jnp_matmul(linalg_H(vh) * sv, vh) else jnp_matmul(u * sv, linalg_H(u))
  } else {
    jax_error("Unknown polar decomposition method ", format(method), ".", class = "jax_value_error")
  }
  list(unitary, posdef)
}

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_polar <- function(a, side = "right", method = "qdwh", eps = NULL,
                               max_iterations = NULL) {
  # Port of jax/_src/scipy/linalg.py:polar
  arr <- as_jax(a)
  if (scl_ndim(arr) < 2L)
    jax_error("The input `a` must be at least a 2-D array.", class = "jax_value_error")
  if (!(identical(side, "right") || identical(side, "left")))
    jax_error("The argument `side` must be either 'right' or 'left'.", class = "jax_value_error")
  scl_vectorize(function(x) scl_polar_2d(x, side, method, eps, max_iterations), 2L)(arr)
}

# Port of jax/_src/scipy/linalg.py:_sqrtm_triu. JAX starts the inner sum at 0.0 and
# lets the while loop promote it; here it starts as a zero of the dtype of T.
scl_sqrtm_triu <- function(T) {
  diag <- jnp_sqrt(extract_diagonal(T))
  n <- scl_shape(diag)
  U <- construct_diagonal(diag)
  zero <- lax_full(integer(0), 0, scl_dtype(U))
  i_loop <- function(l, data) {
    j <- data[[1L]]
    U <- data[[2L]]
    i <- j - 1L - l
    s <- lax_fori_loop(i + 1L, j, function(k, val) val + U[i, k] * U[k, j], zero)
    value <- jnp_where(T[i, j] == s, 0.0, (T[i, j] - s) / (diag[i] + diag[j]))
    list(j, at_set(U, i, j, value = value))
  }
  lax_fori_loop(0L, n, function(j, U) lax_fori_loop(0L, j, i_loop, list(j, U))[[2L]], U)
}

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_sqrtm <- function(A, blocksize = 1L) {
  # Port of jax/_src/scipy/linalg.py:sqrtm and _sqrtm
  if (blocksize > 1L)
    jax_error("Blocked version is not implemented yet.", class = "jax_not_implemented_error")
  tz <- scipy_linalg_schur(as_jax(A), output = "complex")
  Z <- tz[[2L]]
  sqrt_T <- scl_sqrtm_triu(tz[[1L]])
  scl_hi_matmul(scl_hi_matmul(Z, sqrt_T), jnp_conjugate(jnp_transpose(Z)))
}

# Port of jax/_src/scipy/linalg.py:_rsf2csf_2d.
scl_rsf2csf_2d <- function(T, Z) {
  tz <- scl_promote_complex(list(T, Z))
  T <- tz[[1L]]
  Z <- tz[[2L]]
  dt <- scl_dtype(T)
  eps <- finfo(dt)$eps
  N <- scl_shape(T)[1L]
  if (N == 1L) return(list(T, Z))
  update_T_Z <- function(m, T, Z) {
    mu <- jnp_linalg_eigvals(lax_dynamic_slice(T, list(m - 1L, m - 1L), c(2L, 2L))) - T[m, m]
    r <- jnp_astype(jnp_linalg_norm(jnp_stack(list(mu[0L], T[m, m - 1L]))), dt)
    c <- mu[0L] / r
    s <- T[m, m - 1L] / r
    G <- jnp_astype(jnp_stack(list(jnp_stack(list(jnp_conjugate(c), s)),
                                   jnp_stack(list(-s, c)))), dt)
    GH <- jnp_transpose(jnp_conjugate(G))
    T_rows <- lax_dynamic_slice_in_dim(T, m - 1L, 2L, axis = 0L)
    col_mask <- jnp_arange(N) >= m - 1L
    G_dot_T_zeroed_cols <- jnp_matmul(G, jnp_where(col_mask, T_rows, 0))
    T_rows_new <- jnp_where(!col_mask, T_rows, G_dot_T_zeroed_cols)
    T <- lax_dynamic_update_slice_in_dim(T, T_rows_new, m - 1L, axis = 0L)
    T_cols <- lax_dynamic_slice_in_dim(T, m - 1L, 2L, axis = 1L)
    row_mask <- lax_expand_dims(jnp_arange(N), 1L) < m + 1L
    T_zeroed_rows_dot_GH <- jnp_matmul(jnp_where(row_mask, T_cols, 0), GH)
    T_cols_new <- jnp_where(!row_mask, T_cols, T_zeroed_rows_dot_GH)
    T <- lax_dynamic_update_slice_in_dim(T, T_cols_new, m - 1L, axis = 1L)
    Z_cols <- lax_dynamic_slice_in_dim(Z, m - 1L, 2L, axis = 1L)
    Z <- lax_dynamic_update_slice_in_dim(Z, jnp_matmul(Z_cols, GH), m - 1L, axis = 1L)
    list(T, Z)
  }
  lax_fori_loop(1L, N, function(i, TZ) {
    m <- N - i
    T <- TZ[[1L]]
    Z <- TZ[[2L]]
    TZ <- lax_cond(jnp_abs(T[m, m - 1L]) > eps * (jnp_abs(T[m - 1L, m - 1L]) + jnp_abs(T[m, m])),
                   update_T_Z, function(m, T, Z) list(T, Z), m, T, Z)
    list(at_set(TZ[[1L]], m, m - 1L, value = 0.0), TZ[[2L]])
  }, list(T, Z))
}

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_rsf2csf <- function(T, Z, check_finite = TRUE) {
  # Port of jax/_src/scipy/linalg.py:rsf2csf
  T_arr <- as_jax(T)
  Z_arr <- as_jax(Z)
  scl_check_square(T_arr, "Input 'T' must be square.")
  scl_check_square(Z_arr, "Input 'Z' must be square.")
  if (utils::tail(scl_shape(T_arr), 1L) != utils::tail(scl_shape(Z_arr), 1L))
    jax_error("Input array shapes must match: Z: ", py_shape(scl_shape(Z_arr)), " vs. T: ",
              py_shape(scl_shape(T_arr)), class = "jax_value_error")
  scl_vectorize(scl_rsf2csf_2d, c(2L, 2L))(T_arr, Z_arr)
}

# Port of jax/_src/third_party/scipy/linalg.py:_algorithm_11_1_1.
scl_algorithm_11_1_1 <- function(F, T) {
  N <- scl_shape(T)[1L]
  minden <- jnp_abs(T[0L, 0L])
  iota <- jnp_arange(N)
  inner_loop <- function(i, p_F_minden) {
    p <- p_F_minden[[1L]]
    F <- p_F_minden[[2L]]
    minden <- p_F_minden[[3L]]
    j <- i + p
    s <- T[i - 1L, j - 1L] * (F[j - 1L, j - 1L] - F[i - 1L, i - 1L])
    T_row <- T[i - 1L]
    T_col <- T[, j - 1L]
    F_row <- F[i - 1L]
    F_col <- F[, j - 1L]
    ind <- (iota >= i) & (iota < j - 1L)
    val <- jnp_matmul(jnp_where(ind, T_row, 0), jnp_where(ind, F_col, 0)) -
      jnp_matmul(jnp_where(ind, F_row, 0), jnp_where(ind, T_col, 0))
    s <- s + val
    den <- T[j - 1L, j - 1L] - T[i - 1L, i - 1L]
    s <- jnp_where(den != 0, s / den, s)
    F <- at_set(F, i - 1L, j - 1L, value = s)
    minden <- jnp_minimum(minden, jnp_abs(den))
    list(p, F, minden)
  }
  outer_loop <- function(p, F_minden) {
    r <- lax_fori_loop(1L, N - p + 1L, inner_loop, list(p, F_minden[[1L]], F_minden[[2L]]))
    list(r[[2L]], r[[3L]])
  }
  lax_fori_loop(1L, N, outer_loop, list(F, minden))
}

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_funm <- function(A, func, disp = TRUE) {
  # Port of jax/_src/third_party/scipy/linalg.py:funm
  A_arr <- as_jax(A)
  s <- scl_shape(A_arr)
  if (length(s) != 2L || s[1L] != s[2L])
    jax_error("expected square array_like input", class = "jax_value_error")
  tz <- scipy_linalg_schur(A_arr)
  tz <- scipy_linalg_rsf2csf(tz[[1L]], tz[[2L]])
  T <- tz[[1L]]
  Z <- tz[[2L]]
  F <- construct_diagonal(func(extract_diagonal(T)))
  F <- jnp_astype(F, scl_dtype(T))
  fm <- scl_algorithm_11_1_1(F, T)
  F <- jnp_matmul(jnp_matmul(Z, fm[[1L]]), jnp_conjugate(jnp_transpose(Z)))
  if (isTRUE(disp)) return(F)
  # JAX's dtype test overwrites the float16 tolerance with the float64 one.
  tol <- if (scl_dtype(F) %in% c("float32", "complex64")) finfo("float32")$eps else finfo("float64")$eps
  minden <- jnp_where(fm[[2L]] == 0.0, tol, fm[[2L]])
  err <- jnp_where(jnp_any(jnp_isinf(F)), Inf,
                   jnp_minimum(1, jnp_maximum(tol, tol / minden *
                                                jnp_linalg_norm(linalg_triu(T, 1L), 1))))
  list(F, err)
}

# Port of jax/_src/scipy/linalg.py:_solve_sylvester_triangular_scan.
scl_sylvester_triangular <- function(R, S, F) {
  # promote_args_inexact; the operands have equal ranks, so JAX does no shape promotion.
  rsf <- promote_dtypes_inexact(list(R, S, F))
  R <- rsf[[1L]]
  S <- rsf[[2L]]
  F <- rsf[[3L]]
  m <- scl_shape(F)[1L]
  n <- scl_shape(F)[2L]
  total <- m * n
  Y0 <- jnp_zeros(total, dtype = scl_dtype(F))
  k_row <- jnp_arange(m)
  k_col <- jnp_arange(n)
  Y_flat <- lax_scan(function(Y_flat, idx) {
    i <- idx %/% n
    j <- idx %% n
    Y <- jnp_reshape(Y_flat, c(m, n))
    rhs <- F[i, j]
    row_term <- jnp_sum(jnp_where(k_row > i, R[i, ] * Y[, j], 0.0))
    col_term <- jnp_sum(jnp_where(k_col > j, Y[i, ] * S[, j], 0.0))
    rhs <- rhs - (row_term + col_term)
    val <- rhs / (R[i, i] + S[j, j])
    list(at_set(Y_flat, i * n + j, value = val), NULL)
  }, Y0, jnp_arange(total - 1L, -1L, -1L))
  jnp_reshape(Y_flat[[1L]], c(m, n))
}

# Port of jax/_src/scipy/linalg.py:_solve_sylvester_2d.
scl_sylvester_2d <- function(A, B, C, method, tol) {
  m <- utils::tail(scl_shape(C), 2L)[1L]
  n <- utils::tail(scl_shape(C), 1L)
  H <- function(x) jnp_conjugate(jnp_transpose(x))
  if (identical(method, "schur")) {
    RU <- scipy_linalg_schur(A, output = "complex")
    SV <- scipy_linalg_schur(H(B), output = "complex")
    R <- RU[[1L]]
    U <- RU[[2L]]
    V <- SV[[2L]]
    F <- jnp_matmul(jnp_matmul(H(U), jnp_astype(C, scl_dtype(R))), V)
    Y <- scl_sylvester_triangular(R, H(SV[[1L]]), F)
    X <- jnp_matmul(jnp_matmul(U, Y), H(V))
  } else if (identical(method, "eigen")) {
    ea <- jnp_linalg_eig(A)
    eb <- jnp_linalg_eig(B)
    RA <- ea[[1L]]
    UA <- ea[[2L]]
    UB <- eb[[2L]]
    F <- scipy_linalg_solve(UA, jnp_matmul(jnp_astype(C, scl_dtype(RA)), UB))
    W <- lax_expand_dims(RA, 1L) + lax_expand_dims(eb[[1L]], 0L)
    Y <- F / W
    X <- jnp_matmul(jnp_matmul(lax_slice(UA, c(0L, 0L), c(m, m)), lax_slice(Y, c(0L, 0L), c(m, n))),
                    lax_slice(scipy_linalg_inv(UB), c(0L, 0L), c(n, n)))
  } else {
    jax_error("Unrecognized method ", format(method), ". The two valid methods are either ",
              "\"schur\" or \"eigen\".", class = "jax_value_error")
  }
  if (!is_complex_dtype(scl_dtype(C))) X <- jnp_real(X)
  ev <- lax_expand_dims(jnp_linalg_eigvals(A), 1L) + lax_expand_dims(jnp_linalg_eigvals(B), 0L)
  lax_cond(jnp_any(jnp_abs(ev) < tol), function() jnp_zeros_like(X) * NaN, function() X)
}

#' @rdname scipy_linalg_matrix_functions
#' @export
scipy_linalg_solve_sylvester <- function(A, B, C, method = "schur", tol = 1e-08) {
  # Port of jax/_src/scipy/linalg.py:solve_sylvester
  abc <- promote_dtypes_inexact(list(A, B, C))
  cs <- scl_shape(abc[[3L]])
  m <- utils::tail(cs, 2L)[1L]
  n <- utils::tail(cs, 1L)
  if (!identical(utils::tail(scl_shape(abc[[1L]]), 2L), c(m, m)) ||
      !identical(utils::tail(scl_shape(abc[[2L]]), 2L), c(n, n)))
    jax_error("Incompatible shapes for Sylvester equation:\nA: ", py_shape(scl_shape(abc[[1L]])),
              "\nB: ", py_shape(scl_shape(abc[[2L]])), "\nC: ", py_shape(cs),
              class = "jax_value_error")
  scl_vectorize(function(a, b, c) scl_sylvester_2d(a, b, c, method, tol),
                c(2L, 2L, 2L))(abc[[1L]], abc[[2L]], abc[[3L]])
}

# ---- special matrices ------------------------------------------------------------------

#' Special matrices (jax.scipy.linalg)
#'
#' Ports of the matrix constructors of `jax.scipy.linalg`. `toeplitz`, `hankel`,
#' `circulant`, `leslie`, `companion`, `fiedler`, `fiedler_companion` and
#' `convolution_matrix` work on batches of vectors (leading dimensions).
#' @param ... Arrays of at most 2 dimensions (for `block_diag`).
#' @param c First column (`toeplitz`, `hankel`, `circulant`).
#' @param r Last (`hankel`) or first (`toeplitz`) row; `NULL` for the default.
#' @param f,s Fecundity coefficients and survival rates (`leslie`).
#' @param a Coefficients (`companion`, `fiedler`, `fiedler_companion`) or the filter
#'   (`convolution_matrix`).
#' @param n Matrix size (or the input length for `convolution_matrix`).
#' @param mode `"full"`, `"valid"` or `"same"`.
#' @param kind `"symmetric"` (default), `"lower"` or `"upper"`.
#' @param full Whether `helmert` includes the first row.
#' @param dtype Result dtype (`hadamard`: default integer; `dft`: default complex).
#' @param scale `NULL`, `"sqrtn"` or `"n"`.
#' @return The matrix (or batch of matrices).
#' @name scipy_linalg_special_matrices
NULL

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_block_diag <- function(...) {
  # Port of jax/_src/scipy/linalg.py:block_diag
  arrs <- list(...)
  if (!length(arrs)) arrs <- list(jnp_zeros(c(1L, 0L)))
  arrs <- promote_dtypes(arrs)
  bad <- which(vapply(arrs, scl_ndim, 1L) > 2L)
  if (length(bad))
    jax_error("Arguments to jax.scipy.linalg.block_diag must have at most 2 dimensions, got ",
              aval_str(aval_of(arrs[[bad[1L]]])), " at argument ", bad[1L] - 1L, ".",
              class = "jax_value_error")
  converted <- lapply(arrs, scl_atleast_2d)
  dt <- scl_dtype(converted[[1L]])
  cols <- vapply(converted, function(a) scl_shape(a)[2L], 1L)
  total <- sum(cols)
  start <- cumsum(c(0L, cols))
  padded <- lapply(seq_along(converted), function(i) {
    lax_pad(converted[[i]], lax_full(integer(0), 0, dt),
            list(c(0L, 0L, 0L), c(start[i], total - cols[i] - start[i], 0L)))
  })
  jnp_concatenate(padded, axis = 0L)
}

# Port of lax.conv_general_dilated_patches(x.reshape(1, L, 1), (width,), (1,), "VALID",
# dimension_numbers = ("NTC", "IOT", "NTC"), precision = HIGHEST)[0] for a vector x.
scl_patches_1d <- function(x, width) {
  L <- scl_shape(x)
  rhs <- lax_reshape(linalg_eye(scl_dtype(x), c(width, width)), c(1L, width, width))
  out <- lax_conv_general_dilated(lax_reshape(x, c(1L, L, 1L)), rhs, 1L, "VALID",
                                  dimension_numbers = c("NTC", "IOT", "NTC"),
                                  precision = "highest")
  lax_index_in_dim(out, 0L, keepdims = FALSE)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_toeplitz <- function(c, r = NULL) {
  # Port of jax/_src/scipy/linalg.py:toeplitz and _toeplitz
  c <- as_jax(c)
  r <- if (is.null(r)) jnp_conjugate(c) else as_jax(r)
  scl_vectorize(function(c, r) {
    ncols <- scl_shape(c)
    nrows <- scl_shape(r)
    if (ncols == 0L || nrows == 0L)
      return(jnp_zeros(c(ncols, nrows), dtype = result_type(scl_dtype(c), scl_dtype(r))))
    elems <- jnp_concatenate(list(lax_rev(c, 0L), lax_slice(r, 1L, nrows)))
    lax_rev(scl_patches_1d(elems, nrows), 0L)
  }, c(1L, 1L))(scl_atleast_1d(c), scl_atleast_1d(r))
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_hankel <- function(c, r = NULL) {
  # Port of jax/_src/scipy/linalg.py:hankel and _hankel
  c <- as_jax(c)
  r <- if (is.null(r)) jnp_zeros_like(c) else as_jax(r)
  if (scl_ndim(c) == 0L)
    jax_error("hankel: c must be at least 1-dimensional, got a scalar.", class = "jax_value_error")
  if (scl_ndim(r) == 0L)
    jax_error("hankel: r must be at least 1-dimensional, got a scalar.", class = "jax_value_error")
  if (scl_ndim(c) < scl_ndim(r)) {
    c <- lax_expand_dims(c, seq_len(scl_ndim(r) - scl_ndim(c)) - 1L)
  } else if (scl_ndim(r) < scl_ndim(c)) {
    r <- lax_expand_dims(r, seq_len(scl_ndim(c) - scl_ndim(r)) - 1L)
  }
  scl_vectorize(function(c, r) {
    ncols <- scl_shape(c)
    nrows <- scl_shape(r)
    if (ncols == 0L || nrows == 0L)
      return(jnp_zeros(c(ncols, nrows), dtype = result_type(c, r)))
    v <- jnp_concatenate(list(c, lax_slice(r, 1L, nrows)))
    scl_patches_1d(v, nrows)
  }, c(1L, 1L))(c, r)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_circulant <- function(c) {
  # Port of jax/_src/scipy/linalg.py:circulant and _circulant
  scl_vectorize(function(c) {
    n <- scl_shape(c)
    a <- jnp_arange(n)
    c[jnp_remainder(lax_expand_dims(a, 1L) - lax_expand_dims(a, 0L), n)]
  }, 1L)(scl_atleast_1d(c))
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_leslie <- function(f, s) {
  # Port of jax/_src/scipy/linalg.py:leslie and _leslie
  f_arr <- scl_atleast_1d(f)
  s_arr <- scl_atleast_1d(s)
  fs <- scl_shape(f_arr)
  ss <- scl_shape(s_arr)
  if (utils::tail(fs, 1L) < 2L)
    jax_error("The length of f along the last axis must be at least 2; got shape ", py_shape(fs),
              ".", class = "jax_value_error")
  if (utils::tail(ss, 1L) != utils::tail(fs, 1L) - 1L)
    jax_error("Incorrect lengths for f and s. The length of s along the last axis must be one ",
              "less than the length of f; got f shape ", py_shape(fs), " and s shape ",
              py_shape(ss), ".", class = "jax_value_error")
  scl_vectorize(function(f, s) {
    fs <- promote_dtypes(list(f, s))
    # jnp.diag(s, k = -1): s on the first subdiagonal.
    d <- lax_pad(construct_diagonal(fs[[2L]]), lax_full(integer(0), 0, scl_dtype(fs[[2L]])),
                 list(c(1L, 0L, 0L), c(0L, 1L, 0L)))
    at_set(d, 0L, value = fs[[1L]])
  }, c(1L, 1L))(f_arr, s_arr)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_companion <- function(a) {
  # Port of jax/_src/scipy/linalg.py:companion and _companion
  a <- scl_atleast_1d(promote_args_inexact(a)[[1L]])
  if (utils::tail(scl_shape(a), 1L) < 2L)
    jax_error("The length of `a` along the last axis must be at least 2; got shape ",
              py_shape(scl_shape(a)), ".", class = "jax_value_error")
  scl_vectorize(function(a) {
    n <- scl_shape(a)
    first_row <- -lax_slice(a, 1L, n) / a[0L]
    m <- n - 1L
    out <- jnp_eye(m, m, k = -1L, dtype = scl_dtype(first_row))
    at_set(out, 0L, value = first_row)
  }, 1L)(a)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_fiedler <- function(a) {
  # Port of jax/_src/scipy/linalg.py:fiedler
  arr <- scl_atleast_1d(a)
  nd <- scl_ndim(arr)
  jnp_abs(lax_expand_dims(arr, nd) - lax_expand_dims(arr, nd - 1L))
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_fiedler_companion <- function(a) {
  # Port of jax/_src/scipy/linalg.py:fiedler_companion and _fiedler_companion
  a <- scl_atleast_1d(promote_args_inexact(a)[[1L]])
  if (utils::tail(scl_shape(a), 1L) == 0L)
    jax_error("fiedler_companion requires the last axis of 'a' to have nonzero length, but got ",
              "an array of shape ", py_shape(scl_shape(a)), ".", class = "jax_value_error")
  scl_vectorize(function(a) {
    n <- scl_shape(a) - 1L
    if (n == 0L) return(jnp_zeros(c(0L, 0L), dtype = scl_dtype(a)))
    a <- a / a[0L]
    if (n == 1L) return(jnp_reshape(-lax_slice(a, 1L, 2L), c(1L, 1L)))
    i <- matrix(seq_len(n) - 1L, n, n)
    j <- t(i)
    super_val <- -a[pmin(i + 2L, n)]
    sub_val <- -a[pmin(i + 1L, n)]
    i_even <- i %% 2L == 0L
    i_odd <- i %% 2L == 1L
    one <- as_jax(1, scl_dtype(a))
    zero <- as_jax(0, scl_dtype(a))
    conditions <- list((i == 0L) & (j == 0L), (i == 1L) & (j == 0L), (j == i + 1L) & i_even,
                       (j == i + 2L) & i_even, (j == i - 1L) & i_even & (i >= 2L),
                       (j == i - 2L) & i_odd & (i >= 3L))
    choices <- list(-a[1L], one, super_val, one, sub_val, one)
    # jnp.select: the first matching condition wins.
    out <- jnp_broadcast_to(zero, c(n, n))
    for (q in rev(seq_along(conditions))) out <- jnp_where(conditions[[q]], choices[[q]], out)
    out
  }, 1L)(a)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_convolution_matrix <- function(a, n, mode = "full") {
  # Port of jax/_src/scipy/linalg.py:convolution_matrix
  if (!is.numeric(n) || length(n) != 1L || n != round(n))
    jax_error("n must be an integer; got ", format(n), ".", class = "jax_type_error")
  n <- as.integer(n)
  if (n <= 0L) jax_error("n must be a positive integer; got ", n, ".", class = "jax_value_error")
  a_arr <- as_jax(a)
  if (scl_ndim(a_arr) == 0L)
    jax_error("convolution_matrix: a must be at least 1-dimensional, got a scalar.",
              class = "jax_value_error")
  m <- utils::tail(scl_shape(a_arr), 1L)
  if (m < 1L)
    jax_error("len(a) must be at least 1; got shape ", py_shape(scl_shape(a_arr)), ".",
              class = "jax_value_error")
  if (!(identical(mode, "full") || identical(mode, "valid") || identical(mode, "same")))
    jax_error("mode must be one of 'full', 'valid', 'same'; got '", mode, "'.",
              class = "jax_value_error")
  nd <- scl_ndim(a_arr)
  pad_cfg <- c(rep(list(c(0L, 0L, 0L)), nd - 1L), list(c(0L, n - 1L, 0L)))
  zero <- lax_full(integer(0), 0, scl_dtype(a_arr))
  az <- lax_pad(a_arr, zero, pad_cfg)
  raz <- lax_pad(lax_rev(a_arr, nd - 1L), zero, pad_cfg)
  L <- m + n - 1L
  if (mode == "same") {
    trim <- min(n, m) - 1L
    tb <- trim %/% 2L
    te <- trim - tb
  } else if (mode == "valid") {
    tb <- min(n, m) - 1L
    te <- tb
  } else {
    tb <- 0L
    te <- 0L
  }
  col0 <- lax_slice_in_dim(az, tb, L - te, axis = -1L)
  row0 <- lax_slice_in_dim(raz, L - n - tb, L - tb, axis = -1L)
  scipy_linalg_toeplitz(col0, row0)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_hilbert <- function(n) {
  # Port of jax/_src/scipy/linalg.py:hilbert
  a <- lax_broadcasted_iota(default_float(), c(as.integer(n), 1L), 0L)
  1 / (a + jnp_transpose(a) + 1)
}

# Port of jax/_src/scipy/special.py:comb (repetition = FALSE).
scl_comb <- function(N, k) {
  Nk <- promote_args_inexact(N, k)
  N <- Nk[[1L]]
  k <- Nk[[2L]]
  cond <- (k <= N) & (N >= 0) & (k >= 0)
  safe_N <- jnp_where(cond, N, 0.0)
  safe_k <- jnp_where(cond, k, 0.0)
  result <- lax_exp(lax_lgamma(safe_N + 1) - lax_lgamma(safe_k + 1) -
                      lax_lgamma(safe_N + 1 - safe_k))
  jnp_where(cond, result, 0.0)
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_invhilbert <- function(n) {
  # Port of jax/_src/scipy/linalg.py:invhilbert
  n <- as.integer(n)
  a <- jnp_arange(n, dtype = default_float())
  i <- lax_expand_dims(a, 1L)
  j <- lax_expand_dims(a, 0L)
  s <- i + j
  jnp_power(-1.0, s) * (s + 1.0) * scl_comb(n + i, n - j - 1L) * scl_comb(n + j, n - i - 1L) *
    lax_integer_pow(scl_comb(s, i), 2L)
}

# Port of jax/_src/scipy/linalg.py:_binom.
scl_binom <- function(n, k) {
  lax_exp(lax_lgamma(n + 1.0) - lax_lgamma(n - k + 1.0) - lax_lgamma(k + 1.0))
}

scl_check_kind <- function(kind, verb) {
  if (is.null(kind)) kind <- "symmetric"
  if (!(identical(kind, "symmetric") || identical(kind, "lower") || identical(kind, "upper")))
    jax_error("Expected kind to be ", verb, " of: ['symmetric', 'lower', 'upper']; got ",
              format(kind), class = "jax_value_error")
  kind
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_pascal <- function(n, kind = NULL) {
  # Port of jax/_src/scipy/linalg.py:pascal
  kind <- scl_check_kind(kind, "on")
  a <- jnp_arange(as.integer(n), dtype = "float32")
  L_n <- scl_binom(lax_expand_dims(a, 1L), lax_expand_dims(a, 0L))
  switch(kind, lower = L_n, upper = jnp_transpose(L_n), jnp_dot(L_n, jnp_transpose(L_n)))
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_invpascal <- function(n, kind = NULL) {
  # Port of jax/_src/scipy/linalg.py:invpascal
  kind <- scl_check_kind(kind, "one")
  a <- jnp_arange(as.integer(n), dtype = default_float())
  i <- lax_expand_dims(a, 1L)
  j <- lax_expand_dims(a, 0L)
  L_inv <- jnp_power(-1.0, i - j) * scl_binom(i, j)
  switch(kind, lower = L_inv, upper = jnp_transpose(L_inv),
         jnp_dot(jnp_transpose(L_inv), L_inv))
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_helmert <- function(n, full = FALSE) {
  # Port of jax/_src/scipy/linalg.py:helmert
  n <- as.integer(n)
  if (n < 1L) jax_error("n must be a positive integer; got ", n, ".", class = "jax_value_error")
  i <- lax_expand_dims(jnp_arange(n - 1L), 1L)
  j <- lax_expand_dims(jnp_arange(n), 0L)
  k <- i + 1L
  k_f <- jnp_astype(k, default_float())
  denom <- jnp_sqrt(k_f * (k_f + 1))
  H <- jnp_where(j <= i, 1.0 / denom, jnp_where(j == k, -k_f / denom, jnp_zeros_like(denom)))
  if (isTRUE(full)) {
    first_row <- jnp_full(c(1L, n), 1.0 / jnp_sqrt(n), dtype = scl_dtype(H))
    H <- jnp_concatenate(list(first_row, H), axis = 0L)
  }
  H
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_hadamard <- function(n, dtype = "int") {
  # Port of jax/_src/scipy/linalg.py:hadamard
  lg2 <- if (is.numeric(n) && length(n) == 1L && n >= 1) log2(n) else NA
  if (is.na(lg2) || lg2 != round(lg2))
    jax_error("n must be a positive power of 2; got ", format(n), ".", class = "jax_value_error")
  H <- jnp_ones(c(1L, 1L), dtype = dtype)
  for (q in seq_len(lg2)) {
    H <- jnp_concatenate(list(jnp_concatenate(list(H, H), axis = 1L),
                              jnp_concatenate(list(H, -H), axis = 1L)), axis = 0L)
  }
  H
}

#' @rdname scipy_linalg_special_matrices
#' @export
scipy_linalg_dft <- function(n, scale = NULL, dtype = NULL) {
  # Port of jax/_src/scipy/linalg.py:dft
  if (!is.null(scale) && !(identical(scale, "sqrtn") || identical(scale, "n")))
    jax_error("scale must be None, 'sqrtn', or 'n'; got '", format(scale), "'.",
              class = "jax_value_error")
  if (is.null(dtype)) {
    dtype <- default_complex()
  } else {
    dtype <- canonicalize_dtype(dtype(dtype))
    if (!is_complex_dtype(dtype))
      jax_error("dtype must be a complex floating-point type; got ", dtype, ".",
                class = "jax_value_error")
  }
  n <- as.integer(n)
  a <- jnp_arange(n, dtype = dtype)
  omegas <- jnp_exp(-2i * pi * lax_expand_dims(a, 1L) * lax_expand_dims(a, 0L) / n)
  if (identical(scale, "sqrtn")) {
    omegas <- omegas / jnp_sqrt(n)
  } else if (identical(scale, "n")) {
    omegas <- omegas / n
  }
  omegas
}

# ---- jax.scipy.sparse.linalg ----------------------------------------------------------

scs_dot <- function(a, b) jnp_dot(a, b, precision = "highest")
# jnp.vdot(a, b, precision = HIGHEST).
scs_vdot <- function(a, b) {
  a <- jnp_ravel(a)
  if (is_complex_dtype(scl_dtype(a))) a <- jnp_conjugate(a)
  scs_dot(a, jnp_ravel(b))
}
scs_is_complex <- function(x) is_complex_dtype(scl_dtype(as_jax(x)))

# Port of sparse/linalg.py:_vdot_real_part.
scs_vdot_real_part <- function(x, y) {
  result <- scs_vdot(jnp_real(x), jnp_real(y))
  if (scs_is_complex(x) || scs_is_complex(y)) result <- result + scs_vdot(jnp_imag(x), jnp_imag(y))
  result
}
scs_sum <- function(xs) Reduce(`+`, xs, 0)
scs_vdot_real_tree <- function(x, y) scs_sum(tree_leaves(tree_map(scs_vdot_real_part, x, y)))
scs_vdot_tree <- function(x, y) scs_sum(tree_leaves(tree_map(scs_vdot, x, y)))
scs_norm <- function(x) {
  xs <- tree_leaves(x)
  jnp_sqrt(scs_sum(Map(scs_vdot_real_part, xs, xs)))
}
scs_mul <- function(scalar, tree) tree_map(function(l) scalar * l, tree)
scs_add <- function(x, y) tree_map(`+`, x, y)
scs_sub <- function(x, y) tree_map(`-`, x, y)
scs_identity <- function(x) x

# dtypes.lattice_result_type of the leaves of a pytree: list(dtype, weak).
scs_lattice <- function(tree) {
  avs <- lapply(tree_leaves(tree), function(l) aval_of(as_jax(l)))
  promote_types(vapply(avs, `[[`, "", "dtype"), vapply(avs, `[[`, NA, "weak"))
}

# Port of sparse/linalg.py:_normalize_matvec.
scs_normalize_matvec <- function(f) {
  if (is.function(f)) return(f)
  if (is_jax_value(f) || is.numeric(f) || is.complex(f)) {
    f <- as_jax(f)
    s <- scl_shape(f)
    if (length(s) != 2L || s[1L] != s[2L])
      jax_error("linear operator must be a square matrix, but has shape: ", py_shape(s),
                class = "jax_value_error")
    return(function(x) scs_dot(f, x))
  }
  jax_error("linear operator must be either a function or ndarray: ", format(f),
            class = "jax_type_error")
}

# Port of sparse/linalg.py:_cg_solve.
scs_cg_solve <- function(A, b, x0, maxiter, tol = 1e-5, atol = 0.0, M = scs_identity) {
  bs <- scs_vdot_real_tree(b, b)
  atol2 <- jnp_maximum(jnp_square(tol) * bs, jnp_square(atol))
  m_is_identity <- identical(M, scs_identity)
  r0 <- scs_sub(b, A(x0))
  p0 <- z0 <- M(r0)
  dt <- scs_lattice(p0)$dtype
  gamma0 <- jnp_astype(scs_vdot_real_tree(r0, z0), dt)
  cond_fun <- function(value) {
    rs <- if (m_is_identity) jnp_real(value[[3L]]) else scs_vdot_real_tree(value[[2L]], value[[2L]])
    (rs > atol2) & (value[[5L]] < maxiter)
  }
  body_fun <- function(value) {
    x <- value[[1L]]
    r <- value[[2L]]
    gamma <- value[[3L]]
    p <- value[[4L]]
    Ap <- A(p)
    alpha <- gamma / jnp_astype(scs_vdot_real_tree(p, Ap), dt)
    x_ <- scs_add(x, scs_mul(alpha, p))
    r_ <- scs_sub(r, scs_mul(alpha, Ap))
    z_ <- M(r_)
    gamma_ <- jnp_astype(scs_vdot_real_tree(r_, z_), dt)
    beta_ <- gamma_ / gamma
    p_ <- scs_add(z_, scs_mul(beta_, p))
    list(x_, r_, gamma_, p_, value[[5L]] + 1L)
  }
  lax_while_loop(cond_fun, body_fun, list(x0, r0, gamma0, p0, 0L))[[1L]]
}

# Port of sparse/linalg.py:_bicgstab_solve.
scs_bicgstab_solve <- function(A, b, x0, maxiter, tol = 1e-5, atol = 0.0, M = scs_identity) {
  bs <- scs_vdot_real_tree(b, b)
  atol2 <- jnp_maximum(jnp_square(tol) * bs, jnp_square(atol))
  cond_fun <- function(value) {
    r <- value[[2L]]
    k <- value[[9L]]
    rs <- scs_vdot_real_tree(r, r)
    (rs > atol2) & (k < maxiter) & (k >= 0L)
  }
  body_fun <- function(value) {
    x <- value[[1L]]
    r <- value[[2L]]
    rhat <- value[[3L]]
    alpha <- value[[4L]]
    omega <- value[[5L]]
    rho <- value[[6L]]
    p <- value[[7L]]
    q <- value[[8L]]
    k <- value[[9L]]
    rho_ <- scs_vdot_tree(rhat, r)
    beta <- rho_ / rho * alpha / omega
    p_ <- scs_add(r, scs_mul(beta, scs_sub(p, scs_mul(omega, q))))
    phat <- M(p_)
    q_ <- A(phat)
    alpha_ <- rho_ / scs_vdot_tree(rhat, q_)
    s <- scs_sub(r, scs_mul(alpha_, q_))
    exit_early <- scs_vdot_real_tree(s, s) < atol2
    shat <- M(s)
    t <- A(shat)
    omega_ <- scs_vdot_tree(t, s) / scs_vdot_tree(t, t)
    x_ <- tree_map(function(u, v) jnp_where(exit_early, u, v), scs_add(x, scs_mul(alpha_, phat)),
                   scs_add(x, scs_add(scs_mul(alpha_, phat), scs_mul(omega_, shat))))
    r_ <- tree_map(function(u, v) jnp_where(exit_early, u, v), s, scs_sub(s, scs_mul(omega_, t)))
    k_ <- jnp_where((omega_ == 0) | (alpha_ == 0), -11L, k + 1L)
    k_ <- jnp_where(rho_ == 0, -10L, k_)
    list(x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_)
  }
  r0 <- scs_sub(b, A(x0))
  lt <- scs_lattice(b)
  one <- lax_convert_element_type(1, lt$dtype, weak_type = lt$weak)
  lax_while_loop(cond_fun, body_fun, list(x0, r0, r0, one, one, one, r0, r0, 0L))[[1L]]
}

scs_shapes <- function(tree) lapply(tree_leaves(tree), function(l) scl_shape(as_jax(l)))
scs_size <- function(tree) sum(vapply(tree_leaves(tree), function(l) prod_int(scl_shape(as_jax(l))), 1L))
scs_shapes_str <- function(tree) {
  paste0("[", paste(vapply(scs_shapes(tree), py_shape, ""), collapse = ", "), "]")
}

# Port of sparse/linalg.py:_isolve.
scs_isolve <- function(solve_fn, A, b, x0 = NULL, tol = 1e-5, atol = 0.0, maxiter = NULL,
                       M = NULL, check_symmetric = FALSE) {
  if (is.null(x0)) x0 <- tree_map(jnp_zeros_like, b)
  b <- tree_map(as_jax, b)
  x0 <- tree_map(as_jax, x0)
  if (is.null(maxiter)) maxiter <- 10L * scs_size(b)
  if (is.null(M)) M <- scs_identity
  A <- scs_normalize_matvec(A)
  M <- scs_normalize_matvec(M)
  if (!treedef_equal(tree_structure(x0), tree_structure(b)))
    jax_error("x0 and b must have matching tree structure: ", format(tree_structure(x0)), " vs ",
              format(tree_structure(b)), class = "jax_value_error")
  if (!identical(scs_shapes(x0), scs_shapes(b)))
    jax_error("arrays in x0 and b must have matching shapes: ", scs_shapes_str(x0), " vs ",
              scs_shapes_str(b), class = "jax_value_error")
  isolve_solve <- function(matvec, b) {
    solve_fn(matvec, b, x0 = x0, maxiter = maxiter, tol = tol, atol = atol, M = M)
  }
  symmetric <- if (check_symmetric) !any(vapply(tree_leaves(b), scs_is_complex, NA)) else FALSE
  x <- lax_custom_linear_solve(A, b, solve = isolve_solve, transpose_solve = isolve_solve,
                               symmetric = symmetric)
  list(x, NULL)
}

#' Iterative linear solvers (jax.scipy.sparse.linalg)
#'
#' Ports of `jax.scipy.sparse.linalg.cg`, `bicgstab` and `gmres`. The linear operator
#' `A` is a matrix or a function computing `A %*% x` for a pytree `x`; gradients are
#' computed by implicit differentiation ([lax_custom_linear_solve()]). `cg` and
#' `bicgstab` return `list(x, NULL)` (no convergence information, as in JAX); `gmres`
#' returns `list(x, info)` where `info` is `-1` if the solution contains `NaN` and `0`
#' otherwise.
#' @param A Square matrix or linear function.
#' @param b Right hand side (array or pytree).
#' @param x0 Starting guess (default zeros).
#' @param tol,atol Relative and absolute tolerances: the iteration stops when
#'   `norm(residual) <= max(tol * norm(b), atol)`.
#' @param maxiter Maximum number of iterations (default `10 * size(b)`).
#' @param M Preconditioner (matrix or function approximating the inverse of `A`).
#' @param restart Size of the Krylov subspace between restarts (`gmres`).
#' @param solve_method `"batched"` or `"incremental"` (`gmres`).
#' @return `list(x, info)`.
#' @name scipy_sparse_linalg
NULL

#' @rdname scipy_sparse_linalg
#' @export
scipy_sparse_linalg_cg <- function(A, b, x0 = NULL, tol = 1e-5, atol = 0.0, maxiter = NULL,
                                   M = NULL) {
  # Port of jax/_src/scipy/sparse/linalg.py:cg
  scs_isolve(scs_cg_solve, A = A, b = b, x0 = x0, tol = tol, atol = atol, maxiter = maxiter,
             M = M, check_symmetric = TRUE)
}

#' @rdname scipy_sparse_linalg
#' @export
scipy_sparse_linalg_bicgstab <- function(A, b, x0 = NULL, tol = 1e-5, atol = 0.0,
                                         maxiter = NULL, M = NULL) {
  # Port of jax/_src/scipy/sparse/linalg.py:bicgstab
  scs_isolve(scs_bicgstab_solve, A = A, b = b, x0 = x0, tol = tol, atol = atol,
             maxiter = maxiter, M = M)
}

# Port of sparse/linalg.py:_safe_normalize.
scs_safe_normalize <- function(x, thresh = NULL) {
  norm <- scs_norm(x)
  lt <- scs_lattice(x)
  thresh <- if (is.null(thresh)) finfo(scl_dtype(norm))$eps else jnp_real(jnp_astype(thresh, lt$dtype))
  use_norm <- norm > thresh
  norm_cast <- lax_convert_element_type(norm, lt$dtype, weak_type = lt$weak)
  normalized_x <- tree_map(function(y) jnp_where(use_norm, y / norm_cast, 0.0), x)
  list(normalized_x, jnp_where(use_norm, norm, 0.0))
}

# Port of sparse/linalg.py:_project_on_columns: einsum("...n,...->n", conj(X), y).
scs_project_on_columns <- function(A, v) {
  v_proj <- tree_map(function(X, y) {
    r <- scl_ndim(y)
    lax_dot_general(jnp_conjugate(X), y, list(list(seq_len(r) - 1L, seq_len(r) - 1L),
                                              list(integer(0), integer(0))),
                    precision = "highest")
  }, A, v)
  Reduce(`+`, tree_leaves(v_proj))
}

# Port of sparse/linalg.py:_iterative_classical_gram_schmidt.
scs_icgs <- function(Q, x, xnorm, max_iterations = 2L) {
  Q0 <- tree_leaves(Q)[[1L]]
  r <- jnp_zeros(utils::tail(scl_shape(Q0), 1L), dtype = scl_dtype(Q0))
  q <- x
  xnorm_scaled <- xnorm / jnp_sqrt(2.0)
  body_function <- function(carry) {
    k <- carry[[1L]]
    q <- carry[[2L]]
    r <- carry[[3L]]
    qnorm_scaled <- carry[[4L]]
    h <- scs_project_on_columns(Q, q)
    Qh <- tree_map(function(X) scs_dot(X, h), Q)
    q <- scs_sub(q, Qh)
    r <- scs_add(r, h)
    qnorm_cond <- function(carry) carry[[2L]] & (carry[[1L]] < max_iterations - 1L)
    qnorm <- function(carry) {
      qn <- scs_safe_normalize(carry[[3L]])[[2L]]
      list(carry[[1L]], FALSE, carry[[3L]], qn / jnp_sqrt(2.0))
    }
    res <- lax_while_loop(qnorm_cond, qnorm, list(k, TRUE, q, qnorm_scaled))
    list(k + 1L, res[[3L]], r, res[[4L]])
  }
  cond_function <- function(carry) {
    rnorm <- scs_safe_normalize(carry[[3L]])[[2L]]
    (carry[[1L]] < max_iterations - 1L) & (rnorm < carry[[4L]])
  }
  carry <- body_function(list(0L, q, r, xnorm_scaled))
  carry <- lax_while_loop(cond_function, body_function, carry)
  list(carry[[2L]], carry[[3L]])
}

# Port of sparse/linalg.py:_kth_arnoldi_iteration.
scs_kth_arnoldi_iteration <- function(k, A, M, V, H) {
  dt <- scs_lattice(V)$dtype
  eps <- finfo(dt)$eps
  v <- tree_map(function(x) lax_dynamic_index_in_dim(x, k, axis = -1L, keepdims = FALSE), V)
  v <- M(A(v))
  v_norm_0 <- scs_safe_normalize(v)[[2L]]
  vh <- scs_icgs(V, v, v_norm_0, max_iterations = 2L)
  v <- vh[[1L]]
  h <- vh[[2L]]
  tol <- eps * v_norm_0
  un <- scs_safe_normalize(v, thresh = tol)
  unit_v <- un[[1L]]
  v_norm_1 <- un[[2L]]
  V <- tree_map(function(X, y) at_set(X, ellipsis, k + 1L, value = y), V, unit_v)
  h <- at_set(h, k + 1L, value = jnp_astype(v_norm_1, dt))
  H <- at_set(H, k, value = h)
  list(V, H, v_norm_1 == 0.0)
}

# Port of sparse/linalg.py:_rotate_vectors.
scs_rotate_vectors <- function(H, i, cs, sn) {
  x1 <- H[i]
  y1 <- H[i + 1L]
  x2 <- jnp_conjugate(cs) * x1 - jnp_conjugate(sn) * y1
  y2 <- sn * x1 + cs * y1
  H <- at_set(H, i, value = x2)
  at_set(H, i + 1L, value = y2)
}

# Port of sparse/linalg.py:_givens_rotation.
scs_givens_rotation <- function(a, b) {
  b_zero <- jnp_abs(b) == 0
  a_lt_b <- jnp_abs(a) < jnp_abs(b)
  t <- -jnp_where(a_lt_b, a, b) / jnp_where(a_lt_b, b, a)
  r <- jnp_astype(lax_rsqrt(1 + lax_integer_pow(jnp_abs(t), 2L)), scl_dtype(t))
  cs <- jnp_where(b_zero, 1, jnp_where(a_lt_b, r * t, r))
  sn <- jnp_where(b_zero, 0, jnp_where(a_lt_b, r, r * t))
  list(cs, sn)
}

# Port of sparse/linalg.py:_apply_givens_rotations.
scs_apply_givens_rotations <- function(H_row, givens, k) {
  R_row <- lax_fori_loop(0L, k, function(i, H_row) {
    scs_rotate_vectors(H_row, i, givens[i, 0L], givens[i, 1L])
  }, H_row)
  gf <- scs_givens_rotation(R_row[k], R_row[k + 1L])
  givens <- at_set(givens, k, value = jnp_stack(gf))
  list(scs_rotate_vectors(R_row, k, gf[[1L]], gf[[2L]]), givens)
}

scs_pad_krylov <- function(unit_residual, restart) {
  tree_map(function(x) {
    lax_pad(lax_expand_dims(x, -1L), lax_full(integer(0), 0, scl_dtype(x)),
            c(rep(list(c(0L, 0L, 0L)), scl_ndim(x)), list(c(0L, restart, 0L))))
  }, unit_residual)
}

# The update x0 + V[..., :-1] y and the new normalized residual of a GMRES cycle.
scs_gmres_update <- function(A, b, x0, V, y, M) {
  dx <- tree_map(function(X) scs_dot(lax_slice_in_dim(X, 0L, utils::tail(scl_shape(X), 1L) - 1L,
                                                      axis = -1L), y), V)
  x <- scs_add(x0, dx)
  residual <- M(scs_sub(b, A(x)))
  un <- scs_safe_normalize(residual)
  list(x, un[[1L]], un[[2L]])
}

# Port of sparse/linalg.py:_gmres_incremental.
scs_gmres_incremental <- function(A, b, x0, unit_residual, residual_norm, ptol, restart, M) {
  V <- scs_pad_krylov(unit_residual, restart)
  dt <- scs_lattice(b)$dtype
  R <- jnp_eye(restart, restart + 1L, dtype = dt)
  givens <- jnp_zeros(c(restart, 2L), dtype = dt)
  beta_vec <- at_set(jnp_zeros(restart + 1L, dtype = dt), 0L, value = jnp_astype(residual_norm, dt))
  loop_cond <- function(carry) (carry[[1L]] < restart) & (carry[[2L]] > ptol)
  arnoldi_qr_step <- function(carry) {
    k <- carry[[1L]]
    VHb <- scs_kth_arnoldi_iteration(k, A, M, carry[[3L]], carry[[4L]])
    rg <- scs_apply_givens_rotations(VHb[[2L]][k, ], carry[[6L]], k)
    R <- at_set(carry[[4L]], k, value = rg[[1L]])
    givens <- rg[[2L]]
    beta_vec <- scs_rotate_vectors(carry[[5L]], k, givens[k, 0L], givens[k, 1L])
    list(k + 1L, jnp_abs(beta_vec[k + 1L]), VHb[[1L]], R, beta_vec, givens)
  }
  carry <- lax_while_loop(loop_cond, arnoldi_qr_step,
                          list(0L, residual_norm, V, R, beta_vec, givens))
  V <- carry[[3L]]
  R <- carry[[4L]]
  beta_vec <- carry[[5L]]
  y <- scipy_linalg_solve_triangular(jnp_transpose(lax_slice(R, c(0L, 0L), c(restart, restart))),
                                     lax_slice(beta_vec, 0L, restart))
  scs_gmres_update(A, b, x0, V, y, M)
}

# Port of sparse/linalg.py:_lstsq.
scs_lstsq <- function(a, b) {
  aH <- jnp_conjugate(jnp_transpose(a))
  scipy_linalg_solve(scs_dot(aH, a), scs_dot(aH, b), assume_a = "pos")
}

# Port of sparse/linalg.py:_gmres_batched.
scs_gmres_batched <- function(A, b, x0, unit_residual, residual_norm, ptol, restart, M) {
  V <- scs_pad_krylov(unit_residual, restart)
  lt <- scs_lattice(b)
  H <- lax_convert_element_type(jnp_eye(restart, restart + 1L, dtype = lt$dtype), lt$dtype,
                                weak_type = lt$weak)
  loop_cond <- function(carry) (carry[[4L]] < restart) & !carry[[3L]]
  arnoldi_process <- function(carry) {
    k <- carry[[4L]]
    VHb <- scs_kth_arnoldi_iteration(k, A, M, carry[[1L]], carry[[2L]])
    list(VHb[[1L]], VHb[[2L]], VHb[[3L]], k + 1L)
  }
  carry <- lax_while_loop(loop_cond, arnoldi_process, list(V, H, FALSE, 0L))
  V <- carry[[1L]]
  H <- carry[[2L]]
  beta_vec <- at_set(jnp_zeros_like(H, shape = restart + 1L), 0L,
                     value = jnp_astype(residual_norm, lt$dtype))
  y <- scs_lstsq(jnp_transpose(H), beta_vec)
  scs_gmres_update(A, b, x0, V, y, M)
}

# Port of sparse/linalg.py:_gmres_solve.
scs_gmres_solve <- function(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func) {
  residual <- M(scs_sub(b, A(x0)))
  un <- scs_safe_normalize(residual)
  cond_fun <- function(value) (value[[2L]] < maxiter) & (value[[4L]] > atol)
  body_fun <- function(value) {
    r <- gmres_func(A, b, value[[1L]], value[[3L]], value[[4L]], ptol, restart, M)
    list(r[[1L]], value[[2L]] + 1L, r[[2L]], r[[3L]])
  }
  lax_while_loop(cond_fun, body_fun, list(x0, 0L, un[[1L]], un[[2L]]))[[1L]]
}

#' @rdname scipy_sparse_linalg
#' @export
scipy_sparse_linalg_gmres <- function(A, b, x0 = NULL, tol = 1e-5, atol = 0.0, restart = 20L,
                                      maxiter = NULL, M = NULL, solve_method = "batched") {
  # Port of jax/_src/scipy/sparse/linalg.py:gmres
  if (is.null(x0)) x0 <- tree_map(jnp_zeros_like, b)
  if (is.null(M)) M <- scs_identity
  A <- scs_normalize_matvec(A)
  M <- scs_normalize_matvec(M)
  b <- tree_map(as_jax, b)
  x0 <- tree_map(as_jax, x0)
  size <- scs_size(b)
  if (is.null(maxiter)) maxiter <- 10L * size
  restart <- as.integer(min(restart, size))
  if (!treedef_equal(tree_structure(x0), tree_structure(b)))
    jax_error("x0 and b must have matching tree structure: ", format(tree_structure(x0)), " vs ",
              format(tree_structure(b)), class = "jax_value_error")
  b_norm <- scs_norm(b)
  atol <- jnp_maximum(tol * b_norm, atol)
  Mb_norm <- scs_norm(M(b))
  ptol <- Mb_norm * jnp_minimum(1.0, atol / b_norm)
  gmres_func <- if (identical(solve_method, "incremental")) {
    scs_gmres_incremental
  } else if (identical(solve_method, "batched")) {
    scs_gmres_batched
  } else {
    jax_error("invalid solve_method ", format(solve_method),
              ", must be either 'incremental' or 'batched'", class = "jax_value_error")
  }
  solve <- function(A, b) scs_gmres_solve(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func)
  x <- lax_custom_linear_solve(A, b, solve = solve, transpose_solve = solve)
  failed <- jnp_isnan(scs_norm(x))
  list(x, jnp_where(failed, -1L, 0L))
}
