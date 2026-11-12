# jax.experimental.sparse.linalg (port of jax/experimental/sparse/linalg.py): the LOBPCG
# eigensolver and the sparse direct solver.

splin_mm <- function(a, b) lax_dot(a, b, precision = "highest")
splin_colnorm <- function(x, keepdims = TRUE) jnp_linalg_norm(x, ord = 2L, axis = 0L, keepdims = keepdims)

# ---- lobpcg_standard ----------------------------------------------------------------------

#' Top eigenvalues with LOBPCG
#'
#' Port of `jax.experimental.sparse.linalg.lobpcg_standard`: computes the `k` largest
#' eigenvalues of a Hermitian matrix with the locally optimal block preconditioned
#' conjugate gradient method. Only the standard problem `A U = lambda U` is supported,
#' without preconditioning, and gradients are not available.
#' @param A An `(n, n)` array, or a function computing its action on an `(n, j)` array.
#' @param X An `(n, k)` array of initial search directions, numerically linearly
#'   independent, with `0 < 5 k < n`.
#' @param m Maximum number of iterations.
#' @param tol Convergence tolerance: an eigenpair `(lambda, v)` has converged when its
#'   residual norm `|A v - lambda v|` is below `tol * 10 * n * (lambda + |A v|)`.
#'   Defaults to the machine epsilon of the dtype of `X`.
#' @return `list(theta, U, i)`: the `k` eigenvalues, the `(n, k)` eigenvectors and the
#'   number of iterations performed.
#' @examples
#' \dontrun{
#' A <- jnp_diag(jnp_arange(20, dtype = "float32"))
#' X <- random_normal(random_key(0), c(20L, 2L))
#' experimental_sparse_linalg_lobpcg_standard(A, X)[[1L]]  # 19, 18
#' }
#' @export
experimental_sparse_linalg_lobpcg_standard <- function(A, X, m = 100L, tol = NULL) {
  # Compiled once per matrix shape when A is a matrix.
  if (is.function(A)) splin_lobpcg(A, X, m, tol) else splin_lobpcg_matrix(A, X, m, tol)
}

splin_lobpcg_matrix <- jit(function(A, X, m, tol) {
  splin_lobpcg(function(v) splin_mm(A, v), X, m, tol)
}, static_argnums = 2L)

splin_lobpcg <- function(A, X, m, tol) {
  X <- as_jax(X)
  shape <- aval_of(X)$shape
  if (length(shape) != 2L)
    jax_error("X must be an (n, k) array, got shape ", py_tuple(shape), class = "jax_value_error")
  n <- shape[1L]; k <- shape[2L]
  dt <- aval_of(X)$dtype
  splin_check_inputs(A, X)
  if (is.null(tol)) tol <- finfo(dt)$eps

  X <- splin_orthonormalize(X)
  P <- splin_extend_basis(X, k)
  # X holds the current best eigenvectors, P the search directions and R the residuals.
  AX <- A(X)
  theta <- jnp_sum(X * AX, axis = 0L, keepdims = TRUE)
  R <- AX - theta * X

  cond <- function(state) jnp_logical_and(state[[1L]] < m, state[[5L]] < k)
  body <- function(state) {
    # Invariants: X, P and R are kept orthonormal. Some columns of R and P may be zero
    # (basis truncation by the orthogonalization routines), but none of X.
    X <- state[[2L]]; P <- state[[3L]]
    R <- splin_project_out(jnp_concatenate(list(X, P), axis = 1L), state[[4L]])
    XPR <- jnp_concatenate(list(X, P, R), axis = 1L)
    # Projected eigensolve.
    rr <- splin_rayleigh_ritz_orth(A, XPR)
    theta <- rr[[1L]][sl(stop = k)]
    Q <- rr[[2L]]
    # Eigenvector extraction.
    B <- Q[, sl(stop = k)]
    B <- B / splin_colnorm(B)
    X <- splin_mm(XPR, B)
    X <- X / splin_colnorm(X)
    # Difference terms: concat(0, Q[k:, :k]) orthogonalized against Q[:, :k].
    q <- jnp_linalg_qr(jnp_transpose(Q[sl(stop = k), sl(k)]))[[1L]]
    P <- splin_mm(XPR, splin_mm(Q[, sl(k)], q))
    normP <- splin_colnorm(P)
    P <- P / jnp_where(normP == 0L, 1.0, normP)
    # New residuals, converged when small relative to the floating point error expected
    # from computing them.
    AX <- A(X)
    R <- AX - lax_expand_dims(theta, 0L) * X
    reltol <- (splin_colnorm(AX, FALSE) + theta) * n * 10L
    converged <- jnp_sum(splin_colnorm(R, FALSE) < tol * reltol)
    list(state[[1L]] + 1L, X, P, R, converged, lax_expand_dims(theta, 0L))
  }
  state <- lax_while_loop(cond, body, list(0L, X, P, R, 0L, theta))
  list(state[[6L]][0L, ], state[[2L]], state[[1L]])
}

splin_check_inputs <- function(A, X) {
  n <- aval_of(X)$shape[1L]; k <- aval_of(X)$shape[2L]
  dt <- aval_of(X)$dtype
  if (k == 0L) jax_error("must have search dim > 0, got ", k, class = "jax_value_error")
  if (k * 5L >= n)
    jax_error("expected search dim * 5 < matrix dim (got ", k * 5L, ", ", n, ")", class = "jax_value_error")
  out <- aval_of(as_jax(A(jnp_zeros(c(n, 1L), dtype = dt))))
  if (out$dtype != dt)
    jax_error("A, X must have same dtypes (were ", out$dtype, ", ", dt, ")", class = "jax_value_error")
  if (!identical(as.integer(out$shape), c(n, 1L)))
    jax_error("A must be (", n, ", ", n, ") matrix A, got output ", py_tuple(out$shape),
              class = "jax_value_error")
}

# Eigenvalues in decreasing order (upstream _eigh_ascending).
splin_eigh <- function(A) {
  e <- jnp_linalg_eigh(A)
  list(jnp_flip(e[[1L]], 0L), jnp_flip(e[[2L]], 1L))
}

# SVQB: a truncated orthonormal basis of X from the eigenbasis of X^T X. Columns in
# numerically degenerate directions are zeroed.
splin_svqb <- function(X) {
  dt <- aval_of(X)$dtype
  norms <- splin_colnorm(X)
  X <- X / jnp_where(norms == 0L, 1.0, norms)
  inner <- splin_mm(jnp_transpose(X), X)
  e <- splin_eigh(inner)
  w <- e[[1L]]
  # An eigenvalue below max eigenvalue * eps marks a degenerate direction.
  tau <- finfo(dt)$eps * w[0L]
  # tau == 0 when X is all zeros.
  sqrted <- jnp_where(tau > 0L, jnp_maximum(w, tau), 1.0)^(-0.5)
  orthoX <- splin_mm(X, e[[2L]] * lax_expand_dims(sqrted, 0L))
  keep <- lax_expand_dims(jnp_logical_and(w > tau, jnp_diag(inner) > 0), 0L)
  orthoX <- orthoX * jnp_astype(keep, dt)
  norms <- splin_colnorm(orthoX)
  keep <- jnp_logical_and(keep, norms > 0)
  orthoX / jnp_where(keep, norms, 1.0)
}

# Twice is enough.
splin_orthonormalize <- function(basis) splin_svqb(splin_svqb(basis))

# The component of U in the orthogonal complement of `basis` (orthonormal, zero columns
# allowed); suspicious columns are zeroed so that [basis, U] stays zero or orthogonal.
splin_project_out <- function(basis, U) {
  bt <- jnp_transpose(basis)
  sub <- function(U) U - splin_mm(basis, splin_mm(bt, U))
  for (i in 1:2) U <- splin_orthonormalize(sub(U))
  # It is crucial to end on a subtraction of the original basis.
  U <- sub(sub(U))
  U * jnp_astype(splin_colnorm(U) >= 0.99, aval_of(U)$dtype)
}

splin_rayleigh_ritz_orth <- function(A, S) splin_eigh(splin_mm(jnp_transpose(S), A(S)))

# Extend the orthonormal (n, k) basis X by m orthonormal columns with a block
# Householder reflector (deterministic, unlike projecting out a random basis).
splin_extend_basis <- function(X, m) {
  n <- aval_of(X)$shape[1L]; k <- aval_of(X)$shape[2L]
  dt <- aval_of(X)$dtype
  Xupper <- X[sl(stop = k), ]
  usv <- jnp_linalg_svd(Xupper)
  u <- usv[[1L]]; s <- usv[[2L]]; vt <- usv[[3L]]
  y <- jnp_concatenate(list(Xupper + splin_mm(u, vt), X[sl(k), ]), axis = 0L)
  other <- jnp_concatenate(list(jnp_eye(m, dtype = dt), jnp_zeros(c(n - k - m, m), dtype = dt)), axis = 0L)
  w <- splin_mm(y, jnp_transpose(vt) * lax_expand_dims((2L * (1L + s))^(-0.5), 0L))
  h <- -2L * jnp_linalg_multi_dot(list(w, jnp_transpose(w[sl(k), ]), other), precision = "highest")
  at_add(h, sl(k), value = other)
}

# ---- spsolve ------------------------------------------------------------------------------

spsolve_p <- new_primitive("spsolve")
spsolve_p$abstract_eval <- function(avals, params) {
  data <- avals[[1L]]; indices <- avals[[2L]]; indptr <- avals[[3L]]; b <- avals[[4L]]
  shapes <- function() paste0("data.shape=", py_tuple(data$shape), " indices.shape=", py_tuple(indices$shape),
                              " indptr.shape=", py_tuple(indptr$shape))
  if (data$dtype != b$dtype)
    jax_error("data types do not match: data.dtype=", data$dtype, " b.dtype=", b$dtype, class = "jax_value_error")
  if (!is_int_dtype(indices$dtype) || !is_int_dtype(indptr$dtype))
    jax_error("index arrays must be integer typed; got indices.dtype=", indices$dtype, " indptr.dtype=",
              indptr$dtype, class = "jax_value_error")
  if (any(lengths(list(data$shape, indices$shape, indptr$shape, b$shape)) != 1L))
    jax_error("Arrays must be one-dimensional. Got ", shapes(), " b.shape=", py_tuple(b$shape),
              class = "jax_value_error")
  if (indptr$shape != b$shape + 1L || data$shape != indices$shape)
    jax_error("Invalid CSR buffer sizes: ", shapes(), class = "jax_value_error")
  if (!isTRUE(params$reorder %in% 0:3))
    jax_error("reorder=", params$reorder, " not valid, must be one of [1, 2, 3, 4]", class = "jax_value_error")
  new_aval(b$shape, b$dtype)
}

# The host solve, as scipy.sparse.linalg.spsolve upstream: a sparse LU factorization
# with the Matrix package for real matrices, a dense solve otherwise. Duplicate entries
# add up; a singular matrix gives NaN.
splin_spsolve_host <- function(data, indices, indptr, b) {
  n <- length(b)
  row <- rep(seq_len(n), diff(as.vector(indptr)))
  col <- as.vector(indices) + 1L
  data <- as.vector(data); b <- as.vector(b)
  tryCatch({
    if (!is.complex(data) && requireNamespace("Matrix", quietly = TRUE)) {
      as.vector(Matrix::solve(Matrix::sparseMatrix(i = row, j = col, x = data, dims = c(n, n)), b))
    } else {
      # ponytail: dense O(n^3) fallback; a sparse complex LU if this ever matters.
      A <- matrix(if (is.complex(data)) 0i else 0, n, n)
      pos <- row + (col - 1L) * n
      A[sort(unique(pos))] <- vapply(split(data, pos), sum, data[1L])
      solve(A, b)
    }
  }, error = function(e) rep(NaN, n))
}

spsolve_p$lowering <- lower_fun(function(args, params) {
  b <- aval_of(args[[4L]])
  pure_callback(splin_spsolve_host, ShapeDtypeStruct(b$shape, b$dtype), args[[1L]], args[[2L]],
                args[[3L]], args[[4L]])
})

defjvp(spsolve_p,
       # d/dM M^-1 b = -M^-1 M_dot M^-1 b
       function(t, primals, params) {
         p <- bind(spsolve_p, primals, params)
         n <- aval_of(primals[[4L]])$shape
         q <- bind(csr_matvec_p, list(t, primals[[2L]], primals[[3L]], p),
                   list(shape = c(n, n), transpose = FALSE))
         -bind(spsolve_p, list(primals[[1L]], primals[[2L]], primals[[3L]], q), params)
       },
       NULL, NULL,
       # d/db M^-1 b = M^-1 b_dot
       function(t, primals, params) bind(spsolve_p, list(primals[[1L]], primals[[2L]], primals[[3L]], t), params))

# Transpose of a square CSR matrix.
splin_csr_transpose <- function(data, indices, indptr) {
  m <- aval_of(indptr)$shape - 1L
  row <- jnp_cumsum(at_add(jnp_zeros_like(indices), indptr, value = 1L)) - 1L
  s <- lax_sort(list(indices, row, data), num_keys = 2L)
  counts <- jnp_astype(jnp_cumsum(jnp_bincount(s[[1L]], length = m)), aval_of(indptr)$dtype)
  list(s[[3L]], s[[2L]], at_set(jnp_zeros_like(indptr), sl(1L), value = counts))
}

spsolve_p$transpose <- function(ct, args, params) {
  if (!is_undefined(args[[4L]]))
    # Not reached: the JVP is linear in the data through csr_matvec.
    jax_error("spsolve transpose with respect to data", class = "jax_not_implemented_error")
  tr <- splin_csr_transpose(args[[1L]], args[[2L]], args[[3L]])
  list(NULL, NULL, NULL, bind(spsolve_p, c(tr, list(instantiate_zero(ct))), params))
}

#' Sparse direct solver
#'
#' Port of `jax.experimental.sparse.linalg.spsolve`: solves `A x = b` for a square
#' matrix `A` given by its CSR buffers. As upstream on CPU, the solve runs on the host:
#' a sparse LU factorization with the Matrix package (a dense solve for complex data or
#' without Matrix), so `tol` and `reorder` (GPU solver options) are only validated. It
#' works under [jit()] and is differentiable with respect to `data` and `b`; batching
#' with [vmap()] is not supported.
#' @param data The stored values of the CSR matrix.
#' @param indices The 0-based column indices.
#' @param indptr The row pointer array, of length `n + 1`.
#' @param b The right hand side, of length `n`.
#' @param tol Singularity tolerance of the GPU solver.
#' @param reorder Fill-in reducing reordering scheme of the GPU solver (0 to 3).
#' @return The solution, with the dtype and size of `b`.
#' @examples
#' \dontrun{
#' m <- experimental_sparse_CSR$fromdense(matrix(c(4, 1, 0, 1, 3, 0, 0, 0, 2), 3))
#' experimental_sparse_linalg_spsolve(m$data, m$indices, m$indptr, jnp_array(c(1, 2, 3)))
#' }
#' @export
experimental_sparse_linalg_spsolve <- function(data, indices, indptr, b, tol = 1e-6, reorder = 1L) {
  bind(spsolve_p, lapply(list(data, indices, indptr, b), as_jax), list(tol = as.numeric(tol), reorder = reorder))
}
