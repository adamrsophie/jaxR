# Linear algebra (jax.numpy.linalg)

Ports of the functions of `jax.numpy.linalg`. Inputs are promoted to an
inexact dtype and the matrix functions work on batches (leading
dimensions). Functions that return named tuples in JAX return named
lists here: `jnp_linalg_svd()` gives `list(U, S, Vh)`,
`jnp_linalg_eigh()` and `jnp_linalg_eig()` give
`list(eigenvalues, eigenvectors)`, `jnp_linalg_qr()` gives `list(Q, R)`
and `jnp_linalg_slogdet()` gives `list(sign, logabsdet)`.

## Usage

``` r
jnp_linalg_cholesky(a, upper = FALSE, symmetrize_input = TRUE)

jnp_linalg_svd(
  a,
  full_matrices = TRUE,
  compute_uv = TRUE,
  hermitian = FALSE,
  subset_by_index = NULL
)

jnp_linalg_svdvals(x)

jnp_linalg_matrix_power(a, n)

jnp_linalg_matrix_rank(M, rtol = NULL, hermitian = FALSE, tol = NULL)

jnp_linalg_slogdet(a, method = NULL)

jnp_linalg_det(a)

jnp_linalg_eig(a)

jnp_linalg_eigvals(a)

jnp_linalg_eigh(a, UPLO = NULL, symmetrize_input = TRUE)

jnp_linalg_eigvalsh(a, UPLO = "L", symmetrize_input = TRUE)

jnp_linalg_pinv(a, rtol = NULL, hermitian = FALSE, rcond = NULL)

jnp_linalg_inv(a)

jnp_linalg_solve(a, b)

jnp_linalg_lstsq(a, b, rcond = NULL, numpy_resid = FALSE)

jnp_linalg_norm(x, ord = NULL, axis = NULL, keepdims = FALSE)

jnp_linalg_matrix_norm(x, keepdims = FALSE, ord = "fro")

jnp_linalg_vector_norm(x, axis = NULL, keepdims = FALSE, ord = 2)

jnp_linalg_cond(x, p = NULL)

jnp_linalg_qr(a, mode = "reduced")

jnp_linalg_cross(x1, x2, axis = -1L)

jnp_linalg_outer(x1, x2)

jnp_linalg_matrix_transpose(x)

jnp_linalg_vecdot(
  x1,
  x2,
  axis = -1L,
  precision = NULL,
  preferred_element_type = NULL
)

jnp_linalg_matmul(x1, x2, precision = NULL, preferred_element_type = NULL)

jnp_linalg_tensordot(
  x1,
  x2,
  axes = 2L,
  precision = NULL,
  preferred_element_type = NULL
)

jnp_linalg_diagonal(x, offset = 0L)

jnp_linalg_trace(x, offset = 0L, dtype = NULL)

jnp_linalg_tensorinv(a, ind = 2L)

jnp_linalg_tensorsolve(a, b, axes = NULL)

jnp_linalg_multi_dot(arrays, precision = NULL)
```

## Arguments

- a, x, M:

  Input array.

- upper:

  If `TRUE` return the upper triangular factor.

- symmetrize_input:

  If `TRUE` the matrix is symmetrized before the decomposition.

- full_matrices:

  Whether full or reduced matrices are returned.

- compute_uv:

  Whether to compute the singular vectors.

- hermitian:

  Assume a Hermitian matrix (uses the eigendecomposition).

- subset_by_index:

  Optional 0-based index range of singular values.

- n:

  Integer exponent.

- rtol, tol, rcond:

  Relative tolerance below which singular values count as zero (`tol`
  and `rcond` are older names of `rtol`).

- method:

  `"lu"` (default) or `"qr"`.

- UPLO:

  `"L"` or `"U"`: which triangle of the matrix is used.

- b:

  Right hand side.

- numpy_resid:

  If `TRUE` the residuals follow the NumPy convention (empty unless the
  system is overdetermined and of full rank).

- ord:

  Order of the norm.

- axis:

  Axis (0-based) or axes over which to operate.

- keepdims:

  Keep the reduced dimensions with size one.

- p:

  Order of the norm (see `jnp_linalg_norm()`).

- mode:

  `"reduced"`, `"complete"`, `"r"` or `"raw"`.

- x1, x2:

  Input arrays.

- precision:

  Precision of the underlying products.

- preferred_element_type:

  Optional result dtype.

- axes:

  Number of axes to contract, or `list(axes1, axes2)`; for
  `jnp_linalg_tensorsolve()` the axes of `a` to move to the end.

- offset:

  Offset of the diagonal.

- dtype:

  Optional result dtype.

- ind:

  Number of leading dimensions that form the rows of the inverted
  tensor.

- arrays:

  List of arrays.

## Value

An array or a (named) list of arrays.
