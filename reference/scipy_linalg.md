# Linear algebra (jax.scipy.linalg)

Ports of the decompositions and solvers of `jax.scipy.linalg`. Inputs
are promoted to an inexact dtype and most functions work on batches
(leading dimensions). Functions that return tuples in JAX return unnamed
lists: `scipy_linalg_svd()` gives `list(u, s, vh)`,
`scipy_linalg_eigh()` gives `list(w, v)`, `scipy_linalg_lu()` gives
`list(p, l, u)` (or `list(pl, u)`), `scipy_linalg_qr()` gives
`list(q, r)` (plus the pivots, or only `list(r)` for `mode = "r"`),
`scipy_linalg_schur()` gives `list(t, z)`. The scipy arguments
`overwrite_a`, `overwrite_b`, `overwrite_c`, `check_finite`, `lwork`,
`turbo`, `debug` and `lapack_driver` are accepted and ignored, as in
JAX.

## Usage

``` r
scipy_linalg_cholesky(
  a,
  lower = FALSE,
  overwrite_a = FALSE,
  check_finite = TRUE
)

scipy_linalg_cho_factor(
  a,
  lower = FALSE,
  overwrite_a = FALSE,
  check_finite = TRUE
)

scipy_linalg_cho_solve(
  c_and_lower,
  b,
  overwrite_b = FALSE,
  check_finite = TRUE
)

scipy_linalg_svd(
  a,
  full_matrices = TRUE,
  compute_uv = TRUE,
  overwrite_a = FALSE,
  check_finite = TRUE,
  lapack_driver = "gesdd"
)

scipy_linalg_det(a, overwrite_a = FALSE, check_finite = TRUE)

scipy_linalg_eigh(
  a,
  b = NULL,
  lower = TRUE,
  eigvals_only = FALSE,
  overwrite_a = FALSE,
  overwrite_b = FALSE,
  turbo = TRUE,
  eigvals = NULL,
  type = 1L,
  check_finite = TRUE
)

scipy_linalg_schur(a, output = "real")

scipy_linalg_inv(a, overwrite_a = FALSE, check_finite = TRUE)

scipy_linalg_lu_factor(a, overwrite_a = FALSE, check_finite = TRUE)

scipy_linalg_lu_solve(
  lu_and_piv,
  b,
  trans = 0L,
  overwrite_b = FALSE,
  check_finite = TRUE
)

scipy_linalg_lu(a, permute_l = FALSE, overwrite_a = FALSE, check_finite = TRUE)

scipy_linalg_qr(
  a,
  overwrite_a = FALSE,
  lwork = NULL,
  mode = "full",
  pivoting = FALSE,
  check_finite = TRUE
)

scipy_linalg_qr_multiply(
  a,
  c,
  mode = "right",
  pivoting = FALSE,
  conjugate = FALSE,
  overwrite_a = FALSE,
  overwrite_c = FALSE
)

scipy_linalg_solve(
  a,
  b,
  lower = FALSE,
  overwrite_a = FALSE,
  overwrite_b = FALSE,
  debug = FALSE,
  check_finite = TRUE,
  assume_a = "gen"
)

scipy_linalg_solve_triangular(
  a,
  b,
  trans = 0L,
  lower = FALSE,
  unit_diagonal = FALSE,
  overwrite_b = FALSE,
  debug = NULL,
  check_finite = TRUE
)

scipy_linalg_hessenberg(
  a,
  calc_q = FALSE,
  overwrite_a = FALSE,
  check_finite = TRUE
)
```

## Arguments

- a, b:

  Arrays (`a` is a matrix or a batch of matrices).

- lower:

  Whether the lower triangle is used (or returned).

- overwrite_a, overwrite_b, overwrite_c, check_finite, lwork, turbo,
  debug, lapack_driver:

  Ignored (scipy compatibility).

- c_and_lower:

  `list(c, lower)` as returned by `scipy_linalg_cho_factor()`.

- full_matrices:

  Whether to compute the full `u` and `vh`.

- compute_uv:

  Whether to compute `u` and `vh`.

- eigvals_only:

  Whether to return only the eigenvalues.

- eigvals:

  Must be `NULL` (only the full spectrum is implemented).

- type:

  Generalized problem type: 1 (`a v = w b v`), 2 (`a b v = w v`) or 3
  (`b a v = w v`).

- output:

  `"real"` or `"complex"` Schur form.

- lu_and_piv:

  `list(lu, piv)` as returned by `scipy_linalg_lu_factor()`.

- trans:

  0 (`"N"`), 1 (`"T"`) or 2 (`"C"`): which system to solve.

- permute_l:

  Whether to return `list(p %*% l, u)`.

- mode:

  For `qr`: `"full"`, `"economic"` or `"r"`; for `qr_multiply`:
  `"right"` (`c %*% q`) or `"left"` (`q %*% c`).

- pivoting:

  Whether to use column pivoting (the pivots are 0-based).

- c:

  Array multiplied by `q`.

- conjugate:

  Whether to use the conjugate transpose of `q`.

- assume_a:

  `"gen"`, `"sym"`, `"her"` or `"pos"` (only `"pos"` uses a Cholesky
  solve).

- unit_diagonal:

  Whether the diagonal of `a` is assumed to be ones.

- calc_q:

  Whether to also return the unitary factor `q` (`list(h, q)`).

## Value

Arrays, or lists of arrays as described above.
