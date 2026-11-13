# Eigenvalues of a symmetric tridiagonal matrix

Port of `jax.scipy.linalg.eigh_tridiagonal`: bisection with Sturm counts
for the eigenvalues and inverse iteration (with reorthogonalization of
clusters) for the eigenvectors.

## Usage

``` r
scipy_linalg_eigh_tridiagonal(
  d,
  e,
  eigvals_only = FALSE,
  select = "a",
  select_range = NULL,
  tol = NULL,
  key = NULL
)
```

## Arguments

- d:

  Diagonal (length `n`).

- e:

  Off-diagonal (length `n - 1`), same dtype as `d`.

- eigvals_only:

  Whether to return only the eigenvalues.

- select:

  `"a"` (all eigenvalues) or `"i"` (an index range); `"v"` is not
  implemented.

- select_range:

  For `select = "i"`, the 0-based inclusive index range `c(lo, hi)`.

- tol:

  Optional absolute tolerance of the eigenvalues.

- key:

  PRNG key for the random start vectors of the inverse iteration
  (default `random_key(42)`).

## Value

The eigenvalues, or `list(w, v)` with the eigenvectors in the columns of
`v`.
