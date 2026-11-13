# Top eigenvalues with LOBPCG

Port of `jax.experimental.sparse.linalg.lobpcg_standard`: computes the
`k` largest eigenvalues of a Hermitian matrix with the locally optimal
block preconditioned conjugate gradient method. Only the standard
problem `A U = lambda U` is supported, without preconditioning, and
gradients are not available.

## Usage

``` r
experimental_sparse_linalg_lobpcg_standard(A, X, m = 100L, tol = NULL)
```

## Arguments

- A:

  An `(n, n)` array, or a function computing its action on an `(n, j)`
  array.

- X:

  An `(n, k)` array of initial search directions, numerically linearly
  independent, with `0 < 5 k < n`.

- m:

  Maximum number of iterations.

- tol:

  Convergence tolerance: an eigenpair `(lambda, v)` has converged when
  its residual norm `|A v - lambda v|` is below
  `tol * 10 * n * (lambda + |A v|)`. Defaults to the machine epsilon of
  the dtype of `X`.

## Value

`list(theta, U, i)`: the `k` eigenvalues, the `(n, k)` eigenvectors and
the number of iterations performed.

## Examples

``` r
if (FALSE) { # \dontrun{
A <- jnp_diag(jnp_arange(20, dtype = "float32"))
X <- random_normal(random_key(0), c(20L, 2L))
experimental_sparse_linalg_lobpcg_standard(A, X)[[1L]]  # 19, 18
} # }
```
