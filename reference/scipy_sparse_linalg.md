# Iterative linear solvers (jax.scipy.sparse.linalg)

Ports of `jax.scipy.sparse.linalg.cg`, `bicgstab` and `gmres`. The
linear operator `A` is a matrix or a function computing `A %*% x` for a
pytree `x`; gradients are computed by implicit differentiation
([`lax_custom_linear_solve()`](https://adamrsophie.github.io/jaxR/reference/lax_custom_linear_solve.md)).
`cg` and `bicgstab` return `list(x, NULL)` (no convergence information,
as in JAX); `gmres` returns `list(x, info)` where `info` is `-1` if the
solution contains `NaN` and `0` otherwise.

## Usage

``` r
scipy_sparse_linalg_cg(
  A,
  b,
  x0 = NULL,
  tol = 1e-05,
  atol = 0,
  maxiter = NULL,
  M = NULL
)

scipy_sparse_linalg_bicgstab(
  A,
  b,
  x0 = NULL,
  tol = 1e-05,
  atol = 0,
  maxiter = NULL,
  M = NULL
)

scipy_sparse_linalg_gmres(
  A,
  b,
  x0 = NULL,
  tol = 1e-05,
  atol = 0,
  restart = 20L,
  maxiter = NULL,
  M = NULL,
  solve_method = "batched"
)
```

## Arguments

- A:

  Square matrix or linear function.

- b:

  Right hand side (array or pytree).

- x0:

  Starting guess (default zeros).

- tol, atol:

  Relative and absolute tolerances: the iteration stops when
  `norm(residual) <= max(tol * norm(b), atol)`.

- maxiter:

  Maximum number of iterations (default `10 * size(b)`).

- M:

  Preconditioner (matrix or function approximating the inverse of `A`).

- restart:

  Size of the Krylov subspace between restarts (`gmres`).

- solve_method:

  `"batched"` or `"incremental"` (`gmres`).

## Value

`list(x, info)`.
