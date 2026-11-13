# Inexact line search satisfying the strong Wolfe conditions

Port of `jax._src.scipy.optimize.line_search.line_search` (algorithm 3.5
of Nocedal and Wright, Numerical Optimization), the line search used by
[`scipy_optimize_minimize()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_minimize.md).
It is not part of the public `jax.scipy.optimize` namespace.

## Usage

``` r
scipy_optimize_line_search(
  f,
  xk,
  pk,
  old_fval = NULL,
  old_old_fval = NULL,
  gfk = NULL,
  c1 = 1e-04,
  c2 = 0.9,
  maxiter = 20L
)
```

## Arguments

- f:

  Function of `x` returning a scalar.

- xk:

  Starting point.

- pk:

  Search direction.

- old_fval, gfk:

  Value and gradient of `f` at `xk` (computed if `NULL`).

- old_old_fval:

  Value of `f` at the previous point, used for the initial step.

- c1, c2:

  Constants of the Wolfe conditions.

- maxiter:

  Maximum number of iterations.

## Value

A named list with `failed`, `nit`, `nfev`, `ngev`, `k`, `a_k` (the
step), `f_k`, `g_k` (value and gradient at the new point) and `status`
(0: converged, 1: the zoom failed, 3: `maxiter` reached).
