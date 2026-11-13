# Results of an optimization

Port of the `jax.scipy.optimize.OptimizeResults` named tuple: a named
list (a pytree, so it can be returned from
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md)).

## Usage

``` r
scipy_optimize_OptimizeResults(
  x,
  success,
  status,
  fun,
  jac,
  hess_inv,
  nfev,
  njev,
  nit
)
```

## Arguments

- x:

  Solution.

- success:

  Whether the optimization succeeded.

- status:

  Solver specific status code (0 means converged).

- fun:

  Function value at `x`.

- jac:

  Gradient at `x`.

- hess_inv:

  Approximate inverse Hessian at `x` (`NULL` for L-BFGS).

- nfev, njev:

  Numbers of function and gradient evaluations.

- nit:

  Number of iterations.

## Value

A named list with these fields.
