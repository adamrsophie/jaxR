# Minimize a scalar function of one or more variables

Port of `jax.scipy.optimize.minimize`. Supported methods are `"BFGS"`
and `"l-bfgs-experimental-do-not-rely-on-this"` (case insensitive). The
whole optimization is one
[`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md),
so the function can be wrapped in
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and
mapped with
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md).

## Usage

``` r
scipy_optimize_minimize(
  fun,
  x0,
  args = list(),
  method,
  tol = NULL,
  options = NULL
)
```

## Arguments

- fun:

  Objective `fun(x, ...)` returning a scalar; `x` is a 1-D array.

- x0:

  Initial guess (1-D).

- args:

  List of extra arguments passed to `fun` after `x`.

- method:

  Solver name.

- tol:

  Ignored (as in JAX); use the solver options.

- options:

  Named list of solver options.

## Value

A
[`scipy_optimize_OptimizeResults()`](https://adamrsophie.github.io/jaxR/reference/scipy_optimize_OptimizeResults.md)
list.

## Details

Options for BFGS: `maxiter` (default `200 * length(x0)`), `norm` (order
of the gradient norm, default `Inf`), `gtol` (default `1e-5`),
`line_search_maxiter` (default 10). Status codes: 0 converged, 1
`maxiter` reached, 2 + the line search status if the line search failed,
-1 otherwise.

Options for L-BFGS: `maxiter`, `norm`, `maxcor` (history size, default
10), `ftol`, `gtol`, `maxfun`, `maxgrad`, `maxls` (line search
iterations, default 20).
