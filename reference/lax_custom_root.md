# Differentiable root finding

Port of `jax.lax.custom_root`: solves `f(x) = 0` with
`solve(f, initial_guess)` and defines derivatives with respect to the
values `f` closes over by implicit differentiation at the solution
(assuming `f(solution) == 0`), without differentiating through `solve`.

## Usage

``` r
lax_custom_root(f, initial_guess, solve, tangent_solve, has_aux = FALSE)
```

## Arguments

- f:

  Function of a pytree returning a pytree of the same structure.

- initial_guess:

  Pytree of arrays.

- solve:

  Function `(f, initial_guess)` returning the root (or `list(root, aux)`
  with `has_aux`).

- tangent_solve:

  Function `(g, y)` solving the linear system `g(x) = y`, where `g` is
  `f` linearized at the root; for scalars `function(g, y) y / g(1)`.

- has_aux:

  Whether `solve` returns `list(root, aux)`.

## Value

The result of `solve(f, initial_guess)`.

## Examples

``` r
if (FALSE) { # \dontrun{
sqrt_newton <- function(a) {
  f <- function(x) x^2 - a
  solve <- function(f, x0) lax_fori_loop(0L, 20L, function(i, x) x - f(x) / (2 * x), x0)
  lax_custom_root(f, 1, solve, function(g, y) y / g(1))
}
grad(sqrt_newton)(2)  # 1 / (2 sqrt(2))
} # }
```
