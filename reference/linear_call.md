# Linear function with a custom transpose

Port of `jax.custom_derivatives.linear_call`: calls
`fun(residual_args, linear_args)`, a function linear in `linear_args`,
whose transpose (used by reverse mode differentiation and
[`linear_transpose()`](https://adamrsophie.github.io/jaxR/reference/linear_transpose.md))
is `fun_transpose(residual_args, ct)`. The transpose of the transpose is
`fun` again. There is no batching rule, as in JAX.

## Usage

``` r
linear_call(fun, fun_transpose, residual_args, linear_args)
```

## Arguments

- fun:

  Function `(residual_args, linear_args)` linear in its second argument.

- fun_transpose:

  Function `(residual_args, ct)` returning a pytree with the structure
  of `linear_args`.

- residual_args:

  Pytree of arguments in which `fun` need not be linear.

- linear_args:

  Pytree of the linear arguments.

## Value

`fun(residual_args, linear_args)`.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- function(r, x) x / r
t <- function(r, ct) ct / r
div_add <- function(x, denom) x + linear_call(f, t, denom, x)
linear_transpose(function(x) div_add(x, 3), 1)(18)  # 24
} # }
```
