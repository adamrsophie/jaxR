# Define a function with a custom JVP rule

Port of `jax.custom_jvp`. The rule receives `(primals, tangents)` (lists
of the differentiable arguments, preceded by any `nondiff_argnums`
arguments) and returns `list(primal_out, tangent_out)`. Each tangent
output must have the shape and dtype of its primal output (float0 for an
integer output); `NULL` or a
[`SymbolicZero()`](https://adamrsophie.github.io/jaxR/reference/SymbolicZero.md)
stands for a zero tangent.

## Usage

``` r
custom_jvp(fun, jvp = NULL, nondiff_argnums = integer(0))

custom_jvp_defjvp(f, jvp, symbolic_zeros = FALSE)

custom_jvp_defjvps(f, ...)
```

## Arguments

- fun:

  The function.

- jvp:

  Optional JVP rule (can be set later with `custom_jvp_defjvp()`).

- nondiff_argnums:

  0-based positions of non-differentiable (static) arguments.

- f:

  A function created by `custom_jvp()`.

- symbolic_zeros:

  Whether the rule receives a
  [`SymbolicZero()`](https://adamrsophie.github.io/jaxR/reference/SymbolicZero.md)
  object in place of each tangent that is known to be zero (instead of
  an array of zeros), so that it can skip work.

- ...:

  One JVP function per argument, `g(tangent, primal_out, ...primals)`,
  or `NULL` for arguments without a derivative.

## Value

A function with a custom JVP rule.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- custom_jvp(function(x) sin(x), jvp = function(primals, tangents) {
  list(sin(primals[[1]]), cos(primals[[1]]) * tangents[[1]])
})
} # }
```
