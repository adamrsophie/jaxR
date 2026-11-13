# Evaluate eagerly at trace time

Port of `jax.ensure_compile_time_eval`: inside a staged function
([`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
[`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)
bodies, ...), operations in `code` whose operands are all concrete are
evaluated now (constant folding) instead of being staged, so their
results can drive R control flow. Operations on traced values are staged
as usual.

## Usage

``` r
ensure_compile_time_eval(code)
```

## Arguments

- code:

  Expression to evaluate.

## Value

The value of `code`.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) {
  z <- ensure_compile_time_eval(sin(sin(as_jax(3))) > 0)
  if (as.logical(z)) sin(x) else cos(x)
})
} # }
```
