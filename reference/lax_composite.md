# Composite operation

Port of `jax.lax.composite`: wraps `decomposition` as a named composite
operation that compilers may replace by a specialized implementation.
Unnamed arguments of the returned function are inputs; named arguments
are attributes passed to `decomposition` (they must not be traced
values). On CPU the decomposition is compiled. Composites are not
differentiable (use
[`custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md))
and have no batching rule, as in JAX.

## Usage

``` r
lax_composite(decomposition, name, version = 0L)
```

## Arguments

- decomposition:

  Function implementing the operation.

- name:

  Name of the operation.

- version:

  Version of its semantics.

## Value

A function.

## Examples

``` r
if (FALSE) { # \dontrun{
tangent <- lax_composite(function(x) lax_sin(x) / lax_cos(x), name = "my.tangent")
tangent(c(0, pi / 4))
} # }
```
