# A partially applied function that is a pytree

Port of `jax.tree_util.Partial`: `jax_Partial(func, ...)` calls
`func(<bound args>, <new args>)`. Its bound arguments are the leaves of
a pytree and `func` is static data, so it can be passed to transformed
functions such as
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md). The R
name carries the `jax_` prefix because `Partial` is a base R function in
recent versions of R.

## Usage

``` r
jax_Partial(func, ...)
```

## Arguments

- func:

  A function.

- ...:

  Arguments to bind (named arguments are bound by name).

## Value

A function of class `jax_Partial`.

## Examples

``` r
if (FALSE) { # \dontrun{
add_one <- jax_Partial(jnp_add, 1)
jit(function(f, x) f(x))(add_one, 2)
} # }
```
