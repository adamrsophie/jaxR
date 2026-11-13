# Cast between replication types

Port of `jax.lax.pcast` for `vmap` axes. jaxR has no `shard_map`, so
values are never varying across a named axis in the type system: the
cast to `"varying"` is the identity (after checking that the axis names
are bound), and the casts to `"reduced"` and `"unreduced"` raise as they
do in JAX for invariant values.

## Usage

``` r
lax_pcast(x, axis_name, to)
```

## Arguments

- x:

  A pytree of arrays.

- axis_name:

  Axis name or character vector of names (`NULL` or empty returns `x`).

- to:

  One of `"varying"`, `"reduced"` or `"unreduced"`.

## Value

`x`.

## Examples

``` r
if (FALSE) { # \dontrun{
vmap(function(v) lax_pcast(v, "i", to = "varying"), axis_name = "i")(c(1, 2))
} # }
```
