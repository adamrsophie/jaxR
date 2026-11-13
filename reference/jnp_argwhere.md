# Indices of nonzero elements

Ports of `jnp.argwhere` (a `(N, ndim)` array of coordinates) and
`jnp.flatnonzero` (indices into the flattened array). The number of
results depends on the data, so inside `jit`, `vmap` or `grad` a static
`size` is required; missing entries are padded with `fill_value`
(default 0) and extra entries are dropped.

## Usage

``` r
jnp_argwhere(a, size = NULL, fill_value = NULL)

jnp_flatnonzero(a, size = NULL, fill_value = NULL)
```

## Arguments

- a:

  Array.

- size:

  Optional static number of results.

- fill_value:

  Optional padding value.

## Value

An integer array.
