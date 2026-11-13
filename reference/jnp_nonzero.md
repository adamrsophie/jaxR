# Indices of nonzero elements

Port of `jnp.nonzero` (and of the one argument form of `jnp.where`).
Without `size` the number of nonzero elements must be concrete, so
`size` is required under
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and other
transformations; extra slots are filled with `fill_value` (default 0)
and extra nonzero entries are dropped.

## Usage

``` r
jnp_nonzero(a, size = NULL, fill_value = NULL)
```

## Arguments

- a:

  Array of rank at least one.

- size:

  Optional static number of indices per dimension.

- fill_value:

  Scalar, or a list with one scalar per dimension.

## Value

A list of integer arrays, one per dimension (0-based).
