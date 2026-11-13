# Indices of triangles and masks

Ports of `jnp.triu_indices`, `jnp.tril_indices`,
`jnp.triu_indices_from`, `jnp.tril_indices_from` and `jnp.mask_indices`.
Indices are 0-based.

## Usage

``` r
jnp_triu_indices(n, k = 0L, m = NULL)

jnp_tril_indices(n, k = 0L, m = NULL)

jnp_triu_indices_from(arr, k = 0L)

jnp_tril_indices_from(arr, k = 0L)

jnp_mask_indices(n, mask_func, k = 0L, size = NULL)
```

## Arguments

- n, m:

  Number of rows and columns (`m` defaults to `n`).

- k:

  Diagonal offset.

- arr:

  Two dimensional array (or anything with a shape).

- mask_func:

  Function `(array, k)` returning a mask array, such as
  [`jnp_triu()`](https://adamrsophie.github.io/jaxR/reference/jnp_tri.md).

- size:

  Static number of indices to return (see
  [`jnp_nonzero()`](https://adamrsophie.github.io/jaxR/reference/jnp_nonzero.md)).

## Value

A list of two integer arrays.
