# Diagonals

Ports of `jnp.diagonal` (diagonal of two axes, moved to the end),
`jnp.diag` (extract a diagonal of a 2-D array or build a 2-D array from
a 1-D diagonal), `jnp.diagflat` (2-D array with the flattened input on a
diagonal), `jnp.diag_indices`, `jnp.diag_indices_from` (lists of index
arrays for the main diagonal) and `jnp.fill_diagonal`.

## Usage

``` r
jnp_diagonal(a, offset = 0L, axis1 = 0L, axis2 = 1L)

jnp_diag(v, k = 0L)

jnp_diagflat(v, k = 0L)

jnp_diag_indices(n, ndim = 2L)

jnp_diag_indices_from(arr)

jnp_fill_diagonal(a, val, wrap = FALSE, inplace = TRUE)
```

## Arguments

- a, v, arr:

  Array.

- offset, k:

  0-based diagonal offset (positive above the main diagonal).

- axis1, axis2:

  0-based axes.

- n:

  Size of each dimension.

- ndim:

  Number of dimensions.

- val:

  Value(s) written on the diagonal (repeated as needed).

- wrap:

  Not supported (must be `FALSE`).

- inplace:

  Must be `FALSE` (arrays are immutable).

## Value

An array, or a list of index arrays.
