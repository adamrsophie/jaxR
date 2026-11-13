# Slice a BCOO array

Ports of `jax.experimental.sparse.bcoo_slice` (static bounds) and
`bcoo_dynamic_slice` (traced start indices, static sizes).

## Usage

``` r
experimental_sparse_bcoo_slice(
  mat,
  start_indices,
  limit_indices,
  strides = NULL
)

experimental_sparse_bcoo_dynamic_slice(mat, start_indices, slice_sizes)
```

## Arguments

- mat:

  A BCOO array.

- start_indices, limit_indices, strides:

  Static 0-based bounds and strides.

- slice_sizes:

  Static slice sizes.

## Value

A BCOO array.
