# Broadcast a BCOO array

Port of `jax.experimental.sparse.bcoo_broadcast_in_dim`.

## Usage

``` r
experimental_sparse_bcoo_broadcast_in_dim(
  mat,
  shape,
  broadcast_dimensions,
  sharding = NULL
)
```

## Arguments

- mat:

  A BCOO array.

- shape:

  Output shape.

- broadcast_dimensions:

  0-based output dimension of each input dimension.

- sharding:

  Ignored.

## Value

A BCOO array.
