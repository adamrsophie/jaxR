# Reshape a BCOO array

Port of `jax.experimental.sparse.bcoo_reshape`: the new shape may not
mix batch, sparse and dense dimensions.

## Usage

``` r
experimental_sparse_bcoo_reshape(
  mat,
  new_sizes,
  dimensions = NULL,
  sharding = NULL
)
```

## Arguments

- mat:

  A BCOO array.

- new_sizes:

  New shape.

- dimensions:

  Optional 0-based permutation applied first.

- sharding:

  Ignored.

## Value

A BCOO array.
