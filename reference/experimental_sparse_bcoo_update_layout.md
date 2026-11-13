# Change the batch and dense dimensions of a BCOO array

Port of `jax.experimental.sparse.bcoo_update_layout`.

## Usage

``` r
experimental_sparse_bcoo_update_layout(
  mat,
  n_batch = NULL,
  n_dense = NULL,
  on_inefficient = "error"
)
```

## Arguments

- mat:

  A BCOO array.

- n_batch, n_dense:

  New numbers of batch and dense dimensions.

- on_inefficient:

  `"error"`, `"warn"` or `NULL`: what to do when the new layout stores
  many explicit zeros.

## Value

A BCOO array.
