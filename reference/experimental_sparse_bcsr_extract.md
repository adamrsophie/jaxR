# Extract values of a dense array at the indices of a BCSR array

Port of `jax.experimental.sparse.bcsr_extract`.

## Usage

``` r
experimental_sparse_bcsr_extract(indices, indptr, mat)
```

## Arguments

- indices, indptr:

  BCSR index buffers.

- mat:

  A dense array.

## Value

The data buffer of a BCSR array.
