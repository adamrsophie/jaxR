# Extract values of a dense array at the indices of a BCOO array

Port of `jax.experimental.sparse.bcoo_extract`.

## Usage

``` r
experimental_sparse_bcoo_extract(sparr, arr, assume_unique = NULL)
```

## Arguments

- sparr:

  A BCOO array.

- arr:

  A dense array of the same shape.

- assume_unique:

  Whether the indices are unique (default: `sparr$unique_indices`).

## Value

A BCOO array with the indices of `sparr`.
