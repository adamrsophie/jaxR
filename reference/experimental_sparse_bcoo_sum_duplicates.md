# Sum duplicate entries of a BCOO array

Ports of `jax.experimental.sparse.bcoo_sum_duplicates` and
`bcoo_eliminate_zeros` (the latter is reached through
`mat$sum_duplicates(remove_zeros = TRUE)`). `nse` must be given under
transformations.

## Usage

``` r
experimental_sparse_bcoo_sum_duplicates(mat, nse = NULL)
```

## Arguments

- mat:

  A BCOO array.

- nse:

  Number of stored elements of the result.

## Value

A BCOO array with sorted, unique indices.
