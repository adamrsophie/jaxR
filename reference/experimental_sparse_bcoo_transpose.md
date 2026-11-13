# Transpose a BCOO array

Port of `jax.experimental.sparse.bcoo_transpose`: batch, sparse and
dense dimensions may only be permuted among themselves.

## Usage

``` r
experimental_sparse_bcoo_transpose(mat, permutation)
```

## Arguments

- mat:

  A BCOO array.

- permutation:

  0-based permutation.

## Value

A BCOO array.
