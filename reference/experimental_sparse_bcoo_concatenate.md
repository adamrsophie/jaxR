# Concatenate BCOO arrays

Port of `jax.experimental.sparse.bcoo_concatenate` (along batch or
sparse dimensions).

## Usage

``` r
experimental_sparse_bcoo_concatenate(operands, dimension)
```

## Arguments

- operands:

  A list of BCOO arrays.

- dimension:

  0-based dimension.

## Value

A BCOO array.
