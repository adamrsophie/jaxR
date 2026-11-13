# Elementwise products with BCOO arrays

Ports of `jax.experimental.sparse.bcoo_multiply_sparse` (two BCOO
arrays, the result stores the intersection of their indices) and
`bcoo_multiply_dense` (the result is the data of a BCOO array with the
indices of `sp_mat`).

## Usage

``` r
experimental_sparse_bcoo_multiply_sparse(lhs, rhs)

experimental_sparse_bcoo_multiply_dense(sp_mat, v)
```

## Arguments

- lhs, rhs, sp_mat:

  BCOO arrays.

- v:

  A dense array that broadcasts to `sp_mat`'s shape.

## Value

A BCOO array, or the data array for `bcoo_multiply_dense`.
