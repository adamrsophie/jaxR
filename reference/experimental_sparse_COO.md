# Coordinate (COO) sparse matrices

Port of `jax.experimental.sparse.COO`: `list(data, row, col)` and a two
dimensional shape.
`experimental_sparse_COO$fromdense(mat, nse, index_dtype)` builds one
from a dense matrix; `$todense()`, `$transpose()` (or `$T`) and `%*%`
work as in JAX.

## Usage

``` r
experimental_sparse_COO(args, shape, rows_sorted = FALSE, cols_sorted = FALSE)
```

## Arguments

- args:

  `list(data, row, col)`.

- shape:

  Shape.

- rows_sorted, cols_sorted:

  Sortedness flags.

## Value

A COO matrix (class `jax_coo`).
