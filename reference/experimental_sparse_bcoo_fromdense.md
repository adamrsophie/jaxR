# Convert between dense arrays and BCOO

Ports of `jax.experimental.sparse.bcoo_fromdense` and `bcoo_todense`.
`nse` (the number of stored elements) must be concrete under
transformations; by default it is the largest number of nonzeros per
batch element.

## Usage

``` r
experimental_sparse_bcoo_fromdense(
  mat,
  nse = NULL,
  n_batch = 0L,
  n_dense = 0L,
  index_dtype = "int32"
)

experimental_sparse_bcoo_todense(mat)
```

## Arguments

- mat:

  A dense array, or for `bcoo_todense` a BCOO array.

- nse:

  Number of stored elements.

- n_batch, n_dense:

  Number of batch and dense dimensions.

- index_dtype:

  Index dtype.

## Value

A BCOO array or a dense array.

## Examples

``` r
if (FALSE) { # \dontrun{
m <- experimental_sparse_bcoo_fromdense(matrix(c(0, 1, 2, 0), 2))
experimental_sparse_bcoo_todense(m)
} # }
```
