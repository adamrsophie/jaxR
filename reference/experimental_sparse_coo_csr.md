# Functions on COO and CSR matrices

Ports of `jax.experimental.sparse.coo_fromdense`, `coo_todense`,
`coo_matvec`, `coo_matmat`, `csr_fromdense`, `csr_todense`, `csr_matvec`
and `csr_matmat` (two dimensional matrices only).

## Usage

``` r
experimental_sparse_coo_fromdense(mat, nse = NULL, index_dtype = "int32")

experimental_sparse_coo_todense(mat)

experimental_sparse_coo_matvec(mat, v, transpose = FALSE)

experimental_sparse_coo_matmat(mat, B, transpose = FALSE)

experimental_sparse_csr_fromdense(mat, nse = NULL, index_dtype = "int32")

experimental_sparse_csr_todense(mat)

experimental_sparse_csr_matvec(mat, v, transpose = FALSE)

experimental_sparse_csr_matmat(mat, B, transpose = FALSE)
```

## Arguments

- mat:

  A dense matrix (`*_fromdense`) or a COO/CSR matrix.

- nse:

  Number of stored elements (default: the number of nonzeros).

- index_dtype:

  Index dtype.

- v:

  A vector.

- transpose:

  Multiply by the transpose of `mat`.

- B:

  A matrix.

## Value

A COO/CSR matrix or a dense array.
