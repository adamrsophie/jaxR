# Functions on BCSR arrays

Ports of `jax.experimental.sparse.bcsr_fromdense`, `bcsr_todense`,
`bcsr_dot_general` (BCSR lhs, dense rhs), `bcsr_broadcast_in_dim`,
`bcsr_concatenate` and `bcsr_sum_duplicates`.

## Usage

``` r
experimental_sparse_bcsr_fromdense(
  mat,
  nse = NULL,
  n_batch = 0L,
  n_dense = 0L,
  index_dtype = "int32"
)

experimental_sparse_bcsr_todense(mat)

experimental_sparse_bcsr_dot_general(
  lhs,
  rhs,
  dimension_numbers,
  precision = NULL,
  preferred_element_type = NULL
)

experimental_sparse_bcsr_broadcast_in_dim(
  mat,
  shape,
  broadcast_dimensions,
  sharding = NULL
)

experimental_sparse_bcsr_concatenate(operands, dimension)

experimental_sparse_bcsr_sum_duplicates(mat, nse = NULL)
```

## Arguments

- mat:

  A dense array (`bcsr_fromdense`) or a BCSR array.

- nse, n_batch, n_dense, index_dtype:

  As for
  [`experimental_sparse_bcoo_fromdense()`](https://adamrsophie.github.io/jaxR/reference/experimental_sparse_bcoo_fromdense.md).

- lhs, rhs:

  Operands of `bcsr_dot_general`.

- dimension_numbers, precision, preferred_element_type:

  As for
  [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md).

- shape, broadcast_dimensions, sharding:

  As for
  [`lax_broadcast_in_dim()`](https://adamrsophie.github.io/jaxR/reference/lax_broadcast_in_dim.md).

- operands:

  A list of BCSR arrays.

- dimension:

  0-based dimension.

## Value

A BCSR or dense array.
