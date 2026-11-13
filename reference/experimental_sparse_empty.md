# Empty and identity sparse arrays

Ports of `jax.experimental.sparse.empty` and `eye`.

## Usage

``` r
experimental_sparse_empty(
  shape,
  dtype = NULL,
  index_dtype = "int32",
  sparse_format = "bcoo",
  ...
)

experimental_sparse_eye(
  N,
  M = NULL,
  k = 0L,
  dtype = NULL,
  index_dtype = "int32",
  sparse_format = "bcoo",
  ...
)
```

## Arguments

- shape:

  Shape.

- dtype:

  Data dtype.

- index_dtype:

  Index dtype.

- sparse_format:

  One of `"bcoo"`, `"bcsr"`, `"coo"`, `"csr"`, `"csc"` (`eye` has no
  `"bcsr"`).

- ...:

  Extra format arguments (`n_batch`, `n_dense`, `nse`).

- N, M:

  Number of rows and columns.

- k:

  Diagonal offset.

## Value

A sparse array.
