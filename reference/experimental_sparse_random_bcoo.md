# Random BCOO arrays

Port of `jax.experimental.sparse.random_bcoo`.

## Usage

``` r
experimental_sparse_random_bcoo(
  key,
  shape,
  dtype = "float",
  indices_dtype = NULL,
  nse = 0.2,
  n_batch = 0L,
  n_dense = 0L,
  unique_indices = TRUE,
  sorted_indices = FALSE,
  generator = random_uniform,
  ...
)
```

## Arguments

- key:

  A PRNG key.

- shape:

  Shape.

- dtype:

  Data dtype.

- indices_dtype:

  Index dtype (default integer type).

- nse:

  Number of stored elements, or a fraction in `(0, 1)` of the sparse
  size.

- n_batch, n_dense:

  Numbers of batch and dense dimensions.

- unique_indices:

  Draw indices without replacement.

- sorted_indices:

  Sort the indices.

- generator:

  Function `(key, shape, dtype, ...)` drawing the data.

- ...:

  Passed to `generator`.

## Value

A BCOO array.
