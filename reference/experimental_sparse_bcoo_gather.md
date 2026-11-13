# Gather from a BCOO array

Port of `jax.experimental.sparse.bcoo_gather` (only
`mode = "promise_in_bounds"`).

## Usage

``` r
experimental_sparse_bcoo_gather(
  operand,
  start_indices,
  dimension_numbers,
  slice_sizes,
  unique_indices = FALSE,
  indices_are_sorted = FALSE,
  mode = NULL,
  fill_value = NULL
)
```

## Arguments

- operand:

  A BCOO array.

- start_indices, dimension_numbers, slice_sizes:

  As for
  [`lax_gather()`](https://adamrsophie.github.io/jaxR/reference/lax_gather.md).

- unique_indices, indices_are_sorted, mode, fill_value:

  As for
  [`lax_gather()`](https://adamrsophie.github.io/jaxR/reference/lax_gather.md).

## Value

A BCOO array.
