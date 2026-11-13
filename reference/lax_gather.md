# Gather

Port of `jax.lax.gather` (XLA Gather). Most code should use NumPy style
indexing (`x[i]`,
[`at_get()`](https://adamrsophie.github.io/jaxR/reference/at.md))
instead.

## Usage

``` r
lax_gather(
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

  Array from which slices are taken.

- start_indices:

  Integer array of start indices; its last dimension is the index vector
  dimension.

- dimension_numbers:

  A
  [`lax_GatherDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_GatherDimensionNumbers.md)
  value.

- slice_sizes:

  Size of each slice, one entry per operand dimension.

- unique_indices:

  Whether the gathered elements are guaranteed not to overlap.

- indices_are_sorted:

  Whether the indices are known to be sorted.

- mode:

  Out of bounds mode: `"clip"`, `"fill"` (or `"drop"`),
  `"promise_in_bounds"`; `NULL` means `"promise_in_bounds"`. See
  [lax_GatherScatterMode](https://adamrsophie.github.io/jaxR/reference/lax_GatherScatterMode.md).

- fill_value:

  Value of out of bounds slices in fill mode. Defaults to NaN for
  inexact types, the minimum for signed integers, the maximum for
  unsigned integers and `TRUE` for booleans.

## Value

The gathered array.
