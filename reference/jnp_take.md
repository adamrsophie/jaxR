# Take elements from an array

Port of `jnp.take`.

## Usage

``` r
jnp_take(
  a,
  indices,
  axis = NULL,
  out = NULL,
  mode = NULL,
  unique_indices = FALSE,
  indices_are_sorted = FALSE,
  fill_value = NULL
)
```

## Arguments

- a:

  Array.

- indices:

  Integer array of 0-based indices.

- axis:

  Axis along which to take; `NULL` takes from the flattened array.

- out:

  Not supported.

- mode:

  Out of bounds mode: `"fill"` (default: out of bounds indices give
  `fill_value`), `"clip"` or `"wrap"`.

- unique_indices:

  Whether the indices are known to be free of duplicates.

- indices_are_sorted:

  Whether the indices are known to be sorted.

- fill_value:

  Fill value for `mode = "fill"`; defaults to NaN for inexact types, the
  minimum for signed integers, the maximum for unsigned integers and
  `TRUE` for booleans.

## Value

An array.
