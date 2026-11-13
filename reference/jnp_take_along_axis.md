# Take values along an axis

Port of `jnp.take_along_axis`.

## Usage

``` r
jnp_take_along_axis(
  arr,
  indices,
  axis = -1L,
  mode = NULL,
  fill_value = NULL,
  wrap_negative_indices = TRUE
)
```

## Arguments

- arr:

  Array.

- indices:

  Integer array with the same number of dimensions as `arr` (broadcast
  against it along the other axes).

- axis:

  0-based axis; `NULL` flattens `arr` (then `indices` must be 1-D).

- mode:

  Out of bounds mode: `"fill"` (default), `"clip"`,
  `"promise_in_bounds"` or `"one_hot"` (a sum over a one hot mask; out
  of bounds indices give zero).

- fill_value:

  Fill value for `mode = "fill"`.

- wrap_negative_indices:

  Whether negative indices count from the end.

## Value

An array.
