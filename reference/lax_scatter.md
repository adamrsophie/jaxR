# Scatter

Ports of `jax.lax.scatter` (updates replace values), `scatter_add`,
`scatter_sub`, `scatter_mul`, `scatter_min`, `scatter_max` and
`scatter_apply` (XLA Scatter). Most code should use
[`at_set()`](https://adamrsophie.github.io/jaxR/reference/at.md),
[`at_add()`](https://adamrsophie.github.io/jaxR/reference/at.md) and
friends instead. With repeated indices, `lax_scatter` applies the
updates in an unspecified order. `lax_scatter_apply` is not compatible
with automatic differentiation.

## Usage

``` r
lax_scatter(
  operand,
  scatter_indices,
  updates,
  dimension_numbers,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)

lax_scatter_add(
  operand,
  scatter_indices,
  updates,
  dimension_numbers,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)

lax_scatter_sub(
  operand,
  scatter_indices,
  updates,
  dimension_numbers,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)

lax_scatter_mul(
  operand,
  scatter_indices,
  updates,
  dimension_numbers,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)

lax_scatter_min(
  operand,
  scatter_indices,
  updates,
  dimension_numbers,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)

lax_scatter_max(
  operand,
  scatter_indices,
  updates,
  dimension_numbers,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)

lax_scatter_apply(
  operand,
  scatter_indices,
  func,
  dimension_numbers,
  update_shape = integer(0),
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL
)
```

## Arguments

- operand:

  Array to which the scatter is applied.

- scatter_indices:

  Integer array of indices; its last dimension is the index vector
  dimension.

- updates:

  Updates scattered onto `operand`.

- dimension_numbers:

  A
  [`lax_ScatterDimensionNumbers()`](https://adamrsophie.github.io/jaxR/reference/lax_GatherDimensionNumbers.md)
  value.

- indices_are_sorted:

  Whether the indices are known to be sorted.

- unique_indices:

  Whether the updated elements are guaranteed not to overlap.

- mode:

  Out of bounds mode: `"clip"` clamps indices, `"fill"` or `"drop"` (the
  default, also for `NULL`) and `"promise_in_bounds"` drop out of bounds
  updates. See
  [lax_GatherScatterMode](https://adamrsophie.github.io/jaxR/reference/lax_GatherScatterMode.md).

- func:

  Unary function of a scalar applied at each index; repeated indices
  apply it several times.

- update_shape:

  Shape of the updates at the given indices.

## Value

The updated array.
