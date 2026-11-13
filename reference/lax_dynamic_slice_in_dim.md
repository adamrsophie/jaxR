# Dynamic slices and updates along one axis

Ports of `jax.lax.dynamic_slice_in_dim`, `dynamic_index_in_dim`,
`dynamic_update_slice_in_dim` and `dynamic_update_index_in_dim`. The
start index may be a traced value; slices that overrun the array are
shifted to fit.

## Usage

``` r
lax_dynamic_slice_in_dim(
  operand,
  start_index,
  slice_size,
  axis = 0L,
  allow_negative_indices = TRUE
)

lax_dynamic_index_in_dim(
  operand,
  index,
  axis = 0L,
  keepdims = TRUE,
  allow_negative_indices = TRUE
)

lax_dynamic_update_slice_in_dim(
  operand,
  update,
  start_index,
  axis,
  allow_negative_indices = TRUE
)

lax_dynamic_update_index_in_dim(
  operand,
  update,
  index,
  axis,
  allow_negative_indices = TRUE
)
```

## Arguments

- operand:

  Array.

- start_index, index:

  0-based (possibly traced) scalar index.

- slice_size:

  Static size of the slice.

- axis:

  0-based axis.

- allow_negative_indices:

  Whether negative indices count from the end. If `FALSE` negative
  indices are out of bounds (clamped to the first index).

- keepdims:

  Keep the indexed dimension (size 1).

- update:

  Array written into `operand`.

## Value

An array.
