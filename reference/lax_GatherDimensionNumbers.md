# Dimension numbers for gather and scatter

Ports of `jax.lax.GatherDimensionNumbers` and
`jax.lax.ScatterDimensionNumbers`. All dimensions are 0-based. The index
vector dimension is implicit: it is always the last dimension of the
indices.

## Usage

``` r
lax_GatherDimensionNumbers(
  offset_dims,
  collapsed_slice_dims,
  start_index_map,
  operand_batching_dims = integer(0),
  start_indices_batching_dims = integer(0)
)

lax_ScatterDimensionNumbers(
  update_window_dims,
  inserted_window_dims,
  scatter_dims_to_operand_dims,
  operand_batching_dims = integer(0),
  scatter_indices_batching_dims = integer(0)
)
```

## Arguments

- offset_dims:

  Output dimensions that offset into the slice taken from the operand
  (ascending).

- collapsed_slice_dims:

  Operand dimensions of slice size 1 that are removed from the output
  (ascending).

- start_index_map:

  For each index component, the operand dimension it indexes.

- operand_batching_dims:

  Operand batching dimensions (ascending).

- start_indices_batching_dims:

  Batching dimensions of the indices, matched with
  `operand_batching_dims`.

- update_window_dims:

  Dimensions of the updates that are window dimensions (ascending).

- inserted_window_dims:

  Size 1 window dimensions inserted into the updates (operand
  dimensions, ascending).

- scatter_dims_to_operand_dims:

  For each index component, the operand dimension it indexes.

- scatter_indices_batching_dims:

  Batching dimensions of the indices, matched with
  `operand_batching_dims`.

## Value

A named list.
