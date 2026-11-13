# Dynamic update slice

Dynamic update slice

## Usage

``` r
lax_dynamic_update_slice(
  operand,
  update,
  start_indices,
  allow_negative_indices = TRUE
)
```

## Arguments

- operand:

  Array.

- update:

  Array written into operand.

- start_indices:

  0-based start indices.

- allow_negative_indices:

  Whether negative start indices count from the end of the axis (one
  flag, or one per axis).
