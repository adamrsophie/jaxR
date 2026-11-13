# Dynamic slice

Dynamic slice

## Usage

``` r
lax_dynamic_slice(
  operand,
  start_indices,
  slice_sizes,
  allow_negative_indices = TRUE
)
```

## Arguments

- operand:

  Array.

- start_indices:

  List (or vector) of 0-based scalar start indices.

- slice_sizes:

  Sizes.

- allow_negative_indices:

  Whether negative start indices count from the end of the axis (one
  flag, or one per axis). When `FALSE` a negative R number is an error
  and a negative array index is clamped like any other out of bounds
  index.
