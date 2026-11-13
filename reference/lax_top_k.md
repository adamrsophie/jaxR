# Top k values and their indices

Port of `jax.lax.top_k`.

## Usage

``` r
lax_top_k(operand, k, axis = -1L, is_stable = TRUE)
```

## Arguments

- operand:

  Array of a non complex dtype.

- k:

  Number of entries.

- axis:

  0-based axis (default last).

- is_stable:

  Preserve the order of equal elements.

## Value

`list(values, indices)`; indices are 0-based `int32`.
