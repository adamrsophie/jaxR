# Functional indexed updates

Ports of the `x.at[idx]` methods of JAX arrays: `at_get` is
`x.at[idx].get()`, `at_set` is `x.at[idx].set(value)`, and likewise
`add`, `subtract`, `multiply`, `divide`, `power`, `min`, `max` and
`apply`. The index arguments in `...` follow the rules of
[`[.jax_value`](https://adamrsophie.github.io/jaxR/reference/jax_indexing.md).
None of these modify `x`; they return a new array. With repeated indices
all updates are applied (in an unspecified order for `at_set`).

## Usage

``` r
at_get(
  x,
  ...,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  mode = NULL,
  fill_value = NULL,
  wrap_negative_indices = TRUE,
  strategy = "auto"
)

at_set(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_add(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_subtract(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_multiply(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_divide(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_power(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_min(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_max(
  x,
  ...,
  value,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)

at_apply(
  x,
  ...,
  func,
  mode = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  wrap_negative_indices = TRUE
)
```

## Arguments

- x:

  Array.

- ...:

  Index arguments (see
  [jax_indexing](https://adamrsophie.github.io/jaxR/reference/jax_indexing.md)).

- indices_are_sorted:

  Whether the indices are known to be sorted.

- unique_indices:

  Whether the indices are known to be free of duplicates.

- mode:

  Out of bounds mode: `"clip"`, `"fill"` or `"drop"`,
  `"promise_in_bounds"`. By default reads clamp indices and updates drop
  out of bounds indices.

- fill_value:

  Value returned for out of bounds indices with `mode = "fill"`.

- wrap_negative_indices:

  Whether negative indices count from the end.

- strategy:

  One of `"auto"`, `"gather"`, `"static_slice"`, `"dynamic_slice"`.

- value:

  Update values, broadcast to the indexed shape.

- func:

  Unary function applied at the indices (not differentiable).

## Value

An array.
